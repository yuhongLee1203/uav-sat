"""Six-order MeanShift / GRU / Kalman architecture ablation for v36_byTeacher.

This file implements the six serial permutations requested in the PPT:
    MKG, MGK, GMK, GKM, KGM, KMG
where M=MeanShift, G=GRU position refiner, K=Kalman filter.

Key correction relative to the old page-1 tracker:
- The GRU no longer predicts a future motion polynomial Delta that is added to the
  previous/final position.
- There is no `final_position + GRU_delta` feedback path.
- The GRU is a current-frame position refiner. It receives the current stage's
  position estimate + uncertainty + temporal visual features and returns a
  refined current-frame position.
- Kalman keeps its own recurrent state; it never receives a GRU motion delta.
- For the controlled six-order experiment, the predefined route reference point
  is used only to open/orient the common forward 3x6 candidate window, matching
  the existing v36_byTeacher reference-prior protocol.

The implementation intentionally keeps the visual retrieval model frozen during
six-order training. Each order trains only its own GRU position-refinement model,
so the comparison isolates operator ordering rather than re-training the visual
backbone differently for each order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from robust_tracker_base import build_route_cache, metric_summary, resolve_device, set_seed
from visual_localizer import (
    CandidateBatch,
    FrozenVisualLocalizer,
    regular_grid_indices,
    soft_mean_shift,
    train_visual_retrieval_a_only,
)


VALID_ORDERS = ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG")

# Separate defaults for this experiment. Existing config values remain untouched.
SIX_ORDER_ROOT = Path(config.BACKBONE_OUTPUT_DIR) / "six_order_ablation"
SIX_ORDER_CHECKPOINT_ROOT = Path(config.CHECKPOINT_DIR) / "six_order_ablation"
GRU_FEATURE_DIM = int(os.environ.get("UAVSAT_SIX_GRU_FEATURE", "128"))
GRU_HIDDEN_DIM = int(os.environ.get("UAVSAT_SIX_GRU_HIDDEN", "256"))
GRU_DROPOUT = float(os.environ.get("UAVSAT_SIX_GRU_DROPOUT", "0.0"))
POSITION_SCALE_M = float(os.environ.get("UAVSAT_SIX_POSITION_SCALE_M", "1000.0"))
VARIANCE_SCALE_M2 = float(os.environ.get("UAVSAT_SIX_VARIANCE_SCALE_M2", "100.0"))
GRU_CORRECTION_SCALE_M = float(os.environ.get("UAVSAT_SIX_GRU_CORRECTION_SCALE_M", "10.0"))
KF_R_FLOOR_M2 = float(os.environ.get("UAVSAT_SIX_KF_R_FLOOR_M2", "1.0"))
KF_R_CEIL_M2 = float(os.environ.get("UAVSAT_SIX_KF_R_CEIL_M2", "2500.0"))
LOSS_POSITION = float(os.environ.get("UAVSAT_SIX_LOSS_POSITION", "1.0"))
LOSS_CORRECTION = float(os.environ.get("UAVSAT_SIX_LOSS_CORRECTION", "0.10"))


@dataclass
class Estimate:
    """A current-frame 2-D position estimate with axis-wise uncertainty."""

    xy: torch.Tensor
    var_xy: torch.Tensor
    source: str


@dataclass
class GRUOutput:
    """Current-frame GRU refinement output.

    `correction_xy` is an error-correction residual for the current incoming
    estimate. It is not a future motion delta and is never added to a previous
    final position outside the GRU stage.
    """

    position_xy: torch.Tensor
    correction_xy: torch.Tensor
    hidden: torch.Tensor


class PositionRefinementGRU(nn.Module):
    """GRU that refines the current estimate rather than predicting future motion."""

    def __init__(self):
        super().__init__()
        f = GRU_FEATURE_DIM
        h = GRU_HIDDEN_DIM
        d = GRU_DROPOUT

        def proj(in_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, f),
                nn.GELU(),
                nn.LayerNorm(f),
            )

        self.position_projector = proj(2)
        self.variance_projector = proj(2)
        self.temporal_mean_projector = proj(int(config.EMBED_DIM))
        self.first_difference_projector = proj(int(config.EMBED_DIM))
        self.gru = nn.GRUCell(4 * f, h)
        self.dropout = nn.Dropout(d)
        self.position_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(d),
            nn.Linear(h // 2, 2),
        )
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

    def initial_hidden(self, batch_size: int, device, dtype):
        return torch.zeros(batch_size, GRU_HIDDEN_DIM, device=device, dtype=dtype)

    def forward_step(
        self,
        estimate_xy: torch.Tensor,
        variance_xy: torch.Tensor,
        z_uav: torch.Tensor,
        previous_z_uav: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
    ) -> GRUOutput:
        if previous_z_uav is None:
            previous_z_uav = z_uav
        if hidden is None:
            hidden = self.initial_hidden(z_uav.shape[0], z_uav.device, z_uav.dtype)

        temporal_mean = 0.5 * (z_uav + previous_z_uav)
        first_difference = z_uav - previous_z_uav
        position_in = estimate_xy.float() / max(POSITION_SCALE_M, 1e-6)
        variance_in = torch.log1p(variance_xy.float().clamp_min(0.0)) / math.log1p(
            max(VARIANCE_SCALE_M2, 1e-6)
        )
        recurrent_input = torch.cat(
            [
                self.position_projector(position_in),
                self.variance_projector(variance_in),
                self.temporal_mean_projector(temporal_mean),
                self.first_difference_projector(first_difference),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        correction_xy = self.position_head(self.dropout(new_hidden)) * GRU_CORRECTION_SCALE_M
        refined_xy = estimate_xy.float() + correction_xy
        return GRUOutput(refined_xy, correction_xy, new_hidden)


class DynamicXYKalman:
    """Constant-velocity XY Kalman with dynamic measurement covariance."""

    def __init__(self, initial_xy: np.ndarray):
        p = np.asarray(initial_xy, dtype=np.float64).reshape(2)
        self.x = np.asarray([p[0], p[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
            ]
        )
        self.F = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self.Q = np.diag(
            [
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_VELOCITY),
                float(config.KALMAN_Q_VELOCITY),
            ]
        )

    def step(self, measurement_xy: np.ndarray, variance_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        r = np.asarray(variance_xy, dtype=np.float64).reshape(2)
        r = np.clip(r, KF_R_FLOOR_M2, KF_R_CEIL_M2)
        R = np.diag(r)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ innovation
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return self.x[:2].copy(), np.diag(self.P)[:2].copy()


def _tensor_xy(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, 2)


def _posterior_centroid_and_variance(
    centers: torch.Tensor,
    probability: torch.Tensor,
    center_xy: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if center_xy is None:
        center_xy = (probability[:, :, None] * centers).sum(dim=1)
    residual = centers - center_xy[:, None, :]
    var_xy = (probability[:, :, None] * residual.square()).sum(dim=1)
    return center_xy, var_xy.clamp_min(1e-4)


def raw_estimate(batch: CandidateBatch) -> Estimate:
    xy, var_xy = _posterior_centroid_and_variance(batch.centers, batch.raw_prob)
    return Estimate(xy=xy, var_xy=var_xy, source="raw_cosine")


def meanshift_estimate(batch: CandidateBatch, source: str) -> Estimate:
    xy, var_xy = _posterior_centroid_and_variance(
        batch.centers, batch.raw_prob, center_xy=batch.softms_xy
    )
    return Estimate(xy=xy, var_xy=var_xy, source=source)


def strict_forward_indices(full_centers: torch.Tensor, heading_rad: torch.Tensor) -> torch.Tensor:
    batch = int(full_centers.shape[0])
    if int(full_centers.shape[1]) != 36:
        raise RuntimeError("forward search requires a 6x6 base lattice")
    headings = heading_rad.reshape(-1).to(full_centers.device, full_centers.dtype)
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match candidate batch")
    geometric_center = full_centers.mean(dim=1, keepdim=True)
    relative = full_centers - geometric_center
    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    use_x = cos_h.abs() >= sin_h.abs()
    direction_sign = torch.where(
        use_x,
        torch.where(cos_h >= 0, torch.ones_like(cos_h), -torch.ones_like(cos_h)),
        torch.where(sin_h >= 0, torch.ones_like(sin_h), -torch.ones_like(sin_h)),
    )
    longitudinal = torch.where(
        use_x[:, None], relative[:, :, 0], relative[:, :, 1]
    ) * direction_sign[:, None]
    selected = torch.topk(longitudinal, k=18, dim=1, largest=True, sorted=False).indices
    lateral = torch.where(use_x[:, None], relative[:, :, 1], relative[:, :, 0])
    chosen_longitudinal = torch.gather(longitudinal, 1, selected)
    chosen_lateral = torch.gather(lateral, 1, selected)
    ordering_key = chosen_longitudinal * 1000.0 + chosen_lateral
    order = torch.argsort(ordering_key, dim=1)
    return torch.gather(selected, 1, order)


def forward_candidate_batch(
    visual: FrozenVisualLocalizer,
    uav_clip: torch.Tensor,
    reference_center_xy: torch.Tensor,
    heading_rad: torch.Tensor,
) -> CandidateBatch:
    full_indices = regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        reference_center_xy,
        6,
        config.SAT_STRIDE,
        visual.device,
    )
    full_centers = visual.gallery["xy"][full_indices]
    selected_local = strict_forward_indices(full_centers, heading_rad)
    indices = torch.gather(full_indices, 1, selected_local)
    centers = visual.gallery["xy"][indices]
    satellite_clip = visual.gallery["clip_feat"][indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    logits = visual.model.logit_scale.exp().clamp(max=100.0) * (z_uav[:, None] * z_sat).sum(dim=2)
    prob = torch.softmax(logits / float(config.MEANSHIFT_SCORE_TAU), dim=1)
    top_index = logits.argmax(dim=1)
    top_xy = centers[torch.arange(centers.shape[0], device=visual.device), top_index]
    soft_xy, support, _, _, mode_weights, _ = soft_mean_shift(
        logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    return CandidateBatch(
        indices=indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=logits,
        raw_prob=prob,
        raw_top1_xy=top_xy,
        softms_xy=soft_xy,
        softms_support=support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
    )


def route_headings(reference_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    delta = np.zeros_like(xy)
    if len(xy) > 1:
        delta[:-1] = xy[1:] - xy[:-1]
        delta[-1] = delta[-2]
    heading = np.zeros(len(xy), dtype=np.float64)
    last = 0.0
    for i, d in enumerate(delta):
        if float(np.linalg.norm(d)) > 1e-8:
            last = float(math.atan2(d[1], d[0]))
        heading[i] = last
    return heading


class SixOrderTracker:
    def __init__(self, order: str, model: PositionRefinementGRU, initial_xy: np.ndarray, device):
        order = order.upper()
        if order not in VALID_ORDERS:
            raise ValueError(f"invalid order {order!r}; expected one of {VALID_ORDERS}")
        self.order = order
        self.model = model
        self.device = device
        self.kalman = DynamicXYKalman(initial_xy)
        self.hidden = None
        self.previous_z = None

    def detach_temporal_state(self):
        if torch.is_tensor(self.hidden):
            self.hidden = self.hidden.detach()
        if torch.is_tensor(self.previous_z):
            self.previous_z = self.previous_z.detach()

    def _apply_k(self, current: Estimate) -> Estimate:
        xy_np, var_np = self.kalman.step(
            current.xy[0].detach().cpu().numpy(),
            current.var_xy[0].detach().cpu().numpy(),
        )
        return Estimate(_tensor_xy(xy_np, self.device), _tensor_xy(var_np, self.device), "kalman")

    def _apply_g(self, current: Estimate, z_uav: torch.Tensor) -> Tuple[Estimate, GRUOutput]:
        output = self.model.forward_step(
            estimate_xy=current.xy,
            variance_xy=current.var_xy,
            z_uav=z_uav,
            previous_z_uav=self.previous_z,
            hidden=self.hidden,
        )
        self.hidden = output.hidden
        return Estimate(output.position_xy, current.var_xy, "gru"), output

    def _apply_center_m(
        self,
        visual: FrozenVisualLocalizer,
        uav_clip: torch.Tensor,
        current: Estimate,
    ) -> Tuple[Estimate, CandidateBatch]:
        center = current.xy.detach()
        batch = visual.candidate_batch(uav_clip=uav_clip, center_xy=center, grid_size=6)
        return meanshift_estimate(batch, "center_meanshift"), batch

    def forward_frame(
        self,
        visual: FrozenVisualLocalizer,
        uav_clip: torch.Tensor,
        reference_center_xy: np.ndarray,
        heading_rad: float,
    ) -> Dict[str, object]:
        with torch.no_grad():
            base_batch = forward_candidate_batch(
                visual,
                uav_clip,
                _tensor_xy(reference_center_xy, self.device),
                torch.as_tensor([heading_rad], dtype=torch.float32, device=self.device),
            )
            current = raw_estimate(base_batch)
        result: Dict[str, object] = {
            "base_batch": base_batch,
            "raw": current,
            "m": None,
            "g": None,
            "k": None,
            "center_m_batch": None,
        }
        for stage_index, stage in enumerate(self.order):
            if stage == "M":
                if stage_index == 0:
                    current = meanshift_estimate(base_batch, "forward_meanshift")
                    result["m"] = current
                else:
                    with torch.no_grad():
                        current, center_batch = self._apply_center_m(visual, uav_clip, current)
                    result["m"] = current
                    result["center_m_batch"] = center_batch
            elif stage == "K":
                current = self._apply_k(current)
                result["k"] = current
            elif stage == "G":
                current, gru_output = self._apply_g(current, base_batch.z_uav)
                result["g"] = current
                result["gru_output"] = gru_output
            else:
                raise RuntimeError(stage)
        self.previous_z = base_batch.z_uav.detach()
        result["final"] = current
        return result


def stage_error(stage: Optional[Estimate], reference_xy: np.ndarray) -> float:
    if stage is None:
        return float("nan")
    xy = stage.xy[0].detach().cpu().numpy().astype(np.float64)
    return float(np.linalg.norm(xy - reference_xy))


def output_paths(order: str) -> Tuple[Path, Path, Path]:
    root = SIX_ORDER_ROOT / order
    checkpoint = SIX_ORDER_CHECKPOINT_ROOT / f"six_order_{order}_{config.BACKBONE_KEY}.pt"
    latest = SIX_ORDER_CHECKPOINT_ROOT / f"six_order_{order}_{config.BACKBONE_KEY}_latest.pt"
    return root, checkpoint, latest


def training_sequence_loss(
    order: str,
    model: PositionRefinementGRU,
    optimizer,
    visual: FrozenVisualLocalizer,
    cache,
    device,
) -> Tuple[float, Dict[str, float]]:
    refs = cache.gt_xy.detach().cpu().numpy().astype(np.float64)
    headings = route_headings(refs)
    tracker = SixOrderTracker(order, model, refs[0], device)
    optimizer.zero_grad(set_to_none=True)
    chunk_loss = None
    chunk_count = 0
    total_loss = []
    position_losses = []
    correction_losses = []
    for i in range(len(cache)):
        frame = tracker.forward_frame(
            visual,
            cache.uav_clip[i : i + 1].to(device).float(),
            refs[i],
            headings[i],
        )
        g: Estimate = frame["g"]
        gout: GRUOutput = frame["gru_output"]
        target = _tensor_xy(refs[i], device)
        pos_loss = F.smooth_l1_loss(g.xy, target)
        g_input_xy = g.xy - gout.correction_xy
        target_correction = target - g_input_xy.detach()
        corr_loss = F.smooth_l1_loss(gout.correction_xy, target_correction)
        loss = LOSS_POSITION * pos_loss + LOSS_CORRECTION * corr_loss
        chunk_loss = loss if chunk_loss is None else chunk_loss + loss
        chunk_count += 1
        position_losses.append(float(pos_loss.detach().cpu()))
        correction_losses.append(float(corr_loss.detach().cpu()))
        is_last = i + 1 == len(cache)
        if chunk_count >= int(config.TBPTT_STEPS) or is_last:
            normalized = chunk_loss / float(chunk_count)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.GRAD_CLIP_NORM))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss.append(float(normalized.detach().cpu()))
            tracker.detach_temporal_state()
            chunk_loss = None
            chunk_count = 0
    return (
        float(np.mean(total_loss)) if total_loss else float("nan"),
        {
            "position": float(np.mean(position_losses)) if position_losses else float("nan"),
            "correction": float(np.mean(correction_losses)) if correction_losses else float("nan"),
        },
    )


@torch.no_grad()
def evaluate_route(
    order: str,
    model: PositionRefinementGRU,
    visual: FrozenVisualLocalizer,
    cache,
    device,
    save_csv: bool = False,
    measure_latency: bool = False,
) -> Dict[str, object]:
    model.eval()
    refs = cache.gt_xy.detach().cpu().numpy().astype(np.float64)
    headings = route_headings(refs)
    tracker = SixOrderTracker(order, model, refs[0], device)
    errors = []
    raw_errors = []
    m_errors = []
    g_errors = []
    k_errors = []
    timings = []
    rows = []
    for i in range(len(cache)):
        uav_clip = cache.uav_clip[i : i + 1].to(device).float()
        if measure_latency and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        frame = tracker.forward_frame(visual, uav_clip, refs[i], headings[i])
        if measure_latency and device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if measure_latency:
            timings.append(elapsed)
        final_xy = frame["final"].xy[0].detach().cpu().numpy().astype(np.float64)
        final_error = float(np.linalg.norm(final_xy - refs[i]))
        errors.append(final_error)
        raw_errors.append(stage_error(frame["raw"], refs[i]))
        m_errors.append(stage_error(frame["m"], refs[i]))
        g_errors.append(stage_error(frame["g"], refs[i]))
        k_errors.append(stage_error(frame["k"], refs[i]))
        if save_csv:
            def xy_of(stage):
                if stage is None:
                    return (float("nan"), float("nan"))
                p = stage.xy[0].detach().cpu().numpy()
                return float(p[0]), float(p[1])
            raw_x, raw_y = xy_of(frame["raw"])
            m_x, m_y = xy_of(frame["m"])
            g_x, g_y = xy_of(frame["g"])
            k_x, k_y = xy_of(frame["k"])
            rows.append(
                {
                    "frame_id": int(cache.frame_ids[i]),
                    "image_path": cache.image_paths[i],
                    "reference_x": float(refs[i, 0]),
                    "reference_y": float(refs[i, 1]),
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "M_x": m_x,
                    "M_y": m_y,
                    "G_x": g_x,
                    "G_y": g_y,
                    "K_x": k_x,
                    "K_y": k_y,
                    "final_x": float(final_xy[0]),
                    "final_y": float(final_xy[1]),
                    "error_final_m": final_error,
                    "latency_ms": elapsed,
                }
            )
        tracker.detach_temporal_state()
    summary = metric_summary(errors)
    def finite_mean(values):
        a = np.asarray(values, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")
    summary.update(
        {
            "Order": order,
            "Route": cache.route_name,
            "RawVisual_MLE_m": finite_mean(raw_errors),
            "M_stage_MLE_m": finite_mean(m_errors),
            "G_stage_MLE_m": finite_mean(g_errors),
            "K_stage_MLE_m": finite_mean(k_errors),
            "Final_MLE_m": float(summary["MLE_m"]),
            "FeedbackRule": "none: no final_position + GRU motion delta",
            "GRURole": "current-frame position error refinement",
            "KalmanRole": "own constant-velocity state + dynamic measurement covariance",
            "ReferenceUsage": "predefined route reference point opens/orients common forward 3x6 only",
        }
    )
    if timings:
        warmup = min(int(getattr(config, "LATENCY_WARMUP_FRAMES", 10)), len(timings))
        steady = np.asarray(timings[warmup:], dtype=np.float64)
        if steady.size == 0:
            steady = np.asarray(timings, dtype=np.float64)
        mean_ms = float(np.mean(steady))
        summary["Timing"] = {
            "mean_ms": mean_ms,
            "median_ms": float(np.median(steady)),
            "p90_ms": float(np.quantile(steady, 0.90)),
            "fps": float(1000.0 / max(mean_ms, 1e-12)),
        }
    if save_csv and rows:
        root, _, _ = output_paths(order)
        root.mkdir(parents=True, exist_ok=True)
        csv_path = root / f"{cache.route_name}_{order}_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["CSV"] = str(csv_path)
    return summary


def train_order(
    order: str,
    visual: FrozenVisualLocalizer,
    route_a,
    route_c,
    device,
    epochs: int,
    patience: int,
    resume: bool,
) -> Tuple[PositionRefinementGRU, float]:
    model = PositionRefinementGRU().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    _, checkpoint, latest = output_paths(order)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_score = float("inf")
    best_state = None
    stale = 0
    if resume and latest.exists():
        payload = torch.load(latest, map_location="cpu")
        if payload.get("order") != order:
            raise RuntimeError("resume checkpoint order mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model")
        stale = int(payload.get("stale", 0))
    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        train_loss, components = training_sequence_loss(order, model, optimizer, visual, route_a, device)
        val = evaluate_route(order, model, visual, route_c, device, save_csv=False)
        score = float(val["MLE_m"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)
        if improved:
            best_score = score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "order": order,
                    "architecture": f"SixOrder_{order}_PositionRefinementGRU",
                    "model": best_state,
                    "epoch": epoch,
                    "validation": val,
                    "training_route": "route_A",
                    "validation_route": "route_C",
                    "gru_semantics": "current-frame position refiner; no future delta/polynomial",
                },
                checkpoint,
            )
        else:
            stale += 1
        torch.save(
            {
                "order": order,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_model": best_state,
                "epoch": epoch,
                "best_score": best_score,
                "stale": stale,
            },
            latest,
        )
        print(
            f"[{order}] epoch={epoch:03d}/{epochs} "
            f"train={train_loss:.5f} pos={components['position']:.4f} corr={components['correction']:.4f} "
            f"C_MLE={val['MLE_m']:.3f} C_P90={val['P90_m']:.3f} "
            f"C_LSR15={val['LSR@15_pct']:.2f}% best={best_score:.3f} stale={stale}/{patience}",
            flush=True,
        )
        if epoch >= int(config.EARLY_STOP_MIN_EPOCH) and stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError(f"{order}: no checkpoint produced")
    model.load_state_dict(best_state)
    return model, best_score


def load_order_model(order: str, device) -> PositionRefinementGRU:
    _, checkpoint, _ = output_paths(order)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("order") != order:
        raise RuntimeError("checkpoint order mismatch")
    model = PositionRefinementGRU().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def prepare_visual_if_needed(args, device):
    if config.VISUAL_CHECKPOINT.exists():
        return
    if not args.train_visual_if_missing:
        raise FileNotFoundError(
            f"visual checkpoint not found: {config.VISUAL_CHECKPOINT}; pass --train-visual-if-missing"
        )
    train_visual_retrieval_a_only(
        device=device,
        epochs=int(args.visual_epochs),
        jitter_m=float(args.jitter_m),
        resume=bool(args.resume_visual),
    )


def build_cache(route_name: str, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


def run(args):
    order = args.order.upper()
    if order not in VALID_ORDERS:
        raise ValueError(order)
    set_seed(int(config.SEED))
    device = resolve_device()
    print(f"device={device} order={order} backbone={config.BACKBONE_KEY}", flush=True)
    prepare_visual_if_needed(args, device)
    if args.mode == "prepare_visual":
        print(f"visual checkpoint ready: {config.VISUAL_CHECKPOINT}", flush=True)
        return
    visual = FrozenVisualLocalizer(device)
    route_a = build_cache("route_A", visual, device) if args.mode in {"train", "train_eval"} else None
    route_c = build_cache("route_C", visual, device)
    if args.mode in {"train", "train_eval"}:
        model, best = train_order(
            order,
            visual,
            route_a,
            route_c,
            device,
            epochs=int(args.temporal_epochs),
            patience=int(args.patience),
            resume=bool(args.resume_temporal),
        )
        print(f"[{order}] best Route-C validation MLE={best:.3f}m", flush=True)
    else:
        model = load_order_model(order, device)
    if args.mode in {"eval", "train_eval"}:
        root, _, _ = output_paths(order)
        root.mkdir(parents=True, exist_ok=True)
        results = {
            "order": order,
            "train": ["route_A"],
            "validation": "route_C",
            "evaluation": list(args.eval_routes),
            "semantics": {
                "M": "MeanShift; forward 3x6 if first, otherwise centered 6x6",
                "G": "current-frame GRU position-error refiner",
                "K": "constant-velocity Kalman with dynamic R",
                "feedback": "no final position + GRU future delta",
            },
            "results": {},
        }
        for route_name in args.eval_routes:
            cache = route_c if route_name == "route_C" else build_cache(route_name, visual, device)
            summary = evaluate_route(
                order,
                model,
                visual,
                cache,
                device,
                save_csv=True,
                measure_latency=bool(args.measure_latency),
            )
            results["results"][route_name] = summary
            print(
                f"[{order}] {route_name}: MLE={summary['MLE_m']:.3f}m "
                f"P90={summary['P90_m']:.3f}m LSR@15={summary['LSR@15_pct']:.2f}%",
                flush=True,
            )
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{order}] summary: {summary_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=VALID_ORDERS, default="MKG")
    p.add_argument("--mode", choices=["prepare_visual", "train", "eval", "train_eval"], default="train_eval")
    p.add_argument("--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS))
    p.add_argument("--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS))
    p.add_argument("--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M))
    p.add_argument("--patience", type=int, default=int(config.EARLY_STOP_PATIENCE))
    p.add_argument("--train-visual-if-missing", action="store_true")
    p.add_argument("--resume-visual", action="store_true")
    p.add_argument("--resume-temporal", action="store_true")
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument(
        "--eval-routes",
        nargs="+",
        choices=config.ROUTE_NAMES,
        default=["route_C", "route_B"],
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
