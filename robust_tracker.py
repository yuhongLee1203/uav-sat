"""Train and evaluate Temporal Motion-Conditioned Retrieval (TMCR).

This file replaces the training-free Kalman/gating tracker.  After five GT
initialisation frames, evaluation is fully closed-loop:

    consecutive UAV spatial features -> learned local motion distribution
    learned motion centre -> fixed local SAT candidate lattice
    temporal query x candidate tokens -> learned candidate posterior
    learned uncertainty gate -> final position

No hand-written Mahalanobis rejection, displacement cap, recovery expansion,
or alpha-beta coefficient is used by the proposed method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import RouteDataset
from visual_localizer import CandidateBatch, FrozenVisualLocalizer
from visual_model import TemporalMotionRetriever


@dataclass
class RouteCache:
    name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    global_features: torch.Tensor
    spatial_features: torch.Tensor

    def __len__(self):
        return int(self.gt_xy.shape[0])


@dataclass
class SplitRange:
    start: int
    end: int

    @property
    def length(self):
        return max(0, self.end - self.start)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_frame_id(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def cache_dtype():
    return torch.float16 if config.FEATURE_CACHE_DTYPE == "float16" else torch.float32


@torch.no_grad()
def build_route_cache(
    root: Path,
    name: str,
    visual: FrozenVisualLocalizer,
    device: torch.device,
) -> RouteCache:
    """Encode every real frame; GT is never used to collapse or select frames."""
    dataset = RouteDataset(
        root,
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )
    frame_ids: List[int] = []
    gt_xy: List[torch.Tensor] = []
    global_features: List[torch.Tensor] = []
    spatial_features: List[torch.Tensor] = []

    batch_size = int(config.EVAL_BATCH_SIZE)
    for start in range(0, len(dataset), batch_size):
        items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        global_batch = visual.encode_uav_clip(uav)
        spatial_batch = visual.encode_uav_spatial(uav)

        global_features.append(global_batch.to("cpu", dtype=cache_dtype()))
        spatial_features.append(spatial_batch.to("cpu", dtype=cache_dtype()))
        gt_xy.extend(item["xy"].float().cpu() for item in items)
        frame_ids.extend(parse_frame_id(item["frame_id"]) for item in items)

    return RouteCache(
        name=name,
        frame_ids=torch.tensor(frame_ids, dtype=torch.long),
        gt_xy=torch.stack(gt_xy).float(),
        global_features=torch.cat(global_features, dim=0),
        spatial_features=torch.cat(spatial_features, dim=0),
    )


def contiguous_splits(length: int) -> Dict[str, SplitRange]:
    guard = int(config.SPLIT_GUARD_FRAMES)
    train_end = int(length * float(config.TRAIN_FRACTION))
    val_end = int(length * (float(config.TRAIN_FRACTION) + float(config.VAL_FRACTION)))

    train = SplitRange(0, max(0, train_end - guard))
    val = SplitRange(min(length, train_end + guard), max(min(length, train_end + guard), val_end - guard))
    test = SplitRange(min(length, val_end + guard), length)
    return {"train": train, "val": val, "test": test, "all": SplitRange(0, length)}


def sequence_windows(
    caches: Sequence[RouteCache],
    split_name: str,
    sequence_length: int,
    stride: int,
) -> List[Tuple[int, int, int]]:
    windows = []
    minimum = int(config.HISTORY) + 2
    for route_index, cache in enumerate(caches):
        segment = contiguous_splits(len(cache))[split_name]
        if segment.length < minimum:
            continue
        effective_length = min(int(sequence_length), segment.length)
        last_start = segment.end - effective_length
        for start in range(segment.start, last_start + 1, int(stride)):
            windows.append((route_index, start, start + effective_length))
        if not any(item[0] == route_index for item in windows):
            windows.append((route_index, segment.start, segment.end))
    return windows


def teacher_forcing_ratio(epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return float(config.TEACHER_FORCING_END)
    progress = epoch / float(epochs - 1)
    return float(config.TEACHER_FORCING_START) + progress * (
        float(config.TEACHER_FORCING_END) - float(config.TEACHER_FORCING_START)
    )


def stable_axes(velocity: torch.Tensor, fallback: torch.Tensor | None = None):
    """Construct trajectory-frame forward/lateral axes without GT."""
    speed = torch.linalg.norm(velocity, dim=1, keepdim=True)
    forward = velocity / speed.clamp_min(1e-6)
    if fallback is None:
        fallback = torch.zeros_like(forward)
        fallback[:, 0] = 1.0
    fallback = F.normalize(fallback, dim=1)
    forward = torch.where((speed > 1e-4).expand_as(forward), forward, fallback)
    lateral = torch.stack([-forward[:, 1], forward[:, 0]], dim=1)
    return forward, lateral, speed.squeeze(1)


def project_local(vector, forward, lateral):
    return torch.stack(
        [(vector * forward).sum(dim=1), (vector * lateral).sum(dim=1)], dim=1
    )


def local_to_map(local, forward, lateral):
    return local[:, :1] * forward + local[:, 1:2] * lateral


def random_jitter(batch_size: int, device: torch.device):
    radius = torch.sqrt(torch.rand(batch_size, 1, device=device)) * float(
        config.TEACHER_JITTER_M
    )
    angle = torch.rand(batch_size, 1, device=device) * (2.0 * math.pi)
    return torch.cat([radius * angle.cos(), radius * angle.sin()], dim=1)


def nearest_candidate_target(candidate_xy, gt_xy):
    distance_squared = (candidate_xy - gt_xy[:, None, :]).square().sum(dim=2)
    return distance_squared.argmin(dim=1)


def initialise_temporal_state(
    model: TemporalMotionRetriever,
    cache: RouteCache,
    start: int,
    device: torch.device,
):
    """Use the first five GT nodes only, as required by the protocol."""
    history = int(config.HISTORY)
    gt = cache.gt_xy[start : start + history].to(device)
    frame_ids = cache.frame_ids[start : start + history].to(device)
    global_features = cache.global_features[start : start + history].to(device).float()
    spatial_features = cache.spatial_features[start : start + history].to(device).float()

    hidden = model.initial_hidden(1, device)
    fallback_displacement = (gt[-1] - gt[0]).unsqueeze(0)
    fallback_axis = F.normalize(
        torch.where(
            torch.linalg.norm(fallback_displacement, dim=1, keepdim=True) > 1e-4,
            fallback_displacement,
            torch.tensor([[1.0, 0.0]], device=device),
        ),
        dim=1,
    )

    previous_local = torch.zeros(1, 2, device=device)
    previous_speed = torch.zeros(1, device=device)
    previous_velocity = torch.zeros(1, 2, device=device)

    for index in range(1, history):
        dt = (frame_ids[index] - frame_ids[index - 1]).clamp_min(1).float().view(1)
        forward, lateral, _ = stable_axes(previous_velocity, fallback_axis)
        hidden, _, _ = model.encode_motion(
            global_features[index - 1 : index],
            global_features[index : index + 1],
            spatial_features[index - 1 : index],
            spatial_features[index : index + 1],
            previous_local,
            previous_speed,
            dt,
            hidden,
        )
        displacement = (gt[index] - gt[index - 1]).view(1, 2)
        previous_velocity = displacement / dt[:, None]
        previous_local = project_local(displacement, forward, lateral)
        previous_speed = torch.linalg.norm(previous_velocity, dim=1)

    previous_xy = gt[-1:].clone()
    previous_previous_xy = gt[-2:-1].clone()
    previous_frame_id = frame_ids[-1:].clone()
    previous_previous_frame_id = frame_ids[-2:-1].clone()
    previous_velocity = (previous_xy - previous_previous_xy) / (
        previous_frame_id - previous_previous_frame_id
    ).clamp_min(1).float()[:, None]
    previous_gt_delta = (gt[-1] - gt[-2]).view(1, 2)

    return {
        "hidden": hidden,
        "previous_xy": previous_xy,
        "previous_previous_xy": previous_previous_xy,
        "previous_frame_id": previous_frame_id,
        "previous_velocity": previous_velocity,
        "previous_local": previous_local,
        "previous_gt_delta": previous_gt_delta,
        "fallback_axis": fallback_axis,
        "previous_prediction_delta": previous_xy - previous_previous_xy,
    }


def one_temporal_step(
    model: TemporalMotionRetriever,
    visual: FrozenVisualLocalizer,
    cache: RouteCache,
    index: int,
    state: dict,
    device: torch.device,
    teacher_ratio: float,
    training: bool,
):
    previous_index = index - 1
    previous_global = cache.global_features[previous_index : previous_index + 1].to(device).float()
    current_global = cache.global_features[index : index + 1].to(device).float()
    previous_spatial = cache.spatial_features[previous_index : previous_index + 1].to(device).float()
    current_spatial = cache.spatial_features[index : index + 1].to(device).float()
    gt_xy = cache.gt_xy[index : index + 1].to(device)
    previous_gt_xy = cache.gt_xy[previous_index : previous_index + 1].to(device)
    frame_id = cache.frame_ids[index : index + 1].to(device)
    dt = (frame_id - state["previous_frame_id"]).clamp_min(1).float()

    forward, lateral, previous_speed = stable_axes(
        state["previous_velocity"], state["fallback_axis"]
    )
    hidden, delta_local, motion_logvar = model.encode_motion(
        previous_global,
        current_global,
        previous_spatial,
        current_spatial,
        state["previous_local"],
        previous_speed,
        dt,
        state["hidden"],
    )
    motion_center = state["previous_xy"] + local_to_map(
        delta_local, forward, lateral
    )

    use_teacher = training and random.random() < float(teacher_ratio)
    candidate_center = (
        gt_xy + random_jitter(1, device)
        if use_teacher
        else motion_center.detach()
    )
    candidates = visual.candidate_batch(
        current_global, candidate_center, grid_size=config.GRID_SIZE
    )
    decoded = model.decode_candidates(
        hidden,
        candidates.z_uav,
        candidates.z_sat,
        candidates.raw_logits,
        candidates.raw_prob,
        candidates.centers,
        motion_center,
        forward,
        lateral,
        motion_logvar,
    )
    posterior_logits, posterior_prob, visual_expectation, final_xy = decoded[:4]
    correction_gate, entropy, spatial_std = decoded[4:]

    gt_delta = gt_xy - previous_gt_xy
    target_local = project_local(gt_delta, forward.detach(), lateral.detach())
    motion_nll = 0.5 * (
        torch.exp(-motion_logvar) * (delta_local - target_local).square()
        + motion_logvar
    ).mean()
    target_index = nearest_candidate_target(candidates.centers, gt_xy)
    candidate_loss = F.cross_entropy(posterior_logits, target_index)
    coordinate_loss = F.smooth_l1_loss(final_xy, gt_xy)

    prediction_delta = final_xy - state["previous_xy"]
    relative_loss = F.smooth_l1_loss(prediction_delta, gt_delta)
    predicted_acceleration = prediction_delta - state["previous_prediction_delta"]
    gt_acceleration = gt_delta - state["previous_gt_delta"]
    acceleration_loss = F.smooth_l1_loss(
        predicted_acceleration, gt_acceleration
    )

    final_error = torch.linalg.norm(final_xy - gt_xy, dim=1).detach()
    uncertainty_loss = F.smooth_l1_loss(
        torch.log1p(spatial_std), torch.log1p(final_error)
    )
    total_loss = (
        float(config.LOSS_COORD) * coordinate_loss
        + float(config.LOSS_CANDIDATE) * candidate_loss
        + float(config.LOSS_MOTION_NLL) * motion_nll
        + float(config.LOSS_RELATIVE) * relative_loss
        + float(config.LOSS_ACCELERATION) * acceleration_loss
        + float(config.LOSS_UNCERTAINTY) * uncertainty_loss
    )

    new_velocity = prediction_delta / dt[:, None]
    new_local = project_local(prediction_delta.detach(), forward, lateral)
    contains_gt = (
        torch.ones(1, dtype=torch.bool, device=device)
        if training
        else visual.candidate_contains_gt_anchor(candidates.indices, gt_xy)
    )

    next_state = {
        "hidden": hidden,
        "previous_xy": final_xy,
        "previous_previous_xy": state["previous_xy"],
        "previous_frame_id": frame_id,
        "previous_velocity": new_velocity,
        "previous_local": new_local,
        "previous_gt_delta": gt_delta.detach(),
        "fallback_axis": forward.detach(),
        "previous_prediction_delta": prediction_delta,
    }
    output = {
        "loss": total_loss,
        "loss_coord": coordinate_loss.detach(),
        "loss_candidate": candidate_loss.detach(),
        "loss_motion": motion_nll.detach(),
        "gt": gt_xy,
        "raw_top1": candidates.raw_top1_xy,
        "hardms": candidates.hardms_xy,
        "motion": motion_center,
        "visual_expectation": visual_expectation,
        "temporal": final_xy,
        "gate": correction_gate,
        "entropy": entropy,
        "spatial_std": spatial_std,
        "motion_std": torch.exp(0.5 * motion_logvar).mean(dim=1),
        "capture": contains_gt,
        "frame_id": frame_id,
        "teacher": use_teacher,
    }
    return next_state, output


def run_training_window(
    model,
    visual,
    cache,
    start,
    end,
    device,
    teacher_ratio,
):
    state = initialise_temporal_state(model, cache, start, device)
    losses = []
    diagnostics = []
    for index in range(start + int(config.HISTORY), end):
        state, output = one_temporal_step(
            model,
            visual,
            cache,
            index,
            state,
            device,
            teacher_ratio=teacher_ratio,
            training=True,
        )
        losses.append(output["loss"])
        diagnostics.append(output)
    if not losses:
        raise RuntimeError("Training window is shorter than HISTORY + 1")
    return torch.stack(losses).mean(), diagnostics


@torch.no_grad()
def validation_mle(model, visual, caches, windows, device):
    model.eval()
    errors = []
    for route_index, start, end in windows:
        cache = caches[route_index]
        state = initialise_temporal_state(model, cache, start, device)
        for index in range(start + int(config.HISTORY), end):
            state, output = one_temporal_step(
                model,
                visual,
                cache,
                index,
                state,
                device,
                teacher_ratio=0.0,
                training=False,
            )
            errors.append(
                torch.linalg.norm(output["temporal"] - output["gt"], dim=1)
                .cpu()
                .item()
            )
    return float(np.mean(errors)) if errors else float("inf")


def train_model(model, visual, caches, device, epochs):
    train_windows = sequence_windows(
        caches,
        "train",
        config.SEQUENCE_LENGTH,
        config.SEQUENCE_STRIDE,
    )
    val_windows = sequence_windows(
        caches,
        "val",
        config.SEQUENCE_LENGTH,
        config.SEQUENCE_STRIDE,
    )
    if not train_windows:
        raise RuntimeError("No training sequence was produced from the route splits")
    if not val_windows:
        val_windows = train_windows

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LR),
        weight_decay=float(config.WEIGHT_DECAY),
    )
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(int(epochs)):
        model.train()
        random.shuffle(train_windows)
        ratio = teacher_forcing_ratio(epoch, int(epochs))
        epoch_losses = []

        for route_index, start, end in train_windows:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = run_training_window(
                model,
                visual,
                caches[route_index],
                start,
                end,
                device,
                ratio,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation = validation_mle(
            model, visual, caches, val_windows, device
        )
        train_loss = float(np.mean(epoch_losses))
        print(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"loss={train_loss:.5f} val_mle={validation:.3f}m "
            f"teacher={ratio:.3f}",
            flush=True,
        )

        if validation < best_validation:
            best_validation = validation
            torch.save(
                {
                    "model": model.state_dict(),
                    "spatial_channels": int(caches[0].spatial_features.shape[1]),
                    "epoch": epoch + 1,
                    "validation_mle": validation,
                    "architecture": "TemporalMotionConditionedRetrieval",
                },
                config.TEMPORAL_CHECKPOINT,
            )

    print(
        f"best validation MLE={best_validation:.3f}m; "
        f"checkpoint={config.TEMPORAL_CHECKPOINT}",
        flush=True,
    )


def metric_block(prediction, gt):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    error = np.linalg.norm(prediction - gt, axis=1)
    if len(prediction) > 1:
        predicted_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)
        rpe = np.linalg.norm(predicted_step - gt_step, axis=1)
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = float(np.percentile(gt_step_length, 99)) + float(
            config.JUMP_TOLERANCE_M
        )
        jump_rate = float(
            (np.linalg.norm(predicted_step, axis=1) > jump_threshold).mean() * 100
        )
        if len(predicted_step) > 1:
            acceleration = np.linalg.norm(np.diff(predicted_step, axis=0), axis=1)
            acceleration_p90 = float(np.percentile(acceleration, 90))
        else:
            acceleration_p90 = 0.0
        path_ratio = float(
            np.linalg.norm(predicted_step, axis=1).sum()
            / max(np.linalg.norm(gt_step, axis=1).sum(), 1e-8)
        )
    else:
        rpe = np.zeros(1)
        jump_rate = 0.0
        acceleration_p90 = 0.0
        path_ratio = 0.0
        jump_threshold = 0.0

    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "ATE_RMSE_m": float(np.sqrt(np.mean(error**2))),
        "LSR@5_pct": float((error <= 5).mean() * 100),
        "LSR@10_pct": float((error <= 10).mean() * 100),
        "LSR@15_pct": float((error <= 15).mean() * 100),
        "LSR@20_pct": float((error <= 20).mean() * 100),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": jump_rate,
        "JumpThreshold_m": jump_threshold,
        "AccelerationP90_m": acceleration_p90,
        "PathLengthRatio": path_ratio,
        "MaxLE_m": float(error.max()),
    }


@torch.no_grad()
def evaluate_route(
    model,
    visual,
    cache,
    device,
    split_name,
):
    model.eval()
    segment = contiguous_splits(len(cache))[split_name]
    if segment.length < int(config.HISTORY) + 2:
        raise RuntimeError(
            f"{cache.name} {split_name} split is too short: {segment.length}"
        )

    state = initialise_temporal_state(model, cache, segment.start, device)
    history_slice = slice(segment.start, segment.start + int(config.HISTORY))
    history_gt = cache.gt_xy[history_slice].to(device)
    history_ids = cache.frame_ids[history_slice].to(device)
    total_dt = (history_ids[-1] - history_ids[0]).clamp_min(1).float()
    constant_velocity = (history_gt[-1] - history_gt[0]) / total_dt
    constant_xy = history_gt[-1].clone()
    constant_frame_id = history_ids[-1].clone()

    rows = []
    for index in range(segment.start + int(config.HISTORY), segment.end):
        current_frame_id = cache.frame_ids[index].to(device)
        constant_dt = (current_frame_id - constant_frame_id).clamp_min(1).float()
        constant_xy = constant_xy + constant_velocity * constant_dt
        constant_frame_id = current_frame_id

        state, output = one_temporal_step(
            model,
            visual,
            cache,
            index,
            state,
            device,
            teacher_ratio=0.0,
            training=False,
        )
        rows.append(
            {
                "frame_id": int(output["frame_id"].item()),
                "gt": output["gt"].squeeze(0).cpu().tolist(),
                "constant_velocity": constant_xy.cpu().tolist(),
                "raw_top1": output["raw_top1"].squeeze(0).cpu().tolist(),
                "hardms": output["hardms"].squeeze(0).cpu().tolist(),
                "motion": output["motion"].squeeze(0).cpu().tolist(),
                "visual_expectation": output["visual_expectation"].squeeze(0).cpu().tolist(),
                "temporal": output["temporal"].squeeze(0).cpu().tolist(),
                "gate": float(output["gate"].item()),
                "entropy": float(output["entropy"].item()),
                "visual_std": float(output["spatial_std"].item()),
                "motion_std": float(output["motion_std"].item()),
                "capture": bool(output["capture"].item()),
            }
        )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / f"{cache.name}_robust_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame_id",
                "gt_x",
                "gt_y",
                "constant_velocity_x",
                "constant_velocity_y",
                "raw_top1_x",
                "raw_top1_y",
                "hardms_x",
                "hardms_y",
                "motion_x",
                "motion_y",
                "visual_expectation_x",
                "visual_expectation_y",
                "temporal_x",
                "temporal_y",
                "correction_gate",
                "posterior_entropy",
                "visual_std_m",
                "motion_std_m",
                "candidate_capture",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["frame_id"],
                    *row["gt"],
                    *row["constant_velocity"],
                    *row["raw_top1"],
                    *row["hardms"],
                    *row["motion"],
                    *row["visual_expectation"],
                    *row["temporal"],
                    row["gate"],
                    row["entropy"],
                    row["visual_std"],
                    row["motion_std"],
                    int(row["capture"]),
                ]
            )

    gt = [row["gt"] for row in rows]
    summary = {
        "route": cache.name,
        "split": split_name,
        "ConstantVelocity": metric_block(
            [row["constant_velocity"] for row in rows], gt
        ),
        "RawTop1": metric_block([row["raw_top1"] for row in rows], gt),
        "FixedHardMS": metric_block([row["hardms"] for row in rows], gt),
        "LearnedMotionOnly": metric_block([row["motion"] for row in rows], gt),
        "TemporalVisualExpectation": metric_block(
            [row["visual_expectation"] for row in rows], gt
        ),
        "TMCR": metric_block([row["temporal"] for row in rows], gt),
        "CandidateCaptureRate_pct": float(
            np.mean([row["capture"] for row in rows]) * 100
        ),
        "MeanCorrectionGate": float(np.mean([row["gate"] for row in rows])),
        "MeanPosteriorEntropy": float(
            np.mean([row["entropy"] for row in rows])
        ),
        "MeanVisualStd_m": float(
            np.mean([row["visual_std"] for row in rows])
        ),
        "MeanMotionStd_m": float(
            np.mean([row["motion_std"] for row in rows])
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def load_temporal_checkpoint(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Temporal checkpoint not found: {config.TEMPORAL_CHECKPOINT}. "
            "Run --mode train_eval first."
        )
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def selected_routes(names: Iterable[str] | None):
    if not names:
        return list(zip(config.ROUTE_ROOTS, config.ROUTE_NAMES))
    requested = set(names)
    pairs = [
        (root, name)
        for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES)
        if name in requested
    ]
    missing = requested - {name for _, name in pairs}
    if missing:
        raise ValueError(f"Unknown routes: {sorted(missing)}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_eval"),
        default="train_eval",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument(
        "--eval-split", choices=("test", "all"), default="test"
    )
    parser.add_argument("--routes", nargs="*", default=None)
    args = parser.parse_args()

    set_seed(int(config.SEED))
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    visual = FrozenVisualLocalizer(device)

    caches = []
    for root, name in selected_routes(args.routes):
        print(f"encoding {name}: {root}", flush=True)
        caches.append(build_route_cache(root, name, visual, device))
        print(
            f"  frames={len(caches[-1])}, "
            f"spatial={tuple(caches[-1].spatial_features.shape[1:])}",
            flush=True,
        )

    if not caches:
        raise RuntimeError("No route cache was created")
    spatial_channels = int(caches[0].spatial_features.shape[1])
    model = TemporalMotionRetriever(spatial_channels).to(device)

    if args.mode in ("train", "train_eval"):
        train_model(model, visual, caches, device, args.epochs)

    if args.mode in ("eval", "train_eval"):
        checkpoint = load_temporal_checkpoint(model, device)
        print(
            f"loaded epoch={checkpoint.get('epoch')} "
            f"val_mle={checkpoint.get('validation_mle')}",
            flush=True,
        )
        summaries = [
            evaluate_route(
                model,
                visual,
                cache,
                device,
                split_name=args.eval_split,
            )
            for cache in caches
        ]
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = config.OUTPUT_DIR / "robust_tracker_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summaries, file, ensure_ascii=False, indent=2)
        print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()