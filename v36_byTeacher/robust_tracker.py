"""v36_byTeacher v8: causal MS1 -> Kalman -> MS2 tracker.

Frame t uses no current-frame reference/GT information as an estimator input.

    previous final X_{t-1} ---------------------------> MS1 center (forward 3x6)
    previous final X_{t-1} + predicted delta_{t-1} -> inertial prior X_pre_t
    current UAV image + MS1 --------------------------> visual measurement X_ms1_t
    X_pre_t + X_ms1_t -------------------------------> position Kalman -> X'_t
    X'_t --------------------------------------------> MS2 center (full 6x6)
    MS2 ----------------------------------------------> final X_t
    MS1 coordinate + temporal visual state ----------> GRU -> v_t, a_t, heading_t
    v_t, a_t, heading_t -----------------------------> polynomial delta_t
    X_t + delta_t -----------------------------------> next inertial prior X_pre_{t+1}

Reference positions are used only for Route-A supervision, Route-B model
selection/metrics, and Route-C final metrics.  There are no reference-dependent
search centers, no GT progress/step caps, and no hand-coded speed/acceleration/
heading/turn-rate limits in the inference path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker_base as b
from visual_localizer import (
    CandidateBatch,
    FrozenVisualLocalizer,
    regular_grid_indices,
    response_variance_xy,
    soft_mean_shift,
    train_visual_retrieval_a_only,
)
from visual_model import ThreeFrameRouteStateGRU

ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)

# Reuse only data/route/cache utilities from the legacy base module.  The old
# RouteKalman, controlled_gt_prior, cap_* and stabilize_* functions are not used.
WaypointRoute = b.WaypointRoute
RouteCache = b.RouteCache
build_route_cache = b.build_route_cache
load_waypoint_xy = b.load_waypoint_xy
resolve_device = b.resolve_device
set_seed = b.set_seed


def _tensor_xy(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, 2)


def _wrap_angle(value):
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))


def _angle_error(a, b):
    return _wrap_angle(float(a) - float(b))


def _metric_summary(errors):
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return {
            "MLE_m": float("inf"),
            "MedLE_m": float("inf"),
            "P90_m": float("inf"),
            "P95_m": float("inf"),
            "P99_m": float("inf"),
            "LSR@5_pct": 0.0,
            "LSR@10_pct": 0.0,
            "LSR@15_pct": 0.0,
            "LSR@20_pct": 0.0,
        }
    return {
        "MLE_m": float(values.mean()),
        "MedLE_m": float(np.median(values)),
        "P90_m": float(np.quantile(values, 0.90)),
        "P95_m": float(np.quantile(values, 0.95)),
        "P99_m": float(np.quantile(values, 0.99)),
        "LSR@5_pct": float((values <= 5.0).mean() * 100.0),
        "LSR@10_pct": float((values <= 10.0).mean() * 100.0),
        "LSR@15_pct": float((values <= 15.0).mean() * 100.0),
        "LSR@20_pct": float((values <= 20.0).mean() * 100.0),
    }


def _score_candidate_indices(visual, uav_clip, selected_indices):
    centers = visual.gallery["xy"][selected_indices]
    satellite_clip = visual.gallery["clip_feat"][selected_indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    raw_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    raw_prob = torch.softmax(raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1)
    raw_index = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=visual.device), raw_index
    ]
    softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(
        raw_logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    variance_xy = response_variance_xy(raw_prob, centers, softms_xy)
    return CandidateBatch(
        indices=selected_indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=softms_xy,
        softms_support=softms_support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
        softms_variance_xy=variance_xy,
    )


@torch.no_grad()
def ms1_forward_3x6(visual, uav_clip, center_xy, heading_rad):
    """MS1: score only the heading-forward 3x6 half of a local 6x6 lattice."""
    grid_size = int(config.MS1_BASE_GRID_SIZE)
    full_indices = regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        center_xy,
        grid_size,
        config.SAT_STRIDE,
        visual.device,
    )

    headings = torch.as_tensor(
        heading_rad, dtype=center_xy.dtype, device=center_xy.device
    ).reshape(-1)
    if headings.numel() == 1 and center_xy.shape[0] > 1:
        headings = headings.expand(center_xy.shape[0])
    if headings.numel() != center_xy.shape[0]:
        raise ValueError("heading count must match center batch size")

    forward_unit = torch.stack([torch.cos(headings), torch.sin(headings)], dim=1)
    cross_unit = torch.stack([-torch.sin(headings), torch.cos(headings)], dim=1)
    backshift = float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M)
    grid_center_xy = center_xy - backshift * forward_unit

    # Rebuild around the shifted center when a nonzero backshift is requested.
    if abs(backshift) > 1e-12:
        full_indices = regular_grid_indices(
            visual.gallery["xy"],
            visual.gallery["pixel"],
            visual.pixel_index,
            grid_center_xy,
            grid_size,
            config.SAT_STRIDE,
            visual.device,
        )

    full_centers = visual.gallery["xy"][full_indices]
    relative = full_centers - grid_center_xy[:, None, :]
    forward_projection = (relative * forward_unit[:, None, :]).sum(dim=2)
    cross_projection = (relative * cross_unit[:, None, :]).sum(dim=2)

    keep_count = int(config.MS1_CANDIDATE_COUNT)
    selected_local = torch.topk(
        forward_projection,
        k=keep_count,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    selected_forward = torch.gather(forward_projection, 1, selected_local)
    selected_cross = torch.gather(cross_projection, 1, selected_local)
    ordering_key = -selected_forward * 1000.0 + selected_cross
    order = torch.argsort(ordering_key, dim=1)
    selected_local = torch.gather(selected_local, 1, order)
    selected_indices = torch.gather(full_indices, 1, selected_local)
    return _score_candidate_indices(visual, uav_clip, selected_indices)


class PositionKalman:
    """Unconstrained 2D position Kalman used between MS1 and MS2.

    The prior mean is supplied explicitly by X_pre = previous_MS2_final +
    previous_GRU_delta.  No velocity state, speed clamp, turn clamp, innovation
    cap, GT cap, route-progress cap or final-step cap exists here.
    """

    def __init__(self):
        init_var = float(config.KALMAN_POSITION_INIT_VAR)
        self.P = np.eye(2, dtype=np.float64) * init_var
        self.Q = np.diag(
            [
                float(config.KALMAN_POSITION_PROCESS_VAR_X),
                float(config.KALMAN_POSITION_PROCESS_VAR_Y),
            ]
        ).astype(np.float64)
        self.last_prior = np.zeros(2, dtype=np.float64)
        self.last_measurement = np.zeros(2, dtype=np.float64)
        self.last_fused = np.zeros(2, dtype=np.float64)
        self.last_gain = np.eye(2, dtype=np.float64)

    def fuse(self, prior_xy, measurement_xy, measurement_variance_xy):
        prior = np.asarray(prior_xy, dtype=np.float64).reshape(2)
        measurement = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        variance = np.asarray(measurement_variance_xy, dtype=np.float64).reshape(2)
        eps = float(config.KALMAN_NUMERICAL_VARIANCE_EPS)
        R = np.diag(np.maximum(variance, eps))
        P_prior = self.P + self.Q
        S = P_prior + R
        K = P_prior @ np.linalg.inv(S)
        fused = prior + K @ (measurement - prior)
        self.P = (np.eye(2, dtype=np.float64) - K) @ P_prior
        self.last_prior = prior.copy()
        self.last_measurement = measurement.copy()
        self.last_fused = fused.copy()
        self.last_gain = K.copy()
        return fused

    def commit_ms2_final(self, final_xy, ms2_variance_xy):
        del final_xy  # mean is carried explicitly by tracker state
        variance = np.asarray(ms2_variance_xy, dtype=np.float64).reshape(2)
        eps = float(config.KALMAN_NUMERICAL_VARIANCE_EPS)
        self.P = np.diag(np.maximum(variance, eps))


@dataclass
class MotionTargets:
    velocity_xy: torch.Tensor
    acceleration_xy: torch.Tensor
    next_step_xy: torch.Tensor
    heading_rad: torch.Tensor
    turn_rate_rad: torch.Tensor
    valid_next: torch.Tensor


def build_motion_targets(cache, initial_heading_rad):
    """Derive v/a/heading supervision directly from reference XY positions.

    These tensors are labels only.  They are never passed to MS1, Kalman, MS2,
    the GRU input, or any inference-time state update.
    """
    p = cache.gt_xy.detach().cpu().double().numpy()
    n = len(p)
    velocity = np.zeros((n, 2), dtype=np.float64)
    acceleration = np.zeros((n, 2), dtype=np.float64)
    next_step = np.zeros((n, 2), dtype=np.float64)
    valid_next = np.zeros(n, dtype=np.bool_)

    if n >= 2:
        velocity[0] = p[1] - p[0]
        velocity[-1] = p[-1] - p[-2]
        next_step[:-1] = p[1:] - p[:-1]
        valid_next[:-1] = True
    if n >= 3:
        velocity[1:-1] = 0.5 * (p[2:] - p[:-2])
        acceleration[1:-1] = p[2:] - 2.0 * p[1:-1] + p[:-2]
        acceleration[0] = acceleration[1]
        acceleration[-1] = acceleration[-2]
    elif n == 2:
        acceleration[:] = 0.0

    heading = np.zeros(n, dtype=np.float64)
    last_heading = float(initial_heading_rad)
    for index in range(n):
        if valid_next[index] and np.linalg.norm(next_step[index]) > 1e-9:
            last_heading = math.atan2(next_step[index, 1], next_step[index, 0])
        heading[index] = last_heading

    turn_rate = np.zeros(n, dtype=np.float64)
    for index in range(1, n):
        turn_rate[index] = _angle_error(heading[index], heading[index - 1])

    return MotionTargets(
        velocity_xy=torch.tensor(velocity, dtype=torch.float32),
        acceleration_xy=torch.tensor(acceleration, dtype=torch.float32),
        next_step_xy=torch.tensor(next_step, dtype=torch.float32),
        heading_rad=torch.tensor(heading, dtype=torch.float32).reshape(-1, 1),
        turn_rate_rad=torch.tensor(turn_rate, dtype=torch.float32).reshape(-1, 1),
        valid_next=torch.tensor(valid_next, dtype=torch.bool),
    )


def motion_supervision_loss(output, targets, index, device):
    target_v = targets.velocity_xy[index : index + 1].to(device)
    target_a = targets.acceleration_xy[index : index + 1].to(device)
    target_heading = targets.heading_rad[index : index + 1].to(device)
    target_turn = targets.turn_rate_rad[index : index + 1].to(device)

    velocity_loss = F.smooth_l1_loss(output.velocity_xy, target_v)
    acceleration_loss = F.smooth_l1_loss(output.acceleration_xy, target_a)
    heading_delta = torch.atan2(
        torch.sin(output.heading_rad - target_heading),
        torch.cos(output.heading_rad - target_heading),
    )
    heading_loss = (1.0 - torch.cos(heading_delta)).mean()
    turn_loss = F.smooth_l1_loss(output.turn_rate_rad, target_turn)

    if bool(targets.valid_next[index]):
        target_step = targets.next_step_xy[index : index + 1].to(device)
        next_step_loss = F.smooth_l1_loss(output.next_step_xy, target_step)
    else:
        next_step_loss = output.next_step_xy.sum() * 0.0

    total = (
        float(config.LOSS_VELOCITY) * velocity_loss
        + float(config.LOSS_ACCELERATION) * acceleration_loss
        + float(config.LOSS_NEXT_STEP) * next_step_loss
        + float(config.LOSS_HEADING) * heading_loss
        + float(config.LOSS_TURN_RATE) * turn_loss
    )
    return total, {
        "velocity": float(velocity_loss.detach().cpu()),
        "acceleration": float(acceleration_loss.detach().cpu()),
        "next_step": float(next_step_loss.detach().cpu()),
        "heading": float(heading_loss.detach().cpu()),
        "turn_rate": float(turn_loss.detach().cpu()),
    }


@dataclass
class TrackerState:
    final_xy: np.ndarray
    previous_delta_xy: torch.Tensor
    previous_ms1_xy: torch.Tensor | None
    previous_z_uav: torch.Tensor | None
    previous_velocity_xy: torch.Tensor
    previous_acceleration_xy: torch.Tensor
    previous_heading_state: torch.Tensor
    hidden: torch.Tensor | None
    kalman: PositionKalman


def _initial_heading(route):
    delta = np.asarray(route.points[1] - route.points[0], dtype=np.float64)
    if float(np.linalg.norm(delta)) <= 1e-12:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


def initialize_tracker(route, device):
    start_xy = np.asarray(route.points[0], dtype=np.float64).copy()
    heading = _initial_heading(route)
    return TrackerState(
        final_xy=start_xy,
        previous_delta_xy=torch.zeros(1, 2, dtype=torch.float32, device=device),
        previous_ms1_xy=None,
        previous_z_uav=None,
        previous_velocity_xy=torch.zeros(1, 2, dtype=torch.float32, device=device),
        previous_acceleration_xy=torch.zeros(1, 2, dtype=torch.float32, device=device),
        previous_heading_state=torch.tensor(
            [[heading, 0.0]], dtype=torch.float32, device=device
        ),
        hidden=None,
        kalman=PositionKalman(),
    )


def detach_tracker_graph(state):
    state.previous_delta_xy = state.previous_delta_xy.detach()
    state.previous_velocity_xy = state.previous_velocity_xy.detach()
    state.previous_acceleration_xy = state.previous_acceleration_xy.detach()
    state.previous_heading_state = state.previous_heading_state.detach()
    if state.hidden is not None:
        state.hidden = state.hidden.detach()
    if state.previous_z_uav is not None:
        state.previous_z_uav = state.previous_z_uav.detach()
    if state.previous_ms1_xy is not None:
        state.previous_ms1_xy = state.previous_ms1_xy.detach()
    return state


@dataclass
class FrameResult:
    previous_final_xy: np.ndarray
    inertial_prior_xy: np.ndarray
    ms1: CandidateBatch
    kalman_fused_xy: np.ndarray
    ms2: CandidateBatch
    final_xy: np.ndarray
    output: object
    search_heading_rad: float


def tracking_step(model, visual, uav_clip, state, device):
    """Run one frame without reading any reference/GT position."""
    previous_final = np.asarray(state.final_xy, dtype=np.float64).copy()
    center_tensor = _tensor_xy(previous_final, device)
    search_heading = float(state.previous_heading_state[0, 0].detach().cpu())

    ms1 = ms1_forward_3x6(
        visual=visual,
        uav_clip=uav_clip,
        center_xy=center_tensor,
        heading_rad=search_heading,
    )
    ms1_xy = ms1.softms_xy

    output = model.forward_step(
        z_uav=ms1.z_uav,
        previous_z_uav=state.previous_z_uav,
        ms1_xy=ms1_xy,
        previous_ms1_xy=state.previous_ms1_xy,
        previous_final_xy=center_tensor,
        previous_velocity_xy=state.previous_velocity_xy,
        previous_acceleration_xy=state.previous_acceleration_xy,
        previous_heading_state=state.previous_heading_state,
        hidden=state.hidden,
    )

    inertial_prior = previous_final + state.previous_delta_xy.detach().cpu().numpy()[0]
    ms1_xy_np = ms1_xy[0].detach().cpu().numpy().astype(np.float64)
    ms1_var_np = (
        ms1.softms_variance_xy[0].detach().cpu().numpy().astype(np.float64)
    )
    fused_xy = state.kalman.fuse(inertial_prior, ms1_xy_np, ms1_var_np)

    ms2 = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=_tensor_xy(fused_xy, device),
        grid_size=int(config.MS2_GRID_SIZE),
    )
    final_xy = ms2.softms_xy[0].detach().cpu().numpy().astype(np.float64)
    ms2_var_np = (
        ms2.softms_variance_xy[0].detach().cpu().numpy().astype(np.float64)
    )
    state.kalman.commit_ms2_final(final_xy, ms2_var_np)

    state.final_xy = final_xy
    state.previous_delta_xy = output.next_step_xy
    state.previous_ms1_xy = ms1_xy.detach()
    state.previous_z_uav = ms1.z_uav.detach()
    state.previous_velocity_xy = output.velocity_xy
    state.previous_acceleration_xy = output.acceleration_xy
    state.previous_heading_state = torch.cat(
        [output.heading_rad, output.turn_rate_rad], dim=1
    )
    state.hidden = output.hidden

    return FrameResult(
        previous_final_xy=previous_final,
        inertial_prior_xy=inertial_prior,
        ms1=ms1,
        kalman_fused_xy=fused_xy,
        ms2=ms2,
        final_xy=final_xy,
        output=output,
        search_heading_rad=search_heading,
    )


def _candidate_capture(candidate, reference_xy):
    centers = candidate.centers[0].detach().cpu().numpy()
    distance = np.linalg.norm(centers - np.asarray(reference_xy)[None, :], axis=1)
    return bool(distance.min() <= float(config.CANDIDATE_CAPTURE_RADIUS_M))


@torch.no_grad()
def evaluate_closed_loop(model, visual, cache, route, device, collect_rows=False):
    model.eval()
    state = initialize_tracker(route, device)
    targets = build_motion_targets(cache, _initial_heading(route))

    final_errors = []
    ms1_errors = []
    prior_errors = []
    kalman_errors = []
    speed_errors = []
    acceleration_errors = []
    heading_errors = []
    ms1_capture = []
    ms2_capture = []
    rows = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        result = tracking_step(model, visual, uav_clip, state, device)

        # Reference is read only after the estimator has produced every output.
        reference = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        final_errors.append(float(np.linalg.norm(result.final_xy - reference)))
        ms1_xy = result.ms1.softms_xy[0].cpu().numpy().astype(np.float64)
        ms1_errors.append(float(np.linalg.norm(ms1_xy - reference)))
        prior_errors.append(float(np.linalg.norm(result.inertial_prior_xy - reference)))
        kalman_errors.append(float(np.linalg.norm(result.kalman_fused_xy - reference)))

        target_v = targets.velocity_xy[index].numpy().astype(np.float64)
        target_a = targets.acceleration_xy[index].numpy().astype(np.float64)
        pred_v = result.output.velocity_xy[0].cpu().numpy().astype(np.float64)
        pred_a = result.output.acceleration_xy[0].cpu().numpy().astype(np.float64)
        speed_errors.append(float(np.linalg.norm(pred_v - target_v)))
        acceleration_errors.append(float(np.linalg.norm(pred_a - target_a)))
        heading_errors.append(
            abs(
                math.degrees(
                    _angle_error(
                        float(result.output.heading_rad[0, 0].cpu()),
                        float(targets.heading_rad[index, 0]),
                    )
                )
            )
        )
        ms1_capture.append(_candidate_capture(result.ms1, reference))
        ms2_capture.append(_candidate_capture(result.ms2, reference))

        if collect_rows:
            ms1_var = result.ms1.softms_variance_xy[0].cpu().numpy()
            ms2_var = result.ms2.softms_variance_xy[0].cpu().numpy()
            delta = result.output.next_step_xy[0].cpu().numpy()
            rows.append(
                {
                    "frame_id": int(cache.frame_ids[index]),
                    "image_path": cache.image_paths[index],
                    "reference_x": float(reference[0]),
                    "reference_y": float(reference[1]),
                    "previous_final_x": float(result.previous_final_xy[0]),
                    "previous_final_y": float(result.previous_final_xy[1]),
                    "inertial_prior_x": float(result.inertial_prior_xy[0]),
                    "inertial_prior_y": float(result.inertial_prior_xy[1]),
                    "ms1_x": float(ms1_xy[0]),
                    "ms1_y": float(ms1_xy[1]),
                    "ms1_var_x": float(ms1_var[0]),
                    "ms1_var_y": float(ms1_var[1]),
                    "kalman_fused_x": float(result.kalman_fused_xy[0]),
                    "kalman_fused_y": float(result.kalman_fused_xy[1]),
                    "ms2_final_x": float(result.final_xy[0]),
                    "ms2_final_y": float(result.final_xy[1]),
                    "ms2_var_x": float(ms2_var[0]),
                    "ms2_var_y": float(ms2_var[1]),
                    "gru_velocity_x": float(pred_v[0]),
                    "gru_velocity_y": float(pred_v[1]),
                    "gru_acceleration_x": float(pred_a[0]),
                    "gru_acceleration_y": float(pred_a[1]),
                    "gru_heading_deg": float(
                        math.degrees(float(result.output.heading_rad[0, 0].cpu()))
                    ),
                    "gru_turn_rate_deg": float(
                        math.degrees(float(result.output.turn_rate_rad[0, 0].cpu()))
                    ),
                    "polynomial_delta_x": float(delta[0]),
                    "polynomial_delta_y": float(delta[1]),
                    "search_heading_deg": float(math.degrees(result.search_heading_rad)),
                    "error_prior_m": float(prior_errors[-1]),
                    "error_ms1_m": float(ms1_errors[-1]),
                    "error_kalman_m": float(kalman_errors[-1]),
                    "error_final_m": float(final_errors[-1]),
                    "ms1_capture": int(ms1_capture[-1]),
                    "ms2_capture": int(ms2_capture[-1]),
                }
            )

    summary = _metric_summary(final_errors)
    summary.update(
        {
            "MS1_MLE_m": float(np.mean(ms1_errors)),
            "Prior_MLE_m": float(np.mean(prior_errors)),
            "Kalman_MLE_m": float(np.mean(kalman_errors)),
            "VelocityVector_MAE_m_per_frame": float(np.mean(speed_errors)),
            "AccelerationVector_MAE_m_per_frame2": float(np.mean(acceleration_errors)),
            "Heading_MAE_deg": float(np.mean(heading_errors)),
            "MS1_Capture_pct": float(np.mean(ms1_capture) * 100.0),
            "MS2_Capture_pct": float(np.mean(ms2_capture) * 100.0),
        }
    )
    return summary, rows


def train_temporal_model(
    visual,
    train_cache,
    train_route,
    val_cache,
    val_route,
    device,
    epochs,
    patience_limit,
    resume=False,
):
    model = ThreeFrameRouteStateGRU().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    train_targets = build_motion_targets(train_cache, _initial_heading(train_route))

    start_epoch = 1
    best_score = float("inf")
    best_state = None
    patience = 0
    if resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        if payload.get("architecture") != ARCHITECTURE_NAME:
            raise RuntimeError("latest temporal checkpoint architecture mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        state = initialize_tracker(train_route, device)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []

        for index in range(len(train_cache)):
            uav_clip = train_cache.uav_clip[index : index + 1].to(device).float()
            result = tracking_step(model, visual, uav_clip, state, device)
            loss, _ = motion_supervision_loss(
                result.output, train_targets, index, device
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite temporal loss at epoch={epoch} frame={index}"
                )
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            if chunk_count >= int(config.TBPTT_STEPS) or index + 1 >= len(train_cache):
                normalized = chunk_loss / float(chunk_count)
                normalized.backward()
                torch.nn.utils.clip_grad_norm_(params, float(config.GRAD_CLIP_NORM))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_losses.append(float(normalized.detach().cpu()))
                state = detach_tracker_graph(state)
                chunk_loss = None
                chunk_count = 0

        val, _ = evaluate_closed_loop(
            model, visual, val_cache, val_route, device, collect_rows=False
        )
        score = float(val["MLE_m"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)
        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "model": best_state,
                    "epoch": epoch,
                    "validation_route": config.VALIDATION_ROUTE_NAME,
                    "validation": val,
                    "train_route": config.TRAIN_ROUTE_NAME,
                    "test_route": config.TEST_ROUTE_NAME,
                    "reference_as_inference_input": False,
                    "hard_motion_limits": False,
                    "flow": "MS1 -> Kalman(prior+MS1) -> MS2; GRU predicts next delta",
                },
                config.TEMPORAL_CHECKPOINT,
            )
        else:
            patience += 1

        torch.save(
            {
                "architecture": ARCHITECTURE_NAME,
                "model": model.state_dict(),
                "best_model": best_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "patience": patience,
                "train_route": config.TRAIN_ROUTE_NAME,
                "validation_route": config.VALIDATION_ROUTE_NAME,
                "test_route": config.TEST_ROUTE_NAME,
                "reference_as_inference_input": False,
                "hard_motion_limits": False,
            },
            config.LATEST_TEMPORAL_CHECKPOINT,
        )

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(
            f"temporal epoch={epoch:03d}/{epochs} loss={mean_loss:.5f} "
            f"B_MLE={val['MLE_m']:.3f}m B_P90={val['P90_m']:.3f}m "
            f"B_LSR15={val['LSR@15_pct']:.2f}% "
            f"best={best_score:.3f} patience={patience}/{patience_limit}",
            flush=True,
        )

        if (
            epoch >= int(config.EARLY_STOP_MIN_EPOCH)
            and patience >= int(patience_limit)
        ):
            print("temporal early stopping on Route B", flush=True)
            break

    if best_state is None:
        raise RuntimeError("temporal training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


def load_temporal_model(device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.TEMPORAL_CHECKPOINT)
    payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            f"checkpoint architecture mismatch: {payload.get('architecture')}"
        )
    if payload.get("reference_as_inference_input") is not False:
        raise RuntimeError("checkpoint does not prove reference-free inference")
    if payload.get("hard_motion_limits") is not False:
        raise RuntimeError("checkpoint was trained with hard motion limits")
    model = ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _route_objects(route_name, visual, device):
    route_index = config.ROUTE_NAMES.index(route_name)
    cache = build_route_cache(
        route_name,
        config.ROUTE_ROOTS[route_index],
        visual,
        device,
    )
    route = WaypointRoute(
        load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
    )
    return cache, route


def train_pipeline(args, device):
    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=0.0,
            resume=bool(args.resume_visual),
        )

    visual = FrozenVisualLocalizer(device)
    train_cache, train_route = _route_objects(config.TRAIN_ROUTE_NAME, visual, device)
    val_cache, val_route = _route_objects(config.VALIDATION_ROUTE_NAME, visual, device)

    _, score = train_temporal_model(
        visual=visual,
        train_cache=train_cache,
        train_route=train_route,
        val_cache=val_cache,
        val_route=val_route,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume_temporal),
    )
    print(f"best whole-Route-B validation MLE={score:.3f}m", flush=True)


def run_route_inference(route_name, role, visual, model, cache, route, device, measure_latency=False):
    timing_ms = []
    warmup = int(config.LATENCY_WARMUP_FRAMES)

    if measure_latency:
        # Timed run is separate from the metric run so metrics remain simple and
        # deterministic.  No reference is consulted inside the timed loop.
        state = initialize_tracker(route, device)
        model.eval()
        with torch.no_grad():
            for index in range(len(cache)):
                uav_clip = cache.uav_clip[index : index + 1].to(device).float()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                tracking_step(model, visual, uav_clip, state, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing_ms.append((time.perf_counter() - start) * 1000.0)

    summary, rows = evaluate_closed_loop(
        model, visual, cache, route, device, collect_rows=True
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / f"{role}_{route_name}_v36_v8_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary["Route"] = route_name
    summary["Role"] = role
    summary["CSV"] = str(csv_path)
    summary["ReferenceAsInferenceInput"] = False
    summary["HardMotionLimits"] = False
    summary["MS1"] = "forward 3x6 centered at previous MS2 final"
    summary["Kalman"] = "2D position fusion of inertial prior and MS1 visual position"
    summary["MS2"] = "full 6x6 centered at Kalman fused position"

    if timing_ms:
        steady = np.asarray(timing_ms[warmup:], dtype=np.float64)
        if steady.size == 0:
            steady = np.asarray(timing_ms, dtype=np.float64)
        mean_ms = float(steady.mean())
        summary["TrackingCoreTiming"] = {
            "definition": "cached UAV backbone -> MS1 -> GRU -> Kalman -> MS2 -> final XY",
            "warmup_frames": int(min(warmup, len(timing_ms))),
            "samples": int(steady.size),
            "mean_ms": mean_ms,
            "median_ms": float(np.median(steady)),
            "p90_ms": float(np.quantile(steady, 0.90)),
            "fps": float(1000.0 / max(mean_ms, 1e-12)),
        }

    timing_text = ""
    if "TrackingCoreTiming" in summary:
        timing_text = (
            f" core={summary['TrackingCoreTiming']['mean_ms']:.2f}ms "
            f"FPS={summary['TrackingCoreTiming']['fps']:.2f}"
        )
    print(
        f"{role} {route_name}: MLE={summary['MLE_m']:.3f}m "
        f"P90={summary['P90_m']:.3f}m LSR@15={summary['LSR@15_pct']:.2f}%"
        f"{timing_text}",
        flush=True,
    )
    return summary


def eval_pipeline(args, device):
    visual = FrozenVisualLocalizer(device)
    model = load_temporal_model(device)

    val_cache, val_route = _route_objects(config.VALIDATION_ROUTE_NAME, visual, device)
    test_cache, test_route = _route_objects(config.TEST_ROUTE_NAME, visual, device)

    validation = run_route_inference(
        config.VALIDATION_ROUTE_NAME,
        "validation",
        visual,
        model,
        val_cache,
        val_route,
        device,
        measure_latency=bool(args.measure_latency),
    )
    test = run_route_inference(
        config.TEST_ROUTE_NAME,
        "test",
        visual,
        model,
        test_cache,
        test_route,
        device,
        measure_latency=bool(args.measure_latency),
    )

    payload = {
        "architecture": ARCHITECTURE_NAME,
        "protocol": config.PROTOCOL_NAME,
        "train_route": config.TRAIN_ROUTE_NAME,
        "validation_route": config.VALIDATION_ROUTE_NAME,
        "test_route": config.TEST_ROUTE_NAME,
        "reference_as_inference_input": False,
        "hard_motion_limits": False,
        "validation": validation,
        "test": test,
    }
    summary_path = config.OUTPUT_DIR / "A_train_B_validation_C_test_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary: {summary_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "eval", "train_eval"], default="train_eval"
    )
    parser.add_argument("--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS))
    parser.add_argument("--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS))
    parser.add_argument("--patience", type=int, default=int(config.EARLY_STOP_PATIENCE))
    parser.add_argument("--reuse-visual", action="store_true")
    parser.add_argument("--resume-visual", action="store_true")
    parser.add_argument("--resume-temporal", action="store_true")
    parser.add_argument("--measure-latency", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = resolve_device()

    print("=" * 100, flush=True)
    print(ARCHITECTURE_NAME, flush=True)
    print(config.PROTOCOL_NAME, flush=True)
    print(
        "Reference/GT estimator input: OFF | hard speed/acceleration/heading/turn limits: OFF",
        flush=True,
    )
    print(f"output={config.OUTPUT_DIR}", flush=True)
    print("=" * 100, flush=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "train_eval"):
        train_pipeline(args, device)
    if args.mode in ("eval", "train_eval"):
        eval_pipeline(args, device)


if __name__ == "__main__":
    main()
