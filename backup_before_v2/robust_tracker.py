"""Leakage-free, turn-aware straight-line HardMS tracker.

Main corrections over the previous version:
1. Frames are sampled by image order only; future GT never selects frames.
2. Velocity is metres per original frame ID and prediction uses the real dt.
3. Fixed HardMS retains Top-M spatial modes.
4. A grid-size-invariant concentration score selects the temporal mode.
5. Position measurements update both position and velocity.
6. Recovery searches a corridor from the last reliable position to prediction.
7. A real turn is allowed; the old heading is not enforced forever.
"""
from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

import config
from data import RouteDataset
from visual_localizer import FrozenVisualLocalizer, VisualMeasurement


INIT_HISTORY = int(config.HISTORY)
EPS = 1e-8


def tracking_frames(root, origin_lat, origin_lon):
    """Select frames without consulting GT motion.

    The old motion_nodes() grouped frames according to future GPS displacement,
    which leaked GT into closed-loop inference and created irregular, unknown
    node intervals. Here selection depends only on image order.
    """
    dataset = RouteDataset(
        root,
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    stride = max(1, int(config.TRACK_FRAME_STRIDE))
    indices = list(range(0, len(dataset), stride))
    if indices[-1] != len(dataset) - 1:
        indices.append(len(dataset) - 1)
    return dataset, indices


def robust_initial_velocity(
    gt_init: np.ndarray,
    frame_ids: np.ndarray,
) -> np.ndarray:
    """Robustly estimate metres per original frame ID from five GT nodes."""
    slopes = []
    for i in range(len(gt_init)):
        for j in range(i + 1, len(gt_init)):
            dt = float(frame_ids[j] - frame_ids[i])
            if dt <= 0:
                continue
            slopes.append((gt_init[j] - gt_init[i]) / dt)

    if not slopes:
        return np.zeros(2, dtype=np.float32)

    velocity = np.median(np.asarray(slopes, dtype=np.float64), axis=0)
    speed = float(np.linalg.norm(velocity))
    if speed < float(config.MIN_NONZERO_SPEED_M_PER_FRAME):
        return np.zeros(2, dtype=np.float32)
    if speed > float(config.MAX_SPEED_M_PER_FRAME):
        velocity *= float(config.MAX_SPEED_M_PER_FRAME) / max(speed, EPS)
    return velocity.astype(np.float32)


def robust_history_velocity(
    positions: deque,
    times: deque,
    confidences: deque,
    fallback: torch.Tensor,
):
    """Fit a short straight line with real frame times and confidence weights."""
    if len(positions) < 3:
        return fallback

    pos = torch.stack(list(positions), dim=0)
    t = torch.tensor(
        list(times), device=pos.device, dtype=pos.dtype
    )
    w = torch.tensor(
        list(confidences), device=pos.device, dtype=pos.dtype
    ).clamp(min=0.05, max=1.0)

    t = t - (w * t).sum() / w.sum().clamp_min(EPS)
    mean = (w[:, None] * pos).sum(dim=0) / w.sum().clamp_min(EPS)
    numerator = (w[:, None] * t[:, None] * (pos - mean)).sum(dim=0)
    denominator = (w * t.square()).sum().clamp_min(EPS)
    velocity = numerator / denominator

    if not torch.isfinite(velocity).all() or velocity.norm() < 1e-6:
        return fallback
    return limit_speed(velocity)


def limit_speed(velocity: torch.Tensor):
    speed = velocity.norm()
    maximum = float(config.MAX_SPEED_M_PER_FRAME)
    if speed > maximum:
        velocity = velocity * (maximum / speed.clamp_min(EPS))
    return velocity


def constrain_turn(
    previous_velocity: torch.Tensor,
    candidate_velocity: torch.Tensor,
    dt: float,
    recovery: bool,
):
    """Limit one-frame heading rotation without freezing the old direction."""
    candidate_velocity = limit_speed(candidate_velocity)
    prev_speed = previous_velocity.norm()
    cand_speed = candidate_velocity.norm()
    if prev_speed < 1e-5 or cand_speed < 1e-5:
        return candidate_velocity

    prev_dir = previous_velocity / prev_speed
    cand_dir = candidate_velocity / cand_speed
    dot = torch.clamp((prev_dir * cand_dir).sum(), -1.0, 1.0)
    angle = torch.acos(dot)
    per_frame = (
        config.RECOVERY_MAX_TURN_DEG_PER_FRAME
        if recovery
        else config.MAX_TURN_DEG_PER_FRAME
    )
    max_angle = torch.deg2rad(
        torch.tensor(
            min(179.0, float(per_frame) * max(float(dt), 1.0)),
            device=previous_velocity.device,
            dtype=previous_velocity.dtype,
        )
    )
    if angle <= max_angle:
        return candidate_velocity

    cross = prev_dir[0] * cand_dir[1] - prev_dir[1] * cand_dir[0]
    sign = torch.sign(cross)
    if sign == 0:
        sign = torch.tensor(
            1.0, device=previous_velocity.device, dtype=previous_velocity.dtype
        )
    signed = sign * max_angle
    c, s = torch.cos(signed), torch.sin(signed)
    direction = torch.stack(
        [
            c * prev_dir[0] - s * prev_dir[1],
            s * prev_dir[0] + c * prev_dir[1],
        ]
    )
    return direction * cand_speed


def predict(
    state: torch.Tensor,
    covariance: torch.Tensor,
    dt: float,
):
    dtype, device = state.dtype, state.device
    dt_tensor = torch.tensor(float(dt), device=device, dtype=dtype)

    transition = torch.eye(4, device=device, dtype=dtype)
    transition[0, 2] = dt_tensor
    transition[1, 3] = dt_tensor

    # White-noise acceleration model. The exact scale is less important than
    # preserving position-velocity cross covariance for visual velocity updates.
    q_pos = float(config.PROCESS_POSITION_STD_M) ** 2 * max(float(dt), 1.0)
    q_vel = (
        float(config.PROCESS_VELOCITY_STD_M_PER_FRAME) ** 2
        * max(float(dt), 1.0)
    )
    process = torch.diag(
        torch.tensor([q_pos, q_pos, q_vel, q_vel], device=device, dtype=dtype)
    )

    predicted_state = transition @ state
    predicted_cov = transition @ covariance @ transition.T + process
    max_var = float(config.MAX_POSITION_STD_M) ** 2
    predicted_cov[0, 0] = predicted_cov[0, 0].clamp_max(max_var)
    predicted_cov[1, 1] = predicted_cov[1, 1].clamp_max(max_var)
    predicted_cov = 0.5 * (predicted_cov + predicted_cov.T)
    return predicted_state, predicted_cov


def choose_search(lost_streak: int, covariance: torch.Tensor):
    position_std = float(
        torch.sqrt(torch.diagonal(covariance[:2, :2]).max()).item()
    )
    if (
        lost_streak >= int(config.LOST_STREAK_RECOVERY)
        or position_std >= 24.0
    ):
        return int(config.GRID_SIZE_RECOVERY), "recovery"
    if (
        lost_streak >= int(config.LOST_STREAK_MEDIUM)
        or position_std >= 11.0
    ):
        return int(config.GRID_SIZE_MEDIUM), "medium"
    return int(config.GRID_SIZE), "normal"


def build_search_centers(
    predicted_xy: torch.Tensor,
    last_reliable_xy: torch.Tensor,
    level: str,
):
    if level == "normal":
        centers = [predicted_xy]
    elif level == "medium":
        centers = [predicted_xy, 0.5 * (predicted_xy + last_reliable_xy)]
    else:
        centers = [
            predicted_xy,
            0.5 * (predicted_xy + last_reliable_xy),
            last_reliable_xy,
        ]

    unique = []
    for centre in centers:
        if not any(
            float(torch.norm(centre - old).item())
            < 0.5 * float(config.CANDIDATE_SPACING_M)
            for old in unique
        ):
            unique.append(centre)
    return torch.stack(unique[: int(config.MAX_SEARCH_CENTERS)], dim=0)


def search_parameters(level: str):
    if level == "recovery":
        return (
            float(config.RECOVERY_SIGMA_ALONG_M),
            float(config.RECOVERY_SIGMA_CROSS_M),
            float(config.MOTION_PRIOR_WEIGHT_RECOVERY),
            float(config.MOTION_SCORE_WEIGHT_RECOVERY),
        )
    if level == "medium":
        return (
            0.5 * (
                float(config.MOTION_SIGMA_ALONG_M)
                + float(config.RECOVERY_SIGMA_ALONG_M)
            ),
            0.5 * (
                float(config.MOTION_SIGMA_CROSS_M)
                + float(config.RECOVERY_SIGMA_CROSS_M)
            ),
            float(config.MOTION_PRIOR_WEIGHT_MEDIUM),
            float(config.MOTION_SCORE_WEIGHT_MEDIUM),
        )
    return (
        float(config.MOTION_SIGMA_ALONG_M),
        float(config.MOTION_SIGMA_CROSS_M),
        float(config.MOTION_PRIOR_WEIGHT_NORMAL),
        float(config.MOTION_SCORE_WEIGHT_NORMAL),
    )


def motion_components(
    point_xy: torch.Tensor,
    predicted_state: torch.Tensor,
):
    velocity = predicted_state[2:]
    speed = velocity.norm()
    if speed > 1e-6:
        direction = velocity / speed
    else:
        direction = torch.tensor(
            [1.0, 0.0], device=velocity.device, dtype=velocity.dtype
        )
    normal = torch.stack([-direction[1], direction[0]])
    innovation = point_xy - predicted_state[:2]
    along = (innovation * direction).sum()
    cross = (innovation * normal).sum()
    return innovation, along, cross, direction, normal


def select_temporal_mode(
    measurement: VisualMeasurement,
    predicted_state: torch.Tensor,
    dt: float,
    level: str,
    last_visual_xy: torch.Tensor | None,
):
    sigma_along, sigma_cross, _, motion_weight = search_parameters(level)
    candidate_count = int(measurement.centers.shape[1])

    best = None
    details = []
    for mode_id in range(int(config.TOP_MODES)):
        xy = measurement.mode_xy[0, mode_id]
        support = measurement.mode_support[0, mode_id].clamp_min(EPS)
        local_mass = measurement.mode_local_mass[0, mode_id].clamp_min(EPS)
        spatial_std = measurement.mode_spatial_std[0, mode_id]
        peak = measurement.mode_peak_prob[0, mode_id].clamp_min(EPS)

        _, along, cross, _, _ = motion_components(xy, predicted_state)
        motion_cost = 0.5 * (
            (along / sigma_along).square()
            + (cross / sigma_cross).square()
        )

        temporal_cost = torch.tensor(
            0.0, device=xy.device, dtype=xy.dtype
        )
        if last_visual_xy is not None:
            distance = torch.norm(xy - last_visual_xy)
            plausible = (
                float(config.MAX_SPEED_M_PER_FRAME)
                * max(float(dt), 1.0)
                * 1.8
            )
            temporal_cost = torch.relu(
                distance - plausible
            ) / max(float(config.CANDIDATE_SPACING_M), EPS)

        visual_score = (
            torch.log(local_mass)
            + 0.50 * torch.log(support)
            + 0.15 * torch.log(peak * candidate_count + 1.0)
            - float(config.SPATIAL_STD_PENALTY)
            * spatial_std
            / max(float(config.MODE_LOCAL_RADIUS_M), EPS)
        )
        score = (
            float(config.VISUAL_SCORE_WEIGHT) * visual_score
            - motion_weight * motion_cost
            - float(config.TEMPORAL_MODE_WEIGHT) * temporal_cost
        )

        boundary = FrozenVisualLocalizer.mode_at_search_boundary(
            measurement, mode_id
        )
        details.append(
            {
                "mode_id": mode_id,
                "xy": xy,
                "support": support,
                "local_mass": local_mass,
                "spatial_std": spatial_std,
                "peak": peak,
                "along": along,
                "cross": cross,
                "motion_cost": motion_cost,
                "temporal_cost": temporal_cost,
                "visual_score": visual_score,
                "score": score,
                "boundary": boundary,
            }
        )
        if best is None or float(score.item()) > float(best["score"].item()):
            best = details[-1]

    if best is None:
        raise RuntimeError("HardMS did not return a temporal mode")

    # Confidence uses ratios/concentration, not an absolute Top-1 probability.
    local_q = torch.clamp(best["local_mass"] / 0.25, 0.0, 1.0)
    support_q = torch.clamp(best["support"], 0.0, 1.0)
    std_q = torch.exp(
        -best["spatial_std"] / max(float(config.MODE_LOCAL_RADIUS_M), EPS)
    )
    peak_ratio_q = torch.clamp(
        torch.log1p(best["peak"] * candidate_count) / np.log(9.0),
        0.0,
        1.0,
    )
    visual_confidence = (
        0.40 * local_q
        + 0.25 * support_q
        + 0.20 * std_q
        + 0.15 * peak_ratio_q
    )
    geometry = torch.exp(-best["motion_cost"].clamp(max=12.0))
    if level == "recovery":
        confidence = visual_confidence * (0.75 + 0.25 * geometry)
    elif level == "medium":
        confidence = visual_confidence * (0.55 + 0.45 * geometry)
    else:
        confidence = visual_confidence * (0.35 + 0.65 * geometry)
    if best["boundary"]:
        confidence = confidence * 0.75

    best["confidence"] = float(confidence.clamp(0.0, 1.0).item())
    best["visual_confidence"] = float(
        visual_confidence.clamp(0.0, 1.0).item()
    )
    return best, details


def robust_kalman_update(
    previous_state: torch.Tensor,
    predicted_state: torch.Tensor,
    predicted_cov: torch.Tensor,
    selected: dict,
    dt: float,
    level: str,
):
    confidence = float(selected["confidence"])
    visual_xy = selected["xy"]
    _, along, cross, direction, normal = motion_components(
        visual_xy, predicted_state
    )

    recovery = level == "recovery"
    if recovery:
        cap = (
            float(config.RECOVERY_MAX_CORRECTION_M_PER_FRAME)
            * max(float(dt), 1.0)
        )
        bounded_along = along.clamp(-cap, cap)
        bounded_cross = cross.clamp(-cap, cap)
    else:
        along_cap = (
            float(config.MAX_ALONG_CORRECTION_M_PER_FRAME)
            * max(float(dt), 1.0)
        )
        cross_cap = (
            float(config.MAX_CROSS_CORRECTION_M_PER_FRAME)
            * max(float(dt), 1.0)
        )
        bounded_along = along.clamp(-along_cap, along_cap)
        bounded_cross = cross.clamp(-cross_cap, cross_cap)
    bounded_innovation = bounded_along * direction + bounded_cross * normal

    spatial_std = float(selected["spatial_std"].item())
    measurement_std = spatial_std / max(0.35 + confidence, 0.2)
    measurement_std = float(
        np.clip(
            measurement_std,
            config.MEASUREMENT_STD_MIN_M,
            config.MEASUREMENT_STD_MAX_M,
        )
    )

    dtype, device = predicted_state.dtype, predicted_state.device
    observation = torch.zeros((2, 4), device=device, dtype=dtype)
    observation[0, 0] = 1.0
    observation[1, 1] = 1.0
    measurement_cov = torch.eye(2, device=device, dtype=dtype) * (
        measurement_std**2
    )

    innovation_cov = (
        observation @ predicted_cov @ observation.T + measurement_cov
    )
    gain = predicted_cov @ observation.T @ torch.linalg.inv(innovation_cov)
    effective_gain = gain * confidence

    updated_state = predicted_state + effective_gain @ bounded_innovation
    identity = torch.eye(4, device=device, dtype=dtype)
    correction = identity - effective_gain @ observation
    updated_cov = (
        correction @ predicted_cov @ correction.T
        + effective_gain @ measurement_cov @ effective_gain.T
    )

    # Explicitly update velocity from the corrected displacement. This is the
    # crucial turn fix missing in the previous position-only update.
    observed_velocity = (
        updated_state[:2] - previous_state[:2]
    ) / max(float(dt), 1.0)
    beta = min(
        0.75 if recovery else 0.50,
        float(config.VELOCITY_OBSERVATION_GAIN) * confidence,
    )
    velocity_candidate = (
        (1.0 - beta) * updated_state[2:] + beta * observed_velocity
    )
    updated_state[2:] = constrain_turn(
        previous_state[2:],
        velocity_candidate,
        dt,
        recovery=recovery,
    )

    updated_cov = 0.5 * (updated_cov + updated_cov.T)
    innovation_norm = float(torch.norm(visual_xy - predicted_state[:2]).item())
    return {
        "state": updated_state,
        "covariance": updated_cov,
        "measurement_std": measurement_std,
        "innovation_norm": innovation_norm,
        "along": float(along.item()),
        "cross": float(cross.item()),
        "bounded_along": float(bounded_along.item()),
        "bounded_cross": float(bounded_cross.item()),
        "velocity_gain": beta,
    }


def update_recovery(
    lost_streak: int,
    good_streak: int,
    confidence: float,
    boundary: bool,
):
    if confidence < float(config.WEAK_CONFIDENCE) or boundary:
        return min(lost_streak + 1, 100), 0

    good_streak += 1
    if confidence >= float(config.RELIABLE_CONFIDENCE):
        lost_streak = max(0, lost_streak - 1)
    if good_streak >= int(config.GOOD_STREAK_TO_SHRINK):
        lost_streak = max(0, lost_streak - 1)
        good_streak = 0
    return lost_streak, good_streak


def metrics(rows, key):
    pred = np.asarray([r[key] for r in rows], dtype=np.float64)
    gt = np.asarray([r["gt"] for r in rows], dtype=np.float64)
    error = np.linalg.norm(pred - gt, axis=1)

    step = np.diff(pred, axis=0)
    gt_step = np.diff(gt, axis=0)
    velocity_error = np.linalg.norm(step - gt_step, axis=1)
    pred_length = float(np.linalg.norm(step, axis=1).sum()) if len(step) else 0.0
    gt_length = float(np.linalg.norm(gt_step, axis=1).sum()) if len(gt_step) else 0.0

    if len(gt_step):
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = float(
            np.percentile(gt_step_length, 99) + config.JUMP_TOLERANCE_M
        )
        jump = np.linalg.norm(step, axis=1) > jump_threshold
    else:
        jump_threshold = 0.0
        jump = np.zeros(0, dtype=bool)

    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "MaxLE_m": float(error.max()),
        "LSR@5_pct": float((error <= 5).mean() * 100),
        "LSR@10_pct": float((error <= 10).mean() * 100),
        "LSR@15_pct": float((error <= 15).mean() * 100),
        "LSR@20_pct": float((error <= 20).mean() * 100),
        "RPE_m": float(velocity_error.mean()) if len(velocity_error) else 0.0,
        "JumpThreshold_m": jump_threshold,
        "JumpRate_pct": float(jump.mean() * 100) if len(jump) else 0.0,
        "PathLengthRatio": pred_length / max(gt_length, EPS),
    }


def write_route_csv(path: Path, rows):
    columns = [
        "frame_id", "dt_raw", "dt_used",
        "gt_x", "gt_y",
        "raw_visual_x", "raw_visual_y",
        "fused_mode1_x", "fused_mode1_y",
        "selected_visual_x", "selected_visual_y",
        "prediction_x", "prediction_y",
        "final_x", "final_y",
        "velocity_x", "velocity_y",
        "grid_size", "search_level", "search_center_count",
        "candidate_captured", "selected_mode", "boundary",
        "entropy", "mode_support", "mode_local_mass",
        "mode_spatial_std", "mode_peak_prob",
        "visual_confidence", "confidence", "measurement_std",
        "innovation_m", "along_innovation_m", "cross_innovation_m",
        "bounded_along_m", "bounded_cross_m", "velocity_gain",
        "lost_streak",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {
                "frame_id": row["frame_id"],
                "dt_raw": row["dt_raw"],
                "dt_used": row["dt_used"],
                "gt_x": row["gt"][0],
                "gt_y": row["gt"][1],
                "raw_visual_x": row["raw_visual"][0],
                "raw_visual_y": row["raw_visual"][1],
                "fused_mode1_x": row["fused_mode1"][0],
                "fused_mode1_y": row["fused_mode1"][1],
                "selected_visual_x": row["selected_visual"][0],
                "selected_visual_y": row["selected_visual"][1],
                "prediction_x": row["prediction"][0],
                "prediction_y": row["prediction"][1],
                "final_x": row["final"][0],
                "final_y": row["final"][1],
                "velocity_x": row["velocity"][0],
                "velocity_y": row["velocity"][1],
                "grid_size": row["grid_size"],
                "search_level": row["search_level"],
                "search_center_count": row["search_center_count"],
                "candidate_captured": int(row["candidate_captured"]),
                "selected_mode": row["selected_mode"],
                "boundary": int(row["boundary"]),
                "entropy": row["entropy"],
                "mode_support": row["mode_support"],
                "mode_local_mass": row["mode_local_mass"],
                "mode_spatial_std": row["mode_spatial_std"],
                "mode_peak_prob": row["mode_peak_prob"],
                "visual_confidence": row["visual_confidence"],
                "confidence": row["confidence"],
                "measurement_std": row["measurement_std"],
                "innovation_m": row["innovation_m"],
                "along_innovation_m": row["along_innovation_m"],
                "cross_innovation_m": row["cross_innovation_m"],
                "bounded_along_m": row["bounded_along_m"],
                "bounded_cross_m": row["bounded_cross_m"],
                "velocity_gain": row["velocity_gain"],
                "lost_streak": row["lost_streak"],
            }
            writer.writerow(flat)


@torch.no_grad()
def run_route(root, name, visual, device):
    dataset, indices = tracking_frames(
        root, visual.origin_lat, visual.origin_lon
    )
    if len(indices) <= INIT_HISTORY:
        raise ValueError(f"{name} has only {len(indices)} sampled frames")

    # Read coordinates/frame IDs from metadata without loading all images.
    # UAV tensors are encoded in bounded batches and CLIP features are kept on
    # CPU, preventing TRACK_FRAME_STRIDE=1 from exhausting GPU memory.
    sampled_meta = [dataset.samples[i] for i in indices]
    frame_ids = np.asarray(
        [int(sample["frame_id"]) for sample in sampled_meta],
        dtype=np.int64,
    )
    eval_gt = np.asarray(
        [[sample["x_meter"], sample["y_meter"]] for sample in sampled_meta],
        dtype=np.float32,
    )

    clip_batches = []
    for start in range(0, len(indices), 64):
        chunk = indices[start:start + 64]
        uav_batch = torch.stack([dataset[i]["uav"] for i in chunk], dim=0)
        clip_batches.append(visual.encode_uav_clip(uav_batch).cpu())
    clips = torch.cat(clip_batches, dim=0)

    state = torch.zeros(4, device=device)
    state[:2] = torch.from_numpy(eval_gt[INIT_HISTORY - 1]).to(device)
    initial_velocity = robust_initial_velocity(
        eval_gt[:INIT_HISTORY], frame_ids[:INIT_HISTORY]
    )
    state[2:] = torch.from_numpy(initial_velocity).to(device)
    initial_speed = float(state[2:].norm().item())

    covariance = torch.diag(
        torch.tensor(
            [
                config.POSITION_STD_M**2,
                config.POSITION_STD_M**2,
                config.VELOCITY_STD_M_PER_FRAME**2,
                config.VELOCITY_STD_M_PER_FRAME**2,
            ],
            device=device,
            dtype=state.dtype,
        )
    )

    history_positions = deque(maxlen=int(config.LINE_FIT_HISTORY))
    history_times = deque(maxlen=int(config.LINE_FIT_HISTORY))
    history_confidences = deque(maxlen=int(config.LINE_FIT_HISTORY))
    for p, frame_id in zip(
        eval_gt[:INIT_HISTORY], frame_ids[:INIT_HISTORY]
    ):
        history_positions.append(torch.from_numpy(p).to(device))
        history_times.append(float(frame_id))
        history_confidences.append(1.0)

    last_reliable_xy = state[:2].clone()
    last_visual_xy = None
    lost_streak = 0
    good_streak = 0
    previous_frame_id = int(frame_ids[INIT_HISTORY - 1])
    rows = []

    for t in range(INIT_HISTORY, len(indices)):
        frame_id = int(frame_ids[t])
        dt_raw = max(1, frame_id - previous_frame_id)
        dt = float(min(dt_raw, int(config.MAX_FRAME_ID_GAP)))

        predicted_state, predicted_cov = predict(state, covariance, dt)
        grid_size, level = choose_search(lost_streak, predicted_cov)
        search_centers = build_search_centers(
            predicted_state[:2], last_reliable_xy, level
        )
        sigma_along, sigma_cross, prior_weight, _ = search_parameters(level)

        measurement = visual.measure(
            clips[t:t + 1].to(device, non_blocking=True),
            predicted_state[:2].unsqueeze(0),
            predicted_state[2:].unsqueeze(0),
            search_centers=search_centers,
            grid_size=grid_size,
            sigma_along=sigma_along,
            sigma_cross=sigma_cross,
            prior_weight=prior_weight,
        )
        selected, _ = select_temporal_mode(
            measurement,
            predicted_state,
            dt,
            level,
            last_visual_xy,
        )
        update = robust_kalman_update(
            state,
            predicted_state,
            predicted_cov,
            selected,
            dt,
            level,
        )
        state = update["state"]
        covariance = update["covariance"]

        confidence = float(selected["confidence"])
        boundary = bool(selected["boundary"])
        lost_streak, good_streak = update_recovery(
            lost_streak,
            good_streak,
            confidence,
            boundary,
        )

        history_positions.append(state[:2].clone())
        history_times.append(float(frame_id))
        history_confidences.append(max(confidence, 0.05))
        if confidence >= float(config.WEAK_CONFIDENCE):
            fitted_velocity = robust_history_velocity(
                history_positions,
                history_times,
                history_confidences,
                state[2:],
            )
            blend = float(config.VELOCITY_HISTORY_BLEND) * confidence
            candidate_velocity = (
                (1.0 - blend) * state[2:] + blend * fitted_velocity
            )
            state[2:] = constrain_turn(
                state[2:],
                candidate_velocity,
                dt=1.0,
                recovery=(level == "recovery"),
            )

        if confidence >= float(config.RELIABLE_CONFIDENCE) and not boundary:
            last_reliable_xy = state[:2].clone()
        if confidence >= float(config.WEAK_CONFIDENCE):
            if last_visual_xy is None:
                last_visual_xy = selected["xy"].clone()
            else:
                maximum = (
                    float(config.MAX_SPEED_M_PER_FRAME)
                    * max(float(dt), 1.0)
                    * 2.0
                )
                if float(torch.norm(selected["xy"] - last_visual_xy).item()) <= maximum:
                    last_visual_xy = selected["xy"].clone()

        # Evaluation only: this must never alter state/search/recovery.
        gt_tensor = torch.from_numpy(eval_gt[t:t + 1]).to(device)
        captured = bool(
            visual.candidate_contains_gt(measurement, gt_tensor).item()
        )

        rows.append(
            {
                "frame_id": frame_id,
                "dt_raw": dt_raw,
                "dt_used": dt,
                "gt": eval_gt[t].tolist(),
                "raw_visual": measurement.raw_visual_xy[0].cpu().tolist(),
                "fused_mode1": measurement.fused_xy[0].cpu().tolist(),
                "selected_visual": selected["xy"].cpu().tolist(),
                "prediction": predicted_state[:2].cpu().tolist(),
                "final": state[:2].cpu().tolist(),
                "velocity": state[2:].cpu().tolist(),
                "grid_size": int(grid_size),
                "search_level": level,
                "search_center_count": int(search_centers.shape[0]),
                "candidate_captured": captured,
                "selected_mode": int(selected["mode_id"]),
                "boundary": boundary,
                "entropy": float(measurement.entropy.item()),
                "mode_support": float(selected["support"].item()),
                "mode_local_mass": float(selected["local_mass"].item()),
                "mode_spatial_std": float(selected["spatial_std"].item()),
                "mode_peak_prob": float(selected["peak"].item()),
                "visual_confidence": float(selected["visual_confidence"]),
                "confidence": confidence,
                "measurement_std": update["measurement_std"],
                "innovation_m": update["innovation_norm"],
                "along_innovation_m": update["along"],
                "cross_innovation_m": update["cross"],
                "bounded_along_m": update["bounded_along"],
                "bounded_cross_m": update["bounded_cross"],
                "velocity_gain": update["velocity_gain"],
                "lost_streak": int(lost_streak),
            }
        )
        previous_frame_id = frame_id

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_route_csv(
        config.OUTPUT_DIR / f"{name}_straight_line_v2_frames.csv",
        rows,
    )

    return {
        "route": name,
        "GTLeakageFreeSampling": True,
        "track_frame_stride": int(config.TRACK_FRAME_STRIDE),
        "sampled_frames": len(indices),
        "initial_speed_m_per_frame_id": initial_speed,
        "RawVisualHardMS": metrics(rows, "raw_visual"),
        "FusedMode1": metrics(rows, "fused_mode1"),
        "SelectedTemporalMode": metrics(rows, "selected_visual"),
        "MotionPrediction": metrics(rows, "prediction"),
        "StraightLineTemporalHardMSV2": metrics(rows, "final"),
        "CandidateCaptureRate_pct": float(
            np.mean([r["candidate_captured"] for r in rows]) * 100
        ),
        "GridUsage": {
            str(n): int(sum(r["grid_size"] == n for r in rows))
            for n in [
                config.GRID_SIZE,
                config.GRID_SIZE_MEDIUM,
                config.GRID_SIZE_RECOVERY,
            ]
        },
        "SearchLevelUsage": {
            level: int(sum(r["search_level"] == level for r in rows))
            for level in ["normal", "medium", "recovery"]
        },
        "MeanConfidence": float(np.mean([r["confidence"] for r in rows])),
        "ReliableRate_pct": float(
            np.mean(
                [
                    r["confidence"] >= config.RELIABLE_CONFIDENCE
                    and not r["boundary"]
                    for r in rows
                ]
            )
            * 100
        ),
        "BoundaryRate_pct": float(
            np.mean([r["boundary"] for r in rows]) * 100
        ),
    }


def main():
    torch.manual_seed(int(config.SEED))
    np.random.seed(int(config.SEED))
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    visual = FrozenVisualLocalizer(device)

    results = [
        run_route(root, name, visual, device)
        for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES)
    ]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.OUTPUT_DIR / "straight_line_tracker_v2_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    for result in results:
        metric = result["StraightLineTemporalHardMSV2"]
        print(
            f"{result['route']}: "
            f"MLE={metric['MLE_m']:.2f}m | "
            f"P90={metric['P90_m']:.2f}m | "
            f"jump={metric['JumpRate_pct']:.2f}% | "
            f"CCR={result['CandidateCaptureRate_pct']:.2f}% | "
            f"reliable={result['ReliableRate_pct']:.2f}%"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()