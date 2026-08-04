"""Straight-line prior-guided HardMS tracker.

This is a deterministic replacement for the old fixed-initial-velocity tracker.
It is designed specifically for the teacher's "straight-line prediction"
requirement and for suppressing left/right/front/back jumps:

1. Fit a short straight line to recent posterior positions.
2. Extrapolate one node forward.
3. Search with an anisotropic prior (wide along track, narrow across track).
4. Use fused HardMS, not raw HardMS.
5. Apply bounded continuous correction rather than a binary gate.
6. Expand candidates when confidence is poor or the mode reaches the edge.
"""
from __future__ import annotations

import csv
import json
from collections import deque

import numpy as np
import torch

import config
from data import RouteDataset
from visual_localizer import FrozenVisualLocalizer


INIT_HISTORY = int(config.HISTORY)


def motion_nodes(root, origin_lat, origin_lon):
    """Collapse consecutive images with the same logged GPS position."""
    dataset = RouteDataset(
        root,
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    groups, current = [], [0]
    for i in range(1, len(dataset.samples)):
        a, b = dataset.samples[i - 1], dataset.samples[i]
        distance = np.hypot(
            b["x_meter"] - a["x_meter"],
            b["y_meter"] - a["y_meter"],
        )
        if distance <= config.TEMPORAL_POSITION_MERGE_M:
            current.append(i)
        else:
            groups.append(current[len(current) // 2])
            current = [i]
    groups.append(current[len(current) // 2])
    return dataset, groups


def robust_initial_velocity(gt_init: np.ndarray) -> np.ndarray:
    """Estimate initial straight-line motion without averaging take-off pauses."""
    steps = np.diff(gt_init, axis=0)
    speeds = np.linalg.norm(steps, axis=1)
    valid = speeds > 1e-3
    if not np.any(valid):
        return np.zeros(2, dtype=np.float32)

    steps = steps[valid]
    speeds = speeds[valid]
    # Use the most recent three non-zero steps. Median speed is robust to one
    # irregular GPS interval; summed unit directions suppress lateral noise.
    steps = steps[-3:]
    speeds = speeds[-3:]
    unit = steps / np.maximum(speeds[:, None], 1e-6)
    direction = unit.sum(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = unit[-1]
    else:
        direction = direction / norm
    speed = float(np.median(speeds))
    speed = min(speed, float(config.ABSOLUTE_MAX_SPEED_M_PER_NODE))
    return (direction * speed).astype(np.float32)


def line_fit_velocity(position_history: deque, previous_velocity: torch.Tensor):
    """Fit x(t), y(t) with least squares, then constrain speed and turning."""
    positions = torch.stack(list(position_history), dim=0)
    if positions.shape[0] < 2:
        return previous_velocity

    t = torch.arange(
        positions.shape[0],
        device=positions.device,
        dtype=positions.dtype,
    )
    t = t - t.mean()
    slope = (t[:, None] * (positions - positions.mean(dim=0))).sum(dim=0)
    slope = slope / t.square().sum().clamp_min(1e-6)

    prev_speed = previous_velocity.norm()
    candidate_speed = slope.norm()
    if candidate_speed < 1e-6:
        return previous_velocity

    # Smooth the estimated velocity before applying geometric constraints.
    candidate = (
        (1.0 - float(config.VELOCITY_EMA)) * previous_velocity
        + float(config.VELOCITY_EMA) * slope
    )
    candidate_speed = candidate.norm().clamp_min(1e-6)

    if prev_speed > 1e-6:
        min_speed = prev_speed * float(config.MIN_SPEED_RATIO)
        max_speed = prev_speed * float(config.MAX_SPEED_RATIO)
        target_speed = candidate_speed.clamp(min=min_speed, max=max_speed)

        prev_dir = previous_velocity / prev_speed
        cand_dir = candidate / candidate_speed
        dot = torch.clamp((prev_dir * cand_dir).sum(), -1.0, 1.0)
        angle = torch.acos(dot)
        max_angle = torch.deg2rad(
            torch.tensor(
                float(config.MAX_TURN_DEG_PER_NODE),
                device=positions.device,
                dtype=positions.dtype,
            )
        )
        if angle > max_angle:
            # 2-D signed rotation from previous direction toward candidate.
            cross = prev_dir[0] * cand_dir[1] - prev_dir[1] * cand_dir[0]
            signed = torch.sign(cross) * max_angle
            c, s = torch.cos(signed), torch.sin(signed)
            constrained_dir = torch.stack(
                [c * prev_dir[0] - s * prev_dir[1],
                 s * prev_dir[0] + c * prev_dir[1]]
            )
        else:
            constrained_dir = cand_dir
    else:
        target_speed = candidate_speed
        constrained_dir = candidate / candidate_speed

    target_speed = target_speed.clamp_max(
        float(config.ABSOLUTE_MAX_SPEED_M_PER_NODE)
    )
    return constrained_dir * target_speed


def predict(state: torch.Tensor, covariance: torch.Tensor):
    dtype, device = state.dtype, state.device
    F = torch.eye(4, device=device, dtype=dtype)
    F[0, 2] = 1.0
    F[1, 3] = 1.0
    Q = torch.diag(
        torch.tensor(
            [
                config.PROCESS_POSITION_STD_M**2,
                config.PROCESS_POSITION_STD_M**2,
                config.PROCESS_VELOCITY_STD_M**2,
                config.PROCESS_VELOCITY_STD_M**2,
            ],
            device=device,
            dtype=dtype,
        )
    )
    predicted_state = F @ state
    predicted_cov = F @ covariance @ F.T + Q
    predicted_cov[0, 0] = predicted_cov[0, 0].clamp_max(
        config.MAX_POSITION_STD_M**2
    )
    predicted_cov[1, 1] = predicted_cov[1, 1].clamp_max(
        config.MAX_POSITION_STD_M**2
    )
    return predicted_state, predicted_cov


def choose_grid_size(lost_streak: int, covariance: torch.Tensor):
    position_std = float(
        torch.sqrt(torch.diagonal(covariance[:2, :2]).max()).item()
    )
    if lost_streak >= config.LOST_STREAK_RECOVERY or position_std >= 18.0:
        return config.GRID_SIZE_RECOVERY
    if lost_streak >= config.LOST_STREAK_LARGE or position_std >= 12.0:
        return config.GRID_SIZE_LARGE
    if lost_streak >= config.LOST_STREAK_MEDIUM or position_std >= 8.0:
        return config.GRID_SIZE_MEDIUM
    return config.GRID_SIZE


def measurement_quality(measurement, edge: bool):
    entropy_quality = (1.0 - measurement.entropy).clamp(0.0, 1.0)
    peak_quality = (
        measurement.peak_prob / max(float(config.LOW_PEAK_PROB), 1e-6)
    ).clamp(0.0, 1.0)
    margin_quality = (
        measurement.margin_prob / max(float(config.LOW_MARGIN_PROB), 1e-6)
    ).clamp(0.0, 1.0)
    quality = entropy_quality * torch.sqrt(peak_quality * margin_quality)
    if edge:
        quality = quality * 0.35
    return float(quality.item())


def bounded_continuous_update(
    predicted_state: torch.Tensor,
    predicted_cov: torch.Tensor,
    visual_xy: torch.Tensor,
    quality: float,
):
    """Apply a bounded along/cross-track correction with continuous weight."""
    velocity = predicted_state[2:]
    speed = velocity.norm()
    if speed > 1e-6:
        direction = velocity / speed
    else:
        direction = torch.tensor(
            [1.0, 0.0], device=velocity.device, dtype=velocity.dtype
        )
    normal = torch.stack([-direction[1], direction[0]])

    innovation = visual_xy - predicted_state[:2]
    innovation_norm = float(innovation.norm().item())
    along = (innovation * direction).sum()
    cross = (innovation * normal).sum()

    # Never allow one retrieval peak to rewrite the track in a single frame.
    bounded_along = along.clamp(
        -float(config.MAX_ALONG_CORRECTION_M),
        float(config.MAX_ALONG_CORRECTION_M),
    )
    bounded_cross = cross.clamp(
        -float(config.MAX_CROSS_CORRECTION_M),
        float(config.MAX_CROSS_CORRECTION_M),
    )
    bounded = bounded_along * direction + bounded_cross * normal

    geometry_weight = float(
        torch.exp(
            -0.5
            * (
                (along / float(config.RECOVERY_SIGMA_ALONG_M)) ** 2
                + (cross / float(config.RECOVERY_SIGMA_CROSS_M)) ** 2
            )
        ).item()
    )
    alpha = min(
        float(config.MAX_UPDATE_ALPHA),
        max(0.0, quality * geometry_weight),
    )
    if innovation_norm > float(config.EXTREME_INNOVATION_M):
        alpha = 0.0

    updated_state = predicted_state.clone()
    updated_state[:2] = predicted_state[:2] + alpha * bounded

    updated_cov = predicted_cov.clone()
    # A confident continuous correction reduces position uncertainty; a weak
    # correction leaves it large so the next candidate window can expand.
    reduction = max(0.25, 1.0 - 0.75 * alpha)
    updated_cov[:2, :2] = updated_cov[:2, :2] * reduction
    updated_cov = 0.5 * (updated_cov + updated_cov.T)

    return updated_state, updated_cov, alpha, innovation_norm, float(along), float(cross)


def update_recovery_state(lost_streak, good_streak, measurement, quality, edge):
    poor = (
        edge
        or float(measurement.entropy.item()) >= config.HIGH_ENTROPY
        or float(measurement.peak_prob.item()) <= config.LOW_PEAK_PROB
        or float(measurement.margin_prob.item()) <= config.LOW_MARGIN_PROB
        or quality < config.MIN_CONFIDENT_ALPHA
    )
    if poor:
        return min(lost_streak + 1, 100), 0
    good_streak += 1
    if good_streak >= config.GOOD_STREAK_TO_SHRINK:
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
    }


@torch.no_grad()
def run_route(root, name, visual, device):
    dataset, indices = motion_nodes(root, visual.origin_lat, visual.origin_lon)
    frames = [dataset[i] for i in indices]
    if len(frames) <= INIT_HISTORY:
        raise ValueError(f"{name} has only {len(frames)} motion nodes")

    uav = torch.stack([f["uav"] for f in frames]).to(device)
    gt = torch.stack([f["xy"] for f in frames]).cpu().numpy()
    clips = torch.cat(
        [visual.encode_uav_clip(uav[i:i + 64]) for i in range(0, len(uav), 64)],
        dim=0,
    )

    state = torch.zeros(4, device=device)
    state[:2] = torch.from_numpy(gt[INIT_HISTORY - 1]).to(device)
    state[2:] = torch.from_numpy(
        robust_initial_velocity(gt[:INIT_HISTORY])
    ).to(device)
    initial_speed = float(state[2:].norm().item())

    covariance = torch.diag(
        torch.tensor(
            [
                config.POSITION_STD_M**2,
                config.POSITION_STD_M**2,
                config.VELOCITY_STD_M**2,
                config.VELOCITY_STD_M**2,
            ],
            device=device,
        )
    )

    position_history = deque(maxlen=int(config.LINE_FIT_HISTORY))
    for p in gt[:INIT_HISTORY]:
        position_history.append(torch.from_numpy(p).to(device))

    lost_streak = 0
    good_streak = 0
    rows = []

    for t in range(INIT_HISTORY, len(frames)):
        # Re-estimate a smooth straight line from recent posterior positions.
        state[2:] = line_fit_velocity(position_history, state[2:])
        predicted_state, predicted_cov = predict(state, covariance)

        grid_size = choose_grid_size(lost_streak, predicted_cov)
        recovery = grid_size >= config.GRID_SIZE_LARGE
        sigma_along = (
            config.RECOVERY_SIGMA_ALONG_M
            if recovery else config.MOTION_SIGMA_ALONG_M
        )
        sigma_cross = (
            config.RECOVERY_SIGMA_CROSS_M
            if recovery else config.MOTION_SIGMA_CROSS_M
        )

        measurement = visual.measure(
            clips[t:t + 1],
            predicted_state[:2].unsqueeze(0),
            predicted_state[2:].unsqueeze(0),
            grid_size=grid_size,
            sigma_along=sigma_along,
            sigma_cross=sigma_cross,
        )
        edge = bool(visual.at_grid_edge(measurement).item())
        quality = measurement_quality(measurement, edge)

        # IMPORTANT: use fused_xy. The old code calculated fused_xy but then
        # accidentally updated with raw visual_xy.
        state, covariance, alpha, innovation_norm, along, cross = (
            bounded_continuous_update(
                predicted_state,
                predicted_cov,
                measurement.fused_xy.squeeze(0),
                quality,
            )
        )

        lost_streak, good_streak = update_recovery_state(
            lost_streak,
            good_streak,
            measurement,
            quality,
            edge,
        )

        position_history.append(state[:2].clone())
        gt_tensor = torch.from_numpy(gt[t:t + 1]).to(device)
        captured = bool(
            visual.candidate_contains_gt(measurement, gt_tensor).item()
        )

        rows.append(
            {
                "frame_id": frames[t]["frame_id"],
                "gt": gt[t].tolist(),
                "raw_visual": measurement.raw_visual_xy.squeeze(0).cpu().tolist(),
                "fused_visual": measurement.fused_xy.squeeze(0).cpu().tolist(),
                "prediction": predicted_state[:2].cpu().tolist(),
                "straight_line": state[:2].cpu().tolist(),
                "velocity": state[2:].cpu().tolist(),
                "grid_size": int(grid_size),
                "candidate_captured": captured,
                "edge": edge,
                "entropy": float(measurement.entropy.item()),
                "peak_prob": float(measurement.peak_prob.item()),
                "margin_prob": float(measurement.margin_prob.item()),
                "quality": quality,
                "alpha": alpha,
                "innovation_m": innovation_norm,
                "along_innovation_m": along,
                "cross_innovation_m": cross,
                "lost_streak": int(lost_streak),
            }
        )

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_straight_line_frames.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_id",
                "gt_x", "gt_y",
                "raw_visual_x", "raw_visual_y",
                "fused_visual_x", "fused_visual_y",
                "prediction_x", "prediction_y",
                "final_x", "final_y",
                "velocity_x", "velocity_y",
                "grid_size", "candidate_captured", "edge",
                "entropy", "peak_prob", "margin_prob", "quality",
                "update_alpha", "innovation_m",
                "along_innovation_m", "cross_innovation_m",
                "lost_streak",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["frame_id"],
                    *r["gt"],
                    *r["raw_visual"],
                    *r["fused_visual"],
                    *r["prediction"],
                    *r["straight_line"],
                    *r["velocity"],
                    r["grid_size"],
                    int(r["candidate_captured"]),
                    int(r["edge"]),
                    r["entropy"],
                    r["peak_prob"],
                    r["margin_prob"],
                    r["quality"],
                    r["alpha"],
                    r["innovation_m"],
                    r["along_innovation_m"],
                    r["cross_innovation_m"],
                    r["lost_streak"],
                ]
            )

    return {
        "route": name,
        "initial_speed_m_per_node": initial_speed,
        "RawVisualHardMS": metrics(rows, "raw_visual"),
        "MotionPrediction": metrics(rows, "prediction"),
        "StraightLineTemporalHardMS": metrics(rows, "straight_line"),
        "CandidateCaptureRate_pct": float(
            np.mean([r["candidate_captured"] for r in rows]) * 100
        ),
        "GridUsage": {
            str(n): int(sum(r["grid_size"] == n for r in rows))
            for n in [
                config.GRID_SIZE,
                config.GRID_SIZE_MEDIUM,
                config.GRID_SIZE_LARGE,
                config.GRID_SIZE_RECOVERY,
            ]
        },
        "MeanUpdateAlpha": float(np.mean([r["alpha"] for r in rows])),
        "EdgeRate_pct": float(np.mean([r["edge"] for r in rows]) * 100),
    }


def main():
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    visual = FrozenVisualLocalizer(device)
    results = [
        run_route(root, name, visual, device)
        for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES)
    ]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT_DIR / "straight_line_tracker_summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    for result in results:
        metric = result["StraightLineTemporalHardMS"]
        print(
            f"{result['route']}: "
            f"MLE={metric['MLE_m']:.2f}m | "
            f"P90={metric['P90_m']:.2f}m | "
            f"jump={metric['JumpRate_pct']:.2f}% | "
            f"CCR={result['CandidateCaptureRate_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()