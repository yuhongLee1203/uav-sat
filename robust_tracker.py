"""Oracle-candidate causal straight-line HardMS smoothing.

The current goal is deliberately narrow: first prove that temporal smoothing can
remove left/right/front/back jitter when the correct local satellite region is
available.  GT (or GT plus bounded deterministic jitter) is therefore used only
as the 6x6 candidate-window centre.  After the first five-frame
initialisation, GT is never passed to the temporal prediction or correction.

This is an Oracle Candidate Temporal Smoothing diagnostic, not a closed-loop
tracking claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

import config
from data import RouteDataset
from visual_localizer import FrozenVisualLocalizer, VisualMeasurement

EPS = 1e-8
INIT_HISTORY = int(config.HISTORY)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-center",
        choices=["gt", "gt_jitter"],
        default=str(config.CANDIDATE_CENTER_MODE),
        help="Use exact GT or GT plus deterministic bounded jitter as the local-window centre.",
    )
    parser.add_argument(
        "--jitter-m",
        type=float,
        default=float(config.GT_JITTER_MAX_M),
        help="Maximum radial jitter in metres for --candidate-center gt_jitter.",
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=None,
        help="Optional subset, for example: --routes route_A route_C",
    )
    return parser.parse_args()


def tracking_frames(root, origin_lat, origin_lon):
    """Sample frames by image order only; GT motion never selects frames."""
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


def robust_initial_velocity(gt_init: np.ndarray, frame_ids: np.ndarray):
    """Median pairwise slope, in metres per original frame ID."""
    slopes = []
    for i in range(len(gt_init)):
        for j in range(i + 1, len(gt_init)):
            dt = float(frame_ids[j] - frame_ids[i])
            if dt > 0:
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


def deterministic_jitter(frame_ids: np.ndarray, route_index: int, radius_m: float):
    """Independent, reproducible uniform-in-disc candidate-centre perturbations."""
    if radius_m <= 0:
        return np.zeros((len(frame_ids), 2), dtype=np.float32)
    rng = np.random.default_rng(int(config.SEED) + 1009 * int(route_index))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=len(frame_ids))
    radius = float(radius_m) * np.sqrt(rng.uniform(0.0, 1.0, size=len(frame_ids)))
    jitter = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return jitter.astype(np.float32)


def limit_speed(velocity: torch.Tensor):
    speed = velocity.norm()
    maximum = float(config.MAX_SPEED_M_PER_FRAME)
    if float(speed.item()) > maximum:
        velocity = velocity * (maximum / speed.clamp_min(EPS))
    return velocity


def constrain_turn(previous: torch.Tensor, candidate: torch.Tensor, dt: float):
    """Limit abrupt heading changes while allowing gradual real turns."""
    candidate = limit_speed(candidate)
    prev_speed = previous.norm()
    cand_speed = candidate.norm()
    if float(prev_speed.item()) < 1e-5 or float(cand_speed.item()) < 1e-5:
        return candidate

    prev_dir = previous / prev_speed
    cand_dir = candidate / cand_speed
    angle = torch.acos(torch.clamp((prev_dir * cand_dir).sum(), -1.0, 1.0))
    max_angle = torch.deg2rad(
        torch.tensor(
            min(179.0, float(config.MAX_TURN_DEG_PER_FRAME) * max(float(dt), 1.0)),
            device=previous.device,
            dtype=previous.dtype,
        )
    )
    if bool(angle <= max_angle):
        return candidate

    cross = prev_dir[0] * cand_dir[1] - prev_dir[1] * cand_dir[0]
    sign = torch.sign(cross)
    if float(sign.item()) == 0.0:
        sign = torch.tensor(1.0, device=previous.device, dtype=previous.dtype)
    signed = sign * max_angle
    c, s = torch.cos(signed), torch.sin(signed)
    direction = torch.stack(
        [
            c * prev_dir[0] - s * prev_dir[1],
            s * prev_dir[0] + c * prev_dir[1],
        ]
    )
    return direction * cand_speed


def predict(state: torch.Tensor, covariance: torch.Tensor, dt: float):
    dtype, device = state.dtype, state.device
    transition = torch.eye(4, device=device, dtype=dtype)
    transition[0, 2] = float(dt)
    transition[1, 3] = float(dt)

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
    return predicted_state, 0.5 * (predicted_cov + predicted_cov.T)


def line_fit_velocity(
    positions: deque,
    times: deque,
    fallback: torch.Tensor,
):
    """Robust causal local-linear velocity fitted to filtered positions."""
    if len(positions) < 3:
        return fallback

    pos = torch.stack(list(positions), dim=0)
    t = torch.tensor(list(times), device=pos.device, dtype=pos.dtype)
    t = t - t.mean()
    weights = torch.ones(len(pos), device=pos.device, dtype=pos.dtype)
    slope = fallback

    for _ in range(4):
        w_sum = weights.sum().clamp_min(EPS)
        t_mean = (weights * t).sum() / w_sum
        p_mean = (weights[:, None] * pos).sum(dim=0) / w_sum
        tc = t - t_mean
        denominator = (weights * tc.square()).sum().clamp_min(EPS)
        slope = (
            weights[:, None] * tc[:, None] * (pos - p_mean)
        ).sum(dim=0) / denominator
        intercept = p_mean - slope * t_mean
        residual = torch.norm(pos - (intercept + t[:, None] * slope), dim=1)
        delta = float(config.LINE_HUBER_M)
        weights = torch.where(
            residual <= delta,
            torch.ones_like(residual),
            delta / residual.clamp_min(EPS),
        )

    if not torch.isfinite(slope).all():
        return fallback
    return limit_speed(slope)


def motion_axes(predicted_state: torch.Tensor, candidate_xy: torch.Tensor | None = None):
    velocity = predicted_state[2:]
    speed = velocity.norm()
    if float(speed.item()) >= float(config.MIN_NONZERO_SPEED_M_PER_FRAME):
        direction = velocity / speed
    elif candidate_xy is not None:
        residual = candidate_xy - predicted_state[:2]
        norm = residual.norm()
        if float(norm.item()) > 1e-6:
            direction = residual / norm
        else:
            direction = torch.tensor(
                [1.0, 0.0], device=velocity.device, dtype=velocity.dtype
            )
    else:
        direction = torch.tensor(
            [1.0, 0.0], device=velocity.device, dtype=velocity.dtype
        )
    normal = torch.stack([-direction[1], direction[0]])
    return direction, normal


def visual_confidence(measurement: VisualMeasurement, mode_id: int):
    """Relative concentration within a known-correct local candidate region."""
    probability = measurement.raw_prob[0]
    k = min(2, int(probability.numel()))
    top = torch.topk(probability, k=k).values
    p1 = top[0]
    p2 = top[1] if k > 1 else torch.zeros_like(p1)
    margin = ((p1 - p2) / p1.clamp_min(EPS)).clamp(0.0, 1.0)

    n = max(int(probability.numel()), 2)
    entropy = -(
        probability * probability.clamp_min(EPS).log()
    ).sum() / np.log(float(n))
    concentration = (1.0 - entropy).clamp(0.0, 1.0)

    local_mass = measurement.mode_local_mass[0, mode_id]
    mass_quality = (local_mass / 0.35).clamp(0.0, 1.0)
    spatial_std = measurement.mode_spatial_std[0, mode_id]
    spatial_quality = torch.exp(
        -spatial_std / max(float(config.MODE_LOCAL_RADIUS_M), EPS)
    )

    confidence = (
        0.32 * margin
        + 0.30 * concentration
        + 0.23 * mass_quality
        + 0.15 * spatial_quality
    ).clamp(0.0, 1.0)
    return {
        "confidence": float(confidence.item()),
        "margin": float(margin.item()),
        "entropy": float(entropy.item()),
        "local_mass": float(local_mass.item()),
        "spatial_std": float(spatial_std.item()),
        "peak_prob": float(measurement.mode_peak_prob[0, mode_id].item()),
    }


def select_temporal_mode(
    measurement: VisualMeasurement,
    predicted_state: torch.Tensor,
):
    """Softly rank raw HardMS modes using visual evidence and straight motion."""
    candidate_count = int(measurement.centers.shape[1])
    best = None

    for mode_id in range(min(int(config.TOP_MODES), measurement.mode_xy.shape[1])):
        xy = measurement.mode_xy[0, mode_id]
        direction, normal = motion_axes(predicted_state, xy)
        residual = xy - predicted_state[:2]
        along = (residual * direction).sum()
        cross = (residual * normal).sum()
        distance_units = (
            residual.norm() / max(float(config.CANDIDATE_SPACING_M), EPS)
        ).clamp(max=float(config.MODE_MAX_DISTANCE_UNITS))
        cross_units = (
            cross.abs() / max(float(config.CANDIDATE_SPACING_M), EPS)
        ).clamp(max=float(config.MODE_MAX_DISTANCE_UNITS))

        local_mass = measurement.mode_local_mass[0, mode_id].clamp_min(EPS)
        peak = measurement.mode_peak_prob[0, mode_id].clamp_min(EPS)
        spatial_std = measurement.mode_spatial_std[0, mode_id]
        visual_score = (
            torch.log(local_mass)
            + 0.30 * torch.log1p(peak * candidate_count)
            - 0.18 * spatial_std / max(float(config.MODE_LOCAL_RADIUS_M), EPS)
        )
        score = (
            float(config.MODE_VISUAL_WEIGHT) * visual_score
            - float(config.MODE_MOTION_WEIGHT) * distance_units
            - float(config.MODE_CROSS_WEIGHT) * cross_units
        )
        item = {
            "mode_id": int(mode_id),
            "xy": xy,
            "score": score,
            "along": float(along.item()),
            "cross": float(cross.item()),
        }
        if best is None or float(score.item()) > float(best["score"].item()):
            best = item

    if best is None:
        raise RuntimeError("No HardMS mode was returned")

    confidence = visual_confidence(measurement, best["mode_id"])
    best.update(confidence)
    best["boundary"] = FrozenVisualLocalizer.mode_at_search_boundary(
        measurement, best["mode_id"]
    )
    if best["boundary"]:
        best["confidence"] *= 0.80
    return best


def straight_line_update(
    previous_state: torch.Tensor,
    predicted_state: torch.Tensor,
    predicted_cov: torch.Tensor,
    selected: dict,
    dt: float,
):
    """Continuous robust alpha-beta correction; no binary Mahalanobis gate."""
    visual_xy = selected["xy"]
    direction, normal = motion_axes(predicted_state, visual_xy)
    innovation = visual_xy - predicted_state[:2]
    along = (innovation * direction).sum()
    cross = (innovation * normal).sum()
    innovation_norm = innovation.norm()

    huber = min(
        1.0,
        float(config.INNOVATION_HUBER_M)
        / max(float(innovation_norm.item()), EPS),
    )
    confidence = max(float(config.CONFIDENCE_FLOOR), float(selected["confidence"]))
    effective = confidence * huber

    along_cap = float(config.MAX_ALONG_CORRECTION_M_PER_FRAME) * max(float(dt), 1.0)
    cross_cap = float(config.MAX_CROSS_CORRECTION_M_PER_FRAME) * max(float(dt), 1.0)
    bounded_along = along.clamp(-along_cap, along_cap)
    bounded_cross = cross.clamp(-cross_cap, cross_cap)

    alpha_along = (
        float(config.ALPHA_ALONG_MIN)
        + (float(config.ALPHA_ALONG_MAX) - float(config.ALPHA_ALONG_MIN))
        * confidence
    ) * huber
    alpha_cross = (
        float(config.ALPHA_CROSS_MIN)
        + (float(config.ALPHA_CROSS_MAX) - float(config.ALPHA_CROSS_MIN))
        * confidence
    ) * huber

    correction = (
        alpha_along * bounded_along * direction
        + alpha_cross * bounded_cross * normal
    )
    updated_state = predicted_state.clone()
    updated_state[:2] = predicted_state[:2] + correction

    observed_velocity = (
        updated_state[:2] - previous_state[:2]
    ) / max(float(dt), 1.0)
    beta = (
        float(config.BETA_MIN)
        + (float(config.BETA_MAX) - float(config.BETA_MIN)) * effective
    )
    velocity_candidate = (
        (1.0 - beta) * predicted_state[2:] + beta * observed_velocity
    )
    updated_state[2:] = constrain_turn(
        previous_state[2:], velocity_candidate, dt
    )

    # The covariance is diagnostic only in this oracle-candidate stage.
    mean_alpha = 0.5 * (alpha_along + alpha_cross)
    updated_cov = predicted_cov.clone()
    updated_cov[:2, :2] *= max(0.20, 1.0 - 0.60 * mean_alpha)
    updated_cov = 0.5 * (updated_cov + updated_cov.T)

    return {
        "state": updated_state,
        "covariance": updated_cov,
        "innovation_m": float(innovation_norm.item()),
        "along_innovation_m": float(along.item()),
        "cross_innovation_m": float(cross.item()),
        "bounded_along_m": float(bounded_along.item()),
        "bounded_cross_m": float(bounded_cross.item()),
        "alpha_along": float(alpha_along),
        "alpha_cross": float(alpha_cross),
        "beta": float(beta),
        "huber_weight": float(huber),
    }


def metrics(rows, key):
    pred = np.asarray([r[key] for r in rows], dtype=np.float64)
    gt = np.asarray([r["gt"] for r in rows], dtype=np.float64)
    error = np.linalg.norm(pred - gt, axis=1)

    pred_step = np.diff(pred, axis=0)
    gt_step = np.diff(gt, axis=0)
    pred_step_len = np.linalg.norm(pred_step, axis=1) if len(pred_step) else np.zeros(0)
    gt_step_len = np.linalg.norm(gt_step, axis=1) if len(gt_step) else np.zeros(0)
    rpe = np.linalg.norm(pred_step - gt_step, axis=1) if len(pred_step) else np.zeros(0)

    if len(gt_step_len):
        jump_threshold = float(
            np.percentile(gt_step_len, 99) + float(config.JUMP_TOLERANCE_M)
        )
        jump_rate = float((pred_step_len > jump_threshold).mean() * 100)
        stationary = gt_step_len <= float(config.STATIONARY_GT_STEP_M)
        stationary_drift = pred_step_len[stationary]
    else:
        jump_threshold = 0.0
        jump_rate = 0.0
        stationary_drift = np.zeros(0)

    acceleration = (
        np.linalg.norm(np.diff(pred_step, axis=0), axis=1)
        if len(pred_step) >= 2
        else np.zeros(0)
    )
    pred_length = float(pred_step_len.sum())
    gt_length = float(gt_step_len.sum())

    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "MaxLE_m": float(error.max()),
        "LSR@5_pct": float((error <= 5.0).mean() * 100),
        "LSR@10_pct": float((error <= 10.0).mean() * 100),
        "LSR@15_pct": float((error <= 15.0).mean() * 100),
        "LSR@20_pct": float((error <= 20.0).mean() * 100),
        "RPE_m": float(rpe.mean()) if len(rpe) else 0.0,
        "JumpThreshold_m": jump_threshold,
        "JumpRate_pct": jump_rate,
        "PathLengthRatio": pred_length / max(gt_length, EPS),
        "AccelerationP90_m": (
            float(np.percentile(acceleration, 90)) if len(acceleration) else 0.0
        ),
        "StationaryDriftMean_m": (
            float(stationary_drift.mean()) if len(stationary_drift) else 0.0
        ),
        "StationaryDriftP90_m": (
            float(np.percentile(stationary_drift, 90))
            if len(stationary_drift)
            else 0.0
        ),
    }


def write_route_csv(path: Path, rows):
    fields = [
        "frame_id", "dt_raw", "dt_used",
        "gt_x", "gt_y",
        "candidate_center_x", "candidate_center_y", "candidate_jitter_m",
        "raw_top1_x", "raw_top1_y",
        "raw_hardms_x", "raw_hardms_y",
        "selected_visual_x", "selected_visual_y",
        "prediction_x", "prediction_y",
        "final_x", "final_y",
        "velocity_x", "velocity_y",
        "candidate_captured", "selected_mode", "boundary",
        "confidence", "margin", "entropy", "mode_local_mass",
        "mode_spatial_std", "mode_peak_prob",
        "innovation_m", "along_innovation_m", "cross_innovation_m",
        "bounded_along_m", "bounded_cross_m",
        "alpha_along", "alpha_cross", "beta", "huber_weight",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "dt_raw": row["dt_raw"],
                    "dt_used": row["dt_used"],
                    "gt_x": row["gt"][0],
                    "gt_y": row["gt"][1],
                    "candidate_center_x": row["candidate_center"][0],
                    "candidate_center_y": row["candidate_center"][1],
                    "candidate_jitter_m": row["candidate_jitter_m"],
                    "raw_top1_x": row["raw_top1"][0],
                    "raw_top1_y": row["raw_top1"][1],
                    "raw_hardms_x": row["raw_hardms"][0],
                    "raw_hardms_y": row["raw_hardms"][1],
                    "selected_visual_x": row["selected_visual"][0],
                    "selected_visual_y": row["selected_visual"][1],
                    "prediction_x": row["prediction"][0],
                    "prediction_y": row["prediction"][1],
                    "final_x": row["final"][0],
                    "final_y": row["final"][1],
                    "velocity_x": row["velocity"][0],
                    "velocity_y": row["velocity"][1],
                    "candidate_captured": int(row["candidate_captured"]),
                    "selected_mode": row["selected_mode"],
                    "boundary": int(row["boundary"]),
                    "confidence": row["confidence"],
                    "margin": row["margin"],
                    "entropy": row["entropy"],
                    "mode_local_mass": row["mode_local_mass"],
                    "mode_spatial_std": row["mode_spatial_std"],
                    "mode_peak_prob": row["mode_peak_prob"],
                    "innovation_m": row["innovation_m"],
                    "along_innovation_m": row["along_innovation_m"],
                    "cross_innovation_m": row["cross_innovation_m"],
                    "bounded_along_m": row["bounded_along_m"],
                    "bounded_cross_m": row["bounded_cross_m"],
                    "alpha_along": row["alpha_along"],
                    "alpha_cross": row["alpha_cross"],
                    "beta": row["beta"],
                    "huber_weight": row["huber_weight"],
                }
            )


def run_route(
    root,
    name,
    route_index,
    visual,
    device,
    candidate_mode: str,
    jitter_m: float,
):
    dataset, indices = tracking_frames(root, visual.origin_lat, visual.origin_lon)
    if len(indices) <= INIT_HISTORY:
        raise ValueError(f"{name} needs more than {INIT_HISTORY} frames")

    sampled_meta = [dataset.samples[i] for i in indices]
    frame_ids = np.asarray(
        [int(sample["frame_id"]) for sample in sampled_meta], dtype=np.int64
    )
    eval_gt = np.asarray(
        [[sample["x_meter"], sample["y_meter"]] for sample in sampled_meta],
        dtype=np.float32,
    )

    jitter = deterministic_jitter(frame_ids, route_index, jitter_m)
    if candidate_mode == "gt":
        jitter[:] = 0.0
    candidate_centers = eval_gt + jitter

    # Encode UAVs in bounded batches.  Cached CLIP features stay on CPU.
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

    covariance = torch.diag(
        torch.tensor(
            [
                float(config.POSITION_STD_M) ** 2,
                float(config.POSITION_STD_M) ** 2,
                float(config.VELOCITY_STD_M_PER_FRAME) ** 2,
                float(config.VELOCITY_STD_M_PER_FRAME) ** 2,
            ],
            device=device,
            dtype=state.dtype,
        )
    )

    history_positions = deque(maxlen=int(config.LINE_FIT_HISTORY))
    history_times = deque(maxlen=int(config.LINE_FIT_HISTORY))
    for p, frame_id in zip(eval_gt[:INIT_HISTORY], frame_ids[:INIT_HISTORY]):
        history_positions.append(torch.from_numpy(p).to(device))
        history_times.append(float(frame_id))

    previous_frame_id = int(frame_ids[INIT_HISTORY - 1])
    rows = []

    for t in range(INIT_HISTORY, len(indices)):
        frame_id = int(frame_ids[t])
        dt_raw = max(1, frame_id - previous_frame_id)
        dt = float(min(dt_raw, int(config.MAX_FRAME_ID_GAP)))

        previous_state = state.clone()
        predicted_state, predicted_cov = predict(state, covariance, dt)
        search_center = torch.from_numpy(candidate_centers[t]).to(device)
        measurement = visual.measure(
            clips[t:t + 1].to(device, non_blocking=True),
            predicted_state[:2].unsqueeze(0),
            predicted_state[2:].unsqueeze(0),
            search_centers=search_center.unsqueeze(0),
            grid_size=int(config.GRID_SIZE),
            sigma_along=float(config.MOTION_SIGMA_ALONG_M),
            sigma_cross=float(config.MOTION_SIGMA_CROSS_M),
            prior_weight=float(config.MOTION_PRIOR_WEIGHT),
        )

        selected = select_temporal_mode(measurement, predicted_state)
        update = straight_line_update(
            previous_state,
            predicted_state,
            predicted_cov,
            selected,
            dt,
        )
        state = update["state"]
        covariance = update["covariance"]

        history_positions.append(state[:2].clone())
        history_times.append(float(frame_id))
        fitted_velocity = line_fit_velocity(
            history_positions, history_times, state[2:]
        )
        blend = float(config.VELOCITY_LINE_BLEND) * max(
            float(config.CONFIDENCE_FLOOR), float(selected["confidence"])
        )
        velocity_candidate = (
            (1.0 - blend) * state[2:] + blend * fitted_velocity
        )
        state[2:] = constrain_turn(state[2:], velocity_candidate, dt=1.0)

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
                "candidate_center": candidate_centers[t].tolist(),
                "candidate_jitter_m": float(np.linalg.norm(jitter[t])),
                "raw_top1": measurement.raw_top1_xy[0].cpu().tolist(),
                "raw_hardms": measurement.raw_visual_xy[0].cpu().tolist(),
                "selected_visual": selected["xy"].cpu().tolist(),
                "prediction": predicted_state[:2].cpu().tolist(),
                "final": state[:2].cpu().tolist(),
                "velocity": state[2:].cpu().tolist(),
                "candidate_captured": captured,
                "selected_mode": int(selected["mode_id"]),
                "boundary": bool(selected["boundary"]),
                "confidence": float(selected["confidence"]),
                "margin": float(selected["margin"]),
                "entropy": float(selected["entropy"]),
                "mode_local_mass": float(selected["local_mass"]),
                "mode_spatial_std": float(selected["spatial_std"]),
                "mode_peak_prob": float(selected["peak_prob"]),
                **update,
            }
        )
        # Remove tensor-valued fields copied from update.
        rows[-1].pop("state", None)
        rows[-1].pop("covariance", None)
        previous_frame_id = frame_id

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_route_csv(config.OUTPUT_DIR / f"{name}_robust_frames.csv", rows)

    return {
        "route": name,
        "Experiment": "Oracle Candidate Temporal Smoothing",
        "CandidateCenterMode": candidate_mode,
        "GTJitterMax_m": float(jitter_m if candidate_mode == "gt_jitter" else 0.0),
        "GTUsedAfterInitializationByFilter": False,
        "GTUsedForCandidateCenter": True,
        "GridSize": int(config.GRID_SIZE),
        "SampledFrames": len(indices),
        "InitialSpeed_m_per_frame_id": float(np.linalg.norm(initial_velocity)),
        "RawTop1": metrics(rows, "raw_top1"),
        "RawHardMS": metrics(rows, "raw_hardms"),
        "TemporalSelectedMode": metrics(rows, "selected_visual"),
        "StraightPrediction": metrics(rows, "prediction"),
        "StraightLineSmoothedHardMS": metrics(rows, "final"),
        "CandidateCaptureRate_pct": float(
            np.mean([r["candidate_captured"] for r in rows]) * 100.0
        ),
        "MeanVisualConfidence": float(
            np.mean([r["confidence"] for r in rows])
        ),
        "BoundaryRate_pct": float(
            np.mean([r["boundary"] for r in rows]) * 100.0
        ),
    }


def main():
    args = parse_args()
    torch.manual_seed(int(config.SEED))
    np.random.seed(int(config.SEED))
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    visual = FrozenVisualLocalizer(device)

    selected_routes = set(args.routes) if args.routes else None
    route_items = [
        (root, name, i)
        for i, (root, name) in enumerate(zip(config.ROUTE_ROOTS, config.ROUTE_NAMES))
        if selected_routes is None or name in selected_routes
    ]
    if not route_items:
        raise ValueError(f"No valid routes selected: {args.routes}")

    results = [
        run_route(
            root,
            name,
            route_index,
            visual,
            device,
            args.candidate_center,
            max(0.0, float(args.jitter_m)),
        )
        for root, name, route_index in route_items
    ]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.OUTPUT_DIR / "robust_tracker_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    for result in results:
        raw = result["RawHardMS"]
        smooth = result["StraightLineSmoothedHardMS"]
        print(
            f"{result['route']}: "
            f"raw MLE={raw['MLE_m']:.2f}m, jump={raw['JumpRate_pct']:.2f}% | "
            f"smooth MLE={smooth['MLE_m']:.2f}m, "
            f"P90={smooth['P90_m']:.2f}m, "
            f"jump={smooth['JumpRate_pct']:.2f}%, "
            f"stationary-P90={smooth['StationaryDriftP90_m']:.2f}m | "
            f"CCR={result['CandidateCaptureRate_pct']:.2f}%"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()