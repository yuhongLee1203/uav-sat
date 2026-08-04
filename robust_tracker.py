"""Training-free temporal prior-guided HardMS tracker.

The tracker keeps a position/velocity state. HardMS is a visual measurement,
not a command: an innovation-consistent update is applied only to the extent
allowed by a robust Mahalanobis gate. This file intentionally contains no GRU,
no learned residual and no recovery/global search.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import config
from data import RouteDataset
from visual_localizer import FrozenVisualLocalizer


INIT_HISTORY = 5
POSITION_STD_M = 4.0
VELOCITY_STD_M = 3.0
PROCESS_POSITION_STD_M = 1.0
PROCESS_VELOCITY_STD_M = 1.5
MAX_POSITION_STD_M = 10.0
CHI2_99_2D = 9.21  # 99% chi-square gate for a two-dimensional innovation.
JUMP_M = 15.0


def motion_nodes(root, origin_lat, origin_lon):
    """Collapse consecutive images with the same logged GPS position."""
    dataset = RouteDataset(root, train=False, origin_lat=origin_lat, origin_lon=origin_lon)
    groups, current = [], [0]
    for i in range(1, len(dataset.samples)):
        a, b = dataset.samples[i - 1], dataset.samples[i]
        if np.hypot(b["x_meter"] - a["x_meter"], b["y_meter"] - a["y_meter"]) <= config.TEMPORAL_POSITION_MERGE_M:
            current.append(i)
        else:
            groups.append(current[len(current) // 2]); current = [i]
    groups.append(current[len(current) // 2])
    return dataset, groups


def robust_update(state, covariance, measurement, entropy):
    """One constant-velocity prediction followed by a Huberised Kalman update."""
    dtype, device = state.dtype, state.device
    F = torch.eye(4, device=device, dtype=dtype)
    F[0, 2] = F[1, 3] = 1.0
    Q = torch.diag(torch.tensor([
        PROCESS_POSITION_STD_M**2, PROCESS_POSITION_STD_M**2,
        PROCESS_VELOCITY_STD_M**2, PROCESS_VELOCITY_STD_M**2,
    ], device=device, dtype=dtype))
    predicted_state = F @ state
    predicted_cov = F @ covariance @ F.T + Q
    # Entropy may widen R slightly, but must never make any far response look
    # statistically plausible. The old 3--10m range disabled the gate.
    visual_std = 2.5 + 2.0 * float(entropy)
    R = torch.eye(2, device=device, dtype=dtype) * visual_std**2
    H = torch.zeros((2, 4), device=device, dtype=dtype); H[0, 0] = H[1, 1] = 1.0
    innovation = measurement - H @ predicted_state
    S = H @ predicted_cov @ H.T + R
    d2 = innovation @ torch.linalg.solve(S, innovation)
    # Standard innovation gating: an observation outside the 99% predicted
    # ellipse is rejected, rather than partially dragging the state outward.
    robust_weight = 1.0 if float(d2) <= CHI2_99_2D else 0.0
    # Visual localization corrects position only. It must not rewrite the
    # inertial velocity state: an early false peak otherwise creates a wrong
    # velocity which continues to grow after later visual updates are rejected.
    position_gain = predicted_cov[:2, :2] @ torch.linalg.inv(S)
    updated_state = predicted_state.clone()
    updated_state[:2] = predicted_state[:2] + robust_weight * (position_gain @ innovation)
    updated_cov = predicted_cov.clone()
    updated_cov[:2, :2] = (torch.eye(2, device=device, dtype=dtype) - robust_weight * position_gain) @ predicted_cov[:2, :2]
    updated_cov[:2, 2:] = 0.0
    updated_cov[2:, :2] = 0.0
    updated_cov = 0.5 * (updated_cov + updated_cov.T)
    # Keep state uncertainty bounded: prolonged visual loss must not turn into
    # an unrestricted future visual update.
    updated_cov[0, 0] = updated_cov[0, 0].clamp_max(MAX_POSITION_STD_M**2)
    updated_cov[1, 1] = updated_cov[1, 1].clamp_max(MAX_POSITION_STD_M**2)
    updated_cov[2, 2] = updated_cov[2, 2].clamp_max(VELOCITY_STD_M**2)
    updated_cov[3, 3] = updated_cov[3, 3].clamp_max(VELOCITY_STD_M**2)
    return predicted_state, updated_state, updated_cov, float(d2), robust_weight


def metrics(rows, key):
    pred = np.asarray([r[key] for r in rows]); gt = np.asarray([r["gt"] for r in rows])
    error = np.linalg.norm(pred - gt, axis=1)
    step = np.diff(pred, axis=0); gt_step = np.diff(gt, axis=0)
    velocity_error = np.linalg.norm(step - gt_step, axis=1)
    absolute_jump = np.linalg.norm(step, axis=1) > JUMP_M
    unexpected_jump = velocity_error > JUMP_M
    # Jump then return: a large move whose next output falls near the previous
    # estimate, a direct signature of a transient false visual peak.
    return_event = np.zeros(len(rows) - 2, dtype=bool)
    if len(rows) >= 3:
        return_event = (np.linalg.norm(step[:-1], axis=1) > JUMP_M) & (np.linalg.norm(pred[2:] - pred[:-2], axis=1) < JUMP_M)
    return {
        "MLE_m": float(error.mean()), "MedLE_m": float(np.median(error)), "P90_m": float(np.percentile(error, 90)),
        "LSR@5_pct": float((error <= 5).mean() * 100), "LSR@10_pct": float((error <= 10).mean() * 100),
        "LSR@15_pct": float((error <= 15).mean() * 100), "LSR@20_pct": float((error <= 20).mean() * 100),
        "RPE_m": float(velocity_error.mean()), "AbsoluteJump@15_pct": float(absolute_jump.mean() * 100),
        "UnexpectedJump@15_pct": float(unexpected_jump.mean() * 100), "JumpReturn@15_pct": float(return_event.mean() * 100) if len(return_event) else 0.0,
    }


@torch.no_grad()
def run_route(root, name, visual, device):
    dataset, indices = motion_nodes(root, visual.origin_lat, visual.origin_lon)
    frames = [dataset[i] for i in indices]
    uav = torch.stack([f["uav"] for f in frames]).to(device)
    gt = torch.stack([f["xy"] for f in frames]).cpu().numpy()
    clips = torch.cat([visual.encode_uav_clip(uav[i:i + 64]) for i in range(0, len(uav), 64)], dim=0)
    state = torch.zeros(4, device=device)
    state[:2] = torch.from_numpy(gt[INIT_HISTORY - 1]).to(device)
    velocity = torch.from_numpy((gt[1:INIT_HISTORY] - gt[:INIT_HISTORY - 1]).mean(axis=0)).to(device)
    state[2:] = velocity
    covariance = torch.diag(torch.tensor([POSITION_STD_M**2, POSITION_STD_M**2, VELOCITY_STD_M**2, VELOCITY_STD_M**2], device=device))
    rows = []
    for t in range(INIT_HISTORY, len(frames)):
        predicted_xy = state[:2].unsqueeze(0) + state[2:].unsqueeze(0)
        measurement = visual.measure(clips[t:t+1], predicted_xy, grid_size=config.GRID_SIZE, motion_sigma=config.MOTION_SIGMA_M)
        pred_state, state, covariance, d2, weight = robust_update(state, covariance, measurement.visual_xy.squeeze(0), measurement.entropy.item())
        rows.append({
            "frame_id": frames[t]["frame_id"], "gt": gt[t].tolist(), "visual": measurement.visual_xy.squeeze(0).cpu().tolist(),
            "prediction": pred_state[:2].cpu().tolist(), "robust": state[:2].cpu().tolist(), "d2": d2, "weight": weight,
        })
    out_dir = config.OUTPUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_robust_frames.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["frame_id", "gt_x", "gt_y", "visual_x", "visual_y", "prediction_x", "prediction_y", "robust_x", "robust_y", "mahalanobis_d2", "visual_update_weight"])
        for r in rows: writer.writerow([r["frame_id"], *r["gt"], *r["visual"], *r["prediction"], *r["robust"], r["d2"], r["weight"]])
    return {"route": name, "VisualHardMS": metrics(rows, "visual"), "MotionPrediction": metrics(rows, "prediction"), "RobustTemporalHardMS": metrics(rows, "robust"), "mean_visual_update_weight": float(np.mean([r["weight"] for r in rows]))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual = FrozenVisualLocalizer(device)
    results = [run_route(root, name, visual, device) for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES)]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT_DIR / "robust_tracker_summary.json").open("w") as f: json.dump(results, f, indent=2)
    for r in results:
        m = r["RobustTemporalHardMS"]
        print(f"{r['route']}: MLE={m['MLE_m']:.2f}m | unexpected-jump={m['UnexpectedJump@15_pct']:.2f}% | jump-return={m['JumpReturn@15_pct']:.2f}%")


if __name__ == "__main__": main()
