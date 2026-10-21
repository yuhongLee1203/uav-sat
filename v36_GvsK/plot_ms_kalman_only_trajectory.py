#!/usr/bin/env python3
"""Plot MS + Kalman-only localization trajectories and motion stability.

Reads the per-frame CSVs produced by run_ms_kalman_only_eval.sh and writes:
  - route_B_trajectory.png / route_C_trajectory.png
  - route_B_motion_stability.png / route_C_motion_stability.png

Trajectory plot:
  reference trajectory (GT/reference positions), raw SoftMS visual anchor,
  Kalman prior prediction, and final Kalman posterior.

Motion-stability plot:
  final estimator step, reference-frame step, localization error, and marks
  frames already classified as abnormal jumps by the evaluation code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def col(rows, name, default=np.nan):
    values = []
    for row in rows:
        value = row.get(name, "")
        if value in (None, ""):
            values.append(default)
        else:
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def plot_trajectory(rows, route_name: str, out_dir: Path):
    gt_x = col(rows, "gt_x")
    gt_y = col(rows, "gt_y")
    ms_x = col(rows, "visual_anchor_x")
    ms_y = col(rows, "visual_anchor_y")
    prior_x = col(rows, "predicted_x")
    prior_y = col(rows, "predicted_y")
    final_x = col(rows, "final_x")
    final_y = col(rows, "final_y")

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(gt_x, gt_y, linewidth=2.2, label="Reference trajectory")
    ax.plot(ms_x, ms_y, linewidth=0.8, alpha=0.45, label="SoftMS measurement")
    ax.plot(prior_x, prior_y, linewidth=1.0, alpha=0.75, label="Kalman prior")
    ax.plot(final_x, final_y, linewidth=1.8, label="Kalman final")

    # Mark start/end clearly without overcrowding the trajectory.
    ax.scatter([gt_x[0]], [gt_y[0]], marker="o", s=70, label="Start")
    ax.scatter([gt_x[-1]], [gt_y[-1]], marker="X", s=85, label="End")

    ax.set_title(f"{route_name}: SoftMS + Kalman-only trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{route_name}_trajectory.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_motion_stability(rows, route_name: str, out_dir: Path):
    frame = col(rows, "frame_id")
    final_step = col(rows, "final_step_m")
    ref_step = col(rows, "gt_step_for_jump_m")
    error = col(rows, "error_final_m")
    abnormal = col(rows, "abnormal_jump", default=0.0) > 0.5

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(frame, final_step, linewidth=1.0, label="Kalman final step")
    ax.plot(frame, ref_step, linewidth=1.0, alpha=0.75, label="Reference step")
    ax.plot(frame, error, linewidth=0.9, alpha=0.75, label="Localization error")
    if abnormal.any():
        ax.scatter(frame[abnormal], final_step[abnormal], marker="x", s=45, label="Abnormal jump")

    ax.set_title(f"{route_name}: motion stability / jitter check")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Meters")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{route_name}_motion_stability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def summarize(rows, route_name: str):
    final_step = col(rows, "final_step_m")
    ref_step = col(rows, "gt_step_for_jump_m")
    error = col(rows, "error_final_m")
    abnormal = col(rows, "abnormal_jump", default=0.0) > 0.5
    excess = np.maximum(final_step - ref_step, 0.0)
    print(
        f"{route_name}: frames={len(rows)} "
        f"MLE={np.nanmean(error):.3f}m "
        f"max_step={np.nanmax(final_step):.3f}m "
        f"mean_excess_step={np.nanmean(excess):.3f}m "
        f"abnormal_jumps={int(abnormal.sum())} ({100.0*np.mean(abnormal):.3f}%)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "ms_kalman_only",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(out_dir.glob("route_*_controlled_gtprior_*_frames.csv"))
    if not csvs:
        raise FileNotFoundError(f"no route frame CSVs found in {out_dir}")

    created = []
    for path in csvs:
        route_name = path.name.split("_controlled_gtprior", 1)[0]
        rows = read_csv(path)
        summarize(rows, route_name)
        created.append(plot_trajectory(rows, route_name, out_dir))
        created.append(plot_motion_stability(rows, route_name, out_dir))

    print("Created plots:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
