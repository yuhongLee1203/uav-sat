#!/usr/bin/env python3
"""Render matching full-V36 (forward 3x6) Route-B/C trajectory figures."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
RUN = EXP / "outputs" / "internal" / "waypoint486_650_3x6_BC" / "full_v36"
BASE = ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
OUT = EXP / "figures" / "ours"


def affine_world_to_pixel():
    checkpoint = torch.load(BASE / "checkpoints" / "visual_retrieval_A_only.pt", map_location="cpu")
    gallery = checkpoint["gallery"]
    xy, pixel = gallery["xy"].float().numpy(), gallery["pixel"].float().numpy()
    return np.linalg.lstsq(np.column_stack((xy, np.ones(len(xy)))), pixel, rcond=None)[0]


def pixels(xy, affine):
    return np.column_stack((xy, np.ones(len(xy)))) @ affine


def render(route_name: str):
    files = sorted(RUN.glob(f"{route_name}_*_frames.csv"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No completed {route_name} frames under {RUN}")
    with files[-1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frame = np.asarray([int(row["frame_id"]) for row in rows])
    gt = np.asarray([[float(row["gt_x"]), float(row["gt_y"])] for row in rows])
    final = np.asarray([[float(row["final_x"]), float(row["final_y"])] for row in rows])
    waypoints = json.loads((ROOT / "route_waypoints" / f"{route_name}_waypoints.json").read_text())["waypoints"]
    waypoint_ids = [int(point["frame_index"]) for point in waypoints]
    waypoint_rows = [int(np.argmin(np.abs(frame - value))) for value in waypoint_ids]
    spec = importlib.util.spec_from_file_location("v36_config", EXP / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    with Image.open(config.SAT_IMAGE) as raw:
        image = np.asarray(ImageEnhance.Brightness(raw.convert("RGB")).enhance(1.15))
    affine = affine_world_to_pixel()
    gt_px, final_px = pixels(gt, affine), pixels(final, affine)
    waypoint_px = gt_px[waypoint_rows]
    fig, ax = plt.subplots(figsize=(8.5, 11), constrained_layout=True)
    ax.imshow(image)
    # Keep the route large enough to inspect; no city-scale empty margin or in-map legend.
    all_px = np.vstack((gt_px, final_px))
    margin = 85
    ax.set_xlim(all_px[:, 0].min() - margin, all_px[:, 0].max() + margin)
    ax.set_ylim(all_px[:, 1].max() + margin, all_px[:, 1].min() - margin)
    ax.plot(gt_px[:, 0], gt_px[:, 1], color="white", lw=4.0, alpha=0.9, zorder=3)
    ax.plot(final_px[:, 0], final_px[:, 1], color="#9BFF3A", lw=2.0, alpha=0.95, zorder=4)
    ax.scatter(waypoint_px[:, 0], waypoint_px[:, 1], marker="X", s=58, c="#FFD60A", edgecolors="black", linewidths=0.7, zorder=5)
    ax.scatter(gt_px[0, 0], gt_px[0, 1], c="#39E75F", edgecolors="black", s=48, zorder=6)
    ax.scatter(gt_px[-1, 0], gt_px[-1, 1], c="#FF3B30", edgecolors="white", s=48, zorder=6)
    ax.set_title(f"V36 3×6 — {route_name.replace('_', ' ').title()}", fontsize=15, weight="bold", pad=8)
    ax.set_axis_off()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"v36_3x6_waypoint486_650_{route_name}_actual_trajectory.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    print(output)


def main():
    for route_name in ("route_B", "route_C"):
        render(route_name)


if __name__ == "__main__":
    main()
