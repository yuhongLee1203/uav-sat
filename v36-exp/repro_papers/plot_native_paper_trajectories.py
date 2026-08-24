#!/usr/bin/env python3
"""Render actual B/C trajectories emitted by the adapted native-paper runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
BASE = ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
DEFAULT_ROOT = EXP / "outputs" / "native-paper-trajectories"
DEFAULT_OUT = EXP / "figures" / "other_papers"
REFERENCE_RUN_B = EXP / "outputs" / "internal" / "softms_vs_centroid_rerun_20260820" / "full_v36"
REFERENCE_RUN_C = EXP / "outputs" / "internal" / "waypoint650_3x6" / "full_v36"


def world_to_pixel_affine():
    checkpoint = torch.load(
        BASE / "checkpoints" / "visual_retrieval_A_only.pt", map_location="cpu"
    )
    gallery = checkpoint["gallery"]
    xy = gallery["xy"].float().cpu().numpy()
    pixel = gallery["pixel"].float().cpu().numpy()
    # [world_x, world_y, 1] @ affine = [pixel_x, pixel_y]
    affine, *_ = np.linalg.lstsq(
        np.column_stack((xy, np.ones(len(xy)))), pixel, rcond=None
    )
    return affine


def as_pixel(xy, affine):
    return np.column_stack((xy, np.ones(len(xy)))) @ affine


def chronological_route(route):
    root = REFERENCE_RUN_B if route == "route_B" else REFERENCE_RUN_C
    files = sorted(root.glob(f"{route}_*_frames.csv"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"Missing chronological {route} reference CSV under {root}")
    with files[-1].open(newline="", encoding="utf-8") as handle:
        return np.asarray([[float(row["gt_x"]), float(row["gt_y"])] for row in csv.DictReader(handle)])


def draw_panel(ax, image, method, route, payload, affine):
    # These native baselines are frame-independent global localizers.  Joining
    # their outputs with a line would fabricate a temporal trajectory, so show
    # their actual per-frame locations as points instead.
    ax.imshow(image)
    gt = as_pixel(chronological_route(route), affine)
    pred = as_pixel(payload["pred_xy"], affine)
    ax.plot(gt[:, 0], gt[:, 1], color="white", lw=2.1, alpha=0.9, zorder=3)
    ax.scatter(pred[:, 0], pred[:, 1], c="#FF5A36", s=5, alpha=0.45, edgecolors="none", zorder=4)
    ax.scatter(gt[0, 0], gt[0, 1], c="#39E75F", edgecolors="black", linewidths=0.55, s=34, zorder=5)
    ax.scatter(gt[-1, 0], gt[-1, 1], c="#FF3B30", edgecolors="white", linewidths=0.55, s=34, zorder=5)
    ax.set_title(f"{method} — {route.replace('_', ' ').title()}", fontsize=12, weight="bold")
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--methods", nargs="+",
        default=["DenseUAV", "Sample4Geo", "Game4Loc", "InfoGeo", "Bearing-UAV"],
    )
    args = parser.parse_args()
    known = {name: name for name in ("DenseUAV", "Sample4Geo", "Game4Loc", "InfoGeo", "Bearing-UAV")}
    invalid = [name for name in args.methods if name not in known]
    if invalid:
        raise ValueError(f"Unknown method(s): {invalid}")
    methods = [(name, known[name]) for name in args.methods]
    missing = [
        args.trajectory_root / folder / f"{route}_predictions.npz"
        for _, folder in methods for route in ("route_B", "route_C")
        if not (args.trajectory_root / folder / f"{route}_predictions.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing trajectory output(s):\n" + "\n".join(map(str, missing)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("v36_config", EXP / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    with Image.open(config.SAT_IMAGE) as raw:
        image = np.asarray(ImageEnhance.Brightness(raw.convert("RGB")).enhance(1.15))
    affine = world_to_pixel_affine()
    fig, axes = plt.subplots(2 * len(methods), 1, figsize=(8.5, 11 * len(methods)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for method_index, (title, folder) in enumerate(methods):
        for route_index, route in enumerate(("route_B", "route_C")):
            payload = np.load(args.trajectory_root / folder / f"{route}_predictions.npz")
            draw_panel(axes[2 * method_index + route_index], image, title, route, payload, affine)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ("_".join(args.methods) + "_prediction_locations_BC.png")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
