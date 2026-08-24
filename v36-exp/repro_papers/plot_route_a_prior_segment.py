#!/usr/bin/env python3
"""Plot the executed Route-A start-to-waypoint-1 local-prior experiment."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image, ImageEnhance


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
RUN = EXP / "outputs" / "internal" / "routeA_start_to_wp1_3x6" / "full_v36"
CHECKPOINT = ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman" / "checkpoints" / "visual_retrieval_A_only.pt"
OUT = ROOT / "masterpaper" / "figures_v2"


def world_to_pixel_affine():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    gallery = checkpoint["gallery"]
    xy, pixel = gallery["xy"].float().numpy(), gallery["pixel"].float().numpy()
    return np.linalg.lstsq(np.column_stack((xy, np.ones(len(xy)))), pixel, rcond=None)[0]


def pixels(xy, affine):
    return np.column_stack((xy, np.ones(len(xy)))) @ affine


def map_image():
    spec = importlib.util.spec_from_file_location("v36_config", EXP / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    with Image.open(config.SAT_IMAGE) as raw:
        return np.asarray(ImageEnhance.Brightness(raw.convert("RGB")).enhance(1.18))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frames = sorted(RUN.glob("route_A_*_frames.csv"), key=lambda path: path.stat().st_mtime)
    if not frames:
        raise FileNotFoundError(f"Missing Route-A segment output under {RUN}")
    with frames[-1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    gt = np.asarray([[float(r["gt_x"]), float(r["gt_y"])] for r in rows])
    prior = np.asarray([[float(r["prior_center_x"]), float(r["prior_center_y"])] for r in rows])
    pred = np.asarray([[float(r["final_x"]), float(r["final_y"])] for r in rows])
    error = np.asarray([float(r["error_final_m"]) for r in rows])
    affine = world_to_pixel_affine()
    original_image = map_image()
    # This first leg is nearly north--south. Rotate map and coordinates together
    # so the dense frame references can be inspected without a page of whitespace.
    image = np.rot90(original_image)
    image_width = original_image.shape[1]
    def rotated(points):
        return np.column_stack((points[:, 1], image_width - 1 - points[:, 0]))
    gt_px = rotated(pixels(gt, affine))
    prior_px = rotated(pixels(prior, affine))
    pred_px = rotated(pixels(pred, affine))
    all_px = np.vstack((gt_px, prior_px, pred_px))
    # Retain generous cross-track map context; this is a geographic figure, not
    # a thin line strip.
    along_margin, cross_margin = 115, 390
    limits = (all_px[:, 0].min() - along_margin, all_px[:, 0].max() + along_margin,
              all_px[:, 1].max() + cross_margin, all_px[:, 1].min() - cross_margin)
    def base_axis(title):
        fig, ax = plt.subplots(figsize=(11.2, 5.6))
        ax.imshow(image)
        ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[2], limits[3])
        ax.set_title(title, fontsize=13, weight="bold", pad=7)
        ax.set_axis_off()
        return fig, ax

    def endpoints(ax):
        # Keep the waypoint X convention, then overlay explicitly identifiable
        # start/end markers so a reader never has to infer direction.
        ax.scatter(gt_px[[0, -1], 0], gt_px[[0, -1], 1], marker="X", s=100,
                   c="#FFD60A", edgecolors="black", linewidths=.8, zorder=6)
        ax.scatter(gt_px[0, 0], gt_px[0, 1], s=68, c="#39E75F",
                   edgecolors="black", linewidths=.9, zorder=7)
        ax.scatter(gt_px[-1, 0], gt_px[-1, 1], s=68, c="#FF3B30",
                   edgecolors="white", linewidths=.9, zorder=7)

    def legend(ax, with_prediction):
        handles = [
            Line2D([], [], color="white", lw=3.6, label="GT path"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#FF9F1C", markeredgecolor="none", markersize=7, label="Dense prior reference"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#39E75F", markeredgecolor="black", markersize=8, label="Start"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#FF3B30", markeredgecolor="white", markersize=8, label="Waypoint 1 / end"),
        ]
        if with_prediction:
            handles.insert(2, Line2D([], [], color="#9BFF3A", lw=2.4, label="FieldAnchor-LR prediction"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.11),
                  ncol=(5 if with_prediction else 4), frameon=True, fontsize=9.5,
                  facecolor="#1E252B", edgecolor="none", labelcolor="white",
                  handlelength=1.8, columnspacing=1.4)

    fig, ax = base_axis("Route A: dense local-prior reference points (start → waypoint 1)")
    ax.plot(gt_px[:, 0], gt_px[:, 1], color="white", lw=3.6, alpha=.96, zorder=2)
    ax.scatter(prior_px[:, 0], prior_px[:, 1], s=12, c="#FF9F1C", alpha=.72, edgecolors="none", zorder=3)
    endpoints(ax)
    legend(ax, with_prediction=False)
    fig.subplots_adjust(bottom=.18)
    save(fig, "routeA_start_to_waypoint1_gt_dense_prior_points")

    fig, ax = base_axis("Route A: FieldAnchor-LR 3×6 result (start → waypoint 1)")
    ax.scatter(prior_px[:, 0], prior_px[:, 1], s=9, c="#FF9F1C", alpha=.42, edgecolors="none", zorder=2)
    ax.plot(gt_px[:, 0], gt_px[:, 1], color="white", lw=3.7, alpha=.96, zorder=3)
    ax.plot(pred_px[:, 0], pred_px[:, 1], color="#9BFF3A", lw=2.2, alpha=.98, zorder=4)
    endpoints(ax)
    legend(ax, with_prediction=True)
    fig.subplots_adjust(bottom=.18)
    save(fig, "routeA_start_to_waypoint1_v36_prediction")
    print(f"frames={len(rows)} MLE={error.mean():.4f} P90={np.quantile(error, .90):.4f}")


if __name__ == "__main__":
    main()
