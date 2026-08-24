#!/usr/bin/env python3
"""Create the two data-backed figures reserved in FieldAnchor0821v2."""

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
OUT = ROOT / "masterpaper" / "figures_v2"
NATIVE = EXP / "outputs" / "native-paper-trajectories"
V36_BC = EXP / "outputs" / "internal" / "waypoint486_650_3x6_BC" / "full_v36"
ABLATION = EXP / "outputs" / "internal" / "corrected_v2"
VISUAL_CHECKPOINT = ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman" / "checkpoints" / "visual_retrieval_A_only.pt"


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def route_rows(root: Path, route: str):
    files = sorted(root.glob(f"{route}_*_frames.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No {route} frames under {root}")
    return load_rows(files[-1])


def affine_world_to_pixel():
    checkpoint = torch.load(VISUAL_CHECKPOINT, map_location="cpu")
    gallery = checkpoint["gallery"]
    xy, pixel = gallery["xy"].float().numpy(), gallery["pixel"].float().numpy()
    return np.linalg.lstsq(np.column_stack((xy, np.ones(len(xy)))), pixel, rcond=None)[0]


def pixels(xy, affine):
    return np.column_stack((xy, np.ones(len(xy)))) @ affine


def satellite_image():
    spec = importlib.util.spec_from_file_location("v36_config", EXP / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    with Image.open(config.SAT_IMAGE) as raw:
        return np.asarray(ImageEnhance.Brightness(raw.convert("RGB")).enhance(1.22))


def save_both(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure10():
    """Same Route-B frame and same map crop for every native method plus ours."""
    route, frame_id = "route_B", 1212
    methods = ["DenseUAV", "Sample4Geo", "Game4Loc", "InfoGeo", "Bearing-UAV"]
    rows = route_rows(V36_BC, route)
    row = next(row for row in rows if int(row["frame_id"]) == frame_id)
    gt_xy = np.array([float(row["gt_x"]), float(row["gt_y"])])
    ours_xy = np.array([float(row["final_x"]), float(row["final_y"])])
    payloads = []
    for method in methods:
        data = np.load(NATIVE / method / f"{route}_predictions.npz")
        index = int(np.where(data["frame_id"] == frame_id)[0][0])
        predicted_xy = np.asarray(data["pred_xy"][index], dtype=np.float64)
        # Every panel must use the exact same recorded coordinate as its answer.
        # Native table metrics may use a positive gallery-patch centre instead,
        # but that protocol-specific target is not appropriate for this figure.
        payloads.append((method, predicted_xy, float(np.linalg.norm(predicted_xy - gt_xy))))
    ours_error = float(np.linalg.norm(ours_xy - gt_xy))
    payloads.append(("FieldAnchor-LR", ours_xy, ours_error))

    affine, image = affine_world_to_pixel(), satellite_image()
    gt_px = pixels(gt_xy[None, :], affine)[0]
    pred_px = np.asarray([pixels(pred[None, :], affine)[0] for _, pred, _ in payloads])
    all_px = np.vstack((gt_px[None, :], pred_px))
    # One common extent is essential: it makes the spatial errors comparable.
    margin = 80
    x0, x1 = all_px[:, 0].min() - margin, all_px[:, 0].max() + margin
    y0, y1 = all_px[:, 1].min() - margin, all_px[:, 1].max() + margin

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0))
    fig.subplots_adjust(left=.028, right=.992, bottom=.10, top=.83, hspace=.15, wspace=.035)
    fig.suptitle("Cross-method localization on the same held-out UAV query", y=.975, fontsize=17, weight="bold")
    fig.text(.172, .895, f"Route B · frame {frame_id} · identical map extent in every panel", fontsize=10.5, color="#4D5966")
    query = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    query_ax = fig.add_axes([.035, .855, .105, .085])
    query_ax.imshow(query)
    query_ax.set_title("Shared UAV query", fontsize=8.5, pad=2, color="#27313B")
    for spine in query_ax.spines.values():
        spine.set_visible(True); spine.set_color("#27313B"); spine.set_linewidth(.8)
    query_ax.set_axis_off()
    for ax, (method, pred_xy, error) in zip(axes.flat, payloads):
        ax.imshow(image)
        ax.scatter(gt_px[0], gt_px[1], s=88, marker="o", c="white", edgecolors="#15202B", linewidths=1.35, zorder=3)
        pred = pixels(pred_xy[None, :], affine)[0]
        ax.scatter(pred[0], pred[1], s=92, marker="^", c="#E74C3C", edgecolors="white", linewidths=1.0, zorder=4)
        ax.plot([gt_px[0], pred[0]], [gt_px[1], pred[1]], color="#F4C542", lw=1.65, alpha=.95, zorder=2)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off()
        title_color = "#168A71" if method == "FieldAnchor-LR" else "#17212B"
        ax.set_title(f"{method}  |  {error:.1f} m", fontsize=10.5, weight="bold", color=title_color, pad=5)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#15202B", markersize=8, label="GT / correct location"),
            Line2D([], [], marker="^", linestyle="", markerfacecolor="#E74C3C", markeredgecolor="white", markersize=8, label="Method prediction"),
            Line2D([], [], color="#F4C542", lw=2, label="Localization error"),
        ],
        loc="lower center", bbox_to_anchor=(.5, .007), ncol=3, frameon=True,
        facecolor="#1E252B", edgecolor="none", labelcolor="white", fontsize=10,
        columnspacing=2.4, handlelength=1.8,
    )
    save_both(fig, "fig10_cross_method_same_query")


def figure11():
    """Raw (unsmoothed) per-frame errors on one shared Route-C interval."""
    route, start, end = "route_C", 253, 402
    variants = [
        ("MS only", "softms_only", "#8E44AD"),
        ("MS + 3-frame GRU", "softms_gru", "#E67E22"),
        ("MS + GRU + inertia", "softms_gru_poly", "#2980B9"),
        ("FieldAnchor-LR (full)", "full_v36", "#16A085"),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.8), constrained_layout=True)
    for label, folder, color in variants:
        rows = route_rows(ABLATION / folder, route)
        frame = np.asarray([int(row["frame_id"]) for row in rows])
        error = np.asarray([float(row["error_final_m"]) for row in rows])
        keep = (frame >= start) & (frame <= end)
        # Plot the recorded outputs exactly; no rolling mean or visual smoothing.
        ax.plot(frame[keep], error[keep], color=color, lw=1.45, alpha=.95, label=label)
    ax.set_title("Temporal stability on one contiguous held-out Route C sequence", fontsize=15, weight="bold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Localization error (m)")
    ax.set_xlim(start, end)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", lw=.8)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.text(.01, .96, "Raw per-frame outputs; no smoothing", transform=ax.transAxes, va="top", fontsize=9.5, color="#555555")
    save_both(fig, "fig11_temporal_stability_routeC_frames253_402")


if __name__ == "__main__":
    figure10()
    figure11()
