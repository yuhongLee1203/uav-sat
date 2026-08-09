#!/usr/bin/env python3
"""Render identical straight and turning route segments for every method.

The script only reads completed predictions.  It does not train or rerun any
model.  Every panel uses the same frame IDs, coordinate crop, GT, and map.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import SatGeoMapper, twd97_from_latlon


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "same_route_allPaper_exp"
RUN = ROOT / "outputs" / "strict_train_A_test_BC_t2only_w3"
COMPARE = ROOT.parents[0] / "Go_aaai" / "comapred_paper" / "results" / "newdata_local36_20260806_222100"
BEARING = ROOT.parents[0] / "Go_aaai" / "bearinguav_results"
SAT_IMAGE = ROOT.parents[0] / "sim_data" / "sim_competition_crop_check" / "sim_map_competition_roi_crop.png"
SAT_JSON = ROOT.parents[0] / "sim_data" / "sim_competition_crop_check" / "sim_map_competition_roi_crop_worldfile_epsg3826.json"

COLORS = {
    "GT": "#111827",
    "Raw Top-1": "#d55e00",
    "Fixed HardMS": "#0072b2",
    "RTL-CRF (3 frames)": "#009e73",
    "MobileCLIP": "#7c3aed",
    "Sample4Geo": "#cc79a7",
    "DenseUAV": "#e69f00",
    "Game4Loc": "#56b4e9",
    "Bearing-UAV": "#6b7280",
}


def xy_to_pixel(xy: np.ndarray, origin_lat: float, origin_lon: float, mapper: SatGeoMapper) -> np.ndarray:
    pixels = []
    cos_lat = math.cos(math.radians(origin_lat))
    for x_m, y_m in np.asarray(xy, dtype=float):
        lat = origin_lat + math.degrees(y_m / 6378137.0)
        lon = origin_lon + math.degrees(x_m / (6378137.0 * cos_lat))
        pixels.append(mapper.latlon_to_pixel(lat, lon))
    return np.asarray(pixels, dtype=float)


def load_external(name: str, label: str) -> pd.DataFrame:
    frame = pd.read_csv(COMPARE / name / "test_predictions.csv")
    centers = [np.asarray(json.loads(value), dtype=float)[int(idx)] for value, idx in zip(frame.candidate_xy_json, frame.top1_idx)]
    arr = np.asarray(centers)
    result = pd.DataFrame({
        "route": frame.route.str.lower().map({"route_b": "route_B", "route_c": "route_C"}),
        "frame_id": frame.frame_id.astype(int),
        f"{label}_x": arr[:, 0],
        f"{label}_y": arr[:, 1],
    })
    return result


def load_bearing() -> pd.DataFrame:
    frames = []
    for route, path in (
        ("route_B", BEARING / "newdata_eval_B_test" / "predictions.csv"),
        ("route_C", BEARING / "newdata_eval_C_short" / "predictions.csv"),
    ):
        # Bearing-UAV stores timestamp IDs while all other result files use
        # the ordinal acquisition index. Keep the ordering and reconstruct
        # that shared ordinal ID; the following inner join then retains the
        # exact frames present in the T2-only run.
        frame = pd.read_csv(path).sort_values("frame_id").reset_index(drop=True)
        frames.append(pd.DataFrame({
            "route": route,
            "frame_id": np.arange(len(frame), dtype=int),
            "Bearing-UAV_pred_px_x": frame.pred_pixel_x.to_numpy(float),
            "Bearing-UAV_pred_px_y": frame.pred_pixel_y.to_numpy(float),
            "Bearing-UAV_gt_px_x": frame.gt_pixel_x.to_numpy(float),
            "Bearing-UAV_gt_px_y": frame.gt_pixel_y.to_numpy(float),
            "Bearing-UAV_error_m": frame.error_m.to_numpy(float),
        }))
    return pd.concat(frames, ignore_index=True)


def load_all() -> pd.DataFrame:
    base = []
    for route in ("route_B", "route_C"):
        frame = pd.read_csv(RUN / f"{route}_robust_frames.csv")
        base.append(pd.DataFrame({
            "route": route,
            "frame_id": frame.frame_id.astype(int),
            "gt_x": frame.gt_x,
            "gt_y": frame.gt_y,
            "Raw Top-1_x": frame.raw_top1_x,
            "Raw Top-1_y": frame.raw_top1_y,
            "Fixed HardMS_x": frame.hardms_x,
            "Fixed HardMS_y": frame.hardms_y,
            "RTL-CRF (3 frames)_x": frame.temporal_x,
            "RTL-CRF (3 frames)_y": frame.temporal_y,
        }))
    merged = pd.concat(base, ignore_index=True)
    for folder, label in (
        ("sample4geo", "Sample4Geo"),
        ("denseuav", "DenseUAV"),
        ("game4loc", "Game4Loc"),
    ):
        merged = merged.merge(load_external(folder, label), on=["route", "frame_id"], how="inner")
    merged = merged.merge(load_bearing(), on=["route", "frame_id"], how="inner")
    # Bearing-UAV archived files use an independent pixel origin. Its reported
    # metre error is valid, but absolute pixel coordinates cannot be plotted on
    # this orthomosaic by a plain scale multiplication.  Convert the *per-frame
    # offset* from Bearing pixels to metres and add it to the shared GT XY.
    pred_px = merged[["Bearing-UAV_pred_px_x", "Bearing-UAV_pred_px_y"]].to_numpy(float)
    gt_px = merged[["Bearing-UAV_gt_px_x", "Bearing-UAV_gt_px_y"]].to_numpy(float)
    delta_px = pred_px - gt_px
    delta_norm = np.linalg.norm(delta_px, axis=1)
    scale = np.median(merged["Bearing-UAV_error_m"].to_numpy(float) / np.maximum(delta_norm, 1e-8))
    merged["Bearing-UAV_x"] = merged.gt_x + delta_px[:, 0] * scale
    merged["Bearing-UAV_y"] = merged.gt_y + delta_px[:, 1] * scale
    return merged.sort_values(["route", "frame_id"]).reset_index(drop=True)


def choose_segment(frame: pd.DataFrame, kind: str, length: int = 52) -> pd.DataFrame:
    """Choose a well-travelled low-turn or high-turn GT segment automatically."""
    xy = frame[["gt_x", "gt_y"]].to_numpy(float)
    best_score, best_start = None, 0
    for start in range(0, len(frame) - length + 1):
        part = xy[start:start + length]
        mid = length // 2
        first = part[mid] - part[0]
        second = part[-1] - part[mid]
        a, b = np.linalg.norm(first), np.linalg.norm(second)
        if min(a, b) < 5.0:
            continue
        cosine = np.clip(float(np.dot(first, second) / (a * b)), -1.0, 1.0)
        angle = math.degrees(math.acos(cosine))
        travel = float(np.linalg.norm(np.diff(part, axis=0), axis=1).sum())
        if kind == "straight":
            score = travel - 4.0 * angle
            valid = angle <= 18.0
        else:
            score = travel + 4.0 * angle
            valid = angle >= 30.0
        if valid and (best_score is None or score > best_score):
            best_score, best_start = score, start
    if best_score is None:
        raise RuntimeError(f"Could not find a {kind} segment in {frame.route.iloc[0]}")
    return frame.iloc[best_start:best_start + length].copy()


def crop_map(points_px: list[np.ndarray], image: Image.Image, pad: int = 120) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    all_points = np.concatenate(points_px, axis=0)
    left = max(0, int(np.floor(all_points[:, 0].min())) - pad)
    right = min(image.width, int(np.ceil(all_points[:, 0].max())) + pad)
    top = max(0, int(np.floor(all_points[:, 1].min())) - pad)
    bottom = min(image.height, int(np.ceil(all_points[:, 1].max())) + pad)
    return np.asarray(image.crop((left, top, right, bottom))), (left, right, bottom, top)


def draw_path(ax, bg: np.ndarray, extent: tuple[float, float, float, float], series: dict[str, np.ndarray], labels: list[str], title: str) -> None:
    ax.imshow(bg, extent=extent, origin="upper")
    gt = series["GT"]
    ax.plot(gt[:, 0], gt[:, 1], color=COLORS["GT"], linewidth=2.8, label="Ground truth", zorder=8)
    for label in labels:
        point = series[label]
        ax.plot(point[:, 0], point[:, 1], color=COLORS[label], linewidth=1.55, alpha=0.92, label=label, zorder=6)
        ax.scatter(point[::5, 0], point[::5, 1], color=COLORS[label], s=10, zorder=7)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92)


def render_segment(segment: pd.DataFrame, tag: str, mapper: SatGeoMapper, origin_lat: float, origin_lon: float, sat: Image.Image) -> dict:
    labels = ["Raw Top-1", "Fixed HardMS", "RTL-CRF (3 frames)", "Sample4Geo", "DenseUAV", "Game4Loc", "Bearing-UAV"]
    series = {"GT": xy_to_pixel(segment[["gt_x", "gt_y"]].to_numpy(), origin_lat, origin_lon, mapper)}
    for label in labels:
        series[label] = xy_to_pixel(segment[[f"{label}_x", f"{label}_y"]].to_numpy(), origin_lat, origin_lon, mapper)
    bg, extent = crop_map(list(series.values()), sat)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    draw_path(axes[0, 0], bg, extent, series, ["Raw Top-1", "Fixed HardMS", "RTL-CRF (3 frames)"], "Our visual and temporal predictions")
    draw_path(axes[0, 1], bg, extent, series, ["Sample4Geo", "DenseUAV", "Game4Loc", "Bearing-UAV"], "Published-method adaptations")
    index = np.arange(len(segment))
    gt_xy = segment[["gt_x", "gt_y"]].to_numpy()
    for label in ["Raw Top-1", "Fixed HardMS", "RTL-CRF (3 frames)"]:
        error = np.linalg.norm(segment[[f"{label}_x", f"{label}_y"]].to_numpy() - gt_xy, axis=1)
        axes[1, 0].plot(index, error, label=label, color=COLORS[label], linewidth=1.8)
    axes[1, 0].axhline(15, linestyle="--", color="#374151", linewidth=1, label="15 m")
    axes[1, 0].set(title="Our per-frame localization error", xlabel="Frame within displayed segment", ylabel="Error (m)")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.25)
    for label in ["Sample4Geo", "DenseUAV", "Game4Loc", "Bearing-UAV"]:
        error = np.linalg.norm(segment[[f"{label}_x", f"{label}_y"]].to_numpy() - gt_xy, axis=1)
        axes[1, 1].plot(index, error, label=label, color=COLORS[label], linewidth=1.55)
    axes[1, 1].axhline(15, linestyle="--", color="#374151", linewidth=1, label="15 m")
    axes[1, 1].set(title="External-method per-frame localization error", xlabel="Frame within displayed segment", ylabel="Error (m)")
    axes[1, 1].legend(fontsize=8, ncol=2); axes[1, 1].grid(alpha=0.25)
    route = segment.route.iloc[0]
    frame_start, frame_end = int(segment.frame_id.iloc[0]), int(segment.frame_id.iloc[-1])
    fig.suptitle(f"{tag.title()} segment: {route}, frames {frame_start}-{frame_end} (identical frames and map crop)", fontsize=14, weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{tag}_{route}_all_methods.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"segment": tag, "route": route, "first_frame": frame_start, "last_frame": frame_end, "frames": len(segment)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(RUN / "checkpoints" / "visual_retrieval_A_only.pt", map_location="cpu")
    mapper = SatGeoMapper(SAT_JSON, SAT_IMAGE)
    with Image.open(SAT_IMAGE) as image:
        sat = image.convert("RGB").copy()
    frame = load_all()
    # Route B provides a clear long straight portion; Route C provides a turn.
    straight = choose_segment(frame[frame.route.eq("route_B")].reset_index(drop=True), "straight")
    turn = choose_segment(frame[frame.route.eq("route_C")].reset_index(drop=True), "turn")
    meta = [
        render_segment(straight, "straight", mapper, float(checkpoint["origin_lat"]), float(checkpoint["origin_lon"]), sat),
        render_segment(turn, "turn", mapper, float(checkpoint["origin_lat"]), float(checkpoint["origin_lon"]), sat),
    ]
    (OUT / "README.md").write_text(
        "# Same-route all-method visual comparison\n\n"
        "The two figures use exactly the same frame IDs, ground truth, satellite crop, and coordinate system for every row. "
        "They are diagnostic visualizations of completed predictions; no model was retrained.\n\n"
        + "| Segment | Route | Frames | Frame IDs |\n| --- | --- | ---: | --- |\n"
        + "\n".join(f"| {x['segment']} | {x['route']} | {x['frames']} | {x['first_frame']}-{x['last_frame']} |" for x in meta)
        + "\n\nExternal rows are local-36 adaptations. Bearing-UAV is from archived predictions, not an official author-reported result. RTL-CRF uses three consecutive frames and prior-frame state.\n",
        encoding="utf-8",
    )
    (OUT / "segments.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
