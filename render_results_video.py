"""Render clean, aspect-ratio-preserving route videos for temporal diagnostics.

The layout deliberately follows the earlier fineXY visualisation convention:
the full orthomosaic is shown on the right and the current UAV frame on the
left.  Neither image is stretched.  Each frame is an independent visual
measurement; the traces simply reveal how predictions evolve in acquisition
order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from robust_tracker import motion_nodes


VIDEO_WIDTH = 1600
VIDEO_HEIGHT = 900
FPS = 8

# BGR palette chosen to remain legible over both fields and roads.
GT_COLOR = (50, 220, 70)
VISUAL_COLOR = (45, 80, 235)
ROBUST_COLOR = (235, 185, 35)
TEXT_COLOR = (242, 242, 242)


def fit_xy_to_pixel(dataset):
    xy = np.asarray([[s["x_meter"], s["y_meter"], 1.0] for s in dataset.samples])
    pixel = np.asarray([[s["pixel_x"], s["pixel_y"]] for s in dataset.samples])
    return np.linalg.lstsq(xy, pixel, rcond=None)[0]


def transform(xy, affine):
    xy = np.asarray(xy, dtype=np.float64)
    return np.c_[xy, np.ones(len(xy))] @ affine


def contain_image(image, width, height, fill=(0, 0, 0)):
    """Letterbox an image into a panel while retaining its aspect ratio."""
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=interp)
    panel = np.full((height, width, 3), fill, dtype=np.uint8)
    x = (width - dst_w) // 2
    y = (height - dst_h) // 2
    panel[y : y + dst_h, x : x + dst_w] = resized
    return panel, x, y, dst_w, dst_h


def add_top_bar(canvas, text):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 43), (0, 0, 0), -1)
    cv2.putText(canvas, text, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_trace(canvas, points, color, panel_x, panel_y, panel_w, panel_h, thickness):
    """Draw only valid contiguous segments so off-map drift cannot make a line mess."""
    if len(points) < 2:
        return
    last = None
    max_step = 0.18 * float(np.hypot(panel_w, panel_h))
    for point in points:
        x, y = int(point[0]), int(point[1])
        valid = panel_x <= x < panel_x + panel_w and panel_y <= y < panel_y + panel_h
        if valid and last is not None:
            if float(np.hypot(x - last[0], y - last[1])) <= max_step:
                cv2.line(canvas, last, (x, y), color, thickness, cv2.LINE_AA)
        last = (x, y) if valid else None


def draw_marker(canvas, point, color, kind):
    x, y = int(point[0]), int(point[1])
    if kind == "cross":
        cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, 15, 2, cv2.LINE_AA)
    elif kind == "diamond":
        cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_DIAMOND, 15, 2, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)


def in_panel(point, x, y, w, h):
    return x <= point[0] < x + w and y <= point[1] < y + h


def render_route(root, name, out_dir):
    result_csv = config.OUTPUT_DIR / f"{name}_robust_frames.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Missing {result_csv}; run robust_tracker.py first.")
    rows = pd.read_csv(result_csv)

    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    dataset, _ = motion_nodes(root, float(checkpoint["origin_lat"]), float(checkpoint["origin_lon"]))
    by_id = {str(s["frame_id"]): s for s in dataset.samples}
    affine = fit_xy_to_pixel(dataset)
    gt_xy = rows[["gt_x", "gt_y"]].to_numpy()
    visual_xy = rows[["visual_x", "visual_y"]].to_numpy()
    robust_xy = rows[["robust_x", "robust_y"]].to_numpy()
    gt_px, visual_px, robust_px = (transform(value, affine) for value in (gt_xy, visual_xy, robust_xy))

    # The full orthomosaic defines the map panel, exactly as in the reference video.
    with Image.open(config.SAT_IMAGE) as image:
        image = image.convert("RGB")
        map_width = int(round(image.width * (VIDEO_HEIGHT / image.height)))
        map_rgb = np.asarray(image.resize((map_width, VIDEO_HEIGHT), Image.Resampling.LANCZOS))
        source_w, source_h = image.width, image.height
    map_panel = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
    uav_width = VIDEO_WIDTH - map_width
    if uav_width < 480:
        raise RuntimeError("Satellite map is too wide for the selected video dimensions.")

    # Map source pixels directly into the full-map panel; no route crop / square warp.
    gt_panel = np.c_[gt_px[:, 0] * map_width / source_w + uav_width, gt_px[:, 1] * VIDEO_HEIGHT / source_h].astype(np.int32)
    visual_panel = np.c_[visual_px[:, 0] * map_width / source_w + uav_width, visual_px[:, 1] * VIDEO_HEIGHT / source_h].astype(np.int32)
    robust_panel = np.c_[robust_px[:, 0] * map_width / source_w + uav_width, robust_px[:, 1] * VIDEO_HEIGHT / source_h].astype(np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_robust_tracker_reference_style.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    for i, row in rows.iterrows():
        canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        canvas[:, uav_width:] = map_panel
        sample = by_id.get(str(int(row.frame_id)))
        if sample is not None:
            image = cv2.imread(sample["image_path"])
            if image is not None:
                uav_panel, _, _, _, _ = contain_image(image, uav_width, VIDEO_HEIGHT, fill=(0, 0, 0))
                canvas[:, :uav_width] = uav_panel

        draw_trace(canvas, gt_panel[: i + 1], GT_COLOR, uav_width, 0, map_width, VIDEO_HEIGHT, 2)
        draw_trace(canvas, visual_panel[: i + 1], VISUAL_COLOR, uav_width, 0, map_width, VIDEO_HEIGHT, 1)
        draw_trace(canvas, robust_panel[: i + 1], ROBUST_COLOR, uav_width, 0, map_width, VIDEO_HEIGHT, 2)

        for point, color, kind in ((gt_panel[i], GT_COLOR, "dot"), (visual_panel[i], VISUAL_COLOR, "cross"), (robust_panel[i], ROBUST_COLOR, "diamond")):
            if in_panel(point, uav_width, 0, map_width, VIDEO_HEIGHT):
                draw_marker(canvas, point, color, kind)

        visual_error = float(np.linalg.norm(visual_xy[i] - gt_xy[i]))
        robust_error = float(np.linalg.norm(robust_xy[i] - gt_xy[i]))
        accepted = "accepted" if row.visual_update_weight > 0 else "rejected"
        add_top_bar(canvas, f"{name.upper()}   frame {int(row.frame_id)}   visual {visual_error:.1f} m   temporal {robust_error:.1f} m   update {accepted}")
        cv2.rectangle(canvas, (uav_width + 10, VIDEO_HEIGHT - 75), (VIDEO_WIDTH - 10, VIDEO_HEIGHT - 10), (0, 0, 0), -1)
        cv2.putText(canvas, "GT", (uav_width + 24, VIDEO_HEIGHT - 48), cv2.FONT_HERSHEY_SIMPLEX, .46, GT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Raw HardMS", (uav_width + 82, VIDEO_HEIGHT - 48), cv2.FONT_HERSHEY_SIMPLEX, .46, VISUAL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Temporal", (uav_width + 205, VIDEO_HEIGHT - 48), cv2.FONT_HERSHEY_SIMPLEX, .46, ROBUST_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Trajectories show independent frame predictions in acquisition order.", (uav_width + 24, VIDEO_HEIGHT - 26), cv2.FONT_HERSHEY_SIMPLEX, .39, TEXT_COLOR, 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=[*config.ROUTE_NAMES, "all"], default="all")
    args = parser.parse_args()
    output = config.OUTPUT_DIR / "videos_reference_style"
    for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES):
        if args.route in ("all", name):
            print(render_route(root, name, output))


if __name__ == "__main__":
    main()
