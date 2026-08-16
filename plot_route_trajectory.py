#!/usr/bin/env python3
"""Create a full-route and a zoomed SAT-map trajectory PNG from tracker CSVs."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from data import RouteDataset
from render_results_video import load_waypoint_pixels, xy_to_source_pixels


GT_COLOR = (50, 220, 50)       # BGR
PRED_COLOR = (255, 60, 255)
WAYPOINT_COLOR = (0, 215, 255)
JUMP_COLOR = (0, 60, 255)


def draw_polyline(image, points, color, thickness=3):
    points = np.rint(points).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def annotate(image, lines):
    box_height = 20 + 30 * len(lines)
    cv2.rectangle(image, (12, 12), (670, box_height), (15, 15, 15), -1)
    for index, text in enumerate(lines):
        cv2.putText(image, text, (24, 42 + index * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default="route_B", choices=["route_B", "route_C"])
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--start-frame", type=int, default=None)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    csv_path = config.OUTPUT_DIR / (
        args.route + "_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = pd.read_csv(csv_path)
    route_index = config.ROUTE_NAMES.index(args.route)
    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    dataset = RouteDataset(Path(config.ROUTE_ROOTS[route_index]), train=False, origin_lat=origin_lat, origin_lon=origin_lon)

    gt_px = xy_to_source_pixels(rows[["gt_x", "gt_y"]].to_numpy(float), dataset, origin_lat, origin_lon)
    pred_px = xy_to_source_pixels(rows[["final_x", "final_y"]].to_numpy(float), dataset, origin_lat, origin_lon)
    waypoint_px = xy_to_source_pixels(load_waypoint_pixels(args.route, dataset, origin_lat, origin_lon), dataset, origin_lat, origin_lon)

    with Image.open(config.SAT_IMAGE) as sat:
        map_bgr = cv2.cvtColor(np.asarray(sat.convert("RGB")), cv2.COLOR_RGB2BGR)

    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Complete-route overview, resized only for a readable PNG.
    scale = min(1.0, 2400.0 / max(map_bgr.shape[:2]))
    overview = cv2.resize(map_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else map_bgr.copy()
    draw_polyline(overview, gt_px * scale, GT_COLOR, 3)
    draw_polyline(overview, pred_px * scale, PRED_COLOR, 3)
    for point in waypoint_px * scale:
        cv2.circle(overview, tuple(np.rint(point).astype(int)), 5, WAYPOINT_COLOR, -1, cv2.LINE_AA)
    jump_rows = rows[rows["abnormal_jump"].astype(int) == 1]
    if not jump_rows.empty:
        jump_px = xy_to_source_pixels(jump_rows[["final_x", "final_y"]].to_numpy(float), dataset, origin_lat, origin_lon)
        for point in jump_px * scale:
            cv2.circle(overview, tuple(np.rint(point).astype(int)), 9, JUMP_COLOR, 2, cv2.LINE_AA)
    annotate(overview, [
        f"{args.route}: complete trajectory ({len(rows)} frames)",
        "Green = GT trajectory   Magenta = final Kalman trajectory",
        "Yellow = waypoint   Red circle = abnormal jump (> GT step + 5m)",
    ])
    full_path = output_dir / (args.route + "_full_trajectory.png")
    cv2.imwrite(str(full_path), overview)

    # Default zoom is centred on the largest excess-step event, i.e. the most
    # useful local segment for checking a visible jump.
    if args.start_frame is None:
        center_frame = int(rows.loc[rows["excess_step_over_gt_m"].idxmax(), "frame_id"])
        start_frame = max(int(rows["frame_id"].min()), center_frame - args.frames // 2)
    else:
        start_frame = int(args.start_frame)
    segment = rows[rows["frame_id"] >= start_frame].iloc[: args.frames].copy()
    if segment.empty:
        raise RuntimeError("No frames selected")
    first_frame, last_frame = int(segment.iloc[0].frame_id), int(segment.iloc[-1].frame_id)
    selected = rows.index.isin(segment.index)
    segment_gt = gt_px[selected]
    segment_pred = pred_px[selected]
    all_points = np.vstack([segment_gt, segment_pred])
    min_xy = np.floor(all_points.min(axis=0)).astype(int)
    max_xy = np.ceil(all_points.max(axis=0)).astype(int)
    # At least 700 source pixels around the selected trajectory: a true zoom,
    # but wide enough to retain road/map context.
    margin = max(350, int(0.65 * max((max_xy - min_xy).max(), 1)))
    x0, y0 = np.maximum(min_xy - margin, 0)
    x1, y1 = np.minimum(max_xy + margin, [map_bgr.shape[1] - 1, map_bgr.shape[0] - 1])
    crop = map_bgr[y0:y1 + 1, x0:x1 + 1].copy()
    local_gt = segment_gt - np.array([x0, y0])
    local_pred = segment_pred - np.array([x0, y0])
    draw_polyline(crop, local_gt, GT_COLOR, 5)
    draw_polyline(crop, local_pred, PRED_COLOR, 5)
    cv2.circle(crop, tuple(np.rint(local_gt[0]).astype(int)), 9, GT_COLOR, -1, cv2.LINE_AA)
    cv2.circle(crop, tuple(np.rint(local_gt[-1]).astype(int)), 10, GT_COLOR, 2, cv2.LINE_AA)
    for (_, row), point in zip(segment.iterrows(), local_pred):
        if int(row.abnormal_jump):
            cv2.circle(crop, tuple(np.rint(point).astype(int)), 15, JUMP_COLOR, 3, cv2.LINE_AA)
    zoom_scale = min(2.0, 1800.0 / max(crop.shape[:2]))
    if zoom_scale > 1.0:
        crop = cv2.resize(crop, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_CUBIC)
    annotate(crop, [
        f"{args.route}: zoomed trajectory, frames {first_frame}-{last_frame}",
        "Green = GT   Magenta = final Kalman position",
        "Red circle = abnormal jump (> GT step + 5m)",
    ])
    zoom_path = output_dir / (args.route + f"_zoom_frames_{first_frame:04d}_{last_frame:04d}.png")
    cv2.imwrite(str(zoom_path), crop)
    print(full_path)
    print(zoom_path)


if __name__ == "__main__":
    main()
