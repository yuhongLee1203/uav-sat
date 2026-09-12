#!/usr/bin/env python3
"""Render synchronized UAV + SAT-map videos for the autonomous tracker."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from data import RouteDataset, meters_from_latlon

REFERENCE_COLOR = (0, 255, 0)
FINAL_COLOR = (255, 0, 255)
MS1_COLOR = (255, 255, 0)
KALMAN_COLOR = (0, 165, 255)
PRIOR_COLOR = (255, 128, 0)
WAYPOINT_COLOR = (0, 215, 255)


def meters_to_latlon(x_m, y_m, origin_lat, origin_lon):
    radius = 6378137.0
    lat = float(origin_lat) + math.degrees(float(y_m) / radius)
    lon = float(origin_lon) + math.degrees(
        float(x_m)
        / (
            radius
            * max(abs(math.cos(math.radians(float(origin_lat)))), 1e-12)
        )
    )
    return lat, lon


def xy_to_source_pixels(xy, dataset, origin_lat, origin_lon):
    rows = []
    for x_m, y_m in np.asarray(xy, dtype=np.float64):
        lat, lon = meters_to_latlon(x_m, y_m, origin_lat, origin_lon)
        px, py = dataset.mapper.latlon_to_pixel(lat, lon)
        rows.append([float(px), float(py)])
    return np.asarray(rows, dtype=np.float64)


def contain_image(image, width, height):
    src_h, src_w = image.shape[:2]
    scale = min(float(width) / src_w, float(height) / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(
        image,
        (dst_w, dst_h),
        interpolation=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    ox = (width - dst_w) // 2
    oy = (height - dst_h) // 2
    panel[oy : oy + dst_h, ox : ox + dst_w] = resized
    return panel


def load_waypoint_xy(route_name, origin_lat, origin_lon):
    payload = json.loads(Path(config.WAYPOINT_FILES[route_name]).read_text(encoding="utf-8"))
    items = sorted(payload["waypoints"], key=lambda x: int(x["waypoint_order"]))
    result = []
    for item in items:
        x_m, y_m = meters_from_latlon(
            item["latitude"], item["longitude"], origin_lat, origin_lon
        )
        result.append([float(x_m), float(y_m)])
    return np.asarray(result, dtype=np.float64)


def _required_columns(rows):
    required = {
        "frame_id", "image_path",
        "reference_x", "reference_y",
        "prior_x", "prior_y",
        "ms1_x", "ms1_y",
        "kalman_x_prime_x", "kalman_x_prime_y",
        "final_x", "final_y",
        "next_prior_x", "next_prior_y",
        "pred_speed_m_per_frame",
        "pred_acceleration_m_per_frame2",
        "pred_heading_deg",
        "delta_x", "delta_y",
        "error_final_m",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise RuntimeError(f"autonomous result CSV missing columns: {sorted(missing)}")


def render_route(route_name, start_frame=None, frame_count=None):
    csv_path = Path(config.OUTPUT_DIR) / (
        f"{route_name}_autonomous_ms1_kf_gru_ms2_frames.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path}\nRun: python robust_tracker.py --mode eval "
            f"--eval-routes {route_name}"
        )

    rows = pd.read_csv(csv_path)
    _required_columns(rows)
    if start_frame is not None:
        rows = rows[rows["frame_id"] >= int(start_frame)]
    if frame_count is not None:
        rows = rows.iloc[: int(frame_count)]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise RuntimeError("No rows remain after frame selection")

    route_index = config.ROUTE_NAMES.index(route_name)
    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    dataset = RouteDataset(
        Path(config.ROUTE_ROOTS[route_index]),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    with Image.open(config.SAT_IMAGE) as image:
        source_width, source_height = image.size
        map_height = int(config.VIDEO_HEIGHT)
        map_width = min(
            int(round(source_width * (float(map_height) / float(source_height)))),
            int(config.VIDEO_WIDTH * 0.63),
        )
        map_rgb = np.asarray(
            image.convert("RGB").resize((map_width, map_height), Image.Resampling.LANCZOS)
        )
    map_panel = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)

    width = int(config.VIDEO_WIDTH)
    height = int(config.VIDEO_HEIGHT)
    uav_width = width - map_width
    scale_x = float(map_width) / float(source_width)
    scale_y = float(map_height) / float(source_height)

    def xy_to_canvas(xy):
        source = xy_to_source_pixels(xy, dataset, origin_lat, origin_lon)
        result = np.empty_like(source)
        result[:, 0] = source[:, 0] * scale_x + uav_width
        result[:, 1] = source[:, 1] * scale_y
        return result

    reference = rows[["reference_x", "reference_y"]].to_numpy(float)
    final = rows[["final_x", "final_y"]].to_numpy(float)
    ms1 = rows[["ms1_x", "ms1_y"]].to_numpy(float)
    kalman = rows[["kalman_x_prime_x", "kalman_x_prime_y"]].to_numpy(float)
    prior = rows[["prior_x", "prior_y"]].to_numpy(float)
    reference_canvas = xy_to_canvas(reference)
    final_canvas = xy_to_canvas(final)
    ms1_canvas = xy_to_canvas(ms1)
    kalman_canvas = xy_to_canvas(kalman)
    prior_canvas = xy_to_canvas(prior)
    waypoint_canvas = xy_to_canvas(load_waypoint_xy(route_name, origin_lat, origin_lon))

    output_dir = Path(config.OUTPUT_DIR) / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if start_frame is not None or frame_count is not None:
        suffix = f"_frames_{int(rows.iloc[0].frame_id):04d}_{int(rows.iloc[-1].frame_id):04d}"
    output_path = output_dir / f"{route_name}{suffix}_autonomous_inference.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.VIDEO_FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {output_path}")

    try:
        for i, row in rows.iterrows():
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:, uav_width:] = map_panel
            uav_image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if uav_image is not None:
                canvas[:, :uav_width] = contain_image(uav_image, uav_width, height)

            if len(waypoint_canvas) >= 2:
                cv2.polylines(
                    canvas, [waypoint_canvas.astype(np.int32)], False,
                    WAYPOINT_COLOR, 1, cv2.LINE_AA
                )
            if i > 0:
                cv2.polylines(
                    canvas, [reference_canvas[: i + 1].astype(np.int32)], False,
                    REFERENCE_COLOR, 3, cv2.LINE_AA
                )
                cv2.polylines(
                    canvas, [final_canvas[: i + 1].astype(np.int32)], False,
                    FINAL_COLOR, 3, cv2.LINE_AA
                )

            for point, color, marker in [
                (prior_canvas[i], PRIOR_COLOR, cv2.MARKER_DIAMOND),
                (ms1_canvas[i], MS1_COLOR, cv2.MARKER_SQUARE),
                (kalman_canvas[i], KALMAN_COLOR, cv2.MARKER_TILTED_CROSS),
                (reference_canvas[i], REFERENCE_COLOR, cv2.MARKER_CROSS),
                (final_canvas[i], FINAL_COLOR, cv2.MARKER_STAR),
            ]:
                cv2.drawMarker(
                    canvas, (int(point[0]), int(point[1])), color,
                    marker, 18, 2, cv2.LINE_AA
                )

            heading = math.radians(float(row["pred_heading_deg"]))
            arrow_len_m = max(8.0, abs(float(row["pred_speed_m_per_frame"])) * 4.0)
            arrow_xy = np.asarray([[
                float(row["final_x"]) + arrow_len_m * math.cos(heading),
                float(row["final_y"]) + arrow_len_m * math.sin(heading),
            ]])
            arrow_point = xy_to_canvas(arrow_xy)[0]
            final_point = final_canvas[i]
            cv2.arrowedLine(
                canvas,
                (int(final_point[0]), int(final_point[1])),
                (int(arrow_point[0]), int(arrow_point[1])),
                FINAL_COLOR, 2, cv2.LINE_AA, tipLength=0.25,
            )

            labels = [
                "AUTONOMOUS: PRIOR -> MS#1 3x6 -> [KALMAN || GRU] -> MS#2 6x6 -> FINAL",
                "Reference=GREEN  Prior=BLUE  MS#1=CYAN  Kalman X'=ORANGE  Final=MAGENTA",
                f"frame={int(row['frame_id'])}  error={float(row['error_final_m']):.2f} m",
                (
                    f"v={float(row['pred_speed_m_per_frame']):.3f} m/frame  "
                    f"a={float(row['pred_acceleration_m_per_frame2']):.3f} m/frame^2  "
                    f"heading={float(row['pred_heading_deg']):.2f} deg"
                ),
                (
                    f"Delta=({float(row['delta_x']):.3f}, {float(row['delta_y']):.3f})  "
                    f"next prior=({float(row['next_prior_x']):.2f}, {float(row['next_prior_y']):.2f})"
                ),
                "No current-frame reference/GT is used to select the runtime search center.",
            ]
            for j, text in enumerate(labels):
                cv2.putText(
                    canvas, text, (18, 30 + j * 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
            writer.write(canvas)
    finally:
        writer.release()

    print(f"[OK] video: {output_path}", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routes", nargs="+", choices=config.ROUTE_NAMES, default=["route_C", "route_B"]
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    args = parser.parse_args()
    for route_name in args.routes:
        render_route(route_name, args.start_frame, args.frame_count)


if __name__ == "__main__":
    main()
