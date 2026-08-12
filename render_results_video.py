import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from data import RouteDataset


GT_COLOR = (0, 255, 0)
PRED_COLOR = (255, 0, 255)
WAYPOINT_COLOR = (0, 215, 255)


def meters_to_latlon(x_m, y_m, origin_lat, origin_lon):
    radius = 6378137.0
    lat = float(origin_lat) + math.degrees(float(y_m) / radius)
    lon_scale = radius * math.cos(math.radians(float(origin_lat)))
    lon = float(origin_lon) + math.degrees(float(x_m) / lon_scale)
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
    scale = min(float(width) / float(src_w), float(height) / float(src_h))
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


def load_waypoint_pixels(route_name, dataset, origin_lat, origin_lon):
    import json
    from data import meters_from_latlon

    path = Path(config.WAYPOINT_FILES[route_name])
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = sorted(
        payload["waypoints"], key=lambda item: int(item["waypoint_order"])
    )
    xy = []
    for item in items:
        x_m, y_m = meters_from_latlon(
            item["latitude"], item["longitude"], origin_lat, origin_lon
        )
        xy.append([float(x_m), float(y_m)])
    return np.asarray(xy, dtype=np.float64)


def render_video(route_name, rows, dataset, origin_lat, origin_lon, output_dir):
    with Image.open(config.SAT_IMAGE) as image:
        source_width, source_height = image.size
        map_height = int(config.VIDEO_HEIGHT)
        map_width = min(
            int(round(source_width * (float(map_height) / float(source_height)))),
            int(config.VIDEO_WIDTH * 0.63),
        )
        map_rgb = np.asarray(
            image.convert("RGB").resize(
                (map_width, map_height), Image.Resampling.LANCZOS
            )
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

    gt_xy = rows[["gt_x", "gt_y"]].to_numpy(dtype=float)
    pred_xy = rows[["final_x", "final_y"]].to_numpy(dtype=float)
    waypoint_xy = load_waypoint_pixels(route_name, dataset, origin_lat, origin_lon)
    gt_canvas = xy_to_canvas(gt_xy)
    pred_canvas = xy_to_canvas(pred_xy)
    waypoint_canvas = xy_to_canvas(waypoint_xy)

    output_path = output_dir / (route_name + "_synchronized_inference.mp4")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.VIDEO_FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot create video: %s" % output_path)

    try:
        for row_index, row in rows.iterrows():
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:, uav_width:] = map_panel

            uav_image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if uav_image is not None:
                canvas[:, :uav_width] = contain_image(
                    uav_image, uav_width, height
                )

            if len(waypoint_canvas) >= 2:
                cv2.polylines(
                    canvas,
                    [waypoint_canvas.astype(np.int32)],
                    False,
                    WAYPOINT_COLOR,
                    1,
                    cv2.LINE_AA,
                )
            for waypoint_index, point in enumerate(waypoint_canvas):
                cv2.circle(
                    canvas,
                    (int(point[0]), int(point[1])),
                    4,
                    WAYPOINT_COLOR,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "W%d" % waypoint_index,
                    (int(point[0]) + 5, int(point[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    WAYPOINT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            if row_index > 0:
                cv2.polylines(
                    canvas,
                    [gt_canvas[: row_index + 1].astype(np.int32)],
                    False,
                    GT_COLOR,
                    3,
                    cv2.LINE_AA,
                )
                cv2.polylines(
                    canvas,
                    [pred_canvas[: row_index + 1].astype(np.int32)],
                    False,
                    PRED_COLOR,
                    3,
                    cv2.LINE_AA,
                )

            gt_point = gt_canvas[row_index]
            pred_point = pred_canvas[row_index]
            cv2.drawMarker(
                canvas,
                (int(gt_point[0]), int(gt_point[1])),
                GT_COLOR,
                cv2.MARKER_CROSS,
                20,
                3,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                canvas,
                (int(pred_point[0]), int(pred_point[1])),
                PRED_COLOR,
                cv2.MARKER_STAR,
                22,
                3,
                cv2.LINE_AA,
            )

            # Estimated ground-track heading is part of the model state and is
            # used by the polynomial predictor. Draw it from the final position.
            if "estimated_heading_deg" in rows.columns:
                heading_rad = math.radians(float(row.get("polynomial_heading_deg", row.get("estimated_heading_deg", 0.0))))
                arrow_len_m = 14.0
                arrow_xy = np.asarray([[
                    float(row["final_x"]) + arrow_len_m * math.cos(heading_rad),
                    float(row["final_y"]) + arrow_len_m * math.sin(heading_rad),
                ]], dtype=np.float64)
                arrow_canvas = xy_to_canvas(arrow_xy)[0]
                cv2.arrowedLine(
                    canvas,
                    (int(pred_point[0]), int(pred_point[1])),
                    (int(arrow_canvas[0]), int(arrow_canvas[1])),
                    PRED_COLOR,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

            lines = [
                "CONTROLLED GT+SMOOTH-JITTER / HEADING-AWARE POLYNOMIAL / KALMAN FINAL",
                "GT = GREEN    FINAL CONSTRAINED KALMAN = MAGENTA    WAYPOINT = YELLOW",
                "frame=%d  target=W%d  pred_leg=%d  gt_leg=%d"
                % (
                    int(row["frame_id"]),
                    int(row["target_waypoint"]),
                    int(row["waypoint_leg"]),
                    int(row.get("gt_waypoint_leg", -1)),
                ),
                "v=%.2f gt_v=%.2f  step=%.2f gt_step=%.2f  speed_err=%.2f"
                % (
                    float(row["v_parallel"]),
                    float(row.get("gt_velocity_parallel", 0.0)),
                    float(row["poly_next_step_parallel"]),
                    float(row.get("gt_step_parallel", 0.0)),
                    float(row.get("speed_error_m_per_frame", 0.0)),
                ),
                "causal heading=%.1fdeg gt=%.1fdeg turn=%.1fdeg/f err=%.1fdeg"
                % (
                    float(row.get("estimated_heading_deg", 0.0)),
                    float(row.get("gt_heading_deg", 0.0)),
                    float(row.get("turn_rate_deg_per_frame", 0.0)),
                    float(row.get("heading_error_deg", 0.0)),
                ),
                "progress=%.1f gt=%.1f progress_err=%.1f  selected=%d bank=%d"
                % (
                    float(row["final_progress_s"]),
                    float(row.get("gt_progress_s", 0.0)),
                    float(row.get("progress_error_m", 0.0)),
                    int(row.get("selected_candidate_capture", row.get("candidate_capture", 0))),
                    int(row.get("bank_candidate_capture", 0)),
                ),
                "visual_conf=%.3f step_limit=%.2fm limited=%d  selected_s=%.1f"
                % (
                    float(row.get("local_visual_confidence", row.get("acquisition_confidence", 0.0))),
                    float(row.get("kalman_step_limit_m", 0.0)),
                    int(row.get("kalman_step_limited", 0)),
                    float(row.get("selected_hypothesis_center_s", 0.0)),
                ),
                "final_step=%.2fm err=%.2fm H=%.2f margin=%.4f Rscale=%.2f"
                % (
                    float(row["final_step_m"]),
                    float(row["error_final_m"]),
                    float(row.get("visual_entropy", 0.0)),
                    float(row.get("visual_margin", 0.0)),
                    float(row.get("kalman_r_scale", 1.0)),
                ),
            ]
            for line_index, text in enumerate(lines):
                cv2.putText(
                    canvas,
                    text,
                    (18, 30 + line_index * 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
    finally:
        writer.release()
    return output_path


def render_route(route_name):
    csv_path = config.OUTPUT_DIR / (
        route_name + "_controlled_gtprior_causal_heading_rnn_polynomial_kalman_frames.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            "Inference CSV missing: %s\nRun: bash run_robust_tracker.sh --mode eval"
            % csv_path
        )
    rows = pd.read_csv(csv_path)
    if rows.empty:
        raise RuntimeError("CSV is empty: %s" % csv_path)

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
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video = render_video(
        route_name=route_name,
        rows=rows,
        dataset=dataset,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        output_dir=config.OUTPUT_DIR,
    )
    print("rendered:", video, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route", choices=["route_B", "route_C", "all"], default="all"
    )
    args = parser.parse_args()
    if args.route == "all":
        for route_name in ["route_B", "route_C"]:
            render_route(route_name)
    else:
        render_route(args.route)


if __name__ == "__main__":
    main()
