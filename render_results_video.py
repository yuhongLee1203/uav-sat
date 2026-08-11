import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from data import RouteDataset, meters_from_latlon


def load_waypoints(route_name, origin_lat, origin_lon):
    payload = json.loads(
        Path(config.WAYPOINT_FILES[route_name]).read_text(encoding="utf-8")
    )
    raw = sorted(payload["waypoints"], key=lambda item: int(item["waypoint_order"]))
    rows = []
    for item in raw:
        x_m, y_m = meters_from_latlon(
            item["latitude"],
            item["longitude"],
            origin_lat,
            origin_lon,
        )
        rows.append(
            {
                "order": int(item["waypoint_order"]),
                "x": float(x_m),
                "y": float(y_m),
            }
        )
    return rows


def meters_to_latlon(x_m, y_m, origin_lat, origin_lon):
    radius = 6378137.0
    lat = float(origin_lat) + math.degrees(float(y_m) / radius)
    lon_scale = radius * math.cos(math.radians(float(origin_lat)))
    lon = float(origin_lon) + math.degrees(float(x_m) / lon_scale)
    return lat, lon


def xy_to_source_pixels(xy, dataset, origin_lat, origin_lon):
    rows = []
    for x_m, y_m in np.asarray(xy, dtype=np.float64):
        lat, lon = meters_to_latlon(
            x_m,
            y_m,
            origin_lat,
            origin_lon,
        )
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


def render_overview(route_name, rows, waypoints, output_dir):
    gt = rows[["gt_x", "gt_y"]].to_numpy(dtype=float)
    visual = rows[["visual_x", "visual_y"]].to_numpy(dtype=float)
    final = rows[["final_x", "final_y"]].to_numpy(dtype=float)

    route_xy = np.asarray(
        [[wp["x"], wp["y"]] for wp in waypoints],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(
        route_xy[:, 0],
        route_xy[:, 1],
        "--",
        linewidth=1.2,
        label="Waypoint route prior",
    )
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        linewidth=2.0,
        label="GT (evaluation only)",
    )
    ax.plot(
        visual[:, 0],
        visual[:, 1],
        linewidth=1.5,
        label="RNN visual progress",
    )
    ax.plot(
        final[:, 0],
        final[:, 1],
        linewidth=1.8,
        label="progress-only Kalman",
    )

    for wp in waypoints:
        ax.scatter([wp["x"]], [wp["y"]], marker="X", s=80)
        ax.annotate(
            "W%d" % wp["order"],
            (wp["x"], wp["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(
        "%s: Continuous-Progress Visual RNN v11\n"
        "RNNCell + 0..3m/frame + forward 3x6 + route progress"
        % route_name
    )
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_dir / (route_name + "_overview_frames.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def render_process(route_name, rows, output_dir):
    count = min(int(config.PROCESS_SNAPSHOT_COUNT), len(rows))
    indices = np.unique(
        np.linspace(0, len(rows) - 1, count).round().astype(int)
    )

    columns = 3
    row_count = int(math.ceil(len(indices) / float(columns)))
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(16, 4.8 * row_count),
    )
    axes = np.asarray(axes).reshape(-1)

    for plot_index, data_index in enumerate(indices):
        row = rows.iloc[data_index]
        ax = axes[plot_index]

        points = {
            "GT": (row["gt_x"], row["gt_y"], "o"),
            "Selected SAT": (
                row["selected_patch_x"],
                row["selected_patch_y"],
                "D",
            ),
            "RNN XY": (row["visual_x"], row["visual_y"], "P"),
            "Kalman XY": (row["final_x"], row["final_y"], "*"),
        }

        for label, (x, y, marker) in points.items():
            ax.scatter(
                [float(x)],
                [float(y)],
                marker=marker,
                s=95 if label == "Kalman XY" else 55,
                label=label,
            )

        xy = np.asarray(
            [[float(value[0]), float(value[1])] for value in points.values()]
        )
        center = xy.mean(axis=0)
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 30.0)
        margin = 0.75 * span

        ax.set_xlim(center[0] - margin, center[0] + margin)
        ax.set_ylim(center[1] - margin, center[1] + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            "f%d W%d->W%d\n"
            "step=%.2fm gate=%.2f visualCap=%.2f inertiaCap=%.2f\n"
            "heading(route/est)=%.1f/%.1f deg pmax=%.2f err=%.1fm"
            % (
                int(row["frame_id"]),
                int(row["active_waypoint_from"]),
                int(row["active_waypoint_to"]),
                float(row["predicted_step_m"]),
                float(row["move_gate"]),
                float(row["visual_step_cap_m"]),
                float(row["inertia_cap_m"]),
                float(row["route_heading_deg"]),
                float(row["estimated_heading_deg"]),
                float(row["candidate_probability_max"]),
                float(row["error_final_m"]),
            ),
            fontsize=8,
        )
        if plot_index == 0:
            ax.legend(fontsize=6)

    for index in range(len(indices), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        "%s: v11 continuous route progress" % route_name,
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / (route_name + "_process_frames.png")
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def render_video(
    route_name,
    rows,
    waypoints,
    dataset,
    origin_lat,
    origin_lon,
    output_dir,
):
    with Image.open(config.SAT_IMAGE) as image:
        source_width, source_height = image.size
        map_height = int(config.VIDEO_HEIGHT)
        map_width = min(
            int(round(source_width * (float(map_height) / float(source_height)))),
            int(config.VIDEO_WIDTH * 0.63),
        )
        map_rgb = np.asarray(
            image.convert("RGB").resize(
                (map_width, map_height),
                Image.Resampling.LANCZOS,
            )
        )

    map_panel = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
    width = int(config.VIDEO_WIDTH)
    height = int(config.VIDEO_HEIGHT)
    uav_width = width - map_width
    scale_x = float(map_width) / float(source_width)
    scale_y = float(map_height) / float(source_height)

    def xy_to_canvas(xy):
        source = xy_to_source_pixels(
            xy,
            dataset,
            origin_lat,
            origin_lon,
        )
        result = np.empty_like(source)
        result[:, 0] = source[:, 0] * scale_x + uav_width
        result[:, 1] = source[:, 1] * scale_y
        return result

    gt_canvas = xy_to_canvas(
        rows[["gt_x", "gt_y"]].to_numpy(dtype=float)
    )
    final_canvas = xy_to_canvas(
        rows[["final_x", "final_y"]].to_numpy(dtype=float)
    )
    selected_canvas = xy_to_canvas(
        rows[["selected_patch_x", "selected_patch_y"]].to_numpy(dtype=float)
    )
    waypoint_canvas = xy_to_canvas(
        np.asarray([[wp["x"], wp["y"]] for wp in waypoints], dtype=float)
    )
    route_canvas = waypoint_canvas.copy()

    path = output_dir / (route_name + "_synchronized_inference.mp4")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.VIDEO_FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot create video: %s" % path)

    try:
        for row_index, row in rows.iterrows():
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:, uav_width:] = map_panel

            image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if image is not None:
                canvas[:, :uav_width] = contain_image(
                    image,
                    uav_width,
                    height,
                )

            if len(route_canvas) > 1:
                cv2.polylines(
                    canvas,
                    [route_canvas.astype(np.int32)],
                    False,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            for wp_index, point in enumerate(waypoint_canvas):
                cv2.drawMarker(
                    canvas,
                    (int(point[0]), int(point[1])),
                    (0, 220, 255),
                    cv2.MARKER_TILTED_CROSS,
                    13,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "W%d" % waypoints[wp_index]["order"],
                    (int(point[0]) + 4, int(point[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            if row_index > 0:
                cv2.polylines(
                    canvas,
                    [gt_canvas[: row_index + 1].astype(np.int32)],
                    False,
                    (90, 220, 90),
                    2,
                    cv2.LINE_AA,
                )
                cv2.polylines(
                    canvas,
                    [final_canvas[: row_index + 1].astype(np.int32)],
                    False,
                    (0, 140, 255),
                    2,
                    cv2.LINE_AA,
                )

            gt_point = gt_canvas[row_index]
            pred_point = final_canvas[row_index]
            selected_point = selected_canvas[row_index]

            cv2.drawMarker(
                canvas,
                (int(gt_point[0]), int(gt_point[1])),
                (90, 255, 90),
                cv2.MARKER_CROSS,
                17,
                2,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                canvas,
                (int(pred_point[0]), int(pred_point[1])),
                (0, 140, 255),
                cv2.MARKER_STAR,
                19,
                2,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                canvas,
                (int(selected_point[0]), int(selected_point[1])),
                (255, 220, 0),
                cv2.MARKER_DIAMOND,
                13,
                1,
                cv2.LINE_AA,
            )

            heading_rad = math.radians(float(row["estimated_heading_deg"]))
            arrow_length = 38.0
            arrow_end = (
                int(pred_point[0] + arrow_length * math.cos(heading_rad)),
                int(pred_point[1] - arrow_length * math.sin(heading_rad)),
            )
            cv2.arrowedLine(
                canvas,
                (int(pred_point[0]), int(pred_point[1])),
                arrow_end,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

            text_rows = [
                "frame=%d  W%d->W%d"
                % (
                    int(row["frame_id"]),
                    int(row["active_waypoint_from"]),
                    int(row["active_waypoint_to"]),
                ),
                "step=%.2f m  gate=%.2f  max=%.1f m/frame"
                % (
                    float(row["predicted_step_m"]),
                    float(row["move_gate"]),
                    float(config.MAX_STEP_M_PER_FRAME),
                ),
                "heading route=%.1f deg  est=%.1f deg"
                % (
                    float(row["route_heading_deg"]),
                    float(row["estimated_heading_deg"]),
                ),
                "18-patch pmax=%.2f margin=%.2f"
                % (
                    float(row["candidate_probability_max"]),
                    float(row["candidate_probability_margin"]),
                ),
                "error=%.2f m" % float(row["error_final_m"]),
            ]

            for line_index, text in enumerate(text_rows):
                cv2.putText(
                    canvas,
                    text,
                    (18, 30 + line_index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(canvas)
    finally:
        writer.release()

    return path


def render_route(route_name):
    csv_path = (
        config.OUTPUT_DIR
        / (route_name + "_continuous_progress_rnn_frames.csv")
    )
    if not csv_path.exists():
        raise FileNotFoundError("Inference CSV missing: %s" % csv_path)

    rows = pd.read_csv(csv_path)
    if rows.empty:
        raise RuntimeError("CSV is empty: %s" % csv_path)

    route_index = config.ROUTE_NAMES.index(route_name)

    checkpoint = torch.load(
        config.VISUAL_CHECKPOINT,
        map_location="cpu",
    )
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])

    dataset = RouteDataset(
        Path(config.ROUTE_ROOTS[route_index]),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    waypoints = load_waypoints(
        route_name,
        origin_lat,
        origin_lon,
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overview = render_overview(
        route_name,
        rows,
        waypoints,
        config.OUTPUT_DIR,
    )
    process = render_process(
        route_name,
        rows,
        config.OUTPUT_DIR,
    )
    video = render_video(
        route_name,
        rows,
        waypoints,
        dataset,
        origin_lat,
        origin_lon,
        config.OUTPUT_DIR,
    )

    print("rendered:", overview, flush=True)
    print("rendered:", process, flush=True)
    print("rendered:", video, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        choices=["route_B", "route_C", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.route == "all":
        for route_name in ["route_B", "route_C"]:
            render_route(route_name)
    else:
        render_route(args.route)


if __name__ == "__main__":
    main()
