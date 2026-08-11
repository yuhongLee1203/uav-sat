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
    payload = json.loads(Path(config.WAYPOINT_FILES[route_name]).read_text(encoding="utf-8"))
    raw = sorted(payload["waypoints"], key=lambda item: int(item["waypoint_order"]))
    result = []
    for item in raw:
        x_m, y_m = meters_from_latlon(
            item["latitude"], item["longitude"], origin_lat, origin_lon
        )
        result.append(
            {
                "order": int(item["waypoint_order"]),
                "x": float(x_m),
                "y": float(y_m),
            }
        )
    return result


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
    panel[oy:oy + dst_h, ox:ox + dst_w] = resized
    return panel


def render_overview(route_name, rows, waypoints, output_dir):
    gt = rows[["gt_x", "gt_y"]].to_numpy(dtype=float)
    visual = rows[["visual_x", "visual_y"]].to_numpy(dtype=float)
    final = rows[["final_x", "final_y"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(gt[:, 0], gt[:, 1], linewidth=2.0, label="GT")
    ax.plot(visual[:, 0], visual[:, 1], linewidth=1.5, label="v10 reversible visual")
    ax.plot(final[:, 0], final[:, 1], linewidth=1.8, label="post-model Kalman")

    for waypoint in waypoints:
        ax.scatter([waypoint["x"]], [waypoint["y"]], marker="X", s=90, zorder=5)
        ax.annotate(
            "W%d" % waypoint["order"],
            (waypoint["x"], waypoint["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    switched = rows[rows["waypoint_switched_after_frame"] > 0]
    for _, row in switched.iterrows():
        ax.scatter(
            [float(row["visual_x"])],
            [float(row["visual_y"])],
            facecolors="none",
            edgecolors="black",
            s=100,
            linewidths=1.5,
        )

    ax.set_title(
        "%s: Reversible Topology Recovery LSTM v10\n"
        "Forward 3x6 tracking + non-forward topology recovery + reversible leg state" % route_name
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
    indices = np.unique(np.linspace(0, len(rows) - 1, count).round().astype(int))
    columns = 3
    row_count = int(math.ceil(len(indices) / float(columns)))
    fig, axes = plt.subplots(row_count, columns, figsize=(16, 4.8 * row_count))
    axes = np.asarray(axes).reshape(-1)

    for plot_index, data_index in enumerate(indices):
        row = rows.iloc[data_index]
        ax = axes[plot_index]
        points = {
            "GT": (row["gt_x"], row["gt_y"], "o"),
            "HOLD": (row["hold_x"], row["hold_y"], "s"),
            "LOCAL": (row["local_x"], row["local_y"], "D"),
            "RECOVERY": (row["recovery_x"], row["recovery_y"], "^"),
            "VISUAL": (row["visual_x"], row["visual_y"], "P"),
            "KALMAN": (row["final_x"], row["final_y"], "*"),
        }
        for label, (x, y, marker) in points.items():
            ax.scatter([float(x)], [float(y)], marker=marker, s=90 if label == "KALMAN" else 55, label=label)

        xy = np.asarray([[float(v[0]), float(v[1])] for v in points.values()])
        center = xy.mean(axis=0)
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 35.0)
        margin = 0.75 * span
        ax.set_xlim(center[0] - margin, center[0] + margin)
        ax.set_ylim(center[1] - margin, center[1] + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            "f%d W%d->W%d heading=%.1f°\n"
            "H/L/R=%.2f/%.2f/%.2f leg P/C/N=%.2f/%.2f/%.2f\n"
            "visual=%.1fm kalman=%.1fm"
            % (
                int(row["frame_id"]),
                int(row["active_waypoint_from"]),
                int(row["active_waypoint_to"]),
                float(row["search_heading_deg"]),
                float(row["fusion_hold"]),
                float(row["fusion_local"]),
                float(row["fusion_recovery"]),
                float(row["leg_probability_previous"]),
                float(row["leg_probability_current"]),
                float(row["leg_probability_next"]),
                float(row["error_visual_m"]),
                float(row["error_final_m"]),
            ),
            fontsize=8,
        )
        if plot_index == 0:
            ax.legend(fontsize=6)

    for index in range(len(indices), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        "%s: forward 3x6 local tracking + reversible topology recovery" % route_name,
        fontsize=14,
    )
    fig.tight_layout()
    path = output_dir / (route_name + "_process_frames.png")
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def draw_marker(canvas, point, marker_type, size, thickness):
    cv2.drawMarker(
        canvas,
        (int(round(point[0])), int(round(point[1]))),
        (255, 255, 255),
        marker_type,
        int(size),
        int(thickness),
        cv2.LINE_AA,
    )


def render_video(route_name, rows, waypoints, dataset, origin_lat, origin_lon, output_dir):
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

    gt_canvas = xy_to_canvas(rows[["gt_x", "gt_y"]].to_numpy(dtype=float))
    visual_canvas = xy_to_canvas(rows[["visual_x", "visual_y"]].to_numpy(dtype=float))
    final_canvas = xy_to_canvas(rows[["final_x", "final_y"]].to_numpy(dtype=float))
    local_canvas = xy_to_canvas(rows[["local_x", "local_y"]].to_numpy(dtype=float))
    recovery_canvas = xy_to_canvas(rows[["recovery_x", "recovery_y"]].to_numpy(dtype=float))
    waypoint_canvas = xy_to_canvas(np.asarray([[wp["x"], wp["y"]] for wp in waypoints], dtype=float))

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
                canvas[:, :uav_width] = contain_image(image, uav_width, height)

            for wp_index, point in enumerate(waypoint_canvas):
                cv2.drawMarker(
                    canvas,
                    (int(point[0]), int(point[1])),
                    (0, 220, 255),
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "W%d" % waypoints[wp_index]["order"],
                    (int(point[0]) + 4, int(point[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            end = row_index + 1
            if end > 1:
                cv2.polylines(canvas, [np.round(gt_canvas[:end]).astype(np.int32).reshape(-1, 1, 2)], False, (60, 210, 80), 2, cv2.LINE_AA)
                cv2.polylines(canvas, [np.round(visual_canvas[:end]).astype(np.int32).reshape(-1, 1, 2)], False, (190, 80, 255), 2, cv2.LINE_AA)
                cv2.polylines(canvas, [np.round(final_canvas[:end]).astype(np.int32).reshape(-1, 1, 2)], False, (20, 140, 255), 3, cv2.LINE_AA)

            draw_marker(canvas, gt_canvas[row_index], cv2.MARKER_CROSS, 20, 3)
            draw_marker(canvas, local_canvas[row_index], cv2.MARKER_DIAMOND, 16, 2)
            draw_marker(canvas, recovery_canvas[row_index], cv2.MARKER_TRIANGLE_UP, 17, 2)
            draw_marker(canvas, visual_canvas[row_index], cv2.MARKER_STAR, 22, 3)
            draw_marker(canvas, final_canvas[row_index], cv2.MARKER_STAR, 28, 3)

            # Search-heading arrow is route tangent, not camera yaw.
            heading_rad = math.radians(float(row["search_heading_deg"]))
            start = final_canvas[row_index]
            arrow_len = 45.0
            end_point = (
                int(round(start[0] + arrow_len * math.cos(heading_rad))),
                int(round(start[1] - arrow_len * math.sin(heading_rad))),
            )
            cv2.arrowedLine(
                canvas,
                (int(round(start[0])), int(round(start[1]))),
                end_point,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

            cv2.rectangle(canvas, (0, 0), (width, 60), (0, 0, 0), -1)
            text = (
                "%s f=%d W%d->W%d heading=%.1fdeg | H/L/R=%.2f/%.2f/%.2f | "
                "leg P/C/N=%.2f/%.2f/%.2f switch=%d | err=%.1fm"
                % (
                    route_name.upper(),
                    int(row["frame_id"]),
                    int(row["active_waypoint_from"]),
                    int(row["active_waypoint_to"]),
                    float(row["search_heading_deg"]),
                    float(row["fusion_hold"]),
                    float(row["fusion_local"]),
                    float(row["fusion_recovery"]),
                    float(row["leg_probability_previous"]),
                    float(row["leg_probability_current"]),
                    float(row["leg_probability_next"]),
                    int(row["waypoint_switched_after_frame"]),
                    float(row["error_final_m"]),
                )
            )
            cv2.putText(canvas, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)

            info = [
                "WHITE ARROW = current route-tangent search direction (not camera yaw)",
                "LOCAL = forward 3x6 only; RECOVERY is not forward-limited",
                "RECOVERY = full PREVIOUS/CURRENT/NEXT leg corridor current-image search",
                "LEG STATE is reversible PREVIOUS/CURRENT/NEXT; rollback is allowed",
                "PURPLE = recurrent visual output; ORANGE = post-model position-only Kalman",
            ]
            box_y = height - 175
            cv2.rectangle(canvas, (10, box_y), (uav_width - 10, height - 10), (0, 0, 0), -1)
            for i, line in enumerate(info):
                cv2.putText(
                    canvas,
                    line,
                    (20, box_y + 27 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(canvas)
            if (row_index + 1) % 250 == 0:
                print("render %s: %d/%d" % (route_name, row_index + 1, len(rows)), flush=True)
    finally:
        writer.release()
    return path


def route_dataset(route_name, origin_lat, origin_lon):
    index = config.ROUTE_NAMES.index(route_name)
    return RouteDataset(
        Path(config.ROUTE_ROOTS[index]),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def render_route(route_name):
    csv_path = config.OUTPUT_DIR / (route_name + "_reversible_topology_frames.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "missing inference CSV: %s\nRun robust_tracker.py --mode eval/train_eval first." % csv_path
        )
    rows = pd.read_csv(csv_path)
    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    waypoints = load_waypoints(route_name, origin_lat, origin_lon)
    dataset = route_dataset(route_name, origin_lat, origin_lon)
    output_dir = config.OUTPUT_DIR / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("overview:", render_overview(route_name, rows, waypoints, output_dir), flush=True)
    print("process:", render_process(route_name, rows, output_dir), flush=True)
    print("video:", render_video(route_name, rows, waypoints, dataset, origin_lat, origin_lon, output_dir), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("route_B", "route_C", "all"), default="all")
    args = parser.parse_args()
    routes = ["route_B", "route_C"] if args.route == "all" else [args.route]
    for route_name in routes:
        render_route(route_name)


if __name__ == "__main__":
    main()
