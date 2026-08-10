import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

import config
from data import RouteDataset, meters_from_latlon


# One semantic color scheme for ALL generated visualizations.
# OpenCV BGR colors:
GT_BGR = (60, 200, 70)          # green
FINAL_BGR = (20, 140, 255)      # orange
PRED_BGR = (255, 210, 60)       # cyan
HARDMS_BGR = (210, 80, 220)     # purple
WAYPOINT_BGR = (0, 220, 255)    # yellow
TEXT_BGR = (245, 245, 245)
BLACK = (0, 0, 0)


def load_waypoints(route_name, origin_lat, origin_lon):
    payload = json.loads(
        Path(config.WAYPOINT_FILES[route_name]).read_text(encoding="utf-8")
    )
    rows = sorted(
        payload["waypoints"], key=lambda item: int(item["waypoint_order"])
    )
    result = []
    for item in rows:
        x, y = meters_from_latlon(
            item["latitude"], item["longitude"], origin_lat, origin_lon
        )
        result.append(
            {
                "order": int(item["waypoint_order"]),
                "role": str(item.get("role", "waypoint")),
                "frame": int(item["frame_index"]),
                "timestamp_ns": int(item["timestamp_ns"]),
                "x": float(x),
                "y": float(y),
            }
        )
    return result


def route_dataset(route_name, origin_lat, origin_lon):
    route_index = config.ROUTE_NAMES.index(route_name)
    return RouteDataset(
        Path(config.ROUTE_ROOTS[route_index]),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def meters_to_latlon(x_meter, y_meter, origin_lat, origin_lon):
    earth_radius_m = 6378137.0
    origin_lat_rad = math.radians(float(origin_lat))
    lat = float(origin_lat) + math.degrees(float(y_meter) / earth_radius_m)
    lon = float(origin_lon) + math.degrees(
        float(x_meter) / (earth_radius_m * math.cos(origin_lat_rad))
    )
    return lat, lon


def xy_to_source_pixels(xy, dataset, origin_lat, origin_lon):
    rows = []
    for x_meter, y_meter in np.asarray(xy, dtype=np.float64):
        lat, lon = meters_to_latlon(
            x_meter, y_meter, origin_lat, origin_lon
        )
        pixel_x, pixel_y = dataset.mapper.latlon_to_pixel(lat, lon)
        rows.append([float(pixel_x), float(pixel_y)])
    return np.asarray(rows, dtype=np.float64)


def contain_image(image, width, height):
    source_h, source_w = image.shape[:2]
    scale = min(float(width) / source_w, float(height) / source_h)
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - target_w) // 2
    y0 = (height - target_h) // 2
    panel[y0 : y0 + target_h, x0 : x0 + target_w] = resized
    return panel


def draw_marker(canvas, point, color, marker_type, size=18, thickness=2):
    point = (int(round(point[0])), int(round(point[1])))
    cv2.drawMarker(
        canvas,
        point,
        color,
        marker_type,
        size,
        thickness,
        cv2.LINE_AA,
    )


def draw_history(canvas, points, color, thickness):
    if len(points) < 2:
        return
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)


def draw_text_panel(canvas, lines, x, y, width):
    line_height = 26
    height = 16 + line_height * len(lines)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), BLACK, -1)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            str(line),
            (x + 10, y + 24 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            TEXT_BGR,
            1,
            cv2.LINE_AA,
        )


def draw_legend(canvas, x, y):
    rows = [
        ("GT (evaluation only)", GT_BGR, cv2.MARKER_CROSS),
        ("Final Kalman output", FINAL_BGR, cv2.MARKER_STAR),
        ("Motion prediction", PRED_BGR, cv2.MARKER_SQUARE),
        ("HardMS visual", HARDMS_BGR, cv2.MARKER_DIAMOND),
        ("Mission waypoint", WAYPOINT_BGR, cv2.MARKER_TILTED_CROSS),
    ]
    width = 270
    height = 14 + 28 * len(rows)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), BLACK, -1)
    for index, (label, color, marker_type) in enumerate(rows):
        row_y = y + 24 + 28 * index
        cv2.drawMarker(
            canvas,
            (x + 20, row_y - 5),
            color,
            marker_type,
            14,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x + 40, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_BGR,
            1,
            cv2.LINE_AA,
        )


def render_overview(route_name, rows, waypoints, output_dir):
    gt = rows[["gt_x", "gt_y"]].to_numpy(dtype=float)
    final = rows[["final_x", "final_y"]].to_numpy(dtype=float)
    hardms = rows[["hardms_x", "hardms_y"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(gt[:, 0], gt[:, 1], color="tab:green", linewidth=2.0, label="GT")
    ax.plot(
        final[:, 0],
        final[:, 1],
        color="tab:orange",
        linewidth=1.8,
        label="Final Kalman",
    )

    sample_stride = max(1, len(hardms) // 200)
    ax.scatter(
        hardms[::sample_stride, 0],
        hardms[::sample_stride, 1],
        color="tab:purple",
        s=10,
        alpha=0.30,
        label="HardMS samples",
    )

    for waypoint in waypoints:
        ax.scatter(
            [waypoint["x"]],
            [waypoint["y"]],
            color="gold",
            edgecolors="black",
            marker="X",
            s=90,
            zorder=5,
        )
        ax.annotate(
            f"W{waypoint['order']} / f{waypoint['frame']}",
            (waypoint["x"], waypoint["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    interval = max(1, int(config.FRAME_LABEL_INTERVAL))
    for index in range(0, len(rows), interval):
        frame_id = int(rows.iloc[index]["frame_id"])
        ax.scatter(
            [final[index, 0]], [final[index, 1]], color="tab:orange", s=22
        )
        ax.annotate(
            f"f{frame_id}",
            (final[index, 0], final[index, 1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=7,
            color="tab:orange",
        )

    ax.set_title(
        f"{route_name}: localization overview\n"
        "All mission waypoints + frame-labelled final positions"
    )
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{route_name}_overview_frames.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def render_process_frames(route_name, rows, waypoints, output_dir):
    count = min(int(config.PROCESS_SNAPSHOT_COUNT), len(rows))
    indices = np.unique(
        np.linspace(0, len(rows) - 1, count).round().astype(int)
    )
    columns = 3
    row_count = int(math.ceil(len(indices) / float(columns)))
    fig, axes = plt.subplots(row_count, columns, figsize=(16, 4.8 * row_count))
    axes = np.asarray(axes).reshape(-1)
    waypoint_xy = np.asarray(
        [[item["x"], item["y"]] for item in waypoints], dtype=float
    )
    waypoint_by_order = {
        int(item["order"]): np.asarray([item["x"], item["y"]], dtype=float)
        for item in waypoints
    }

    for subplot_index, data_index in enumerate(indices):
        ax = axes[subplot_index]
        row = rows.iloc[data_index]
        gt = np.asarray([row["gt_x"], row["gt_y"]], dtype=float)
        final = np.asarray([row["final_x"], row["final_y"]], dtype=float)
        pred = np.asarray([row["prediction_x"], row["prediction_y"]], dtype=float)
        hardms = np.asarray([row["hardms_x"], row["hardms_y"]], dtype=float)
        leg_from = int(row["waypoint_from"])
        leg_to = int(row["waypoint_to"])
        start = waypoint_by_order[leg_from]
        end = waypoint_by_order[leg_to]

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="0.65",
            linewidth=2.0,
            label="active mission leg",
        )
        ax.scatter(gt[0], gt[1], color="tab:green", marker="o", s=85, label="GT")
        ax.scatter(
            final[0], final[1], color="tab:orange", marker="*", s=130, label="Final"
        )
        ax.scatter(
            pred[0], pred[1], color="tab:cyan", marker="s", s=60, label="Prediction"
        )
        ax.scatter(
            hardms[0], hardms[1], color="tab:purple", marker="D", s=60, label="HardMS"
        )
        ax.plot(
            [gt[0], final[0]],
            [gt[1], final[1]],
            linestyle="--",
            color="0.35",
            linewidth=1.2,
        )

        current = np.vstack([gt, final, pred, hardms, start, end])
        center = current.mean(axis=0)
        span = max(np.ptp(current[:, 0]), np.ptp(current[:, 1]), 50.0)
        margin = 0.65 * span
        ax.set_xlim(center[0] - margin, center[0] + margin)
        ax.set_ylim(center[1] - margin, center[1] + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"frame {int(row['frame_id'])} | t={float(row['elapsed_time_s']):.2f}s | "
            f"W{leg_from}->W{leg_to}\nerror={float(row['error_m']):.1f}m"
        )
        if subplot_index == 0:
            ax.legend(fontsize=8)

    for index in range(len(indices), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        f"{route_name}: localization process (Prediction -> HardMS -> Final Kalman)",
        fontsize=14,
    )
    fig.tight_layout()
    path = output_dir / f"{route_name}_process_frames.png"
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
    fps,
    width,
    height,
):
    with Image.open(config.SAT_IMAGE) as image:
        source_width, source_height = image.size
        map_width = min(
            int(round(source_width * float(height) / float(source_height))),
            int(width * 0.63),
        )
        resampling = getattr(Image, "Resampling", Image)
        rgb = np.asarray(
            image.convert("RGB").resize(
                (map_width, height), resampling.LANCZOS
            )
        )
    map_panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    uav_width = width - map_width
    if uav_width < 420:
        raise RuntimeError("VIDEO_WIDTH is too small for synchronized UAV panel")

    scale_x = float(map_width) / float(source_width)
    scale_y = float(height) / float(source_height)

    def xy_to_canvas(xy):
        source = xy_to_source_pixels(xy, dataset, origin_lat, origin_lon)
        result = np.empty_like(source)
        result[:, 0] = source[:, 0] * scale_x + uav_width
        result[:, 1] = source[:, 1] * scale_y
        return result

    gt = rows[["gt_x", "gt_y"]].to_numpy(dtype=float)
    final = rows[["final_x", "final_y"]].to_numpy(dtype=float)
    pred = rows[["prediction_x", "prediction_y"]].to_numpy(dtype=float)
    hardms = rows[["hardms_x", "hardms_y"]].to_numpy(dtype=float)
    waypoint_xy = np.asarray(
        [[item["x"], item["y"]] for item in waypoints], dtype=float
    )

    gt_canvas = xy_to_canvas(gt)
    final_canvas = xy_to_canvas(final)
    pred_canvas = xy_to_canvas(pred)
    hardms_canvas = xy_to_canvas(hardms)
    waypoint_canvas = xy_to_canvas(waypoint_xy)

    sample_by_frame = {
        int(sample["frame_id"]): sample for sample in dataset.samples
    }
    output_path = output_dir / f"{route_name}_synchronized_inference.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    try:
        for index, row in rows.iterrows():
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:, uav_width:] = map_panel
            frame_id = int(row["frame_id"])
            sample = sample_by_frame.get(frame_id)
            if sample is not None:
                image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
                if image is not None:
                    canvas[:, :uav_width] = contain_image(
                        image, uav_width, height
                    )

            # Every clicked/recorded mission waypoint is always visible.
            for waypoint_index, point in enumerate(waypoint_canvas):
                draw_marker(
                    canvas,
                    point,
                    WAYPOINT_BGR,
                    cv2.MARKER_TILTED_CROSS,
                    size=13,
                    thickness=2,
                )
                wp = waypoints[waypoint_index]
                position = (int(point[0]) + 4, int(point[1]) - 4)
                label = f"W{wp['order']}/f{wp['frame']}"
                cv2.putText(
                    canvas,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    BLACK,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    WAYPOINT_BGR,
                    1,
                    cv2.LINE_AA,
                )

            end = index + 1
            draw_history(canvas, gt_canvas[:end], GT_BGR, 2)
            draw_history(canvas, final_canvas[:end], FINAL_BGR, 3)

            current_gt = gt_canvas[index]
            current_final = final_canvas[index]
            current_pred = pred_canvas[index]
            current_hardms = hardms_canvas[index]
            draw_marker(canvas, current_gt, GT_BGR, cv2.MARKER_CROSS, 22, 3)
            draw_marker(canvas, current_final, FINAL_BGR, cv2.MARKER_STAR, 25, 3)
            draw_marker(canvas, current_pred, PRED_BGR, cv2.MARKER_SQUARE, 18, 2)
            draw_marker(canvas, current_hardms, HARDMS_BGR, cv2.MARKER_DIAMOND, 18, 2)
            cv2.line(
                canvas,
                tuple(np.round(current_gt).astype(int)),
                tuple(np.round(current_final).astype(int)),
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

            elapsed = float(row["elapsed_time_s"])
            top = (
                f"{route_name.upper()} | source frame={frame_id} | t={elapsed:.2f}s | "
                f"timestamp_ns={int(row['timestamp_ns'])} | "
                f"active W{int(row['waypoint_from'])}->W{int(row['waypoint_to'])} | "
                f"error={float(row['error_m']):.1f}m"
            )
            cv2.rectangle(canvas, (0, 0), (width, 46), BLACK, -1)
            cv2.putText(
                canvas,
                top,
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                TEXT_BGR,
                1,
                cv2.LINE_AA,
            )

            draw_text_panel(
                canvas,
                [
                    "LEFT = exact UAV source image used at this inference step",
                    f"frame={frame_id} | elapsed={elapsed:.2f}s | dt={float(row['dt_seconds']):.3f}s",
                    f"Prediction XY=({float(row['prediction_x']):.1f}, {float(row['prediction_y']):.1f}) m",
                    f"HardMS XY=({float(row['hardms_x']):.1f}, {float(row['hardms_y']):.1f}) m",
                    f"Final XY=({float(row['final_x']):.1f}, {float(row['final_y']):.1f}) m",
                    f"State s={float(row['final_s']):.1f}m, v={float(row['final_v_mps']):.2f}m/s, d={float(row['final_d']):.1f}m",
                ],
                10,
                height - 190,
                uav_width - 20,
            )
            draw_legend(canvas, uav_width + 10, height - 170)
            writer.write(canvas)

            if (index + 1) % 250 == 0:
                print(
                    f"rendering {route_name}: {index + 1}/{len(rows)}",
                    flush=True,
                )
    finally:
        writer.release()

    return output_path


def render_route(route_name, output_dir, fps, width, height):
    csv_path = config.OUTPUT_DIR / f"{route_name}_route_coordinate_frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = pd.read_csv(csv_path)

    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    dataset = route_dataset(route_name, origin_lat, origin_lon)
    waypoints = load_waypoints(route_name, origin_lat, origin_lon)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = render_overview(route_name, rows, waypoints, output_dir)
    process = render_process_frames(route_name, rows, waypoints, output_dir)
    video = render_video(
        route_name,
        rows,
        waypoints,
        dataset,
        origin_lat,
        origin_lon,
        output_dir,
        fps,
        width,
        height,
    )
    print("overview:", overview, flush=True)
    print("process frames:", process, flush=True)
    print("video:", video, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route", choices=("route_B", "route_C", "all"), default="all"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "visualizations",
    )
    parser.add_argument("--fps", type=float, default=float(config.VIDEO_FPS))
    parser.add_argument("--width", type=int, default=int(config.VIDEO_WIDTH))
    parser.add_argument("--height", type=int, default=int(config.VIDEO_HEIGHT))
    args = parser.parse_args()

    routes = ["route_B", "route_C"] if args.route == "all" else [args.route]
    for route_name in routes:
        render_route(
            route_name,
            args.output_dir,
            float(args.fps),
            int(args.width),
            int(args.height),
        )


if __name__ == "__main__":
    main()
