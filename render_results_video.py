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


GT_COLOR = (60, 210, 80)
HOLD_COLOR = (170, 170, 170)
LOCAL_COLOR = (255, 150, 50)
RECOVERY_COLOR = (220, 80, 220)
WAYPOINT_BRANCH_COLOR = (0, 220, 255)
VISUAL_COLOR = (190, 80, 255)
FINAL_COLOR = (20, 140, 255)
POLY_COLOR = (255, 210, 60)
TEXT_COLOR = (245, 245, 245)
BLACK = (0, 0, 0)


def branch_name(value):
    value = int(value)

    names = {
        int(config.HYPOTHESIS_HOLD): "HOLD",
        int(config.HYPOTHESIS_LOCAL): "LOCAL",
        int(config.HYPOTHESIS_RECOVERY): "RECOVERY",
        int(config.HYPOTHESIS_WAYPOINT): "WAYPOINT",
    }

    return names.get(value, "UNKNOWN")


def load_waypoints(
    route_name,
    origin_lat,
    origin_lon,
):
    payload = json.loads(
        Path(
            config.WAYPOINT_FILES[
                route_name
            ]
        ).read_text(
            encoding="utf-8"
        )
    )

    raw = sorted(
        payload["waypoints"],
        key=lambda item: int(
            item["waypoint_order"]
        ),
    )

    result = []

    for item in raw:
        x_m, y_m = meters_from_latlon(
            item["latitude"],
            item["longitude"],
            origin_lat,
            origin_lon,
        )

        result.append(
            {
                "order": int(
                    item["waypoint_order"]
                ),
                "x": float(x_m),
                "y": float(y_m),
            }
        )

    return result


def meters_to_latlon(
    x_meter,
    y_meter,
    origin_lat,
    origin_lon,
):
    radius = 6378137.0

    latitude = (
        float(origin_lat)
        + math.degrees(
            float(y_meter) / radius
        )
    )

    longitude_scale = (
        radius
        * math.cos(
            math.radians(
                float(origin_lat)
            )
        )
    )

    longitude = (
        float(origin_lon)
        + math.degrees(
            float(x_meter)
            / longitude_scale
        )
    )

    return latitude, longitude


def xy_to_source_pixels(
    xy,
    dataset,
    origin_lat,
    origin_lon,
):
    points = []

    for x_meter, y_meter in np.asarray(
        xy,
        dtype=np.float64,
    ):
        lat, lon = meters_to_latlon(
            x_meter,
            y_meter,
            origin_lat,
            origin_lon,
        )

        pixel_x, pixel_y = (
            dataset.mapper.latlon_to_pixel(
                lat,
                lon,
            )
        )

        points.append(
            [
                float(pixel_x),
                float(pixel_y),
            ]
        )

    return np.asarray(
        points,
        dtype=np.float64,
    )


def contain_image(
    image,
    width,
    height,
):
    source_height, source_width = (
        image.shape[:2]
    )

    scale = min(
        float(width)
        / float(source_width),
        float(height)
        / float(source_height),
    )

    destination_width = max(
        1,
        int(
            round(
                source_width * scale
            )
        ),
    )

    destination_height = max(
        1,
        int(
            round(
                source_height * scale
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            destination_width,
            destination_height,
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale <= 1.0
            else cv2.INTER_LINEAR
        ),
    )

    panel = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        width - destination_width
    ) // 2

    offset_y = (
        height - destination_height
    ) // 2

    panel[
        offset_y:
        offset_y + destination_height,
        offset_x:
        offset_x + destination_width,
    ] = resized

    return panel


def draw_marker(
    canvas,
    point,
    color,
    marker_type,
    size,
    thickness,
):
    cv2.drawMarker(
        canvas,
        (
            int(round(point[0])),
            int(round(point[1])),
        ),
        color,
        marker_type,
        int(size),
        int(thickness),
        cv2.LINE_AA,
    )


def draw_history(
    canvas,
    points,
    color,
    thickness,
):
    if len(points) < 2:
        return

    integer_points = np.round(
        points
    ).astype(np.int32)

    cv2.polylines(
        canvas,
        [
            integer_points.reshape(
                -1,
                1,
                2,
            )
        ],
        False,
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def render_overview(
    route_name,
    rows,
    waypoints,
    output_dir,
):
    gt = rows[
        ["gt_x", "gt_y"]
    ].to_numpy(dtype=float)

    visual = rows[
        ["visual_x", "visual_y"]
    ].to_numpy(dtype=float)

    final = rows[
        ["final_x", "final_y"]
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    ax.plot(
        gt[:, 0],
        gt[:, 1],
        color="tab:green",
        linewidth=2.0,
        label="GT (evaluation only)",
    )

    ax.plot(
        visual[:, 0],
        visual[:, 1],
        color="tab:purple",
        linewidth=1.6,
        label="Route-bounded recurrent visual",
    )

    ax.plot(
        final[:, 0],
        final[:, 1],
        color="tab:orange",
        linewidth=1.8,
        label="Final position-only Kalman",
    )

    for waypoint in waypoints:
        ax.scatter(
            [waypoint["x"]],
            [waypoint["y"]],
            marker="X",
            s=90,
            color="gold",
            edgecolors="black",
            zorder=5,
        )

        ax.annotate(
            f"W{waypoint['order']}",
            (
                waypoint["x"],
                waypoint["y"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    interval = max(
        1,
        int(config.FRAME_LABEL_INTERVAL),
    )

    for index in range(
        0,
        len(rows),
        interval,
    ):
        frame_id = int(
            rows.iloc[index]["frame_id"]
        )

        ax.annotate(
            f"f{frame_id}",
            (
                final[index, 0],
                final[index, 1],
            ),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=7,
            color="tab:orange",
        )

    switched = rows[
        rows[
            "waypoint_switched_after_frame"
        ] > 0
    ]

    for _, row in switched.iterrows():
        ax.scatter(
            [float(row["visual_x"])],
            [float(row["visual_y"])],
            facecolors="none",
            edgecolors="black",
            s=100,
            linewidths=1.6,
        )

    ax.set_title(
        (
            f"{route_name}: Route-Bounded Hypothesis LSTM v6\n"
            "HOLD / LOCAL / RECOVERY / WAYPOINT"
        )
    )

    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        / f"{route_name}_overview_frames.png"
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(fig)

    return path


def render_process(
    route_name,
    rows,
    output_dir,
):
    count = min(
        int(config.PROCESS_SNAPSHOT_COUNT),
        len(rows),
    )

    indices = np.unique(
        np.linspace(
            0,
            len(rows) - 1,
            count,
        ).round().astype(int)
    )

    columns = 3

    row_count = int(
        math.ceil(
            len(indices)
            / float(columns)
        )
    )

    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(
            16,
            4.8 * row_count,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for plot_index, data_index in enumerate(
        indices
    ):
        row = rows.iloc[data_index]
        ax = axes[plot_index]

        points = {
            "GT": (
                row["gt_x"],
                row["gt_y"],
                "tab:green",
                "o",
            ),
            "HOLD": (
                row["hold_x"],
                row["hold_y"],
                "0.45",
                "s",
            ),
            "LOCAL": (
                row["local_x"],
                row["local_y"],
                "tab:blue",
                "D",
            ),
            "RECOVERY": (
                row["recovery_x"],
                row["recovery_y"],
                "tab:pink",
                "^",
            ),
            "WAYPOINT": (
                row["waypoint_x"],
                row["waypoint_y"],
                "goldenrod",
                "X",
            ),
            "POLY": (
                row["polynomial_x"],
                row["polynomial_y"],
                "tab:cyan",
                "x",
            ),
            "VISUAL": (
                row["visual_x"],
                row["visual_y"],
                "tab:purple",
                "P",
            ),
            "FINAL": (
                row["final_x"],
                row["final_y"],
                "tab:orange",
                "*",
            ),
        }

        for label, (
            x_value,
            y_value,
            color,
            marker,
        ) in points.items():
            ax.scatter(
                [float(x_value)],
                [float(y_value)],
                color=color,
                marker=marker,
                s=(
                    120
                    if label == "FINAL"
                    else 65
                ),
                label=label,
            )

        xy = np.asarray(
            [
                [
                    float(item[0]),
                    float(item[1]),
                ]
                for item
                in points.values()
            ]
        )

        center = xy.mean(axis=0)

        span = max(
            np.ptp(xy[:, 0]),
            np.ptp(xy[:, 1]),
            45.0,
        )

        margin = 0.7 * span

        ax.set_xlim(
            center[0] - margin,
            center[0] + margin,
        )

        ax.set_ylim(
            center[1] - margin,
            center[1] + margin,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(True, alpha=0.25)

        selected = branch_name(
            row["selected_branch"]
        )

        ax.set_title(
            (
                f"frame {int(row['frame_id'])} "
                f"W{int(row['active_waypoint_from'])}"
                f"->W{int(row['active_waypoint_to'])}\n"
                f"{selected} | "
                f"H={float(row['branch_hold']):.2f} "
                f"L={float(row['branch_local']):.2f} "
                f"R={float(row['branch_recovery']):.2f} "
                f"W={float(row['branch_waypoint']):.2f}\n"
                f"visual err={float(row['error_visual_m']):.1f}m | "
                f"final err={float(row['error_final_m']):.1f}m"
            ),
            fontsize=8,
        )

        if plot_index == 0:
            ax.legend(fontsize=6)

    for index in range(
        len(indices),
        len(axes),
    ):
        axes[index].axis("off")

    fig.suptitle(
        (
            f"{route_name}: current-frame hypothesis process\n"
            "HOLD vs LOCAL vs active-route RECOVERY vs WAYPOINT transition"
        ),
        fontsize=14,
    )

    fig.tight_layout()

    path = (
        output_dir
        / f"{route_name}_process_frames.png"
    )

    fig.savefig(
        path,
        dpi=190,
    )

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
    with Image.open(
        config.SAT_IMAGE
    ) as image:
        source_width, source_height = (
            image.size
        )

        map_height = int(
            config.VIDEO_HEIGHT
        )

        map_width = min(
            int(
                round(
                    source_width
                    * (
                        float(map_height)
                        / float(source_height)
                    )
                )
            ),
            int(
                config.VIDEO_WIDTH * 0.63
            ),
        )

        map_rgb = np.asarray(
            image.convert("RGB").resize(
                (
                    map_width,
                    map_height,
                ),
                Image.Resampling.LANCZOS,
            )
        )

    map_panel = cv2.cvtColor(
        map_rgb,
        cv2.COLOR_RGB2BGR,
    )

    width = int(config.VIDEO_WIDTH)
    height = int(config.VIDEO_HEIGHT)
    uav_width = width - map_width

    scale_x = (
        float(map_width)
        / float(source_width)
    )

    scale_y = (
        float(map_height)
        / float(source_height)
    )

    def xy_to_canvas(xy):
        source = xy_to_source_pixels(
            xy,
            dataset,
            origin_lat,
            origin_lon,
        )

        result = np.empty_like(source)

        result[:, 0] = (
            source[:, 0]
            * scale_x
            + uav_width
        )

        result[:, 1] = (
            source[:, 1]
            * scale_y
        )

        return result

    def canvas_points(x_name, y_name):
        return xy_to_canvas(
            rows[
                [x_name, y_name]
            ].to_numpy(dtype=float)
        )

    gt_canvas = canvas_points(
        "gt_x",
        "gt_y",
    )

    hold_canvas = canvas_points(
        "hold_x",
        "hold_y",
    )

    local_canvas = canvas_points(
        "local_x",
        "local_y",
    )

    recovery_canvas = canvas_points(
        "recovery_x",
        "recovery_y",
    )

    waypoint_hypothesis_canvas = (
        canvas_points(
            "waypoint_x",
            "waypoint_y",
        )
    )

    visual_canvas = canvas_points(
        "visual_x",
        "visual_y",
    )

    final_canvas = canvas_points(
        "final_x",
        "final_y",
    )

    polynomial_canvas = canvas_points(
        "polynomial_x",
        "polynomial_y",
    )

    waypoint_canvas = xy_to_canvas(
        np.asarray(
            [
                [
                    wp["x"],
                    wp["y"],
                ]
                for wp in waypoints
            ],
            dtype=float,
        )
    )

    output_path = (
        output_dir
        / f"{route_name}_synchronized_inference.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(config.VIDEO_FPS),
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create video: {output_path}"
        )

    try:
        for row_index, row in rows.iterrows():
            canvas = np.zeros(
                (
                    height,
                    width,
                    3,
                ),
                dtype=np.uint8,
            )

            canvas[
                :,
                uav_width:
            ] = map_panel

            image = cv2.imread(
                str(row["image_path"]),
                cv2.IMREAD_COLOR,
            )

            if image is not None:
                canvas[
                    :,
                    :uav_width
                ] = contain_image(
                    image,
                    uav_width,
                    height,
                )

            for waypoint_index, point in enumerate(
                waypoint_canvas
            ):
                draw_marker(
                    canvas,
                    point,
                    WAYPOINT_BRANCH_COLOR,
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    2,
                )

                cv2.putText(
                    canvas,
                    f"W{waypoints[waypoint_index]['order']}",
                    (
                        int(point[0]) + 5,
                        int(point[1]) - 5,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    WAYPOINT_BRANCH_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            end_index = row_index + 1

            draw_history(
                canvas,
                gt_canvas[:end_index],
                GT_COLOR,
                2,
            )

            draw_history(
                canvas,
                visual_canvas[:end_index],
                VISUAL_COLOR,
                2,
            )

            draw_history(
                canvas,
                final_canvas[:end_index],
                FINAL_COLOR,
                3,
            )

            draw_marker(
                canvas,
                gt_canvas[row_index],
                GT_COLOR,
                cv2.MARKER_CROSS,
                21,
                3,
            )

            draw_marker(
                canvas,
                hold_canvas[row_index],
                HOLD_COLOR,
                cv2.MARKER_SQUARE,
                14,
                2,
            )

            draw_marker(
                canvas,
                local_canvas[row_index],
                LOCAL_COLOR,
                cv2.MARKER_DIAMOND,
                17,
                2,
            )

            draw_marker(
                canvas,
                recovery_canvas[row_index],
                RECOVERY_COLOR,
                cv2.MARKER_TRIANGLE_UP,
                18,
                2,
            )

            draw_marker(
                canvas,
                waypoint_hypothesis_canvas[
                    row_index
                ],
                WAYPOINT_BRANCH_COLOR,
                cv2.MARKER_TILTED_CROSS,
                20,
                3,
            )

            draw_marker(
                canvas,
                polynomial_canvas[row_index],
                POLY_COLOR,
                cv2.MARKER_CROSS,
                13,
                2,
            )

            draw_marker(
                canvas,
                visual_canvas[row_index],
                VISUAL_COLOR,
                cv2.MARKER_STAR,
                23,
                3,
            )

            draw_marker(
                canvas,
                final_canvas[row_index],
                FINAL_COLOR,
                cv2.MARKER_STAR,
                28,
                3,
            )

            branch = branch_name(
                row["selected_branch"]
            )

            cv2.rectangle(
                canvas,
                (0, 0),
                (width, 58),
                BLACK,
                -1,
            )

            top_text = (
                f"{route_name.upper()} | frame={int(row['frame_id'])} | "
                f"W{int(row['active_waypoint_from'])}"
                f"->W{int(row['active_waypoint_to'])} | "
                f"{branch} | "
                f"H={float(row['branch_hold']):.2f} "
                f"L={float(row['branch_local']):.2f} "
                f"R={float(row['branch_recovery']):.2f} "
                f"W={float(row['branch_waypoint']):.2f} | "
                f"err={float(row['error_final_m']):.1f}m"
            )

            cv2.putText(
                canvas,
                top_text,
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            info = [
                "LEFT = exact current UAV frame",
                "GRAY HOLD = keep previous image-derived position",
                "BLUE LOCAL = current image in previous 6x6 neighborhood",
                "MAGENTA RECOVERY = current image over active Start->End corridor",
                "YELLOW WAYPOINT = current image in endpoint transition neighborhood",
                "PURPLE = recurrent visual; ORANGE = position-only Kalman",
                (
                    f"waypoint_selected={int(row['waypoint_branch_selected'])} "
                    f"switch={int(row['waypoint_switched_after_frame'])} "
                    f"mission_complete={int(row['mission_complete'])}"
                ),
            ]

            box_y = height - 230

            cv2.rectangle(
                canvas,
                (10, box_y),
                (
                    uav_width - 10,
                    height - 10,
                ),
                BLACK,
                -1,
            )

            for line_index, line in enumerate(
                info
            ):
                cv2.putText(
                    canvas,
                    line,
                    (
                        20,
                        box_y
                        + 27
                        + line_index * 28,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            writer.write(canvas)

            if (
                row_index + 1
            ) % 250 == 0:
                print(
                    f"render {route_name}: "
                    f"{row_index + 1}/{len(rows)}",
                    flush=True,
                )

    finally:
        writer.release()

    return output_path


def route_dataset(
    route_name,
    origin_lat,
    origin_lon,
):
    index = config.ROUTE_NAMES.index(
        route_name
    )

    return RouteDataset(
        Path(
            config.ROUTE_ROOTS[index]
        ),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def render_route(route_name):
    csv_path = (
        config.OUTPUT_DIR
        / (
            f"{route_name}_"
            "route_hypothesis_lstm_frames.csv"
        )
    )

    if not csv_path.exists():
        available = sorted(
            str(path)
            for path in config.OUTPUT_DIR.glob("*frames.csv")
        )

        message = [
            f"Missing inference CSV: {csv_path}",
            "",
            "The visualization step cannot run before robust_tracker.py --mode eval",
            "has generated the Route-B/Route-C per-frame CSV files.",
            "",
            "Available *frames.csv files in this experiment directory:",
        ]

        if available:
            message.extend(
                f"  {path}"
                for path in available
            )
        else:
            message.append("  <none>")

        raise FileNotFoundError(
            "\n".join(message)
        )

    rows = pd.read_csv(csv_path)

    checkpoint = torch.load(
        config.VISUAL_CHECKPOINT,
        map_location="cpu",
    )

    origin_lat = float(
        checkpoint["origin_lat"]
    )

    origin_lon = float(
        checkpoint["origin_lon"]
    )

    waypoints = load_waypoints(
        route_name,
        origin_lat,
        origin_lon,
    )

    dataset = route_dataset(
        route_name,
        origin_lat,
        origin_lon,
    )

    output_dir = (
        config.OUTPUT_DIR
        / "visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overview = render_overview(
        route_name,
        rows,
        waypoints,
        output_dir,
    )

    process = render_process(
        route_name,
        rows,
        output_dir,
    )

    video = render_video(
        route_name,
        rows,
        waypoints,
        dataset,
        origin_lat,
        origin_lon,
        output_dir,
    )

    print("overview:", overview, flush=True)
    print("process:", process, flush=True)
    print("video:", video, flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--route",
        choices=(
            "route_B",
            "route_C",
            "all",
        ),
        default="all",
    )

    args = parser.parse_args()

    routes = (
        ["route_B", "route_C"]
        if args.route == "all"
        else [args.route]
    )

    for route_name in routes:
        render_route(route_name)


if __name__ == "__main__":
    main()
