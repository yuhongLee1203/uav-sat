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


GT_COLOR = (60, 210, 80)
VISUAL_COLOR = (210, 80, 220)
FINAL_COLOR = (20, 140, 255)
POLY_COLOR = (255, 210, 60)
WAYPOINT_COLOR = (0, 220, 255)
TEXT_COLOR = (245, 245, 245)
BLACK = (0, 0, 0)


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

    rows = sorted(
        payload[
            "waypoints"
        ],
        key=lambda item: int(
            item[
                "waypoint_order"
            ]
        ),
    )

    result = []

    for item in rows:
        x_meter, y_meter = (
            meters_from_latlon(
                item[
                    "latitude"
                ],
                item[
                    "longitude"
                ],
                origin_lat,
                origin_lon,
            )
        )

        result.append(
            {
                "order": int(
                    item[
                        "waypoint_order"
                    ]
                ),
                "x": float(
                    x_meter
                ),
                "y": float(
                    y_meter
                ),
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
        float(
            origin_lat
        )
        + math.degrees(
            float(
                y_meter
            )
            / radius
        )
    )

    longitude_scale = (
        radius
        * math.cos(
            math.radians(
                float(
                    origin_lat
                )
            )
        )
    )

    longitude = (
        float(
            origin_lon
        )
        + math.degrees(
            float(
                x_meter
            )
            / longitude_scale
        )
    )

    return (
        latitude,
        longitude,
    )


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
        latitude, longitude = (
            meters_to_latlon(
                x_meter,
                y_meter,
                origin_lat,
                origin_lon,
            )
        )

        pixel_x, pixel_y = (
            dataset.mapper.latlon_to_pixel(
                latitude,
                longitude,
            )
        )

        points.append(
            [
                float(
                    pixel_x
                ),
                float(
                    pixel_y
                ),
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
        image.shape[
            :2
        ]
    )

    scale = min(
        float(
            width
        )
        / float(
            source_width
        ),
        float(
            height
        )
        / float(
            source_height
        ),
    )

    destination_width = max(
        1,
        int(
            round(
                source_width
                * scale
            )
        ),
    )

    destination_height = max(
        1,
        int(
            round(
                source_height
                * scale
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
            if scale
            <= 1.0
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
        width
        - destination_width
    ) // 2

    offset_y = (
        height
        - destination_height
    ) // 2

    panel[
        offset_y:
        offset_y
        + destination_height,
        offset_x:
        offset_x
        + destination_width,
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
            int(
                round(
                    point[
                        0
                    ]
                )
            ),
            int(
                round(
                    point[
                        1
                    ]
                )
            ),
        ),
        color,
        marker_type,
        int(
            size
        ),
        int(
            thickness
        ),
        cv2.LINE_AA,
    )


def draw_history(
    canvas,
    points,
    color,
    thickness,
):
    if len(
        points
    ) < 2:
        return

    integer_points = np.round(
        points
    ).astype(
        np.int32
    )

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
        int(
            thickness
        ),
        cv2.LINE_AA,
    )


def phase_name(
    value,
):
    value = int(
        value
    )

    if value == int(
        config.PHASE_STATIONARY
    ):
        return "STATIONARY"

    if value == int(
        config.PHASE_TRANSLATION
    ):
        return "TRANSLATION"

    if value == int(
        config.PHASE_ROTATION
    ):
        return "ROTATION"

    return "UNKNOWN"


def render_overview(
    route_name,
    rows,
    waypoints,
    output_dir,
):
    gt = rows[
        [
            "gt_x",
            "gt_y",
        ]
    ].to_numpy(
        dtype=float
    )

    visual = rows[
        [
            "visual_measurement_x",
            "visual_measurement_y",
        ]
    ].to_numpy(
        dtype=float
    )

    final = rows[
        [
            "final_x",
            "final_y",
        ]
    ].to_numpy(
        dtype=float
    )

    polynomial = rows[
        [
            "polynomial_x",
            "polynomial_y",
        ]
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(
            14,
            9,
        )
    )

    ax.plot(
        gt[
            :,
            0
        ],
        gt[
            :,
            1
        ],
        color="tab:green",
        linewidth=2.0,
        label="GT (evaluation only)",
    )

    ax.plot(
        visual[
            :,
            0
        ],
        visual[
            :,
            1
        ],
        color="tab:purple",
        linewidth=1.5,
        label="Image-motion-gated visual state",
    )

    ax.plot(
        final[
            :,
            0
        ],
        final[
            :,
            1
        ],
        color="tab:orange",
        linewidth=1.8,
        label="Final position-only Kalman",
    )

    stride = max(
        1,
        len(
            polynomial
        )
        // 180,
    )

    ax.scatter(
        polynomial[
            ::stride,
            0
        ],
        polynomial[
            ::stride,
            1
        ],
        s=8,
        alpha=0.30,
        color="tab:cyan",
        label="Observed-motion polynomial prior",
    )

    for waypoint in waypoints:
        ax.scatter(
            [
                waypoint[
                    "x"
                ]
            ],
            [
                waypoint[
                    "y"
                ]
            ],
            marker="X",
            s=90,
            color="gold",
            edgecolors="black",
            zorder=5,
        )

        ax.annotate(
            f"W{waypoint['order']}",
            (
                waypoint[
                    "x"
                ],
                waypoint[
                    "y"
                ],
            ),
            xytext=(
                5,
                5,
            ),
            textcoords="offset points",
            fontsize=8,
        )

    interval = max(
        1,
        int(
            config.FRAME_LABEL_INTERVAL
        ),
    )

    for index in range(
        0,
        len(
            rows
        ),
        interval,
    ):
        frame_id = int(
            rows.iloc[
                index
            ][
                "frame_id"
            ]
        )

        ax.annotate(
            f"f{frame_id}",
            (
                final[
                    index,
                    0
                ],
                final[
                    index,
                    1
                ],
            ),
            xytext=(
                4,
                -12,
            ),
            textcoords="offset points",
            fontsize=7,
            color="tab:orange",
        )

    terminal_rows = rows[
        rows[
            "terminal_locked"
        ]
        > 0
    ]

    if len(
        terminal_rows
    ) > 0:
        first_terminal = (
            terminal_rows.iloc[
                0
            ]
        )

        ax.scatter(
            [
                float(
                    first_terminal[
                        "visual_measurement_x"
                    ]
                )
            ],
            [
                float(
                    first_terminal[
                        "visual_measurement_y"
                    ]
                )
            ],
            s=150,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            label=(
                "Terminal lock "
                f"f{int(first_terminal['frame_id'])}"
            ),
        )

    ax.set_title(
        (
            f"{route_name}: Visual-Motion-Gated Route LSTM v5\n"
            "No fixed speed; current/previous images decide motion phase"
        )
    )

    ax.set_xlabel(
        "Local X (m)"
    )

    ax.set_ylabel(
        "Local Y (m)"
    )

    ax.axis(
        "equal"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

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

    plt.close(
        fig
    )

    return path


def render_process_snapshots(
    route_name,
    rows,
    output_dir,
):
    count = min(
        int(
            config.PROCESS_SNAPSHOT_COUNT
        ),
        len(
            rows
        ),
    )

    indices = np.unique(
        np.linspace(
            0,
            len(
                rows
            )
            - 1,
            count,
        )
        .round()
        .astype(
            int
        )
    )

    columns = 3

    row_count = int(
        math.ceil(
            len(
                indices
            )
            / float(
                columns
            )
        )
    )

    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(
            16,
            4.8
            * row_count,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(
        -1
    )

    for plot_index, row_index in enumerate(
        indices
    ):
        row = rows.iloc[
            row_index
        ]

        ax = axes[
            plot_index
        ]

        points = {
            "GT": (
                float(
                    row[
                        "gt_x"
                    ]
                ),
                float(
                    row[
                        "gt_y"
                    ]
                ),
                "tab:green",
                "o",
            ),
            "Search center": (
                float(
                    row[
                        "search_center_x"
                    ]
                ),
                float(
                    row[
                        "search_center_y"
                    ]
                ),
                "0.45",
                "s",
            ),
            "Proposal": (
                float(
                    row[
                        "proposal_x"
                    ]
                ),
                float(
                    row[
                        "proposal_y"
                    ]
                ),
                "tab:blue",
                "x",
            ),
            "Polynomial": (
                float(
                    row[
                        "polynomial_x"
                    ]
                ),
                float(
                    row[
                        "polynomial_y"
                    ]
                ),
                "tab:cyan",
                "^",
            ),
            "Gated visual": (
                float(
                    row[
                        "visual_measurement_x"
                    ]
                ),
                float(
                    row[
                        "visual_measurement_y"
                    ]
                ),
                "tab:purple",
                "D",
            ),
            "Final": (
                float(
                    row[
                        "final_x"
                    ]
                ),
                float(
                    row[
                        "final_y"
                    ]
                ),
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
                [
                    x_value
                ],
                [
                    y_value
                ],
                color=color,
                marker=marker,
                s=(
                    120
                    if label
                    == "Final"
                    else 65
                ),
                label=label,
            )

        all_xy = np.asarray(
            [
                [
                    item[
                        0
                    ],
                    item[
                        1
                    ],
                ]
                for item
                in points.values()
            ],
            dtype=float,
        )

        center = np.mean(
            all_xy,
            axis=0,
        )

        span = max(
            np.ptp(
                all_xy[
                    :,
                    0
                ]
            ),
            np.ptp(
                all_xy[
                    :,
                    1
                ]
            ),
            45.0,
        )

        margin = (
            0.70
            * span
        )

        ax.set_xlim(
            center[
                0
            ]
            - margin,
            center[
                0
            ]
            + margin,
        )

        ax.set_ylim(
            center[
                1
            ]
            - margin,
            center[
                1
            ]
            + margin,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        phase = phase_name(
            row[
                "predicted_phase"
            ]
        )

        ax.set_title(
            (
                f"frame {int(row['frame_id'])} | "
                f"W{int(row['active_waypoint_from'])}"
                f"->W{int(row['active_waypoint_to'])}\n"
                f"{phase} | move={float(row['translation_probability']):.2f} "
                f"stop={float(row['stationary_probability']):.2f} "
                f"rot={float(row['rotation_probability']):.2f}\n"
                f"visual err={float(row['error_visual_m']):.1f}m | "
                f"final err={float(row['error_final_m']):.1f}m"
            ),
            fontsize=8.5,
        )

        if plot_index == 0:
            ax.legend(
                fontsize=7,
            )

    for index in range(
        len(
            indices
        ),
        len(
            axes
        ),
    ):
        axes[
            index
        ].axis(
            "off"
        )

    fig.suptitle(
        (
            f"{route_name}: current-image localization process\n"
            "image pair motion state -> visual proposal -> translation gate "
            "-> synchronized visual state -> position Kalman"
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

    plt.close(
        fig
    )

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
                        float(
                            map_height
                        )
                        / float(
                            source_height
                        )
                    )
                )
            ),
            int(
                config.VIDEO_WIDTH
                * 0.63
            ),
        )

        map_rgb = np.asarray(
            image.convert(
                "RGB"
            ).resize(
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

    width = int(
        config.VIDEO_WIDTH
    )

    height = int(
        config.VIDEO_HEIGHT
    )

    uav_width = (
        width
        - map_width
    )

    scale_x = (
        float(
            map_width
        )
        / float(
            source_width
        )
    )

    scale_y = (
        float(
            map_height
        )
        / float(
            source_height
        )
    )

    def xy_to_canvas(
        xy,
    ):
        source = xy_to_source_pixels(
            xy,
            dataset,
            origin_lat,
            origin_lon,
        )

        result = np.empty_like(
            source
        )

        result[
            :,
            0
        ] = (
            source[
                :,
                0
            ]
            * scale_x
            + uav_width
        )

        result[
            :,
            1
        ] = (
            source[
                :,
                1
            ]
            * scale_y
        )

        return result

    gt_xy = rows[
        [
            "gt_x",
            "gt_y",
        ]
    ].to_numpy(
        dtype=float
    )

    visual_xy = rows[
        [
            "visual_measurement_x",
            "visual_measurement_y",
        ]
    ].to_numpy(
        dtype=float
    )

    final_xy = rows[
        [
            "final_x",
            "final_y",
        ]
    ].to_numpy(
        dtype=float
    )

    polynomial_xy = rows[
        [
            "polynomial_x",
            "polynomial_y",
        ]
    ].to_numpy(
        dtype=float
    )

    proposal_xy = rows[
        [
            "proposal_x",
            "proposal_y",
        ]
    ].to_numpy(
        dtype=float
    )

    waypoint_xy = np.asarray(
        [
            [
                waypoint[
                    "x"
                ],
                waypoint[
                    "y"
                ],
            ]
            for waypoint
            in waypoints
        ],
        dtype=float,
    )

    gt_canvas = xy_to_canvas(
        gt_xy
    )

    visual_canvas = xy_to_canvas(
        visual_xy
    )

    final_canvas = xy_to_canvas(
        final_xy
    )

    polynomial_canvas = xy_to_canvas(
        polynomial_xy
    )

    proposal_canvas = xy_to_canvas(
        proposal_xy
    )

    waypoint_canvas = xy_to_canvas(
        waypoint_xy
    )

    output_path = (
        output_dir
        / f"{route_name}_synchronized_inference.mp4"
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(
            config.VIDEO_FPS
        ),
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot open video writer: {output_path}"
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
                str(
                    row[
                        "image_path"
                    ]
                ),
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
                    WAYPOINT_COLOR,
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    2,
                )

                cv2.putText(
                    canvas,
                    (
                        f"W"
                        f"{waypoints[waypoint_index]['order']}"
                    ),
                    (
                        int(
                            point[
                                0
                            ]
                        )
                        + 5,
                        int(
                            point[
                                1
                            ]
                        )
                        - 5,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    WAYPOINT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            end_index = (
                row_index
                + 1
            )

            draw_history(
                canvas,
                gt_canvas[
                    :end_index
                ],
                GT_COLOR,
                2,
            )

            draw_history(
                canvas,
                visual_canvas[
                    :end_index
                ],
                VISUAL_COLOR,
                2,
            )

            draw_history(
                canvas,
                final_canvas[
                    :end_index
                ],
                FINAL_COLOR,
                3,
            )

            draw_marker(
                canvas,
                gt_canvas[
                    row_index
                ],
                GT_COLOR,
                cv2.MARKER_CROSS,
                21,
                3,
            )

            draw_marker(
                canvas,
                proposal_canvas[
                    row_index
                ],
                (
                    255,
                    160,
                    70,
                ),
                cv2.MARKER_SQUARE,
                15,
                2,
            )

            draw_marker(
                canvas,
                polynomial_canvas[
                    row_index
                ],
                POLY_COLOR,
                cv2.MARKER_TRIANGLE_UP,
                18,
                2,
            )

            draw_marker(
                canvas,
                visual_canvas[
                    row_index
                ],
                VISUAL_COLOR,
                cv2.MARKER_DIAMOND,
                20,
                3,
            )

            draw_marker(
                canvas,
                final_canvas[
                    row_index
                ],
                FINAL_COLOR,
                cv2.MARKER_STAR,
                25,
                3,
            )

            frame_id = int(
                row[
                    "frame_id"
                ]
            )

            active_from = int(
                row[
                    "active_waypoint_from"
                ]
            )

            active_to = int(
                row[
                    "active_waypoint_to"
                ]
            )

            phase = phase_name(
                row[
                    "predicted_phase"
                ]
            )

            cv2.rectangle(
                canvas,
                (
                    0,
                    0,
                ),
                (
                    width,
                    58,
                ),
                BLACK,
                -1,
            )

            top_text = (
                f"{route_name.upper()} | frame={frame_id} | "
                f"W{active_from}->W{active_to} | {phase} | "
                f"move={float(row['translation_probability']):.2f} "
                f"stop={float(row['stationary_probability']):.2f} "
                f"rot={float(row['rotation_probability']):.2f} | "
                f"err={float(row['error_final_m']):.1f}m"
            )

            cv2.putText(
                canvas,
                top_text,
                (
                    14,
                    34,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            status_lines = [
                (
                    "LEFT = exact current UAV frame; "
                    "RIGHT = synchronized localization"
                ),
                (
                    f"ECC shift={float(row['ecc_center_shift_norm']):.4f} "
                    f"rot={float(row['ecc_rotation_deg']):.1f}deg "
                    f"corr={float(row['ecc_correlation']):.2f}"
                ),
                (
                    f"observed step parallel="
                    f"{float(row['observed_delta_parallel']):.2f}m "
                    f"cross={float(row['observed_delta_cross']):.2f}m"
                ),
                (
                    "Purple = current IMAGE-gated localization; "
                    "cyan = polynomial prior only"
                ),
                (
                    "Orange = position-only Kalman; "
                    "no velocity state / no fixed speed"
                ),
                (
                    f"terminal_locked={int(row['terminal_locked'])}"
                ),
            ]

            box_y = (
                height
                - 205
            )

            cv2.rectangle(
                canvas,
                (
                    10,
                    box_y,
                ),
                (
                    uav_width
                    - 10,
                    height
                    - 10,
                ),
                BLACK,
                -1,
            )

            for line_index, line in enumerate(
                status_lines
            ):
                cv2.putText(
                    canvas,
                    line,
                    (
                        20,
                        box_y
                        + 27
                        + line_index
                        * 28,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            writer.write(
                canvas
            )

            if (
                row_index
                + 1
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
    route_index = (
        config.ROUTE_NAMES.index(
            route_name
        )
    )

    return RouteDataset(
        Path(
            config.ROUTE_ROOTS[
                route_index
            ]
        ),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def render_route(
    route_name,
):
    csv_path = (
        config.OUTPUT_DIR
        / (
            f"{route_name}_"
            "visual_motion_lstm_frames.csv"
        )
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    rows = pd.read_csv(
        csv_path
    )

    checkpoint = torch.load(
        config.VISUAL_CHECKPOINT,
        map_location="cpu",
    )

    origin_lat = float(
        checkpoint[
            "origin_lat"
        ]
    )

    origin_lon = float(
        checkpoint[
            "origin_lon"
        ]
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

    process = render_process_snapshots(
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

    print(
        "overview:",
        overview,
        flush=True,
    )

    print(
        "process:",
        process,
        flush=True,
    )

    print(
        "video:",
        video,
        flush=True,
    )


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
        [
            "route_B",
            "route_C",
        ]
        if args.route
        == "all"
        else [
            args.route
        ]
    )

    for route_name in routes:
        render_route(
            route_name
        )


if __name__ == "__main__":
    main()
