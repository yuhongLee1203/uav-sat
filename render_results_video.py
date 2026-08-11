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


GT_COLOR = (0, 255, 0)       # BGR bright green
PRED_COLOR = (255, 0, 255)   # BGR bright magenta


def meters_to_latlon(x_m, y_m, origin_lat, origin_lon):
    radius = 6378137.0
    lat = (
        float(origin_lat)
        + math.degrees(float(y_m) / radius)
    )
    lon_scale = (
        radius
        * math.cos(math.radians(float(origin_lat)))
    )
    lon = (
        float(origin_lon)
        + math.degrees(float(x_m) / lon_scale)
    )
    return lat, lon


def xy_to_source_pixels(
    xy,
    dataset,
    origin_lat,
    origin_lon,
):
    rows = []

    for x_m, y_m in np.asarray(
        xy,
        dtype=np.float64,
    ):
        lat, lon = meters_to_latlon(
            x_m,
            y_m,
            origin_lat,
            origin_lon,
        )
        px, py = dataset.mapper.latlon_to_pixel(
            lat,
            lon,
        )
        rows.append(
            [float(px), float(py)]
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def contain_image(image, width, height):
    src_h, src_w = image.shape[:2]

    scale = min(
        float(width) / float(src_w),
        float(height) / float(src_h),
    )

    dst_w = max(
        1,
        int(round(src_w * scale)),
    )
    dst_h = max(
        1,
        int(round(src_h * scale)),
    )

    resized = cv2.resize(
        image,
        (dst_w, dst_h),
        interpolation=(
            cv2.INTER_AREA
            if scale <= 1.0
            else cv2.INTER_LINEAR
        ),
    )

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    ox = (width - dst_w) // 2
    oy = (height - dst_h) // 2

    panel[
        oy : oy + dst_h,
        ox : ox + dst_w,
    ] = resized

    return panel


def render_video(
    route_name,
    rows,
    dataset,
    origin_lat,
    origin_lon,
    output_dir,
):
    with Image.open(config.SAT_IMAGE) as image:
        source_width, source_height = image.size

        map_height = int(config.VIDEO_HEIGHT)
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
            int(config.VIDEO_WIDTH * 0.63),
        )

        map_rgb = np.asarray(
            image.convert("RGB").resize(
                (map_width, map_height),
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
            source[:, 0] * scale_x
            + uav_width
        )
        result[:, 1] = (
            source[:, 1] * scale_y
        )
        return result

    gt_xy = rows[
        ["gt_x", "gt_y"]
    ].to_numpy(dtype=float)

    pred_xy = rows[
        ["final_x", "final_y"]
    ].to_numpy(dtype=float)

    gt_canvas = xy_to_canvas(gt_xy)
    pred_canvas = xy_to_canvas(pred_xy)

    output_path = (
        output_dir
        / (
            route_name
            + "_synchronized_inference.mp4"
        )
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.VIDEO_FPS),
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "cannot create video: %s"
            % output_path
        )

    try:
        for row_index, row in rows.iterrows():
            canvas = np.zeros(
                (height, width, 3),
                dtype=np.uint8,
            )
            canvas[:, uav_width:] = map_panel

            image = cv2.imread(
                str(row["image_path"]),
                cv2.IMREAD_COLOR,
            )

            if image is not None:
                canvas[:, :uav_width] = contain_image(
                    image,
                    uav_width,
                    height,
                )

            if row_index > 0:
                cv2.polylines(
                    canvas,
                    [
                        gt_canvas[
                            : row_index + 1
                        ].astype(np.int32)
                    ],
                    False,
                    GT_COLOR,
                    3,
                    cv2.LINE_AA,
                )
                cv2.polylines(
                    canvas,
                    [
                        pred_canvas[
                            : row_index + 1
                        ].astype(np.int32)
                    ],
                    False,
                    PRED_COLOR,
                    3,
                    cv2.LINE_AA,
                )

            gt_point = gt_canvas[row_index]
            pred_point = pred_canvas[row_index]

            cv2.drawMarker(
                canvas,
                (
                    int(gt_point[0]),
                    int(gt_point[1]),
                ),
                GT_COLOR,
                cv2.MARKER_CROSS,
                20,
                3,
                cv2.LINE_AA,
            )

            cv2.drawMarker(
                canvas,
                (
                    int(pred_point[0]),
                    int(pred_point[1]),
                ),
                PRED_COLOR,
                cv2.MARKER_STAR,
                22,
                3,
                cv2.LINE_AA,
            )

            # Do not draw a direction before the temporal image model has a
            # meaningful nonzero motion vector.
            if (
                int(row["heading_valid"]) == 1
                and np.isfinite(
                    float(
                        row[
                            "estimated_heading_deg_enu"
                        ]
                    )
                )
            ):
                heading_rad = math.radians(
                    float(
                        row[
                            "estimated_heading_deg_enu"
                        ]
                    )
                )

                arrow_m = float(
                    config.HEADING_RENDER_ARROW_LENGTH_M
                )

                tip_world = np.asarray(
                    [
                        [
                            float(row["final_x"])
                            + arrow_m
                            * math.cos(heading_rad),
                            float(row["final_y"])
                            + arrow_m
                            * math.sin(heading_rad),
                        ]
                    ],
                    dtype=np.float64,
                )

                # Correct map orientation: transform the WORLD endpoint through
                # the same mapper instead of assuming screen x/y == ENU x/y.
                tip_canvas = xy_to_canvas(
                    tip_world
                )[0]

                cv2.arrowedLine(
                    canvas,
                    (
                        int(pred_point[0]),
                        int(pred_point[1]),
                    ),
                    (
                        int(tip_canvas[0]),
                        int(tip_canvas[1]),
                    ),
                    PRED_COLOR,
                    3,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

            lines = [
                "GT = GREEN    FINAL PRED = MAGENTA",
                "frame=%d"
                % int(row["frame_id"]),
                "RNN motion=(%.2f, %.2f)m |d|=%.2fm stopP=%.2f"
                % (
                    float(row["rnn_motion_dx_m"]),
                    float(row["rnn_motion_dy_m"]),
                    float(row["rnn_motion_magnitude_m"]),
                    float(row["rnn_stop_probability"]),
                ),
                "heading=%s"
                % (
                    (
                        "%.1f deg ENU"
                        % float(
                            row[
                                "estimated_heading_deg_enu"
                            ]
                        )
                    )
                    if int(row["heading_valid"]) == 1
                    else "undefined / stopped"
                ),
                "36-patch pmax=%.2f margin=%.2f  error=%.2fm"
                % (
                    float(
                        row[
                            "candidate_probability_max"
                        ]
                    ),
                    float(
                        row[
                            "candidate_probability_margin"
                        ]
                    ),
                    float(row["error_final_m"]),
                ),
            ]

            for line_index, text in enumerate(lines):
                cv2.putText(
                    canvas,
                    text,
                    (
                        18,
                        30 + line_index * 29,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(canvas)

    finally:
        writer.release()

    return output_path


def render_route(route_name):
    csv_path = (
        config.OUTPUT_DIR
        / (
            route_name
            + "_stable_visual_inertial_rnn_frames.csv"
        )
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            "Inference CSV missing: %s"
            % csv_path
        )

    rows = pd.read_csv(csv_path)

    if rows.empty:
        raise RuntimeError(
            "CSV is empty: %s"
            % csv_path
        )

    route_index = config.ROUTE_NAMES.index(
        route_name
    )

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

    dataset = RouteDataset(
        Path(
            config.ROUTE_ROOTS[
                route_index
            ]
        ),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    video = render_video(
        route_name=route_name,
        rows=rows,
        dataset=dataset,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        output_dir=config.OUTPUT_DIR,
    )

    print(
        "rendered:",
        video,
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        choices=["route_B", "route_C", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.route == "all":
        for route_name in [
            "route_B",
            "route_C",
        ]:
            render_route(route_name)
    else:
        render_route(args.route)


if __name__ == "__main__":
    main()
