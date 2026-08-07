
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from data import RouteDataset

VIDEO_WIDTH = 1600
VIDEO_HEIGHT = 900
DEFAULT_FPS = 8.0
EARTH_RADIUS_M = 6378137.0

# BGR colours.
GT_COLOR = (50, 220, 70)
RTL_COLOR = (220, 70, 220)
TEXT_COLOR = (242, 242, 242)
BACKGROUND_COLOR = (0, 0, 0)

REQUIRED_COLUMNS = {
    "frame_id",
    "gt_x",
    "gt_y",
    "temporal_x",
    "temporal_y",
}


def meters_to_latlon(
    x_meter: float,
    y_meter: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """Exact inverse of data.meters_from_latlon for the fixed route origin."""
    origin_lat_rad = math.radians(float(origin_lat))
    lat = float(origin_lat) + math.degrees(float(y_meter) / EARTH_RADIUS_M)
    lon_scale = EARTH_RADIUS_M * math.cos(origin_lat_rad)
    if abs(lon_scale) < 1e-9:
        raise RuntimeError("Invalid route origin latitude for metre conversion")
    lon = float(origin_lon) + math.degrees(float(x_meter) / lon_scale)
    return lat, lon


def xy_to_source_pixels(
    xy: np.ndarray,
    dataset: RouteDataset,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    """Convert local ENU-like metre coordinates through the calibrated mapper."""
    pixels = []
    for x_meter, y_meter in np.asarray(xy, dtype=np.float64):
        lat, lon = meters_to_latlon(
            x_meter,
            y_meter,
            origin_lat,
            origin_lon,
        )
        pixel_x, pixel_y = dataset.mapper.latlon_to_pixel(lat, lon)
        pixels.append((float(pixel_x), float(pixel_y)))
    return np.asarray(pixels, dtype=np.float64)


def contain_image(
    image: np.ndarray,
    width: int,
    height: int,
    fill: Tuple[int, int, int] = BACKGROUND_COLOR,
) -> np.ndarray:
    """Letterbox only the UAV image while preserving its aspect ratio."""
    source_height, source_width = image.shape[:2]
    scale = min(float(width) / source_width, float(height) / source_height)
    destination_width = max(1, int(round(source_width * scale)))
    destination_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (destination_width, destination_height),
        interpolation=interpolation,
    )
    panel = np.full((height, width, 3), fill, dtype=np.uint8)
    offset_x = (width - destination_width) // 2
    offset_y = (height - destination_height) // 2
    panel[
        offset_y : offset_y + destination_height,
        offset_x : offset_x + destination_width,
    ] = resized
    return panel


def draw_trace(
    canvas: np.ndarray,
    points: np.ndarray,
    colour: Tuple[int, int, int],
    panel_x: int,
    panel_y: int,
    panel_width: int,
    panel_height: int,
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    previous = None
    maximum_step = 0.18 * float(np.hypot(panel_width, panel_height))
    for point in points:
        x, y = int(round(point[0])), int(round(point[1]))
        valid = (
            panel_x <= x < panel_x + panel_width
            and panel_y <= y < panel_y + panel_height
        )
        if valid and previous is not None:
            if float(np.hypot(x - previous[0], y - previous[1])) <= maximum_step:
                cv2.line(
                    canvas,
                    previous,
                    (x, y),
                    colour,
                    thickness,
                    cv2.LINE_AA,
                )
        previous = (x, y) if valid else None


def point_in_panel(
    point: Sequence[float],
    panel_x: int,
    panel_y: int,
    panel_width: int,
    panel_height: int,
) -> bool:
    return (
        panel_x <= point[0] < panel_x + panel_width
        and panel_y <= point[1] < panel_y + panel_height
    )


def draw_marker(
    canvas: np.ndarray,
    point: Sequence[float],
    colour: Tuple[int, int, int],
    marker: int,
    size: int = 15,
) -> None:
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.drawMarker(canvas, (x, y), colour, marker, size, 2, cv2.LINE_AA)


def add_top_bar(canvas: np.ndarray, text: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 43), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def add_legend(canvas: np.ndarray, x: int, y: int) -> None:
    entries = [
        ("GT", GT_COLOR),
        ("RTL-CRF final", RTL_COLOR),
    ]
    cv2.rectangle(canvas, (x, y), (x + 250, y + 74), (0, 0, 0), -1)
    for index, (label, colour) in enumerate(entries):
        row_y = y + 25 + 25 * index
        cv2.line(canvas, (x + 14, row_y - 5), (x + 42, row_y - 5), colour, 3)
        cv2.putText(
            canvas,
            label,
            (x + 52, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def open_video_writer(
    path: Path,
    fps: float,
    size: Tuple[int, int],
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open MP4 writer for {path}")


def direct_gt_pixels(rows: pd.DataFrame, sample_by_id) -> np.ndarray:
    pixels = []
    missing = []
    for frame_id in rows["frame_id"].astype(int).tolist():
        sample = sample_by_id.get(frame_id)
        if sample is None:
            missing.append(frame_id)
            continue
        pixels.append((float(sample["pixel_x"]), float(sample["pixel_y"])))
    if missing:
        raise RuntimeError(
            "Evaluation CSV contains frame IDs that are absent from RouteDataset: "
            f"{missing[:10]}"
        )
    return np.asarray(pixels, dtype=np.float64)


def verify_coordinate_mapping(
    converted_gt: np.ndarray,
    exact_gt: np.ndarray,
    route_name: str,
) -> None:
    errors = np.linalg.norm(converted_gt - exact_gt, axis=1)
    median_error = float(np.median(errors))
    maximum_error = float(np.max(errors))
    print(
        f"{route_name}: map-coordinate verification "
        f"median={median_error:.4f}px max={maximum_error:.4f}px",
        flush=True,
    )
    if median_error > 1.5 or maximum_error > 5.0:
        raise RuntimeError(
            f"{route_name}: local XY and georeferenced map disagree "
            f"(median {median_error:.2f}px, max {maximum_error:.2f}px). "
            "Video rendering stopped instead of drawing a flipped trajectory."
        )


def render_route(
    root: Path,
    name: str,
    output_dir: Path,
    fps: float = DEFAULT_FPS,
    video_width: int = VIDEO_WIDTH,
    video_height: int = VIDEO_HEIGHT,
) -> Path:
    result_csv = config.OUTPUT_DIR / f"{name}_robust_frames.csv"
    if not result_csv.exists():
        raise FileNotFoundError(
            f"Missing {result_csv}; run robust_tracker.py --mode eval first"
        )

    rows = pd.read_csv(result_csv)
    missing_columns = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing_columns:
        raise RuntimeError(
            f"{result_csv} is not an RTL-CRF result CSV; "
            f"missing columns: {missing_columns}"
        )
    if rows.empty:
        raise RuntimeError(f"No inference rows found in {result_csv}")

    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    dataset = RouteDataset(
        root,
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    sample_by_id = {
        int(sample["frame_id"]): sample for sample in dataset.samples
    }

    gt_xy = rows[["gt_x", "gt_y"]].to_numpy(dtype=np.float64)
    temporal_xy = rows[["temporal_x", "temporal_y"]].to_numpy(dtype=np.float64)

    # GT uses exact source pixels from RouteDataset. Predictions use the same
    # calibrated meter -> lat/lon -> source-pixel chain as the dataset.
    gt_source_px = direct_gt_pixels(rows, sample_by_id)
    converted_gt_px = xy_to_source_pixels(
        gt_xy,
        dataset,
        origin_lat,
        origin_lon,
    )
    verify_coordinate_mapping(converted_gt_px, gt_source_px, name)
    temporal_source_px = xy_to_source_pixels(
        temporal_xy,
        dataset,
        origin_lat,
        origin_lon,
    )

    # Restore the earlier reference-style layout: the orthomosaic occupies the
    # full video height and is resized without flipping or stretching.
    with Image.open(config.SAT_IMAGE) as image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        map_width = int(round(source_width * (float(video_height) / source_height)))
        map_rgb = np.asarray(
            image.resize((map_width, video_height), Image.Resampling.LANCZOS)
        )
    map_panel = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
    uav_width = video_width - map_width
    if uav_width < 420:
        raise RuntimeError(
            "The full satellite map is too wide for this video width. "
            "Increase --width while keeping the map aspect ratio."
        )

    scale_x = float(map_width) / source_width
    scale_y = float(video_height) / source_height

    def source_to_canvas(points: np.ndarray) -> np.ndarray:
        result = np.empty_like(points, dtype=np.float64)
        result[:, 0] = points[:, 0] * scale_x + uav_width
        result[:, 1] = points[:, 1] * scale_y
        return result

    gt_panel = source_to_canvas(gt_source_px)
    temporal_panel = source_to_canvas(temporal_source_px)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_rtl_crf_inference.mp4"
    writer = open_video_writer(
        output_path,
        fps,
        (video_width, video_height),
    )

    try:
        for row_index, row in rows.iterrows():
            canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            canvas[:, uav_width:] = map_panel

            frame_id = int(row["frame_id"])
            sample = sample_by_id.get(frame_id)
            if sample is not None:
                uav_image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
                if uav_image is not None:
                    canvas[:, :uav_width] = contain_image(
                        uav_image,
                        uav_width,
                        video_height,
                    )

            end = row_index + 1
            draw_trace(
                canvas,
                gt_panel[:end],
                GT_COLOR,
                uav_width,
                0,
                map_width,
                video_height,
                2,
            )
            draw_trace(
                canvas,
                temporal_panel[:end],
                RTL_COLOR,
                uav_width,
                0,
                map_width,
                video_height,
                3,
            )

            marker_data = [
                (gt_panel[row_index], GT_COLOR, cv2.MARKER_CROSS),
                (temporal_panel[row_index], RTL_COLOR, cv2.MARKER_STAR),
            ]
            for point, colour, marker in marker_data:
                if point_in_panel(
                    point,
                    uav_width,
                    0,
                    map_width,
                    video_height,
                ):
                    draw_marker(canvas, point, colour, marker)

            temporal_error = float(
                np.linalg.norm(temporal_xy[row_index] - gt_xy[row_index])
            )
            add_top_bar(
                canvas,
                (
                    f"{name.upper()}  frame {frame_id}  "
                    f"RTL-CRF final error {temporal_error:.1f} m"
                ),
            )
            add_legend(canvas, uav_width + 12, video_height - 88)
            writer.write(canvas)

            if (row_index + 1) % 250 == 0:
                print(
                    f"rendering {name}: {row_index + 1}/{len(rows)}",
                    flush=True,
                )
    finally:
        writer.release()

    print(f"video written: {output_path}", flush=True)
    return output_path


def selected_route_pairs(route: str) -> List[Tuple[Path, str]]:
    pairs = [
        (Path(root), name)
        for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES)
    ]
    if route == "all":
        eval_names = set(config.EVAL_ROUTE_NAMES)
        return [
            (root, name)
            for root, name in pairs
            if name in eval_names
        ]
    return [(root, name) for root, name in pairs if name == route]


def render_routes(
    route_pairs: Sequence[Tuple[Path, str]],
    fps: float = DEFAULT_FPS,
    video_width: int = VIDEO_WIDTH,
    video_height: int = VIDEO_HEIGHT,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    destination = output_dir or (config.OUTPUT_DIR / "inference_videos")
    output_paths = []
    for root, name in route_pairs:
        output_paths.append(
            render_route(
                root=Path(root),
                name=name,
                output_dir=destination,
                fps=fps,
                video_width=video_width,
                video_height=video_height,
            )
        )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        choices=[*config.ROUTE_NAMES, "all"],
        default="all",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "inference_videos",
    )
    args = parser.parse_args()
    render_routes(
        selected_route_pairs(args.route),
        fps=float(args.fps),
        video_width=int(args.width),
        video_height=int(args.height),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()