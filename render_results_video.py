from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Video colours are BGR.
GT_COLOR = (50, 220, 70)
RTL_COLOR = (220, 70, 220)
TEXT_COLOR = (242, 242, 242)
BACKGROUND_COLOR = (0, 0, 0)

REQUIRED_COLUMNS = {
    "frame_id",
    "gt_x", "gt_y",
    "hardms_x", "hardms_y",
    "temporal_x", "temporal_y",
}


def meters_to_latlon(
    x_meter: float,
    y_meter: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """Exact inverse of the local meter conversion used by the dataset."""
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
            "Evaluation CSV contains frame IDs absent from RouteDataset: "
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
            "Rendering stopped instead of drawing a flipped trajectory."
        )


def contain_image(
    image: np.ndarray,
    width: int,
    height: int,
    fill: Tuple[int, int, int] = BACKGROUND_COLOR,
) -> np.ndarray:
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


def load_route_result(root: Path, name: str):
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
    hardms_xy = rows[["hardms_x", "hardms_y"]].to_numpy(dtype=np.float64)
    temporal_xy = rows[["temporal_x", "temporal_y"]].to_numpy(dtype=np.float64)

    gt_source_px = direct_gt_pixels(rows, sample_by_id)
    converted_gt_px = xy_to_source_pixels(
        gt_xy, dataset, origin_lat, origin_lon
    )
    verify_coordinate_mapping(converted_gt_px, gt_source_px, name)
    hardms_source_px = xy_to_source_pixels(
        hardms_xy, dataset, origin_lat, origin_lon
    )
    temporal_source_px = xy_to_source_pixels(
        temporal_xy, dataset, origin_lat, origin_lon
    )

    return {
        "rows": rows,
        "dataset": dataset,
        "sample_by_id": sample_by_id,
        "gt_xy": gt_xy,
        "hardms_xy": hardms_xy,
        "temporal_xy": temporal_xy,
        "gt_px": gt_source_px,
        "hardms_px": hardms_source_px,
        "temporal_px": temporal_source_px,
    }


def _metric_summary(prediction: np.ndarray, gt: np.ndarray):
    error = np.linalg.norm(prediction - gt, axis=1)
    if len(prediction) > 1:
        pred_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = float(np.percentile(gt_step_length, 99)) + float(
            config.JUMP_TOLERANCE_M
        )
        jump_rate = float(
            (np.linalg.norm(pred_step, axis=1) > jump_threshold).mean() * 100.0
        )
    else:
        jump_rate = 0.0
    return float(error.mean()), float(np.percentile(error, 90)), jump_rate


def _crop_bounds(
    source_width: int,
    source_height: int,
    points: np.ndarray,
    margin: int,
) -> Tuple[int, int, int, int]:
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return 0, 0, source_width, source_height
    left = max(0, int(math.floor(points[:, 0].min())) - margin)
    right = min(source_width, int(math.ceil(points[:, 0].max())) + margin)
    top = max(0, int(math.floor(points[:, 1].min())) - margin)
    bottom = min(source_height, int(math.ceil(points[:, 1].max())) + margin)
    if right <= left or bottom <= top:
        return 0, 0, source_width, source_height
    return left, top, right, bottom


def _localize_pixels(points: np.ndarray, box: Tuple[int, int, int, int]):
    left, top, _, _ = box
    result = np.asarray(points, dtype=np.float64).copy()
    result[:, 0] -= left
    result[:, 1] -= top
    return result


def save_full_route_figure(
    sat_image: Image.Image,
    name: str,
    result,
    output_dir: Path,
) -> Path:
    gt_xy = result["gt_xy"]
    hardms_xy = result["hardms_xy"]
    temporal_xy = result["temporal_xy"]
    gt_px = result["gt_px"]
    hardms_px = result["hardms_px"]
    temporal_px = result["temporal_px"]

    all_points = np.concatenate([gt_px, hardms_px, temporal_px], axis=0)
    box = _crop_bounds(sat_image.width, sat_image.height, all_points, margin=350)
    crop = sat_image.crop(box)
    local_gt = _localize_pixels(gt_px, box)
    local_hard = _localize_pixels(hardms_px, box)
    local_rtl = _localize_pixels(temporal_px, box)

    hard_mle, hard_p90, hard_jump = _metric_summary(hardms_xy, gt_xy)
    rtl_mle, rtl_p90, rtl_jump = _metric_summary(temporal_xy, gt_xy)

    figure, axis = plt.subplots(figsize=(14, 10))
    axis.imshow(crop)
    axis.plot(
        local_gt[:, 0], local_gt[:, 1],
        linewidth=2.4, label="GT trajectory",
    )
    axis.plot(
        local_hard[:, 0], local_hard[:, 1],
        linewidth=1.2, linestyle="--", alpha=0.75,
        label=(
            f"Fixed HardMS (single-frame): MLE {hard_mle:.2f} m, "
            f"P90 {hard_p90:.2f} m, Jump {hard_jump:.2f}%"
        ),
    )
    axis.plot(
        local_rtl[:, 0], local_rtl[:, 1],
        linewidth=2.2,
        label=(
            f"RTL-CRF final: MLE {rtl_mle:.2f} m, "
            f"P90 {rtl_p90:.2f} m, Jump {rtl_jump:.2f}%"
        ),
    )
    axis.scatter(local_gt[0, 0], local_gt[0, 1], marker="o", s=70, label="Start")
    axis.scatter(local_gt[-1, 0], local_gt[-1, 1], marker="*", s=130, label="End")
    axis.set_title(
        f"{name.upper()} — complete unseen-route localization\n"
        "Single-frame HardMS vs. RTL-CRF temporal localization"
    )
    axis.legend(loc="best", fontsize=9)
    axis.axis("off")
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_full_localization.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"full-route figure written: {path}", flush=True)
    return path


def _select_jump_events(
    gt_xy: np.ndarray,
    hardms_xy: np.ndarray,
    temporal_xy: np.ndarray,
    number: int,
    min_separation: int,
) -> List[int]:
    if len(gt_xy) < 2 or number <= 0:
        return []

    gt_step = np.diff(gt_xy, axis=0)
    hard_step = np.diff(hardms_xy, axis=0)
    rtl_step = np.diff(temporal_xy, axis=0)

    hard_rpe = np.linalg.norm(hard_step - gt_step, axis=1)
    rtl_rpe = np.linalg.norm(rtl_step - gt_step, axis=1)
    improvement = hard_rpe - rtl_rpe

    # Prefer transitions where the single-frame result jumps substantially more
    # than RTL-CRF.  The index is shifted by +1 because a transition i-1 -> i
    # is visualized at frame i.
    priority = np.argsort(improvement)[::-1]
    selected: List[int] = []
    for transition_index in priority.tolist():
        frame_index = int(transition_index + 1)
        if improvement[transition_index] <= 0:
            continue
        if all(abs(frame_index - old) >= min_separation for old in selected):
            selected.append(frame_index)
        if len(selected) >= number:
            return selected

    # If temporal separation leaves fewer than requested, fill with the largest
    # single-frame transition errors so the script still writes exactly N plots.
    fallback = np.argsort(hard_rpe)[::-1]
    for transition_index in fallback.tolist():
        frame_index = int(transition_index + 1)
        if frame_index not in selected:
            selected.append(frame_index)
        if len(selected) >= number:
            break
    return selected


def save_zoom_jump_figures(
    sat_image: Image.Image,
    name: str,
    result,
    output_dir: Path,
    number: int = 15,
    half_window: int = 15,
) -> List[Path]:
    rows = result["rows"]
    gt_xy = result["gt_xy"]
    hardms_xy = result["hardms_xy"]
    temporal_xy = result["temporal_xy"]
    gt_px = result["gt_px"]
    hardms_px = result["hardms_px"]
    temporal_px = result["temporal_px"]

    selected = _select_jump_events(
        gt_xy,
        hardms_xy,
        temporal_xy,
        number=number,
        min_separation=max(5, half_window),
    )

    gt_step = np.diff(gt_xy, axis=0)
    hard_step = np.diff(hardms_xy, axis=0)
    rtl_step = np.diff(temporal_xy, axis=0)
    hard_rpe = np.linalg.norm(hard_step - gt_step, axis=1)
    rtl_rpe = np.linalg.norm(rtl_step - gt_step, axis=1)

    paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, center in enumerate(selected, start=1):
        start = max(0, center - half_window)
        end = min(len(rows), center + half_window + 1)
        points = np.concatenate(
            [gt_px[start:end], hardms_px[start:end], temporal_px[start:end]],
            axis=0,
        )
        box = _crop_bounds(
            sat_image.width,
            sat_image.height,
            points,
            margin=220,
        )
        crop = sat_image.crop(box)
        local_gt = _localize_pixels(gt_px[start:end], box)
        local_hard = _localize_pixels(hardms_px[start:end], box)
        local_rtl = _localize_pixels(temporal_px[start:end], box)

        event_local = center - start
        frame_id = int(rows.iloc[center]["frame_id"])
        previous_frame_id = int(rows.iloc[max(0, center - 1)]["frame_id"])
        event_hard_rpe = float(hard_rpe[center - 1]) if center > 0 else 0.0
        event_rtl_rpe = float(rtl_rpe[center - 1]) if center > 0 else 0.0

        figure, axis = plt.subplots(figsize=(10, 9))
        axis.imshow(crop)
        axis.plot(
            local_gt[:, 0], local_gt[:, 1],
            marker="o", markersize=3.0, linewidth=2.2,
            label="GT",
        )
        axis.plot(
            local_hard[:, 0], local_hard[:, 1],
            marker="s", markersize=3.0, linewidth=1.3, linestyle="--",
            label="Fixed HardMS (single-frame)",
        )
        axis.plot(
            local_rtl[:, 0], local_rtl[:, 1],
            marker="o", markersize=3.0, linewidth=2.3,
            label="RTL-CRF final",
        )

        if 0 <= event_local < len(local_gt):
            axis.scatter(
                local_hard[event_local, 0], local_hard[event_local, 1],
                marker="X", s=130, zorder=5,
                label="Selected HardMS jump frame",
            )
            axis.scatter(
                local_rtl[event_local, 0], local_rtl[event_local, 1],
                marker="*", s=150, zorder=6,
                label="RTL-CRF at same frame",
            )

        axis.set_title(
            f"{name.upper()} zoom {rank:02d} — frames {previous_frame_id} -> {frame_id}\n"
            f"single-frame transition error {event_hard_rpe:.2f} m  |  "
            f"RTL-CRF transition error {event_rtl_rpe:.2f} m"
        )
        axis.legend(loc="best", fontsize=8)
        axis.axis("off")
        figure.tight_layout()

        path = output_dir / f"{name}_jump_comparison_{rank:02d}.png"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
        print(f"zoom comparison written: {path}", flush=True)

    return paths


def render_video(
    sat_image: Image.Image,
    name: str,
    result,
    output_dir: Path,
    fps: float,
    video_width: int,
    video_height: int,
) -> Path:
    rows = result["rows"]
    sample_by_id = result["sample_by_id"]
    gt_xy = result["gt_xy"]
    temporal_xy = result["temporal_xy"]
    gt_source_px = result["gt_px"]
    temporal_source_px = result["temporal_px"]

    source_width, source_height = sat_image.size
    map_width = int(round(source_width * (float(video_height) / source_height)))
    map_rgb = np.asarray(
        sat_image.resize((map_width, video_height), Image.Resampling.LANCZOS)
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
        result_px = np.empty_like(points, dtype=np.float64)
        result_px[:, 0] = points[:, 0] * scale_x + uav_width
        result_px[:, 1] = points[:, 1] * scale_y
        return result_px

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


def render_route(
    root: Path,
    name: str,
    output_dir: Path,
    fps: float,
    video_width: int,
    video_height: int,
    num_zoom: int,
    zoom_half_window: int,
    no_video: bool,
) -> List[Path]:
    result = load_route_result(root, name)
    with Image.open(config.SAT_IMAGE) as image:
        sat_image = image.convert("RGB")

    figure_dir = output_dir / "figures"
    paths: List[Path] = []
    paths.append(
        save_full_route_figure(
            sat_image,
            name,
            result,
            figure_dir,
        )
    )
    paths.extend(
        save_zoom_jump_figures(
            sat_image,
            name,
            result,
            figure_dir,
            number=num_zoom,
            half_window=zoom_half_window,
        )
    )

    if not no_video:
        paths.append(
            render_video(
                sat_image,
                name,
                result,
                output_dir,
                fps,
                video_width,
                video_height,
            )
        )
    return paths


def selected_route_pairs(route: str) -> List[Tuple[Path, str]]:
    pairs = [
        (Path(root), name)
        for root, name in zip(config.ROUTE_ROOTS, config.ROUTE_NAMES)
    ]
    if route == "all":
        eval_names = set(config.EVAL_ROUTE_NAMES)
        return [(root, name) for root, name in pairs if name in eval_names]
    return [(root, name) for root, name in pairs if name == route]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        choices=[*config.ROUTE_NAMES, "all"],
        default="route_B",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "inference_videos",
    )
    parser.add_argument(
        "--num-zoom",
        type=int,
        default=15,
        help="Number of real jump-comparison figures to save per route.",
    )
    parser.add_argument(
        "--zoom-half-window",
        type=int,
        default=15,
        help="Frames before/after the selected jump frame in each zoom figure.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Write PNG localization figures only; skip MP4 rendering.",
    )
    args = parser.parse_args()

    all_outputs: List[Path] = []
    for root, name in selected_route_pairs(args.route):
        all_outputs.extend(
            render_route(
                root=root,
                name=name,
                output_dir=args.output_dir,
                fps=float(args.fps),
                video_width=int(args.width),
                video_height=int(args.height),
                num_zoom=max(0, int(args.num_zoom)),
                zoom_half_window=max(1, int(args.zoom_half_window)),
                no_video=bool(args.no_video),
            )
        )

    print("\nGenerated outputs:", flush=True)
    for path in all_outputs:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
