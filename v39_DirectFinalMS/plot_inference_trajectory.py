#!/usr/bin/env python3
"""Plot original v39 DirectFinalMS Route-B/C inference trajectories.

This script is diagnostic only. It reads the per-frame inference CSVs already
written by robust_tracker.py and projects their metric XY coordinates back onto
the original satellite image using the same map geometry as the v39 data code.
No training/inference behavior is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
BASE_SRC = ROOT / "base_src"
DEFAULT_DATA_ROOT = REPO_ROOT / "v36_GvsK" / "v36_training_data"
DEFAULT_VISUAL_CKPT = (
    REPO_ROOT
    / "forNX"
    / "weights"
    / "v36_mobilenet_v3_small"
    / "checkpoints"
    / "visual_retrieval_A_only.pt"
)
EARTH_RADIUS_M = 6378137.0
Point = Tuple[float, float]


def _float(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return None if not math.isfinite(value) else value


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _metric_to_latlon(x_m: float, y_m: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    lat = float(origin_lat) + math.degrees(float(y_m) / EARTH_RADIUS_M)
    cos_lat = max(abs(math.cos(math.radians(float(origin_lat)))), 1e-9)
    lon = float(origin_lon) + math.degrees(float(x_m) / (EARTH_RADIUS_M * cos_lat))
    return lat, lon


def _load_mapper(data_root: Path):
    # data.py imports config.py, so point config at the same original data root
    # before importing the canonical v39 geometry helper.
    os.environ["UAVSAT_DATA_ROOT"] = str(data_root)
    sys.path.insert(0, str(BASE_SRC))
    try:
        from data import SatGeoMapper
    finally:
        try:
            sys.path.remove(str(BASE_SRC))
        except ValueError:
            pass

    sat_image = data_root / "satellite" / "sim_map_competition_roi_crop.png"
    sat_json = data_root / "satellite" / "sim_map_competition_roi_crop_worldfile_epsg3826.json"
    if not sat_image.exists():
        raise FileNotFoundError(f"Satellite image not found: {sat_image}")
    if not sat_json.exists():
        raise FileNotFoundError(f"Satellite world-file JSON not found: {sat_json}")
    return SatGeoMapper(sat_json, sat_image), sat_image


def _pixel_points(
    rows: Sequence[Mapping[str, str]],
    x_key: str,
    y_key: str,
    mapper,
    origin_lat: float,
    origin_lon: float,
) -> List[Point]:
    points: List[Point] = []
    for row in rows:
        x = _float(row, x_key)
        y = _float(row, y_key)
        if x is None or y is None:
            continue
        lat, lon = _metric_to_latlon(x, y, origin_lat, origin_lon)
        px, py = mapper.latlon_to_pixel(lat, lon)
        points.append((float(px), float(py)))
    return points


def _draw_line(draw: ImageDraw.ImageDraw, points: Sequence[Point], fill, width: int) -> None:
    if len(points) >= 2:
        draw.line([(round(x), round(y)) for x, y in points], fill=fill, width=width, joint="curve")


def _draw_marker(draw: ImageDraw.ImageDraw, point: Point, fill, radius: int) -> None:
    x, y = point
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=(255, 255, 255, 255),
        width=2,
    )


def _crop_bounds(point_sets: Iterable[Sequence[Point]], width: int, height: int, margin: int = 180) -> Tuple[int, int, int, int]:
    all_points = [point for points in point_sets for point in points]
    if not all_points:
        return 0, 0, width, height
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    left = max(0, int(min(xs)) - margin)
    top = max(0, int(min(ys)) - margin)
    right = min(width, int(max(xs)) + margin)
    bottom = min(height, int(max(ys)) + margin)
    if right - left < 700:
        extra = (700 - (right - left)) // 2
        left, right = max(0, left - extra), min(width, right + extra)
    if bottom - top < 700:
        extra = (700 - (bottom - top)) // 2
        top, bottom = max(0, top - extra), min(height, bottom + extra)
    return left, top, right, bottom


def _find_csv(route_name: str, output_dir: Path, route_summary: Mapping[str, object]) -> Path:
    explicit = route_summary.get("CSV")
    if isinstance(explicit, str):
        p = Path(explicit)
        if p.exists():
            return p
        candidate = output_dir / p.name
        if candidate.exists():
            return candidate
    matches = sorted(output_dir.glob(f"{route_name}_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"No {route_name} inference CSV found under {output_dir}")
    return matches[-1]


def _diagnostic_box(draw: ImageDraw.ImageDraw, route_name: str, summary: Mapping[str, object]) -> None:
    lines = [
        f"Original v39 DirectFinalMS - {route_name}",
        "reference: green | visual: yellow | Kalman: orange | final: red",
    ]
    fields = (
        ("MLE", "MLE_m", " m"),
        ("P90", "P90_m", " m"),
        ("LSR@5", "LSR@5_pct", "%"),
        ("Visual MAE", "VisualMeasurement_MAE_m", " m"),
        ("Kalman MAE", "Kalman_MAE_m", " m"),
        ("MS shift", "MS_MeanShiftFromKalman_m", " m"),
        ("Jump rate", "JumpRate_pct", "%"),
    )
    for label, key, suffix in fields:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"{label}: {float(value):.2f}{suffix}")
    box_w = 690
    box_h = 24 + 27 * len(lines)
    draw.rounded_rectangle((18, 18, 18 + box_w, 18 + box_h), radius=12, fill=(0, 0, 0, 185))
    for i, text in enumerate(lines):
        draw.text((34, 31 + i * 27), text, fill=(255, 255, 255, 255))


def render_route(
    route_name: str,
    output_dir: Path,
    route_summary: Mapping[str, object],
    mapper,
    satellite_path: Path,
    origin_lat: float,
    origin_lon: float,
) -> Dict[str, Path]:
    csv_path = _find_csv(route_name, output_dir, route_summary)
    rows = _read_rows(csv_path)

    reference = _pixel_points(rows, "gt_x", "gt_y", mapper, origin_lat, origin_lon)
    visual = _pixel_points(rows, "visual_anchor_x", "visual_anchor_y", mapper, origin_lat, origin_lon)
    kalman = _pixel_points(rows, "kalman_x", "kalman_y", mapper, origin_lat, origin_lon)
    final = _pixel_points(rows, "final_x", "final_y", mapper, origin_lat, origin_lon)
    if not reference or not final:
        raise RuntimeError(f"CSV lacks usable reference/final coordinates: {csv_path}")

    image = Image.open(satellite_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    line_width = max(5, image.width // 700)

    # Draw sparse reference-to-final error connectors so accumulated offset is
    # visible while keeping the satellite background readable.
    stride = max(1, min(len(reference), len(final)) // 28)
    for index in range(0, min(len(reference), len(final)), stride):
        draw.line(
            (reference[index], final[index]),
            fill=(255, 255, 255, 80),
            width=max(2, line_width // 2),
        )

    _draw_line(draw, reference, (40, 255, 100, 245), line_width + 3)
    _draw_line(draw, visual, (255, 230, 40, 220), line_width)
    _draw_line(draw, kalman, (255, 145, 25, 220), line_width)
    _draw_line(draw, final, (255, 45, 60, 255), line_width + 2)

    _draw_marker(draw, reference[0], (40, 255, 100, 255), max(9, line_width + 4))
    _draw_marker(draw, reference[-1], (40, 255, 100, 255), max(9, line_width + 4))
    _draw_marker(draw, final[0], (255, 45, 60, 255), max(7, line_width + 2))
    _draw_marker(draw, final[-1], (255, 45, 60, 255), max(7, line_width + 2))
    _diagnostic_box(draw, route_name, route_summary)

    full_path = output_dir / f"{route_name}_inference_trajectory_full.jpg"
    image.save(full_path, quality=95)

    bounds = _crop_bounds((reference, visual, kalman, final), image.width, image.height)
    zoom_path = output_dir / f"{route_name}_inference_trajectory_zoom.jpg"
    image.crop(bounds).save(zoom_path, quality=95)

    print(f"[PLOT] {route_name} full: {full_path}", flush=True)
    print(f"[PLOT] {route_name} zoom: {zoom_path}", flush=True)
    return {"full": full_path, "zoom": zoom_path, "csv": csv_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "output_wc_single"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--visual-checkpoint", default=str(DEFAULT_VISUAL_CKPT))
    parser.add_argument("--routes", nargs="+", default=["route_B", "route_C"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    visual_checkpoint = Path(args.visual_checkpoint).resolve()

    summary_path = output_dir / "robust_tracker_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing inference summary: {summary_path}")
    if not visual_checkpoint.exists():
        raise FileNotFoundError(f"Missing visual checkpoint: {visual_checkpoint}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = _load_checkpoint(visual_checkpoint)
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    mapper, satellite_path = _load_mapper(data_root)

    for route_name in args.routes:
        route_summary = summary.get(route_name)
        if not isinstance(route_summary, dict):
            raise KeyError(f"Summary has no route result: {route_name}")
        render_route(
            route_name,
            output_dir,
            route_summary,
            mapper,
            satellite_path,
            origin_lat,
            origin_lon,
        )


if __name__ == "__main__":
    main()
