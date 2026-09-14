#!/usr/bin/env python3
"""Visualize Bearing-UAV inference trajectories on the full satellite map.

The plot is diagnostic only.  It does not participate in training or inference.
Coordinates in the inference CSV are metric city-map coordinates and are mapped
back to RSI pixels using the Bearing-UAV map MPP stored in bearing_satellite.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw


Point = Tuple[float, float]


def _float(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_points(rows: Sequence[Mapping[str, str]], x_key: str, y_key: str, mpp: float) -> List[Point]:
    points: List[Point] = []
    for row in rows:
        x = _float(row, x_key)
        y = _float(row, y_key)
        if x is None or y is None:
            continue
        points.append((x / mpp, y / mpp))
    return points


def _draw_line(draw: ImageDraw.ImageDraw, points: Sequence[Point], fill, width: int) -> None:
    if len(points) >= 2:
        draw.line([(round(x), round(y)) for x, y in points], fill=fill, width=width, joint="curve")


def _draw_marker(draw: ImageDraw.ImageDraw, point: Point, fill, outline=(255, 255, 255, 255), radius: int = 10) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)


def _diagnostic_box(draw: ImageDraw.ImageDraw, route_name: str, summary: Mapping[str, object]) -> None:
    lines = [
        f"{route_name} held-out inference",
        "GT/reference: green | visual: yellow | Kalman: orange | final: red",
    ]
    for label, key in (
        ("MLE", "MLE_m"),
        ("P90", "P90_m"),
        ("LSR@15", "LSR@15_pct"),
        ("Visual MAE", "VisualMeasurement_MAE_m"),
        ("Kalman MAE", "Kalman_MAE_m"),
        ("Progress err", "MeanProgressError_m"),
    ):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            suffix = "%" if "LSR" in label else " m"
            lines.append(f"{label}: {float(value):.2f}{suffix}")
    box_w = 660
    box_h = 24 + 27 * len(lines)
    draw.rounded_rectangle((18, 18, 18 + box_w, 18 + box_h), radius=12, fill=(0, 0, 0, 185))
    for i, text in enumerate(lines):
        draw.text((34, 31 + i * 27), text, fill=(255, 255, 255, 255))


def _crop_bounds(point_sets: Iterable[Sequence[Point]], width: int, height: int, margin: int = 180) -> Tuple[int, int, int, int]:
    all_points = [p for points in point_sets for p in points]
    if not all_points:
        return (0, 0, width, height)
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    left = max(0, int(min(xs)) - margin)
    top = max(0, int(min(ys)) - margin)
    right = min(width, int(max(xs)) + margin)
    bottom = min(height, int(max(ys)) + margin)
    if right - left < 600:
        extra = (600 - (right - left)) // 2
        left, right = max(0, left - extra), min(width, right + extra)
    if bottom - top < 600:
        extra = (600 - (bottom - top)) // 2
        top, bottom = max(0, top - extra), min(height, bottom + extra)
    return (left, top, right, bottom)


def _find_csv(route_name: str, output_dir: Path, summary: Mapping[str, object]) -> Path:
    explicit = summary.get("CSV")
    if isinstance(explicit, str):
        p = Path(explicit)
        if p.exists():
            return p
    matches = sorted(output_dir.glob(f"{route_name}_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"No inference frames CSV found for {route_name} under {output_dir}")
    return matches[-1]


def render_route_from_output(
    route_name: str,
    prepared_root: Path,
    output_dir: Path,
    summary: Mapping[str, object],
) -> Dict[str, Path]:
    """Render full-map and zoomed inference trajectories for one held-out route."""
    prepared_root = Path(prepared_root)
    output_dir = Path(output_dir)
    sat_meta = json.loads((prepared_root / "bearing_satellite.json").read_text(encoding="utf-8"))
    satellite_path = Path(sat_meta["satellite_image"])
    mpp = float(sat_meta["mpp"])
    csv_path = _find_csv(route_name, output_dir, summary)
    rows = _read_rows(csv_path)

    gt = _metric_points(rows, "gt_x", "gt_y", mpp)
    visual = _metric_points(rows, "visual_anchor_x", "visual_anchor_y", mpp)
    kalman = _metric_points(rows, "kalman_x", "kalman_y", mpp)
    final = _metric_points(rows, "final_x", "final_y", mpp)
    if not gt or not final:
        raise RuntimeError(f"{csv_path} does not contain usable gt_x/gt_y and final_x/final_y columns")

    image = Image.open(satellite_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    base_width = max(5, image.width // 700)

    # Sparse error connectors make along-track lag immediately visible without
    # obscuring the satellite image.
    stride = max(1, len(rows) // 24)
    for i in range(0, min(len(gt), len(final)), stride):
        draw.line((gt[i], final[i]), fill=(255, 255, 255, 90), width=max(2, base_width // 2))

    _draw_line(draw, gt, (40, 255, 100, 245), base_width + 3)
    _draw_line(draw, visual, (255, 230, 40, 220), base_width)
    _draw_line(draw, kalman, (255, 145, 25, 220), base_width)
    _draw_line(draw, final, (255, 45, 60, 255), base_width + 2)

    _draw_marker(draw, gt[0], (40, 255, 100, 255), radius=max(9, base_width + 4))
    _draw_marker(draw, gt[-1], (40, 255, 100, 255), radius=max(9, base_width + 4))
    _draw_marker(draw, final[0], (255, 45, 60, 255), radius=max(7, base_width + 2))
    _draw_marker(draw, final[-1], (255, 45, 60, 255), radius=max(7, base_width + 2))
    _diagnostic_box(draw, route_name, summary)

    full_path = output_dir / f"{route_name}_inference_trajectory_full.jpg"
    image.save(full_path, quality=95)

    bounds = _crop_bounds((gt, visual, kalman, final), image.width, image.height)
    zoom = image.crop(bounds)
    zoom_path = output_dir / f"{route_name}_inference_trajectory_zoom.jpg"
    zoom.save(zoom_path, quality=95)

    print(f"[PLOT] {route_name} full: {full_path}", flush=True)
    print(f"[PLOT] {route_name} zoom: {zoom_path}", flush=True)
    return {"full": full_path, "zoom": zoom_path, "csv": csv_path}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--routes", nargs="+", default=["test_01", "test_02"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepared_root = Path(args.prepared_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else prepared_root / "v39_output_corrected"
    summary_path = output_dir / "bearing_v39_summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for route_name in args.routes:
        render_route_from_output(route_name, prepared_root, output_dir, summaries.get(route_name, {}))


if __name__ == "__main__":
    main()
