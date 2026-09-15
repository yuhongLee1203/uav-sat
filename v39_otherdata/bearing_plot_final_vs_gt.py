#!/usr/bin/env python3
"""Plot Bearing reference route, true sampled GT, and final v39 prediction.

Bearing-UAV observations are independent samples, not frames from one recorded
flight.  Therefore the planned/reference route and the true per-frame sampled GT
must be shown separately:

  green solid line : planned/reference waypoint polyline
  cyan dots        : true selected Bearing sample coordinates (metric GT)
  red solid line   : raw final v39 prediction after final MeanShift

The true GT dots are deliberately not connected, because connecting independent
samples creates a misleading small zig-zag/caterpillar trajectory.  Metrics are
still computed against the true per-frame GT coordinates.  A second diagnostic
image also overlays the pre-final-MS Kalman trajectory in orange so any extra
final-MeanShift wobble can be identified without changing the model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _points(rows: Sequence[Mapping[str, str]], x_key: str, y_key: str, mpp: float):
    out = []
    for row in rows:
        try:
            x = float(row[x_key]) / mpp
            y = float(row[y_key]) / mpp
        except (KeyError, TypeError, ValueError):
            continue
        if x == x and y == y:
            out.append((x, y))
    return out


def _reference_waypoints(prepared_root: Path, route: str) -> List[Point]:
    path = prepared_root / "routes" / route / "waypoints.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = [
        (float(item["pixel_x"]), float(item["pixel_y"]))
        for item in payload.get("waypoints", [])
    ]
    if len(points) < 2:
        raise RuntimeError(f"Missing planned waypoints for {route}: {path}")
    return points


def _find_csv(route: str, output_dir: Path, summary: Mapping[str, object]) -> Path:
    explicit = summary.get("CSV")
    if isinstance(explicit, str):
        p = Path(explicit)
        if p.exists():
            return p
        q = output_dir / p.name
        if q.exists():
            return q
    matches = sorted(output_dir.glob(f"{route}_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"No inference CSV for {route} under {output_dir}")
    return matches[-1]


def _crop_bounds(groups, width: int, height: int, margin: int = 180):
    all_points = [p for group in groups for p in group]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    left = max(0, int(min(xs)) - margin)
    top = max(0, int(min(ys)) - margin)
    right = min(width, int(max(xs)) + margin)
    bottom = min(height, int(max(ys)) + margin)
    return left, top, right, bottom


def _draw_reference(draw: ImageDraw.ImageDraw, reference: Sequence[Point], width: int):
    draw.line(reference, fill=(40, 255, 100, 250), width=width + 3, joint="curve")
    corner_r = max(6, width + 1)
    for x, y in reference:
        draw.ellipse(
            (x - corner_r, y - corner_r, x + corner_r, y + corner_r),
            fill=(40, 255, 100, 255),
            outline=(255, 255, 255, 220),
            width=2,
        )


def _draw_sparse_gt_dots(
    draw: ImageDraw.ImageDraw,
    gt: Sequence[Point],
    radius: int,
    stride: int = 4,
):
    # Cyan, not green, so true sampled GT can never be confused with the green
    # planned/reference route.  Samples remain unconnected by design.
    for i, (x, y) in enumerate(gt):
        if i % max(1, int(stride)) != 0 and i != len(gt) - 1:
            continue
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(40, 225, 255, 210),
            outline=(255, 255, 255, 150),
            width=1,
        )


def _draw_info_box(
    draw: ImageDraw.ImageDraw,
    route: str,
    summary: Mapping[str, object],
    *,
    diagnostic: bool,
):
    lines = [
        f"{route} held-out inference",
        "Reference route: green | True sampled GT: cyan dots",
        "Final prediction: red" + (" | Kalman before final MS: orange" if diagnostic else ""),
        "Metrics use final prediction vs true sampled GT",
    ]
    for label, key, suffix in (
        ("MLE", "MLE_m", " m"),
        ("P90", "P90_m", " m"),
        ("LSR@15", "LSR@15_pct", "%"),
        ("Jump", "JumpRate_pct", "%"),
        ("MS shift", "MS_MeanShiftFromKalman_m", " m"),
    ):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"{label}: {float(value):.2f}{suffix}")
    box_w = 820 if diagnostic else 730
    box_h = 24 + 27 * len(lines)
    draw.rounded_rectangle(
        (18, 18, 18 + box_w, 18 + box_h),
        radius=12,
        fill=(0, 0, 0, 185),
    )
    for i, text in enumerate(lines):
        draw.text((34, 31 + i * 27), text, fill=(255, 255, 255, 255))


def render(route: str, prepared_root: Path, output_dir: Path, summary: Mapping[str, object]):
    sat_meta = json.loads(
        (prepared_root / "bearing_satellite.json").read_text(encoding="utf-8")
    )
    sat_path = Path(sat_meta["satellite_image"])
    mpp = float(sat_meta["mpp"])
    rows = _read_rows(_find_csv(route, output_dir, summary))

    true_gt = _points(rows, "gt_x", "gt_y", mpp)
    final = _points(rows, "final_x", "final_y", mpp)
    kalman = _points(rows, "kalman_x", "kalman_y", mpp)
    reference = _reference_waypoints(prepared_root, route)
    if not true_gt or not final:
        raise RuntimeError(f"Missing GT/final coordinates for {route}")

    base_image = Image.open(sat_path).convert("RGB")
    width = max(5, base_image.width // 700)

    # Clean publication view: planned route + true GT dots + raw final output.
    image = base_image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_reference(draw, reference, width)
    _draw_sparse_gt_dots(draw, true_gt, radius=max(2, width // 2), stride=4)
    draw.line(final, fill=(255, 45, 60, 255), width=width + 2, joint="curve")
    _draw_info_box(draw, route, summary, diagnostic=False)

    full = output_dir / f"{route}_final_vs_gt_full.jpg"
    image.save(full, quality=95)
    bounds = _crop_bounds((reference, true_gt, final), image.width, image.height)
    zoom = output_dir / f"{route}_final_vs_gt_zoom.jpg"
    image.crop(bounds).save(zoom, quality=95)

    # Diagnostic view: same truth/reference, plus Kalman state before final MS.
    diagnostic = base_image.copy()
    ddraw = ImageDraw.Draw(diagnostic, "RGBA")
    _draw_reference(ddraw, reference, width)
    _draw_sparse_gt_dots(ddraw, true_gt, radius=max(2, width // 2), stride=4)
    if kalman:
        ddraw.line(kalman, fill=(255, 180, 30, 220), width=max(3, width), joint="curve")
    ddraw.line(final, fill=(255, 45, 60, 255), width=width + 2, joint="curve")
    _draw_info_box(ddraw, route, summary, diagnostic=True)
    diag_groups = (reference, true_gt, final, kalman) if kalman else (reference, true_gt, final)
    diag_bounds = _crop_bounds(diag_groups, diagnostic.width, diagnostic.height)
    diag_zoom = output_dir / f"{route}_kalman_vs_final_diagnostic.jpg"
    diagnostic.crop(diag_bounds).save(diag_zoom, quality=95)

    print(f"[PLOT] {full}", flush=True)
    print(f"[PLOT] {zoom}", flush=True)
    print(f"[PLOT] {diag_zoom}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--routes", nargs="+", default=["test_01", "test_02"])
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else prepared_root / "v39_output_corrected"
    )
    summary_path = output_dir / "bearing_v39_summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    for route in args.routes:
        render(route, prepared_root, output_dir, summaries[route])


if __name__ == "__main__":
    main()
