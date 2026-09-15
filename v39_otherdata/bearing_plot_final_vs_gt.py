#!/usr/bin/env python3
"""Plot the planned piecewise-linear reference route and final prediction.

Important: Bearing-UAV observations are independent samples, not frames from a
single recorded flight. Their true coordinates are therefore scattered around
the planned route.  Connecting those true sample coordinates produces a fake
"caterpillar" polyline that looks like many small turns.  This visualizer keeps
those coordinates for the localization metrics, but does NOT connect them into
a route.  The green line is the planned reference route from waypoints.json;
true sampled GT positions are shown only as sparse dots.  Final prediction is
shown as the red trajectory.
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
    points = []
    for item in payload.get("waypoints", []):
        points.append((float(item["pixel_x"]), float(item["pixel_y"])))
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


def _draw_sparse_gt_dots(draw: ImageDraw.ImageDraw, gt: Sequence[Point], radius: int, stride: int = 6):
    # The true Bearing sample positions remain visible for transparency, but are
    # deliberately not connected. Connecting independent observations is what
    # created the misleading small zig-zag/caterpillar line.
    for i, (x, y) in enumerate(gt):
        if i % max(1, int(stride)) != 0 and i != len(gt) - 1:
            continue
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(80, 255, 130, 145),
            outline=(255, 255, 255, 120),
            width=1,
        )


def render(route: str, prepared_root: Path, output_dir: Path, summary: Mapping[str, object]):
    sat_meta = json.loads((prepared_root / "bearing_satellite.json").read_text(encoding="utf-8"))
    sat_path = Path(sat_meta["satellite_image"])
    mpp = float(sat_meta["mpp"])
    rows = _read_rows(_find_csv(route, output_dir, summary))

    # True per-frame GT is still used by the tracker metrics.
    true_gt = _points(rows, "gt_x", "gt_y", mpp)
    final = _points(rows, "final_x", "final_y", mpp)
    reference = _reference_waypoints(prepared_root, route)
    if not true_gt or not final:
        raise RuntimeError(f"Missing GT/final coordinates for {route}")

    image = Image.open(sat_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width = max(5, image.width // 700)

    # EXACT piecewise-linear reference route: straight -> major corner -> straight.
    # Do not connect the scattered true GT observations.
    draw.line(reference, fill=(40, 255, 100, 250), width=width + 3, joint="curve")
    corner_r = max(6, width + 1)
    for x, y in reference:
        draw.ellipse(
            (x - corner_r, y - corner_r, x + corner_r, y + corner_r),
            fill=(40, 255, 100, 255),
            outline=(255, 255, 255, 220),
            width=2,
        )

    # Sparse actual GT dots show where the selected Bearing images truly are,
    # without turning that sample scatter into a fake wiggly route.
    _draw_sparse_gt_dots(draw, true_gt, radius=max(2, width // 2), stride=6)
    draw.line(final, fill=(255, 45, 60, 255), width=width + 2, joint="curve")

    lines = [
        f"{route} held-out inference",
        "Planned reference route: green | Final prediction: red",
        "True sampled GT: green dots (not connected)",
        "Metrics below use final prediction vs true sampled GT",
    ]
    for label, key, suffix in (
        ("MLE", "MLE_m", " m"),
        ("P90", "P90_m", " m"),
        ("LSR@15", "LSR@15_pct", "%"),
        ("Jump", "JumpRate_pct", "%"),
    ):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"{label}: {float(value):.2f}{suffix}")
    box_w = 720
    box_h = 24 + 27 * len(lines)
    draw.rounded_rectangle((18, 18, 18 + box_w, 18 + box_h), radius=12, fill=(0, 0, 0, 185))
    for i, text in enumerate(lines):
        draw.text((34, 31 + i * 27), text, fill=(255, 255, 255, 255))

    full = output_dir / f"{route}_final_vs_gt_full.jpg"
    image.save(full, quality=95)
    crop = image.crop(_crop_bounds((reference, true_gt, final), image.width, image.height))
    zoom = output_dir / f"{route}_final_vs_gt_zoom.jpg"
    crop.save(zoom, quality=95)
    print(f"[PLOT] {full}", flush=True)
    print(f"[PLOT] {zoom}", flush=True)


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
