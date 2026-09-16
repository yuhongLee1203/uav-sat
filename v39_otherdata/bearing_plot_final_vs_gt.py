#!/usr/bin/env python3
"""Render only the final Bearing-v39 result for each held-out route.

Coordinate contract:
- inference CSV gt/final coordinates are metres RELATIVE to the route_A origin;
- Bearing manifests and satellite pixels are ABSOLUTE map coordinates.

The plotter therefore restores the route_A origin before converting final XY to
satellite pixels.  It also verifies CSV GT against the manifest and recomputes
MLE before any image is written.  No Kalman diagnostic image, no extra full-size
copy, and no cosmetic trajectory smoothing are produced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _route_origin_m(prepared_root: Path) -> Tuple[float, float]:
    rows = _read_rows(prepared_root / "routes" / "route_A" / "manifest.csv")
    if not rows:
        raise RuntimeError("route_A manifest is empty")
    return float(rows[0]["x_m"]), float(rows[0]["y_m"])


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


def _reference_waypoints(prepared_root: Path, route: str) -> List[Point]:
    payload = json.loads(
        (prepared_root / "routes" / route / "waypoints.json").read_text(
            encoding="utf-8"
        )
    )
    points = [
        (float(item["pixel_x"]), float(item["pixel_y"]))
        for item in sorted(
            payload["waypoints"], key=lambda item: int(item["waypoint_order"])
        )
    ]
    if len(points) < 2:
        raise RuntimeError(f"{route}: fewer than two waypoints")
    return points


def _relative_to_absolute_pixel(
    x_rel_m: float,
    y_rel_m: float,
    origin_x_m: float,
    origin_y_m: float,
    mpp: float,
) -> Point:
    return (
        (float(x_rel_m) + float(origin_x_m)) / float(mpp),
        (float(y_rel_m) + float(origin_y_m)) / float(mpp),
    )


def _audit_and_points(
    route: str,
    prepared_root: Path,
    output_dir: Path,
    summary: Mapping[str, object],
    mpp: float,
    image_size: Tuple[int, int],
    origin_x_m: float,
    origin_y_m: float,
):
    csv_path = _find_csv(route, output_dir, summary)
    rows = _read_rows(csv_path)
    manifest = _read_rows(prepared_root / "routes" / route / "manifest.csv")
    if not rows:
        raise RuntimeError(f"{route}: inference CSV is empty")
    if len(rows) != len(manifest):
        raise RuntimeError(
            f"{route}: CSV/manifest frame count mismatch: {len(rows)} vs {len(manifest)}"
        )

    final_px: List[Point] = []
    errors = []
    max_gt_manifest_error = 0.0
    width, height = image_size

    for index, (row, gt_row) in enumerate(zip(rows, manifest)):
        csv_gt_abs_x = float(row["gt_x"]) + origin_x_m
        csv_gt_abs_y = float(row["gt_y"]) + origin_y_m
        manifest_x = float(gt_row["x_m"])
        manifest_y = float(gt_row["y_m"])
        gt_manifest_error = math.hypot(
            csv_gt_abs_x - manifest_x, csv_gt_abs_y - manifest_y
        )
        max_gt_manifest_error = max(max_gt_manifest_error, gt_manifest_error)

        fx = float(row["final_x"])
        fy = float(row["final_y"])
        gx = float(row["gt_x"])
        gy = float(row["gt_y"])
        errors.append(math.hypot(fx - gx, fy - gy))

        px = _relative_to_absolute_pixel(
            fx, fy, origin_x_m, origin_y_m, mpp
        )
        if not (-1.0 <= px[0] <= width and -1.0 <= px[1] <= height):
            raise RuntimeError(
                f"{route}: final point {index} is outside satellite after coordinate restore: {px}"
            )
        final_px.append(px)

    if max_gt_manifest_error > 1e-3:
        raise RuntimeError(
            f"{route}: CSV GT/origin does not reproduce manifest GT; max error={max_gt_manifest_error:.6f}m"
        )

    mle = float(np.mean(np.asarray(errors, dtype=np.float64)))
    summary_mle = float(summary["MLE_m"])
    if abs(mle - summary_mle) > 1e-5:
        raise RuntimeError(
            f"{route}: summary MLE mismatch: CSV={mle:.9f} summary={summary_mle:.9f}"
        )

    print(
        f"[FINAL-PLOT-AUDIT] {route}: PASS | frames={len(rows)} "
        f"MLE={mle:.3f}m GT-coordinate-max-error={max_gt_manifest_error:.6f}m",
        flush=True,
    )
    return final_px


def _crop_bounds(groups: Sequence[Sequence[Point]], width: int, height: int):
    points = [p for group in groups for p in group]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = 160
    return (
        max(0, int(min(xs)) - margin),
        max(0, int(min(ys)) - margin),
        min(width, int(max(xs)) + margin),
        min(height, int(max(ys)) + margin),
    )


def render(
    route: str,
    prepared_root: Path,
    output_dir: Path,
    summary: Mapping[str, object],
):
    sat_meta = json.loads(
        (prepared_root / "bearing_satellite.json").read_text(encoding="utf-8")
    )
    sat_path = Path(sat_meta["satellite_image"])
    mpp = float(sat_meta["mpp"])
    base = Image.open(sat_path).convert("RGB")
    origin_x_m, origin_y_m = _route_origin_m(prepared_root)
    reference = _reference_waypoints(prepared_root, route)
    final = _audit_and_points(
        route,
        prepared_root,
        output_dir,
        summary,
        mpp,
        base.size,
        origin_x_m,
        origin_y_m,
    )

    draw = ImageDraw.Draw(base, "RGBA")
    width = max(5, base.width // 700)
    draw.line(reference, fill=(30, 255, 95, 255), width=width + 3, joint="curve")
    draw.line(final, fill=(255, 45, 55, 255), width=width + 2, joint="curve")

    # Mark only route start/end so the final figure stays clean.
    r = max(6, width + 1)
    for x, y in (reference[0], reference[-1]):
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(30, 255, 95, 255),
            outline=(255, 255, 255, 230),
            width=2,
        )

    lines = [
        f"{route} final result",
        f"MLE {float(summary['MLE_m']):.2f} m | P90 {float(summary['P90_m']):.2f} m",
        f"LSR@15 {float(summary['LSR@15_pct']):.1f}%",
        "green: reference route | red: final prediction",
    ]
    box_w, box_h = 520, 30 + 27 * len(lines)
    draw.rounded_rectangle(
        (18, 18, 18 + box_w, 18 + box_h),
        radius=10,
        fill=(0, 0, 0, 170),
    )
    for i, text in enumerate(lines):
        draw.text((34, 32 + i * 27), text, fill=(255, 255, 255, 255))

    bounds = _crop_bounds((reference, final), base.width, base.height)
    out = output_dir / f"{route}_final_result.jpg"
    base.crop(bounds).save(out, quality=95)
    print(f"[FINAL-PLOT] {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--routes", nargs="+", default=["test_01", "test_02"])
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    summaries = json.loads(
        (output_dir / "bearing_v39_summary.json").read_text(encoding="utf-8")
    )
    for route in args.routes:
        render(route, prepared_root, output_dir, summaries[route])


if __name__ == "__main__":
    main()
