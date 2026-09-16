#!/usr/bin/env python3
"""Render paper-style final Bearing-v39 result figures.

The numerical result is never smoothed or altered.  This module only changes
presentation so the trajectory stays readable on a busy satellite background.
The visual convention follows the Bearing-UAV paper figure style:

- reference / ground-truth route: purple dashed line;
- final prediction: red solid line;
- high-contrast white halo under both trajectories;
- clearly visible waypoint/start/end markers;
- compact metric legend added AFTER cropping, so it can never be cropped away.

Coordinate contract:
- inference CSV gt/final coordinates are metres RELATIVE to train_01 frame 0;
- Bearing manifests and satellite pixels are ABSOLUTE map coordinates.

The plotter restores the exact train_01 origin, verifies CSV GT against the
manifest, and recomputes MLE before writing an image.  No Kalman diagnostic,
trajectory smoothing, projection, or cosmetic coordinate modification is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

Point = Tuple[float, float]

PRED_COLOR = (238, 45, 45, 255)          # paper-like red
REF_COLOR = (176, 78, 245, 255)          # high-contrast purple
HALO_COLOR = (255, 255, 255, 235)
WAYPOINT_FILL = (176, 78, 245, 255)
PRED_MARKER = (238, 45, 45, 255)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _route_origin_m(prepared_root: Path) -> Tuple[float, float]:
    rows = _read_rows(prepared_root / "routes" / "train_01" / "manifest.csv")
    if not rows:
        raise RuntimeError("train_01 manifest is empty")
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
        (prepared_root / "routes" / route / "waypoints.json").read_text(encoding="utf-8")
    )
    points = [
        (float(item["pixel_x"]), float(item["pixel_y"]))
        for item in sorted(payload["waypoints"], key=lambda item: int(item["waypoint_order"]))
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
) -> List[Point]:
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
    errors: List[float] = []
    max_gt_manifest_error = 0.0
    width, height = image_size

    for index, (row, gt_row) in enumerate(zip(rows, manifest)):
        csv_gt_abs_x = float(row["gt_x"]) + origin_x_m
        csv_gt_abs_y = float(row["gt_y"]) + origin_y_m
        manifest_x = float(gt_row["x_m"])
        manifest_y = float(gt_row["y_m"])
        gt_manifest_error = math.hypot(
            csv_gt_abs_x - manifest_x,
            csv_gt_abs_y - manifest_y,
        )
        max_gt_manifest_error = max(max_gt_manifest_error, gt_manifest_error)

        fx, fy = float(row["final_x"]), float(row["final_y"])
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        errors.append(math.hypot(fx - gx, fy - gy))

        px = _relative_to_absolute_pixel(fx, fy, origin_x_m, origin_y_m, mpp)
        if not (-1.0 <= px[0] <= width and -1.0 <= px[1] <= height):
            raise RuntimeError(
                f"{route}: final point {index} outside satellite after coordinate restore: {px}"
            )
        final_px.append(px)

    if max_gt_manifest_error > 1e-3:
        raise RuntimeError(
            f"{route}: CSV GT/origin does not reproduce manifest GT; "
            f"max error={max_gt_manifest_error:.6f}m"
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
    margin = max(220, int(0.055 * max(max(xs) - min(xs), max(ys) - min(ys), 1)))
    return (
        max(0, int(min(xs)) - margin),
        max(0, int(min(ys)) - margin),
        min(width, int(max(xs)) + margin),
        min(height, int(max(ys)) + margin),
    )


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Point],
    *,
    fill,
    width: int,
    dash: float,
    gap: float,
) -> None:
    if len(points) < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = map(float, p0)
        x1, y1 = map(float, p1)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        ux, uy = dx / length, dy / length
        s = 0.0
        while s < length:
            e = min(length, s + dash)
            draw.line(
                (x0 + ux * s, y0 + uy * s, x0 + ux * e, y0 + uy * e),
                fill=fill,
                width=width,
            )
            s += dash + gap


def _font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _circle(draw, point: Point, radius: int, fill, outline=HALO_COLOR, outline_width=3):
    x, y = point
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=outline,
        width=outline_width,
    )


def _draw_legend(image: Image.Image, route: str, summary: Mapping[str, object]) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    scale = max(1.0, min(image.width, image.height) / 1200.0)
    title_font = _font(max(22, int(28 * scale)), bold=True)
    body_font = _font(max(18, int(22 * scale)), bold=False)

    pad = max(18, int(22 * scale))
    line_h = max(30, int(36 * scale))
    box_w = max(620, int(690 * scale))
    box_h = pad * 2 + line_h * 5
    x0, y0 = pad, pad
    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=max(12, int(16 * scale)),
        fill=(0, 0, 0, 178),
        outline=(255, 255, 255, 175),
        width=max(2, int(2 * scale)),
    )

    tx = x0 + pad
    ty = y0 + pad
    draw.text((tx, ty), f"{route} - Bearing-v39 final result", font=title_font, fill=(255, 255, 255, 255))
    ty += line_h
    draw.text(
        (tx, ty),
        f"MLE {float(summary['MLE_m']):.2f} m   P90 {float(summary['P90_m']):.2f} m   LSR@15 {float(summary['LSR@15_pct']):.1f}%",
        font=body_font,
        fill=(255, 255, 255, 255),
    )
    ty += line_h + 4

    sw = max(100, int(125 * scale))
    lw = max(8, int(10 * scale))
    # Prediction swatch.
    draw.line((tx, ty + 10, tx + sw, ty + 10), fill=HALO_COLOR, width=lw + 6)
    draw.line((tx, ty + 10, tx + sw, ty + 10), fill=PRED_COLOR, width=lw)
    draw.text((tx + sw + 18, ty - 3), "Final prediction", font=body_font, fill=(255, 255, 255, 255))
    ty += line_h

    # Reference dashed swatch.
    _draw_dashed_polyline(
        draw,
        [(tx, ty + 10), (tx + sw, ty + 10)],
        fill=HALO_COLOR,
        width=lw + 6,
        dash=24 * scale,
        gap=14 * scale,
    )
    _draw_dashed_polyline(
        draw,
        [(tx, ty + 10), (tx + sw, ty + 10)],
        fill=REF_COLOR,
        width=lw,
        dash=24 * scale,
        gap=14 * scale,
    )
    draw.text((tx + sw + 18, ty - 3), "Reference / GT route", font=body_font, fill=(255, 255, 255, 255))
    ty += line_h
    draw.text(
        (tx, ty),
        "Paper-style visualization only; coordinates and metrics are unchanged.",
        font=body_font,
        fill=(225, 225, 225, 255),
    )

    image.alpha_composite(overlay)


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
    source = Image.open(sat_path).convert("RGB")
    # Slightly dim a busy satellite background so red/purple trajectories remain legible.
    base = ImageEnhance.Brightness(source).enhance(0.88).convert("RGBA")

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
    scale = max(1.0, base.width / 4096.0)
    pred_w = max(14, int(round(16 * scale)))
    ref_w = max(11, int(round(13 * scale)))
    halo_extra = max(6, int(round(7 * scale)))
    dash = max(28.0, 34.0 * scale)
    gap = max(16.0, 20.0 * scale)

    # Draw reference first: white dashed halo + purple dashed route.
    _draw_dashed_polyline(
        draw, reference, fill=HALO_COLOR, width=ref_w + halo_extra, dash=dash, gap=gap
    )
    _draw_dashed_polyline(
        draw, reference, fill=REF_COLOR, width=ref_w, dash=dash, gap=gap
    )

    # Final prediction: solid red with a white halo.  This is intentionally drawn
    # above the reference so small deviations stay visible when the two overlap.
    draw.line(final, fill=HALO_COLOR, width=pred_w + halo_extra, joint="curve")
    draw.line(final, fill=PRED_COLOR, width=pred_w, joint="curve")

    # Reference waypoints make turns obvious without changing the trajectory.
    wp_r = max(9, int(round(11 * scale)))
    for point in reference:
        _circle(draw, point, wp_r, WAYPOINT_FILL, outline=HALO_COLOR, outline_width=max(3, int(3 * scale)))

    # Prediction start/end points are larger red markers.
    pred_r = max(11, int(round(14 * scale)))
    _circle(draw, final[0], pred_r, PRED_MARKER, outline=HALO_COLOR, outline_width=max(3, int(4 * scale)))
    _circle(draw, final[-1], pred_r, PRED_MARKER, outline=HALO_COLOR, outline_width=max(3, int(4 * scale)))

    bounds = _crop_bounds((reference, final), base.width, base.height)
    cropped = base.crop(bounds).convert("RGBA")
    _draw_legend(cropped, route, summary)

    out = output_dir / f"{route}_final_result.jpg"
    cropped.convert("RGB").save(out, quality=98, subsampling=0)
    print(
        f"[FINAL-PLOT] {out} | paper-style red solid prediction / purple dashed reference | "
        f"pred_width={pred_w}px ref_width={ref_w}px",
        flush=True,
    )


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
