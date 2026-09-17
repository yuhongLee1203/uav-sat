#!/usr/bin/env python3
"""Render Bearing-v39 final navigation figures.

Visualization policy:
  - GT is the predefined Bearing-UAV waypoint route polyline, drawn as a GREEN
    SOLID line. Per-frame independently sampled GT observations are NOT joined,
    because that would create artificial zig-zag motion that is not the route.
  - Prediction is the RAW model output from the inference CSV (final_x/final_y),
    drawn as a RED SOLID polyline in frame order.
  - Prediction receives NO moving average, interpolation, spline fitting,
    resampling, denoising, corner rounding, or other display post-processing.

This file never changes inference, saved CSV values, evaluation GT, MLE/P90/LSR,
or any model component. It only renders the already-produced results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

Point = Tuple[float, float]
PRED = (228, 44, 52, 255)
GT = (40, 180, 70, 255)
HALO = (255, 255, 255, 220)
TEXTBG = (0, 0, 0, 170)


def _rows(p: Path) -> List[Dict[str, str]]:
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _origin(root: Path) -> Tuple[float, float]:
    rows = _rows(root / "routes" / "train_01" / "manifest.csv")
    if not rows:
        raise RuntimeError("train_01 manifest is empty")
    return float(rows[0]["x_m"]), float(rows[0]["y_m"])


def _find_csv(route: str, out: Path, summary: dict) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.exists():
        return p
    q = out / p.name
    if q.exists():
        return q
    matches = sorted(out.glob(f"{route}_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"No inference CSV for {route}")
    return matches[-1]


def _official_trajectory(root: Path, route: str) -> List[Point]:
    """Return the official/predefined route as straight waypoint-to-waypoint legs."""
    payload = json.loads(
        (root / "routes" / route / "waypoints.json").read_text(encoding="utf-8")
    )
    pts = [
        (float(wp["pixel_x"]), float(wp["pixel_y"]))
        for wp in sorted(payload["waypoints"], key=lambda x: int(x["waypoint_order"]))
    ]
    if len(pts) < 2:
        raise RuntimeError(f"{route}: predefined trajectory has <2 waypoints")
    return pts


def _abs_px(x: float, y: float, ox: float, oy: float, mpp: float) -> Point:
    return ((x + ox) / mpp, (y + oy) / mpp)


def _audit_and_raw_prediction(
    route: str,
    root: Path,
    out: Path,
    summary: dict,
    mpp: float,
    size: Tuple[int, int],
    ox: float,
    oy: float,
) -> List[Point]:
    """Validate metrics/coordinates and return raw final_x/final_y prediction pixels."""
    rows = _rows(_find_csv(route, out, summary))
    manifest = _rows(root / "routes" / route / "manifest.csv")
    if not rows or len(rows) != len(manifest):
        raise RuntimeError(f"{route}: CSV/manifest count mismatch")

    pred: List[Point] = []
    errors: List[float] = []
    max_contract_error = 0.0
    width, height = size

    for i, (row, man) in enumerate(zip(rows, manifest)):
        gx_rel = float(row["gt_x"])
        gy_rel = float(row["gt_y"])
        gx_abs = gx_rel + ox
        gy_abs = gy_rel + oy
        max_contract_error = max(
            max_contract_error,
            math.hypot(gx_abs - float(man["x_m"]), gy_abs - float(man["y_m"])),
        )

        # IMPORTANT: these are the model's saved final outputs. Do not modify.
        fx = float(row["final_x"])
        fy = float(row["final_y"])
        errors.append(math.hypot(fx - gx_rel, fy - gy_rel))
        point = _abs_px(fx, fy, ox, oy, mpp)
        if not (-1 <= point[0] <= width and -1 <= point[1] <= height):
            raise RuntimeError(f"{route}: pred {i} outside RSI")
        pred.append(point)

    if max_contract_error > 1e-3:
        raise RuntimeError(
            f"{route}: sample-GT coordinate contract mismatch {max_contract_error:.6f}m"
        )

    mle = float(np.mean(errors))
    if abs(mle - float(summary["MLE_m"])) > 1e-5:
        raise RuntimeError(
            f"{route}: MLE mismatch raw-CSV={mle:.9f} summary={float(summary['MLE_m']):.9f}"
        )

    print(
        f"[FINAL-PLOT-AUDIT] {route}: PASS frames={len(rows)} "
        f"raw-model-MLE={mle:.3f}m pred_source=CSV(final_x,final_y) postprocess=NONE",
        flush=True,
    )
    return pred


def _font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _bounds(groups: Tuple[List[Point], ...], width: int, height: int):
    pts = [p for group in groups for p in group]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    margin = max(180, int(0.05 * span))
    return (
        max(0, int(min(xs)) - margin),
        max(0, int(min(ys)) - margin),
        min(width, int(max(xs)) + margin),
        min(height, int(max(ys)) + margin),
    )


def _legend(img: Image.Image) -> None:
    """Minimal, high-visibility legend: only GT and Predict."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    scale = max(1.0, min(img.size) / 1200.0)
    font = _font(max(22, int(28 * scale)), True)
    pad = max(14, int(18 * scale))
    line_h = max(36, int(42 * scale))
    swatch = max(95, int(120 * scale))
    box_w = min(img.width - 2 * pad, max(330, int(390 * scale)))
    box_h = pad * 2 + line_h * 2

    draw.rounded_rectangle(
        (pad, pad, pad + box_w, pad + box_h),
        radius=12,
        fill=TEXTBG,
        outline=(255, 255, 255, 150),
        width=2,
    )

    x = pad + 18
    y = pad + 10
    gt_w = max(5, int(6 * scale))
    pred_w = max(5, int(6 * scale))

    draw.line((x, y + 14, x + swatch, y + 14), fill=HALO, width=gt_w + 4)
    draw.line((x, y + 14, x + swatch, y + 14), fill=GT, width=gt_w)
    draw.text((x + swatch + 18, y), "GT", font=font, fill="white")

    y += line_h
    draw.line((x, y + 14, x + swatch, y + 14), fill=HALO, width=pred_w + 4)
    draw.line((x, y + 14, x + swatch, y + 14), fill=PRED, width=pred_w)
    draw.text((x + swatch + 18, y), "Predict", font=font, fill="white")

    img.alpha_composite(overlay)


def render(route: str, root: Path, out: Path, summary: dict) -> None:
    sat_meta = json.loads((root / "bearing_satellite.json").read_text(encoding="utf-8"))
    mpp = float(sat_meta["mpp"])
    src = Image.open(sat_meta["satellite_image"]).convert("RGB")
    base = ImageEnhance.Brightness(src).enhance(0.84).convert("RGBA")
    ox, oy = _origin(root)

    # GT display: official waypoint geometry, joined as clean route legs.
    gt = _official_trajectory(root, route)

    # Prediction display: EXACT frame-order final_x/final_y from model inference.
    pred = _audit_and_raw_prediction(route, root, out, summary, mpp, base.size, ox, oy)

    draw = ImageDraw.Draw(base, "RGBA")
    scale = max(1.0, base.width / 4096.0)

    # Keep GT visually clean/prominent. This is route geometry, not model output.
    gt_w = max(5, int(6 * scale))
    draw.line(gt, fill=HALO, width=gt_w + 5, joint="curve")
    draw.line(gt, fill=GT, width=gt_w, joint="curve")

    # RAW PREDICTION: no joint='curve' and no smoothing of any kind.
    pred_w = max(4, int(5 * scale))
    draw.line(pred, fill=HALO, width=pred_w + 4)
    draw.line(pred, fill=PRED, width=pred_w)

    crop = base.crop(_bounds((gt, pred), *base.size)).convert("RGBA")
    _legend(crop)
    dest = out / f"{route}_final_result.jpg"
    crop.convert("RGB").save(dest, quality=98, subsampling=0)
    print(
        f"[FINAL-PLOT] {dest} | GT=official waypoint polyline | "
        f"PRED=raw CSV final_x/final_y | prediction_postprocess=NONE",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--routes", nargs="+", default=["test_01", "test_02"])
    args = parser.parse_args()

    root = Path(args.prepared_root).resolve()
    out = Path(args.output_dir).resolve()
    summaries = json.loads(
        (out / "bearing_v39_summary.json").read_text(encoding="utf-8")
    )
    for route in args.routes:
        render(route, root, out, summaries[route])


if __name__ == "__main__":
    main()
