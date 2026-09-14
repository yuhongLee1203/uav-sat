#!/usr/bin/env python3
"""Plot only GT/reference and final prediction for Bearing v39 inference."""
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


def render(route: str, prepared_root: Path, output_dir: Path, summary: Mapping[str, object]):
    sat_meta = json.loads((prepared_root / "bearing_satellite.json").read_text(encoding="utf-8"))
    sat_path = Path(sat_meta["satellite_image"])
    mpp = float(sat_meta["mpp"])
    rows = _read_rows(_find_csv(route, output_dir, summary))
    gt = _points(rows, "gt_x", "gt_y", mpp)
    final = _points(rows, "final_x", "final_y", mpp)
    if not gt or not final:
        raise RuntimeError(f"Missing GT/final coordinates for {route}")

    image = Image.open(sat_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width = max(5, image.width // 700)
    draw.line(gt, fill=(40, 255, 100, 245), width=width + 3, joint="curve")
    draw.line(final, fill=(255, 45, 60, 255), width=width + 2, joint="curve")

    lines = [
        f"{route} held-out inference",
        "GT/reference: green | Final prediction: red",
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
    box_w = 600
    box_h = 24 + 27 * len(lines)
    draw.rounded_rectangle((18, 18, 18 + box_w, 18 + box_h), radius=12, fill=(0, 0, 0, 185))
    for i, text in enumerate(lines):
        draw.text((34, 31 + i * 27), text, fill=(255, 255, 255, 255))

    full = output_dir / f"{route}_final_vs_gt_full.jpg"
    image.save(full, quality=95)
    crop = image.crop(_crop_bounds((gt, final), image.width, image.height))
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
