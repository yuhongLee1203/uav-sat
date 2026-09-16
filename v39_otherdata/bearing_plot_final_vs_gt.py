#!/usr/bin/env python3
"""Plot Bearing reference route, true sampled GT, Kalman, and final v39 output.

IMPORTANT COORDINATE CONTRACT
-----------------------------
The v39 runtime works in XY metres RELATIVE to the visual checkpoint origin.
The Bearing manifest/waypoints and satellite image use ABSOLUTE map coordinates.
Therefore every inference CSV coordinate (gt_x/final_x/kalman_x/...) must have
(route_A first sample) origin added back before it is converted to satellite
pixels.  Older versions of this plotter forgot that translation, which drew a
correct relative prediction hundreds of pixels away from the green route.

Display:
  green solid line : planned/reference waypoint polyline (absolute map pixels)
  cyan dots        : true selected Bearing sample coordinates (absolute GT)
  red solid line   : raw final v39 prediction, translated back to absolute map
  orange diagnostic: pre-final-MS Kalman, also translated back to absolute map

No output trajectory is cosmetically smoothed or projected onto the route.
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
    """Return the exact origin used by the visual checkpoint/runtime.

    train_visual_retrieval_a_only() creates RouteDataset(route_A) without an
    externally supplied origin, and RouteDataset defines that origin as its
    first manifest sample.  The checkpoint stores the same values.  Because the
    wrapper always starts from a clean prepared root, this is the authoritative
    runtime origin for the current experiment.
    """
    manifest = prepared_root / "routes" / "route_A" / "manifest.csv"
    rows = _read_rows(manifest)
    if not rows:
        raise RuntimeError(f"Empty route_A manifest: {manifest}")
    return float(rows[0]["x_m"]), float(rows[0]["y_m"])


def _relative_points_to_absolute_pixels(
    rows: Sequence[Mapping[str, str]],
    x_key: str,
    y_key: str,
    mpp: float,
    origin_x_m: float,
    origin_y_m: float,
) -> List[Point]:
    out: List[Point] = []
    for row in rows:
        try:
            x_rel = float(row[x_key])
            y_rel = float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x_rel) and math.isfinite(y_rel):
            out.append(
                (
                    (x_rel + float(origin_x_m)) / float(mpp),
                    (y_rel + float(origin_y_m)) / float(mpp),
                )
            )
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


def _manifest_by_frame(prepared_root: Path, route: str) -> Dict[int, Dict[str, str]]:
    manifest = prepared_root / "routes" / route / "manifest.csv"
    rows = _read_rows(manifest)
    result: Dict[int, Dict[str, str]] = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        if frame_id in result:
            raise RuntimeError(f"Duplicate frame_id={frame_id} in {manifest}")
        result[frame_id] = row
    if not result:
        raise RuntimeError(f"Empty manifest: {manifest}")
    return result


def _find_csv(route: str, output_dir: Path, summary: Mapping[str, object]) -> Path:
    explicit = summary.get("CSV")
    if isinstance(explicit, str):
        p = Path(explicit)
        if p.exists():
            # Do not silently accept a CSV from another output directory.
            if p.resolve().parent == output_dir.resolve():
                return p
        q = output_dir / p.name
        if q.exists():
            return q
    matches = sorted(output_dir.glob(f"{route}_*_frames.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one inference CSV for {route} under {output_dir}; "
            f"found {len(matches)}: {[p.name for p in matches]}"
        )
    return matches[0]


def _audit_csv_alignment(
    route: str,
    rows: Sequence[Mapping[str, str]],
    prepared_root: Path,
    summary: Mapping[str, object],
    origin_x_m: float,
    origin_y_m: float,
) -> None:
    """Hard-fail if CSV, manifest, origin, or summary are not the same run."""
    manifest = _manifest_by_frame(prepared_root, route)
    gt_manifest_errors = []
    final_errors = []
    image_mismatches = []

    for row in rows:
        frame_id = int(row["frame_id"])
        if frame_id not in manifest:
            raise RuntimeError(
                f"{route}: CSV frame_id={frame_id} not present in current manifest"
            )
        m = manifest[frame_id]

        csv_image = str(row.get("image_path", ""))
        manifest_image = str(m.get("image_path", ""))
        if csv_image and manifest_image and Path(csv_image).name != Path(manifest_image).name:
            image_mismatches.append((frame_id, Path(csv_image).name, Path(manifest_image).name))

        gt_abs = np.asarray(
            [
                float(row["gt_x"]) + float(origin_x_m),
                float(row["gt_y"]) + float(origin_y_m),
            ],
            dtype=np.float64,
        )
        manifest_abs = np.asarray(
            [float(m["x_m"]), float(m["y_m"])], dtype=np.float64
        )
        gt_manifest_errors.append(float(np.linalg.norm(gt_abs - manifest_abs)))

        final_rel = np.asarray(
            [float(row["final_x"]), float(row["final_y"])], dtype=np.float64
        )
        gt_rel = np.asarray(
            [float(row["gt_x"]), float(row["gt_y"])], dtype=np.float64
        )
        final_errors.append(float(np.linalg.norm(final_rel - gt_rel)))

    if image_mismatches:
        raise RuntimeError(
            f"{route}: CSV/manifest image mismatch; first={image_mismatches[0]}"
        )

    max_gt_manifest = max(gt_manifest_errors) if gt_manifest_errors else float("inf")
    if max_gt_manifest > 0.02:
        raise RuntimeError(
            f"{route}: coordinate-origin audit failed: reconstructed CSV GT differs "
            f"from manifest by up to {max_gt_manifest:.6f} m"
        )

    recomputed_mle = float(np.mean(final_errors)) if final_errors else float("nan")
    summary_mle = float(summary.get("MLE_m", float("nan")))
    if not (math.isfinite(recomputed_mle) and math.isfinite(summary_mle)):
        raise RuntimeError(f"{route}: invalid MLE during CSV/summary audit")
    if abs(recomputed_mle - summary_mle) > 1e-4:
        raise RuntimeError(
            f"{route}: stale/wrong CSV: recomputed MLE={recomputed_mle:.6f} m "
            f"but summary MLE={summary_mle:.6f} m"
        )

    print(
        f"[COORD-AUDIT] {route}: PASS | origin=({origin_x_m:.6f}, "
        f"{origin_y_m:.6f})m | max(csvGT+origin - manifest)="
        f"{max_gt_manifest:.6f}m | recomputed MLE={recomputed_mle:.6f}m",
        flush=True,
    )


def _crop_bounds(groups, width: int, height: int, margin: int = 180):
    all_points = [p for group in groups for p in group]
    if not all_points:
        return 0, 0, width, height
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
    stride: int = 3,
):
    for i, (x, y) in enumerate(gt):
        if i % max(1, int(stride)) != 0 and i != len(gt) - 1:
            continue
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(40, 225, 255, 230),
            outline=(255, 255, 255, 180),
            width=1,
        )


def _point_to_polyline_distance_m(
    point_px: Point, reference_px: Sequence[Point], mpp: float
) -> float:
    p = np.asarray(point_px, dtype=np.float64)
    best = float("inf")
    for a_raw, b_raw in zip(reference_px, reference_px[1:]):
        a = np.asarray(a_raw, dtype=np.float64)
        b = np.asarray(b_raw, dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            q = a
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            q = a + t * ab
        best = min(best, float(np.linalg.norm(p - q)) * float(mpp))
    return best


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
        "Coordinate audit: relative runtime XY -> absolute satellite XY",
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
    box_w = 900 if diagnostic else 820
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
    csv_path = _find_csv(route, output_dir, summary)
    rows = _read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"Empty inference CSV: {csv_path}")

    origin_x_m, origin_y_m = _route_origin_m(prepared_root)
    _audit_csv_alignment(
        route,
        rows,
        prepared_root,
        summary,
        origin_x_m,
        origin_y_m,
    )

    true_gt = _relative_points_to_absolute_pixels(
        rows, "gt_x", "gt_y", mpp, origin_x_m, origin_y_m
    )
    final = _relative_points_to_absolute_pixels(
        rows, "final_x", "final_y", mpp, origin_x_m, origin_y_m
    )
    kalman = _relative_points_to_absolute_pixels(
        rows, "kalman_x", "kalman_y", mpp, origin_x_m, origin_y_m
    )
    reference = _reference_waypoints(prepared_root, route)
    if not true_gt or not final:
        raise RuntimeError(f"Missing GT/final coordinates for {route}")

    base_image = Image.open(sat_path).convert("RGB")
    width = max(5, base_image.width // 700)

    # Hard bounds check catches another unit/origin mistake before any image is saved.
    for label, points in (("GT", true_gt), ("Final", final), ("Kalman", kalman)):
        if not points:
            continue
        outside = [
            p for p in points
            if p[0] < -1 or p[1] < -1 or p[0] > base_image.width or p[1] > base_image.height
        ]
        if outside:
            raise RuntimeError(
                f"{route}: {label} contains {len(outside)} points outside satellite bounds; "
                "coordinate conversion is not trustworthy"
            )

    route_dist = np.asarray(
        [_point_to_polyline_distance_m(p, reference, mpp) for p in final],
        dtype=np.float64,
    )
    print(
        f"[ROUTE-AUDIT] {route}: final-to-reference mean={route_dist.mean():.3f}m "
        f"p90={np.percentile(route_dist, 90):.3f}m max={route_dist.max():.3f}m",
        flush=True,
    )

    image = base_image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_reference(draw, reference, width)
    _draw_sparse_gt_dots(draw, true_gt, radius=max(3, width // 2), stride=3)
    draw.line(final, fill=(255, 45, 60, 255), width=width + 2, joint="curve")
    _draw_info_box(draw, route, summary, diagnostic=False)

    full = output_dir / f"{route}_final_vs_gt_full.jpg"
    image.save(full, quality=95)
    bounds = _crop_bounds((reference, true_gt, final), image.width, image.height)
    zoom = output_dir / f"{route}_final_vs_gt_zoom.jpg"
    image.crop(bounds).save(zoom, quality=95)

    diagnostic = base_image.copy()
    ddraw = ImageDraw.Draw(diagnostic, "RGBA")
    _draw_reference(ddraw, reference, width)
    _draw_sparse_gt_dots(ddraw, true_gt, radius=max(3, width // 2), stride=3)
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
        if route not in summaries:
            raise RuntimeError(f"Summary missing route {route}: {summary_path}")
        render(route, prepared_root, output_dir, summaries[route])


if __name__ == "__main__":
    main()
