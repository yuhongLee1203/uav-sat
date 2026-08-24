#!/usr/bin/env python3
"""Plot Route-B/C final localization trajectories on the georeferenced SAT map.

This reproduces the purpose/style of the old v36-exp *_actual_trajectory.png
figures for v36_byTeacher.  Only the reference trajectory and the FINAL
GRU+Polynomial+Kalman output are drawn; MeanShift is intentionally omitted.

Coordinates in the per-frame CSV are local metric XY, created by
meters_from_latlon().  They are converted back to latitude/longitude with the
same local tangent-plane definition and then mapped to the SAT image through
SatGeoMapper / the configured world-file affine transform.  No min/max fitting
or visual rescaling of trajectory coordinates is used.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from data import SatGeoMapper


EARTH_RADIUS_M = 6378137.0


def local_xy_to_latlon(x_m, y_m, origin_lat, origin_lon):
    """Exact inverse of data.meters_from_latlon for this fixed origin."""
    lat = float(origin_lat) + math.degrees(float(y_m) / EARTH_RADIUS_M)
    cos_lat0 = math.cos(math.radians(float(origin_lat)))
    lon = float(origin_lon) + math.degrees(
        float(x_m) / (EARTH_RADIUS_M * max(abs(cos_lat0), 1e-12))
    )
    return lat, lon


def xy_to_pixel(xy, mapper, origin_lat, origin_lon):
    pixels = []
    for x_m, y_m in np.asarray(xy, dtype=np.float64):
        lat, lon = local_xy_to_latlon(x_m, y_m, origin_lat, origin_lon)
        px, py = mapper.latlon_to_pixel(lat, lon)
        pixels.append((float(px), float(py)))
    return np.asarray(pixels, dtype=np.float64)


def read_final_csv(path):
    reference = []
    final = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"reference_x", "reference_y", "final_x", "final_y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"{path} is missing required final-trajectory fields: {sorted(missing)}"
            )
        for row in reader:
            reference.append((float(row["reference_x"]), float(row["reference_y"])))
            final.append((float(row["final_x"]), float(row["final_y"])))
    if not reference:
        raise RuntimeError(f"empty trajectory CSV: {path}")
    return np.asarray(reference), np.asarray(final)


def load_visual_origin():
    checkpoint_path = Path(config.VISUAL_CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"visual checkpoint not found: {checkpoint_path}\n"
            "The plot must use the same coordinate origin as the completed run."
        )
    payload = torch.load(checkpoint_path, map_location="cpu")
    return float(payload["origin_lat"]), float(payload["origin_lon"])


def route_crop_bounds(reference_px, final_px, image_size, padding_px):
    """Route-centred crop while retaining every in-map final prediction."""
    width, height = image_size
    points = np.concatenate([reference_px, final_px], axis=0)
    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1)
    )
    points = points[inside]
    if points.size == 0:
        return 0.0, float(width - 1), 0.0, float(height - 1)

    x0 = max(0.0, float(points[:, 0].min()) - padding_px)
    x1 = min(float(width - 1), float(points[:, 0].max()) + padding_px)
    y0 = max(0.0, float(points[:, 1].min()) - padding_px)
    y1 = min(float(height - 1), float(points[:, 1].max()) + padding_px)

    # Keep a useful minimum field of view for near-straight route segments.
    min_span = 500.0
    if x1 - x0 < min_span:
        cx = 0.5 * (x0 + x1)
        x0 = max(0.0, cx - 0.5 * min_span)
        x1 = min(float(width - 1), cx + 0.5 * min_span)
    if y1 - y0 < min_span:
        cy = 0.5 * (y0 + y1)
        y0 = max(0.0, cy - 0.5 * min_span)
        y1 = min(float(height - 1), cy + 0.5 * min_span)
    return x0, x1, y0, y1


def plot_route(route_name, csv_path, output_path, mapper, origin_lat, origin_lon, padding_px):
    reference_xy, final_xy = read_final_csv(csv_path)
    reference_px = xy_to_pixel(reference_xy, mapper, origin_lat, origin_lon)
    final_px = xy_to_pixel(final_xy, mapper, origin_lat, origin_lon)

    with Image.open(config.SAT_IMAGE) as image:
        sat = np.asarray(image.convert("RGB"))
        image_size = image.size

    x0, x1, y0, y1 = route_crop_bounds(
        reference_px, final_px, image_size, float(padding_px)
    )

    # Match the old v36-exp actual-trajectory presentation: satellite image as
    # the real background, continuous reference/final trajectories, and clear
    # start/end markers.  No MeanShift points/line are shown here.
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.imshow(sat, origin="upper")

    ax.plot(
        reference_px[:, 0],
        reference_px[:, 1],
        color="red",
        linewidth=2.4,
        linestyle="-",
        label="Reference trajectory",
        zorder=4,
    )
    ax.plot(
        final_px[:, 0],
        final_px[:, 1],
        color="deepskyblue",
        linewidth=2.2,
        linestyle="-",
        label="Final localization",
        zorder=5,
    )

    ax.scatter(
        reference_px[0, 0], reference_px[0, 1],
        s=95, marker="o", color="lime", edgecolors="black", linewidths=0.8,
        label="Start", zorder=7,
    )
    ax.scatter(
        reference_px[-1, 0], reference_px[-1, 1],
        s=115, marker="X", color="yellow", edgecolors="black", linewidths=0.8,
        label="End", zorder=7,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{route_name.replace('_', ' ').title()} - Final Localization Trajectory",
        fontsize=15,
    )
    ax.set_xlabel("Satellite-map pixel X")
    ax.set_ylabel("Satellite-map pixel Y")
    ax.legend(loc="best", framealpha=0.90)
    ax.grid(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {route_name}: {output_path}", flush=True)


def resolve_csv(output_root, route_name, frame_count, forward_rows):
    frame_dir = Path(output_root) / f"{int(frame_count)}frame"
    exact = frame_dir / (
        f"{route_name}_forward{int(forward_rows)}x6_v36_byTeacher_frames.csv"
    )
    if exact.exists():
        return exact

    matches = sorted(
        p for p in frame_dir.glob(f"{route_name}_*frames.csv")
        if f"forward{int(forward_rows)}x6" in p.name
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"cannot find {route_name} final CSV under {frame_dir} for "
            f"forward {forward_rows}x6"
        )
    raise RuntimeError(f"ambiguous {route_name} CSVs: {matches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-count", type=int, default=2)
    parser.add_argument("--forward-rows", type=int, default=3, choices=(3, 4, 5, 6))
    parser.add_argument("--padding-px", type=float, default=140.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: output/<backbone>/<frame>frame/figures",
    )
    args = parser.parse_args()

    output_root = Path(config.BACKBONE_OUTPUT_DIR)
    frame_dir = output_root / f"{args.frame_count}frame"
    output_dir = args.output_dir or (frame_dir / "figures")

    origin_lat, origin_lon = load_visual_origin()
    mapper = SatGeoMapper(config.SAT_JSON, config.SAT_IMAGE)

    print(f"SAT background: {config.SAT_IMAGE}", flush=True)
    print(f"SAT georef:     {config.SAT_JSON}", flush=True)
    print(f"origin:         ({origin_lat:.9f}, {origin_lon:.9f})", flush=True)

    for route_name in ("route_B", "route_C"):
        csv_path = resolve_csv(
            output_root, route_name, args.frame_count, args.forward_rows
        )
        output_path = output_dir / (
            f"v36_byTeacher_{args.frame_count}frame_"
            f"{args.forward_rows}x6_{route_name}_actual_trajectory.png"
        )
        plot_route(
            route_name,
            csv_path,
            output_path,
            mapper,
            origin_lat,
            origin_lon,
            args.padding_px,
        )


if __name__ == "__main__":
    main()
