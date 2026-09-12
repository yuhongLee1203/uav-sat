#!/usr/bin/env python3
"""Plot autonomous Route-B/C final trajectories on the georeferenced SAT map."""

import argparse
import csv
import math
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
    lat = float(origin_lat) + math.degrees(float(y_m) / EARTH_RADIUS_M)
    lon = float(origin_lon) + math.degrees(
        float(x_m)
        / (
            EARTH_RADIUS_M
            * max(abs(math.cos(math.radians(float(origin_lat)))), 1e-12)
        )
    )
    return lat, lon


def xy_to_pixel(xy, mapper, origin_lat, origin_lon):
    result = []
    for x_m, y_m in np.asarray(xy, dtype=np.float64):
        lat, lon = local_xy_to_latlon(x_m, y_m, origin_lat, origin_lon)
        px, py = mapper.latlon_to_pixel(lat, lon)
        result.append([float(px), float(py)])
    return np.asarray(result, dtype=np.float64)


def read_csv(path):
    reference, final, ms1, kalman = [], [], [], []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "reference_x", "reference_y", "final_x", "final_y",
            "ms1_x", "ms1_y", "kalman_x_prime_x", "kalman_x_prime_y",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} missing fields: {sorted(missing)}")
        for row in reader:
            reference.append([float(row["reference_x"]), float(row["reference_y"])])
            final.append([float(row["final_x"]), float(row["final_y"])])
            ms1.append([float(row["ms1_x"]), float(row["ms1_y"])])
            kalman.append(
                [float(row["kalman_x_prime_x"]), float(row["kalman_x_prime_y"])]
            )
    if not reference:
        raise RuntimeError(f"empty CSV: {path}")
    return (
        np.asarray(reference),
        np.asarray(final),
        np.asarray(ms1),
        np.asarray(kalman),
    )


def load_origin():
    payload = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    return float(payload["origin_lat"]), float(payload["origin_lon"])


def crop_bounds(*pixel_sets, image_size, padding_px):
    points = np.concatenate(pixel_sets, axis=0)
    width, height = image_size
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if not len(points):
        return 0.0, float(width - 1), 0.0, float(height - 1)
    x0 = max(0.0, float(points[:, 0].min()) - padding_px)
    x1 = min(float(width - 1), float(points[:, 0].max()) + padding_px)
    y0 = max(0.0, float(points[:, 1].min()) - padding_px)
    y1 = min(float(height - 1), float(points[:, 1].max()) + padding_px)
    return x0, x1, y0, y1


def plot_route(route_name, csv_path, output_path, show_intermediate=False, padding_px=140.0):
    origin_lat, origin_lon = load_origin()
    mapper = SatGeoMapper(config.SAT_JSON, config.SAT_IMAGE)
    reference, final, ms1, kalman = read_csv(csv_path)
    reference_px = xy_to_pixel(reference, mapper, origin_lat, origin_lon)
    final_px = xy_to_pixel(final, mapper, origin_lat, origin_lon)
    ms1_px = xy_to_pixel(ms1, mapper, origin_lat, origin_lon)
    kalman_px = xy_to_pixel(kalman, mapper, origin_lat, origin_lon)

    with Image.open(config.SAT_IMAGE) as image:
        sat = np.asarray(image.convert("RGB"))
        image_size = image.size

    x0, x1, y0, y1 = crop_bounds(
        reference_px, final_px, image_size=image_size, padding_px=float(padding_px)
    )

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.imshow(sat, origin="upper")
    ax.plot(reference_px[:, 0], reference_px[:, 1], linewidth=2.8, label="Reference")
    if show_intermediate:
        ax.plot(ms1_px[:, 0], ms1_px[:, 1], linewidth=1.0, alpha=0.7, label="MS #1")
        ax.plot(
            kalman_px[:, 0], kalman_px[:, 1],
            linewidth=1.0, alpha=0.7, label="Kalman X'"
        )
    ax.plot(final_px[:, 0], final_px[:, 1], linewidth=2.2, label="Final MS #2")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] {route_name}: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routes", nargs="+", choices=config.ROUTE_NAMES, default=["route_C", "route_B"]
    )
    parser.add_argument("--show-intermediate", action="store_true")
    parser.add_argument("--padding-px", type=float, default=140.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (Path(config.OUTPUT_DIR) / "figures")
    for route_name in args.routes:
        csv_path = Path(config.OUTPUT_DIR) / (
            f"{route_name}_autonomous_ms1_kf_gru_ms2_frames.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path}\nRun autonomous evaluation first: "
                "python robust_tracker.py --mode eval --eval-routes route_C route_B"
            )
        plot_route(
            route_name,
            csv_path,
            output_dir / f"{route_name}_autonomous_final_trajectory.png",
            show_intermediate=bool(args.show_intermediate),
            padding_px=float(args.padding_px),
        )


if __name__ == "__main__":
    main()
