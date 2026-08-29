#!/usr/bin/env python3
"""Visualize one BearingUAV pseudo-route on the full satellite city image.

The thick polyline is the planned multi-segment route used to order actual
BearingUAV samples.  If generated route metadata already exists, the selected
actual-sample trajectory is overlaid as a thinner line.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import prepare_bearinguav_routes as prep


def local_pixel_from_latlon(lat, lon, city, bounds):
    tl = bounds["geo_bounds"]["top_left"]
    br = bounds["geo_bounds"]["bottom_right"]
    gx = (float(lon) - float(tl["longitude"])) / (
        float(br["longitude"]) - float(tl["longitude"])
    ) * (prep.MAP_PX * 3 - 1)
    gy = (float(tl["latitude"]) - float(lat)) / (
        float(tl["latitude"]) - float(br["latitude"])
    ) * (prep.MAP_PX - 1)
    return np.asarray([gx - prep.CITY_OFFSETS[city], gy], dtype=np.float64)


def load_actual_path(route_name, city):
    sensor_path = prep.OUTPUT_ROOT / route_name / "sensor_with_yaw.json"
    geo_path = prep.OUTPUT_ROOT / "bearing_cities_abc_geo.json"
    if not sensor_path.exists() or not geo_path.exists():
        return None
    sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
    bounds = json.loads(geo_path.read_text(encoding="utf-8"))
    rows = sensor.get("timestamp", [])
    if not rows:
        return None
    return np.stack([
        local_pixel_from_latlon(row["latitude"], row["longitude"], city, bounds)
        for row in rows
    ])


def draw_route(route_name, output_path=None, show=False):
    if route_name not in prep.ROUTES:
        raise KeyError(f"Unknown route {route_name!r}; choose from {sorted(prep.ROUTES)}")

    city, waypoints = prep.ROUTES[route_name]
    image_path = prep.CITY_IMAGES[city]
    if not image_path.exists():
        raise FileNotFoundError(
            f"Satellite city image not found: {image_path}\n"
            "Check DATA_ROOT in prepare_bearinguav_routes.py."
        )

    if output_path is None:
        out_dir = prep.OUTPUT_ROOT / "route_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{route_name}_full_satellite_route.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        satellite = im.convert("RGB")

    wp = np.asarray(waypoints, dtype=np.float64)
    actual = load_actual_path(route_name, city)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=160)
    ax.imshow(satellite, extent=(0, prep.MAP_PX, prep.MAP_PX, 0))
    ax.plot(wp[:, 0], wp[:, 1], linewidth=3.0, label="Planned multi-segment route")
    ax.scatter(wp[:, 0], wp[:, 1], s=34, marker="o", label="Turn / route points")
    ax.scatter([wp[0, 0]], [wp[0, 1]], s=100, marker="^", label="Start")
    ax.scatter([wp[-1, 0]], [wp[-1, 1]], s=100, marker="s", label="End")

    if actual is not None and len(actual) > 1:
        ax.plot(
            actual[:, 0], actual[:, 1], linewidth=1.1, alpha=0.75,
            label="Selected actual BearingUAV samples",
        )

    ax.set_xlim(0, prep.MAP_PX)
    ax.set_ylim(prep.MAP_PX, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Satellite image x (px)")
    ax.set_ylabel("Satellite image y (px)")
    ax.set_title(f"BearingUAV {route_name} ({city}) - Full Satellite Route")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path.resolve())
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default="train_1", choices=sorted(prep.ROUTES))
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Write train_1/train_2/train_3 overview images instead of only --route.",
    )
    args = parser.parse_args()

    if args.all_train:
        for route_name in ("train_1", "train_2", "train_3"):
            draw_route(route_name, show=False)
    else:
        draw_route(args.route, args.output, args.show)


if __name__ == "__main__":
    main()
