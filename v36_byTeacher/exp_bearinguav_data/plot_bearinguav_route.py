#!/usr/bin/env python3
"""Visualize a BearingUAV multi-turn pseudo-route on the full city satellite map."""

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


def load_generated(route_name, city):
    geo_path = prep.OUTPUT_ROOT / "bearing_cities_abc_geo.json"
    sensor_path = prep.OUTPUT_ROOT / route_name / "sensor_with_yaw.json"
    waypoint_path = prep.OUTPUT_ROOT / "waypoints" / f"{route_name}_waypoints.json"
    if not geo_path.exists():
        return None, []
    bounds = json.loads(geo_path.read_text(encoding="utf-8"))

    actual = None
    if sensor_path.exists():
        rows = json.loads(sensor_path.read_text(encoding="utf-8")).get("timestamp", [])
        if rows:
            actual = np.stack([
                local_pixel_from_latlon(row["latitude"], row["longitude"], city, bounds)
                for row in rows
            ])

    generated_waypoints = []
    if waypoint_path.exists():
        rows = json.loads(waypoint_path.read_text(encoding="utf-8")).get("waypoints", [])
        for row in rows:
            generated_waypoints.append({
                **row,
                "pixel": local_pixel_from_latlon(row["latitude"], row["longitude"], city, bounds),
            })
    return actual, generated_waypoints


def draw_route(route_name, output_path=None, show=False):
    if route_name not in prep.ROUTES:
        raise KeyError(f"Unknown route {route_name!r}; choose from {sorted(prep.ROUTES)}")

    city, planned_points = prep.ROUTES[route_name]
    profile = prep.ROUTE_PROFILES[route_name]
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

    planned = np.asarray(planned_points, dtype=np.float64)
    actual, generated_waypoints = load_generated(route_name, city)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=160)
    ax.imshow(satellite, extent=(0, prep.MAP_PX, prep.MAP_PX, 0))
    ax.plot(planned[:, 0], planned[:, 1], linewidth=3.0, label="Planned straight-leg route")
    ax.scatter(planned[:, 0], planned[:, 1], s=28, marker="o", label="Planned turn vertices")
    ax.scatter([planned[0, 0]], [planned[0, 1]], s=110, marker="^", label="Planned start")
    ax.scatter([planned[-1, 0]], [planned[-1, 1]], s=110, marker="s", label="Planned end")

    if actual is not None and len(actual) > 1:
        ax.plot(actual[:, 0], actual[:, 1], linewidth=1.2, alpha=0.78,
                label="Selected real BearingUAV samples")

    if generated_waypoints:
        pixels = np.stack([row["pixel"] for row in generated_waypoints])
        ax.scatter(pixels[:, 0], pixels[:, 1], s=58, marker="D",
                   label="Generated flight waypoints")
        for row in generated_waypoints:
            x, y = row["pixel"]
            order = int(row["waypoint_order"])
            role = str(row["role"])
            ax.annotate(f"W{order}:{role}", (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7)

    ax.set_xlim(0, prep.MAP_PX)
    ax.set_ylim(prep.MAP_PX, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Satellite image x (px)")
    ax.set_ylabel("Satellite image y (px)")
    ax.set_title(
        f"BearingUAV {route_name} ({city}) - multi-turn route | "
        f"{profile['name']} target {profile['base_step_m']:.1f} m/frame"
    )
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
        help="Write train_1/train_2/train_3 full-satellite route images.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write train/validation/test full-satellite route images.",
    )
    args = parser.parse_args()

    if args.all:
        for route_name in prep.ROUTES:
            draw_route(route_name, show=False)
    elif args.all_train:
        for route_name in ("train_1", "train_2", "train_3"):
            draw_route(route_name, show=False)
    else:
        draw_route(args.route, args.output, args.show)


if __name__ == "__main__":
    main()
