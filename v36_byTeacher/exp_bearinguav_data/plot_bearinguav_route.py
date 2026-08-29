#!/usr/bin/env python3
"""Visualize same-scene BearingUAV routes on the city-A satellite image."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import prepare_bearinguav_routes as prep


def pixel_from_latlon(lat, lon, bounds):
    tl = bounds["geo_bounds"]["top_left"]
    br = bounds["geo_bounds"]["bottom_right"]
    x = (float(lon) - float(tl["longitude"])) / (
        float(br["longitude"]) - float(tl["longitude"])
    ) * (prep.MAP_PX - 1)
    y = (float(tl["latitude"]) - float(lat)) / (
        float(tl["latitude"]) - float(br["latitude"])
    ) * (prep.MAP_PX - 1)
    return np.asarray([x, y], dtype=np.float64)


def load_actual_path(route_name):
    sensor_path = prep.OUTPUT_ROOT / route_name / "sensor_with_yaw.json"
    geo_path = prep.OUTPUT_ROOT / "bearing_citya_geo.json"
    if not sensor_path.exists() or not geo_path.exists():
        return None
    sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
    bounds = json.loads(geo_path.read_text(encoding="utf-8"))
    rows = sensor.get("timestamp", [])
    if not rows:
        return None
    return np.stack([
        pixel_from_latlon(row["latitude"], row["longitude"], bounds)
        for row in rows
    ])


def load_waypoint_pixels(route_name):
    waypoint_path = prep.OUTPUT_ROOT / "waypoints" / f"{route_name}_waypoints.json"
    geo_path = prep.OUTPUT_ROOT / "bearing_citya_geo.json"
    if not waypoint_path.exists() or not geo_path.exists():
        return None, []
    payload = json.loads(waypoint_path.read_text(encoding="utf-8"))
    bounds = json.loads(geo_path.read_text(encoding="utf-8"))
    waypoints = payload.get("waypoints", [])
    if not waypoints:
        return None, []
    pixels = np.stack([
        pixel_from_latlon(row["latitude"], row["longitude"], bounds)
        for row in waypoints
    ])
    return pixels, waypoints


def draw_route(route_name, output_path=None, show=False):
    if route_name not in prep.ROUTES:
        raise KeyError(f"Unknown route {route_name!r}; choose from {sorted(prep.ROUTES)}")

    city, planned_waypoints = prep.ROUTES[route_name]
    if city != prep.SCENE:
        raise RuntimeError(f"{route_name} is not on the configured same scene {prep.SCENE}")
    image_path = prep.CITY_IMAGE
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if output_path is None:
        out_dir = prep.OUTPUT_ROOT / "route_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{route_name}_full_satellite_route.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        satellite = image.convert("RGB")

    planned = np.asarray(planned_waypoints, dtype=np.float64)
    actual = load_actual_path(route_name)
    selected_wp, waypoint_rows = load_waypoint_pixels(route_name)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=160)
    ax.imshow(satellite, extent=(0, prep.MAP_PX, prep.MAP_PX, 0))
    ax.plot(planned[:, 0], planned[:, 1], linewidth=2.8, label="Planned multi-L route")
    ax.scatter(planned[:, 0], planned[:, 1], s=24, marker="o", label="Planned turn points")

    if actual is not None and len(actual) > 1:
        ax.plot(actual[:, 0], actual[:, 1], linewidth=1.0, alpha=0.72,
                label="Selected 1000 real samples")

    if selected_wp is not None:
        ax.scatter(selected_wp[:, 0], selected_wp[:, 1], s=55, marker="x",
                   label="Generated waypoints")
        for p, row in zip(selected_wp, waypoint_rows):
            label = f"W{row['waypoint_order']}:{row['role']}"
            ax.annotate(label, (p[0], p[1]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    ax.scatter([planned[0, 0]], [planned[0, 1]], s=90, marker="^", label="Start")
    ax.scatter([planned[-1, 0]], [planned[-1, 1]], s=90, marker="s", label="Planned end")
    ax.set_xlim(0, prep.MAP_PX)
    ax.set_ylim(prep.MAP_PX, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Satellite image x (px)")
    ax.set_ylabel("Satellite image y (px)")
    ax.set_title(f"BearingUAV {route_name} - SAME city-A satellite scene")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path.resolve())
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def draw_all_same_scene(output_path=None, show=False):
    if output_path is None:
        out_dir = prep.OUTPUT_ROOT / "route_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "all_routes_same_satellite_scene.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(prep.CITY_IMAGE) as image:
        satellite = image.convert("RGB")

    fig, ax = plt.subplots(figsize=(12, 12), dpi=170)
    ax.imshow(satellite, extent=(0, prep.MAP_PX, prep.MAP_PX, 0))

    for route_name in ("train_1", "train_2", "val_1"):
        _, planned_waypoints = prep.ROUTES[route_name]
        planned = np.asarray(planned_waypoints, dtype=np.float64)
        ax.plot(planned[:, 0], planned[:, 1], linewidth=2.4, label=f"{route_name} planned")
        actual = load_actual_path(route_name)
        if actual is not None and len(actual) > 1:
            ax.plot(actual[:, 0], actual[:, 1], linewidth=0.9, alpha=0.65,
                    label=f"{route_name} selected")
        selected_wp, _ = load_waypoint_pixels(route_name)
        if selected_wp is not None:
            ax.scatter(selected_wp[:, 0], selected_wp[:, 1], s=32, marker="x")

    ax.set_xlim(0, prep.MAP_PX)
    ax.set_ylim(prep.MAP_PX, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Satellite image x (px)")
    ax.set_ylabel("Satellite image y (px)")
    ax.set_title("BearingUAV: 2 Train + 1 Validation on ONE city-A Satellite Image")
    ax.legend(loc="best", fontsize=8)
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
    parser.add_argument("--all", action="store_true", help="Draw all 3 routes on one satellite image.")
    parser.add_argument("--all-train", action="store_true", help="Write train_1 and train_2 route figures.")
    args = parser.parse_args()

    if args.all:
        draw_all_same_scene(args.output, args.show)
    elif args.all_train:
        for route_name in ("train_1", "train_2"):
            draw_route(route_name, show=False)
    else:
        draw_route(args.route, args.output, args.show)


if __name__ == "__main__":
    main()
