#!/usr/bin/env python3
"""Visualize exact route references separately from BearingUAV source samples.

The previous figure connected selected independent BearingUAV samples with a
polyline.  That was visually misleading: the resulting jagged line was NOT the
route/reference line.  This version draws the exact piecewise-straight planned
route as the route line and draws source samples only as points.

Use --show-jitter to additionally preview the same bounded deterministic smooth
jitter pattern used by the shared v36 controlled local-prior protocol.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
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


def load_bounds():
    path = prep.OUTPUT_ROOT / "bearing_citya_geo.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_sensor_geometry(route_name):
    sensor_path = prep.OUTPUT_ROOT / route_name / "sensor_with_yaw.json"
    bounds = load_bounds()
    if not sensor_path.exists() or bounds is None:
        return None, None, []
    payload = json.loads(sensor_path.read_text(encoding="utf-8"))
    rows = payload.get("timestamp", [])
    if not rows:
        return None, None, []

    source = np.stack([
        pixel_from_latlon(row["latitude"], row["longitude"], bounds)
        for row in rows
    ])
    if all("reference_latitude" in row and "reference_longitude" in row for row in rows):
        reference = np.stack([
            pixel_from_latlon(row["reference_latitude"], row["reference_longitude"], bounds)
            for row in rows
        ])
    else:
        reference = None
    return source, reference, rows


def load_waypoint_pixels(route_name):
    waypoint_path = prep.OUTPUT_ROOT / "waypoints" / f"{route_name}_waypoints.json"
    bounds = load_bounds()
    if not waypoint_path.exists() or bounds is None:
        return None, []
    payload = json.loads(waypoint_path.read_text(encoding="utf-8"))
    rows = payload.get("waypoints", [])
    if not rows:
        return None, []
    pixels = np.stack([
        pixel_from_latlon(row["latitude"], row["longitude"], bounds)
        for row in rows
    ])
    return pixels, rows


def deterministic_smooth_jitter_pixels(route_name, source_pixels, rows, maximum_m):
    """Preview robust_tracker_base.controlled_gt_prior_se smooth jitter."""
    maximum = float(maximum_m)
    if maximum <= 0.0 or source_pixels is None:
        return source_pixels

    route_code = sum(ord(ch) for ch in str(route_name))
    angular_rate = float(config.CONTROLLED_GT_PRIOR_JITTER_ANGULAR_RATE)
    radius_rate = float(config.CONTROLLED_GT_PRIOR_JITTER_RADIUS_RATE)
    lo = float(config.CONTROLLED_GT_PRIOR_JITTER_MIN_FRACTION)
    hi = float(config.CONTROLLED_GT_PRIOR_JITTER_MAX_FRACTION)

    output = []
    for index, (base, row) in enumerate(zip(source_pixels, rows)):
        frame_id = int(index)
        image = str(row.get("image", ""))
        if image.startswith("vi_"):
            try:
                frame_id = int(Path(image).stem.split("_")[-1])
            except ValueError:
                pass
        angle = 0.11 * route_code + angular_rate * float(frame_id)
        radius_phase = 0.07 * route_code + radius_rate * float(frame_id)
        radius_fraction = lo + (hi - lo) * (0.5 + 0.5 * math.sin(radius_phase))
        radius_m = maximum * radius_fraction
        jitter_xy_m = np.asarray([
            radius_m * math.cos(angle),
            radius_m * math.sin(angle),
        ])
        # Generated georef is x-east / y-north while image y grows downward.
        jitter_px = np.asarray([
            jitter_xy_m[0] / prep.METERS_PER_PIXEL,
            -jitter_xy_m[1] / prep.METERS_PER_PIXEL,
        ])
        output.append(base + jitter_px)
    return np.stack(output)


def draw_route(route_name, output_path=None, show=False, show_jitter=False, jitter_m=8.0):
    if route_name not in prep.ROUTES:
        raise KeyError(f"Unknown route {route_name!r}; choose from {sorted(prep.ROUTES)}")

    city, planned_waypoints = prep.ROUTES[route_name]
    if city != prep.SCENE:
        raise RuntimeError(f"{route_name} is not on the configured scene {prep.SCENE}")
    if not prep.CITY_IMAGE.exists():
        raise FileNotFoundError(prep.CITY_IMAGE)

    if output_path is None:
        out_dir = prep.OUTPUT_ROOT / "route_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{route_name}_full_satellite_route.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(prep.CITY_IMAGE) as image:
        satellite = image.convert("RGB")

    planned = np.asarray(planned_waypoints, dtype=np.float64)
    source, reference, sensor_rows = load_sensor_geometry(route_name)
    waypoint_pixels, waypoint_rows = load_waypoint_pixels(route_name)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=160)
    ax.imshow(satellite, extent=(0, prep.MAP_PX, prep.MAP_PX, 0))

    ax.plot(
        planned[:, 0], planned[:, 1], linewidth=3.0,
        label="Exact straight-segment route/reference line",
    )
    ax.scatter(
        planned[:, 0], planned[:, 1], s=30, marker="o",
        label="Planned segment junctions",
    )

    if reference is not None and len(reference) > 0:
        ax.scatter(
            reference[:, 0], reference[:, 1], s=7, alpha=0.45,
            label=f"Per-frame route reference points (n={len(reference)})",
        )

    # Do NOT connect independent BearingUAV samples.  They are image/position
    # source samples near the route, not the route/reference trajectory itself.
    if source is not None and len(source) > 0:
        ax.scatter(
            source[:, 0], source[:, 1], s=9, alpha=0.35,
            label=f"Selected real source samples (points only, n={len(source)})",
        )

    if show_jitter and source is not None:
        jittered = deterministic_smooth_jitter_pixels(
            route_name, source, sensor_rows, maximum_m=jitter_m
        )
        ax.scatter(
            jittered[:, 0], jittered[:, 1], s=8, alpha=0.45,
            label=f"Runtime smooth-jitter prior preview ({jitter_m:g} m max)",
        )

    if waypoint_pixels is not None:
        ax.scatter(
            waypoint_pixels[:, 0], waypoint_pixels[:, 1], s=58, marker="x",
            label="Exact route waypoints",
        )
        for p, row in zip(waypoint_pixels, waypoint_rows):
            label = f"W{row['waypoint_order']}:{row['role']}"
            ax.annotate(
                label, (p[0], p[1]), xytext=(4, 4),
                textcoords="offset points", fontsize=7,
            )

    ax.scatter([planned[0, 0]], [planned[0, 1]], s=90, marker="^", label="Planned start")
    ax.scatter([planned[-1, 0]], [planned[-1, 1]], s=90, marker="s", label="Planned end")
    ax.set_xlim(0, prep.MAP_PX)
    ax.set_ylim(prep.MAP_PX, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Satellite image x (px)")
    ax.set_ylabel("Satellite image y (px)")
    ax.set_title(f"BearingUAV {route_name}: exact route line vs. source samples")
    ax.legend(loc="best", fontsize=8)
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
        ax.plot(
            planned[:, 0], planned[:, 1], linewidth=2.6,
            label=f"{route_name} exact reference route",
        )
        ax.scatter(planned[:, 0], planned[:, 1], s=20, marker="o")
        source, _reference, _rows = load_sensor_geometry(route_name)
        if source is not None:
            ax.scatter(
                source[:, 0], source[:, 1], s=6, alpha=0.20,
                label=f"{route_name} source samples",
            )

    ax.set_xlim(0, prep.MAP_PX)
    ax.set_ylim(prep.MAP_PX, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Satellite image x (px)")
    ax.set_ylabel("Satellite image y (px)")
    ax.set_title("BearingUAV: exact piecewise-straight references on ONE city-A scene")
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
    parser.add_argument("--show-jitter", action="store_true")
    parser.add_argument("--jitter-m", type=float, default=8.0)
    parser.add_argument("--all", action="store_true", help="Draw all 3 exact routes on one satellite image.")
    parser.add_argument("--all-train", action="store_true", help="Write train_1 and train_2 route figures.")
    args = parser.parse_args()

    if args.all:
        draw_all_same_scene(args.output, args.show)
    elif args.all_train:
        for route_name in ("train_1", "train_2"):
            draw_route(route_name, show=False, show_jitter=args.show_jitter, jitter_m=args.jitter_m)
    else:
        draw_route(
            args.route, args.output, args.show,
            show_jitter=args.show_jitter, jitter_m=args.jitter_m,
        )


if __name__ == "__main__":
    main()
