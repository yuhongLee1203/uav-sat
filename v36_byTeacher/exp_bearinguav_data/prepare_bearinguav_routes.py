"""Build 3 train / 1 validation / 1 test ordered routes from BearingUAV."""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/yh/study/cvpr_data/Bearing_UAV_90K")
OUTPUT_ROOT = HERE / "generated_routes_3train_1val_1test"
MAP_PX = 4096
METERS_PER_PIXEL = 0.25
CITY_OFFSETS = {"citya": 0, "cityb": MAP_PX, "cityc": MAP_PX * 2}
CITY_IMAGES = {
    "citya": DATA_ROOT / "city_rsi/35.67091338738739_139.69289911300856_1791.95_1024_1024_4326_city.jpg",
    "cityb": DATA_ROOT / "city_rsi/25.030947387387386_121.51462868800057_1791.95_1024_1024_4326_city.jpg",
    "cityc": DATA_ROOT / "city_rsi/1.2897673873873876_103.84197619336068_1791.95_1024_1024_4326_city.jpg",
}


def horizontal_snake(rows=12, x0=384, x1=3712, y0=400, y1=3696):
    ys = np.linspace(y0, y1, rows)
    return [(x0, float(y)) if index % 2 == 0 else (x1, float(y))
            for index, y in enumerate(ys)]


def vertical_snake(cols=12, y0=384, y1=3712, x0=400, x1=3696):
    xs = np.linspace(x0, x1, cols)
    return [(float(x), y0) if index % 2 == 0 else (float(x), y1)
            for index, x in enumerate(xs)]


def diagonal_snake(legs=14, x0=500, x1=3596, y0=400, y1=3696):
    ys = np.linspace(y0, y1, legs)
    return [(x0, float(y)) if index % 2 == 0 else (x1, float(y))
            for index, y in enumerate(ys)]


ROUTES = {
    "train_1": ("citya", horizontal_snake()),
    "train_2": ("citya", vertical_snake()),
    "train_3": ("citya", diagonal_snake()),
    # Validation and test use entirely different cities, so neither UAV views
    # nor satellite appearance leak from training.
    "val_1": ("cityb", horizontal_snake(rows=12, x0=384, x1=3712, y0=400, y1=3696)),
    "test_1": ("cityc", vertical_snake(cols=12, y0=384, y1=3712, x0=400, x1=3696)),
}


def resolve_path(raw):
    path = Path(raw)
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if parts and parts[0] == "Bearing_UAV_90K":
        parts = parts[1:]
    return DATA_ROOT.joinpath(*parts)


def row_pixel(row):
    return np.asarray([
        int(row["block_x"]) * 256.0 + (float(row["x_norm"]) + 1.0) * 256.0,
        int(row["block_y"]) * 256.0 + (float(row["y_norm"]) + 1.0) * 256.0,
    ], dtype=np.float64)


def sample_polyline(waypoints, spacing_m):
    spacing_px = spacing_m / METERS_PER_PIXEL
    samples = []
    for leg, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        steps = max(1, int(round(np.linalg.norm(end - start) / spacing_px)))
        for step in range(steps + 1):
            if leg > 0 and step == 0:
                continue
            samples.append({
                "pixel": start + (step / float(steps)) * (end - start),
                "waypoint_order": leg if step == 0 else (leg + 1 if step == steps else None),
            })
    return samples


def attach_nearby_views(desired, tree):
    recent = deque(maxlen=8)
    selected = []
    for item in desired:
        distances, indices = tree.query(item["pixel"], k=32)
        choice = None
        for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if int(index) not in recent:
                choice = (int(index), float(distance) * METERS_PER_PIXEL)
                break
        if choice is None:
            choice = (int(np.atleast_1d(indices)[0]),
                      float(np.atleast_1d(distances)[0]) * METERS_PER_PIXEL)
        recent.append(choice[0])
        selected.append(dict(item, source_index=choice[0], source_error_m=choice[1]))
    return selected


def make_mosaic(rebuild=False):
    path = OUTPUT_ROOT / "bearing_cities_abc.jpg"
    if path.exists() and not rebuild:
        return path
    mosaic = Image.new("RGB", (MAP_PX * 3, MAP_PX))
    for city, image_path in CITY_IMAGES.items():
        with Image.open(image_path) as image:
            mosaic.paste(image.convert("RGB"), (CITY_OFFSETS[city], 0))
    mosaic.save(path, quality=95)
    return path


def mosaic_georef():
    width_px, height_px = MAP_PX * 3, MAP_PX
    top_lat, left_lon = 23.600000, 120.000000
    lat_deg_per_m = 1.0 / 111320.0
    lon_deg_per_m = 1.0 / (111320.0 * math.cos(math.radians(top_lat)))
    bottom_lat = top_lat - height_px * METERS_PER_PIXEL * lat_deg_per_m
    right_lon = left_lon + width_px * METERS_PER_PIXEL * lon_deg_per_m
    payload = {
        "geo_bounds": {
            "top_left": {"latitude": top_lat, "longitude": left_lon},
            "bottom_right": {"latitude": bottom_lat, "longitude": right_lon},
        },
        "meters_per_pixel": METERS_PER_PIXEL,
    }
    path = OUTPUT_ROOT / "bearing_cities_abc_geo.json"
    path.write_text(json.dumps(payload, indent=2))
    return payload, path


def global_pixel_to_latlon(pixel, bounds):
    top_left = bounds["geo_bounds"]["top_left"]
    bottom_right = bounds["geo_bounds"]["bottom_right"]
    lon = top_left["longitude"] + pixel[0] / (MAP_PX * 3 - 1) * (
        bottom_right["longitude"] - top_left["longitude"]
    )
    lat = top_left["latitude"] - pixel[1] / (MAP_PX - 1) * (
        top_left["latitude"] - bottom_right["latitude"]
    )
    return float(lat), float(lon)


def write_route(name, city, selected, rows, bounds, spacing_m, rebuild=False):
    root, vi = OUTPUT_ROOT / name, OUTPUT_ROOT / name / "vi"
    root.mkdir(parents=True, exist_ok=True)
    vi.mkdir(parents=True, exist_ok=True)
    timestamps, waypoints = [], []
    x_offset = CITY_OFFSETS[city]
    base_timestamp = 1_900_000_000_000_000_000
    for frame_index, item in enumerate(selected):
        row = rows[item["source_index"]]
        source = resolve_path(row.get("target_patch_3d") or row["target_path"])
        target = vi / ("vi_%06d.jpg" % frame_index)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
        global_pixel = item["pixel"] + np.asarray([x_offset, 0.0])
        lat, lon = global_pixel_to_latlon(global_pixel, bounds)
        timestamp_ns = base_timestamp + frame_index * 333_333_333
        timestamps.append({"timestamp_ns": timestamp_ns, "image": target.name,
                           "latitude": lat, "longitude": lon, "altitude": 120.0})
        if item["waypoint_order"] is not None:
            waypoints.append({
                "waypoint_order": int(item["waypoint_order"]),
                "role": "start" if int(item["waypoint_order"]) == 0 else "turn",
                "frame_index": frame_index, "image": target.name,
                "timestamp_ns": timestamp_ns, "latitude": lat, "longitude": lon,
                "altitude_m": 120.0, "source_bearinguav_index": int(item["source_index"]),
            })
    (root / "sensor_with_yaw.json").write_text(json.dumps({"timestamp": timestamps}, indent=2))
    waypoint_dir = OUTPUT_ROOT / "waypoints"; waypoint_dir.mkdir(exist_ok=True)
    (waypoint_dir / (name + "_waypoints.json")).write_text(json.dumps({
        "route": name, "split": name.split("_")[0], "city": city,
        "reference_spacing_m": spacing_m, "waypoints": waypoints,
    }, indent=2))
    steps = [np.linalg.norm(selected[i]["pixel"] - selected[i - 1]["pixel"]) * METERS_PER_PIXEL
             for i in range(1, len(selected))]
    errors = [item["source_error_m"] for item in selected]
    summary = {
        "route": name, "split": name.split("_")[0], "city": city,
        "frames": len(selected), "waypoints": len(waypoints),
        "step_mean_m": float(np.mean(steps)), "step_p90_m": float(np.quantile(steps, .9)),
        "source_match_mean_m": float(np.mean(errors)),
        "source_match_p90_m": float(np.quantile(errors, .9)),
    }
    (root / "route_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing-m", type=float, default=4.5)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mosaic = make_mosaic(args.rebuild)
    bounds, georef = mosaic_georef()
    city_data = {}
    for city in CITY_OFFSETS:
        metadata = DATA_ROOT / city / "rawmetadata.csv"
        rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
        points = np.stack([row_pixel(row) for row in rows])
        city_data[city] = (rows, cKDTree(points))
    summaries = []
    for name, (city, waypoint_pixels) in ROUTES.items():
        rows, tree = city_data[city]
        selected = attach_nearby_views(sample_polyline(waypoint_pixels, args.spacing_m), tree)
        summary = write_route(name, city, selected, rows, bounds, args.spacing_m, args.rebuild)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (OUTPUT_ROOT / "generation_summary.json").write_text(json.dumps({
        "split": {"train": ["train_1", "train_2", "train_3"],
                  "validation": ["val_1"], "test": ["test_1"]},
        "satellite_mosaic": str(mosaic), "satellite_georef": str(georef),
        "routes": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
