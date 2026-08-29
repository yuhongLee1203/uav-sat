"""Build physically ordered BearingUAV routes using each UAV sample's actual pose.

This adapter keeps v36's sequential route format but avoids the previous
fixed-spacing synthetic labels.  A smooth planned polyline is used only to pick
an ordered set of nearby BearingUAV samples.  The written position for every
frame is the selected source sample's own metadata position, so image and label
refer to the same place.  Consecutive frames are filtered to a natural variable
step range rather than being forced to one constant displacement.
"""

import argparse
import csv
import json
import math
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
    """Actual BearingUAV sample center in the 4096x4096 city map."""
    return np.asarray([
        int(row["block_x"]) * 256.0 + (float(row["x_norm"]) + 1.0) * 256.0,
        int(row["block_y"]) * 256.0 + (float(row["y_norm"]) + 1.0) * 256.0,
    ], dtype=np.float64)


def sample_polyline(waypoints, spacing_m):
    """Dense query points used only to discover nearby real UAV samples."""
    spacing_px = spacing_m / METERS_PER_PIXEL
    samples = []
    for leg, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        steps = max(1, int(math.ceil(np.linalg.norm(end - start) / spacing_px)))
        for step in range(steps + 1):
            if leg > 0 and step == 0:
                continue
            alpha = step / float(steps)
            samples.append({
                "pixel": start + alpha * (end - start),
                "leg": int(leg),
                "at_leg_start": bool(step == 0),
                "at_leg_end": bool(step == steps),
            })
    return samples


def select_actual_sequence(query_points, tree, source_points, min_step_m, max_step_m,
                           max_query_error_m, k=64):
    """Select an ordered, variable-step sequence of real BearingUAV samples.

    Query points define only the desired route shape.  The returned frame pose is
    always source_points[source_index].  Repeated samples and implausible temporal
    jumps are rejected.  This produces natural non-constant frame displacement.
    """
    selected = []
    used = set()
    min_step_px = float(min_step_m) / METERS_PER_PIXEL
    max_step_px = float(max_step_m) / METERS_PER_PIXEL
    max_query_error_px = float(max_query_error_m) / METERS_PER_PIXEL

    last_source_pixel = None
    last_leg = None
    for query in query_points:
        distances, indices = tree.query(query["pixel"], k=min(k, len(source_points)))
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        choice = None

        for query_distance_px, source_index in zip(distances, indices):
            source_index = int(source_index)
            if source_index in used:
                continue
            if float(query_distance_px) > max_query_error_px:
                break
            source_pixel = source_points[source_index]
            if last_source_pixel is not None:
                step_px = float(np.linalg.norm(source_pixel - last_source_pixel))
                # Permit a somewhat larger step exactly across a planned corner,
                # otherwise keep the temporal displacement close to the original
                # v36 native-frame scale.
                corner = last_leg is not None and int(query["leg"]) != int(last_leg)
                allowed_max = max_step_px * (1.5 if corner else 1.0)
                if step_px < min_step_px or step_px > allowed_max:
                    continue
            choice = (source_index, source_pixel.copy(), float(query_distance_px))
            break

        if choice is None:
            continue

        source_index, source_pixel, query_distance_px = choice
        used.add(source_index)
        selected.append({
            "source_index": source_index,
            "source_pixel": source_pixel,
            "query_pixel": np.asarray(query["pixel"], dtype=np.float64),
            "query_error_m": query_distance_px * METERS_PER_PIXEL,
            "leg": int(query["leg"]),
        })
        last_source_pixel = source_pixel
        last_leg = int(query["leg"])

    if len(selected) < 32:
        raise RuntimeError(
            "Too few physically ordered BearingUAV samples were selected; "
            "increase --max-query-error-m or --max-step-m."
        )
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


def _heading_from_points(points, index):
    if len(points) <= 1:
        return 0.0
    if index == 0:
        delta = points[1] - points[0]
    elif index == len(points) - 1:
        delta = points[-1] - points[-2]
    else:
        delta = points[index + 1] - points[index - 1]
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _waypoint_indices(selected):
    """Use real selected samples nearest each planned leg boundary as waypoints."""
    result = [0]
    for index in range(1, len(selected)):
        if selected[index]["leg"] != selected[index - 1]["leg"]:
            result.append(index)
    if result[-1] != len(selected) - 1:
        result.append(len(selected) - 1)
    return sorted(set(result))


def write_route(name, city, selected, rows, bounds, query_spacing_m, rebuild=False):
    root, vi = OUTPUT_ROOT / name, OUTPUT_ROOT / name / "vi"
    root.mkdir(parents=True, exist_ok=True)
    vi.mkdir(parents=True, exist_ok=True)
    timestamps, waypoints = [], []
    x_offset = CITY_OFFSETS[city]
    base_timestamp = 1_900_000_000_000_000_000
    actual_pixels = np.stack([item["source_pixel"] for item in selected])
    waypoint_indices = set(_waypoint_indices(selected))
    waypoint_order = 0

    for frame_index, item in enumerate(selected):
        row = rows[item["source_index"]]
        source = resolve_path(row.get("target_patch_3d") or row["target_path"])
        target = vi / ("vi_%06d.jpg" % frame_index)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

        # IMPORTANT: label from the selected UAV sample's own source position,
        # never from the synthetic query/polyline point.
        global_pixel = item["source_pixel"] + np.asarray([x_offset, 0.0])
        lat, lon = global_pixel_to_latlon(global_pixel, bounds)
        heading_rad = _heading_from_points(actual_pixels, frame_index)
        timestamp_ns = base_timestamp + frame_index * 333_333_333
        timestamps.append({
            "timestamp_ns": timestamp_ns,
            "image": target.name,
            "latitude": lat,
            "longitude": lon,
            "altitude": 120.0,
            "heading_rad": heading_rad,
            "source_bearinguav_index": int(item["source_index"]),
        })

        if frame_index in waypoint_indices:
            waypoints.append({
                "waypoint_order": waypoint_order,
                "role": "start" if waypoint_order == 0 else (
                    "end" if frame_index == len(selected) - 1 else "turn"
                ),
                "frame_index": frame_index,
                "image": target.name,
                "timestamp_ns": timestamp_ns,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": 120.0,
                "source_bearinguav_index": int(item["source_index"]),
            })
            waypoint_order += 1

    (root / "sensor_with_yaw.json").write_text(
        json.dumps({"timestamp": timestamps}, indent=2)
    )
    waypoint_dir = OUTPUT_ROOT / "waypoints"
    waypoint_dir.mkdir(exist_ok=True)
    (waypoint_dir / (name + "_waypoints.json")).write_text(json.dumps({
        "route": name,
        "split": name.split("_")[0],
        "city": city,
        "query_spacing_m": float(query_spacing_m),
        "position_source": "selected BearingUAV sample metadata",
        "waypoints": waypoints,
    }, indent=2))

    steps = np.linalg.norm(np.diff(actual_pixels, axis=0), axis=1) * METERS_PER_PIXEL
    query_errors = np.asarray([item["query_error_m"] for item in selected], dtype=np.float64)
    summary = {
        "route": name,
        "split": name.split("_")[0],
        "city": city,
        "frames": len(selected),
        "waypoints": len(waypoints),
        "step_mean_m": float(np.mean(steps)),
        "step_std_m": float(np.std(steps)),
        "step_p10_m": float(np.quantile(steps, 0.10)),
        "step_p50_m": float(np.quantile(steps, 0.50)),
        "step_p90_m": float(np.quantile(steps, 0.90)),
        "step_min_m": float(np.min(steps)),
        "step_max_m": float(np.max(steps)),
        "image_label_error_mean_m": 0.0,
        "image_label_error_p90_m": 0.0,
        "query_to_source_mean_m": float(np.mean(query_errors)),
        "query_to_source_p90_m": float(np.quantile(query_errors, 0.90)),
    }
    (root / "route_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    # Query spacing is deliberately denser than the desired output step.  Real
    # samples are then filtered using min/max actual displacement, so output
    # steps vary naturally instead of being fixed by construction.
    parser.add_argument("--query-spacing-m", type=float, default=1.5)
    parser.add_argument("--min-step-m", type=float, default=1.5)
    parser.add_argument("--max-step-m", type=float, default=5.0)
    parser.add_argument("--max-query-error-m", type=float, default=8.0)
    parser.add_argument("--rebuild", action="store_true")
    # Backward-compatible alias; it no longer controls label spacing.
    parser.add_argument("--spacing-m", type=float, default=None)
    args = parser.parse_args()
    if args.spacing_m is not None:
        print(
            "NOTE: --spacing-m is deprecated for BearingUAV. The adapter now "
            "uses actual source positions and variable frame steps; using it "
            "only as query spacing.",
            flush=True,
        )
        args.query_spacing_m = float(args.spacing_m)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mosaic = make_mosaic(args.rebuild)
    bounds, georef = mosaic_georef()
    city_data = {}
    for city in CITY_OFFSETS:
        metadata = DATA_ROOT / city / "rawmetadata.csv"
        rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
        points = np.stack([row_pixel(row) for row in rows])
        city_data[city] = (rows, points, cKDTree(points))

    summaries = []
    for name, (city, waypoint_pixels) in ROUTES.items():
        rows, source_points, tree = city_data[city]
        query_points = sample_polyline(waypoint_pixels, args.query_spacing_m)
        selected = select_actual_sequence(
            query_points,
            tree,
            source_points,
            min_step_m=args.min_step_m,
            max_step_m=args.max_step_m,
            max_query_error_m=args.max_query_error_m,
        )
        summary = write_route(
            name, city, selected, rows, bounds, args.query_spacing_m, args.rebuild
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    (OUTPUT_ROOT / "generation_summary.json").write_text(json.dumps({
        "split": {
            "train": ["train_1", "train_2", "train_3"],
            "validation": ["val_1"],
            "test": ["test_1"],
        },
        "adapter": {
            "position_labels": "actual selected BearingUAV sample positions",
            "temporal_order": "physically ordered nearest samples along planned polyline",
            "variable_step": True,
            "query_spacing_m": float(args.query_spacing_m),
            "min_step_m": float(args.min_step_m),
            "max_step_m": float(args.max_step_m),
            "max_query_error_m": float(args.max_query_error_m),
        },
        "satellite_mosaic": str(mosaic),
        "satellite_georef": str(georef),
        "routes": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
