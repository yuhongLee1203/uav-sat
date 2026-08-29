"""Build physically ordered BearingUAV pseudo-sequences from actual sample poses.

The BearingUAV-90K images are independent samples rather than video frames.  This
adapter therefore does not invent fixed-spacing position labels.  A planned
polyline is used only as a spatial ordering guide; every written frame position
is the selected BearingUAV sample's own metadata position.  Consecutive samples
are selected with a preferred variable-step range and a bounded fallback for
sparse regions.
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


def horizontal_lawnmower(rows=10, x0=420, x1=3670, y0=420, y1=3670):
    """Horizontal sweeps with explicit vertical connectors (90-degree turns)."""
    ys = np.linspace(y0, y1, rows)
    points = []
    for i, y in enumerate(ys):
        start_x, end_x = (x0, x1) if i % 2 == 0 else (x1, x0)
        if not points:
            points.append((float(start_x), float(y)))
        elif points[-1] != (float(start_x), float(y)):
            points.append((float(start_x), float(y)))
        points.append((float(end_x), float(y)))
        if i + 1 < len(ys):
            points.append((float(end_x), float(ys[i + 1])))
    return points


def vertical_lawnmower(cols=10, y0=420, y1=3670, x0=420, x1=3670):
    """Vertical sweeps with explicit horizontal connectors."""
    xs = np.linspace(x0, x1, cols)
    points = []
    for i, x in enumerate(xs):
        start_y, end_y = (y0, y1) if i % 2 == 0 else (y1, y0)
        if not points:
            points.append((float(x), float(start_y)))
        elif points[-1] != (float(x), float(start_y)):
            points.append((float(x), float(start_y)))
        points.append((float(x), float(end_y)))
        if i + 1 < len(xs):
            points.append((float(xs[i + 1]), float(end_y)))
    return points


def diagonal_sweep(lines=9, x0=520, x1=3570, y0=480, y1=3610):
    """A second city-A route family with broad diagonal sweeps and connectors."""
    offsets = np.linspace(0.0, 700.0, lines)
    points = []
    for i, off in enumerate(offsets):
        a = (x0 + off, y0)
        b = (x1, min(y1, y1 - 700.0 + off))
        if i % 2:
            a, b = b, a
        a = (float(np.clip(a[0], 350, 3740)), float(np.clip(a[1], 350, 3740)))
        b = (float(np.clip(b[0], 350, 3740)), float(np.clip(b[1], 350, 3740)))
        if not points:
            points.append(a)
        elif points[-1] != a:
            points.append(a)
        points.append(b)
    return points


ROUTES = {
    "train_1": ("citya", horizontal_lawnmower()),
    "train_2": ("citya", vertical_lawnmower()),
    "train_3": ("citya", diagonal_sweep()),
    "val_1": ("cityb", horizontal_lawnmower()),
    "test_1": ("cityc", vertical_lawnmower()),
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
    """Actual BearingUAV sample center in its 4096 x 4096 city map."""
    return np.asarray([
        int(row["block_x"]) * 256.0 + (float(row["x_norm"]) + 1.0) * 256.0,
        int(row["block_y"]) * 256.0 + (float(row["y_norm"]) + 1.0) * 256.0,
    ], dtype=np.float64)


def sample_polyline(waypoints, spacing_m):
    """Dense route query points used only as an ordering/search guide."""
    spacing_px = max(float(spacing_m) / METERS_PER_PIXEL, 1.0)
    samples = []
    route_order = 0
    for leg, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(math.ceil(length / spacing_px)))
        for step in range(steps + 1):
            if leg > 0 and step == 0:
                continue
            alpha = step / float(steps)
            samples.append({
                "pixel": start + alpha * (end - start),
                "leg": int(leg),
                "route_order": int(route_order),
            })
            route_order += 1
    return samples


def _candidate_choice(query, distances, indices, source_points, used,
                      last_source_pixel, preferred_min_px, preferred_max_px,
                      hard_max_px, max_query_error_px, k_nearest):
    """Pick the best unused actual sample for one query point.

    1) Prefer a natural step in [preferred_min, preferred_max].
    2) If the dataset is locally sparse, allow a bounded fallback step.
    The fallback changes only frame spacing; the position label always remains
    the selected source sample's actual pose.
    """
    rows = []
    for query_distance_px, source_index in zip(
        np.atleast_1d(distances)[:k_nearest], np.atleast_1d(indices)[:k_nearest]
    ):
        source_index = int(source_index)
        query_distance_px = float(query_distance_px)
        if source_index in used:
            continue
        if not np.isfinite(query_distance_px) or query_distance_px > max_query_error_px:
            continue
        source_pixel = source_points[source_index]
        if last_source_pixel is None:
            step_px = 0.0
        else:
            step_px = float(np.linalg.norm(source_pixel - last_source_pixel))
            if step_px > hard_max_px:
                continue
            # Avoid duplicate/near-identical positions even if they are distinct samples.
            if step_px < 0.35 / METERS_PER_PIXEL:
                continue
        rows.append((source_index, source_pixel, query_distance_px, step_px))

    if not rows:
        return None
    if last_source_pixel is None:
        best = min(rows, key=lambda x: x[2])
        return best + (False,)

    preferred = [x for x in rows if preferred_min_px <= x[3] <= preferred_max_px]
    if preferred:
        target_px = 0.5 * (preferred_min_px + preferred_max_px)
        best = min(
            preferred,
            key=lambda x: x[2] + 0.15 * abs(x[3] - target_px),
        )
        return best + (False,)

    # Sparse-region fallback.  Prefer a slightly larger step over a very tiny
    # step, because the latter would create many almost-identical pseudo-frames.
    fallback = [x for x in rows if x[3] >= 0.5 * preferred_min_px]
    if not fallback:
        return None
    best = min(
        fallback,
        key=lambda x: x[2] + 0.35 * max(0.0, x[3] - preferred_max_px),
    )
    return best + (True,)


def select_actual_sequence(query_points, tree, source_points, min_step_m,
                           max_step_m, max_query_error_m, hard_max_step_m=12.0,
                           k=128):
    """Select a physically ordered variable-step sequence of actual samples."""
    selected = []
    used = set()
    preferred_min_px = float(min_step_m) / METERS_PER_PIXEL
    preferred_max_px = float(max_step_m) / METERS_PER_PIXEL
    hard_max_px = max(float(hard_max_step_m), float(max_step_m)) / METERS_PER_PIXEL
    max_query_error_px = float(max_query_error_m) / METERS_PER_PIXEL
    last_source_pixel = None

    for query in query_points:
        distances, indices = tree.query(query["pixel"], k=min(k, len(source_points)))
        choice = _candidate_choice(
            query,
            distances,
            indices,
            source_points,
            used,
            last_source_pixel,
            preferred_min_px,
            preferred_max_px,
            hard_max_px,
            max_query_error_px,
            min(k, len(source_points)),
        )
        if choice is None:
            continue

        source_index, source_pixel, query_distance_px, step_px, used_fallback = choice
        # Dense route queries can still map to a new source sample too soon.
        # Keep it only when it advances a meaningful physical distance.
        if last_source_pixel is not None and step_px < 0.5 * preferred_min_px:
            continue

        used.add(int(source_index))
        selected.append({
            "source_index": int(source_index),
            "source_pixel": np.asarray(source_pixel, dtype=np.float64).copy(),
            "query_pixel": np.asarray(query["pixel"], dtype=np.float64).copy(),
            "query_error_m": float(query_distance_px * METERS_PER_PIXEL),
            "step_m": float(step_px * METERS_PER_PIXEL) if last_source_pixel is not None else 0.0,
            "step_fallback": bool(used_fallback),
            "leg": int(query["leg"]),
            "route_order": int(query["route_order"]),
        })
        last_source_pixel = np.asarray(source_pixel, dtype=np.float64).copy()

    if len(selected) < 64:
        raise RuntimeError(
            "Too few actual BearingUAV samples were selected (%d). "
            "The route corridor is too sparse even with the bounded fallback. "
            "Try --max-query-error-m 12 or --hard-max-step-m 14." % len(selected)
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
    if float(np.linalg.norm(delta)) < 1e-9:
        return 0.0
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _waypoint_indices(selected):
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
    if rebuild:
        for stale in vi.glob("vi_*.jpg"):
            stale.unlink()

    timestamps, waypoints = [], []
    x_offset = CITY_OFFSETS[city]
    base_timestamp = 1_900_000_000_000_000_000
    actual_pixels = np.stack([item["source_pixel"] for item in selected])
    waypoint_indices = set(_waypoint_indices(selected))
    waypoint_order = 0

    for frame_index, item in enumerate(selected):
        row = rows[item["source_index"]]
        source = resolve_path(row.get("target_patch_3d") or row["target_path"])
        if not source.exists():
            raise FileNotFoundError(source)
        target = vi / ("vi_%06d.jpg" % frame_index)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

        # Crucial: the label is the selected source image's own actual position.
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
            "yaw": heading_rad,
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
        "position_source": "selected BearingUAV sample actual metadata position",
        "waypoints": waypoints,
    }, indent=2))

    steps = np.linalg.norm(np.diff(actual_pixels, axis=0), axis=1) * METERS_PER_PIXEL
    query_errors = np.asarray([item["query_error_m"] for item in selected], dtype=np.float64)
    fallback = np.asarray([item["step_fallback"] for item in selected[1:]], dtype=np.float64)
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
        "fallback_step_pct": float(100.0 * np.mean(fallback)) if fallback.size else 0.0,
        "image_label_error_mean_m": 0.0,
        "image_label_error_p90_m": 0.0,
        "query_to_source_mean_m": float(np.mean(query_errors)),
        "query_to_source_p90_m": float(np.quantile(query_errors, 0.90)),
    }
    (root / "route_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def validate_summary(summary):
    if int(summary["frames"]) < 64:
        raise RuntimeError("route has too few frames")
    if float(summary["image_label_error_mean_m"]) != 0.0:
        raise RuntimeError("image/label position mismatch detected")
    if float(summary["step_std_m"]) < 0.05:
        raise RuntimeError("step distribution is still effectively fixed")
    if abs(float(summary["step_p90_m"]) - float(summary["step_p10_m"])) < 0.10:
        raise RuntimeError("step P10/P90 are too similar; sequence is not variable-step")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-spacing-m", type=float, default=1.5)
    parser.add_argument("--min-step-m", type=float, default=1.5)
    parser.add_argument("--max-step-m", type=float, default=5.0)
    parser.add_argument("--hard-max-step-m", type=float, default=12.0)
    parser.add_argument("--max-query-error-m", type=float, default=8.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--spacing-m", type=float, default=None)
    args = parser.parse_args()
    if args.spacing_m is not None:
        print(
            "NOTE: --spacing-m is deprecated. It now changes only the dense "
            "route-query spacing; frame labels remain actual BearingUAV poses.",
            flush=True,
        )
        args.query_spacing_m = float(args.spacing_m)

    if args.min_step_m <= 0 or args.max_step_m <= args.min_step_m:
        raise ValueError("require 0 < min-step-m < max-step-m")
    if args.hard_max_step_m < args.max_step_m:
        raise ValueError("hard-max-step-m must be >= max-step-m")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mosaic = make_mosaic(args.rebuild)
    bounds, georef = mosaic_georef()
    city_data = {}
    for city in CITY_OFFSETS:
        metadata = DATA_ROOT / city / "rawmetadata.csv"
        rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
        points = np.stack([row_pixel(row) for row in rows])
        city_data[city] = (rows, points, cKDTree(points))
        print("%s raw samples=%d" % (city, len(rows)), flush=True)

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
            hard_max_step_m=args.hard_max_step_m,
            max_query_error_m=args.max_query_error_m,
        )
        summary = write_route(
            name, city, selected, rows, bounds, args.query_spacing_m, args.rebuild
        )
        validate_summary(summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    payload = {
        "split": {
            "train": ["train_1", "train_2", "train_3"],
            "validation": ["val_1"],
            "test": ["test_1"],
        },
        "adapter": {
            "position_labels": "actual selected BearingUAV sample positions",
            "temporal_order": "spatially ordered pseudo-sequence along planned route",
            "route_geometry": "lawnmower/diagonal sweeps with explicit connectors",
            "variable_step": True,
            "preferred_query_spacing_m": float(args.query_spacing_m),
            "preferred_min_step_m": float(args.min_step_m),
            "preferred_max_step_m": float(args.max_step_m),
            "hard_max_step_m": float(args.hard_max_step_m),
            "max_query_error_m": float(args.max_query_error_m),
        },
        "satellite_mosaic": str(mosaic),
        "satellite_georef": str(georef),
        "routes": summaries,
    }
    (OUTPUT_ROOT / "generation_summary.json").write_text(json.dumps(payload, indent=2))
    print("generation summary:", OUTPUT_ROOT / "generation_summary.json", flush=True)


if __name__ == "__main__":
    main()
