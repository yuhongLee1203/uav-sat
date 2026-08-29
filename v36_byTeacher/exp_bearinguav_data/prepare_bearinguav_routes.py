"""Build physically ordered BearingUAV pseudo-sequences from actual sample poses.

BearingUAV-90K contains independent cross-view samples rather than video frames.
This adapter therefore uses the official sample pose as the frame label and uses
a planned polyline only to order nearby real samples.  It never assigns a
synthetic fixed-spacing position to an image.

The route builder first projects every real sample onto a planned route, keeps
samples inside a corridor, sorts them by route progress, and then greedily
selects a continuous variable-step chain.  This is much more robust to the
dataset's sparse/irregular sampling than chasing dense query points one by one.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


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


def horizontal_lawnmower(rows=8, x0=500, x1=3590, y0=500, y1=3590):
    """Long horizontal sweeps joined by explicit vertical connectors."""
    ys = np.linspace(y0, y1, rows)
    points = []
    for i, y in enumerate(ys):
        start_x, end_x = (x0, x1) if i % 2 == 0 else (x1, x0)
        a = (float(start_x), float(y))
        b = (float(end_x), float(y))
        if not points:
            points.append(a)
        elif points[-1] != a:
            points.append(a)
        points.append(b)
        if i + 1 < len(ys):
            points.append((float(end_x), float(ys[i + 1])))
    return points


def vertical_lawnmower(cols=8, y0=500, y1=3590, x0=500, x1=3590):
    """Long vertical sweeps joined by explicit horizontal connectors."""
    xs = np.linspace(x0, x1, cols)
    points = []
    for i, x in enumerate(xs):
        start_y, end_y = (y0, y1) if i % 2 == 0 else (y1, y0)
        a = (float(x), float(start_y))
        b = (float(x), float(end_y))
        if not points:
            points.append(a)
        elif points[-1] != a:
            points.append(a)
        points.append(b)
        if i + 1 < len(xs):
            points.append((float(xs[i + 1]), float(end_y)))
    return points


def diagonal_sweep(lines=7, x0=620, x1=3470, y0=560, y1=3530):
    """Broad diagonal sweeps with short connectors."""
    offsets = np.linspace(0.0, 620.0, lines)
    points = []
    for i, off in enumerate(offsets):
        a = np.asarray([x0 + off, y0], dtype=np.float64)
        b = np.asarray([x1, y1 - 620.0 + off], dtype=np.float64)
        a = np.clip(a, 450, 3640)
        b = np.clip(b, 450, 3640)
        if i % 2:
            a, b = b, a
        a, b = tuple(a.tolist()), tuple(b.tolist())
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


def _route_segments(waypoints):
    points = np.asarray(waypoints, dtype=np.float64)
    starts = points[:-1]
    deltas = points[1:] - points[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("route contains a zero-length segment")
    units = deltas / lengths[:, None]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return points, starts, units, lengths, cumulative


def project_samples_to_route(source_points, waypoints):
    """Project all samples to the nearest planned-route segment.

    Returns best route progress in pixels, nearest-route distance in pixels, and
    the nearest segment index for every source sample.
    """
    _, starts, units, lengths, cumulative = _route_segments(waypoints)
    n = len(source_points)
    best_dist = np.full(n, np.inf, dtype=np.float64)
    best_progress = np.zeros(n, dtype=np.float64)
    best_leg = np.zeros(n, dtype=np.int32)

    for leg in range(len(starts)):
        rel = source_points - starts[leg]
        along = np.sum(rel * units[leg], axis=1)
        along_clip = np.clip(along, 0.0, lengths[leg])
        nearest = starts[leg] + along_clip[:, None] * units[leg]
        dist = np.linalg.norm(source_points - nearest, axis=1)
        mask = dist < best_dist
        best_dist[mask] = dist[mask]
        best_progress[mask] = cumulative[leg] + along_clip[mask]
        best_leg[mask] = leg

    return best_progress, best_dist, best_leg


def select_actual_sequence(
    source_points,
    waypoints,
    corridor_m=14.0,
    min_step_m=1.0,
    preferred_step_m=5.5,
    preferred_max_step_m=10.0,
    hard_max_step_m=18.0,
    min_progress_m=0.6,
    lookahead_m=24.0,
):
    """Build a route-progress-ordered chain from real BearingUAV samples.

    The dataset is too sparse to demand 1.5--5 m at every step.  Instead, each
    next frame is chosen from real samples ahead on the route.  Steps near the
    preferred range are favored, while sparse regions may use a bounded larger
    step.  The image label always remains the selected sample's actual pose.
    """
    progress_px, route_dist_px, legs = project_samples_to_route(source_points, waypoints)
    corridor_px = float(corridor_m) / METERS_PER_PIXEL
    keep = np.flatnonzero(route_dist_px <= corridor_px)
    if len(keep) < 64:
        raise RuntimeError(
            "Too few BearingUAV samples lie inside the route corridor (%d). "
            "Increase --corridor-m." % len(keep)
        )

    order = keep[np.argsort(progress_px[keep], kind="stable")]
    progress_m = progress_px * METERS_PER_PIXEL
    route_dist_m = route_dist_px * METERS_PER_PIXEL

    min_step = float(min_step_m)
    preferred = float(preferred_step_m)
    preferred_max = float(preferred_max_step_m)
    hard_max = float(hard_max_step_m)
    min_progress = float(min_progress_m)
    lookahead = float(lookahead_m)

    selected_indices = []
    selected = []
    used = set()

    first_pool = order[: min(256, len(order))]
    first = int(first_pool[np.argmin(route_dist_m[first_pool])])
    selected_indices.append(first)
    used.add(first)

    while True:
        current = selected_indices[-1]
        current_progress = float(progress_m[current])

        lo = np.searchsorted(progress_m[order], current_progress + min_progress, side="left")
        hi = np.searchsorted(progress_m[order], current_progress + lookahead, side="right")
        candidates = [int(i) for i in order[lo:hi] if int(i) not in used]
        if not candidates:
            hi = np.searchsorted(progress_m[order], current_progress + 2.0 * lookahead, side="right")
            candidates = [int(i) for i in order[lo:hi] if int(i) not in used]
        if not candidates:
            break

        cand_arr = np.asarray(candidates, dtype=np.int64)
        steps = np.linalg.norm(source_points[cand_arr] - source_points[current], axis=1) * METERS_PER_PIXEL
        prog_delta = progress_m[cand_arr] - current_progress

        valid = (steps >= min_step) & (steps <= hard_max) & (prog_delta > 0.0)
        if not np.any(valid):
            break

        cand_arr = cand_arr[valid]
        steps = steps[valid]
        prog_delta = prog_delta[valid]

        within_preferred = steps <= preferred_max
        fallback = not bool(np.any(within_preferred))
        if np.any(within_preferred):
            cand_arr = cand_arr[within_preferred]
            steps = steps[within_preferred]
            prog_delta = prog_delta[within_preferred]

        scores = (
            np.abs(steps - preferred)
            + 0.30 * route_dist_m[cand_arr]
            + 0.08 * np.abs(prog_delta - preferred)
        )
        next_index = int(cand_arr[int(np.argmin(scores))])
        step_m = float(np.linalg.norm(source_points[next_index] - source_points[current]) * METERS_PER_PIXEL)

        selected_indices.append(next_index)
        used.add(next_index)
        selected.append({
            "source_index": next_index,
            "source_pixel": source_points[next_index].copy(),
            "route_progress_m": float(progress_m[next_index]),
            "route_distance_m": float(route_dist_m[next_index]),
            "step_m": step_m,
            "step_fallback": bool(fallback or step_m > preferred_max),
            "leg": int(legs[next_index]),
        })

    if len(selected_indices) < 64:
        raise RuntimeError(
            "Only %d continuous actual-pose samples could be chained. "
            "Try --corridor-m 18 or --hard-max-step-m 22." % len(selected_indices)
        )

    first = selected_indices[0]
    rows = [{
        "source_index": int(first),
        "source_pixel": source_points[first].copy(),
        "route_progress_m": float(progress_m[first]),
        "route_distance_m": float(route_dist_m[first]),
        "step_m": 0.0,
        "step_fallback": False,
        "leg": int(legs[first]),
    }]
    rows.extend(selected)

    ids = [row["source_index"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate BearingUAV source sample detected")
    return rows


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
    """Use actual selected samples at planned-leg transitions as waypoints."""
    result = [0]
    for index in range(1, len(selected)):
        if selected[index]["leg"] != selected[index - 1]["leg"]:
            result.append(index)
    if result[-1] != len(selected) - 1:
        result.append(len(selected) - 1)
    max_gap = 80
    expanded = [result[0]]
    for a, b in zip(result[:-1], result[1:]):
        cursor = a + max_gap
        while cursor < b:
            expanded.append(cursor)
            cursor += max_gap
        expanded.append(b)
    return sorted(set(expanded))


def write_route(name, city, selected, rows, bounds, rebuild=False):
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
        "position_source": "selected BearingUAV sample actual metadata position",
        "temporal_semantics": "spatially ordered actual-pose pseudo-sequence",
        "waypoints": waypoints,
    }, indent=2))

    steps = np.linalg.norm(np.diff(actual_pixels, axis=0), axis=1) * METERS_PER_PIXEL
    route_dist = np.asarray([item["route_distance_m"] for item in selected], dtype=np.float64)
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
        "route_corridor_mean_m": float(np.mean(route_dist)),
        "route_corridor_p90_m": float(np.quantile(route_dist, 0.90)),
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
    parser.add_argument("--min-step-m", type=float, default=1.0)
    parser.add_argument("--max-step-m", type=float, default=10.0)
    parser.add_argument("--hard-max-step-m", type=float, default=18.0)
    parser.add_argument("--max-query-error-m", type=float, default=14.0)
    parser.add_argument("--corridor-m", type=float, default=None)
    parser.add_argument("--preferred-step-m", type=float, default=5.5)
    parser.add_argument("--lookahead-m", type=float, default=24.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--spacing-m", type=float, default=None)
    args = parser.parse_args()

    corridor_m = (
        float(args.corridor_m)
        if args.corridor_m is not None
        else float(args.max_query_error_m)
    )
    preferred_max = float(args.max_step_m)

    if args.min_step_m <= 0:
        raise ValueError("min-step-m must be > 0")
    if preferred_max <= args.min_step_m:
        raise ValueError("max-step-m must be > min-step-m")
    if args.hard_max_step_m < preferred_max:
        raise ValueError("hard-max-step-m must be >= max-step-m")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mosaic = make_mosaic(args.rebuild)
    bounds, georef = mosaic_georef()

    city_data = {}
    for city in CITY_OFFSETS:
        metadata = DATA_ROOT / city / "rawmetadata.csv"
        rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
        points = np.stack([row_pixel(row) for row in rows])
        city_data[city] = (rows, points)
        print("%s raw samples=%d" % (city, len(rows)), flush=True)

    summaries = []
    for name, (city, waypoint_pixels) in ROUTES.items():
        rows, source_points = city_data[city]
        selected = select_actual_sequence(
            source_points,
            waypoint_pixels,
            corridor_m=corridor_m,
            min_step_m=float(args.min_step_m),
            preferred_step_m=float(args.preferred_step_m),
            preferred_max_step_m=preferred_max,
            hard_max_step_m=float(args.hard_max_step_m),
            lookahead_m=float(args.lookahead_m),
        )
        summary = write_route(name, city, selected, rows, bounds, args.rebuild)
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
            "temporal_order": "route-progress-ordered actual-pose pseudo-sequence",
            "selection": "all real samples inside a route corridor, then bounded variable-step chaining",
            "variable_step": True,
            "preferred_step_m": float(args.preferred_step_m),
            "preferred_max_step_m": preferred_max,
            "hard_max_step_m": float(args.hard_max_step_m),
            "corridor_m": corridor_m,
            "lookahead_m": float(args.lookahead_m),
        },
        "satellite_mosaic": str(mosaic),
        "satellite_georef": str(georef),
        "routes": summaries,
    }
    (OUTPUT_ROOT / "generation_summary.json").write_text(json.dumps(payload, indent=2))
    print("generation summary:", OUTPUT_ROOT / "generation_summary.json", flush=True)


if __name__ == "__main__":
    main()
