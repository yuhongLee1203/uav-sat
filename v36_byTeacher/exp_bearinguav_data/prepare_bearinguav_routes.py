"""Create v36-compatible BearingUAV pseudo-sequences from real sample poses.

BearingUAV-90K contains independent cross-view samples, not video.  A planned
route is used only to order nearby real samples.  Every frame label is the
selected sample's own pose; frame spacing is variable and no fixed-spacing
synthetic label is created.
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
    ys = np.linspace(y0, y1, rows)
    out = []
    for i, y in enumerate(ys):
        a = (x0, y) if i % 2 == 0 else (x1, y)
        b = (x1, y) if i % 2 == 0 else (x0, y)
        a, b = tuple(map(float, a)), tuple(map(float, b))
        if not out or out[-1] != a:
            out.append(a)
        out.append(b)
        if i + 1 < rows:
            out.append((float(b[0]), float(ys[i + 1])))
    return out


def vertical_lawnmower(cols=8, x0=500, x1=3590, y0=500, y1=3590):
    xs = np.linspace(x0, x1, cols)
    out = []
    for i, x in enumerate(xs):
        a = (x, y0) if i % 2 == 0 else (x, y1)
        b = (x, y1) if i % 2 == 0 else (x, y0)
        a, b = tuple(map(float, a)), tuple(map(float, b))
        if not out or out[-1] != a:
            out.append(a)
        out.append(b)
        if i + 1 < cols:
            out.append((float(xs[i + 1]), float(b[1])))
    return out


def rectangular_spiral(levels=4, x0=500, x1=3590, y0=500, y1=3590, inset=300):
    """A long non-self-intersecting route distinct from the two lawnmowers."""
    out = []
    left, right, top, bottom = map(float, (x0, x1, y0, y1))
    for level in range(int(levels)):
        if left >= right or top >= bottom:
            break
        corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
        if not out or out[-1] != corners[0]:
            out.append(corners[0])
        out.extend(corners[1:])
        nl, nr = left + inset, right - inset
        nt, nb = top + inset, bottom - inset
        if level + 1 < levels and nl < nr and nt < nb:
            out.append((float(nl), float(nb)))
        left, right, top, bottom = nl, nr, nt, nb
    return out


ROUTES = {
    "train_1": ("citya", horizontal_lawnmower()),
    "train_2": ("citya", vertical_lawnmower()),
    "train_3": ("citya", rectangular_spiral()),
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
    return np.asarray([
        int(row["block_x"]) * 256.0 + (float(row["x_norm"]) + 1.0) * 256.0,
        int(row["block_y"]) * 256.0 + (float(row["y_norm"]) + 1.0) * 256.0,
    ], dtype=np.float64)


def project_samples_to_route(points, waypoints):
    wp = np.asarray(waypoints, dtype=np.float64)
    starts = wp[:-1]
    delta = wp[1:] - wp[:-1]
    lengths = np.linalg.norm(delta, axis=1)
    units = delta / np.maximum(lengths[:, None], 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    best_dist = np.full(len(points), np.inf, dtype=np.float64)
    best_progress = np.zeros(len(points), dtype=np.float64)
    best_leg = np.zeros(len(points), dtype=np.int32)
    for leg, (start, unit, length) in enumerate(zip(starts, units, lengths)):
        rel = points - start
        along = np.clip(np.sum(rel * unit, axis=1), 0.0, length)
        nearest = start + along[:, None] * unit
        dist = np.linalg.norm(points - nearest, axis=1)
        mask = dist < best_dist
        best_dist[mask] = dist[mask]
        best_progress[mask] = cumulative[leg] + along[mask]
        best_leg[mask] = leg
    return best_progress, best_dist, best_leg


def select_actual_sequence(
    points,
    waypoints,
    corridor_m,
    min_step_m,
    preferred_step_m,
    preferred_max_step_m,
    hard_max_step_m,
    lookahead_m,
):
    """Select a continuous variable-step chain from actual sample positions."""
    progress_px, route_dist_px, legs = project_samples_to_route(points, waypoints)
    progress_m = progress_px * METERS_PER_PIXEL
    route_dist_m = route_dist_px * METERS_PER_PIXEL
    keep = np.flatnonzero(route_dist_m <= float(corridor_m))
    if len(keep) < 64:
        raise RuntimeError(
            f"Only {len(keep)} samples are inside the {corridor_m:.1f} m route corridor."
        )

    order = keep[np.argsort(progress_m[keep], kind="stable")]
    ordered_progress = progress_m[order]
    first_pool = order[: min(256, len(order))]
    first = int(first_pool[np.argmin(route_dist_m[first_pool])])
    chain = [first]
    used = {first}
    fallback_flags = [False]

    while True:
        current = chain[-1]
        s0 = float(progress_m[current])
        lo = int(np.searchsorted(ordered_progress, s0 + 0.5, side="left"))
        hi = int(np.searchsorted(ordered_progress, s0 + float(lookahead_m), side="right"))
        cand = np.asarray(
            [int(i) for i in order[lo:hi] if int(i) not in used],
            dtype=np.int64,
        )
        if cand.size == 0:
            hi = int(np.searchsorted(
                ordered_progress, s0 + 2.0 * float(lookahead_m), side="right"
            ))
            cand = np.asarray(
                [int(i) for i in order[lo:hi] if int(i) not in used],
                dtype=np.int64,
            )
        if cand.size == 0:
            break

        step = np.linalg.norm(points[cand] - points[current], axis=1) * METERS_PER_PIXEL
        ds = progress_m[cand] - s0
        valid = (
            (step >= float(min_step_m))
            & (step <= float(hard_max_step_m))
            & (ds > 0.0)
        )
        if not np.any(valid):
            break
        cand, step, ds = cand[valid], step[valid], ds[valid]

        preferred = step <= float(preferred_max_step_m)
        fallback = not bool(np.any(preferred))
        if np.any(preferred):
            cand, step, ds = cand[preferred], step[preferred], ds[preferred]

        score = (
            np.abs(step - float(preferred_step_m))
            + 0.30 * route_dist_m[cand]
            + 0.08 * np.abs(ds - float(preferred_step_m))
        )
        nxt = int(cand[int(np.argmin(score))])
        actual_step = float(np.linalg.norm(points[nxt] - points[current]) * METERS_PER_PIXEL)
        chain.append(nxt)
        used.add(nxt)
        fallback_flags.append(bool(fallback or actual_step > float(preferred_max_step_m)))

    if len(chain) < 64:
        raise RuntimeError(
            f"Only {len(chain)} continuous samples could be chained. "
            "Try --corridor-m 18 --hard-max-step-m 22."
        )

    rows = []
    previous = None
    for source_index, fallback in zip(chain, fallback_flags):
        step_m = 0.0 if previous is None else float(
            np.linalg.norm(points[source_index] - points[previous]) * METERS_PER_PIXEL
        )
        rows.append({
            "source_index": int(source_index),
            "source_pixel": points[source_index].copy(),
            "route_progress_m": float(progress_m[source_index]),
            "route_distance_m": float(route_dist_m[source_index]),
            "step_m": step_m,
            "step_fallback": bool(fallback),
            "leg": int(legs[source_index]),
        })
        previous = source_index
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
    top_lat, left_lon = 23.6, 120.0
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(top_lat)))
    bottom_lat = top_lat - height_px * METERS_PER_PIXEL * lat_per_m
    right_lon = left_lon + width_px * METERS_PER_PIXEL * lon_per_m
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
    tl = bounds["geo_bounds"]["top_left"]
    br = bounds["geo_bounds"]["bottom_right"]
    lon = tl["longitude"] + pixel[0] / (MAP_PX * 3 - 1) * (br["longitude"] - tl["longitude"])
    lat = tl["latitude"] - pixel[1] / (MAP_PX - 1) * (tl["latitude"] - br["latitude"])
    return float(lat), float(lon)


def heading_from_points(points, index):
    if len(points) <= 1:
        return 0.0
    if index == 0:
        delta = points[1] - points[0]
    elif index == len(points) - 1:
        delta = points[-1] - points[-2]
    else:
        delta = points[index + 1] - points[index - 1]
    return float(math.atan2(float(delta[1]), float(delta[0]))) if np.linalg.norm(delta) > 1e-9 else 0.0


def waypoint_indices(selected):
    indices = [0]
    for i in range(1, len(selected)):
        if selected[i]["leg"] != selected[i - 1]["leg"]:
            indices.append(i)
    if indices[-1] != len(selected) - 1:
        indices.append(len(selected) - 1)

    out = [indices[0]]
    for a, b in zip(indices[:-1], indices[1:]):
        j = a + 80
        while j < b:
            out.append(j)
            j += 80
        out.append(b)
    return sorted(set(out))


def write_route(name, city, selected, metadata_rows, bounds, rebuild):
    root = OUTPUT_ROOT / name
    vi = root / "vi"
    root.mkdir(parents=True, exist_ok=True)
    vi.mkdir(parents=True, exist_ok=True)
    if rebuild:
        for stale in vi.glob("vi_*.jpg"):
            stale.unlink()

    actual = np.stack([x["source_pixel"] for x in selected])
    wp_indices = set(waypoint_indices(selected))
    timestamps, waypoints = [], []
    waypoint_order = 0
    x_offset = CITY_OFFSETS[city]
    base_timestamp = 1_900_000_000_000_000_000

    for frame_index, item in enumerate(selected):
        meta = metadata_rows[item["source_index"]]
        source = resolve_path(meta.get("target_patch_3d") or meta["target_path"])
        if not source.exists():
            raise FileNotFoundError(source)
        target = vi / f"vi_{frame_index:06d}.jpg"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

        global_pixel = item["source_pixel"] + np.asarray([x_offset, 0.0])
        lat, lon = global_pixel_to_latlon(global_pixel, bounds)
        heading = heading_from_points(actual, frame_index)
        timestamp_ns = base_timestamp + frame_index * 333_333_333
        timestamps.append({
            "timestamp_ns": timestamp_ns,
            "image": target.name,
            "latitude": lat,
            "longitude": lon,
            "altitude": 120.0,
            "heading_rad": heading,
            "yaw": heading,
            "source_bearinguav_index": int(item["source_index"]),
        })

        if frame_index in wp_indices:
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

    (root / "sensor_with_yaw.json").write_text(json.dumps({"timestamp": timestamps}, indent=2))
    wp_dir = OUTPUT_ROOT / "waypoints"
    wp_dir.mkdir(exist_ok=True)
    (wp_dir / f"{name}_waypoints.json").write_text(json.dumps({
        "route": name,
        "split": name.split("_")[0],
        "city": city,
        "position_source": "selected BearingUAV sample actual metadata position",
        "temporal_semantics": "spatially ordered actual-pose pseudo-sequence",
        "waypoints": waypoints,
    }, indent=2))

    steps = np.linalg.norm(np.diff(actual, axis=0), axis=1) * METERS_PER_PIXEL
    corridor = np.asarray([x["route_distance_m"] for x in selected], dtype=np.float64)
    fallback = np.asarray([x["step_fallback"] for x in selected[1:]], dtype=np.float64)
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
        "route_corridor_mean_m": float(np.mean(corridor)),
        "route_corridor_p90_m": float(np.quantile(corridor, 0.90)),
    }
    (root / "route_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def validate_summary(summary):
    if summary["frames"] < 64:
        raise RuntimeError(f'{summary["route"]}: too few frames')
    if summary["image_label_error_mean_m"] != 0.0:
        raise RuntimeError(f'{summary["route"]}: image/label mismatch')
    if summary["step_std_m"] < 0.05:
        raise RuntimeError(f'{summary["route"]}: step distribution is still fixed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-spacing-m", type=float, default=1.5)
    parser.add_argument("--max-query-error-m", type=float, default=14.0)
    parser.add_argument("--spacing-m", type=float, default=None)

    parser.add_argument("--corridor-m", type=float, default=14.0)
    parser.add_argument("--min-step-m", type=float, default=1.0)
    parser.add_argument("--preferred-step-m", type=float, default=5.5)
    parser.add_argument("--max-step-m", type=float, default=10.0)
    parser.add_argument("--hard-max-step-m", type=float, default=18.0)
    parser.add_argument("--lookahead-m", type=float, default=24.0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

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
        city_data[city] = (rows, points)
        print(f"{city} raw samples={len(rows)}", flush=True)

    summaries = []
    for name, (city, planned_route) in ROUTES.items():
        rows, points = city_data[city]
        selected = select_actual_sequence(
            points,
            planned_route,
            corridor_m=args.corridor_m,
            min_step_m=args.min_step_m,
            preferred_step_m=args.preferred_step_m,
            preferred_max_step_m=args.max_step_m,
            hard_max_step_m=args.hard_max_step_m,
            lookahead_m=args.lookahead_m,
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
            "selection": "corridor collection plus bounded variable-step chaining",
            "variable_step": True,
            "corridor_m": float(args.corridor_m),
            "preferred_step_m": float(args.preferred_step_m),
            "preferred_max_step_m": float(args.max_step_m),
            "hard_max_step_m": float(args.hard_max_step_m),
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
