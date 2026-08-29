"""Create same-scene BearingUAV pseudo-sequences for v36_byTeacher.

BearingUAV-90K contains independent cross-view samples rather than a recorded
video trajectory. This adapter therefore constructs spatial pseudo-sequences
from REAL city-A samples only. Every output frame keeps the selected source
sample's own position label.

Protocol in this file:
- one satellite scene only: city-A;
- train_1 and train_2: exactly 1000 frames each;
- val_1: exactly 1000 frames;
- all three routes are spatially separated bands on the SAME satellite image;
- each route is a chain of many straight legs connected by 90-degree turns;
- every traversed planned turn is written as a waypoint;
- train_1 is slower, train_2 is faster, and val_1 is between them;
- effective speed means displacement per spatial pseudo-frame, not measured UAV
  velocity, because BearingUAV is not a temporal video sequence.
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
OUTPUT_ROOT = HERE / "generated_routes_2train_1val_samecity"
MAP_PX = 4096
METERS_PER_PIXEL = 0.25
PSEUDO_DT_S = 1.0 / 3.0
SCENE = "citya"
CITY_IMAGE = DATA_ROOT / "city_rsi/35.67091338738739_139.69289911300856_1791.95_1024_1024_4326_city.jpg"


def horizontal_multi_l(rows, x0, x1, y0, y1, reverse=False):
    """Long connected L-shaped route made from straight horizontal legs."""
    ys = np.linspace(float(y0), float(y1), int(rows))
    if reverse:
        ys = ys[::-1]
    out = []
    for i, y in enumerate(ys):
        left_to_right = (i % 2 == 0) ^ bool(reverse)
        a = (float(x0), float(y)) if left_to_right else (float(x1), float(y))
        b = (float(x1), float(y)) if left_to_right else (float(x0), float(y))
        if not out or out[-1] != a:
            out.append(a)
        out.append(b)
        if i + 1 < len(ys):
            out.append((float(b[0]), float(ys[i + 1])))
    return out


# Three disjoint bands of the SAME city-A satellite image. The gaps are much
# wider than the default 14 m corridor, so train/validation source pools do not
# collapse into the same local strip.
ROUTES = {
    "train_1": (
        SCENE,
        horizontal_multi_l(rows=12, x0=250, x1=3840, y0=250, y1=1200, reverse=False),
    ),
    "val_1": (
        SCENE,
        horizontal_multi_l(rows=11, x0=250, x1=3840, y0=1575, y1=2525, reverse=True),
    ),
    "train_2": (
        SCENE,
        horizontal_multi_l(rows=12, x0=250, x1=3840, y0=2900, y1=3850, reverse=True),
    ),
}

# Requested effective pseudo-motion order:
# train_1 (slow) < val_1 (intermediate) < train_2 (fast).
ROUTE_PROFILES = {
    "train_1": {
        "name": "train_slow",
        "base_step_m": 3.8,
        "min_target_m": 2.7,
        "max_target_m": 5.0,
        "phase": 0,
    },
    "val_1": {
        "name": "validation_intermediate",
        "base_step_m": 5.5,
        "min_target_m": 4.0,
        "max_target_m": 7.0,
        "phase": 2,
    },
    "train_2": {
        "name": "train_fast",
        "base_step_m": 7.3,
        "min_target_m": 5.6,
        "max_target_m": 9.4,
        "phase": 4,
    },
}
SPEED_PATTERN = np.asarray([0.82, 1.04, 0.91, 1.18, 0.88, 1.11], dtype=np.float64)


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


def route_geometry(waypoints):
    wp = np.asarray(waypoints, dtype=np.float64)
    starts = wp[:-1]
    delta = wp[1:] - wp[:-1]
    lengths = np.linalg.norm(delta, axis=1)
    units = delta / np.maximum(lengths[:, None], 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return wp, starts, delta, lengths, units, cumulative


def project_samples_to_route(points, waypoints):
    _wp, starts, _delta, lengths, units, cumulative = route_geometry(waypoints)
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


def target_step_for_frame(profile, frame_index):
    block = int(frame_index) // 45
    phase = int(profile.get("phase", 0))
    factor = float(SPEED_PATTERN[(block + phase) % len(SPEED_PATTERN)])
    wobble = 1.0 + 0.055 * math.sin(0.17 * float(frame_index) + 0.8 * phase)
    value = float(profile["base_step_m"]) * factor * wobble
    return float(np.clip(value, profile["min_target_m"], profile["max_target_m"]))


def _candidate_slice(order, ordered_progress, s0, lookahead_m):
    lo = int(np.searchsorted(ordered_progress, s0 + 0.20, side="left"))
    hi = int(np.searchsorted(ordered_progress, s0 + float(lookahead_m), side="right"))
    return order[lo:hi]


def _build_chain_from_start(
    start_index,
    points,
    progress_m,
    route_dist_m,
    legs,
    order,
    ordered_progress,
    target_frames,
    profile,
    min_step_m,
    preferred_max_step_m,
    hard_max_step_m,
    lookahead_m,
):
    chain = [int(start_index)]
    used = {int(start_index)}
    fallback_flags = [False]

    while len(chain) < int(target_frames):
        current = chain[-1]
        s0 = float(progress_m[current])
        target = target_step_for_frame(profile, len(chain))

        cand = _candidate_slice(order, ordered_progress, s0, lookahead_m)
        cand = np.asarray([int(i) for i in cand if int(i) not in used], dtype=np.int64)
        if cand.size == 0:
            cand = _candidate_slice(order, ordered_progress, s0, 2.0 * float(lookahead_m))
            cand = np.asarray([int(i) for i in cand if int(i) not in used], dtype=np.int64)
        if cand.size == 0:
            break

        ds = progress_m[cand] - s0
        step = np.linalg.norm(points[cand] - points[current], axis=1) * METERS_PER_PIXEL
        leg_delta = legs[cand] - int(legs[current])
        valid = (
            (ds > 0.15)
            & (step >= float(min_step_m))
            & (step <= float(hard_max_step_m))
            & (leg_delta >= 0)
            & (leg_delta <= 1)
        )
        if not np.any(valid):
            break
        cand, ds, step = cand[valid], ds[valid], step[valid]

        route_specific_preferred_max = min(
            float(preferred_max_step_m),
            max(float(profile["max_target_m"]) * 1.30, target * 1.35),
        )
        preferred_min = max(float(min_step_m), target * 0.45)
        preferred = (step >= preferred_min) & (step <= route_specific_preferred_max)
        fallback = not bool(np.any(preferred))
        if np.any(preferred):
            cand, ds, step = cand[preferred], ds[preferred], step[preferred]

        score = (
            np.abs(step - target)
            + 0.45 * np.abs(ds - target)
            + 0.34 * route_dist_m[cand]
            + 0.15 * np.maximum(0.0, legs[cand] - int(legs[current]))
        )
        nxt = int(cand[int(np.argmin(score))])
        actual_step = float(np.linalg.norm(points[nxt] - points[current]) * METERS_PER_PIXEL)
        chain.append(nxt)
        used.add(nxt)
        fallback_flags.append(bool(fallback or actual_step > route_specific_preferred_max))

    return chain, fallback_flags


def select_actual_sequence(
    points,
    waypoints,
    corridor_m,
    min_step_m,
    preferred_max_step_m,
    hard_max_step_m,
    lookahead_m,
    target_frames,
    profile,
):
    """Select exactly target_frames real samples with variable effective speed."""
    progress_px, route_dist_px, legs = project_samples_to_route(points, waypoints)
    progress_m = progress_px * METERS_PER_PIXEL
    route_dist_m = route_dist_px * METERS_PER_PIXEL
    keep = np.flatnonzero(route_dist_m <= float(corridor_m))
    if len(keep) < int(target_frames):
        raise RuntimeError(
            f"Only {len(keep)} source samples are inside the {corridor_m:.1f} m route corridor; "
            f"need at least {target_frames}."
        )

    order = keep[np.argsort(progress_m[keep], kind="stable")]
    ordered_progress = progress_m[order]
    first_pool = order[: min(768, len(order))]
    start_score = route_dist_m[first_pool] + 0.015 * (
        progress_m[first_pool] - progress_m[first_pool].min()
    )
    start_candidates = first_pool[np.argsort(start_score)[: min(32, len(first_pool))]]

    best_chain, best_flags = [], []
    for start in start_candidates:
        chain, flags = _build_chain_from_start(
            int(start), points, progress_m, route_dist_m, legs,
            order, ordered_progress, target_frames, profile,
            min_step_m, preferred_max_step_m, hard_max_step_m, lookahead_m,
        )
        if len(chain) > len(best_chain):
            best_chain, best_flags = chain, flags
        if len(chain) >= int(target_frames):
            best_chain = chain[: int(target_frames)]
            best_flags = flags[: int(target_frames)]
            break

    if len(best_chain) < int(target_frames):
        raise RuntimeError(
            f"Only {len(best_chain)}/{target_frames} continuous real samples could be chained "
            f"for {profile['name']}. Do not train yet; inspect this route and adjust the corridor."
        )

    rows = []
    previous = None
    for frame_index, (source_index, fallback) in enumerate(zip(best_chain, best_flags)):
        step_m = 0.0 if previous is None else float(
            np.linalg.norm(points[source_index] - points[previous]) * METERS_PER_PIXEL
        )
        rows.append({
            "source_index": int(source_index),
            "source_pixel": points[source_index].copy(),
            "route_progress_m": float(progress_m[source_index]),
            "route_distance_m": float(route_dist_m[source_index]),
            "step_m": step_m,
            "target_step_m": 0.0 if frame_index == 0 else target_step_for_frame(profile, frame_index),
            "step_fallback": bool(fallback),
            "leg": int(legs[source_index]),
        })
        previous = source_index
    return rows


def make_satellite_image(rebuild=False):
    path = OUTPUT_ROOT / "bearing_citya.jpg"
    if path.exists() and not rebuild:
        return path
    if not CITY_IMAGE.exists():
        raise FileNotFoundError(CITY_IMAGE)
    with Image.open(CITY_IMAGE) as image:
        image.convert("RGB").save(path, quality=95)
    return path


def satellite_georef():
    width_px = height_px = MAP_PX
    top_lat, left_lon = 23.6, 120.0
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(top_lat)))
    bottom_lat = top_lat - height_px * METERS_PER_PIXEL * lat_per_m
    right_lon = left_lon + width_px * METERS_PER_PIXEL * lon_per_m
    payload = {
        "scene": SCENE,
        "geo_bounds": {
            "top_left": {"latitude": top_lat, "longitude": left_lon},
            "bottom_right": {"latitude": bottom_lat, "longitude": right_lon},
        },
        "meters_per_pixel": METERS_PER_PIXEL,
        "width_px": MAP_PX,
        "height_px": MAP_PX,
    }
    path = OUTPUT_ROOT / "bearing_citya_geo.json"
    path.write_text(json.dumps(payload, indent=2))
    return payload, path


def pixel_to_latlon(pixel, bounds):
    tl = bounds["geo_bounds"]["top_left"]
    br = bounds["geo_bounds"]["bottom_right"]
    lon = tl["longitude"] + pixel[0] / (MAP_PX - 1) * (br["longitude"] - tl["longitude"])
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
    if np.linalg.norm(delta) <= 1e-9:
        return 0.0
    return float(math.atan2(float(delta[1]), float(delta[0])))


def planned_waypoint_indices(selected, planned_route):
    """Map every traversed planned 90-degree turn to the nearest selected frame."""
    _wp, _starts, _delta, _lengths, _units, cumulative_px = route_geometry(planned_route)
    turn_progress_m = cumulative_px * METERS_PER_PIXEL
    selected_progress = np.asarray([x["route_progress_m"] for x in selected], dtype=np.float64)
    covered_min = float(selected_progress[0])
    covered_max = float(selected_progress[-1])

    indices = [0]
    for progress in turn_progress_m[1:-1]:
        if covered_min - 1.0 <= progress <= covered_max + 1.0:
            indices.append(int(np.argmin(np.abs(selected_progress - float(progress)))))
    indices.append(len(selected) - 1)
    return sorted(set(indices))


def write_route(name, city, planned_route, selected, metadata_rows, bounds, rebuild, profile, target_frames):
    root = OUTPUT_ROOT / name
    vi = root / "vi"
    root.mkdir(parents=True, exist_ok=True)
    vi.mkdir(parents=True, exist_ok=True)
    if rebuild:
        for stale in vi.glob("vi_*.jpg"):
            stale.unlink()

    actual = np.stack([x["source_pixel"] for x in selected])
    wp_indices = set(planned_waypoint_indices(selected, planned_route))
    timestamps, waypoints = [], []
    waypoint_order = 0
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

        lat, lon = pixel_to_latlon(item["source_pixel"], bounds)
        heading = heading_from_points(actual, frame_index)
        timestamp_ns = base_timestamp + int(round(frame_index * PSEUDO_DT_S * 1e9))
        timestamps.append({
            "timestamp_ns": timestamp_ns,
            "image": target.name,
            "latitude": lat,
            "longitude": lon,
            "altitude": 120.0,
            "heading_rad": heading,
            "yaw": heading,
            "source_bearinguav_index": int(item["source_index"]),
            "route_progress_m": float(item["route_progress_m"]),
            "effective_step_m": float(item["step_m"]),
        })

        if frame_index in wp_indices:
            role = "start" if frame_index == 0 else (
                "end" if frame_index == len(selected) - 1 else "turn"
            )
            waypoints.append({
                "waypoint_order": waypoint_order,
                "role": role,
                "frame_index": frame_index,
                "image": target.name,
                "timestamp_ns": timestamp_ns,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": 120.0,
                "route_progress_m": float(item["route_progress_m"]),
                "source_bearinguav_index": int(item["source_index"]),
            })
            waypoint_order += 1

    (root / "sensor_with_yaw.json").write_text(json.dumps({"timestamp": timestamps}, indent=2))
    wp_dir = OUTPUT_ROOT / "waypoints"
    wp_dir.mkdir(exist_ok=True)
    (wp_dir / f"{name}_waypoints.json").write_text(json.dumps({
        "route": name,
        "split": "validation" if name == "val_1" else "train",
        "scene": city,
        "position_source": "selected BearingUAV sample actual metadata position",
        "temporal_semantics": "spatially ordered actual-pose pseudo-sequence",
        "waypoint_policy": "start + every traversed planned 90-degree turn + end",
        "waypoints": waypoints,
    }, indent=2))

    steps = np.linalg.norm(np.diff(actual, axis=0), axis=1) * METERS_PER_PIXEL
    targets = np.asarray([x["target_step_m"] for x in selected[1:]], dtype=np.float64)
    corridor = np.asarray([x["route_distance_m"] for x in selected], dtype=np.float64)
    fallback = np.asarray([x["step_fallback"] for x in selected[1:]], dtype=np.float64)
    turn_count = sum(1 for x in waypoints if x["role"] == "turn")
    summary = {
        "route": name,
        "split": "validation" if name == "val_1" else "train",
        "scene": city,
        "frames": len(selected),
        "target_frames": int(target_frames),
        "waypoints": len(waypoints),
        "turn_waypoints": int(turn_count),
        "speed_profile": str(profile["name"]),
        "target_step_base_m": float(profile["base_step_m"]),
        "target_step_mean_m": float(np.mean(targets)),
        "target_step_min_m": float(profile["min_target_m"]),
        "target_step_max_m": float(profile["max_target_m"]),
        "step_mean_m": float(np.mean(steps)),
        "step_std_m": float(np.std(steps)),
        "step_p10_m": float(np.quantile(steps, 0.10)),
        "step_p50_m": float(np.quantile(steps, 0.50)),
        "step_p90_m": float(np.quantile(steps, 0.90)),
        "step_min_m": float(np.min(steps)),
        "step_max_m": float(np.max(steps)),
        "effective_speed_mean_mps": float(np.mean(steps) / PSEUDO_DT_S),
        "effective_speed_note": "derived from spatial pseudo-frame spacing; not recorded UAV velocity",
        "fallback_step_pct": float(100.0 * np.mean(fallback)) if fallback.size else 0.0,
        "image_label_error_mean_m": 0.0,
        "image_label_error_p90_m": 0.0,
        "route_corridor_mean_m": float(np.mean(corridor)),
        "route_corridor_p90_m": float(np.quantile(corridor, 0.90)),
    }
    (root / "route_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def validate_summary(summary):
    if summary["scene"] != SCENE:
        raise RuntimeError(f'{summary["route"]}: wrong scene {summary["scene"]}')
    if summary["frames"] != summary["target_frames"]:
        raise RuntimeError(
            f'{summary["route"]}: expected exactly {summary["target_frames"]} frames, got {summary["frames"]}'
        )
    if summary["image_label_error_mean_m"] != 0.0:
        raise RuntimeError(f'{summary["route"]}: image/label mismatch')
    if summary["step_std_m"] < 0.10:
        raise RuntimeError(f'{summary["route"]}: step distribution is still too fixed')
    if summary["turn_waypoints"] < 4:
        raise RuntimeError(f'{summary["route"]}: too few turn waypoints for a multi-segment route')


def validate_speed_order(summaries):
    by_name = {x["route"]: x for x in summaries}
    t1 = float(by_name["train_1"]["step_mean_m"])
    va = float(by_name["val_1"]["step_mean_m"])
    t2 = float(by_name["train_2"]["step_mean_m"])
    if not (t1 < va < t2):
        raise RuntimeError(
            "actual effective-speed order must be train_1 < val_1 < train_2, got "
            f"{t1:.3f} < {va:.3f} < {t2:.3f} m/frame"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-spacing-m", type=float, default=None)
    parser.add_argument("--max-query-error-m", type=float, default=None)
    parser.add_argument("--spacing-m", type=float, default=None)
    parser.add_argument("--preferred-step-m", type=float, default=None)
    parser.add_argument("--corridor-m", type=float, default=14.0)
    parser.add_argument("--min-step-m", type=float, default=0.8)
    parser.add_argument("--max-step-m", type=float, default=13.0)
    parser.add_argument("--hard-max-step-m", type=float, default=22.0)
    parser.add_argument("--lookahead-m", type=float, default=32.0)
    parser.add_argument("--target-train-frames", type=int, default=1000)
    parser.add_argument("--target-eval-frames", type=int, default=1000)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.min_step_m <= 0 or args.max_step_m <= args.min_step_m:
        raise ValueError("require 0 < min-step-m < max-step-m")
    if args.hard_max_step_m < args.max_step_m:
        raise ValueError("hard-max-step-m must be >= max-step-m")
    if args.target_train_frames < 64 or args.target_eval_frames < 64:
        raise ValueError("target frame counts must be >= 64")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    satellite = make_satellite_image(args.rebuild)
    bounds, georef = satellite_georef()

    metadata = DATA_ROOT / SCENE / "rawmetadata.csv"
    metadata_rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
    points = np.stack([row_pixel(row) for row in metadata_rows])
    print(f"same-scene protocol: scene={SCENE}, raw samples={len(metadata_rows)}", flush=True)

    summaries = []
    for name in ("train_1", "train_2", "val_1"):
        city, planned_route = ROUTES[name]
        profile = dict(ROUTE_PROFILES[name])
        target_frames = args.target_eval_frames if name == "val_1" else args.target_train_frames
        print(
            f"building {name}: SAME scene={city}, target_frames={target_frames}, "
            f"profile={profile['name']}, base_step={profile['base_step_m']:.2f}m/frame, "
            f"planned_turns={len(planned_route)-2}",
            flush=True,
        )
        selected = select_actual_sequence(
            points,
            planned_route,
            corridor_m=args.corridor_m,
            min_step_m=args.min_step_m,
            preferred_max_step_m=args.max_step_m,
            hard_max_step_m=args.hard_max_step_m,
            lookahead_m=args.lookahead_m,
            target_frames=target_frames,
            profile=profile,
        )
        summary = write_route(
            name, city, planned_route, selected, metadata_rows,
            bounds, args.rebuild, profile, target_frames,
        )
        validate_summary(summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    validate_speed_order(summaries)

    payload = {
        "split": {
            "train": ["train_1", "train_2"],
            "validation": ["val_1"],
            "test": [],
        },
        "scene_policy": {
            "same_satellite_scene_for_all_splits": True,
            "scene": SCENE,
            "satellite_image": str(satellite),
            "train_validation_spatial_bands_are_separate": True,
        },
        "adapter": {
            "position_labels": "actual selected BearingUAV sample positions",
            "temporal_order": "route-progress-ordered actual-pose pseudo-sequence",
            "selection": "same-city multi-turn route corridor plus exact-count variable-speed chaining",
            "variable_step": True,
            "target_train_frames": int(args.target_train_frames),
            "target_eval_frames": int(args.target_eval_frames),
            "corridor_m": float(args.corridor_m),
            "preferred_max_step_m": float(args.max_step_m),
            "hard_max_step_m": float(args.hard_max_step_m),
            "lookahead_m": float(args.lookahead_m),
            "waypoint_policy": "start + every traversed planned 90-degree turn + end",
            "speed_order": "train_1 < val_1 < train_2",
            "speed_semantics": "effective displacement per spatial pseudo-frame, not recorded UAV velocity",
        },
        "satellite_image": str(satellite),
        "satellite_georef": str(georef),
        "routes": summaries,
    }
    (OUTPUT_ROOT / "generation_summary.json").write_text(json.dumps(payload, indent=2))
    print("generation summary:", OUTPUT_ROOT / "generation_summary.json", flush=True)


if __name__ == "__main__":
    main()
