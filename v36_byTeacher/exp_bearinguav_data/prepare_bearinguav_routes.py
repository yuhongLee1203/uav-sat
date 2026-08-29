"""Create same-scene BearingUAV pseudo-sequences for v36_byTeacher.

BearingUAV-90K contains independent cross-view samples rather than a recorded
video trajectory.  This adapter therefore keeps two different geometric objects
explicit instead of drawing them as if they were the same thing:

1. The route/reference geometry is an exact piecewise-straight polyline.  It may
   contain horizontal, vertical, and diagonal legs.  Waypoints are placed on
   this exact polyline.
2. UAV images are real BearingUAV samples selected near that polyline.  Their
   own source positions remain the visual labels; they are never snapped onto
   the route.  The shared v36 controlled protocol later applies its normal
   deterministic smooth jitter to the local search prior.

All splits use city-A only.  train_1/train_2 are training routes and val_1 is a
validation route.  Different routes may cross, source images are not reused
across routes, and no generated route exceeds 600 frames.
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
MAX_ROUTE_FRAMES = 600
DEFAULT_MIN_ACCEPTED_FRAMES = 400


def irregular_polyline(points):
    route = [tuple(map(float, p)) for p in points]
    if len(route) < 3:
        raise ValueError("an irregular route needs at least three vertices")
    return route


# Sparse, normal-looking sorties across the same map.  Individual legs are
# straight; directions are intentionally varied and different routes may cross.
ROUTES = {
    "train_1": (
        SCENE,
        irregular_polyline([
            (450, 650),
            (1250, 350),
            (2050, 950),
            (3100, 550),
            (3500, 1500),
            (2700, 2200),
            (3350, 3150),
            (2200, 3600),
            (1250, 2850),
            (500, 3450),
        ]),
    ),
    "train_2": (
        SCENE,
        irregular_polyline([
            (650, 3300),
            (900, 2350),
            (450, 1550),
            (1500, 900),
            (2450, 1450),
            (3350, 650),
            (3600, 1850),
            (2550, 2550),
            (3450, 3500),
            (1800, 3150),
            (850, 3750),
        ]),
    ),
    "val_1": (
        SCENE,
        irregular_polyline([
            (500, 2150),
            (1200, 1750),
            (850, 900),
            (2050, 500),
            (2700, 1150),
            (2050, 1950),
            (3000, 2700),
            (2350, 3450),
            (1200, 3050),
            (550, 2450),
        ]),
    ),
}

# Variable effective displacement per pseudo-frame.  This is a spatial
# pseudo-motion rate, not a measured physical UAV velocity.
ROUTE_PROFILES = {
    "train_1": {
        "name": "train_slow",
        "base_step_m": 3.2,
        "min_target_m": 2.2,
        "max_target_m": 4.2,
        "phase": 0,
    },
    "val_1": {
        "name": "validation_intermediate",
        "base_step_m": 3.9,
        "min_target_m": 2.8,
        "max_target_m": 5.0,
        "phase": 2,
    },
    "train_2": {
        "name": "train_fast",
        "base_step_m": 4.7,
        "min_target_m": 3.5,
        "max_target_m": 6.1,
        "phase": 4,
    },
}
SPEED_PATTERN = np.asarray([0.84, 1.05, 0.92, 1.16, 0.89, 1.10], dtype=np.float64)


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
    crosses = np.stack([-units[:, 1], units[:, 0]], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return wp, starts, delta, lengths, units, crosses, cumulative


def project_samples_to_route(points, waypoints):
    _wp, starts, _delta, lengths, units, crosses, cumulative = route_geometry(waypoints)
    best_dist = np.full(len(points), np.inf, dtype=np.float64)
    best_progress = np.zeros(len(points), dtype=np.float64)
    best_leg = np.zeros(len(points), dtype=np.int32)
    best_cross = np.zeros(len(points), dtype=np.float64)

    for leg, (start, unit, cross, length) in enumerate(zip(starts, units, crosses, lengths)):
        rel = points - start
        along = np.clip(np.sum(rel * unit, axis=1), 0.0, length)
        nearest = start + along[:, None] * unit
        residual = points - nearest
        dist = np.linalg.norm(residual, axis=1)
        signed_cross = np.sum(residual * cross, axis=1)
        mask = dist < best_dist
        best_dist[mask] = dist[mask]
        best_progress[mask] = cumulative[leg] + along[mask]
        best_leg[mask] = leg
        best_cross[mask] = signed_cross[mask]

    return best_progress, best_dist, best_leg, best_cross


def reference_pixel_at_progress(waypoints, progress_m):
    wp, starts, _delta, lengths, units, _crosses, cumulative = route_geometry(waypoints)
    progress_px = float(progress_m) / METERS_PER_PIXEL
    progress_px = float(np.clip(progress_px, 0.0, cumulative[-1]))
    leg = int(np.searchsorted(cumulative, progress_px, side="right") - 1)
    leg = int(np.clip(leg, 0, len(lengths) - 1))
    along = float(np.clip(progress_px - cumulative[leg], 0.0, lengths[leg]))
    return starts[leg] + along * units[leg], leg


def route_heading_at_progress(waypoints, progress_m):
    _wp, _starts, _delta, lengths, units, _crosses, cumulative = route_geometry(waypoints)
    progress_px = float(progress_m) / METERS_PER_PIXEL
    progress_px = float(np.clip(progress_px, 0.0, cumulative[-1]))
    leg = int(np.searchsorted(cumulative, progress_px, side="right") - 1)
    leg = int(np.clip(leg, 0, len(lengths) - 1))
    unit = units[leg]
    return float(math.atan2(float(unit[1]), float(unit[0])))


def target_step_for_frame(profile, frame_index):
    block = int(frame_index) // 36
    phase = int(profile.get("phase", 0))
    factor = float(SPEED_PATTERN[(block + phase) % len(SPEED_PATTERN)])
    wobble = 1.0 + 0.06 * math.sin(0.19 * float(frame_index) + 0.7 * phase)
    value = float(profile["base_step_m"]) * factor * wobble
    return float(np.clip(value, profile["min_target_m"], profile["max_target_m"]))


def _candidate_slice(order, ordered_progress, s0, lookahead_m):
    lo = int(np.searchsorted(ordered_progress, s0 + 0.15, side="left"))
    hi = int(np.searchsorted(ordered_progress, s0 + float(lookahead_m), side="right"))
    return order[lo:hi]


def _build_chain_from_start(
    start_index,
    points,
    progress_m,
    route_dist_m,
    route_cross_m,
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
            (ds > 0.10)
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
            max(float(profile["max_target_m"]) * 1.35, target * 1.40),
        )
        preferred_min = max(float(min_step_m), target * 0.40)
        preferred = (step >= preferred_min) & (step <= route_specific_preferred_max)
        fallback = not bool(np.any(preferred))
        if np.any(preferred):
            cand, ds, step = cand[preferred], ds[preferred], step[preferred]

        # Strongly prefer samples close to the straight reference segment and
        # avoid alternating from one side of the segment to the other.  This
        # removes the artificial saw-tooth look without moving any source label.
        lateral_change = np.abs(route_cross_m[cand] - route_cross_m[current])
        score = (
            np.abs(step - target)
            + 0.42 * np.abs(ds - target)
            + 0.85 * route_dist_m[cand]
            + 0.22 * lateral_change
            + 0.10 * np.maximum(0.0, legs[cand] - int(legs[current]))
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
    min_accepted_frames,
    profile,
    forbidden_indices=None,
):
    """Select up to target_frames real samples near one exact straight polyline."""
    target_frames = min(int(target_frames), MAX_ROUTE_FRAMES)
    progress_px, route_dist_px, legs, route_cross_px = project_samples_to_route(points, waypoints)
    progress_m = progress_px * METERS_PER_PIXEL
    route_dist_m = route_dist_px * METERS_PER_PIXEL
    route_cross_m = route_cross_px * METERS_PER_PIXEL

    keep_mask = route_dist_m <= float(corridor_m)
    if forbidden_indices:
        forbidden = np.fromiter((int(x) for x in forbidden_indices), dtype=np.int64)
        forbidden = forbidden[(forbidden >= 0) & (forbidden < len(points))]
        keep_mask[forbidden] = False
    keep = np.flatnonzero(keep_mask)
    if len(keep) < int(min_accepted_frames):
        raise RuntimeError(
            f"Only {len(keep)} source samples are inside the {corridor_m:.1f} m corridor; "
            f"need at least {min_accepted_frames}."
        )

    order = keep[np.argsort(progress_m[keep], kind="stable")]
    ordered_progress = progress_m[order]
    first_pool = order[: min(1024, len(order))]
    start_score = 1.20 * route_dist_m[first_pool] + 0.012 * (
        progress_m[first_pool] - progress_m[first_pool].min()
    )
    start_candidates = first_pool[np.argsort(start_score)[: min(48, len(first_pool))]]

    best_chain, best_flags = [], []
    for start in start_candidates:
        chain, flags = _build_chain_from_start(
            int(start), points, progress_m, route_dist_m, route_cross_m, legs,
            order, ordered_progress, target_frames, profile,
            min_step_m, preferred_max_step_m, hard_max_step_m, lookahead_m,
        )
        if len(chain) > len(best_chain):
            best_chain, best_flags = chain, flags
        if len(chain) >= target_frames:
            best_chain = chain[:target_frames]
            best_flags = flags[:target_frames]
            break

    if len(best_chain) < int(min_accepted_frames):
        raise RuntimeError(
            f"Only {len(best_chain)} continuous real samples could be chained for {profile['name']}; "
            f"minimum accepted is {min_accepted_frames}."
        )
    if len(best_chain) < target_frames:
        print(
            f"WARNING {profile['name']}: requested max {target_frames} frames, "
            f"using longest valid chain of {len(best_chain)} frames",
            flush=True,
        )

    rows = []
    previous = None
    for frame_index, (source_index, fallback) in enumerate(zip(best_chain, best_flags)):
        step_m = 0.0 if previous is None else float(
            np.linalg.norm(points[source_index] - points[previous]) * METERS_PER_PIXEL
        )
        reference_pixel, _ = reference_pixel_at_progress(waypoints, progress_m[source_index])
        rows.append({
            "source_index": int(source_index),
            "source_pixel": points[source_index].copy(),
            "reference_pixel": reference_pixel.copy(),
            "route_progress_m": float(progress_m[source_index]),
            "route_distance_m": float(route_dist_m[source_index]),
            "route_cross_m": float(route_cross_m[source_index]),
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
    top_lat, left_lon = 23.6, 120.0
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(top_lat)))
    bottom_lat = top_lat - MAP_PX * METERS_PER_PIXEL * lat_per_m
    right_lon = left_lon + MAP_PX * METERS_PER_PIXEL * lon_per_m
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
    lon = tl["longitude"] + float(pixel[0]) / (MAP_PX - 1) * (br["longitude"] - tl["longitude"])
    lat = tl["latitude"] - float(pixel[1]) / (MAP_PX - 1) * (tl["latitude"] - br["latitude"])
    return float(lat), float(lon)


def reference_waypoint_rows(selected, planned_route):
    """Build exact route waypoints while attaching the nearest selected frame id."""
    _wp, _starts, _delta, _lengths, _units, _crosses, cumulative_px = route_geometry(planned_route)
    selected_progress = np.asarray([x["route_progress_m"] for x in selected], dtype=np.float64)
    covered_min = float(selected_progress[0])
    covered_max = float(selected_progress[-1])

    candidates = [(covered_min, "start")]
    for progress_px in cumulative_px[1:-1]:
        progress_m = float(progress_px * METERS_PER_PIXEL)
        if covered_min + 1e-6 < progress_m < covered_max - 1e-6:
            candidates.append((progress_m, "turn"))
    candidates.append((covered_max, "end"))

    rows = []
    for progress_m, role in candidates:
        frame_index = int(np.argmin(np.abs(selected_progress - progress_m)))
        reference_pixel, _ = reference_pixel_at_progress(planned_route, progress_m)
        rows.append({
            "progress_m": float(progress_m),
            "role": role,
            "frame_index": frame_index,
            "reference_pixel": reference_pixel,
        })
    return rows


def write_route(name, city, planned_route, selected, metadata_rows, bounds, rebuild, profile, target_frames):
    root = OUTPUT_ROOT / name
    vi = root / "vi"
    root.mkdir(parents=True, exist_ok=True)
    vi.mkdir(parents=True, exist_ok=True)
    if rebuild:
        for stale in vi.glob("vi_*.jpg"):
            stale.unlink()

    actual = np.stack([x["source_pixel"] for x in selected])
    reference = np.stack([x["reference_pixel"] for x in selected])
    base_timestamp = 1_900_000_000_000_000_000
    timestamps = []

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
        ref_lat, ref_lon = pixel_to_latlon(item["reference_pixel"], bounds)
        heading = route_heading_at_progress(planned_route, item["route_progress_m"])
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
            "reference_latitude": ref_lat,
            "reference_longitude": ref_lon,
            "source_to_reference_m": float(item["route_distance_m"]),
        })

    (root / "sensor_with_yaw.json").write_text(json.dumps({"timestamp": timestamps}, indent=2))

    ref_dir = OUTPUT_ROOT / "reference_points"
    ref_dir.mkdir(exist_ok=True)
    ref_payload = {
        "route": name,
        "scene": city,
        "note": "exact piecewise-straight route reference points; source UAV labels remain unsnapped",
        "points": [
            {
                "frame_index": i,
                "route_progress_m": float(item["route_progress_m"]),
                "latitude": pixel_to_latlon(item["reference_pixel"], bounds)[0],
                "longitude": pixel_to_latlon(item["reference_pixel"], bounds)[1],
            }
            for i, item in enumerate(selected)
        ],
    }
    (ref_dir / f"{name}_reference_points.json").write_text(json.dumps(ref_payload, indent=2))

    waypoint_rows = reference_waypoint_rows(selected, planned_route)
    waypoints = []
    for order, wp in enumerate(waypoint_rows):
        frame_index = int(wp["frame_index"])
        item = selected[frame_index]
        ref_lat, ref_lon = pixel_to_latlon(wp["reference_pixel"], bounds)
        waypoints.append({
            "waypoint_order": order,
            "role": wp["role"],
            "frame_index": frame_index,
            "image": f"vi_{frame_index:06d}.jpg",
            "timestamp_ns": int(timestamps[frame_index]["timestamp_ns"]),
            "latitude": ref_lat,
            "longitude": ref_lon,
            "altitude_m": 120.0,
            "route_progress_m": float(wp["progress_m"]),
            "nearest_source_bearinguav_index": int(item["source_index"]),
            "geometry_source": "exact planned polyline",
        })

    wp_dir = OUTPUT_ROOT / "waypoints"
    wp_dir.mkdir(exist_ok=True)
    (wp_dir / f"{name}_waypoints.json").write_text(json.dumps({
        "route": name,
        "split": "validation" if name == "val_1" else "train",
        "scene": city,
        "position_source": "exact planned route geometry",
        "uav_label_source": "selected BearingUAV sample actual metadata position",
        "temporal_semantics": "spatially ordered actual-pose pseudo-sequence",
        "route_layout": "exact sparse piecewise-straight polyline; diagonal legs allowed",
        "waypoint_policy": "exact centerline start + traversed planned vertices + exact centerline end",
        "waypoints": waypoints,
    }, indent=2))

    steps = np.linalg.norm(np.diff(actual, axis=0), axis=1) * METERS_PER_PIXEL
    reference_steps = np.linalg.norm(np.diff(reference, axis=0), axis=1) * METERS_PER_PIXEL
    targets = np.asarray([x["target_step_m"] for x in selected[1:]], dtype=np.float64)
    corridor = np.asarray([x["route_distance_m"] for x in selected], dtype=np.float64)
    fallback = np.asarray([x["step_fallback"] for x in selected[1:]], dtype=np.float64)
    turn_count = sum(1 for x in waypoints if x["role"] == "turn")

    summary = {
        "route": name,
        "split": "validation" if name == "val_1" else "train",
        "scene": city,
        "frames": len(selected),
        "requested_max_frames": int(target_frames),
        "waypoints": len(waypoints),
        "turn_waypoints": int(turn_count),
        "planned_vertices": int(len(planned_route)),
        "reference_geometry": "exact piecewise-straight polyline",
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
        "reference_step_mean_m": float(np.mean(reference_steps)),
        "effective_speed_mean_mps": float(np.mean(steps) / PSEUDO_DT_S),
        "effective_speed_note": "derived from spatial pseudo-frame spacing; not a recorded UAV velocity",
        "fallback_step_pct": float(100.0 * np.mean(fallback)) if fallback.size else 0.0,
        "image_label_error_mean_m": 0.0,
        "image_label_error_p90_m": 0.0,
        "source_to_reference_mean_m": float(np.mean(corridor)),
        "source_to_reference_p90_m": float(np.quantile(corridor, 0.90)),
        "route_corridor_mean_m": float(np.mean(corridor)),
        "route_corridor_p90_m": float(np.quantile(corridor, 0.90)),
    }
    (root / "route_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def validate_summary(summary, min_accepted_frames):
    if summary["scene"] != SCENE:
        raise RuntimeError(f'{summary["route"]}: wrong scene {summary["scene"]}')
    if summary["frames"] > MAX_ROUTE_FRAMES:
        raise RuntimeError(f'{summary["route"]}: exceeds {MAX_ROUTE_FRAMES}-frame cap')
    if summary["frames"] < int(min_accepted_frames):
        raise RuntimeError(
            f'{summary["route"]}: only {summary["frames"]} frames; minimum is {min_accepted_frames}'
        )
    if summary["image_label_error_mean_m"] != 0.0:
        raise RuntimeError(f'{summary["route"]}: image/label mismatch')
    if summary["reference_geometry"] != "exact piecewise-straight polyline":
        raise RuntimeError(f'{summary["route"]}: reference route is not exact straight-segment geometry')
    if summary["step_std_m"] < 0.10:
        raise RuntimeError(f'{summary["route"]}: step distribution is still too fixed')
    if summary["turn_waypoints"] < 4:
        raise RuntimeError(f'{summary["route"]}: too few segment-junction waypoints')


def validate_speed_order(summaries):
    by_name = {x["route"]: x for x in summaries}
    t1 = float(by_name["train_1"]["step_mean_m"])
    va = float(by_name["val_1"]["step_mean_m"])
    t2 = float(by_name["train_2"]["step_mean_m"])
    if not (t1 < va < t2):
        print(
            "WARNING actual effective-speed order is not strictly train_1 < val_1 < train_2: "
            f"train_1={t1:.3f}, val_1={va:.3f}, train_2={t2:.3f} m/frame",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-spacing-m", type=float, default=None)
    parser.add_argument("--max-query-error-m", type=float, default=None)
    parser.add_argument("--spacing-m", type=float, default=None)
    parser.add_argument("--preferred-step-m", type=float, default=None)
    parser.add_argument("--corridor-m", type=float, default=18.0)
    parser.add_argument("--min-step-m", type=float, default=0.8)
    parser.add_argument("--max-step-m", type=float, default=13.0)
    parser.add_argument("--hard-max-step-m", type=float, default=22.0)
    parser.add_argument("--lookahead-m", type=float, default=34.0)
    parser.add_argument("--target-train-frames", type=int, default=MAX_ROUTE_FRAMES)
    parser.add_argument("--target-eval-frames", type=int, default=MAX_ROUTE_FRAMES)
    parser.add_argument("--min-accepted-frames", type=int, default=DEFAULT_MIN_ACCEPTED_FRAMES)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.min_step_m <= 0 or args.max_step_m <= args.min_step_m:
        raise ValueError("require 0 < min-step-m < max-step-m")
    if args.hard_max_step_m < args.max_step_m:
        raise ValueError("hard-max-step-m must be >= max-step-m")
    if not 64 <= args.min_accepted_frames <= MAX_ROUTE_FRAMES:
        raise ValueError(f"min-accepted-frames must be between 64 and {MAX_ROUTE_FRAMES}")
    if not 64 <= args.target_train_frames <= MAX_ROUTE_FRAMES:
        raise ValueError(f"target-train-frames must be between 64 and {MAX_ROUTE_FRAMES}")
    if not 64 <= args.target_eval_frames <= MAX_ROUTE_FRAMES:
        raise ValueError(f"target-eval-frames must be between 64 and {MAX_ROUTE_FRAMES}")
    if args.min_accepted_frames > min(args.target_train_frames, args.target_eval_frames):
        raise ValueError("min-accepted-frames cannot exceed target frame counts")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    satellite = make_satellite_image(args.rebuild)
    bounds, georef = satellite_georef()

    metadata = DATA_ROOT / SCENE / "rawmetadata.csv"
    metadata_rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
    points = np.stack([row_pixel(row) for row in metadata_rows])
    print(f"same-scene exact-reference protocol: scene={SCENE}, raw samples={len(metadata_rows)}", flush=True)

    summaries = []
    used_source_indices = set()
    for name in ("train_1", "train_2", "val_1"):
        city, planned_route = ROUTES[name]
        profile = dict(ROUTE_PROFILES[name])
        target_frames = args.target_eval_frames if name == "val_1" else args.target_train_frames
        print(
            f"building {name}: SAME scene={city}, max_frames={target_frames}, "
            f"profile={profile['name']}, base_step={profile['base_step_m']:.2f}m/frame, "
            f"straight_legs={len(planned_route)-1}, diagonal_allowed=True",
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
            min_accepted_frames=args.min_accepted_frames,
            profile=profile,
            forbidden_indices=used_source_indices,
        )
        used_source_indices.update(int(x["source_index"]) for x in selected)
        summary = write_route(
            name, city, planned_route, selected, metadata_rows,
            bounds, args.rebuild, profile, target_frames,
        )
        validate_summary(summary, args.min_accepted_frames)
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
            "route_layout": "irregular sparse exact straight-segment polylines across the full map",
            "diagonal_segments_allowed": True,
            "inter_route_crossings_allowed": True,
            "exact_source_image_reuse_across_routes": False,
        },
        "adapter": {
            "position_labels": "actual selected BearingUAV sample positions",
            "reference_geometry": "exact piecewise-straight planned route",
            "reference_points": "per-frame projection on the exact planned route; stored for inspection only",
            "local_prior_jitter": "shared v36 deterministic smooth jitter is applied at runtime; labels are not jittered",
            "temporal_order": "route-progress-ordered actual-pose pseudo-sequence",
            "selection": "same-city exact straight-polyline corridor plus smooth bounded variable-speed chaining",
            "variable_step": True,
            "max_route_frames": MAX_ROUTE_FRAMES,
            "target_train_frames": int(args.target_train_frames),
            "target_eval_frames": int(args.target_eval_frames),
            "min_accepted_frames": int(args.min_accepted_frames),
            "corridor_m": float(args.corridor_m),
            "preferred_max_step_m": float(args.max_step_m),
            "hard_max_step_m": float(args.hard_max_step_m),
            "lookahead_m": float(args.lookahead_m),
            "waypoint_policy": "exact centerline start + traversed planned vertices + exact centerline end",
            "speed_order_target": "train_1 < val_1 < train_2",
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
