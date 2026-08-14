#!/usr/bin/env python3
"""Derive GPS turn-waypoint manifests and review figures for three routes.

The source datasets expose sampled GPS but no PX4 mission-item list.  A marked
waypoint is therefore the sampled GPS vertex closest to a geometrically
detected direction change, not a claim about an unavailable flight-controller
command.  Each pair of consecutive waypoints defines one straight flight leg.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Reuse the same EPSG:3826/world-file conversion as the localization code.
# The route utility lives at repository root while the mapper belongs to CRF.
from CRF.data import SatGeoMapper


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "route_waypoints"
SAT_IMAGE = ROOT.parents[0] / "sim_data" / "sim_competition_crop_check" / "sim_map_competition_roi_crop.png"
SAT_JSON = ROOT.parents[0] / "sim_data" / "sim_competition_crop_check" / "sim_map_competition_roi_crop_worldfile_epsg3826.json"
ROUTES = {
    "route_A": ROOT.parents[0] / "new_data_2" / "model_dataset_new_1_flight" / "sensor_with_yaw.json",
    "route_B": ROOT.parents[0] / "new_data_2" / "model_dataset_new_2_flight" / "sensor_with_yaw.json",
    "route_C": ROOT.parents[0] / "new_data" / "model_dataset_flight" / "sensor_with_yaw.json",
}

# A waypoint is detected from the *full* GPS trajectory.  The heading before
# and after a frame is estimated over this many samples, which makes the
# detector robust to per-frame GPS noise while keeping a real turn localized.
HEADING_WINDOW_FRAMES = 30
POSITION_SMOOTHING_FRAMES = 11
# High-recall segmentation: even a shallow, sustained bend should become a
# boundary, because the intended downstream units are straight flight legs.
# The spatial checks below suppress one-frame GPS jitter instead of relying on
# a large angle threshold that would hide genuine small turns.
MIN_TURN_DEGREES = 6.0
MAJOR_TURN_DEGREES = 45.0
MIN_MINOR_TURN_TRAVEL_M = 8.0
MIN_TURN_SEPARATION_FRAMES = 20
MIN_LEG_LENGTH_M = 20.0
MERGE_TURN_DISTANCE_M = 20.0
TERMINAL_SETTLE_DISTANCE_M = 20.0


def local_meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius = 6378137.0
    lat0 = float(lat[0])
    x = np.radians(lon - lon[0]) * radius * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * radius
    return np.column_stack([x, y])


def smooth_positions(points: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average without moving either endpoint."""
    if window <= 1:
        return points.copy()
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=float)
    return np.column_stack([
        np.convolve(padded[:, 0], kernel, mode="valid"),
        np.convolve(padded[:, 1], kernel, mode="valid"),
    ])


def wrapped_angle_degrees(angle: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))


def turn_candidates(points: np.ndarray) -> tuple[list[int], dict[int, float], np.ndarray]:
    """Find separated heading-change maxima on the complete GPS polyline."""
    window = HEADING_WINDOW_FRAMES
    smoothed = smooth_positions(points, POSITION_SMOOTHING_FRAMES)
    if len(points) <= 2 * window:
        return [], {}, smoothed

    before = smoothed[window:-window] - smoothed[:-2 * window]
    after = smoothed[2 * window:] - smoothed[window:-window]
    heading_before = np.arctan2(before[:, 1], before[:, 0])
    heading_after = np.arctan2(after[:, 1], after[:, 0])
    signed_change = wrapped_angle_degrees(heading_after - heading_before)
    magnitude = np.abs(signed_change)

    # Start from local extrema, rather than accepting every intermediate GPS
    # sample on a smooth curve. Major turns remain valid when the aircraft
    # pauses at a waypoint; shallow bends need real motion on both sides.
    local_peaks: list[int] = []
    for offset in range(1, len(magnitude) - 1):
        if magnitude[offset] < MIN_TURN_DEGREES:
            continue
        if magnitude[offset] < magnitude[offset - 1] or magnitude[offset] < magnitude[offset + 1]:
            continue
        frame = int(offset + window)
        if magnitude[offset] < MAJOR_TURN_DEGREES:
            left = max(0, frame - window)
            right = min(len(smoothed) - 1, frame + window)
            if min(np.linalg.norm(smoothed[frame] - smoothed[left]), np.linalg.norm(smoothed[right] - smoothed[frame])) < MIN_MINOR_TURN_TRAVEL_M:
                continue
        local_peaks.append(offset)

    # Greedily keep only the strongest peak in a short temporal neighborhood.
    selected: list[int] = []
    for offset in sorted(local_peaks, key=lambda index: magnitude[index], reverse=True):
        frame = int(offset + window)
        if all(abs(frame - prior) >= MIN_TURN_SEPARATION_FRAMES for prior in selected):
            selected.append(frame)
    selected.sort()

    turns: dict[int, float] = {}
    for frame in selected:
        turns[frame] = float(signed_change[frame - window])
    return selected, turns, smoothed


def consolidate_turns(candidates: list[int], turns: dict[int, float], xy: np.ndarray) -> tuple[list[int], int]:
    """Merge multiple heading peaks from one physical turn and trim endpoint dwell."""
    if not candidates:
        return [], len(xy) - 1

    groups: list[list[int]] = [[candidates[0]]]
    for frame in candidates[1:]:
        # A tight group of temporal peaks at the same world location is one
        # gradual turn, not multiple short straight missions.
        if np.linalg.norm(xy[frame] - xy[groups[-1][-1]]) < MERGE_TURN_DISTANCE_M:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    merged = [max(group, key=lambda frame: abs(turns[frame])) for group in groups]

    # If the last peak is already at the final stationary GPS position, it is
    # the mission end, rather than an extra turn followed by a zero-length leg.
    end_frame = len(xy) - 1
    if merged and np.linalg.norm(xy[merged[-1]] - xy[-1]) < TERMINAL_SETTLE_DISTANCE_M:
        end_frame = merged.pop()

    # Ensure each retained pair really forms a spatially meaningful straight
    # leg. In a rare near-duplicate pair, retain the stronger heading peak.
    result: list[int] = []
    for frame in merged:
        if result and np.linalg.norm(xy[frame] - xy[result[-1]]) < MIN_LEG_LENGTH_M:
            if abs(turns[frame]) > abs(turns[result[-1]]):
                result[-1] = frame
        else:
            result.append(frame)
    return result, end_frame


def route_waypoints(sensor_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    samples = json.loads(sensor_path.read_text(encoding="utf-8"))["timestamp"]
    lat = np.asarray([row["latitude"] for row in samples], dtype=float)
    lon = np.asarray([row["longitude"] for row in samples], dtype=float)
    xy = local_meters(lat, lon)

    candidates, turns, _ = turn_candidates(xy)
    waypoint_frames, end_frame = consolidate_turns(candidates, turns, xy)
    waypoint_frames = [0, *waypoint_frames, end_frame]

    result = []
    for order, frame_index in enumerate(waypoint_frames):
        row = samples[frame_index]
        record = {
            "waypoint_order": order,
            "role": "start" if order == 0 else "end" if order == len(waypoint_frames) - 1 else "turn",
            "frame_index": frame_index,
            "image": row["image"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "altitude_m": float(row["altitude"]),
            "local_x_m": float(xy[frame_index, 0]),
            "local_y_m": float(xy[frame_index, 1]),
            "turn_degrees": None if frame_index not in turns else float(turns[frame_index]),
        }
        result.append(record)
    return result, xy, np.asarray(samples, dtype=object)


def to_pixels(records: list[dict], mapper: SatGeoMapper) -> np.ndarray:
    return np.asarray([
        mapper.latlon_to_pixel(record["latitude"], record["longitude"])
        for record in records
    ], dtype=float)


def crop_background(image: Image.Image, points: np.ndarray, pad: int = 260) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    left = max(0, int(np.floor(points[:, 0].min())) - pad)
    right = min(image.width, int(np.ceil(points[:, 0].max())) + pad)
    top = max(0, int(np.floor(points[:, 1].min())) - pad)
    bottom = min(image.height, int(np.ceil(points[:, 1].max())) + pad)
    return np.asarray(image.crop((left, top, right, bottom))), (left, right, bottom, top)


def draw_route(route: str, records: list[dict], samples: np.ndarray, mapper: SatGeoMapper, sat: Image.Image) -> None:
    full = np.asarray([
        mapper.latlon_to_pixel(float(row["latitude"]), float(row["longitude"]))
        for row in samples
    ], dtype=float)
    points = to_pixels(records, mapper)
    background, extent = crop_background(sat, np.vstack([full, points]))
    figure, axis = plt.subplots(figsize=(11, 10), constrained_layout=True)
    axis.imshow(background, extent=extent, origin="upper")

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(records) - 1)))
    for leg, (start, end) in enumerate(zip(points[:-1], points[1:]), start=1):
        axis.plot([start[0], end[0]], [start[1], end[1]], linewidth=3.4, color=colors[leg - 1], label=f"Leg {leg}", zorder=3)
    axis.plot(full[:, 0], full[:, 1], color="#111827", alpha=0.45, linewidth=1.0, label="Sampled GPS trajectory", zorder=2)
    axis.scatter(points[0, 0], points[0, 1], s=110, marker="o", color="#22c55e", edgecolor="white", linewidth=1.1, label="Start", zorder=5)
    axis.scatter(points[-1, 0], points[-1, 1], s=135, marker="s", color="#ef4444", edgecolor="white", linewidth=1.1, label="End", zorder=5)
    if len(points) > 2:
        axis.scatter(points[1:-1, 0], points[1:-1, 1], s=120, marker="X", color="#facc15", edgecolor="#111827", linewidth=0.8, label="GPS-derived turn waypoint", zorder=6)
    for record, point in zip(records, points):
        label = "S" if record["role"] == "start" else "E" if record["role"] == "end" else str(record["waypoint_order"])
        axis.annotate(label, point, xytext=(6, 6), textcoords="offset points", color="white", fontsize=9, weight="bold", bbox={"facecolor": "#111827", "alpha": 0.84, "pad": 1.5, "edgecolor": "none"}, zorder=8)
    axis.set_title(f"{route}: GPS-derived straight-leg waypoint boundaries", weight="bold", fontsize=14)
    axis.set_xticks([]); axis.set_yticks([])
    axis.legend(loc="upper right", fontsize=8, framealpha=0.94, ncol=2)
    for suffix in ("png",):
        figure.savefig(OUT / f"{route}_gps_waypoints.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def draw_turn_diagnostic(route: str, records: list[dict], samples: np.ndarray, mapper: SatGeoMapper, sat: Image.Image) -> None:
    """A transparent diagnostic: full GPS curve first, detected turns second."""
    full = np.asarray([
        mapper.latlon_to_pixel(float(row["latitude"]), float(row["longitude"]))
        for row in samples
    ], dtype=float)
    points = to_pixels(records, mapper)
    background, extent = crop_background(sat, np.vstack([full, points]))
    figure, axis = plt.subplots(figsize=(11, 10), constrained_layout=True)
    axis.imshow(background, extent=extent, origin="upper", alpha=0.72)
    axis.plot(full[:, 0], full[:, 1], color="#f8fafc", linewidth=3.3, alpha=0.95, zorder=2, label="Complete sampled GPS path")
    axis.plot(full[:, 0], full[:, 1], color="#111827", linewidth=1.25, alpha=0.95, zorder=3)
    axis.scatter(points[0, 0], points[0, 1], s=115, marker="o", color="#22c55e", edgecolor="white", linewidth=1.2, label="Start", zorder=6)
    axis.scatter(points[-1, 0], points[-1, 1], s=140, marker="s", color="#ef4444", edgecolor="white", linewidth=1.2, label="End", zorder=6)
    if len(points) > 2:
        axis.scatter(points[1:-1, 0], points[1:-1, 1], s=135, marker="X", color="#facc15", edgecolor="#111827", linewidth=1.0, label="Detected turn waypoint", zorder=7)
    for record, point in zip(records, points):
        label = "S" if record["role"] == "start" else "E" if record["role"] == "end" else str(record["waypoint_order"])
        axis.annotate(label, point, xytext=(7, 7), textcoords="offset points", color="white", fontsize=9, weight="bold", bbox={"facecolor": "#111827", "alpha": 0.88, "pad": 1.5, "edgecolor": "none"}, zorder=8)
    axis.set_title(f"{route}: complete GPS path and detected turn boundaries", weight="bold", fontsize=14)
    axis.set_xticks([]); axis.set_yticks([])
    axis.legend(loc="upper right", fontsize=9, framealpha=0.95)
    for suffix in ("png",):
        figure.savefig(OUT / f"{route}_turn_alignment_check.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=sorted(ROUTES), help="Regenerate one route only.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = SatGeoMapper(SAT_JSON, SAT_IMAGE)
    with Image.open(SAT_IMAGE) as image:
        satellite = image.convert("RGB").copy()
    overview = {
        "method": "smoothed full-GPS heading-change peaks with non-maximum suppression",
        "heading_window_frames": HEADING_WINDOW_FRAMES,
        "position_smoothing_frames": POSITION_SMOOTHING_FRAMES,
        "minimum_turn_degrees": MIN_TURN_DEGREES,
        "major_turn_degrees": MAJOR_TURN_DEGREES,
        "minimum_minor_turn_travel_m": MIN_MINOR_TURN_TRAVEL_M,
        "minimum_turn_separation_frames": MIN_TURN_SEPARATION_FRAMES,
        "minimum_adjacent_leg_length_m": MIN_LEG_LENGTH_M,
        "merge_turn_distance_m": MERGE_TURN_DISTANCE_M,
        "terminal_settle_distance_m": TERMINAL_SETTLE_DISTANCE_M,
        "important_limit": "The source folders contain sampled GPS but no PX4 mission-item/waypoint command list. Turn waypoints are GPS-derived estimates, not asserted flight-controller command coordinates.",
        "routes": {},
    }
    selected_routes = ROUTES if args.route is None else {args.route: ROUTES[args.route]}
    for route, source in selected_routes.items():
        records, _, samples = route_waypoints(source)
        payload = {
            "route": route,
            "source": str(source),
            "waypoints": records,
            "straight_legs": [
                {
                    "leg": number,
                    "start_waypoint_order": left["waypoint_order"],
                    "end_waypoint_order": right["waypoint_order"],
                    "start_frame_index": left["frame_index"],
                    "end_frame_index": right["frame_index"],
                }
                for number, (left, right) in enumerate(zip(records[:-1], records[1:]), start=1)
            ],
        }
        (OUT / f"{route}_waypoints.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overview["routes"][route] = {
            "source": str(source),
            "waypoint_count": len(records),
            "turn_count": max(0, len(records) - 2),
            "leg_count": max(0, len(records) - 1),
        }
        draw_route(route, records, samples, mapper, satellite)
        draw_turn_diagnostic(route, records, samples, mapper, satellite)
    # A one-route regeneration must not discard the summaries for the other
    # already generated routes.
    for route, source in ROUTES.items():
        manifest = OUT / f"{route}_waypoints.json"
        if not manifest.exists():
            continue
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        overview["routes"][route] = {
            "source": str(source),
            "waypoint_count": len(stored["waypoints"]),
            "turn_count": max(0, len(stored["waypoints"]) - 2),
            "leg_count": len(stored["straight_legs"]),
        }
    (OUT / "waypoint_detection_summary.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# GPS-derived route waypoint manifests\n\n"
        "Each `route_*_waypoints.json` gives an ordered start, GPS-derived turn waypoints, and end. "
        "Each adjacent pair defines one straight leg for later segment-wise training/inference. "
        "The figures are mandatory visual checks before using the manifests.\n\n"
        "The source telemetry has no PX4 mission-item list, so these are geometrically derived GPS waypoint estimates rather than confirmed autopilot command waypoints.\n",
        encoding="utf-8",
    )
    print(f"Wrote GPS waypoint manifests and figures to {OUT}")


if __name__ == "__main__":
    main()
