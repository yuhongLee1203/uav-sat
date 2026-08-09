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

RDP_EPSILON_M = 8.0
MIN_TURN_DEGREES = 45.0
MIN_LEG_LENGTH_M = 20.0


def local_meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius = 6378137.0
    lat0 = float(lat[0])
    x = np.radians(lon - lon[0]) * radius * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * radius
    return np.column_stack([x, y])


def rdp_indices(points: np.ndarray, epsilon_m: float) -> np.ndarray:
    """Return sampled vertices for a polyline under point-to-segment error."""
    if len(points) < 3:
        return np.arange(len(points), dtype=int)

    chosen = [0]

    def split(start: int, end: int) -> None:
        if end - start <= 1:
            return
        edge = points[end] - points[start]
        length = float(np.linalg.norm(edge))
        interior = points[start + 1:end]
        if length < 1e-8:
            distance = np.linalg.norm(interior - points[start], axis=1)
        else:
            delta = interior - points[start]
            distance = np.abs(delta[:, 0] * edge[1] - delta[:, 1] * edge[0]) / length
        if len(distance) and float(distance.max()) > epsilon_m:
            pivot = start + 1 + int(distance.argmax())
            split(start, pivot)
            chosen.append(pivot)
            split(pivot, end)

    split(0, len(points) - 1)
    chosen.append(len(points) - 1)
    return np.asarray(sorted(set(chosen)), dtype=int)


def signed_turn_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.dot(a, b))
    return float(math.degrees(math.atan2(cross, dot)))


def route_waypoints(sensor_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    samples = json.loads(sensor_path.read_text(encoding="utf-8"))["timestamp"]
    lat = np.asarray([row["latitude"] for row in samples], dtype=float)
    lon = np.asarray([row["longitude"] for row in samples], dtype=float)
    xy = local_meters(lat, lon)

    # GPS positions repeat between receiver updates. Keep only meaningful
    # movement for geometry, while retaining the original frame ID for output.
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]
    compact_xy = xy[keep]
    compact_to_frame = np.flatnonzero(keep)
    vertices = rdp_indices(compact_xy, RDP_EPSILON_M)

    # Only retain geometric vertices that separate two sufficiently long legs
    # and have an actual heading change. These are safe split boundaries.
    waypoint_compact = [int(vertices[0])]
    turns: dict[int, float] = {}
    for previous, current, following in zip(vertices[:-2], vertices[1:-1], vertices[2:]):
        first = compact_xy[current] - compact_xy[previous]
        second = compact_xy[following] - compact_xy[current]
        if min(np.linalg.norm(first), np.linalg.norm(second)) < MIN_LEG_LENGTH_M:
            continue
        turn = signed_turn_degrees(first, second)
        if abs(turn) >= MIN_TURN_DEGREES:
            waypoint_compact.append(int(current))
            turns[int(current)] = turn
    waypoint_compact.append(int(vertices[-1]))
    waypoint_compact = list(dict.fromkeys(waypoint_compact))

    result = []
    for order, compact_index in enumerate(waypoint_compact):
        frame_index = int(compact_to_frame[compact_index])
        row = samples[frame_index]
        record = {
            "waypoint_order": order,
            "role": "start" if order == 0 else "end" if order == len(waypoint_compact) - 1 else "turn",
            "frame_index": frame_index,
            "image": row["image"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "altitude_m": float(row["altitude"]),
            "local_x_m": float(xy[frame_index, 0]),
            "local_y_m": float(xy[frame_index, 1]),
            "turn_degrees": None if compact_index not in turns else float(turns[compact_index]),
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
    for suffix in ("png", "pdf"):
        figure.savefig(OUT / f"{route}_gps_waypoints.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = SatGeoMapper(SAT_JSON, SAT_IMAGE)
    with Image.open(SAT_IMAGE) as image:
        satellite = image.convert("RGB").copy()
    overview = {
        "method": "RDP GPS polyline simplification followed by direction-change filtering",
        "rdp_epsilon_m": RDP_EPSILON_M,
        "minimum_turn_degrees": MIN_TURN_DEGREES,
        "minimum_adjacent_leg_length_m": MIN_LEG_LENGTH_M,
        "important_limit": "The source folders contain sampled GPS but no PX4 mission-item/waypoint command list. Turn waypoints are GPS-derived estimates, not asserted flight-controller command coordinates.",
        "routes": {},
    }
    for route, source in ROUTES.items():
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
