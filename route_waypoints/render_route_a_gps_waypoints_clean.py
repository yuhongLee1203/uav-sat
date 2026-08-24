#!/usr/bin/env python3
"""Render Route A's waypoint map with lines and markers only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
UAVSAT = ROOT.parent
sys.path.insert(0, str(UAVSAT / "v37"))
from data import SatGeoMapper


def crop_background(image, points, pad=260):
    left = max(0, int(np.floor(points[:, 0].min())) - pad)
    right = min(image.width, int(np.ceil(points[:, 0].max())) + pad)
    top = max(0, int(np.floor(points[:, 1].min())) - pad)
    bottom = min(image.height, int(np.ceil(points[:, 1].max())) + pad)
    return np.asarray(image.crop((left, top, right, bottom))), (left, right, bottom, top)


def main():
    waypoint_path = ROOT / "route_A_waypoints.json"
    payload = json.loads(waypoint_path.read_text(encoding="utf-8"))
    waypoints = sorted(payload["waypoints"], key=lambda row: int(row["waypoint_order"]))
    sat_path = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop.png")
    sat_json = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop_worldfile_epsg3826.json")
    mapper = SatGeoMapper(sat_json, sat_path)
    points = np.asarray([mapper.latlon_to_pixel(row["latitude"], row["longitude"]) for row in waypoints])
    with Image.open(sat_path) as image:
        background, extent = crop_background(image.convert("RGB"), points)

    figure, axis = plt.subplots(figsize=(11, 10), constrained_layout=True)
    axis.imshow(background, extent=extent, origin="upper")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(points) - 1)))
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        color = "white" if index == 0 else colors[index]
        axis.plot([start[0], end[0]], [start[1], end[1]], linewidth=3.4, color=color, zorder=3)
    axis.scatter(points[0, 0], points[0, 1], s=110, marker="o", color="#22c55e", edgecolor="white", linewidth=1.1, zorder=5)
    axis.scatter(points[-1, 0], points[-1, 1], s=120, marker="X", color="#facc15", edgecolor="#111827", linewidth=0.8, zorder=6)
    if len(points) > 2:
        axis.scatter(points[1:-1, 0], points[1:-1, 1], s=82, marker="o", color="#ef4444", edgecolor="white", linewidth=1.0, zorder=6)
    axis.set_axis_off()
    output = ROOT / "route_A_gps_waypoints_clean.png"
    figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
