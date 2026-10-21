#!/usr/bin/env python3
"""Overlay the completed V36 MobileCLIP trajectories on bright satellite-map plots."""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "output/original_fornx_benchmark/v36_mobileclip2_s2"
GVSK = ROOT / "output/meanshift_gru"
SATELLITE = ROOT / "v36_training_data/satellite/sim_map_competition_roi_crop.png"
WORLD = ROOT / "v36_training_data/satellite/sim_map_competition_roi_crop_worldfile_epsg3826.json"
CHECKPOINT = ROOT / "../forNX/weights/v36_mobileclip2_s2/checkpoints/visual_retrieval_A_only.pt"

# Bright colors selected for the predominantly dark satellite image.
COLORS = {"gt": "#00E5FF", "kalman": "#FF42C8", "gvsk": "#FFE45C"}


def read_xy(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "gt": np.asarray([(float(r["gt_x"]), float(r["gt_y"])) for r in rows]),
        "final": np.asarray([(float(r["final_x"]), float(r["final_y"])) for r in rows]),
    }


def checkpoint_origin():
    import torch

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return float(payload["origin_lat"]), float(payload["origin_lon"])


def make_mapper():
    # Import only the geographic mapper; it is the exact V36 satellite affine.
    import sys

    sys.path.insert(0, str(ROOT / "meanshift_gru"))
    from data import SatGeoMapper

    return SatGeoMapper(WORLD, SATELLITE)


def local_xy_to_pixel(xy, origin_lat, origin_lon, mapper):
    radius = 6378137.0
    lat = origin_lat + np.degrees(xy[:, 1] / radius)
    lon = origin_lon + np.degrees(xy[:, 0] / (radius * math.cos(math.radians(origin_lat))))
    return np.asarray([mapper.latlon_to_pixel(a, b) for a, b in zip(lat, lon)])


def map_crop(image, all_pixels, margin=420):
    xmin = max(0, int(math.floor(all_pixels[:, 0].min() - margin)))
    xmax = min(image.width, int(math.ceil(all_pixels[:, 0].max() + margin)))
    ymin = max(0, int(math.floor(all_pixels[:, 1].min() - margin)))
    ymax = min(image.height, int(math.ceil(all_pixels[:, 1].max() + margin)))
    return image.crop((xmin, ymin, xmax, ymax)), xmin, ymin


def main():
    if not SATELLITE.is_file():
        raise FileNotFoundError(SATELLITE)
    origin_lat, origin_lon = checkpoint_origin()
    mapper = make_mapper()
    routes = {}
    for route in ("route_B", "route_C"):
        original = read_xy(ORIGINAL / f"{route}_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv")
        gvsk = read_xy(GVSK / f"{route}_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv")
        routes[route] = {
            "gt": local_xy_to_pixel(original["gt"], origin_lat, origin_lon, mapper),
            "kalman": local_xy_to_pixel(original["final"], origin_lat, origin_lon, mapper),
            "gvsk": local_xy_to_pixel(gvsk["final"], origin_lat, origin_lon, mapper),
        }

    Image.MAX_IMAGE_PIXELS = None
    variants = (
        ("kalman", "SoftMS + GRU + Kalman (K)", "with_kalman"),
        ("gvsk", "SoftMS + GRU visual final (no K)", "without_kalman"),
    )
    with Image.open(SATELLITE) as image:
        image = image.convert("RGB")
        for result_key, result_label, suffix in variants:
            figure, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=180, facecolor="#20232A")
            for axis, (route, values) in zip(axes, routes.items()):
                all_pixels = np.concatenate([values["gt"], values[result_key]], axis=0)
                crop, left, top = map_crop(image, all_pixels)
                axis.imshow(crop)
                for name, label, zorder in (("gt", "GT", 2), (result_key, result_label, 4)):
                    xy = values[name].copy()
                    xy[:, 0] -= left
                    xy[:, 1] -= top
                    axis.plot(xy[:, 0], xy[:, 1], color=COLORS[name], linewidth=2.4,
                              alpha=0.94, label=label, zorder=zorder)
                    axis.scatter(xy[0, 0], xy[0, 1], s=30, color=COLORS[name],
                                 edgecolors="#FFFFFF", linewidths=0.5, zorder=zorder + 1)
                axis.set_title(route.replace("_", " ").title(), color="white", fontsize=15, pad=10)
                axis.set_axis_off()
                legend = axis.legend(loc="upper right", fontsize=9, framealpha=0.78,
                                     facecolor="#333840", edgecolor="#FFFFFF")
                for text in legend.get_texts():
                    text.set_color("white")
            figure.suptitle(f"Localization on satellite map: {result_label} (MobileCLIP2-S2)",
                             color="white", fontsize=18, y=0.98)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            output = ROOT / f"output/localization_{suffix}_mobileclip2_s2_map.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, facecolor=figure.get_facecolor(), bbox_inches="tight")
            plt.close(figure)
            print(output)


if __name__ == "__main__":
    main()
