#!/usr/bin/env python3
"""Freeze the dataset route trajectory as a frame-indexed reference manifest.

The tracker never reads current-frame GT to choose its local-search centre once
these files exist.  These manifests are nevertheless derived offline from the
dataset trajectory; they represent a known time-indexed route, not an
independently recorded autopilot mission plan.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from robust_tracker import RouteCache, WaypointRoute, build_gt_route_state, load_waypoint_xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.visual_checkpoint, map_location="cpu")
    origin_lat = float(checkpoint["origin_lat"])
    origin_lon = float(checkpoint["origin_lon"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for route_name in config.ROUTE_NAMES:
        output_path = args.output_dir / (route_name + ".npz")
        if output_path.exists() and not args.force:
            print("%s: reuse frozen dense route reference" % route_name, flush=True)
            continue
        cache_path = args.cache_dir / (route_name + "_uav_clip.pt")
        payload = torch.load(cache_path, map_location="cpu")
        cache = RouteCache(
            route_name=route_name,
            frame_ids=payload["frame_ids"],
            gt_xy=payload["gt_xy"],
            uav_clip=payload["uav_clip"],
            image_paths=payload["image_paths"],
        )
        route = WaypointRoute(load_waypoint_xy(route_name, origin_lat, origin_lon))
        state = build_gt_route_state(cache, route)
        np.savez_compressed(
            output_path,
            frame_ids=cache.frame_ids.cpu().numpy().astype(np.int64),
            reference_se=np.asarray(state["se"], dtype=np.float32),
            source=np.asarray("offline_dataset_route_projection"),
        )
        print("%s: wrote %d references -> %s" % (route_name, len(cache), output_path), flush=True)


if __name__ == "__main__":
    main()
