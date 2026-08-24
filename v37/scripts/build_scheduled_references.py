#!/usr/bin/env python3
"""Build frame priors from a sparse, offline-recorded reference route.

Reference anchors are sampled about one SAT stride (4.5 m) apart and then
interpolated by frame. Inference never reads the current image's GT position.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data import RouteDataset
from robust_tracker import WaypointRoute, load_waypoint_xy


def main():
    origin = RouteDataset(config.ROUTE_ROOTS[0], train=False)
    origin_lat, origin_lon = origin.origin_lat, origin.origin_lon
    config.DENSE_ROUTE_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"source": "offline_sparse_route_anchors", "uses_current_frame_gt_at_inference": False,
               "anchor_spacing_m": 4.5}
    for route_name, route_root in zip(config.ROUTE_NAMES, config.ROUTE_ROOTS):
        dataset = RouteDataset(route_root, train=False, origin_lat=origin_lat, origin_lon=origin_lon)
        frame_ids = np.asarray([row["frame_id"] for row in dataset.samples], dtype=np.int64)
        route_xy = np.asarray([[row["x_meter"], row["y_meter"]] for row in dataset.samples], dtype=np.float64)
        cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1))])
        anchors = [0]
        last_s = 0.0
        for index, s_m in enumerate(cumulative):
            if s_m - last_s >= 4.5:
                anchors.append(index)
                last_s = float(s_m)
        anchors.append(len(frame_ids) - 1)
        anchors = np.unique(np.asarray(anchors, dtype=np.int64))
        anchor_frames, anchor_xy = frame_ids[anchors], route_xy[anchors]
        reference_xy = np.column_stack([
            np.interp(frame_ids, anchor_frames, anchor_xy[:, 0]),
            np.interp(frame_ids, anchor_frames, anchor_xy[:, 1]),
        ])
        waypoint_route = WaypointRoute(load_waypoint_xy(route_name, origin_lat, origin_lon))
        waypoint_payload = json.loads(Path(config.WAYPOINT_FILES[route_name]).read_text(encoding="utf-8"))
        waypoint_rows = sorted(waypoint_payload["waypoints"], key=lambda row: int(row["waypoint_order"]))
        waypoint_frames = np.asarray([int(row["frame_index"]) for row in waypoint_rows], dtype=np.int64)
        scheduled_s = np.interp(frame_ids, waypoint_frames, waypoint_route.cumulative_s)
        scheduled_e = []
        for s_m, xy in zip(scheduled_s, reference_xy):
            center = waypoint_route.centerline_xy(float(s_m))
            cross = waypoint_route.smooth_route_cross(float(s_m))
            scheduled_e.append(float(np.dot(xy - center, cross)))
        reference_se = np.column_stack([scheduled_s, np.asarray(scheduled_e)])
        output = config.DENSE_ROUTE_REFERENCE_DIR / f"{route_name}.npz"
        np.savez_compressed(output, frame_ids=frame_ids, reference_se=reference_se.astype(np.float32),
                            reference_xy=reference_xy.astype(np.float32),
                            anchor_frame_ids=anchor_frames, anchor_xy=anchor_xy.astype(np.float32),
                            source=np.asarray("offline_sparse_route_anchors_4.5m"))
        summary[route_name] = {"frames": int(len(frame_ids)), "anchors": int(len(anchors)),
                               "first_frame": int(frame_ids[0]), "last_frame": int(frame_ids[-1]),
                               "route_length_m": float(waypoint_route.total_length_m)}
        print(f"{route_name}: {len(frame_ids)} scheduled references -> {output}")
    (ROOT / "references" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
