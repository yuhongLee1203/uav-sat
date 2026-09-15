#!/usr/bin/env python3
"""Prepare Bearing-UAV pseudo-flight routes with long straight legs and big turns.

Bearing-UAV images are independent observations rather than consecutive video
frames. We therefore build a pseudo-flight sequence from real Bearing samples.
Each route is intentionally planned as a few long straight legs followed by
clear large turns, rather than frequent alternating bends. Within each straight
leg, the sample selector prefers observations close to the planned centreline
and with stable lateral offset so the GT trajectory is visually easier to read.

The smoothing cost is disabled across a genuine large turn, so it cannot flatten
the deliberate corners. UAV yaw is not used by the localization model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import bearing_prepare as base

SELECTION_VERSION = "soft_sequence_v5_long_straights_big_turns"

# Long-straight / big-turn route geometry. Each route stays in the same broad
# city region as before, but now contains only a few explicit corners. Typical
# straight legs are about 100-275 m and planned turns are roughly 43-101 deg.
# This makes it visually obvious whether a bend belongs to GT or prediction.
# Coordinates are in the canonical 4096x4096 Bearing RSI.
BIG_TURN_ROUTE_SPECS = {
    "train_01": [
        (330, 620),
        (1250, 700),
        (1500, 1150),
        (2600, 1200),
        (3000, 1620),
        (3330, 1480),
    ],
    "train_02": [
        (430, 3080),
        (1350, 3080),
        (1600, 2550),
        (2650, 2550),
        (3000, 3000),
        (3510, 2410),
    ],
    "train_03": [
        (3330, 430),
        (3330, 1350),
        (2950, 1750),
        (3550, 2500),
        (3150, 3050),
        (3310, 3610),
    ],
    "test_01": [
        (560, 1810),
        (1450, 1810),
        (1700, 1450),
        (2550, 1450),
        (2800, 2000),
        (3410, 2050),
    ],
    "test_02": [
        (900, 330),
        (900, 1250),
        (1350, 1600),
        (1350, 2450),
        (950, 2850),
        (1440, 3690),
    ],
}


def _angle_delta_abs_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _sequence_select(
    rows: pd.DataFrame,
    targets: np.ndarray,
    headings: np.ndarray,
    used_global: set,
    *,
    max_sample_distance_m: float,
    preferred_step_m: float,
    safety_max_step_m: float,
    candidate_limit: int,
    beam_width: int,
    skip_penalty: float,
    continuity_weight: float,
    large_step_weight: float,
    cross_weight: float,
    backward_weight: float,
    point_cross_weight: float,
    lateral_smooth_weight: float,
    big_turn_threshold_deg: float,
    min_selected_ratio: float,
):
    xy_px = rows[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    xy_m = xy_px * float(base.MPP)
    target_m = np.asarray(targets, dtype=np.float64) * float(base.MPP)
    identities = rows["target_path"].astype(str).tolist()
    blocked = np.asarray([identity in used_global for identity in identities], dtype=bool)
    max_sample_px = float(max_sample_distance_m) / float(base.MPP)

    heading_rad = np.deg2rad(np.asarray(headings, dtype=np.float64))
    heading_unit = np.stack([np.cos(heading_rad), np.sin(heading_rad)], axis=1)
    target_cross_axis = np.stack([-heading_unit[:, 1], heading_unit[:, 0]], axis=1)

    candidate_lists: List[List[Tuple[int, float]]] = []
    for target in targets:
        spatial_px = np.linalg.norm(xy_px - target[None, :], axis=1)
        valid = np.flatnonzero((spatial_px <= max_sample_px) & (~blocked))
        if valid.size == 0:
            candidate_lists.append([])
            continue
        err_m = spatial_px[valid] * float(base.MPP)
        order = np.argsort(err_m)[: int(candidate_limit)]
        candidate_lists.append([(int(valid[j]), float(err_m[j])) for j in order])

    # state = (cost, last_row, last_target, selected_rows, selected_targets,
    #          used_local, skips)
    beams = [(0.0, None, None, tuple(), tuple(), frozenset(), 0)]

    for target_index, candidates in enumerate(candidate_lists):
        expanded = []
        for state in beams:
            cost, last_idx, last_target, ids, tids, local_used, skips = state

            # Bearing observations are independent, so a target may be skipped.
            expanded.append(
                (
                    cost + float(skip_penalty),
                    last_idx,
                    last_target,
                    ids,
                    tids,
                    local_used,
                    skips + 1,
                )
            )

            for idx, target_err_m in candidates:
                if idx in local_used:
                    continue

                current_offset = xy_m[idx] - target_m[target_index]
                current_lateral = float(
                    np.dot(current_offset, target_cross_axis[target_index])
                )

                # Prefer a real UAV observation close to the planned centreline.
                # This never moves or relabels GT; it only selects a better real
                # Bearing observation from the available pool.
                transition = (
                    float(target_err_m)
                    + float(point_cross_weight) * abs(current_lateral)
                )

                if last_idx is not None:
                    delta = xy_m[idx] - xy_m[int(last_idx)]
                    step = float(np.linalg.norm(delta))
                    if step > float(safety_max_step_m) + 1e-9:
                        continue

                    ta = int(last_target)
                    tb = int(target_index)
                    route_delta = target_m[tb] - target_m[ta]
                    route_norm = float(np.linalg.norm(route_delta))
                    if route_norm <= 1e-9:
                        unit = heading_unit[tb]
                        desired = float(preferred_step_m)
                    else:
                        unit = route_delta / route_norm
                        desired = route_norm
                    cross_axis = np.asarray([-unit[1], unit[0]], dtype=np.float64)
                    along = float(np.dot(delta, unit))
                    cross = abs(float(np.dot(delta, cross_axis)))

                    transition += float(continuity_weight) * abs(step - desired)
                    transition += float(cross_weight) * cross
                    transition += float(backward_weight) * max(0.0, -along)
                    transition += float(large_step_weight) * max(
                        0.0, step - float(preferred_step_m)
                    ) ** 2

                    # Suppress only same-leg left/right wobble. Across an actual
                    # planned corner (>= threshold), this term is disabled so the
                    # large turn remains explicit.
                    planned_turn = _angle_delta_abs_deg(headings[tb], headings[ta])
                    if planned_turn < float(big_turn_threshold_deg):
                        previous_offset = xy_m[int(last_idx)] - target_m[ta]
                        previous_lateral = float(
                            np.dot(previous_offset, target_cross_axis[ta])
                        )
                        transition += float(lateral_smooth_weight) * abs(
                            current_lateral - previous_lateral
                        )

                expanded.append(
                    (
                        cost + transition,
                        idx,
                        target_index,
                        ids + (idx,),
                        tids + (target_index,),
                        local_used | {idx},
                        skips,
                    )
                )

        if not expanded:
            raise RuntimeError(f"sequence search became empty at target {target_index}")

        # First keep as many targets as possible, then choose the smoothest chain.
        expanded.sort(key=lambda s: (-len(s[3]), s[0], s[6]))
        beams = expanded[: int(beam_width)]

    minimum = max(2, int(np.ceil(float(min_selected_ratio) * len(targets))))
    feasible = [state for state in beams if len(state[3]) >= minimum]
    if not feasible:
        best_count = max(len(state[3]) for state in beams)
        raise RuntimeError(
            "No sufficiently dense soft-temporal route: required=%d/%d best=%d. "
            "Try --safety-max-step-m 24 before increasing sample distance."
            % (minimum, len(targets), best_count)
        )

    best = min(feasible, key=lambda s: (s[0], -len(s[3]), s[6]))
    ids = list(best[3])
    target_ids = list(best[4])
    for idx in ids:
        used_global.add(identities[idx])

    selected_idx = np.asarray(ids, dtype=np.int64)
    selected_tid = np.asarray(target_ids, dtype=np.int64)
    selected_xy = xy_m[selected_idx]
    selected_targets_m = target_m[selected_tid]
    selected_cross_axis = target_cross_axis[selected_tid]

    steps = (
        np.linalg.norm(np.diff(selected_xy, axis=0), axis=1)
        if len(ids) > 1
        else np.zeros(0, dtype=np.float64)
    )
    target_error = np.linalg.norm(selected_xy - selected_targets_m, axis=1)
    signed_lateral = np.sum(
        (selected_xy - selected_targets_m) * selected_cross_axis, axis=1
    )
    abs_lateral = np.abs(signed_lateral)

    backward = 0
    cross_values = []
    same_leg_lateral_delta = []
    for k in range(1, len(ids)):
        ta, tb = int(target_ids[k - 1]), int(target_ids[k])
        route_delta = target_m[tb] - target_m[ta]
        norm = float(np.linalg.norm(route_delta))
        if norm > 1e-9:
            unit = route_delta / norm
            cross_axis = np.asarray([-unit[1], unit[0]], dtype=np.float64)
            delta = selected_xy[k] - selected_xy[k - 1]
            along = float(np.dot(delta, unit))
            cross_values.append(abs(float(np.dot(delta, cross_axis))))
            backward += int(along < -1e-6)

        if _angle_delta_abs_deg(headings[tb], headings[ta]) < float(big_turn_threshold_deg):
            same_leg_lateral_delta.append(
                abs(float(signed_lateral[k] - signed_lateral[k - 1]))
            )

    def pct_over(value: float) -> float:
        return float(100.0 * np.mean(steps > value)) if len(steps) else 0.0

    diag = {
        "targets": int(len(targets)),
        "frames": int(len(ids)),
        "selected_ratio": float(len(ids) / max(len(targets), 1)),
        "skipped_targets": int(len(targets) - len(ids)),
        "mean_target_error_m": float(target_error.mean()) if len(target_error) else 0.0,
        "max_target_error_m": float(target_error.max()) if len(target_error) else 0.0,
        "centerline_cross_mean_m": float(abs_lateral.mean()) if len(abs_lateral) else 0.0,
        "centerline_cross_p90_m": float(np.percentile(abs_lateral, 90)) if len(abs_lateral) else 0.0,
        "centerline_cross_max_m": float(abs_lateral.max()) if len(abs_lateral) else 0.0,
        "same_leg_lateral_delta_p90_m": (
            float(np.percentile(same_leg_lateral_delta, 90))
            if same_leg_lateral_delta else 0.0
        ),
        "actual_step_mean_m": float(steps.mean()) if len(steps) else 0.0,
        "actual_step_p50_m": float(np.percentile(steps, 50)) if len(steps) else 0.0,
        "actual_step_p90_m": float(np.percentile(steps, 90)) if len(steps) else 0.0,
        "actual_step_p95_m": float(np.percentile(steps, 95)) if len(steps) else 0.0,
        "actual_step_max_m": float(steps.max()) if len(steps) else 0.0,
        "step_over_7m_pct": pct_over(7.0),
        "step_over_10m_pct": pct_over(10.0),
        "step_over_14m_pct": pct_over(14.0),
        "cross_step_p90_m": float(np.percentile(cross_values, 90)) if cross_values else 0.0,
        "backward_step_pct": float(100.0 * backward / max(len(ids) - 1, 1)),
    }
    return ids, target_ids, diag


def prepare(args):
    dataset_root = Path(args.dataset_root).resolve()
    city = args.city.lower()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(__file__).resolve().parent / "generated" / city
    )
    output_root.mkdir(parents=True, exist_ok=True)

    sat_path = base._find_satellite(dataset_root, city)
    metadata_path = base._find_metadata(dataset_root)
    with Image.open(sat_path) as image:
        width, height = image.size
    if (width, height) != (base.REFERENCE_SIZE, base.REFERENCE_SIZE):
        raise ValueError(f"Expected {base.REFERENCE_SIZE}x{base.REFERENCE_SIZE} RSI")

    rows = base._city_rows(pd.read_csv(metadata_path), city)
    basename_index = base._build_basename_index(dataset_root, city)

    routes = {
        name: base._scale_route(points, width, height)
        for name, points in BIG_TURN_ROUTE_SPECS.items()
    }

    used_global, stats = set(), {}
    for name in (*base.TRAIN_ROUTES, *base.TEST_ROUTES):
        planned = routes[name]
        targets, headings = base._dense_targets(planned, float(args.step_m))
        ids, target_ids, diag = _sequence_select(
            rows,
            targets,
            headings,
            used_global,
            max_sample_distance_m=float(args.max_sample_distance_m),
            preferred_step_m=float(args.preferred_step_m),
            safety_max_step_m=float(args.safety_max_step_m),
            candidate_limit=int(args.candidate_limit),
            beam_width=int(args.beam_width),
            skip_penalty=float(args.skip_penalty),
            continuity_weight=float(args.continuity_weight),
            large_step_weight=float(args.large_step_weight),
            cross_weight=float(args.cross_weight),
            backward_weight=float(args.backward_weight),
            point_cross_weight=float(args.point_cross_weight),
            lateral_smooth_weight=float(args.lateral_smooth_weight),
            big_turn_threshold_deg=float(args.big_turn_threshold_deg),
            min_selected_ratio=float(args.min_selected_ratio),
        )
        selected = rows.iloc[ids].copy()
        paths = [
            base._resolve_image_path(value, dataset_root, city, basename_index)
            for value in selected["target_path"]
        ]
        base._write_route(
            output_root / "routes" / name, name, planned, selected, paths
        )

        stats[name] = {
            "split": "train" if name in base.TRAIN_ROUTES else "inference",
            "planned_length_m": base._route_length_px(planned) * float(base.MPP),
            "waypoints": len(planned),
            "frames": int(diag["frames"]),
            "sample_step_m": float(args.step_m),
            "selection_version": SELECTION_VERSION,
            "max_sample_distance_m": float(args.max_sample_distance_m),
            "preferred_step_m": float(args.preferred_step_m),
            "safety_max_step_m": float(args.safety_max_step_m),
            "point_cross_weight": float(args.point_cross_weight),
            "lateral_smooth_weight": float(args.lateral_smooth_weight),
            "big_turn_threshold_deg": float(args.big_turn_threshold_deg),
            "heading_weight_px_per_deg": 0.0,
            **diag,
        }
        print("[SEQUENCE]", name, json.dumps(stats[name], indent=2), flush=True)

    base._make_train_union(output_root)

    sat_meta = {
        "mode": "bearing_uav_pixel_meter",
        "mpp": base.MPP,
        "width": width,
        "height": height,
        "city": city,
        "rsi_id": base.CITY_TO_RSI[city][0],
        "satellite_image": str(sat_path),
        "metadata_csv": str(metadata_path),
    }
    (output_root / "bearing_satellite.json").write_text(
        json.dumps(sat_meta, indent=2), encoding="utf-8"
    )
    experiment = {
        "dataset_root": str(dataset_root),
        "city": city,
        "satellite_image": str(sat_path),
        "metadata_csv": str(metadata_path),
        "mpp": base.MPP,
        "sequence_selection_version": SELECTION_VERSION,
        "train_routes": list(base.TRAIN_ROUTES),
        "inference_routes": list(base.TEST_ROUTES),
        "route_stats": stats,
        "note": (
            "Long straight route legs with explicit large turns; real Bearing "
            "observations use same-leg centreline/lateral smoothing only. Yaw is not used."
        ),
    }
    (output_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    base._draw_preview(sat_path, routes, output_root / "route_plan_full_satellite.jpg")
    print("[DONE] experiment:", output_root / "experiment.json", flush=True)
    return output_root


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", default="cityb", choices=sorted(base.CITY_TO_RSI))
    p.add_argument("--output-root", default=None)
    p.add_argument("--step-m", type=float, default=8.0)
    p.add_argument("--max-sample-distance-m", type=float, default=15.0)
    p.add_argument("--preferred-step-m", type=float, default=8.0)
    p.add_argument("--safety-max-step-m", type=float, default=22.0)
    p.add_argument("--candidate-limit", type=int, default=64)
    p.add_argument("--beam-width", type=int, default=128)
    p.add_argument("--skip-penalty", type=float, default=30.0)
    p.add_argument("--continuity-weight", type=float, default=1.5)
    p.add_argument("--large-step-weight", type=float, default=0.35)
    p.add_argument("--cross-weight", type=float, default=1.0)
    p.add_argument("--backward-weight", type=float, default=8.0)
    p.add_argument("--point-cross-weight", type=float, default=5.0)
    p.add_argument("--lateral-smooth-weight", type=float, default=5.0)
    p.add_argument("--big-turn-threshold-deg", type=float, default=35.0)
    p.add_argument("--min-selected-ratio", type=float, default=0.70)
    return p


if __name__ == "__main__":
    prepare(build_parser().parse_args())
