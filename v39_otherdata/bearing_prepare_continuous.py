#!/usr/bin/env python3
"""Prepare temporally coherent Bearing-UAV pseudo-flight sequences for v39.

Bearing-UAV observations are independent images.  The old adapter selected one
image independently for every planned route target.  That produced pseudo-video
steps of 15-22+ m even though v39's constrained Kalman is designed for much
smaller frame-to-frame motion.  This adapter keeps the canonical v39 model
unchanged and fixes only external-dataset sequence construction.

Selection rules:
  * train/test image identities remain globally disjoint;
  * UAV yaw is NOT used by default;
  * candidates must stay near the planned route target;
  * consecutive selected observations obey a hard metric step limit;
  * backward and lateral frame-to-frame motion are constrained;
  * a target may be skipped instead of forcing a physically implausible frame;
  * beam search avoids the greedy dead-ends of the old nearest-per-target picker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import bearing_prepare as base

SELECTION_VERSION = "sequence_aware_v2"


def _angle_error_deg(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - float(target) + 180.0) % 360.0 - 180.0)


def _candidate_yaw(rows: pd.DataFrame):
    if "theta" in rows.columns:
        return np.mod(rows["theta"].to_numpy(dtype=np.float64), 360.0)
    if {"x_cosa", "y_sina"}.issubset(rows.columns):
        return np.mod(
            np.degrees(
                np.arctan2(
                    rows["y_sina"].to_numpy(dtype=np.float64),
                    rows["x_cosa"].to_numpy(dtype=np.float64),
                )
            ),
            360.0,
        )
    return None


def _sequence_select(
    rows: pd.DataFrame,
    targets: np.ndarray,
    headings: np.ndarray,
    used_global: set,
    *,
    max_sample_distance_m: float,
    max_frame_step_m: float,
    max_cross_step_m: float,
    max_backward_step_m: float,
    heading_weight_px_per_deg: float,
    continuity_weight: float,
    cross_weight: float,
    backward_weight: float,
    skip_penalty: float,
    candidate_limit: int,
    beam_width: int,
    min_selected_ratio: float,
):
    """Beam-search a coherent observation chain along the planned route.

    State tuple:
      (cost, last_row_idx, last_target_idx, selected_row_ids,
       selected_target_ids, used_local, consecutive_skips)
    """
    xy_px = rows[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    xy_m = xy_px * float(base.MPP)
    yaw = _candidate_yaw(rows)
    identities = rows["target_path"].astype(str).tolist()
    globally_blocked = np.asarray([identity in used_global for identity in identities], dtype=bool)
    max_sample_px = float(max_sample_distance_m) / float(base.MPP)

    # Precompute a small candidate list around every target.  Heading is only an
    # optional tie-break; the default is exactly zero so yaw is not used.
    candidate_lists: List[List[Tuple[int, float, float]]] = []
    for target, heading in zip(targets, headings):
        spatial_px = np.linalg.norm(xy_px - target[None, :], axis=1)
        valid = np.flatnonzero((spatial_px <= max_sample_px) & (~globally_blocked))
        if valid.size == 0:
            candidate_lists.append([])
            continue
        target_err_m = spatial_px[valid] * float(base.MPP)
        score = target_err_m.copy()
        if yaw is not None and float(heading_weight_px_per_deg) > 0.0:
            # Convert legacy px/deg weight to metres/deg before adding to metre score.
            score += (
                float(heading_weight_px_per_deg)
                * float(base.MPP)
                * _angle_error_deg(yaw[valid], float(heading))
            )
        order = np.argsort(score)[: int(candidate_limit)]
        candidate_lists.append(
            [
                (int(valid[j]), float(target_err_m[j]), float(score[j]))
                for j in order
            ]
        )

    # (cost, last_idx, last_target_idx, selected_ids, selected_target_ids,
    #  used_local, consecutive_skips)
    beams = [(0.0, None, None, tuple(), tuple(), frozenset(), 0)]

    for target_index, candidates in enumerate(candidate_lists):
        expanded = []
        for state in beams:
            (
                cost,
                last_idx,
                last_target_idx,
                selected_ids,
                selected_target_ids,
                used_local,
                consecutive_skips,
            ) = state

            # Skipping is always safer than fabricating a 15-25 m pseudo-frame.
            expanded.append(
                (
                    cost + float(skip_penalty),
                    last_idx,
                    last_target_idx,
                    selected_ids,
                    selected_target_ids,
                    used_local,
                    consecutive_skips + 1,
                )
            )

            for idx, target_err_m, base_score in candidates:
                if idx in used_local:
                    continue

                transition_cost = float(base_score)
                if last_idx is not None:
                    delta_m = xy_m[idx] - xy_m[int(last_idx)]
                    step_m = float(np.linalg.norm(delta_m))
                    if step_m > float(max_frame_step_m) + 1e-9:
                        continue

                    previous_target = targets[int(last_target_idx)]
                    current_target = targets[target_index]
                    route_delta = (current_target - previous_target) * float(base.MPP)
                    route_norm = float(np.linalg.norm(route_delta))
                    if route_norm <= 1e-9:
                        heading_rad = np.deg2rad(float(headings[target_index]))
                        route_unit = np.asarray(
                            [np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float64
                        )
                        desired_step_m = 0.0
                    else:
                        route_unit = route_delta / route_norm
                        desired_step_m = route_norm
                    route_cross = np.asarray([-route_unit[1], route_unit[0]])
                    along_m = float(np.dot(delta_m, route_unit))
                    cross_m = abs(float(np.dot(delta_m, route_cross)))
                    if along_m < -float(max_backward_step_m) - 1e-9:
                        continue
                    if cross_m > float(max_cross_step_m) + 1e-9:
                        continue

                    transition_cost += float(continuity_weight) * abs(
                        step_m - min(desired_step_m, float(max_frame_step_m))
                    )
                    transition_cost += float(cross_weight) * cross_m
                    transition_cost += float(backward_weight) * max(0.0, -along_m)

                expanded.append(
                    (
                        cost + transition_cost,
                        idx,
                        target_index,
                        selected_ids + (idx,),
                        selected_target_ids + (target_index,),
                        used_local | {idx},
                        0,
                    )
                )

        # Prefer low cost, then more selected frames, then fewer consecutive skips.
        expanded.sort(key=lambda s: (s[0], -len(s[3]), s[6]))
        beams = expanded[: int(beam_width)]
        if not beams:
            raise RuntimeError("Sequence beam became empty at target %d" % target_index)

    minimum = max(2, int(np.ceil(float(min_selected_ratio) * len(targets))))
    feasible = [state for state in beams if len(state[3]) >= minimum]
    if not feasible:
        best_count = max(len(state[3]) for state in beams)
        raise RuntimeError(
            "No temporally coherent route reached the minimum selected ratio: "
            "required=%d/%d, best=%d. Increase --max-frame-step-m slightly or "
            "increase --max-sample-distance-m; do NOT re-enable yaw weighting."
            % (minimum, len(targets), best_count)
        )
    best = min(feasible, key=lambda s: (s[0], -len(s[3]), s[6]))
    ids = list(best[3])
    target_ids = list(best[4])

    for idx in ids:
        used_global.add(identities[idx])

    selected_xy_m = xy_m[np.asarray(ids, dtype=np.int64)]
    steps = (
        np.linalg.norm(np.diff(selected_xy_m, axis=0), axis=1)
        if len(ids) > 1
        else np.zeros((0,), dtype=np.float64)
    )
    target_xy_px = targets[np.asarray(target_ids, dtype=np.int64)]
    target_error = np.linalg.norm(
        xy_px[np.asarray(ids, dtype=np.int64)] - target_xy_px,
        axis=1,
    ) * float(base.MPP)

    backward_count = 0
    cross_steps = []
    if len(ids) > 1:
        for k in range(1, len(ids)):
            ta = int(target_ids[k - 1])
            tb = int(target_ids[k])
            route_delta = (targets[tb] - targets[ta]) * float(base.MPP)
            norm = float(np.linalg.norm(route_delta))
            if norm <= 1e-9:
                continue
            unit = route_delta / norm
            cross = np.asarray([-unit[1], unit[0]])
            delta = selected_xy_m[k] - selected_xy_m[k - 1]
            along = float(np.dot(delta, unit))
            cross_steps.append(abs(float(np.dot(delta, cross))))
            backward_count += int(along < -1e-6)

    diagnostics = {
        "targets": int(len(targets)),
        "frames": int(len(ids)),
        "selected_ratio": float(len(ids) / max(len(targets), 1)),
        "skipped_targets": int(len(targets) - len(ids)),
        "mean_target_error_m": float(target_error.mean()) if len(target_error) else 0.0,
        "max_target_error_m": float(target_error.max()) if len(target_error) else 0.0,
        "actual_step_mean_m": float(steps.mean()) if len(steps) else 0.0,
        "actual_step_p90_m": float(np.percentile(steps, 90)) if len(steps) else 0.0,
        "actual_step_p95_m": float(np.percentile(steps, 95)) if len(steps) else 0.0,
        "actual_step_max_m": float(steps.max()) if len(steps) else 0.0,
        "cross_step_p90_m": float(np.percentile(cross_steps, 90)) if cross_steps else 0.0,
        "backward_step_pct": float(100.0 * backward_count / max(len(ids) - 1, 1)),
    }
    return ids, target_ids, diagnostics


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
        raise ValueError(
            "Expected official city RSI %dx%d, got %s"
            % (base.REFERENCE_SIZE, base.REFERENCE_SIZE, (width, height))
        )

    rows = base._city_rows(pd.read_csv(metadata_path), city)
    basename_index = base._build_basename_index(dataset_root, city)
    routes = {
        name: base._scale_route(points, width, height)
        for name, points in base.ROUTE_SPECS.items()
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
            max_frame_step_m=float(args.max_frame_step_m),
            max_cross_step_m=float(args.max_cross_step_m),
            max_backward_step_m=float(args.max_backward_step_m),
            heading_weight_px_per_deg=float(args.heading_weight_px_per_deg),
            continuity_weight=float(args.continuity_weight),
            cross_weight=float(args.cross_weight),
            backward_weight=float(args.backward_weight),
            skip_penalty=float(args.skip_penalty),
            candidate_limit=int(args.candidate_limit),
            beam_width=int(args.beam_width),
            min_selected_ratio=float(args.min_selected_ratio),
        )
        selected = rows.iloc[ids].copy()
        paths = [
            base._resolve_image_path(value, dataset_root, city, basename_index)
            for value in selected["target_path"]
        ]
        base._write_route(
            output_root / "routes" / name,
            name,
            planned,
            selected,
            paths,
        )

        stats[name] = {
            "split": "train" if name in base.TRAIN_ROUTES else "inference",
            "planned_length_m": base._route_length_px(planned) * float(base.MPP),
            "waypoints": len(planned),
            "frames": int(diag["frames"]),
            "sample_step_m": float(args.step_m),
            "selection_version": SELECTION_VERSION,
            "max_sample_distance_m": float(args.max_sample_distance_m),
            "max_frame_step_m": float(args.max_frame_step_m),
            "max_cross_step_m": float(args.max_cross_step_m),
            "max_backward_step_m": float(args.max_backward_step_m),
            "heading_weight_px_per_deg": float(args.heading_weight_px_per_deg),
            **diag,
        }
        print("[SEQUENCE]", name, json.dumps(stats[name], indent=2), flush=True)

        # Hard safety audit: the whole reason for this adapter is to stop the
        # 15-22 m pseudo-frame jumps that made the canonical v39 Kalman lag.
        if float(diag["actual_step_max_m"]) > float(args.max_frame_step_m) + 1e-6:
            raise RuntimeError(
                "%s violates max frame step: %.3f > %.3f m"
                % (name, diag["actual_step_max_m"], args.max_frame_step_m)
            )

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
            "Independent Bearing-UAV observations converted into globally-disjoint "
            "pseudo-flight sequences with explicit frame-to-frame continuity constraints."
        ),
    }
    (output_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    preview = output_root / "route_plan_full_satellite.jpg"
    base._draw_preview(sat_path, routes, preview)
    print("[DONE] preview:", preview, flush=True)
    print("[DONE] experiment:", output_root / "experiment.json", flush=True)
    return output_root


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K"
    )
    parser.add_argument(
        "--city", default="cityb", choices=sorted(base.CITY_TO_RSI)
    )
    parser.add_argument("--output-root", default=None)

    # Dense route targets plus a hard actual observation-step cap.  The v39
    # Kalman final-step cap is 7 m; allowing at most 8 m here leaves only a small
    # one-frame mismatch for the final MS to correct instead of 15-22 m jumps.
    parser.add_argument("--step-m", type=float, default=4.0)
    parser.add_argument("--max-sample-distance-m", type=float, default=10.0)
    parser.add_argument("--max-frame-step-m", type=float, default=8.0)
    parser.add_argument("--max-cross-step-m", type=float, default=5.0)
    parser.add_argument("--max-backward-step-m", type=float, default=1.0)

    # Keep yaw out of selection by default.  This is an offline adapter option,
    # not a localization input; 0.0 means it is completely ignored.
    parser.add_argument("--heading-weight-px-per-deg", type=float, default=0.0)

    parser.add_argument("--continuity-weight", type=float, default=2.0)
    parser.add_argument("--cross-weight", type=float, default=1.25)
    parser.add_argument("--backward-weight", type=float, default=6.0)
    parser.add_argument("--skip-penalty", type=float, default=18.0)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--min-selected-ratio", type=float, default=0.55)
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
