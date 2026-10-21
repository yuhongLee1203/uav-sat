#!/usr/bin/env python3
"""Prepare Bearing-UAV pseudo-flight routes from independent observations.

Bearing-UAV does not provide a continuous recorded flight.  We therefore build
an ordered pseudo-flight in two stages:

1. Dense route matching from real Bearing observations.
2. Per-straight-leg physical cleanup.

The cleanup is COVERAGE SAFE: smoothing is never allowed to truncate a leg or
remove later turns.  A smoothed subsequence is accepted only when it spans the
full leg.  Otherwise that leg falls back to its dense real-observation sequence.
Every retained UAV image keeps its original metric coordinate; no GT coordinate
is projected or relabelled and yaw is never used by the localization model.
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

# Keep this identifier because bearing_runner_exact_v39.py explicitly locks to
# this prepared-data family.  The implementation below is the corrected,
# full-route-safe version of the v12 two-stage selector.
SELECTION_VERSION = "soft_sequence_v12_dense_then_physical_leg_prune"
PIECEWISE_ROUTE_SPECS = dict(base.ROUTE_SPECS)


def _angle_delta_abs_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _dense_sequence_select(
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
):
    """Recover a dense route-matched sequence using only soft geometry costs."""
    xy_px = rows[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    xy_m = xy_px * float(base.MPP)
    target_m = np.asarray(targets, dtype=np.float64) * float(base.MPP)
    identities = rows["target_path"].astype(str).tolist()
    blocked = np.asarray(
        [identity in used_global for identity in identities], dtype=bool
    )
    max_sample_px = float(max_sample_distance_m) / float(base.MPP)

    heading_rad = np.deg2rad(np.asarray(headings, dtype=np.float64))
    heading_unit = np.stack(
        [np.cos(heading_rad), np.sin(heading_rad)], axis=1
    )
    target_cross_axis = np.stack(
        [-heading_unit[:, 1], heading_unit[:, 0]], axis=1
    )

    candidate_lists: List[List[Tuple[int, float]]] = []
    for target in targets:
        spatial_px = np.linalg.norm(xy_px - target[None, :], axis=1)
        valid = np.flatnonzero(
            (spatial_px <= max_sample_px) & (~blocked)
        )
        if valid.size == 0:
            candidate_lists.append([])
            continue
        err_m = spatial_px[valid] * float(base.MPP)
        order = np.argsort(err_m)[: int(candidate_limit)]
        candidate_lists.append(
            [(int(valid[j]), float(err_m[j])) for j in order]
        )

    # state = cost, last row, last target, row ids, target ids, local-used, skips
    beams = [(0.0, None, None, tuple(), tuple(), frozenset(), 0)]

    for target_index, candidates in enumerate(candidate_lists):
        expanded = []
        for state in beams:
            cost, last_idx, last_target, ids, tids, local_used, skips = state

            # Independent Bearing observations can be locally sparse.  A skip is
            # always legal; density is favored by beam ordering below.
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
                    np.dot(
                        current_offset,
                        target_cross_axis[target_index],
                    )
                )
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

                    cross_axis = np.asarray(
                        [-unit[1], unit[0]], dtype=np.float64
                    )
                    along = float(np.dot(delta, unit))
                    cross = abs(float(np.dot(delta, cross_axis)))

                    transition += (
                        float(continuity_weight) * abs(step - desired)
                    )
                    transition += float(cross_weight) * cross
                    transition += (
                        float(backward_weight) * max(0.0, -along)
                    )
                    transition += (
                        float(large_step_weight)
                        * max(
                            0.0,
                            step - float(preferred_step_m),
                        )
                        ** 2
                    )

                    planned_turn = _angle_delta_abs_deg(
                        headings[tb], headings[ta]
                    )
                    if planned_turn < float(big_turn_threshold_deg):
                        previous_offset = (
                            xy_m[int(last_idx)] - target_m[ta]
                        )
                        previous_lateral = float(
                            np.dot(
                                previous_offset,
                                target_cross_axis[ta],
                            )
                        )
                        transition += (
                            float(lateral_smooth_weight)
                            * abs(current_lateral - previous_lateral)
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
            raise RuntimeError(
                f"dense sequence search became empty at target {target_index}"
            )

        # Dense first: this reproduces the previously successful high-coverage
        # selection.  Physical cleanup is applied after the whole route exists.
        expanded.sort(key=lambda s: (-len(s[3]), s[0], s[6]))
        beams = expanded[: max(32, int(beam_width))]

    best = min(beams, key=lambda s: (-len(s[3]), s[0], s[6]))
    ids = list(best[3])
    tids = list(best[4])
    if len(ids) < 8:
        raise RuntimeError(
            f"Dense selector found only {len(ids)}/{len(targets)} observations"
        )
    return ids, tids, xy_m, target_m, target_cross_axis


def _split_selected_into_legs(
    target_ids, headings, big_turn_threshold_deg: float
):
    if not target_ids:
        return []
    groups = [[0]]
    for k in range(1, len(target_ids)):
        ta = int(target_ids[k - 1])
        tb = int(target_ids[k])
        turn = _angle_delta_abs_deg(headings[tb], headings[ta])
        if turn >= float(big_turn_threshold_deg):
            groups.append([k])
        else:
            groups[-1].append(k)
    return groups


def _longest_physical_leg(
    dense_ids,
    dense_tids,
    group_positions,
    xy_m,
    target_m,
    target_cross_axis,
    headings,
    *,
    preferred_step_m: float,
    safety_max_step_m: float,
    max_cross_track_m: float,
    max_same_leg_lateral_jump_m: float,
    max_same_leg_backward_m: float,
):
    """Longest physically plausible subsequence inside one straight leg."""
    if not group_positions:
        return []

    pos = list(group_positions)
    n = len(pos)
    valid_node = [False] * n
    lateral = [0.0] * n

    for j, seq_pos in enumerate(pos):
        rid = int(dense_ids[seq_pos])
        tid = int(dense_tids[seq_pos])
        offset = xy_m[rid] - target_m[tid]
        lat = float(np.dot(offset, target_cross_axis[tid]))
        lateral[j] = lat
        valid_node[j] = (
            abs(lat) <= float(max_cross_track_m) + 1e-9
        )

    dp_len = [0] * n
    dp_cost = [float("inf")] * n
    dp_start_tid = [0] * n
    prev = [-1] * n

    for j in range(n):
        if not valid_node[j]:
            continue
        seq_j = pos[j]
        rid_j = int(dense_ids[seq_j])
        tid_j = int(dense_tids[seq_j])
        dp_len[j] = 1
        dp_cost[j] = abs(lateral[j])
        dp_start_tid[j] = tid_j

        for i in range(j):
            if dp_len[i] <= 0:
                continue
            seq_i = pos[i]
            rid_i = int(dense_ids[seq_i])
            tid_i = int(dense_tids[seq_i])

            delta = xy_m[rid_j] - xy_m[rid_i]
            step = float(np.linalg.norm(delta))
            if step > float(safety_max_step_m) + 1e-9:
                continue

            route_delta = target_m[tid_j] - target_m[tid_i]
            route_norm = float(np.linalg.norm(route_delta))
            if route_norm <= 1e-9:
                hr = np.deg2rad(float(headings[tid_j]))
                unit = np.asarray(
                    [np.cos(hr), np.sin(hr)], dtype=np.float64
                )
                desired = float(preferred_step_m)
            else:
                unit = route_delta / route_norm
                desired = route_norm

            along = float(np.dot(delta, unit))
            if along < -float(max_same_leg_backward_m) - 1e-9:
                continue
            if (
                abs(lateral[j] - lateral[i])
                > float(max_same_leg_lateral_jump_m) + 1e-9
            ):
                continue

            new_len = dp_len[i] + 1
            new_cost = (
                dp_cost[i]
                + abs(step - desired)
                + 2.0 * abs(lateral[j] - lateral[i])
                + 0.5 * abs(lateral[j])
            )
            new_start = dp_start_tid[i]
            new_span = tid_j - new_start
            old_span = (
                tid_j - dp_start_tid[j]
                if dp_len[j] > 0
                else -1
            )

            if (
                new_len > dp_len[j]
                or (new_len == dp_len[j] and new_span > old_span)
                or (
                    new_len == dp_len[j]
                    and new_span == old_span
                    and new_cost < dp_cost[j]
                )
            ):
                dp_len[j] = new_len
                dp_cost[j] = new_cost
                dp_start_tid[j] = new_start
                prev[j] = i

    candidates = [j for j in range(n) if dp_len[j] > 0]
    if not candidates:
        return []

    end = max(
        candidates,
        key=lambda j: (
            dp_len[j],
            int(dense_tids[pos[j]]) - dp_start_tid[j],
            -dp_cost[j],
        ),
    )

    chain = []
    while end >= 0:
        chain.append(pos[end])
        end = prev[end]
    chain.reverse()
    return chain


def _physical_prune(
    dense_ids,
    dense_tids,
    xy_m,
    target_m,
    target_cross_axis,
    headings,
    *,
    preferred_step_m: float,
    safety_max_step_m: float,
    big_turn_threshold_deg: float,
    max_cross_track_m: float,
    max_same_leg_lateral_jump_m: float,
    max_same_leg_backward_m: float,
):
    """Coverage-safe leg cleanup.

    The old v12 bug accepted a smooth chain even if it represented only the
    front/middle of a leg.  That could leave many later waypoints with no frames
    and make inference stop early.  Here a chain is accepted only when it spans
    >=85% of the dense target range and reaches within 10% of both boundaries.
    Otherwise the full dense leg is retained.  Boundary samples are also restored
    around accepted chains so every turn remains connected.
    """
    groups = _split_selected_into_legs(
        dense_tids, headings, big_turn_threshold_deg
    )
    kept_positions = []
    per_leg = []

    for leg_index, group in enumerate(groups):
        if not group:
            continue

        chain = _longest_physical_leg(
            dense_ids,
            dense_tids,
            group,
            xy_m,
            target_m,
            target_cross_axis,
            headings,
            preferred_step_m=preferred_step_m,
            safety_max_step_m=safety_max_step_m,
            max_cross_track_m=max_cross_track_m,
            max_same_leg_lateral_jump_m=max_same_leg_lateral_jump_m,
            max_same_leg_backward_m=max_same_leg_backward_m,
        )

        group_start_tid = int(dense_tids[group[0]])
        group_end_tid = int(dense_tids[group[-1]])
        group_span = max(group_end_tid - group_start_tid, 1)
        coverage = 0.0
        use_dense = len(chain) < 2

        if not use_dense:
            chain_start_tid = int(dense_tids[chain[0]])
            chain_end_tid = int(dense_tids[chain[-1]])
            coverage = (
                chain_end_tid - chain_start_tid
            ) / float(group_span)
            start_fraction = (
                chain_start_tid - group_start_tid
            ) / float(group_span)
            end_fraction = (
                group_end_tid - chain_end_tid
            ) / float(group_span)
            if (
                coverage < 0.85
                or start_fraction > 0.10
                or end_fraction > 0.10
            ):
                use_dense = True

        if use_dense:
            selected = list(group)
            mode = "dense_fallback_for_coverage"
            coverage = 1.0
        else:
            first = chain[0]
            last = chain[-1]
            prefix = [p for p in group if p < first]
            suffix = [p for p in group if p > last]
            selected = sorted(set(prefix + list(chain) + suffix))
            mode = "smoothed_full_span"

        kept_positions.extend(selected)
        per_leg.append(
            {
                "leg": int(leg_index),
                "dense": int(len(group)),
                "kept": int(len(selected)),
                "coverage": float(coverage),
                "mode": mode,
                "first_target": int(dense_tids[selected[0]]),
                "last_target": int(dense_tids[selected[-1]]),
            }
        )

    kept_positions = sorted(set(kept_positions))
    return (
        [int(dense_ids[p]) for p in kept_positions],
        [int(dense_tids[p]) for p in kept_positions],
        per_leg,
    )


def _diagnostics(
    ids,
    target_ids,
    xy_m,
    target_m,
    target_cross_axis,
    headings,
    *,
    targets_count: int,
    dense_count: int,
    big_turn_threshold_deg: float,
    max_cross_track_m: float,
    max_same_leg_lateral_jump_m: float,
    max_same_leg_backward_m: float,
    per_leg,
):
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
    target_error = np.linalg.norm(
        selected_xy - selected_targets_m, axis=1
    )
    signed_lateral = np.sum(
        (selected_xy - selected_targets_m) * selected_cross_axis,
        axis=1,
    )
    abs_lateral = np.abs(signed_lateral)

    backward = 0
    same_leg_backward = 0
    cross_values = []
    same_leg_lateral_delta = []

    for k in range(1, len(ids)):
        ta = int(target_ids[k - 1])
        tb = int(target_ids[k])
        route_delta = target_m[tb] - target_m[ta]
        norm = float(np.linalg.norm(route_delta))
        if norm <= 1e-9:
            continue
        unit = route_delta / norm
        cross_axis = np.asarray(
            [-unit[1], unit[0]], dtype=np.float64
        )
        delta = selected_xy[k] - selected_xy[k - 1]
        along = float(np.dot(delta, unit))
        cross_values.append(abs(float(np.dot(delta, cross_axis))))
        backward += int(along < -1e-6)

        same_leg = (
            _angle_delta_abs_deg(headings[tb], headings[ta])
            < float(big_turn_threshold_deg)
        )
        if same_leg:
            same_leg_backward += int(along < -1e-6)
            same_leg_lateral_delta.append(
                abs(
                    float(
                        signed_lateral[k] - signed_lateral[k - 1]
                    )
                )
            )

    def pct_over(value: float) -> float:
        return (
            float(100.0 * np.mean(steps > value))
            if len(steps)
            else 0.0
        )

    return {
        "targets": int(targets_count),
        "dense_frames_before_prune": int(dense_count),
        "dense_selected_ratio": float(
            dense_count / max(targets_count, 1)
        ),
        "frames": int(len(ids)),
        "selected_ratio": float(len(ids) / max(targets_count, 1)),
        "prune_keep_ratio": float(len(ids) / max(dense_count, 1)),
        "skipped_targets": int(targets_count - len(ids)),
        "per_leg_prune": per_leg,
        "mean_target_error_m": (
            float(target_error.mean()) if len(target_error) else 0.0
        ),
        "max_target_error_m": (
            float(target_error.max()) if len(target_error) else 0.0
        ),
        "centerline_cross_mean_m": (
            float(abs_lateral.mean()) if len(abs_lateral) else 0.0
        ),
        "centerline_cross_p90_m": (
            float(np.percentile(abs_lateral, 90))
            if len(abs_lateral)
            else 0.0
        ),
        "centerline_cross_max_m": (
            float(abs_lateral.max()) if len(abs_lateral) else 0.0
        ),
        "same_leg_lateral_delta_p90_m": (
            float(np.percentile(same_leg_lateral_delta, 90))
            if same_leg_lateral_delta
            else 0.0
        ),
        "same_leg_lateral_delta_max_m": (
            float(max(same_leg_lateral_delta))
            if same_leg_lateral_delta
            else 0.0
        ),
        "actual_step_mean_m": (
            float(steps.mean()) if len(steps) else 0.0
        ),
        "actual_step_p50_m": (
            float(np.percentile(steps, 50)) if len(steps) else 0.0
        ),
        "actual_step_p90_m": (
            float(np.percentile(steps, 90)) if len(steps) else 0.0
        ),
        "actual_step_p95_m": (
            float(np.percentile(steps, 95)) if len(steps) else 0.0
        ),
        "actual_step_max_m": (
            float(steps.max()) if len(steps) else 0.0
        ),
        "step_over_7m_pct": pct_over(7.0),
        "step_over_10m_pct": pct_over(10.0),
        "step_over_14m_pct": pct_over(14.0),
        "cross_step_p90_m": (
            float(np.percentile(cross_values, 90))
            if cross_values
            else 0.0
        ),
        "backward_step_pct": float(
            100.0 * backward / max(len(ids) - 1, 1)
        ),
        "same_leg_backward_step_pct": float(
            100.0 * same_leg_backward / max(len(ids) - 1, 1)
        ),
        "max_cross_track_rule_m": float(max_cross_track_m),
        "max_same_leg_lateral_jump_rule_m": float(
            max_same_leg_lateral_jump_m
        ),
        "max_same_leg_backward_rule_m": float(
            max_same_leg_backward_m
        ),
    }


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
    max_cross_track_m: float,
    max_same_leg_lateral_jump_m: float,
    max_same_leg_backward_m: float,
):
    dense_ids, dense_tids, xy_m, target_m, target_cross_axis = (
        _dense_sequence_select(
            rows,
            targets,
            headings,
            used_global,
            max_sample_distance_m=max_sample_distance_m,
            preferred_step_m=preferred_step_m,
            safety_max_step_m=safety_max_step_m,
            candidate_limit=candidate_limit,
            beam_width=beam_width,
            skip_penalty=skip_penalty,
            continuity_weight=continuity_weight,
            large_step_weight=large_step_weight,
            cross_weight=cross_weight,
            backward_weight=backward_weight,
            point_cross_weight=point_cross_weight,
            lateral_smooth_weight=lateral_smooth_weight,
            big_turn_threshold_deg=big_turn_threshold_deg,
        )
    )

    ids, target_ids, per_leg = _physical_prune(
        dense_ids,
        dense_tids,
        xy_m,
        target_m,
        target_cross_axis,
        headings,
        preferred_step_m=preferred_step_m,
        safety_max_step_m=safety_max_step_m,
        big_turn_threshold_deg=big_turn_threshold_deg,
        max_cross_track_m=max_cross_track_m,
        max_same_leg_lateral_jump_m=max_same_leg_lateral_jump_m,
        max_same_leg_backward_m=max_same_leg_backward_m,
    )

    ratio = len(ids) / max(len(targets), 1)
    if len(ids) < 12 or ratio < float(min_selected_ratio):
        raise RuntimeError(
            "Physical prune retained too little: kept=%d/%d (%.1f%%), dense_before=%d"
            % (
                len(ids),
                len(targets),
                100.0 * ratio,
                len(dense_ids),
            )
        )

    identities = rows["target_path"].astype(str).tolist()
    for idx in ids:
        used_global.add(identities[idx])

    diag = _diagnostics(
        ids,
        target_ids,
        xy_m,
        target_m,
        target_cross_axis,
        headings,
        targets_count=len(targets),
        dense_count=len(dense_ids),
        big_turn_threshold_deg=big_turn_threshold_deg,
        max_cross_track_m=max_cross_track_m,
        max_same_leg_lateral_jump_m=max_same_leg_lateral_jump_m,
        max_same_leg_backward_m=max_same_leg_backward_m,
        per_leg=per_leg,
    )
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
        raise ValueError(
            f"Expected {base.REFERENCE_SIZE}x{base.REFERENCE_SIZE} RSI"
        )

    rows = base._city_rows(pd.read_csv(metadata_path), city)
    basename_index = base._build_basename_index(dataset_root, city)
    routes = {
        name: base._scale_route(points, width, height)
        for name, points in PIECEWISE_ROUTE_SPECS.items()
    }

    used_global, stats = set(), {}
    for name in (*base.TRAIN_ROUTES, *base.TEST_ROUTES):
        planned = routes[name]
        targets, headings = base._dense_targets(
            planned, float(args.step_m)
        )
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
            max_cross_track_m=float(args.max_cross_track_m),
            max_same_leg_lateral_jump_m=float(
                args.max_same_leg_lateral_jump_m
            ),
            max_same_leg_backward_m=float(
                args.max_same_leg_backward_m
            ),
        )

        selected = rows.iloc[ids].copy()
        paths = [
            base._resolve_image_path(
                value, dataset_root, city, basename_index
            )
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
            "split": (
                "train"
                if name in base.TRAIN_ROUTES
                else "inference"
            ),
            "planned_length_m": (
                base._route_length_px(planned) * float(base.MPP)
            ),
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
        print(
            "[SEQUENCE-V12-FULLROUTE]",
            name,
            json.dumps(stats[name], indent=2),
            flush=True,
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
            "Dense real-observation route first; coverage-safe per-leg physical "
            "cleanup second. Smoothing may never truncate a leg. Retained GT "
            "coordinates remain real and yaw is not used."
        ),
    }
    (output_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )

    base._draw_preview(
        sat_path,
        routes,
        output_root / "route_plan_full_satellite.jpg",
    )
    print(
        "[DONE] experiment:", output_root / "experiment.json", flush=True
    )
    return output_root


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        default="/yh/study/cvpr_data/Bearing_UAV_90K",
    )
    p.add_argument(
        "--city", default="cityb", choices=sorted(base.CITY_TO_RSI)
    )
    p.add_argument("--output-root", default=None)
    p.add_argument("--step-m", type=float, default=4.0)
    p.add_argument("--max-sample-distance-m", type=float, default=10.0)
    p.add_argument("--preferred-step-m", type=float, default=4.0)
    p.add_argument("--safety-max-step-m", type=float, default=22.0)
    p.add_argument("--candidate-limit", type=int, default=256)
    p.add_argument("--beam-width", type=int, default=512)
    p.add_argument("--skip-penalty", type=float, default=40.0)
    p.add_argument("--continuity-weight", type=float, default=2.5)
    p.add_argument("--large-step-weight", type=float, default=0.85)
    p.add_argument("--cross-weight", type=float, default=2.0)
    p.add_argument("--backward-weight", type=float, default=12.0)
    p.add_argument("--point-cross-weight", type=float, default=20.0)
    p.add_argument("--lateral-smooth-weight", type=float, default=35.0)
    p.add_argument("--big-turn-threshold-deg", type=float, default=25.0)
    p.add_argument("--min-selected-ratio", type=float, default=0.25)
    p.add_argument("--max-cross-track-m", type=float, default=8.5)
    p.add_argument(
        "--max-same-leg-lateral-jump-m", type=float, default=4.0
    )
    p.add_argument(
        "--max-same-leg-backward-m", type=float, default=0.25
    )
    return p


if __name__ == "__main__":
    prepare(build_parser().parse_args())
