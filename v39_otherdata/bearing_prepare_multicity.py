#!/usr/bin/env python3
"""Prepare one Bearing-UAV city for the v39 multi-city experiment.

Why this wrapper exists
-----------------------
The three hand-written training corridors are not equally dense in every city.
The previous multi-city script always forced the same ``train_01`` corridor,
which caused City A to stop after only the first part of the route and therefore
prevented City C/D from ever running.

This file probes the THREE training-only corridor candidates using only the
current city's Bearing observations.  It selects the best *full-route* candidate
and renames it to canonical ``train_01`` (Route A).  The two evaluation routes
are fixed to the official Bearing-UAV navigation waypoint files and are never
used for Route-A selection.

No GT coordinate is moved/projected.  Every retained frame keeps its real
Bearing-UAV coordinate.  If 10 m matching is too sparse we retry 12/15 m, and
record the chosen profile in experiment.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import bearing_prepare as base
import bearing_prepare_sequence_v3 as seq
from bearing_multicity_routes import (
    OFFICIAL_TEST_ROUTE_SOURCE,
    OFFICIAL_TEST_ROUTES,
    audit_test_turn_diversity,
)

SELECTION_VERSION = "soft_sequence_v13_multicity_auto_train_fullroute"
TRAINING_CANDIDATES = {
    name: [tuple(map(int, p)) for p in base.ROUTE_SPECS[name]]
    for name in ("train_01", "train_02", "train_03")
}

PROFILES = (
    dict(name="strict", max_sample_distance_m=10.0, safety_max_step_m=22.0,
         max_cross_track_m=8.5, max_same_leg_lateral_jump_m=4.0),
    dict(name="balanced", max_sample_distance_m=12.0, safety_max_step_m=24.0,
         max_cross_track_m=10.0, max_same_leg_lateral_jump_m=5.0),
    dict(name="relaxed", max_sample_distance_m=15.0, safety_max_step_m=28.0,
         max_cross_track_m=12.0, max_same_leg_lateral_jump_m=6.0),
)


def _selector_kwargs(profile: dict) -> dict:
    return dict(
        max_sample_distance_m=float(profile["max_sample_distance_m"]),
        preferred_step_m=4.0,
        safety_max_step_m=float(profile["safety_max_step_m"]),
        candidate_limit=256,
        beam_width=512,
        skip_penalty=40.0,
        continuity_weight=2.5,
        large_step_weight=0.85,
        cross_weight=2.0,
        backward_weight=12.0,
        point_cross_weight=20.0,
        lateral_smooth_weight=35.0,
        big_turn_threshold_deg=20.0,
        min_selected_ratio=0.20,
        max_cross_track_m=float(profile["max_cross_track_m"]),
        max_same_leg_lateral_jump_m=float(profile["max_same_leg_lateral_jump_m"]),
        max_same_leg_backward_m=0.25,
    )


def _coverage_from_selected(rows, ids, target_ids, planned, targets) -> dict:
    xy_m = rows.iloc[ids][["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64) * float(base.MPP)
    wp_m = np.asarray(planned, dtype=np.float64) * float(base.MPP)
    if len(xy_m) == 0:
        return {"full": False, "reason": "no selected observations"}
    waypoint_nearest = [float(np.linalg.norm(xy_m - wp[None, :], axis=1).min()) for wp in wp_m]
    start = float(np.linalg.norm(xy_m[0] - wp_m[0]))
    end = float(np.linalg.norm(xy_m[-1] - wp_m[-1]))
    first_tid = int(target_ids[0]) if target_ids else -1
    last_tid = int(target_ids[-1]) if target_ids else -1
    last_required = max(len(targets) - 1, 1)
    target_span = (last_tid - first_tid) / float(last_required)
    full = (
        len(ids) >= 40
        and first_tid <= max(3, int(0.05 * len(targets)))
        and last_tid >= int(0.95 * last_required)
        and target_span >= 0.90
        and start <= 20.0
        and end <= 20.0
        and max(waypoint_nearest) <= 20.0
    )
    return {
        "full": bool(full),
        "frames": int(len(ids)),
        "targets": int(len(targets)),
        "selected_ratio": float(len(ids) / max(len(targets), 1)),
        "first_target": first_tid,
        "last_target": last_tid,
        "target_span_ratio": float(target_span),
        "start_m": start,
        "end_m": end,
        "worst_waypoint_m": float(max(waypoint_nearest)),
    }


def _probe_training_candidates(dataset_root: Path, city: str, profile: dict):
    metadata = base._find_metadata(dataset_root)
    rows = base._city_rows(pd.read_csv(metadata), city)
    reports = {}
    valid = []
    kwargs = _selector_kwargs(profile)
    for name, points in TRAINING_CANDIDATES.items():
        planned = base._scale_route(points, base.REFERENCE_SIZE, base.REFERENCE_SIZE)
        targets, headings = base._dense_targets(planned, 4.0)
        try:
            ids, tids, diag = seq._sequence_select(
                rows, targets, headings, set(), **kwargs
            )
            coverage = _coverage_from_selected(rows, ids, tids, planned, targets)
            report = {
                "candidate": name,
                "status": "ok" if coverage["full"] else "incomplete",
                **coverage,
                "centerline_cross_p90_m": float(diag["centerline_cross_p90_m"]),
                "same_leg_lateral_delta_p90_m": float(diag["same_leg_lateral_delta_p90_m"]),
                "backward_step_pct": float(diag["backward_step_pct"]),
                "actual_step_p90_m": float(diag["actual_step_p90_m"]),
            }
            reports[name] = report
            if coverage["full"]:
                # Coverage/density dominates; geometry breaks ties.
                score = (
                    float(coverage["selected_ratio"]),
                    -float(diag["centerline_cross_p90_m"]),
                    -float(diag["backward_step_pct"]),
                )
                valid.append((score, name))
        except Exception as exc:
            reports[name] = {
                "candidate": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if not valid:
        return None, reports
    valid.sort(reverse=True)
    return valid[0][1], reports


def _audit_generated(root: Path) -> dict:
    results = {}
    for route in ("train_01", "test_01", "test_02"):
        manifest_path = root / "routes" / route / "manifest.csv"
        waypoint_path = root / "routes" / route / "waypoints.json"
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        payload = json.loads(waypoint_path.read_text(encoding="utf-8"))
        wps = [
            (float(w["longitude"]), float(w["latitude"]))
            for w in sorted(payload["waypoints"], key=lambda x: int(x["waypoint_order"]))
        ]
        pts = [(float(r["x_m"]), float(r["y_m"])) for r in rows]
        if len(rows) < (40 if route == "train_01" else 30):
            raise RuntimeError(f"{route}: too few frames ({len(rows)})")
        nearest = [min(math.hypot(x-wx, y-wy) for x, y in pts) for wx, wy in wps]
        start = math.hypot(pts[0][0]-wps[0][0], pts[0][1]-wps[0][1])
        end = math.hypot(pts[-1][0]-wps[-1][0], pts[-1][1]-wps[-1][1])
        if start > 22.0 or end > 22.0 or max(nearest) > 22.0:
            raise RuntimeError(
                f"{route}: incomplete route start={start:.2f} end={end:.2f} "
                f"worst_waypoint={max(nearest):.2f}m"
            )
        results[route] = {
            "frames": len(rows),
            "waypoints": len(wps),
            "start_m": start,
            "end_m": end,
            "worst_waypoint_m": max(nearest),
        }
    return results


def prepare(args):
    dataset_root = Path(args.dataset_root).resolve()
    city = str(args.city).lower()
    final_root = Path(args.output_root).resolve()
    if city not in base.CITY_TO_RSI:
        raise ValueError(f"Unsupported city: {city}")

    official_specs = {
        "test_01": [tuple(map(int, p)) for p in OFFICIAL_TEST_ROUTES[city]["test_01"]],
        "test_02": [tuple(map(int, p)) for p in OFFICIAL_TEST_ROUTES[city]["test_02"]],
    }
    turn_specs = {
        "train_01": TRAINING_CANDIDATES["train_01"],
        "train_02": TRAINING_CANDIDATES["train_02"],
        "train_03": TRAINING_CANDIDATES["train_03"],
        **official_specs,
    }
    turn_report = audit_test_turn_diversity(city, turn_specs)
    print("[TURN-DIVERSITY-AUDIT] PASS", flush=True)
    print(json.dumps(turn_report, indent=2), flush=True)

    all_probe_reports = {}
    errors = []
    for profile in PROFILES:
        chosen, probe = _probe_training_candidates(dataset_root, city, profile)
        all_probe_reports[profile["name"]] = probe
        print(f"[TRAIN-ROUTE-PROBE] city={city} profile={profile['name']}", flush=True)
        print(json.dumps(probe, indent=2), flush=True)
        if chosen is None:
            errors.append(f"{profile['name']}: no complete training candidate")
            continue

        tmp_root = final_root.with_name(final_root.name + "__building")
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        if final_root.exists():
            shutil.rmtree(final_root)

        # Canonical v39 uses ONE Route A.  Do not make unused training corridors
        # consume observations or block the experiment.
        base.TRAIN_ROUTES = ("train_01",)
        base.TEST_ROUTES = ("test_01", "test_02")
        seq.base.TRAIN_ROUTES = base.TRAIN_ROUTES
        seq.base.TEST_ROUTES = base.TEST_ROUTES
        seq.SELECTION_VERSION = SELECTION_VERSION
        seq.PIECEWISE_ROUTE_SPECS = {
            "train_01": TRAINING_CANDIDATES[chosen],
            "test_01": official_specs["test_01"],
            "test_02": official_specs["test_02"],
        }

        kwargs = _selector_kwargs(profile)
        prep_args = argparse.Namespace(
            dataset_root=str(dataset_root),
            city=city,
            output_root=str(tmp_root),
            step_m=4.0,
            preferred_step_m=4.0,
            safety_max_step_m=kwargs["safety_max_step_m"],
            max_sample_distance_m=kwargs["max_sample_distance_m"],
            candidate_limit=kwargs["candidate_limit"],
            beam_width=kwargs["beam_width"],
            skip_penalty=kwargs["skip_penalty"],
            continuity_weight=kwargs["continuity_weight"],
            large_step_weight=kwargs["large_step_weight"],
            cross_weight=kwargs["cross_weight"],
            backward_weight=kwargs["backward_weight"],
            point_cross_weight=kwargs["point_cross_weight"],
            lateral_smooth_weight=kwargs["lateral_smooth_weight"],
            big_turn_threshold_deg=kwargs["big_turn_threshold_deg"],
            min_selected_ratio=kwargs["min_selected_ratio"],
            max_cross_track_m=kwargs["max_cross_track_m"],
            max_same_leg_lateral_jump_m=kwargs["max_same_leg_lateral_jump_m"],
            max_same_leg_backward_m=kwargs["max_same_leg_backward_m"],
        )
        try:
            seq.prepare(prep_args)
            coverage = _audit_generated(tmp_root)
            exp_path = tmp_root / "experiment.json"
            exp = json.loads(exp_path.read_text(encoding="utf-8"))
            exp["sequence_selection_version"] = SELECTION_VERSION
            exp["canonical_train_routes"] = ["train_01"]
            exp["route_a_candidate_source"] = chosen
            exp["training_candidate_probes"] = all_probe_reports
            exp["preparation_profile"] = profile
            exp["test_route_source"] = dict(OFFICIAL_TEST_ROUTE_SOURCE[city])
            exp["turn_diversity_audit"] = turn_report
            exp["full_route_coverage_audit"] = coverage
            exp_path.write_text(json.dumps(exp, indent=2), encoding="utf-8")
            (tmp_root / "turn_diversity_audit.json").write_text(
                json.dumps(turn_report, indent=2), encoding="utf-8"
            )
            (tmp_root / "training_route_selection.json").write_text(
                json.dumps({
                    "city": city,
                    "selected_source": chosen,
                    "renamed_as": "train_01",
                    "profile": profile,
                    "candidate_probes": all_probe_reports,
                    "coverage": coverage,
                }, indent=2), encoding="utf-8"
            )
            tmp_root.rename(final_root)
            print(
                f"[MULTICITY-PREP] PASS city={city} Route-A source={chosen} "
                f"profile={profile['name']}", flush=True
            )
            print(json.dumps(coverage, indent=2), flush=True)
            return final_root
        except Exception as exc:
            errors.append(f"{profile['name']}/{chosen}: {type(exc).__name__}: {exc}")
            print(f"[MULTICITY-PREP] retry after failure: {errors[-1]}", flush=True)
            if tmp_root.exists():
                shutil.rmtree(tmp_root)

    raise RuntimeError(
        f"Unable to prepare complete {city} experiment. Attempts:\n- "
        + "\n- ".join(errors)
    )


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=sorted(base.CITY_TO_RSI))
    p.add_argument("--output-root", required=True)
    return p


if __name__ == "__main__":
    prepare(build_parser().parse_args())
