#!/usr/bin/env python3
"""Evaluate Kalman profiles on Route-A validation using FINAL paper outputs only.

No held-out test_01/test_02 route is read.  The architecture/checkpoints remain
unchanged.  The final GT-derived MeanShift reference prior is disabled so the
Kalman contribution is not hidden by an evaluation-time reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

import bearing_iclr_ablation as ab
import bearing_paper_metrics as pm


# Only Kalman-side inference parameters are searched.  GRU weights, visual
# weights, candidate geometry, MeanShift architecture and table metrics stay
# unchanged.  q_scale is relative to the existing city train-validation profile.
PROFILES = [
    {"name": "current", "q_scale": 1.00},
    {"name": "r6_q050", "fixed_variance_m2": 6.0, "q_scale": 0.50},
    {"name": "r9_q050", "fixed_variance_m2": 9.0, "q_scale": 0.50},
    {"name": "r9_q075", "fixed_variance_m2": 9.0, "q_scale": 0.75},
    {"name": "r12_q050", "fixed_variance_m2": 12.0, "q_scale": 0.50},
    {"name": "r12_q075", "fixed_variance_m2": 12.0, "q_scale": 0.75},
    {"name": "r16_q050", "fixed_variance_m2": 16.0, "q_scale": 0.50},
    {"name": "r16_q075", "fixed_variance_m2": 16.0, "q_scale": 0.75},
    {"name": "r16_q100", "fixed_variance_m2": 16.0, "q_scale": 1.00},
    {"name": "r25_q050", "fixed_variance_m2": 25.0, "q_scale": 0.50},
    {"name": "r25_q075", "fixed_variance_m2": 25.0, "q_scale": 0.75},
    {"name": "r36_q050", "fixed_variance_m2": 36.0, "q_scale": 0.50},
    {"name": "r16_q050_conf125", "fixed_variance_m2": 16.0, "q_scale": 0.50, "confidence_power": 1.25},
    {"name": "r16_q050_conf150", "fixed_variance_m2": 16.0, "q_scale": 0.50, "confidence_power": 1.50},
    {"name": "r25_q050_conf125", "fixed_variance_m2": 25.0, "q_scale": 0.50, "confidence_power": 1.25},
    {"name": "r16_q050_tightstep", "fixed_variance_m2": 16.0, "q_scale": 0.50,
     "step_relax_confidence": 0.55, "step_relax_width": 0.08, "step_visual_slack_m": 4.0},
    {"name": "r25_q050_tightstep", "fixed_variance_m2": 25.0, "q_scale": 0.50,
     "step_relax_confidence": 0.55, "step_relax_width": 0.08, "step_visual_slack_m": 4.0},
    {"name": "r16_q025", "fixed_variance_m2": 16.0, "q_scale": 0.25},
]


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_csv(summary: dict, output_dir: Path) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.is_file():
        return p
    q = output_dir / p.name
    if q.is_file():
        return q
    matches = sorted(output_dir.glob("route_A*_frames.csv"))
    if len(matches) == 1:
        return matches[0]
    matches = sorted(output_dir.glob("*_frames.csv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve Route-A frame CSV under {output_dir}: {matches}")


def _base_values(config):
    names = [
        "EXPERIMENT_FIXED_VARIANCE_M2",
        "KALMAN_Q_PROGRESS", "KALMAN_Q_CROSS", "KALMAN_Q_VELOCITY",
        "KALMAN_CONFIDENCE_POWER",
        "KALMAN_STEP_RELAX_CONFIDENCE", "KALMAN_STEP_RELAX_WIDTH",
        "KALMAN_STEP_VISUAL_SLACK_M",
    ]
    values = {}
    for name in names:
        if hasattr(config, name):
            values[name] = float(getattr(config, name))
    return values


def _restore(config, base):
    for key, value in base.items():
        setattr(config, key, float(value))


def _apply_profile(config, base, profile):
    _restore(config, base)
    config.EXPERIMENT_KALMAN = "fixed"
    q_scale = float(profile.get("q_scale", 1.0))
    for key in ("KALMAN_Q_PROGRESS", "KALMAN_Q_CROSS", "KALMAN_Q_VELOCITY"):
        if key in base:
            setattr(config, key, float(base[key]) * q_scale)
    mapping = {
        "fixed_variance_m2": "EXPERIMENT_FIXED_VARIANCE_M2",
        "confidence_power": "KALMAN_CONFIDENCE_POWER",
        "step_relax_confidence": "KALMAN_STEP_RELAX_CONFIDENCE",
        "step_relax_width": "KALMAN_STEP_RELAX_WIDTH",
        "step_visual_slack_m": "KALMAN_STEP_VISUAL_SLACK_M",
    }
    for src, dst in mapping.items():
        if src in profile:
            if not hasattr(config, dst):
                raise RuntimeError(f"runtime missing calibration field {dst}")
            setattr(config, dst, float(profile[src]))


def _metrics(rows, manifest_rows, city_rows, origin_x_m, origin_y_m):
    errors = np.asarray([
        math.hypot(float(r["final_x"]) - float(r["gt_x"]),
                   float(r["final_y"]) - float(r["gt_y"]))
        for r in rows
    ], dtype=np.float64)
    headings = np.asarray([abs(float(r["heading_error_deg"])) for r in rows], dtype=np.float64)
    r1 = pm._same_quadrant_recall(rows, manifest_rows, city_rows, origin_x_m, origin_y_m)
    return {
        "frames": int(len(rows)),
        "R@1*_pct": float(r1),
        "LSR@15_pct": 100.0 * float(np.mean(errors <= 15.0)),
        "HSR@15_pct": 100.0 * float(np.mean(headings <= 15.0)),
        "MLE_m": float(errors.mean()),
        "MHE_deg": float(headings.mean()),
    }


def _margins(full, no_k):
    # Positive means Full is better for every entry.
    return {
        "R@1*_pct": float(full["R@1*_pct"] - no_k["R@1*_pct"]),
        "LSR@15_pct": float(full["LSR@15_pct"] - no_k["LSR@15_pct"]),
        "HSR@15_pct": float(full["HSR@15_pct"] - no_k["HSR@15_pct"]),
        "MLE_m": float(no_k["MLE_m"] - full["MLE_m"]),
        "MHE_deg": float(no_k["MHE_deg"] - full["MHE_deg"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2033)
    cli = p.parse_args()

    # Same final decoder protocol used for the corrected held-out pair.
    os.environ["MS_REFERENCE_PRIOR_WEIGHT"] = "0.0"
    os.environ["MS_KF_PRIOR_WEIGHT"] = "1.50"
    os.environ["MS_REFERENCE_SIGMA_M"] = "4.0"
    os.environ["MS_KF_SIGMA_M"] = "4.0"
    os.environ["MS_MEASURE_LATENCY"] = "0"

    args = ab.build_parser().parse_args([
        "eval", "--suite-root", cli.suite_root, "--dataset-root", cli.dataset_root,
        "--city", cli.city, "--variant", "full", "--train-frames", "3",
        "--gpu", str(cli.gpu), "--seed", str(cli.seed),
    ])
    prepared = ab._prepared_root(args)
    ab._lock_prepared(args, prepared)
    train_root = ab._train_root(args, 3)
    out_root = train_root / "final_output_kalman_trainval"
    runtime = ab._make_runtime(prepared, out_root / "runtime")
    variant = dict(ab.VARIANTS["full"])
    ab._set_environment(args, prepared, out_root, variant, training=False)
    config, tracker, visual_localizer = ab.base._load_runtime_modules(runtime)
    ab._patch_paths(config, args, prepared)
    ab._link_full_checkpoints(config, train_root)
    ab._audit_runtime(config, runtime, variant, training=False)

    device = tracker.resolve_device()
    visual = visual_localizer.FrozenVisualLocalizer(device)
    model = tracker.load_temporal_model(device)
    cache = tracker.build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route = tracker.WaypointRoute(
        tracker.load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon)
    )
    val_start, val_end = [int(x) for x in tracker.split_ranges(len(cache))["val"]]

    exp = json.loads((prepared / "experiment.json").read_text(encoding="utf-8"))
    city_rows = pm.bearing._city_rows(pm.pd.read_csv(exp["metadata_csv"]), cli.city)
    manifest_all = _read_csv(prepared / "routes" / "train_01" / "manifest.csv")
    origin_x_m, origin_y_m = float(manifest_all[0]["x_m"]), float(manifest_all[0]["y_m"])
    manifest_val = manifest_all[val_start:val_end]
    base = _base_values(config)

    # One measured no-Kalman Route-A validation baseline.  Searched parameters
    # below are Kalman-only and therefore cannot change this baseline.
    _restore(config, base)
    config.EXPERIMENT_KALMAN = "none"
    no_k_out = out_root / "no_kalman"
    no_k_out.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = no_k_out
    no_k_summary = tracker.run_route_inference("route_A", visual, model, cache, route, device)
    no_k_rows = _read_csv(_resolve_csv(no_k_summary, no_k_out))[val_start:val_end]
    no_k = _metrics(no_k_rows, manifest_val, city_rows, origin_x_m, origin_y_m)
    print("[TRAINVAL NO-KALMAN]", cli.city, json.dumps(no_k, sort_keys=True), flush=True)

    results = []
    for profile in PROFILES:
        _apply_profile(config, base, profile)
        out = out_root / profile["name"]
        out.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR = out
        print("[TRAINVAL FULL]", cli.city, profile["name"], json.dumps(profile, sort_keys=True), flush=True)
        summary = tracker.run_route_inference("route_A", visual, model, cache, route, device)
        rows = _read_csv(_resolve_csv(summary, out))[val_start:val_end]
        full = _metrics(rows, manifest_val, city_rows, origin_x_m, origin_y_m)
        margins = _margins(full, no_k)
        row = {
            "profile": profile,
            "full": full,
            "no_kalman": no_k,
            "margins_full_better": margins,
            "strict_all_five": bool(all(v > 0.0 for v in margins.values())),
            "positive_metric_count": int(sum(v > 0.0 for v in margins.values())),
        }
        results.append(row)
        print("[TRAINVAL RESULT]", cli.city, profile["name"], json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "selection_source": "Route-A validation only",
        "held_out_navigation_read": False,
        "city": cli.city,
        "validation_range": [val_start, val_end],
        "architecture_changed": False,
        "metric_schema_changed": False,
        "final_reference_prior_weight": 0.0,
        "searched_parameters": "Kalman inference parameters only",
        "no_kalman": no_k,
        "profiles": results,
    }
    out_json = train_root / "final_output_kalman_trainval.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[TRAINVAL JSON]", out_json, flush=True)


if __name__ == "__main__":
    main()
