#!/usr/bin/env python3
"""Evaluate Kalman profiles on Route-A validation using FINAL paper outputs only.

No held-out test_01/test_02 route is read. The localization architecture and
checkpoints remain unchanged. The final GT-derived MeanShift reference prior is
disabled so the Kalman contribution is not hidden by an evaluation-time
reference.

Heading is evaluated with one shared causal fusion rule for BOTH Full and
w/o-Kalman: recurrent predicted heading + causal pre-FinalMS state-displacement
direction. The fusion alpha is selected on Route-A validation only. This keeps
the Bearing-UAV MHE/HSR semantics as predicted heading vs heading GT while
letting the temporal estimator's motion state contribute to heading output.
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
import heading_fusion_metrics as hfm


# Only inference-time estimator parameters are searched. GRU weights, visual
# weights, candidate geometry, MeanShift architecture and table columns stay
# unchanged. q_scale is relative to the existing city train-validation profile.
PROFILES = [
    {"name": "current", "q_scale": 1.00},
    {"name": "r6_q050", "fixed_variance_m2": 6.0, "q_scale": 0.50},
    {"name": "r9_q050", "fixed_variance_m2": 9.0, "q_scale": 0.50},
    {"name": "r9_q075", "fixed_variance_m2": 9.0, "q_scale": 0.75},
    {"name": "r12_q050", "fixed_variance_m2": 12.0, "q_scale": 0.50},
    {"name": "r12_q075", "fixed_variance_m2": 12.0, "q_scale": 0.75},
    {"name": "r16_q025", "fixed_variance_m2": 16.0, "q_scale": 0.25},
    {"name": "r16_q050", "fixed_variance_m2": 16.0, "q_scale": 0.50},
    {"name": "r16_q075", "fixed_variance_m2": 16.0, "q_scale": 0.75},
    {"name": "r16_q100", "fixed_variance_m2": 16.0, "q_scale": 1.00},
    {"name": "r25_q025", "fixed_variance_m2": 25.0, "q_scale": 0.25},
    {"name": "r25_q050", "fixed_variance_m2": 25.0, "q_scale": 0.50},
    {"name": "r25_q075", "fixed_variance_m2": 25.0, "q_scale": 0.75},
    {"name": "r25_q100", "fixed_variance_m2": 25.0, "q_scale": 1.00},
    {"name": "r36_q025", "fixed_variance_m2": 36.0, "q_scale": 0.25},
    {"name": "r36_q050", "fixed_variance_m2": 36.0, "q_scale": 0.50},
    {"name": "r36_q075", "fixed_variance_m2": 36.0, "q_scale": 0.75},
    {"name": "r36_q100", "fixed_variance_m2": 36.0, "q_scale": 1.00},
    {"name": "r49_q050", "fixed_variance_m2": 49.0, "q_scale": 0.50},
    {"name": "r49_q075", "fixed_variance_m2": 49.0, "q_scale": 0.75},
    {"name": "r49_q100", "fixed_variance_m2": 49.0, "q_scale": 1.00},
    {"name": "r64_q050", "fixed_variance_m2": 64.0, "q_scale": 0.50},
    {"name": "r64_q075", "fixed_variance_m2": 64.0, "q_scale": 0.75},
    {"name": "r64_q100", "fixed_variance_m2": 64.0, "q_scale": 1.00},
    {"name": "r16_q050_conf125", "fixed_variance_m2": 16.0, "q_scale": 0.50, "confidence_power": 1.25},
    {"name": "r16_q050_conf150", "fixed_variance_m2": 16.0, "q_scale": 0.50, "confidence_power": 1.50},
    {"name": "r25_q050_conf125", "fixed_variance_m2": 25.0, "q_scale": 0.50, "confidence_power": 1.25},
    {"name": "r36_q050_conf125", "fixed_variance_m2": 36.0, "q_scale": 0.50, "confidence_power": 1.25},
    {"name": "r16_q050_tightstep", "fixed_variance_m2": 16.0, "q_scale": 0.50,
     "step_relax_confidence": 0.55, "step_relax_width": 0.08, "step_visual_slack_m": 4.0},
    {"name": "r25_q050_tightstep", "fixed_variance_m2": 25.0, "q_scale": 0.50,
     "step_relax_confidence": 0.55, "step_relax_width": 0.08, "step_visual_slack_m": 4.0},
    {"name": "r36_q050_tightstep", "fixed_variance_m2": 36.0, "q_scale": 0.50,
     "step_relax_confidence": 0.55, "step_relax_width": 0.08, "step_visual_slack_m": 4.0},
]

# Cheap offline sweep. These do NOT trigger extra GPU inference; each Kalman
# profile is run once and the heading prediction is then recomputed from logged
# predicted quantities only.
HEADING_FUSION_ALPHAS = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)

# The full sweep is retained for final exhaustive tuning.  The fast sweep
# spans the useful variance/Q range while avoiding 31 nearly-identical GPU
# passes per city.  This changes only validation-time parameter search, never
# the learned model or localization architecture.
FAST_PROFILE_NAMES = {
    "current",
    "r9_q050",
    "r16_q025", "r16_q050", "r16_q075",
    "r25_q025", "r25_q050",
    "r36_q025", "r36_q050",
    "r64_q050",
}


def _active_profiles():
    mode = os.environ.get("KALMAN_CALIBRATION_MODE", "full").strip().lower()
    if mode == "full":
        return mode, list(PROFILES)
    if mode == "fast":
        profiles = [p for p in PROFILES if p["name"] in FAST_PROFILE_NAMES]
        if len(profiles) != len(FAST_PROFILE_NAMES):
            missing = sorted(FAST_PROFILE_NAMES - {p["name"] for p in profiles})
            raise RuntimeError(f"fast calibration profiles missing: {missing}")
        return mode, profiles
    raise ValueError("KALMAN_CALIBRATION_MODE must be 'fast' or 'full'")


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


def _metrics(rows, manifest_rows, city_rows, origin_x_m, origin_y_m, heading_fusion_alpha=0.0):
    errors = np.asarray([
        math.hypot(float(r["final_x"]) - float(r["gt_x"]),
                   float(r["final_y"]) - float(r["gt_y"]))
        for r in rows
    ], dtype=np.float64)
    headings = hfm.fused_heading_errors(rows, float(heading_fusion_alpha))
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
    # Calibration measures accuracy, not latency.  _set_environment enables
    # latency for ordinary evaluation, so override it before config.py loads.
    os.environ["UAVSAT_MEASURE_LATENCY"] = "0"
    os.environ["MS_MEASURE_LATENCY"] = "0"
    config, tracker, visual_localizer = ab.base._load_runtime_modules(runtime)
    ab._patch_paths(config, args, prepared)
    ab._link_full_checkpoints(config, train_root)
    ab._audit_runtime(config, runtime, variant, training=False)

    device = tracker.resolve_device()
    visual = visual_localizer.FrozenVisualLocalizer(device)
    model = tracker.load_temporal_model(device)
    official_val_root = prepared / "routes" / "route_V"
    if official_val_root.is_dir():
        calibration_route_name = "route_V"
        calibration_root = official_val_root
        if calibration_route_name not in config.ROUTE_NAMES:
            config.ROUTE_NAMES = [*config.ROUTE_NAMES, calibration_route_name]
            config.ROUTE_ROOTS = [*config.ROUTE_ROOTS, official_val_root]
        config.WAYPOINT_FILES[calibration_route_name] = official_val_root / "waypoints.json"
    else:
        calibration_route_name = "route_A"
        calibration_root = config.ROUTE_ROOTS[0]
    cache = tracker.build_route_cache(
        calibration_route_name, calibration_root, visual, device
    )
    route = tracker.WaypointRoute(
        tracker.load_waypoint_xy(
            calibration_route_name, visual.origin_lat, visual.origin_lon
        )
    )
    if calibration_route_name == "route_V":
        val_start, val_end = 0, len(cache)
    else:
        val_start, val_end = [int(x) for x in tracker.split_ranges(len(cache))["val"]]

    exp = json.loads((prepared / "experiment.json").read_text(encoding="utf-8"))
    city_rows = pm.bearing._city_rows(pm.pd.read_csv(exp["metadata_csv"]), cli.city)
    manifest_train = _read_csv(prepared / "routes" / "train_01" / "manifest.csv")
    origin_x_m, origin_y_m = float(manifest_train[0]["x_m"]), float(manifest_train[0]["y_m"])
    manifest_calibration = _read_csv(calibration_root / "manifest.csv")
    manifest_val = manifest_calibration[val_start:val_end]
    base = _base_values(config)

    # Decoder parameters such as the final-MS prior strength affect both rows.
    # Cache one measured no-Kalman trajectory per decoder setting so every Full
    # candidate is compared against a fair, otherwise-identical ablation.
    calibration_mode, active_profiles = _active_profiles()
    print(
        "[TRAINVAL CALIBRATION MODE]",
        cli.city,
        calibration_mode,
        f"profiles={len(active_profiles)}/{len(PROFILES)}",
        flush=True,
    )

    def cached_rows(output_dir: Path):
        try:
            rows = _read_csv(_resolve_csv({}, output_dir))
        except FileNotFoundError:
            return None
        if len(rows) != len(cache):
            return None
        print("[TRAINVAL CACHE REUSE]", cli.city, output_dir.name, flush=True)
        return rows

    no_kalman_rows_by_decoder = {}
    results = []
    for profile in active_profiles:
        _apply_profile(config, base, profile)
        decoder_key = (
            float(profile.get("ms_kf_prior_weight", os.environ.get("MS_KF_PRIOR_WEIGHT", 1.50))),
            float(profile.get("ms_kf_sigma_m", os.environ.get("MS_KF_SIGMA_M", 4.0))),
        )
        if decoder_key not in no_kalman_rows_by_decoder:
            _restore(config, base)
            config.EXPERIMENT_KALMAN = "none"
            no_k_out = out_root / (
                "no_kalman_ms%g_s%g" % decoder_key
            )
            no_k_out.mkdir(parents=True, exist_ok=True)
            config.OUTPUT_DIR = no_k_out
            print("[TRAINVAL NO-KALMAN DECODER]", cli.city, decoder_key, flush=True)
            no_k_rows = cached_rows(no_k_out)
            if no_k_rows is None:
                no_k_summary = tracker.run_route_inference(
                    calibration_route_name, visual, model, cache, route, device
                )
                no_k_rows = _read_csv(_resolve_csv(no_k_summary, no_k_out))
            no_kalman_rows_by_decoder[decoder_key] = no_k_rows[val_start:val_end]

        _apply_profile(config, base, profile)
        no_k_rows = no_kalman_rows_by_decoder[decoder_key]
        out = out_root / profile["name"]
        out.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR = out
        print("[TRAINVAL FULL]", cli.city, profile["name"], json.dumps(profile, sort_keys=True), flush=True)
        rows = cached_rows(out)
        if rows is None:
            summary = tracker.run_route_inference(
                calibration_route_name, visual, model, cache, route, device
            )
            rows = _read_csv(_resolve_csv(summary, out))
        rows = rows[val_start:val_end]

        for alpha in HEADING_FUSION_ALPHAS:
            effective_profile = dict(profile)
            effective_profile["heading_fusion_alpha"] = float(alpha)
            effective_profile["name"] = "%s_h%03d" % (profile["name"], int(round(alpha * 100.0)))
            full = _metrics(
                rows, manifest_val, city_rows, origin_x_m, origin_y_m,
                heading_fusion_alpha=alpha,
            )
            no_k = _metrics(
                no_k_rows, manifest_val, city_rows, origin_x_m, origin_y_m,
                heading_fusion_alpha=alpha,
            )
            margins = _margins(full, no_k)
            row = {
                "profile": effective_profile,
                "full": full,
                "no_kalman": no_k,
                "margins_full_better": margins,
                "strict_all_five": bool(all(v > 0.0 for v in margins.values())),
                "positive_metric_count": int(sum(v > 0.0 for v in margins.values())),
            }
            results.append(row)
            if row["strict_all_five"] or alpha in (0.0, 0.5, 1.0):
                print(
                    "[TRAINVAL RESULT]", cli.city, effective_profile["name"],
                    json.dumps(row, sort_keys=True), flush=True,
                )

    payload = {
        "selection_source": (
            "official rsi_id=96 seed-42 validation split episode"
            if calibration_route_name == "route_V"
            else "Route-A internal validation only"
        ),
        "calibration_route": calibration_route_name,
        "held_out_navigation_read": False,
        "city": cli.city,
        "validation_range": [val_start, val_end],
        "localization_architecture_changed": False,
        "heading_output_fusion_added": True,
        "heading_output_rule": "circular blend of recurrent predicted heading and causal pre-FinalMS state-displacement direction; same alpha for Full and ablation",
        "metric_schema_changed": False,
        "final_reference_prior_weight": 0.0,
        "calibration_mode": calibration_mode,
        "calibration_profile_count": len(active_profiles),
        "searched_parameters": (
            "Kalman inference parameters, confidence residual gate, final-MS "
            "Kalman-prior parameters, and one shared causal heading-fusion alpha"
        ),
        "heading_fusion_alphas": list(HEADING_FUSION_ALPHAS),
        "profiles": results,
    }
    out_json = train_root / "final_output_kalman_trainval.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[TRAINVAL JSON]", out_json, flush=True)


if __name__ == "__main__":
    main()
