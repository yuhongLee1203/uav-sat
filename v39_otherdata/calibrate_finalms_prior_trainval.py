#!/usr/bin/env python3
"""Calibrate final MeanShift prior weights on Route-A validation only.

This script never reads test_01/test_02. It reuses the already-trained shared
3-frame checkpoints, runs Route A, and selects final-MS prior weights by
validation MLE (+ a small P90 tie-breaker). No retraining is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import bearing_iclr_ablation as ab


PROFILES = [
    {"name": "baseline", "ms_kf_prior_weight": 1.50, "ms_reference_prior_weight": 2.50,
     "ms_kf_sigma_m": 4.0, "ms_reference_sigma_m": 4.0},
    {"name": "balanced", "ms_kf_prior_weight": 2.00, "ms_reference_prior_weight": 2.00,
     "ms_kf_sigma_m": 4.0, "ms_reference_sigma_m": 4.0},
    {"name": "kf2p5_ref1p5", "ms_kf_prior_weight": 2.50, "ms_reference_prior_weight": 1.50,
     "ms_kf_sigma_m": 4.0, "ms_reference_sigma_m": 4.0},
    {"name": "kf3_ref1", "ms_kf_prior_weight": 3.00, "ms_reference_prior_weight": 1.00,
     "ms_kf_sigma_m": 4.0, "ms_reference_sigma_m": 4.0},
    {"name": "kf3p5_ref0p75", "ms_kf_prior_weight": 3.50, "ms_reference_prior_weight": 0.75,
     "ms_kf_sigma_m": 4.0, "ms_reference_sigma_m": 4.0},
    {"name": "kf4_ref0p5", "ms_kf_prior_weight": 4.00, "ms_reference_prior_weight": 0.50,
     "ms_kf_sigma_m": 4.0, "ms_reference_sigma_m": 4.0},
    {"name": "kf3_ref1_sharp", "ms_kf_prior_weight": 3.00, "ms_reference_prior_weight": 1.00,
     "ms_kf_sigma_m": 3.5, "ms_reference_sigma_m": 5.0},
    {"name": "kf4_ref0p5_sharp", "ms_kf_prior_weight": 4.00, "ms_reference_prior_weight": 0.50,
     "ms_kf_sigma_m": 3.0, "ms_reference_sigma_m": 5.0},
]


def _eval_args(cli: argparse.Namespace):
    argv = [
        "eval",
        "--suite-root", cli.suite_root,
        "--dataset-root", cli.dataset_root,
        "--city", cli.city,
        "--variant", "full",
        "--train-frames", "3",
        "--gpu", str(cli.gpu),
        "--seed", str(cli.seed),
    ]
    return ab.build_parser().parse_args(argv)


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
    raise FileNotFoundError(f"cannot resolve Route-A frame CSV under {output_dir}")


def _set_profile(profile: dict) -> None:
    os.environ["MS_KF_PRIOR_WEIGHT"] = str(profile["ms_kf_prior_weight"])
    os.environ["MS_REFERENCE_PRIOR_WEIGHT"] = str(profile["ms_reference_prior_weight"])
    os.environ["MS_KF_SIGMA_M"] = str(profile["ms_kf_sigma_m"])
    os.environ["MS_REFERENCE_SIGMA_M"] = str(profile["ms_reference_sigma_m"])
    os.environ["MS_MEASURE_LATENCY"] = "0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2033)
    cli = p.parse_args()

    args = _eval_args(cli)
    prepared = ab._prepared_root(args)
    ab._lock_prepared(args, prepared)
    train_root = ab._train_root(args, 3)
    calibration_root = train_root / "finalms_trainval_calibration"
    runtime = ab._make_runtime(prepared, calibration_root / "runtime")
    variant = dict(ab.VARIANTS["full"])
    ab._set_environment(args, prepared, calibration_root, variant, training=False)
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

    results = []
    for profile in PROFILES:
        _set_profile(profile)
        out = calibration_root / profile["name"]
        out.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR = out
        print(
            "[FINALMS-TRAINVAL] city=%s profile=%s kf=%.2f ref=%.2f kf_sigma=%.2f ref_sigma=%.2f"
            % (
                cli.city, profile["name"], profile["ms_kf_prior_weight"],
                profile["ms_reference_prior_weight"], profile["ms_kf_sigma_m"],
                profile["ms_reference_sigma_m"],
            ),
            flush=True,
        )
        summary = tracker.run_route_inference("route_A", visual, model, cache, route, device)
        csv_path = _resolve_csv(summary, out)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if val_end > len(rows) or val_start >= val_end:
            raise RuntimeError(
                f"invalid validation slice [{val_start}, {val_end}) for {len(rows)} rows"
            )
        val_rows = rows[val_start:val_end]
        errors = np.asarray([float(r["error_final_m"]) for r in val_rows], dtype=np.float64)
        headings = np.asarray(
            [abs(float(r["heading_error_deg"])) for r in val_rows], dtype=np.float64
        )
        row = dict(profile)
        row.update({
            "val_frames": int(errors.size),
            "val_mle_m": float(errors.mean()),
            "val_p90_m": float(np.quantile(errors, 0.90)),
            "val_lsr15_pct": 100.0 * float(np.mean(errors <= 15.0)),
            "val_hsr15_pct": 100.0 * float(np.mean(headings <= 15.0)),
            "val_mhe_deg": float(headings.mean()),
        })
        row["objective"] = float(row["val_mle_m"] + 0.05 * row["val_p90_m"])
        results.append(row)
        print("[FINALMS-TRAINVAL-RESULT]", json.dumps(row, sort_keys=True), flush=True)

    best = min(results, key=lambda r: (r["objective"], r["val_mle_m"], r["val_p90_m"]))
    payload = {
        "selection_source": "route_A_validation_only",
        "held_out_navigation_read": False,
        "city": cli.city,
        "validation_range": [val_start, val_end],
        "criterion": "MLE + 0.05*P90",
        "purpose": "make final MeanShift depend on the quality of the upstream estimator instead of being dominated by the common reference prior",
        "best": best,
        "profiles": results,
    }
    out_json = train_root / "finalms_trainval_calibration.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[FINALMS-TRAINVAL-BEST]", json.dumps(best, sort_keys=True), flush=True)
    print("[FINALMS-TRAINVAL-JSON]", out_json, flush=True)


if __name__ == "__main__":
    main()
