#!/usr/bin/env python3
"""Eval-only Bearing-UAV runner for the V39 Forward3x6 SoftMS front decoder.

This intentionally reuses the already-trained per-city checkpoints from
v39_output_bearing_adapted.  No visual or temporal training is performed.
The only inference-side architecture change is the front visual decoder:
Forward 3x6 candidates -> Soft MeanShift instead of Weighted Centroid.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

import bearing_runner_multicity_v39 as multi

exact = multi.exact
base = exact.base
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SOFTMS_PATCH = REPO_ROOT / "v39_DirectFinalMS" / "patch_front_softms.py"


def _set_softms_environment(args, prepared_root: Path):
    output = prepared_root / "v39_output_bearing_softms_eval"
    checkpoints = output / "checkpoints"
    feature_cache = prepared_root / "feature_cache_bearing_adapted"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)
    exp = exact._experiment(prepared_root)
    os.environ.update({
        "UAVSAT_DEVICE": f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu",
        "UAVSAT_OUTPUT_DIR": str(output),
        "UAVSAT_CHECKPOINT_DIR": str(checkpoints),
        "UAVSAT_FEATURE_CACHE_DIR": str(feature_cache),
        "UAVSAT_DATA_ROOT": str(prepared_root),
        "UAVSAT_BACKBONE": str(args.backbone),
        "UAVSAT_ARCHITECTURE_NAME": exact.ARCH,
        "UAVSAT_REFERENCE_PROTOCOL": "controlled_gt_jitter",
        "UAVSAT_EXPERIMENT_ANCHOR": "softms",
        "UAVSAT_EXPERIMENT_FRAME_COUNT": "3",
        "UAVSAT_EXPERIMENT_MOTION": "velocity",
        "UAVSAT_EXPERIMENT_KALMAN": "fixed",
        "UAVSAT_EXPERIMENT_DISABLE_GRU": "0",
        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1",
        "UAVSAT_SAT_IMAGE": str(Path(exp["satellite_image"]).resolve()),
        "UAVSAT_SAT_JSON": str((prepared_root / "bearing_satellite.json").resolve()),
        "MS_ENABLED": "1",
        "MS_GRID_SIZE": "6",
        "MS_BANDWIDTH_M": "7.0",
    })
    return output, checkpoints, feature_cache


def _link_existing_checkpoints(config, prepared_root: Path) -> None:
    source = prepared_root / "v39_output_bearing_adapted" / "checkpoints"
    if not source.exists():
        raise FileNotFoundError(
            f"missing existing trained checkpoint directory: {source}; run the completed V39 city experiment first"
        )
    required = [Path(config.VISUAL_CHECKPOINT), Path(config.TEMPORAL_CHECKPOINT)]
    for dest in required:
        src = source / dest.name
        if not src.exists():
            raise FileNotFoundError(f"missing trained checkpoint: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())
        print(f"[SOFTMS-EVAL] reuse checkpoint: {dest.name} -> {src}", flush=True)


def _audit_softms(config, runtime: Path) -> None:
    checks = {
        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",
        "forward_only_3x6": bool(config.FORWARD_ONLY_LOCAL_SEARCH),
        "frame_count_3": int(config.EXPERIMENT_FRAME_COUNT) == 3,
        "constant_velocity": str(config.EXPERIMENT_MOTION) == "velocity",
        "fixed_kalman": str(config.EXPERIMENT_KALMAN) == "fixed",
        "final_ms_6x6": int(os.environ.get("MS_GRID_SIZE", "0")) == 6,
        "final_ms_bw7": abs(float(os.environ.get("MS_BANDWIDTH_M", "0")) - 7.0) < 1e-9,
    }
    text = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    checks["weighted_centroid_front_removed"] = (
        "weighted_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)" not in text
    )
    checks["front_softms_present"] = "Front visual observation: Soft MeanShift" in text
    for name, ok in checks.items():
        print(f"[SOFTMS-AUDIT] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError("SoftMS architecture audit failed: " + ", ".join(bad))


def eval_only(args, prepared_root: Path) -> None:
    exact._require_audited_prepared(args, prepared_root)

    runtime = base._make_runtime(prepared_root)
    if not SOFTMS_PATCH.exists():
        raise FileNotFoundError(SOFTMS_PATCH)
    subprocess.run(
        [sys.executable, str(SOFTMS_PATCH), str(runtime / "robust_tracker.py")],
        check=True,
    )
    exact._patch_final_ms_reference(runtime)
    output, _, _ = _set_softms_environment(args, prepared_root)
    config, tracker, visual_localizer = base._load_runtime_modules(runtime)
    exact._patch_paths_and_scale(config, args, prepared_root)
    _link_existing_checkpoints(config, prepared_root)
    _audit_softms(config, runtime)

    device = tracker.resolve_device()
    visual = visual_localizer.FrozenVisualLocalizer(device)
    model = tracker.load_temporal_model(device)

    cadence = exact._training_only_adaptation(prepared_root)
    geometry = exact._physical_sat_geometry(prepared_root)
    summaries = {}
    for external_name, canonical_name, root in (
        ("test_01", "route_B", config.ROUTE_ROOTS[1]),
        ("test_02", "route_C", config.ROUTE_ROOTS[2]),
    ):
        cache = tracker.build_route_cache(canonical_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(canonical_name, visual.origin_lat, visual.origin_lon)
        )
        print(f"\n=== SOFTMS eval only: {external_name} ({canonical_name}) ===", flush=True)
        result = tracker.run_route_inference(external_name, visual, model, cache, route, device)
        result["BearingAdaptation"] = {
            "uses_test_statistics": False,
            "training_only_cadence": cadence,
            "physical_sat_geometry": geometry,
            "front_decoder": "forward_3x6_soft_mean_shift",
            "checkpoint_retraining": False,
            "checkpoint_source": "v39_output_bearing_adapted/checkpoints",
            "final_ms_reference": "planned_route_centerline_at_true_sample_progress",
            "metric_ground_truth": "true_selected_Bearing_sample_coordinate",
        }
        summaries[external_name] = result

    summary_path = output / "bearing_v39_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=float), encoding="utf-8")
    print("\n[SOFTMS-EVAL] NO TRAINING PERFORMED", flush=True)
    print("[SOFTMS-EVAL] summary:", summary_path, flush=True)


def main() -> None:
    args = base.build_parser().parse_args()
    # The prepared multicity protocol requires these canonical values even
    # though eval-only does not consume the epoch count for training.
    args.epochs_per_route = 60
    args.reuse_visual = True
    args.resume = True
    prepared_root = HERE / "generated" / args.city
    eval_only(args, prepared_root)


if __name__ == "__main__":
    main()
