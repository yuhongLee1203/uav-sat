#!/usr/bin/env python3
"""Run Bearing-UAV with the canonical v39 DirectFinalMS training/inference flow.

Canonical methodological flow reproduced here:

  Route A only -> one continuous 60-epoch temporal training run
  Route B/C    -> held-out inference

  Weighted Centroid -> 3-frame Context-GRU -> velocity motion
  -> fixed-R constrained Kalman -> one final 5x5 MeanShift -> Final Position

Bearing-specific code is restricted to data/coordinate adaptation and pseudo-flight
construction. It must NOT silently switch training routes every 20 epochs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import bearing_runner as base

ORIGINAL_PATCH_PATHS = base._patch_bearing_paths
ORIGINAL_AUDIT = base._audit_canonical

EXACT_ARCH = "V39_GRU_Kalman_MS"
EXPECTED_SELECTION_VERSION = "soft_sequence_v12_dense_then_physical_leg_prune"
CANONICAL_TEMPORAL_EPOCHS = 60


def _require_audited_prepared(args, prepared_root: Path) -> None:
    if bool(getattr(args, "reprepare", False)):
        raise RuntimeError(
            "Exact-v39 runner does not permit --reprepare. Run "
            "run_bearing_v39_sequence_fixed.sh so the audited prepared data is used."
        )

    exp_path = prepared_root / "experiment.json"
    if not exp_path.exists():
        raise RuntimeError(f"Missing audited Bearing experiment: {exp_path}")
    exp = json.loads(exp_path.read_text(encoding="utf-8"))

    errors = []
    if Path(exp.get("dataset_root", "")).resolve() != Path(args.dataset_root).resolve():
        errors.append("dataset_root mismatch")
    if str(exp.get("city", "")).lower() != str(args.city).lower():
        errors.append("city mismatch")
    if exp.get("sequence_selection_version") != EXPECTED_SELECTION_VERSION:
        errors.append(
            "selection_version=%r expected %r"
            % (exp.get("sequence_selection_version"), EXPECTED_SELECTION_VERSION)
        )

    for route_name in (*base.TRAIN_ROUTES, *base.TEST_ROUTES):
        stats = exp.get("route_stats", {}).get(route_name)
        if not isinstance(stats, dict):
            errors.append(f"missing route_stats[{route_name}]")
            continue
        if abs(float(stats.get("sample_step_m", -1.0)) - float(args.step_m)) > 1e-6:
            errors.append(
                f"{route_name} sample_step={stats.get('sample_step_m')} != runner step={args.step_m}"
            )
        for filename in ("manifest.csv", "waypoints.json"):
            path = prepared_root / "routes" / route_name / filename
            if not path.exists():
                errors.append(f"missing {path}")

    if errors:
        raise RuntimeError(
            "Exact-v39 refused non-canonical/stale setup:\n- " + "\n- ".join(errors)
        )

    print(
        "[EXACT-V39] prepared-data lock: PASS | "
        f"selection={EXPECTED_SELECTION_VERSION}",
        flush=True,
    )
    print(
        "[EXACT-V39] training protocol: ONE Route A, 60 continuous epochs -> Route B/C inference",
        flush=True,
    )


def _set_exact_environment(args, prepared_root: Path):
    output = prepared_root / "v39_output_exact"
    checkpoints = output / "checkpoints"
    feature_cache = prepared_root / "feature_cache_exact"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)

    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    env = {
        "UAVSAT_DEVICE": f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu",
        "UAVSAT_OUTPUT_DIR": str(output),
        "UAVSAT_CHECKPOINT_DIR": str(checkpoints),
        "UAVSAT_FEATURE_CACHE_DIR": str(feature_cache),
        "UAVSAT_DATA_ROOT": str(prepared_root),
        "UAVSAT_BACKBONE": str(args.backbone),
        "UAVSAT_ARCHITECTURE_NAME": EXACT_ARCH,
        "UAVSAT_REFERENCE_PROTOCOL": "controlled_gt_jitter",
        "UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid",
        "UAVSAT_EXPERIMENT_FRAME_COUNT": "3",
        "UAVSAT_EXPERIMENT_MOTION": "velocity",
        "UAVSAT_EXPERIMENT_KALMAN": "fixed",
        "UAVSAT_EXPERIMENT_DISABLE_GRU": "0",
        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1",
        "UAVSAT_SAT_IMAGE": str(Path(exp["satellite_image"]).resolve()),
        "UAVSAT_SAT_JSON": str((prepared_root / "bearing_satellite.json").resolve()),
        "MS_ENABLED": "1",
        "MS_GRID_SIZE": "5",
        "MS_BANDWIDTH_M": "7.0",
    }
    os.environ.update(env)
    return output, checkpoints, feature_cache


def _patch_exact_paths(config, args, prepared_root: Path) -> None:
    ORIGINAL_PATCH_PATHS(config, args, prepared_root)

    # Canonical v39: one Route A for visual+temporal training, held-out B/C.
    config.ROUTE_NAMES = ["route_A", "route_B", "route_C"]
    config.ROUTE_ROOTS = [
        prepared_root / "routes" / "train_01",
        prepared_root / "routes" / "test_01",
        prepared_root / "routes" / "test_02",
    ]
    config.WAYPOINT_FILES = {
        "route_A": prepared_root / "routes" / "train_01" / "waypoints.json",
        "route_B": prepared_root / "routes" / "test_01" / "waypoints.json",
        "route_C": prepared_root / "routes" / "test_02" / "waypoints.json",
    }

    # Only unavoidable external-dataset bookkeeping adaptation.
    config.SPLIT_GUARD_FRAMES = 2

    config.TEMPORAL_EPOCHS = CANONICAL_TEMPORAL_EPOCHS
    config.MAX_FORWARD_SPEED_M_PER_FRAME = 14.0
    config.MAX_CROSS_SPEED_M_PER_FRAME = 5.0
    config.MAX_FINAL_CROSS_TRACK_M = 10.0
    config.MAX_FORWARD_ACCEL_M_PER_FRAME2 = 5.0
    config.MAX_CROSS_ACCEL_M_PER_FRAME2 = 4.0
    config.MAX_POLYNOMIAL_STEP_M_PER_FRAME = 14.0
    config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 4.0
    config.MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0
    config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = 5.0
    config.KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = 3.0
    config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = 3.0
    config.KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 1.75
    config.KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = 1.25
    config.KALMAN_FINAL_STEP_SLACK_M = 0.0
    config.KALMAN_FINAL_STEP_MIN_M = 0.0
    config.KALMAN_FINAL_STEP_MAX_M = 7.0
    config.BEARING_CADENCE_ADAPTATION = None


def _exact_model_values(config):
    return {
        "reference_protocol": str(config.REFERENCE_PROTOCOL),
        "visual_decoder": str(config.EXPERIMENT_ANCHOR),
        "frame_count": int(config.EXPERIMENT_FRAME_COUNT),
        "motion": str(config.EXPERIMENT_MOTION),
        "kalman": str(config.EXPERIMENT_KALMAN),
        "forward_only": bool(config.FORWARD_ONLY_LOCAL_SEARCH),
        "temporal_epochs": int(config.TEMPORAL_EPOCHS),
        "temporal_lr": float(config.TEMPORAL_LR),
        "loss_measurement": float(config.LOSS_MEASUREMENT),
        "loss_next_step": float(config.LOSS_NEXT_STEP),
        "loss_velocity": float(config.LOSS_VELOCITY),
        "teacher_ratio_final": float(config.TEACHER_RATIO_FINAL),
        "max_forward_speed_m_per_frame": float(config.MAX_FORWARD_SPEED_M_PER_FRAME),
        "max_cross_speed_m_per_frame": float(config.MAX_CROSS_SPEED_M_PER_FRAME),
        "max_polynomial_step_m_per_frame": float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME),
        "max_measurement_correction_parallel_m": float(config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M),
        "max_measurement_correction_cross_m": float(config.MAX_MEASUREMENT_CORRECTION_CROSS_M),
        "kalman_max_measurement_innovation_progress_m": float(config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M),
        "kalman_max_measurement_innovation_cross_m": float(config.KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M),
        "kalman_max_posterior_correction_progress_m": float(config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M),
        "kalman_max_posterior_correction_cross_m": float(config.KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M),
        "kalman_max_velocity_correction_m_per_frame": float(config.KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME),
        "kalman_final_step_max_m": float(config.KALMAN_FINAL_STEP_MAX_M),
        "ms_grid_size": 5,
        "ms_bandwidth_m": 7.0,
    }


def _audit_exact(config, runtime: Path, args, prepared_root: Path) -> None:
    ORIGINAL_AUDIT(config, runtime, args, prepared_root)
    actual = _exact_model_values(config)
    expected = {
        "reference_protocol": "controlled_gt_jitter",
        "visual_decoder": "weighted_centroid",
        "frame_count": 3,
        "motion": "velocity",
        "kalman": "fixed",
        "forward_only": True,
        "temporal_epochs": 60,
        "temporal_lr": 2e-4,
        "loss_measurement": 1.0,
        "loss_next_step": 3.0,
        "loss_velocity": 0.25,
        "teacher_ratio_final": 1.0,
        "max_forward_speed_m_per_frame": 14.0,
        "max_cross_speed_m_per_frame": 5.0,
        "max_polynomial_step_m_per_frame": 14.0,
        "max_measurement_correction_parallel_m": 4.0,
        "max_measurement_correction_cross_m": 4.0,
        "kalman_max_measurement_innovation_progress_m": 5.0,
        "kalman_max_measurement_innovation_cross_m": 3.0,
        "kalman_max_posterior_correction_progress_m": 3.0,
        "kalman_max_posterior_correction_cross_m": 1.75,
        "kalman_max_velocity_correction_m_per_frame": 1.25,
        "kalman_final_step_max_m": 7.0,
        "ms_grid_size": 5,
        "ms_bandwidth_m": 7.0,
    }
    mismatches = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise RuntimeError("Exact-v39 audit failed: %s" % json.dumps(mismatches, indent=2))

    expected_roots = [
        prepared_root / "routes" / "train_01",
        prepared_root / "routes" / "test_01",
        prepared_root / "routes" / "test_02",
    ]
    if [Path(p).resolve() for p in config.ROUTE_ROOTS] != [p.resolve() for p in expected_roots]:
        raise RuntimeError("Exact-v39 Route A/B/C mapping audit failed")

    audit_path = Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({
        "architecture": EXACT_ARCH,
        "exact_v39_model_inference": True,
        "bearing_cadence_adaptation": False,
        "prepared_selection_version": EXPECTED_SELECTION_VERSION,
        "training_protocol": "single continuous Route-A-only temporal training for 60 epochs",
        "route_mapping": {
            "route_A_train": "train_01",
            "route_B_eval": "test_01",
            "route_C_eval": "test_02",
            "unused_extra_training_routes": ["train_02", "train_03"],
        },
        "exact_v39_values": actual,
        "dataset_only_adaptations": {
            "bearing_coordinate_and_image_adapter": True,
            "pseudo_flight_sequence_construction": True,
            "split_guard_frames": 2,
            "split_reason": "3-frame temporal window on shorter external Route A",
        },
    })
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("[EXACT-V39] model/training/inference settings: PASS", flush=True)
    print(json.dumps(actual, indent=2), flush=True)
    print("[EXACT-V39] Route A=train_01 | Route B=test_01 | Route C=test_02", flush=True)
    print("[EXACT-V39] temporal training = ONE continuous 60-epoch A-only run", flush=True)
    print("[EXACT-V39] Bearing cadence adaptation: DISABLED", flush=True)


def _train_and_infer_exact_a_only(args, prepared_root: Path) -> None:
    runtime = base._make_runtime(prepared_root)
    _set_exact_environment(args, prepared_root)
    config, tracker, visual_localizer = base._load_runtime_modules(runtime)
    _patch_exact_paths(config, args, prepared_root)
    _audit_exact(config, runtime, args, prepared_root)

    device = tracker.resolve_device()
    if not args.resume:
        for path in (
            config.VISUAL_CHECKPOINT,
            config.TEMPORAL_CHECKPOINT,
            config.LATEST_TEMPORAL_CHECKPOINT,
        ):
            if Path(path).exists() or Path(path).is_symlink():
                Path(path).unlink()

    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        visual_localizer.train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume),
        )
    else:
        print("reuse visual checkpoint:", config.VISUAL_CHECKPOINT, flush=True)

    visual = visual_localizer.FrozenVisualLocalizer(device)
    cache_a = tracker.build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route_a = tracker.WaypointRoute(
        tracker.load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon)
    )

    print("\n=== EXACT v39 temporal training: Route A only, 60 epochs ===", flush=True)
    tracker.train_temporal_model(
        visual=visual,
        cache=cache_a,
        route=route_a,
        device=device,
        epochs=CANONICAL_TEMPORAL_EPOCHS,
        patience_limit=int(args.patience),
        resume=bool(args.resume),
    )

    if not config.TEMPORAL_CHECKPOINT.exists():
        raise RuntimeError("Canonical A-only temporal training produced no best checkpoint")
    model = tracker.load_temporal_model(device)

    summaries = {}
    for external_name, canonical_name, root in (
        ("test_01", "route_B", config.ROUTE_ROOTS[1]),
        ("test_02", "route_C", config.ROUTE_ROOTS[2]),
    ):
        cache = tracker.build_route_cache(canonical_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(canonical_name, visual.origin_lat, visual.origin_lon)
        )
        print(
            f"\n=== held-out exact-v39 inference: {external_name} ({canonical_name}) ===",
            flush=True,
        )
        summaries[external_name] = tracker.run_route_inference(
            external_name, visual, model, cache, route, device
        )

    summary_path = Path(config.OUTPUT_DIR) / "bearing_v39_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, default=float), encoding="utf-8"
    )
    print("\n[DONE] summary:", summary_path, flush=True)


base.FINAL_ARCH = EXACT_ARCH
base._ensure_prepared = _require_audited_prepared
base._set_canonical_environment = _set_exact_environment
base._patch_bearing_paths = _patch_exact_paths
base._audit_canonical = _audit_exact
base.train_and_infer = _train_and_infer_exact_a_only


if __name__ == "__main__":
    base.main()
