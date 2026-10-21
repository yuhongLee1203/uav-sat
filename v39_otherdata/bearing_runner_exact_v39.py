#!/usr/bin/env python3
"""Run the selected v39 DirectFinalMS method on Bearing-UAV.

The model/training recipe remains the selected v39 method:

  Route A only -> one continuous 60-epoch temporal training run
  Route B/C    -> held-out inference
  Weighted Centroid -> 3-frame Context-GRU -> Constant Velocity
  -> fixed-R constrained Kalman -> one final 6x6 MeanShift (BW 7 m)

Bearing-UAV differs from the original field data in two unavoidable physical
properties: satellite m/px and pseudo-flight frame cadence.  This adapter keeps
v39's *metric* geometry and estimator role instead of blindly copying pixel and
per-frame constants from a different sampling scale.

Only TRAIN Route A statistics are allowed to determine cadence limits.  Test
routes never tune the estimator.  True Bearing sample coordinates are preserved
for MLE/P90/LSR; only the final-MS spatial reference is the planned route
centerline at the current sample progress, because independent Bearing samples
are not a recorded continuous flight and their lateral scatter must not become a
fake zig-zag motion prior.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import bearing_runner as base

ORIGINAL_PATCH_PATHS = base._patch_bearing_paths
ORIGINAL_AUDIT = base._audit_canonical

ARCH = "V39_ContextGRU_FixedKalman_FinalMS6x6_BearingAdapted"
EXPECTED_SELECTION_VERSION = "soft_sequence_v12_dense_then_physical_leg_prune"
CANONICAL_TEMPORAL_EPOCHS = 60

# Physical geometry of the selected v39 experiment on the original 0.14 m/px map.
CANONICAL_MPP = 0.14
CANONICAL_SAT_STRIDE_PX = 32
CANONICAL_SAT_CROP_PX = 320
CANONICAL_STRIDE_M = CANONICAL_MPP * CANONICAL_SAT_STRIDE_PX   # 4.48 m
CANONICAL_CROP_M = CANONICAL_MPP * CANONICAL_SAT_CROP_PX       # 44.8 m


def _experiment(prepared_root: Path):
    return json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))


def _training_only_adaptation(prepared_root: Path):
    """Derive all cadence values from train_01 only. Never inspect B/C stats."""
    exp = _experiment(prepared_root)
    train = exp["route_stats"]["train_01"]
    p90 = float(train["actual_step_p90_m"])
    p95 = float(train["actual_step_p95_m"])
    mean = float(train["actual_step_mean_m"])

    # Longitudinal limits only. Cross-track limits remain canonical so the
    # external adapter does not gain extra freedom to wiggle sideways.
    values = {
        "source": "train_01_only",
        "train_step_mean_m": mean,
        "train_step_p90_m": p90,
        "train_step_p95_m": p95,
        "max_forward_speed_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),
        "max_polynomial_step_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),
        "max_measurement_correction_parallel_m": max(4.0, min(8.0, 0.55 * p90)),
        "kalman_max_measurement_innovation_progress_m": max(5.0, min(10.0, 0.65 * p90)),
        "kalman_max_posterior_correction_progress_m": max(3.0, min(6.0, 0.45 * p90)),
        "kalman_max_velocity_correction_m_per_frame": max(1.25, min(2.5, 0.18 * p90)),
        "kalman_final_step_max_m": max(7.0, min(14.0, 1.10 * p90)),
    }
    return values


def _physical_sat_geometry(prepared_root: Path):
    exp = _experiment(prepared_root)
    mpp = float(exp["mpp"])
    stride_px = max(1, int(round(CANONICAL_STRIDE_M / mpp)))
    # Keep crop even-sized for a symmetric center pixel convention.
    crop_float = CANONICAL_CROP_M / mpp
    crop_px = max(64, int(round(crop_float / 2.0) * 2))
    return {
        "bearing_mpp": mpp,
        "canonical_stride_m": CANONICAL_STRIDE_M,
        "canonical_crop_m": CANONICAL_CROP_M,
        "sat_stride_px": stride_px,
        "sat_stride_m": stride_px * mpp,
        "sat_crop_px": crop_px,
        "sat_crop_m": crop_px * mpp,
    }


def _require_audited_prepared(args, prepared_root: Path) -> None:
    if bool(getattr(args, "reprepare", False)):
        raise RuntimeError(
            "Bearing-v39 runner does not permit --reprepare. Use "
            "run_bearing_v39_sequence_fixed.sh so the audited route package is used."
        )
    exp_path = prepared_root / "experiment.json"
    if not exp_path.exists():
        raise RuntimeError(f"Missing prepared experiment: {exp_path}")
    exp = _experiment(prepared_root)
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
    if int(args.epochs_per_route) != CANONICAL_TEMPORAL_EPOCHS:
        errors.append(
            f"--epochs-per-route={args.epochs_per_route}; this wrapper requires 60 for the canonical A-only schedule"
        )
    if errors:
        raise RuntimeError("Bearing-v39 refused stale/invalid setup:\n- " + "\n- ".join(errors))
    print(
        "[V39-BEARING] prepared-data lock: PASS | "
        f"selection={EXPECTED_SELECTION_VERSION}",
        flush=True,
    )


def _set_environment(args, prepared_root: Path):
    output = prepared_root / "v39_output_bearing_adapted"
    checkpoints = output / "checkpoints"
    feature_cache = prepared_root / "feature_cache_bearing_adapted"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)
    exp = _experiment(prepared_root)
    os.environ.update({
        "UAVSAT_DEVICE": f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu",
        "UAVSAT_OUTPUT_DIR": str(output),
        "UAVSAT_CHECKPOINT_DIR": str(checkpoints),
        "UAVSAT_FEATURE_CACHE_DIR": str(feature_cache),
        "UAVSAT_DATA_ROOT": str(prepared_root),
        "UAVSAT_BACKBONE": str(args.backbone),
        "UAVSAT_ARCHITECTURE_NAME": ARCH,
        "UAVSAT_REFERENCE_PROTOCOL": "controlled_gt_jitter",
        "UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid",
        "UAVSAT_EXPERIMENT_FRAME_COUNT": "3",
        "UAVSAT_EXPERIMENT_MOTION": "velocity",
        "UAVSAT_EXPERIMENT_KALMAN": "fixed",
        "UAVSAT_EXPERIMENT_DISABLE_GRU": "0",
        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1",
        "UAVSAT_SAT_IMAGE": str(Path(exp["satellite_image"]).resolve()),
        "UAVSAT_SAT_JSON": str((prepared_root / "bearing_satellite.json").resolve()),
        # Current selected v39 setting from README/full_model.
        "MS_ENABLED": "1",
        "MS_GRID_SIZE": "6",
        "MS_BANDWIDTH_M": "7.0",
    })
    return output, checkpoints, feature_cache


def _patch_final_ms_reference(runtime: Path) -> None:
    """Prevent independent Bearing sample scatter from becoming final-MS wobble."""
    path = runtime / "robust_tracker.py"
    text = path.read_text(encoding="utf-8")
    old = '''        # Keep the original v39 predefined frame-reference prior unchanged.\n        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()\n        frame_reference_xy = (\n            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)\n        )\n'''
    new = '''        # Bearing-UAV consists of independent observations, not a recorded\n        # continuous video.  The true sample coordinate remains the metric GT,\n        # but using its lateral sampling scatter as a strong final-MS spatial\n        # prior creates artificial frame-to-frame zig-zags.  Use the planned\n        # route centerline at the sample's true route progress as the spatial\n        # reference, while ALL error metrics below still use cache.gt_xy.\n        reference_progress_s = float(gt_state["se"][index, 0])\n        frame_reference_xy = np.asarray(\n            route.xy_from_se(reference_progress_s, 0.0), dtype=np.float64\n        )\n        frame_reference_xy_t = torch.tensor(\n            frame_reference_xy[None, :], dtype=torch.float32, device=device\n        )\n'''
    if text.count(old) != 1:
        raise RuntimeError(
            "Bearing final-MS centerline patch did not match canonical runtime exactly once"
        )
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")
    print("[V39-BEARING] final-MS route-centerline reference patch: PASS", flush=True)


def _patch_paths_and_scale(config, args, prepared_root: Path) -> None:
    ORIGINAL_PATCH_PATHS(config, args, prepared_root)

    # Exact A-only -> B/C role mapping.
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
    config.SPLIT_GUARD_FRAMES = 2
    config.TEMPORAL_EPOCHS = CANONICAL_TEMPORAL_EPOCHS

    # Match v39 satellite geometry in METRES, not pixels.
    geometry = _physical_sat_geometry(prepared_root)
    config.SAT_STRIDE = int(geometry["sat_stride_px"])
    config.SAT_CROP_SIZE = int(geometry["sat_crop_px"])
    config.GRID_SIZE = 6
    config.CANDIDATE_COUNT = 36

    # Keep canonical lateral constraints. Adapt longitudinal per-frame limits
    # using TRAIN Route A only because Bearing pseudo-flight cadence is sparser.
    cadence = _training_only_adaptation(prepared_root)
    config.MAX_FORWARD_SPEED_M_PER_FRAME = cadence["max_forward_speed_m_per_frame"]
    config.MAX_CROSS_SPEED_M_PER_FRAME = 5.0
    config.MAX_FINAL_CROSS_TRACK_M = 10.0
    config.MAX_FORWARD_ACCEL_M_PER_FRAME2 = 5.0
    config.MAX_CROSS_ACCEL_M_PER_FRAME2 = 4.0
    config.MAX_POLYNOMIAL_STEP_M_PER_FRAME = cadence["max_polynomial_step_m_per_frame"]
    config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M = cadence["max_measurement_correction_parallel_m"]
    config.MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0
    config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = cadence["kalman_max_measurement_innovation_progress_m"]
    config.KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = 3.0
    config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = cadence["kalman_max_posterior_correction_progress_m"]
    config.KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 1.75
    config.KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = cadence["kalman_max_velocity_correction_m_per_frame"]
    config.KALMAN_FINAL_STEP_SLACK_M = 0.0
    config.KALMAN_FINAL_STEP_MIN_M = 0.0
    config.KALMAN_FINAL_STEP_MAX_M = cadence["kalman_final_step_max_m"]
    config.BEARING_CADENCE_ADAPTATION = cadence
    config.BEARING_PHYSICAL_SAT_GEOMETRY = geometry


def _values(config):
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
        "sat_stride_px": int(config.SAT_STRIDE),
        "sat_crop_px": int(config.SAT_CROP_SIZE),
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
        "ms_grid_size": 6,
        "ms_bandwidth_m": 7.0,
    }


def _audit(config, runtime: Path, args, prepared_root: Path) -> None:
    # First prove the runtime still contains the canonical DirectFinalMS and
    # Context-GRU source/patches.
    ORIGINAL_AUDIT(config, runtime, args, prepared_root)

    actual = _values(config)
    fixed_expected = {
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
        "max_cross_speed_m_per_frame": 5.0,
        "max_measurement_correction_cross_m": 4.0,
        "kalman_max_measurement_innovation_cross_m": 3.0,
        "kalman_max_posterior_correction_cross_m": 1.75,
        "ms_grid_size": 6,
        "ms_bandwidth_m": 7.0,
    }
    bad = {
        k: {"actual": actual[k], "expected": v}
        for k, v in fixed_expected.items() if actual[k] != v
    }
    if bad:
        raise RuntimeError("v39 fixed-setting audit failed: %s" % json.dumps(bad, indent=2))

    expected_roots = [
        prepared_root / "routes" / "train_01",
        prepared_root / "routes" / "test_01",
        prepared_root / "routes" / "test_02",
    ]
    if [Path(p).resolve() for p in config.ROUTE_ROOTS] != [p.resolve() for p in expected_roots]:
        raise RuntimeError("Route A/B/C mapping audit failed")

    cadence = _training_only_adaptation(prepared_root)
    geometry = _physical_sat_geometry(prepared_root)
    if int(config.SAT_STRIDE) != int(geometry["sat_stride_px"]):
        raise RuntimeError("SAT stride physical-scale audit failed")
    if int(config.SAT_CROP_SIZE) != int(geometry["sat_crop_px"]):
        raise RuntimeError("SAT crop physical-scale audit failed")
    dynamic_checks = {
        "max_forward_speed_m_per_frame": cadence["max_forward_speed_m_per_frame"],
        "max_polynomial_step_m_per_frame": cadence["max_polynomial_step_m_per_frame"],
        "max_measurement_correction_parallel_m": cadence["max_measurement_correction_parallel_m"],
        "kalman_max_measurement_innovation_progress_m": cadence["kalman_max_measurement_innovation_progress_m"],
        "kalman_max_posterior_correction_progress_m": cadence["kalman_max_posterior_correction_progress_m"],
        "kalman_max_velocity_correction_m_per_frame": cadence["kalman_max_velocity_correction_m_per_frame"],
        "kalman_final_step_max_m": cadence["kalman_final_step_max_m"],
    }
    for key, value in dynamic_checks.items():
        if abs(float(actual[key]) - float(value)) > 1e-9:
            raise RuntimeError(f"training-only cadence audit failed: {key}")

    audit_path = Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({
        "architecture": ARCH,
        "same_v39_architecture_and_training_recipe": True,
        "exact_same_numeric_limits_as_original_dataset": False,
        "reason_numeric_limits_differ": "external pseudo-flight has different frame cadence",
        "training_protocol": "single continuous Route-A-only temporal training for 60 epochs",
        "route_mapping": {
            "route_A_train": "train_01",
            "route_B_eval": "test_01",
            "route_C_eval": "test_02",
            "unused_extra_training_routes": ["train_02", "train_03"],
        },
        "test_statistics_used_for_adaptation": False,
        "training_only_cadence_adaptation": cadence,
        "physical_satellite_geometry_adaptation": geometry,
        "final_ms_reference": "planned route centerline at current true-sample route progress; metric GT remains the true Bearing sample coordinate",
        "selected_v39_final_ms": {"grid": 6, "bandwidth_m": 7.0},
        "effective_values": actual,
        "prepared_selection_version": EXPECTED_SELECTION_VERSION,
    })
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("[V39-BEARING] architecture/training audit: PASS", flush=True)
    print("[V39-BEARING] Route A=train_01 | Route B=test_01 | Route C=test_02", flush=True)
    print("[V39-BEARING] temporal training: ONE continuous 60-epoch A-only run", flush=True)
    print("[V39-BEARING] test statistics used for adaptation: NO", flush=True)
    print("[V39-BEARING] physical SAT geometry:", json.dumps(geometry, indent=2), flush=True)
    print("[V39-BEARING] train_01 cadence adaptation:", json.dumps(cadence, indent=2), flush=True)


def _train_and_infer(args, prepared_root: Path) -> None:
    runtime = base._make_runtime(prepared_root)
    _patch_final_ms_reference(runtime)
    _set_environment(args, prepared_root)
    config, tracker, visual_localizer = base._load_runtime_modules(runtime)
    _patch_paths_and_scale(config, args, prepared_root)
    _audit(config, runtime, args, prepared_root)

    device = tracker.resolve_device()
    if not args.resume:
        for path in (
            config.VISUAL_CHECKPOINT,
            config.TEMPORAL_CHECKPOINT,
            config.LATEST_TEMPORAL_CHECKPOINT,
        ):
            p = Path(path)
            if p.exists() or p.is_symlink():
                p.unlink()

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

    print("\n=== v39 Bearing temporal training: Route A only, 60 epochs ===", flush=True)
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
        raise RuntimeError("A-only temporal training produced no best checkpoint")
    model = tracker.load_temporal_model(device)

    cadence = _training_only_adaptation(prepared_root)
    geometry = _physical_sat_geometry(prepared_root)
    summaries = {}
    for external_name, canonical_name, root in (
        ("test_01", "route_B", config.ROUTE_ROOTS[1]),
        ("test_02", "route_C", config.ROUTE_ROOTS[2]),
    ):
        cache = tracker.build_route_cache(canonical_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(canonical_name, visual.origin_lat, visual.origin_lon)
        )
        print(f"\n=== held-out inference: {external_name} ({canonical_name}) ===", flush=True)
        result = tracker.run_route_inference(external_name, visual, model, cache, route, device)
        result["BearingAdaptation"] = {
            "uses_test_statistics": False,
            "training_only_cadence": cadence,
            "physical_sat_geometry": geometry,
            "final_ms_reference": "planned_route_centerline_at_true_sample_progress",
            "metric_ground_truth": "true_selected_Bearing_sample_coordinate",
        }
        summaries[external_name] = result

    summary_path = Path(config.OUTPUT_DIR) / "bearing_v39_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=float), encoding="utf-8")
    print("\n[DONE] summary:", summary_path, flush=True)


base.FINAL_ARCH = ARCH
base._ensure_prepared = _require_audited_prepared
base._set_canonical_environment = _set_environment
base._patch_bearing_paths = _patch_paths_and_scale
base._audit_canonical = _audit
base.train_and_infer = _train_and_infer


if __name__ == "__main__":
    base.main()
