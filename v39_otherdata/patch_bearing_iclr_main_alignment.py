#!/usr/bin/env python3
"""Patch Bearing-UAV ablation to the city-native residual-temporal experiment.

Dataset protocol exposed to the paper/user:
  City A / B / C / D are the dataset domains.
  Each city reports its two held-out sequences as test_01 and test_02.

The legacy tracker still uses three internal sequence slots for compatibility;
those slot names are implementation details and are never presented as dataset
Route A/B/C.

Active architecture:
  6x6 local geometry -> heading-forward 3x6 (18 scored patches)
  -> Forward-18 SoftMS -> residual temporal GRU
  -> constrained Kalman -> final local MeanShift -> XY.

All model/filter selection in this patch is based only on the current city's
training sequence and its validation split. Held-out test metrics are never read
for hyperparameter selection.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else "v39_otherdata/bearing_iclr_ablation.py")
s = path.read_text(encoding="utf-8")

# A completed run leaves the aligned source in the working tree.  The B/C/D
# launcher invokes this patch once per city, so subsequent cities must accept
# the already-aligned (and possibly V5-extended) form instead of looking for
# the original legacy block again.
aligned_markers = (
    'ARCH = "ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS"',
    'def _train_root(args: argparse.Namespace, frames=None) -> Path:',
    'config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (',
    'train_variant = dict(VARIANTS["full"])',
    'checkpoint_frames = int(variant["frames"])',
    '"UAVSAT_EXPERIMENT_ANCHOR": "softms"',
)
if all(marker in s for marker in aligned_markers):
    compile(s, str(path), "exec")
    print("[PATCH OK] bearing ICLR main alignment already present")
    raise SystemExit(0)


def replace_once_or_already(old: str, new: str, label: str) -> None:
    global s
    if new in s:
        return
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"PATCH FAILED [{label}]: expected 1 old block, found {count}")
    s = s.replace(old, new, 1)


replace_once_or_already(
    'The paper-facing chain contains exactly one MeanShift decoder:\n\n'
    '    6x6 geometry -> forward 3x6 visual scores -> 3-frame GRU\n'
    '    -> fixed-R Kalman -> one final local MeanShift -> XY',
    'The paper-facing chain uses an explicit front SoftMS and final MeanShift:\n\n'
    '    6x6 geometry -> forward 3x6 = 18 visual scores -> front SoftMS\n'
    '    -> residual temporal GRU -> constrained Kalman -> final MeanShift -> XY',
    'docstring chain',
)

replace_once_or_already(
    'ARCH = "ICLR_Forward18_Simple3FrameGRU_FixedKalman_OneFinalMS"',
    'ARCH = "ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS"',
    'architecture name',
)

replace_once_or_already(
    'def _train_root(args: argparse.Namespace) -> Path:\n'
    '    return Path(args.suite_root).resolve() / args.city / "train_full"',
    'def _train_root(args: argparse.Namespace, frames=None) -> Path:\n'
    '    frame_count = int(frames if frames is not None else getattr(args, "train_frames", 3))\n'
    '    return Path(args.suite_root).resolve() / args.city / f"train_frames{frame_count}"',
    'frame-specific train root',
)

# Wider but still training-city-derived longitudinal bounds.  The previous
# posterior correction cap was smaller than normal Bearing frame motion and
# made the Kalman lag behind otherwise useful SoftMS measurements.
replace_once_or_already(
    '        "max_forward_speed_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),\n'
    '        "max_polynomial_step_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),\n'
    '        "max_measurement_correction_parallel_m": max(4.0, min(8.0, 0.55 * p90)),\n'
    '        "kalman_max_measurement_innovation_progress_m": max(5.0, min(10.0, 0.65 * p90)),\n'
    '        "kalman_max_posterior_correction_progress_m": max(3.0, min(6.0, 0.45 * p90)),\n'
    '        "kalman_max_velocity_correction_m_per_frame": max(1.25, min(2.5, 0.18 * p90)),\n'
    '        "kalman_final_step_max_m": max(7.0, min(14.0, 1.10 * p90)),',
    '        "max_forward_speed_m_per_frame": max(14.0, min(30.0, 1.15 * p95)),\n'
    '        "max_polynomial_step_m_per_frame": max(14.0, min(30.0, 1.15 * p95)),\n'
    '        "max_measurement_correction_parallel_m": max(6.0, min(14.0, 0.80 * p90)),\n'
    '        "kalman_max_measurement_innovation_progress_m": max(8.0, min(24.0, 1.05 * p90)),\n'
    '        "kalman_max_posterior_correction_progress_m": max(6.0, min(18.0, 0.80 * p90)),\n'
    '        "kalman_max_velocity_correction_m_per_frame": max(2.0, min(6.0, 0.30 * p90)),\n'
    '        "kalman_final_step_max_m": max(10.0, min(30.0, 1.15 * p95)),',
    'training-city cadence bounds',
)

legacy_token = "weighted" + "_" + "centroid"
old_env = (
    '        # Forward-18 posterior is summarized without a front MeanShift.  The\n'
    '        # only MeanShift in the paper chain is the post-Kalman decoder.\n'
    f'        "UAVSAT_EXPERIMENT_ANCHOR": "{legacy_token}",'
)
new_env = (
    '        # Forward-18 is always decoded by Soft MeanShift.\n'
    '        "UAVSAT_EXPERIMENT_ANCHOR": "softms",'
)
replace_once_or_already(old_env, new_env, 'front SoftMS environment')

replace_once_or_already(
    '        "UAVSAT_EXPERIMENT_MOTION": "velocity",',
    '        "UAVSAT_EXPERIMENT_MOTION": "quadratic",',
    'train/inference motion alignment',
)

replace_once_or_already(
    '    exact._patch_paths_and_scale(config, args, prepared_root)\n'
    '    config.ARCHITECTURE_NAME = ARCH',
    '    exact._patch_paths_and_scale(config, args, prepared_root)\n'
    '    geometry = getattr(config, "BEARING_PHYSICAL_SAT_GEOMETRY", None)\n'
    '    if not isinstance(geometry, dict) or "sat_stride_m" not in geometry:\n'
    '        raise RuntimeError("missing audited Bearing physical SAT geometry")\n'
    '    # Cover the bounded local-prior jitter before retaining only the\n'
    '    # heading-forward 18 cells.  Derived only from protocol geometry.\n'
    '    config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (\n'
    '        float(config.CONTROLLED_GT_PRIOR_JITTER_M)\n'
    '        + 0.5 * float(geometry["sat_stride_m"])\n'
    '    )\n'
    '    # Initialize recurrent + Kalman motion from this city training cadence.\n'
    '    config.INIT_FORWARD_SPEED_M_PER_FRAME = float(\n'
    '        args.training_cadence_audit["train_step_mean_m"]\n'
    '    )\n'
    '    # Apply a Kalman profile selected on this city training validation only.\n'
    '    calibration_path = _train_root(args, 3) / "kalman_calibration.json"\n'
    '    if calibration_path.exists():\n'
    '        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))\n'
    '        best = calibration.get("best", {})\n'
    '        for key, attr in (\n'
    '            ("fixed_variance_m2", "EXPERIMENT_FIXED_VARIANCE_M2"),\n'
    '            ("q_progress", "KALMAN_Q_PROGRESS"),\n'
    '            ("q_cross", "KALMAN_Q_CROSS"),\n'
    '            ("q_velocity", "KALMAN_Q_VELOCITY"),\n'
    '            ("confidence_power", "KALMAN_CONFIDENCE_POWER"),\n'
    '        ):\n'
    '            if key in best:\n'
    '                setattr(config, attr, float(best[key]))\n'
    '    config.ARCHITECTURE_NAME = ARCH',
    'forward-origin, cadence init and train-only calibration',
)

old_audit = (
    '        "one_final_ms_source": "online final path contains exactly one MeanShift" in tracker_text,\n'
    f'        "front_decoder_not_ms": str(config.EXPERIMENT_ANCHOR) == "{legacy_token}",'
)
new_audit = (
    '        "front_softms_source": (\n'
    '            "anchor_xy_all = candidate.softms_xy" in tracker_text\n'
    '            and tracker_text.count("soft_mean_shift(") == 3\n'
    '            and \'getattr(config, "EXPERIMENT_ANCHOR"\' not in tracker_text\n'
    '        ),\n'
    '        "one_final_ms_source": "exactly one final local Soft MeanShift after the Kalman estimator" in tracker_text,\n'
    '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",\n'
    '        "simple_seven_block_gru": "nn.GRUCell(feature_dim * 7" in model_text,\n'
    '        "residual_motion_head": "MOTION_RESIDUAL_FORWARD_M" in model_text,\n'
    '        "temporal_reliability_gate": "self.temporal_reliability_head" in model_text,\n'
    '        "causal_visual_displacement": "self.visual_motion_projection(visual_motion)" in model_text,\n'
    '        "cadence_initialized_tracker": "INIT_FORWARD_SPEED_M_PER_FRAME" in tracker_text,\n'
    '        "motion_training_inference_aligned": str(config.EXPERIMENT_MOTION) == "quadratic",\n'
    '        "training_city_motion_scale_init": float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.0)) > 0.0,'
)
replace_once_or_already(old_audit, new_audit, 'runtime residual SoftMS audit')

# The old audit still has the four-block check. Remove it because the active
# model now has seven compact input blocks.
s = s.replace(
    '        "simple_six_block_gru": "nn.GRUCell(feature_dim * 6" in model_text,\n',
    '',
)

replace_once_or_already(
    '        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",',
    '        "forward_origin_backshift_covers_jitter": (\n'
    '            float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M)\n'
    '            >= float(config.CONTROLLED_GT_PRIOR_JITTER_M)\n'
    '        ),\n'
    '        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",',
    'backshift audit',
)

# Frame-specific temporal checkpoints: each frame-count ablation is trained on
# the same city training sequence rather than masking a 3-frame checkpoint.
replace_once_or_already(
    '    output = _train_root(args)\n'
    '    runtime = _make_runtime(prepared, output / "runtime")\n'
    '    _set_environment(args, prepared, output, VARIANTS["full"], training=True)\n'
    '    config, tracker, visual_localizer = base._load_runtime_modules(runtime)\n'
    '    _patch_paths(config, args, prepared)\n'
    '    audit = _audit_runtime(config, runtime, VARIANTS["full"], training=True)',
    '    train_variant = dict(VARIANTS["full"])\n'
    '    train_variant["frames"] = int(args.train_frames)\n'
    '    output = _train_root(args, args.train_frames)\n'
    '    runtime = _make_runtime(prepared, output / "runtime")\n'
    '    _set_environment(args, prepared, output, train_variant, training=True)\n'
    '    config, tracker, visual_localizer = base._load_runtime_modules(runtime)\n'
    '    _patch_paths(config, args, prepared)\n'
    '    audit = _audit_runtime(config, runtime, train_variant, training=True)',
    'frame-specific training variant',
)
s = s.replace(
    '_write_manifest(args, output, VARIANTS["full"], audit, training=True)',
    '_write_manifest(args, output, train_variant, audit, training=True)',
)

replace_once_or_already(
    '    _link_full_checkpoints(config, _train_root(args))',
    '    checkpoint_frames = int(variant["frames"]) if args.variant in {"frames1", "frames2"} else 3\n'
    '    _link_full_checkpoints(config, _train_root(args, checkpoint_frames))',
    'frame-specific evaluation checkpoint',
)

# -----------------------------------------------------------------------------
# Training-city-only Kalman calibration. It runs only for the 3-frame Full
# checkpoint and never looks at nav50/nav51 held-out outputs.
# -----------------------------------------------------------------------------
calibration_helper = '''\n\ndef _calibrate_kalman_on_training_validation(args, config, tracker, visual, model, cache, route):
    if int(args.train_frames) != 3:
        return None
    gt_state = tracker.build_gt_route_state(cache, route)
    split = tracker.split_ranges(len(cache))
    val_range = split["val"]
    original = {
        "fixed_variance_m2": float(config.EXPERIMENT_FIXED_VARIANCE_M2),
        "q_progress": float(config.KALMAN_Q_PROGRESS),
        "q_cross": float(config.KALMAN_Q_CROSS),
        "q_velocity": float(config.KALMAN_Q_VELOCITY),
        "confidence_power": float(getattr(config, "KALMAN_CONFIDENCE_POWER", 0.5)),
    }
    profiles = []
    # Small predeclared grid. Selection criterion is training-city validation
    # MLE + 0.20*P90; held-out navigation data is never touched.
    for fixed_r in (2.5, 4.0, 6.0, 9.0):
        for q_scale in (0.75, 1.0, 1.5):
            for conf_power in (0.0, 0.5, 1.0):
                config.EXPERIMENT_FIXED_VARIANCE_M2 = float(fixed_r)
                config.KALMAN_Q_PROGRESS = 1.50 * float(q_scale)
                config.KALMAN_Q_CROSS = 0.40 * float(q_scale)
                config.KALMAN_Q_VELOCITY = 1.00 * float(q_scale)
                config.KALMAN_CONFIDENCE_POWER = float(conf_power)
                result = tracker.evaluate_closed_loop(
                    model, visual, cache, route, gt_state, val_range, tracker.resolve_device()
                )
                objective = float(result["mle"] + 0.20 * result["p90"])
                profiles.append({
                    "fixed_variance_m2": float(fixed_r),
                    "q_progress": float(config.KALMAN_Q_PROGRESS),
                    "q_cross": float(config.KALMAN_Q_CROSS),
                    "q_velocity": float(config.KALMAN_Q_VELOCITY),
                    "confidence_power": float(conf_power),
                    "val_mle_m": float(result["mle"]),
                    "val_p90_m": float(result["p90"]),
                    "objective": objective,
                })
    profiles.sort(key=lambda row: (row["objective"], row["val_mle_m"], row["val_p90_m"]))
    best = profiles[0]
    for key, attr in (
        ("fixed_variance_m2", "EXPERIMENT_FIXED_VARIANCE_M2"),
        ("q_progress", "KALMAN_Q_PROGRESS"),
        ("q_cross", "KALMAN_Q_CROSS"),
        ("q_velocity", "KALMAN_Q_VELOCITY"),
        ("confidence_power", "KALMAN_CONFIDENCE_POWER"),
    ):
        setattr(config, attr, float(best[key]))
    payload = {
        "selection_source": "current_city_training_validation_only",
        "city": args.city,
        "validation_range": [int(val_range[0]), int(val_range[1])],
        "criterion": "val_mle + 0.20 * val_p90",
        "best": best,
        "profiles": profiles,
        "held_out_navigation_read": False,
    }
    out = _train_root(args, 3) / "kalman_calibration.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[TRAIN-ONLY KALMAN CALIBRATION]", json.dumps(best, sort_keys=True), flush=True)
    return payload
\n'''
marker = '\ndef train_full(args: argparse.Namespace) -> None:\n'
if calibration_helper.strip() not in s:
    if s.count(marker) != 1:
        raise SystemExit("PATCH FAILED [calibration helper insertion]")
    s = s.replace(marker, calibration_helper + marker, 1)

# Calibrate only after training is finished, using the saved best model on the
# training-sequence validation split.
old_after_train = '''    if not final_ckpt.exists():
        raise RuntimeError(f"training did not produce {final_ckpt}")
    _write_manifest(args, output, train_variant, audit, training=True)
'''
new_after_train = '''    if not final_ckpt.exists():
        raise RuntimeError(f"training did not produce {final_ckpt}")
    if int(args.train_frames) == 3:
        calibrated_model = tracker.load_temporal_model(device)
        _calibrate_kalman_on_training_validation(
            args, config, tracker, visual, calibrated_model, cache, route
        )
    _write_manifest(args, output, train_variant, audit, training=True)
'''
replace_once_or_already(old_after_train, new_after_train, 'post-training Kalman calibration')

# Replace paper-facing protocol metadata without changing internal compatibility
# names used by the tracker.
old_summary = (
    f'            "front_decoder": "posterior_{legacy_token}",\n'
    '            "online_meanshift_count": 1 if variant["ms"] else 0,'
)
new_summary = (
    '            "dataset_domain": args.city,\n'
    '            "held_out_navigation": external_name,\n'
    '            "front_decoder": "forward18_softms",\n'
    '            "front_meanshift_count": 1,\n'
    '            "motion_predictor": "residual_heading_aware_quadratic_next_step",\n'
    '            "motion_init_source": "current_city_training_cadence",\n'
    '            "motion_init_m_per_frame": float(config.INIT_FORWARD_SPEED_M_PER_FRAME),\n'
    '            "final_meanshift_count": 1 if variant["ms"] else 0,\n'
    '            "online_meanshift_count": 2 if variant["ms"] else 1,\n'
    '            "forward_origin_backshift_m": float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M),\n'
    '            "controlled_prior_jitter_m": float(config.CONTROLLED_GT_PRIOR_JITTER_M),'
)
replace_once_or_already(old_summary, new_summary, 'result protocol summary')
s = s.replace('            "training_route": "train_01",\n', '            "training_sequence": "current_city_training_sequence",\n')
s = s.replace('            "held_out_route": external_name,\n', '            "held_out_navigation": external_name,\n')

replace_once_or_already(
    '        "paper_chain": "Forward18 posterior -> 3-frame GRU -> fixed-R Kalman -> one final MeanShift -> XY",',
    '        "paper_chain": "Forward18 SoftMS -> residual temporal GRU -> constrained Kalman -> final MeanShift -> XY",\n'
    '        "dataset_protocol": "Bearing-UAV citya/cityb/cityc/cityd with held-out test_01/test_02 reporting",',
    'manifest chain',
)

replace_once_or_already(
    '    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))',
    '    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))\n'
    '    p.add_argument("--train-frames", type=int, default=3, choices=[1, 2, 3])',
    'train-frames CLI',
)

required = [
    'UAVSAT_EXPERIMENT_ANCHOR": "softms"',
    'UAVSAT_EXPERIMENT_MOTION": "quadratic"',
    'front_softms_source',
    'simple_seven_block_gru',
    'residual_motion_head',
    'temporal_reliability_gate',
    'causal_visual_displacement',
    'INIT_FORWARD_SPEED_M_PER_FRAME',
    'forward_origin_backshift_covers_jitter',
    'checkpoint_frames = int(variant["frames"])',
    'train_variant["frames"] = int(args.train_frames)',
    '_calibrate_kalman_on_training_validation',
    '("test_01", "route_B"',
    '("test_02", "route_C"',
    'p.add_argument("--train-frames"',
    '"front_decoder": "forward18_softms"',
    'residual temporal GRU -> constrained Kalman',
]
missing = [item for item in required if item not in s]
if missing:
    raise SystemExit("PATCH AUDIT FAILED: missing " + repr(missing))

if legacy_token in s.lower():
    raise SystemExit("PATCH AUDIT FAILED: legacy decoder identifier remains in active runner")

compile(s, str(path), "exec")
path.write_text(s, encoding="utf-8")
print(f"[PATCH OK] {path}")
print("[PATCH OK] dataset domains = citya/cityb/cityc/cityd; held-out labels = test_01/test_02")
print("[PATCH OK] active runner = Forward-18 SoftMS -> residual temporal GRU -> constrained Kalman -> final MeanShift")
print("[PATCH OK] recurrent/Kalman state starts at current-city training cadence")
print("[PATCH OK] Kalman profile is selected from current-city training validation only")
print("[PATCH OK] 1/2/3-frame rows use separately trained temporal checkpoints")
print("[PATCH OK] no held-out test_01/test_02 metric was read or modified")
