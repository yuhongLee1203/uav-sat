#!/usr/bin/env python3
"""Run Bearing-UAV with the original v39 estimator settings.

The Bearing adapter is allowed to change only dataset I/O / route bookkeeping.
The inference model and estimator match the saved v39 weighted-centroid main
configuration:

  Weighted Centroid -> 3-frame Context-GRU -> velocity motion
  -> fixed-R constrained Kalman -> one final 5x5 MeanShift -> Final Position

No Bearing cadence statistics are used to change motion or Kalman limits.
The only dataset-specific training bookkeeping change is SPLIT_GUARD_FRAMES=2,
because Bearing train routes are short independent pseudo-flight episodes; this
prevents the 3-frame temporal window from leaking across train/validation while
leaving a non-degenerate validation sequence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import bearing_runner as base

ORIGINAL_SET_ENV = base._set_canonical_environment
ORIGINAL_PATCH_PATHS = base._patch_bearing_paths
ORIGINAL_AUDIT = base._audit_canonical

EXACT_ARCH = "V39_GRU_Kalman_MS"


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
        # Match the saved v39 weighted_centroid_main result exactly.
        "MS_ENABLED": "1",
        "MS_GRID_SIZE": "5",
        "MS_BANDWIDTH_M": "7.0",
    }
    os.environ.update(env)
    return output, checkpoints, feature_cache


def _patch_exact_paths(config, args, prepared_root: Path) -> None:
    # Bearing-only file/coordinate adapter from the existing runner.
    ORIGINAL_PATCH_PATHS(config, args, prepared_root)

    # Dataset bookkeeping only. A 3-frame window needs a two-frame separation;
    # the canonical 16-frame guard leaves only one validation frame on these
    # short external pseudo-flight episodes.
    config.SPLIT_GUARD_FRAMES = 2

    # Restore / lock ALL temporal-estimator values to canonical v39.  Do not use
    # any Bearing route statistics to alter these numbers.
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
    # Preserve the existing static architecture/loss audit first.
    ORIGINAL_AUDIT(config, runtime, args, prepared_root)

    actual = _exact_model_values(config)
    expected = {
        "reference_protocol": "controlled_gt_jitter",
        "visual_decoder": "weighted_centroid",
        "frame_count": 3,
        "motion": "velocity",
        "kalman": "fixed",
        "forward_only": True,
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

    # Verify the short-episode split is healthy. This is dataset bookkeeping,
    # not an inference/model parameter.
    split_rows = {}
    for route_name in base.TRAIN_ROUTES:
        manifest = prepared_root / "routes" / route_name / "manifest.csv"
        with manifest.open("r", encoding="utf-8") as handle:
            length = max(0, sum(1 for _ in handle) - 1)
        split = __import__("robust_tracker").split_ranges(length)
        train_range, val_range = split["train"], split["val"]
        val_frames = max(0, int(val_range[1]) - int(val_range[0]))
        if val_frames < 4:
            raise RuntimeError(
                f"Bearing validation still too short for {route_name}: {val_frames} frames"
            )
        split_rows[route_name] = {
            "length": length,
            "train": list(train_range),
            "val": list(val_range),
            "val_frames": val_frames,
        }

    audit_path = Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["architecture"] = EXACT_ARCH
    audit["exact_v39_model_inference"] = True
    audit["bearing_cadence_adaptation"] = False
    audit["exact_v39_values"] = actual
    audit["dataset_only_adaptations"] = {
        "bearing_coordinate_and_image_adapter": True,
        "pseudo_flight_sequence_construction": True,
        "split_guard_frames": 2,
        "split_reason": "3-frame temporal window on short external episodes",
        "route_splits": split_rows,
    }
    audit["comparison_label"] = (
        "original v39 estimator/settings on Bearing data; no cadence adaptation"
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("[EXACT-V39] model/inference settings: PASS", flush=True)
    print(json.dumps(actual, indent=2), flush=True)
    print("[EXACT-V39] Bearing cadence adaptation: DISABLED", flush=True)


# Monkeypatch only the external runner hooks. Canonical runtime source itself is
# still rebuilt from v39_DirectFinalMS/base_src + patch_direct_finalms.py.
base.FINAL_ARCH = EXACT_ARCH
base._set_canonical_environment = _set_exact_environment
base._patch_bearing_paths = _patch_exact_paths
base._audit_canonical = _audit_exact


if __name__ == "__main__":
    base.main()
