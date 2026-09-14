#!/usr/bin/env python3
"""Run v39 on Bearing with TRAIN-only temporal cadence adaptation.

Architecture remains unchanged:
Weighted Centroid -> 3-frame Context-GRU -> fixed-R Kalman -> one final MS.

Only temporal motion/correction limits are adapted from TRAIN route step
statistics because Bearing-UAV observations are independent images rather than
video frames.  No test-route statistics are used to set these values.

This wrapper also fixes the validation split for short Bearing temporal episodes.
Canonical v39 used SPLIT_GUARD_FRAMES=16 for one long Route-A sequence. Applying
that same guard independently to short train_01/train_02/train_03 episodes can
collapse validation to a single frame. For Bearing we use exactly
TEMPORAL_WINDOW_FRAMES-1 guard frames (=2 for the 3-frame GRU), which prevents
train/validation temporal-window overlap without destroying the validation set.
"""
from __future__ import annotations

import json
from pathlib import Path

import bearing_runner as base

_ORIGINAL_PATCH = base._patch_bearing_paths
_ORIGINAL_AUDIT = base._audit_canonical


def _training_cadence(prepared_root: Path):
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    stats = exp["route_stats"]
    p50 = max(float(stats[name].get("actual_step_p50_m", 0.0)) for name in base.TRAIN_ROUTES)
    p90 = max(float(stats[name].get("actual_step_p90_m", 0.0)) for name in base.TRAIN_ROUTES)
    p95 = max(float(stats[name].get("actual_step_p95_m", 0.0)) for name in base.TRAIN_ROUTES)
    max_step = max(float(stats[name].get("actual_step_max_m", 0.0)) for name in base.TRAIN_ROUTES)
    return {
        "train_step_p50_m": p50,
        "train_step_p90_m": p90,
        "train_step_p95_m": p95,
        "train_step_max_m": max_step,
    }


def _split_preview(length: int, train_fraction: float, val_fraction: float, guard: int):
    train_end_raw = int(length * train_fraction)
    val_end_raw = int(length * (train_fraction + val_fraction))
    train_start = 0
    train_end = max(1, train_end_raw - guard)
    val_start = min(length - 1, train_end_raw + guard)
    val_end = max(
        min(length, val_end_raw - guard),
        min(length - 1, train_end_raw + guard) + 1,
    )
    return {
        "train": [int(train_start), int(train_end)],
        "val": [int(val_start), int(val_end)],
        "train_frames": int(max(0, train_end - train_start)),
        "val_frames": int(max(0, val_end - val_start)),
    }


def _patch_bearing_paths(config, args, prepared_root: Path) -> None:
    _ORIGINAL_PATCH(config, args, prepared_root)
    cadence = _training_cadence(prepared_root)

    # ------------------------------------------------------------------
    # Bearing short-episode validation split fix.
    # ------------------------------------------------------------------
    # Canonical guard=16 was designed for one long continuous Route A. When
    # train_01/02/03 are separate short episodes it can leave only one val frame.
    # With a 3-frame GRU, a 2-frame guard is sufficient to stop temporal-window
    # overlap between train and validation segments.
    config.SPLIT_GUARD_FRAMES = max(0, int(config.TEMPORAL_WINDOW_FRAMES) - 1)

    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    split_diagnostics = {}
    for name in base.TRAIN_ROUTES:
        length = int(exp["route_stats"][name]["frames"])
        split_diagnostics[name] = _split_preview(
            length,
            float(config.TRAIN_FRACTION),
            float(config.VAL_FRACTION),
            int(config.SPLIT_GUARD_FRAMES),
        )
        if split_diagnostics[name]["val_frames"] < 4:
            raise RuntimeError(
                "Bearing temporal validation split is still too small for %s: %s"
                % (name, split_diagnostics[name])
            )

    config.BEARING_SPLIT_ADAPTATION = {
        "split_guard_frames": int(config.SPLIT_GUARD_FRAMES),
        "reason": "3-frame temporal window => 2-frame train/validation separation",
        "routes": split_diagnostics,
    }
    print("[SPLIT] Bearing short-episode validation fix", flush=True)
    print(json.dumps(config.BEARING_SPLIT_ADAPTATION, indent=2), flush=True)

    # ------------------------------------------------------------------
    # TRAIN-only cadence adaptation.
    # ------------------------------------------------------------------
    # The old exact-v39 7 m final-step cap was appropriate for the original
    # continuous video, but it forced 99-100% of Bearing frames into step
    # limiting. Adapt the limits without changing any learned module or
    # estimator topology. Test-route cadence is never used here.
    p90 = max(cadence["train_step_p90_m"], 1.0)
    p95 = max(cadence["train_step_p95_m"], p90)

    config.MAX_FORWARD_SPEED_M_PER_FRAME = max(
        float(config.MAX_FORWARD_SPEED_M_PER_FRAME), min(24.0, 1.20 * p95)
    )
    config.MAX_POLYNOMIAL_STEP_M_PER_FRAME = max(
        float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME), min(24.0, 1.20 * p95)
    )
    config.KALMAN_FINAL_STEP_MAX_M = max(
        float(config.KALMAN_FINAL_STEP_MAX_M), min(20.0, 1.10 * p95)
    )

    config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = max(
        float(config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M),
        min(14.0, 0.80 * p90),
    )
    config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = max(
        float(config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M),
        min(8.0, 0.45 * p90),
    )
    config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M = max(
        float(config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M),
        min(10.0, 0.55 * p90),
    )

    config.BEARING_CADENCE_ADAPTATION = {
        **cadence,
        "max_forward_speed_m_per_frame": float(config.MAX_FORWARD_SPEED_M_PER_FRAME),
        "max_polynomial_step_m_per_frame": float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME),
        "kalman_final_step_max_m": float(config.KALMAN_FINAL_STEP_MAX_M),
        "kalman_max_measurement_innovation_progress_m": float(
            config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M
        ),
        "kalman_max_posterior_correction_progress_m": float(
            config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M
        ),
        "max_measurement_correction_parallel_m": float(
            config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M
        ),
        "source": "TRAIN routes only",
    }
    print("[CADENCE] Bearing TRAIN-only temporal adaptation", flush=True)
    print(json.dumps(config.BEARING_CADENCE_ADAPTATION, indent=2), flush=True)


def _audit_canonical(config, runtime: Path, args, prepared_root: Path) -> None:
    _ORIGINAL_AUDIT(config, runtime, args, prepared_root)
    audit_path = Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["bearing_temporal_cadence_adaptation"] = dict(config.BEARING_CADENCE_ADAPTATION)
    audit["bearing_validation_split_adaptation"] = dict(config.BEARING_SPLIT_ADAPTATION)
    audit["comparison_label"] = (
        "v39 architecture + Bearing TRAIN-cadence adaptation + short-episode split fix"
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


base._patch_bearing_paths = _patch_bearing_paths
base._audit_canonical = _audit_canonical

if __name__ == "__main__":
    base.main()
