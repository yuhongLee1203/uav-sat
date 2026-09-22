#!/usr/bin/env python3
"""Run the existing Bearing temporal ablation with route-reference inference.

This wrapper changes only the reference protocol. Training still uses Route-A
labels for supervision, but inference/acquisition does not read the current
frame GT coordinate or a GT-derived progress cap. All component ablations use
the same protocol.
"""
from __future__ import annotations

import os
from pathlib import Path

import bearing_iclr_ablation as ab

_original_set_environment = ab._set_environment


def _route_reference_environment(args, prepared_root: Path, output: Path, variant: dict, *, training: bool):
    _original_set_environment(args, prepared_root, output, variant, training=training)
    # Must be set after the canonical helper because that helper historically
    # hard-coded controlled_gt_jitter. Runtime modules are loaded only after
    # this function returns, so config.py sees route_reference from import time.
    os.environ["UAVSAT_REFERENCE_PROTOCOL"] = "route_reference"
    os.environ["UAVSAT_ROUTE_REFERENCE_HYPOTHESES"] = os.environ.get(
        "BEARING_ROUTE_REFERENCE_HYPOTHESES", "13"
    )
    os.environ["UAVSAT_ROUTE_REFERENCE_BANK_RADIUS_M"] = os.environ.get(
        "BEARING_ROUTE_REFERENCE_BANK_RADIUS_M", "60.0"
    )


def _route_reference_audit(config, runtime: Path, variant: dict, training: bool) -> dict:
    tracker_text = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    model_text = (runtime / "visual_model.py").read_text(encoding="utf-8")
    checks = {
        "forward_18": int(config.FORWARD_SEARCH_CANDIDATE_COUNT) == 18,
        "base_geometry_6x6": int(config.ACQ_LOCAL_GRID_SIZE) == 6,
        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",
        "one_final_ms_source": "exactly one final local Soft MeanShift after the Kalman estimator" in tracker_text,
        "simple_seven_block_gru": "nn.GRUCell(feature_dim * 7" in model_text,
        "residual_motion_head": "MOTION_RESIDUAL_FORWARD_M" in model_text,
        "temporal_reliability_gate": "self.temporal_reliability_head" in model_text,
        "protocol_route_reference": str(config.REFERENCE_PROTOCOL) == "route_reference",
        "no_current_frame_gt_inference": bool(config.NO_GT_INFERENCE),
        "route_reference_only": bool(config.ROUTE_REFERENCE_ONLY),
        "frame_count": int(config.EXPERIMENT_FRAME_COUNT) == int(variant["frames"]),
        "gru_flag": bool(config.EXPERIMENT_DISABLE_GRU) == bool(variant["disable_gru"]),
        "kalman_flag": str(config.EXPERIMENT_KALMAN) == str(variant["kalman"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"[ROUTE-REF-AUDIT] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    if failed:
        raise RuntimeError("route-reference architecture audit failed: " + ", ".join(failed))
    return checks


# Patch the shared module object so every helper imported afterwards sees the
# same no-GT reference contract.
ab._set_environment = _route_reference_environment
ab._audit_runtime = _route_reference_audit

# Re-export the public helpers used by calibration/evaluation wrappers.
build_parser = ab.build_parser
train_full = ab.train_full
evaluate = ab.evaluate
rebind = getattr(ab, "rebind", None)


if __name__ == "__main__":
    ab.main()
