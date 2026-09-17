#!/usr/bin/env python3
"""Paper-facing Bearing-UAV evaluation wrapper for v39.

This runner keeps the selected v39 architecture/training recipe but removes every
current-frame GT dependency from test-time inference:

  * local SAT search reference = causal predicted route progress
  * no GT progress/step cap in motion or Kalman
  * final MeanShift reference = Kalman's own estimated route progress

Known at inference: ordered waypoint coordinates, previous model state, current
UAV image, cached satellite imagery/features.  Ground-truth coordinates are used
only after inference to compute metrics.

Results are written to ``v39_output_paper_fair`` so they can never be confused
with the older controlled-GT-prior diagnostic experiment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import bearing_runner_exact_v39 as exact

SELECTION_VERSION = "soft_sequence_v13_multicity_auto_train_fullroute"
OUTPUT_NAME = "v39_output_paper_fair"

# Same multicity data contract as the selected official-route experiment.
exact.EXPECTED_SELECTION_VERSION = SELECTION_VERSION
exact.base.TRAIN_ROUTES = ("train_01",)
exact.base.TEST_ROUTES = ("test_01", "test_02")


def _set_environment(args, prepared_root: Path):
    output = prepared_root / OUTPUT_NAME
    checkpoints = output / "checkpoints"
    feature_cache = prepared_root / "feature_cache_paper_fair"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))

    os.environ.update({
        "UAVSAT_DEVICE": f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu",
        "UAVSAT_OUTPUT_DIR": str(output),
        "UAVSAT_CHECKPOINT_DIR": str(checkpoints),
        "UAVSAT_FEATURE_CACHE_DIR": str(feature_cache),
        "UAVSAT_DATA_ROOT": str(prepared_root),
        "UAVSAT_BACKBONE": str(args.backbone),
        "UAVSAT_ARCHITECTURE_NAME": exact.ARCH,
        # CRITICAL paper-facing setting: no current-frame GT at inference.
        "UAVSAT_REFERENCE_PROTOCOL": "route_reference",
        "UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid",
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


def _patch_final_ms_reference(runtime: Path) -> None:
    """Use the model/Kalman-estimated progress, never true sample progress."""
    path = runtime / "robust_tracker.py"
    text = path.read_text(encoding="utf-8")
    old = '''        # Keep the original v39 predefined frame-reference prior unchanged.\n        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()\n        frame_reference_xy = (\n            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)\n        )\n'''
    new = '''        # PAPER-FAIR Bearing evaluation: the final-MS spatial reference must be\n        # causal and model-derived.  Use the current Kalman-estimated progress\n        # projected onto the known waypoint centerline.  Do NOT read gt_state or\n        # cache.gt_xy here; GT is evaluation-only.\n        reference_progress_s = float(kalman_se[0])\n        frame_reference_xy = np.asarray(\n            route.xy_from_se(reference_progress_s, 0.0), dtype=np.float64\n        )\n        frame_reference_xy_t = torch.tensor(\n            frame_reference_xy[None, :], dtype=torch.float32, device=device\n        )\n'''
    if text.count(old) != 1:
        raise RuntimeError(
            "paper-fair final-MS patch did not match canonical runtime exactly once"
        )
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")
    print("[PAPER-FAIR] final-MS reference = Kalman-estimated route progress: PASS", flush=True)


def _audit(config, runtime: Path, args, prepared_root: Path) -> None:
    errors = []
    if str(config.REFERENCE_PROTOCOL) != "route_reference":
        errors.append(f"REFERENCE_PROTOCOL={config.REFERENCE_PROTOCOL!r}")
    if not bool(config.ROUTE_REFERENCE_ONLY):
        errors.append("ROUTE_REFERENCE_ONLY is false")
    if not bool(config.NO_GT_INFERENCE):
        errors.append("NO_GT_INFERENCE is false")
    if bool(config.SCHEDULED_ROUTE_REFERENCE):
        errors.append("scheduled frame-index route reference is enabled")
    if int(config.TEMPORAL_EPOCHS) != int(exact.CANONICAL_TEMPORAL_EPOCHS):
        errors.append(f"TEMPORAL_EPOCHS={config.TEMPORAL_EPOCHS}")
    if int(config.GRID_SIZE) != 6 or int(config.CANDIDATE_COUNT) != 36:
        errors.append(f"candidate geometry={config.GRID_SIZE}x{config.GRID_SIZE}/{config.CANDIDATE_COUNT}")

    source = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    forbidden = 'reference_progress_s = float(gt_state["se"][index, 0])'
    required = 'reference_progress_s = float(kalman_se[0])'
    if forbidden in source:
        errors.append("final MS still reads true GT progress")
    if required not in source:
        errors.append("model-derived final-MS reference patch missing")

    expected_roots = [
        prepared_root / "routes" / "train_01",
        prepared_root / "routes" / "test_01",
        prepared_root / "routes" / "test_02",
    ]
    actual_roots = [Path(p).resolve() for p in config.ROUTE_ROOTS]
    if actual_roots != [p.resolve() for p in expected_roots]:
        errors.append("Route A/B/C mapping mismatch")

    if errors:
        raise RuntimeError("PAPER-FAIR audit failed:\n- " + "\n- ".join(errors))

    audit = {
        "paper_fair": True,
        "architecture": exact.ARCH,
        "reference_protocol": "route_reference",
        "no_gt_inference": True,
        "uses_current_frame_gt_coordinate_at_inference": False,
        "uses_true_route_progress_at_inference": False,
        "uses_gt_motion_or_progress_cap_at_inference": False,
        "known_at_inference": [
            "ordered waypoint coordinates",
            "causal predicted route progress",
            "previous temporal/Kalman state",
            "current UAV image",
            "satellite imagery/features",
        ],
        "metric_gt_usage": "evaluation only after prediction",
        "final_ms_reference": "route centerline at Kalman-estimated progress",
        "train_routes": ["train_01"],
        "test_routes": ["test_01", "test_02"],
        "prepared_selection_version": SELECTION_VERSION,
    }
    audit_path = Path(config.OUTPUT_DIR) / "paper_fair_inference_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("[PAPER-FAIR] NO-GT inference audit: PASS", flush=True)
    print(json.dumps(audit, indent=2), flush=True)


# _train_and_infer resolves these symbols from bearing_runner_exact_v39 globals.
exact._set_environment = _set_environment
exact._patch_final_ms_reference = _patch_final_ms_reference
exact._audit = _audit
exact.base._audit_canonical = _audit
exact.base.train_and_infer = exact._train_and_infer


if __name__ == "__main__":
    exact.base.main()
