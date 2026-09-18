#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: patch_temporal_fusion_v2.py <robust_tracker.py> <config.py>")

tracker_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
tracker = tracker_path.read_text(encoding="utf-8")
config = config_path.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# 1) Final MS must refine the temporal/Kalman estimate, not read current-frame GT
#    as a direct spatial prior.  Keep the controlled GT+jitter protocol used by
#    the front local-search experiment unchanged; only remove the *second* direct
#    current-frame GT pull inside the post-Kalman final MeanShift.
# -----------------------------------------------------------------------------
old_ms = '''            d2_reference = (\n                ms_centers - frame_reference_xy_t[:, None, :]\n            ).square().sum(dim=2)\n\n            sigma_kalman = max(float(env.get("MS_KF_SIGMA_M", "4.0")), 1e-3)\n            sigma_reference = max(float(env.get("MS_REFERENCE_SIGMA_M", "4.0")), 1e-3)\n            weight_kalman = float(env.get("MS_KF_PRIOR_WEIGHT", "1.50"))\n            weight_reference = float(env.get("MS_REFERENCE_PRIOR_WEIGHT", "2.50"))\n\n            combined_log_probability = (\n                visual_log_probability\n                - weight_kalman * d2_kalman / (2.0 * sigma_kalman ** 2)\n                - weight_reference * d2_reference / (2.0 * sigma_reference ** 2)\n            )\n'''
new_ms = '''            # Temporal-fusion v2: final refinement is conditioned only on the\n            # persistent Kalman estimate plus current visual evidence.  Do not\n            # inject current-frame GT/reference as a second spatial prior here.\n            sigma_kalman = max(float(env.get("MS_KF_SIGMA_M", "4.0")), 1e-3)\n            weight_kalman = float(env.get("MS_KF_PRIOR_WEIGHT", "1.50"))\n\n            combined_log_probability = (\n                visual_log_probability\n                - weight_kalman * d2_kalman / (2.0 * sigma_kalman ** 2)\n            )\n'''
if tracker.count(old_ms) != 1:
    raise SystemExit(f"ERROR: final-MS GT-prior block count={tracker.count(old_ms)}")
tracker = tracker.replace(old_ms, new_ms, 1)

old_def = 'summary["MS_Definition"] = "exactly one final local Soft MeanShift after the original v39 Kalman estimator"'
new_def = 'summary["MS_Definition"] = "one final local Soft MeanShift using visual likelihood plus Kalman spatial prior only; no direct current-frame GT/reference prior"'
if tracker.count(old_def) != 1:
    raise SystemExit(f"ERROR: MS definition count={tracker.count(old_def)}")
tracker = tracker.replace(old_def, new_def, 1)

# -----------------------------------------------------------------------------
# 2) Make the learned temporal motion conservative.  Previous runs show that the
#    absolute GRU velocity can over-steer an already strong local visual anchor.
#    Strengthen velocity supervision, reduce frame-to-frame injection, and bound
#    the learned measurement residual more tightly.  These are model changes,
#    not result post-processing.
# -----------------------------------------------------------------------------
replacements = [
    (
        'LOSS_VELOCITY = 0.25',
        'LOSS_VELOCITY = float(__import__("os").environ.get("UAVSAT_LOSS_VELOCITY", "1.0"))',
    ),
    (
        'MOTION_VELOCITY_EMA_ALPHA = 0.55',
        'MOTION_VELOCITY_EMA_ALPHA = float(__import__("os").environ.get("UAVSAT_MOTION_VELOCITY_EMA_ALPHA", "0.30"))',
    ),
    (
        'MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME = 2.0',
        'MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME = float(__import__("os").environ.get("UAVSAT_MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME", "1.0"))',
    ),
    (
        'MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 4.0',
        'MAX_MEASUREMENT_CORRECTION_PARALLEL_M = float(__import__("os").environ.get("UAVSAT_MAX_MEASUREMENT_CORRECTION_PARALLEL_M", "2.0"))',
    ),
    (
        'MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0',
        'MAX_MEASUREMENT_CORRECTION_CROSS_M = float(__import__("os").environ.get("UAVSAT_MAX_MEASUREMENT_CORRECTION_CROSS_M", "2.0"))',
    ),
    (
        'EARLY_STOP_MIN_DELTA = 0.05',
        'EARLY_STOP_MIN_DELTA = float(__import__("os").environ.get("UAVSAT_EARLY_STOP_MIN_DELTA", "0.02"))',
    ),
]
for old, new in replacements:
    if config.count(old) != 1:
        raise SystemExit(f"ERROR: config token count for {old!r} = {config.count(old)}")
    config = config.replace(old, new, 1)

# Static safety checks.
if 'weight_reference * d2_reference' in tracker:
    raise SystemExit("ERROR: direct final-MS reference prior still remains")
if 'LOSS_VELOCITY = 0.25' in config:
    raise SystemExit("ERROR: old velocity-loss weight still remains")

compile(tracker, str(tracker_path), "exec")
compile(config, str(config_path), "exec")
tracker_path.write_text(tracker, encoding="utf-8")
config_path.write_text(config, encoding="utf-8")
print("[OK] temporal-fusion v2: Forward18-ready, conservative GRU motion, Kalman-only final-MS prior")
