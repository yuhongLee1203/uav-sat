#!/usr/bin/env python3
"""Run Route-A validation from ONE global ABCD inference base.

Every candidate below is identical for A/B/C/D. City-local calibration is
explicitly erased before the search. Selection happens only after all four
Route-A validation pools are combined.
"""
from __future__ import annotations

import calibrate_final_output_kalman_city as cal
import global_abcd_runtime_profile as global_base


# Targeted search: 24 GPU profiles instead of re-running the old redundant
# 31-profile city-relative grid. Heading alpha is swept offline and costs no
# additional model inference.
PROFILES = [
    {"name": "r25_q010", "fixed_variance_m2": 25.0, "q_scale": 0.10},
    {"name": "r25_q020", "fixed_variance_m2": 25.0, "q_scale": 0.20},
    {"name": "r25_q030", "fixed_variance_m2": 25.0, "q_scale": 0.30},
    {"name": "r36_q010", "fixed_variance_m2": 36.0, "q_scale": 0.10},
    {"name": "r36_q020", "fixed_variance_m2": 36.0, "q_scale": 0.20},
    {"name": "r36_q030", "fixed_variance_m2": 36.0, "q_scale": 0.30},
    {"name": "r49_q010", "fixed_variance_m2": 49.0, "q_scale": 0.10},
    {"name": "r49_q020", "fixed_variance_m2": 49.0, "q_scale": 0.20},
    {"name": "r49_q030", "fixed_variance_m2": 49.0, "q_scale": 0.30},
    {"name": "r64_q010", "fixed_variance_m2": 64.0, "q_scale": 0.10},
    {"name": "r64_q020", "fixed_variance_m2": 64.0, "q_scale": 0.20},
    {"name": "r64_q030", "fixed_variance_m2": 64.0, "q_scale": 0.30},
    # Confidence-adaptive measurement-preserving variants. These are designed
    # to keep high-confidence visual successes inside LSR@15 while allowing the
    # filtered prior to repair low-confidence frames.
    {"name": "r36_q020_c045_g006", "fixed_variance_m2": 36.0, "q_scale": 0.20,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.06,
     "prior_blend_max": 0.10, "prior_blend_cutoff": 0.45},
    {"name": "r36_q020_c050_g010", "fixed_variance_m2": 36.0, "q_scale": 0.20,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.10,
     "prior_blend_max": 0.15, "prior_blend_cutoff": 0.50},
    {"name": "r36_q025_c055_g014", "fixed_variance_m2": 36.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.14,
     "prior_blend_max": 0.20, "prior_blend_cutoff": 0.55},
    {"name": "r36_q030_c060_g018", "fixed_variance_m2": 36.0, "q_scale": 0.30,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.18,
     "prior_blend_max": 0.28, "prior_blend_cutoff": 0.60},
    {"name": "r49_q015_c045_g006", "fixed_variance_m2": 49.0, "q_scale": 0.15,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.06,
     "prior_blend_max": 0.10, "prior_blend_cutoff": 0.45},
    {"name": "r49_q020_c050_g010", "fixed_variance_m2": 49.0, "q_scale": 0.20,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.10,
     "prior_blend_max": 0.15, "prior_blend_cutoff": 0.50},
    {"name": "r49_q025_c055_g014", "fixed_variance_m2": 49.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.14,
     "prior_blend_max": 0.20, "prior_blend_cutoff": 0.55},
    {"name": "r49_q030_c060_g018", "fixed_variance_m2": 49.0, "q_scale": 0.30,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.18,
     "prior_blend_max": 0.28, "prior_blend_cutoff": 0.60},
    {"name": "r64_q015_c045_g006", "fixed_variance_m2": 64.0, "q_scale": 0.15,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.06,
     "prior_blend_max": 0.10, "prior_blend_cutoff": 0.45},
    {"name": "r64_q020_c050_g010", "fixed_variance_m2": 64.0, "q_scale": 0.20,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.10,
     "prior_blend_max": 0.15, "prior_blend_cutoff": 0.50},
    {"name": "r64_q025_c055_g014", "fixed_variance_m2": 64.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.14,
     "prior_blend_max": 0.20, "prior_blend_cutoff": 0.55},
    {"name": "r64_q030_c060_g018", "fixed_variance_m2": 64.0, "q_scale": 0.30,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.18,
     "prior_blend_max": 0.28, "prior_blend_cutoff": 0.60},
]
cal.PROFILES = PROFILES
cal.HEADING_FUSION_ALPHAS = (
    0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
)


def _base_values(config):
    names = [
        "EXPERIMENT_FIXED_VARIANCE_M2",
        "KALMAN_Q_PROGRESS", "KALMAN_Q_CROSS", "KALMAN_Q_VELOCITY",
        "KALMAN_CONFIDENCE_POWER",
        "KALMAN_PRIOR_BLEND_BASE", "KALMAN_PRIOR_BLEND_LOWCONF_GAIN",
        "KALMAN_PRIOR_BLEND_MAX", "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF",
        "KALMAN_STEP_RELAX_CONFIDENCE", "KALMAN_STEP_RELAX_WIDTH",
        "KALMAN_STEP_VISUAL_SLACK_M",
    ]
    return {name: float(getattr(config, name)) for name in names if hasattr(config, name)}


def _apply_profile(config, base, profile):
    cal._restore(config, base)
    config.EXPERIMENT_KALMAN = "fixed"
    q_scale = float(profile.get("q_scale", 1.0))
    for key in ("KALMAN_Q_PROGRESS", "KALMAN_Q_CROSS", "KALMAN_Q_VELOCITY"):
        if key in base:
            setattr(config, key, float(base[key]) * q_scale)
    mapping = {
        "fixed_variance_m2": "EXPERIMENT_FIXED_VARIANCE_M2",
        "confidence_power": "KALMAN_CONFIDENCE_POWER",
        "prior_blend_base": "KALMAN_PRIOR_BLEND_BASE",
        "prior_blend_lowconf_gain": "KALMAN_PRIOR_BLEND_LOWCONF_GAIN",
        "prior_blend_max": "KALMAN_PRIOR_BLEND_MAX",
        "prior_blend_cutoff": "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF",
        "step_relax_confidence": "KALMAN_STEP_RELAX_CONFIDENCE",
        "step_relax_width": "KALMAN_STEP_RELAX_WIDTH",
        "step_visual_slack_m": "KALMAN_STEP_VISUAL_SLACK_M",
    }
    for src, dst in mapping.items():
        if src in profile:
            if not hasattr(config, dst):
                raise RuntimeError(f"runtime missing global calibration field {dst}")
            setattr(config, dst, float(profile[src]))

cal._base_values = _base_values
cal._apply_profile = _apply_profile

_original_patch_paths = cal.ab._patch_paths

def _global_patch_paths(config, args, prepared_root):
    _original_patch_paths(config, args, prepared_root)
    global_base.apply_global_base(config, args.suite_root)

cal.ab._patch_paths = _global_patch_paths

if __name__ == "__main__":
    cal.main()
