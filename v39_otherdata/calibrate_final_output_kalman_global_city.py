#!/usr/bin/env python3
"""Run Route-A validation from ONE global ABCD inference base.

This wrapper deliberately erases city-local Kalman calibration after the normal
runtime setup, then lets calibrate_final_output_kalman_city evaluate a richer
set of global candidates.  Each city is still a separate temporal episode, but
candidate parameters are numerically identical across A/B/C/D and are selected
only after the four validation pools are combined.
"""
from __future__ import annotations

import calibrate_final_output_kalman_city as cal
import global_abcd_runtime_profile as global_base


EXTRA_PROFILES = [
    # Fine search around the region that already improved pooled MLE/R@1.
    {"name": "r36_q010", "fixed_variance_m2": 36.0, "q_scale": 0.10},
    {"name": "r36_q015", "fixed_variance_m2": 36.0, "q_scale": 0.15},
    {"name": "r49_q010", "fixed_variance_m2": 49.0, "q_scale": 0.10},
    {"name": "r49_q015", "fixed_variance_m2": 49.0, "q_scale": 0.15},
    {"name": "r64_q010", "fixed_variance_m2": 64.0, "q_scale": 0.10},
    {"name": "r64_q025", "fixed_variance_m2": 64.0, "q_scale": 0.25},
    # Measurement-preserving confidence gates.  candidate_x is blended from the
    # raw visual measurement toward the filtered posterior only at low confidence.
    {"name": "r36_q025_pg045_g006", "fixed_variance_m2": 36.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.06,
     "prior_blend_max": 0.10, "prior_blend_cutoff": 0.45},
    {"name": "r36_q025_pg050_g008", "fixed_variance_m2": 36.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.08,
     "prior_blend_max": 0.12, "prior_blend_cutoff": 0.50},
    {"name": "r36_q025_pg055_g012", "fixed_variance_m2": 36.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.12,
     "prior_blend_max": 0.18, "prior_blend_cutoff": 0.55},
    {"name": "r36_q025_pg065_g022", "fixed_variance_m2": 36.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.22,
     "prior_blend_max": 0.32, "prior_blend_cutoff": 0.65},
    {"name": "r49_q015_pg050_g008", "fixed_variance_m2": 49.0, "q_scale": 0.15,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.08,
     "prior_blend_max": 0.12, "prior_blend_cutoff": 0.50},
    {"name": "r49_q025_pg055_g012", "fixed_variance_m2": 49.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.12,
     "prior_blend_max": 0.18, "prior_blend_cutoff": 0.55},
    {"name": "r64_q015_pg050_g008", "fixed_variance_m2": 64.0, "q_scale": 0.15,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.08,
     "prior_blend_max": 0.12, "prior_blend_cutoff": 0.50},
    {"name": "r64_q025_pg060_g018", "fixed_variance_m2": 64.0, "q_scale": 0.25,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.18,
     "prior_blend_max": 0.26, "prior_blend_cutoff": 0.60},
]

# Keep old candidates plus new targeted candidates, with unique names.
_seen = set()
PROFILES = []
for _p in list(cal.PROFILES) + EXTRA_PROFILES:
    if _p["name"] not in _seen:
        PROFILES.append(dict(_p))
        _seen.add(_p["name"])
cal.PROFILES = PROFILES

# The existing alpha sweep remains cheap.  Heading fusion itself is now
# agreement-gated in heading_fusion_metrics.py.
cal.HEADING_FUSION_ALPHAS = (
    0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
)


# Extend the saved/restored base with every confidence-residual field we search.
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

# Erase per-city cadence/Kalman calibration after the normal path setup.
_original_patch_paths = cal.ab._patch_paths

def _global_patch_paths(config, args, prepared_root):
    _original_patch_paths(config, args, prepared_root)
    global_base.apply_global_base(config, args.suite_root)

cal.ab._patch_paths = _global_patch_paths


if __name__ == "__main__":
    cal.main()
