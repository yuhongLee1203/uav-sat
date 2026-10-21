#!/usr/bin/env python3
"""Run Route-A validation from ONE global ABCD inference base.

Every candidate below is identical for A/B/C/D. City-local calibration is
explicitly erased before the search. Selection happens only after all four
Route-A validation pools are combined.
"""
from __future__ import annotations

import os

import calibrate_final_output_kalman_city as cal
import global_abcd_runtime_profile as global_base


# Validation-only search over the existing estimator's inference parameters.
# Besides Q/R, sweep the already-existing confidence residual gate and the
# final MeanShift Kalman-prior strength.  The latter was fixed at 1.50 in the
# old search, even though it is the stage that turns the Kalman state into the
# reported XY coordinate.
PROFILES = [
    {"name": "r25_q005_ms150_s4", "fixed_variance_m2": 25.0, "q_scale": 0.05,
     "ms_kf_prior_weight": 1.50, "ms_kf_sigma_m": 4.0},
    {"name": "r25_q010_ms150_s4", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "ms_kf_prior_weight": 1.50, "ms_kf_sigma_m": 4.0},
    {"name": "r36_q010_ms150_s4", "fixed_variance_m2": 36.0, "q_scale": 0.10,
     "ms_kf_prior_weight": 1.50, "ms_kf_sigma_m": 4.0},
    {"name": "r49_q020_ms150_s4", "fixed_variance_m2": 49.0, "q_scale": 0.20,
     "ms_kf_prior_weight": 1.50, "ms_kf_sigma_m": 4.0},
    {"name": "r25_q010_ms300_s4", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "ms_kf_prior_weight": 3.00, "ms_kf_sigma_m": 4.0},
    {"name": "r25_q010_ms600_s4", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "ms_kf_prior_weight": 6.00, "ms_kf_sigma_m": 4.0},
    {"name": "r25_q010_ms600_s3", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "ms_kf_prior_weight": 6.00, "ms_kf_sigma_m": 3.0},
    {"name": "r25_q010_ms1200_s3", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "ms_kf_prior_weight": 12.0, "ms_kf_sigma_m": 3.0},
    # Conservative low-confidence gates: preserve visual measurements on
    # confident frames and use the motion prior only when confidence drops.
    {"name": "gate04_r25_q010_ms300", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.04,
     "prior_blend_max": 0.06, "prior_blend_cutoff": 0.45,
     "step_relax_confidence": 0.35, "step_relax_width": 0.06,
     "step_visual_slack_m": 8.0, "ms_kf_prior_weight": 3.0, "ms_kf_sigma_m": 4.0},
    {"name": "gate08_r25_q010_ms300", "fixed_variance_m2": 25.0, "q_scale": 0.10,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.08,
     "prior_blend_max": 0.12, "prior_blend_cutoff": 0.50,
     "step_relax_confidence": 0.40, "step_relax_width": 0.08,
     "step_visual_slack_m": 6.0, "ms_kf_prior_weight": 3.0, "ms_kf_sigma_m": 4.0},
    {"name": "gate12_r36_q015_ms600", "fixed_variance_m2": 36.0, "q_scale": 0.15,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.12,
     "prior_blend_max": 0.18, "prior_blend_cutoff": 0.55,
     "step_relax_confidence": 0.45, "step_relax_width": 0.08,
     "step_visual_slack_m": 5.0, "ms_kf_prior_weight": 6.0, "ms_kf_sigma_m": 4.0},
    {"name": "gate18_r49_q020_ms600", "fixed_variance_m2": 49.0, "q_scale": 0.20,
     "prior_blend_base": 0.0, "prior_blend_lowconf_gain": 0.18,
     "prior_blend_max": 0.28, "prior_blend_cutoff": 0.60,
     "step_relax_confidence": 0.50, "step_relax_width": 0.10,
     "step_visual_slack_m": 4.0, "ms_kf_prior_weight": 6.0, "ms_kf_sigma_m": 4.0},
]
cal.PROFILES = PROFILES
# Fast mode must be defined against this wrapper's actual global profile list,
# not the city-local defaults that were replaced above.
cal.FAST_PROFILE_NAMES = {
    "r25_q005_ms150_s4",
    "r25_q010_ms150_s4",
    "r36_q010_ms150_s4",
    "r25_q010_ms300_s4",
    "r25_q010_ms600_s4",
    "r25_q010_ms600_s3",
    "gate04_r25_q010_ms300",
    "gate12_r36_q015_ms600",
}
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
    # patch_direct_finalms reads these values at inference time.  Keeping them
    # in the frozen profile makes validation and held-out evaluation identical.
    if "ms_kf_prior_weight" in profile:
        os.environ["MS_KF_PRIOR_WEIGHT"] = str(float(profile["ms_kf_prior_weight"]))
    if "ms_kf_sigma_m" in profile:
        os.environ["MS_KF_SIGMA_M"] = str(float(profile["ms_kf_sigma_m"]))

cal._base_values = _base_values
cal._apply_profile = _apply_profile

_original_patch_paths = cal.ab._patch_paths

def _global_patch_paths(config, args, prepared_root):
    _original_patch_paths(config, args, prepared_root)
    global_base.apply_global_base(config, args.suite_root)

cal.ab._patch_paths = _global_patch_paths

if __name__ == "__main__":
    cal.main()
