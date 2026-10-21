#!/usr/bin/env python3
"""Evaluate one ablation variant with a frozen Route-A-selected Kalman profile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import bearing_iclr_ablation as ab


def apply_profile(config, profile):
    q_scale = float(profile.get("q_scale", 1.0))
    for key in ("KALMAN_Q_PROGRESS", "KALMAN_Q_CROSS", "KALMAN_Q_VELOCITY"):
        if hasattr(config, key):
            setattr(config, key, float(getattr(config, key)) * q_scale)
    mapping = {
        "fixed_variance_m2": "EXPERIMENT_FIXED_VARIANCE_M2",
        "confidence_power": "KALMAN_CONFIDENCE_POWER",
        "step_relax_confidence": "KALMAN_STEP_RELAX_CONFIDENCE",
        "step_relax_width": "KALMAN_STEP_RELAX_WIDTH",
        "step_visual_slack_m": "KALMAN_STEP_VISUAL_SLACK_M",
    }
    for src, dst in mapping.items():
        if src in profile:
            if not hasattr(config, dst):
                raise RuntimeError(f"runtime missing frozen profile field {dst}")
            setattr(config, dst, float(profile[src]))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile-json", required=True)
    known, remaining = p.parse_known_args()
    payload = json.loads(Path(known.profile_json).read_text(encoding="utf-8"))
    if payload.get("held_out_navigation_read") is not False:
        raise RuntimeError("refusing a profile that was not selected without held-out navigation")
    profile = payload["best"]["profile"]

    os.environ["MS_REFERENCE_PRIOR_WEIGHT"] = "0.0"
    os.environ["MS_KF_PRIOR_WEIGHT"] = "1.50"
    os.environ["MS_REFERENCE_SIGMA_M"] = "4.0"
    os.environ["MS_KF_SIGMA_M"] = "4.0"

    original_patch = ab._patch_paths

    def patched(config, args, prepared_root):
        original_patch(config, args, prepared_root)
        apply_profile(config, profile)
        print("[FROZEN TRAINVAL KALMAN PROFILE]", json.dumps(profile, sort_keys=True), flush=True)

    ab._patch_paths = patched
    args = ab.build_parser().parse_args(remaining)
    if getattr(args, "command", None) not in (None, "eval") and getattr(args, "mode", None) != "eval":
        raise RuntimeError("this wrapper is evaluation-only")
    ab.evaluate(args)


if __name__ == "__main__":
    main()
