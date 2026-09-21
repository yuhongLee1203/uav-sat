#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import bearing_paper_ablation as paper

ab = paper.ab

# Core V5-Restore keeps the architecture unchanged and restores the pre-Smooth-V1
# estimator dynamics that previously gave Full > context truncations on CityA.
CORE_V5 = {
    "corev5_full": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                        forward_only=True, anchor="softms", prior_jitter_m=8.0,
                        heading_feedback=True),
    "corev5_no_gru": dict(frames=3, disable_gru=True, kalman="fixed", ms=True, grid=6,
                          forward_only=True, anchor="softms", prior_jitter_m=8.0,
                          heading_feedback=True),
    "corev5_no_kalman": dict(frames=3, disable_gru=False, kalman="none", ms=True, grid=6,
                             forward_only=True, anchor="softms", prior_jitter_m=8.0,
                             heading_feedback=True),
    "corev5_no_ms": dict(frames=3, disable_gru=False, kalman="fixed", ms=False, grid=6,
                         forward_only=True, anchor="softms", prior_jitter_m=8.0,
                         heading_feedback=True),
    "corev5_ctx1": dict(frames=1, disable_gru=False, kalman="fixed", ms=True, grid=6,
                        forward_only=True, anchor="softms", prior_jitter_m=8.0,
                        heading_feedback=True),
    "corev5_ctx2": dict(frames=2, disable_gru=False, kalman="fixed", ms=True, grid=6,
                        forward_only=True, anchor="softms", prior_jitter_m=8.0,
                        heading_feedback=True),
}
for name, row in CORE_V5.items():
    ab.VARIANTS[name] = row

_ORIG_TRAIN_ROOT = ab._train_root
_ORIG_VARIANT_ROOT = ab._variant_root
_ORIG_SET_ENV = ab._set_environment
_ORIG_PATCH_PATHS = ab._patch_paths
_ORIG_REUSE_VISUAL = ab._reuse_visual_checkpoint
_ORIG_LINK_FULL = ab._link_full_checkpoints

CURRENT_ARGS = None


def _train_root(args):
    if getattr(args, "core_v5_restore", False):
        return Path(args.suite_root).resolve() / args.city / "train_core_v5_restore"
    return _ORIG_TRAIN_ROOT(args)


def _variant_root(args):
    if str(getattr(args, "variant", "")).startswith("corev5_"):
        return Path(args.suite_root).resolve() / args.city / "variants_core_v5_restore" / args.variant
    return _ORIG_VARIANT_ROOT(args)


def _restore_env():
    # Pre-Smooth-V1 V5 estimator dynamics. These values are fixed before any
    # held-out evaluation and are not selected from nav50/nav51.
    values = {
        "UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME": "14.0",
        "UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME": "5.0",
        "UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2": "4.0",
        "UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME": "14.0",
        "UAVSAT_HEADING_STATE_EMA_ALPHA": "0.35",
        "UAVSAT_TURN_RATE_EMA_ALPHA": "0.30",
        "UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME": "5.0",
        "UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2": "5.0",
        "UAVSAT_LOSS_CROSS_MOTION_REG": "0.0",
        "UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M": "1.75",
        "UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME": "1.25",
        "UAVSAT_KALMAN_FINAL_STEP_MAX_M": "7.0",
        "UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M": "24.0",
        "UAVSAT_CORR_PARALLEL_M": "0.75",
        "UAVSAT_CORR_CROSS_M": "0.50",
    }
    os.environ.update(values)
    return values


def _set_environment(args, prepared_root, output, variant, *, training):
    _restore_env()
    _ORIG_SET_ENV(args, prepared_root, output, variant, training=training)
    # _ORIG_SET_ENV may write some generic experiment vars; restore estimator
    # knobs again so runtime config import always sees the V5 values.
    _restore_env()
    os.environ["UAVSAT_EXPERIMENT_FORWARD_ONLY"] = "1"
    os.environ["UAVSAT_EXPERIMENT_ANCHOR"] = "softms"


def _calibration_path(args):
    root = Path(args.suite_root).resolve() / args.city
    for p in (
        root / "train_frames3" / "kalman_calibration.json",
        root / "train_full" / "kalman_calibration.json",
        root / "kalman_calibration.json",
    ):
        if p.is_file():
            return p
    return None


def _finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _select_calibration(args):
    p = _calibration_path(args)
    if p is None:
        return None, {"source": "restored_v5_defaults", "held_out_used_for_selection": False}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("selection_source") != "current_city_training_validation_only":
        raise RuntimeError(f"Refusing non-training-validation calibration: {p}")
    profiles = []
    if isinstance(payload.get("best"), dict):
        profiles.append(dict(payload["best"]))
    profiles.extend(dict(x) for x in payload.get("profiles", []) if isinstance(x, dict))
    profiles = [x for x in profiles if _finite(x.get("val_mle_m"))]
    if not profiles:
        return None, {"source": str(p), "held_out_used_for_selection": False}
    # Predeclared validation-only objective: accuracy first, then P90.
    selected = min(profiles, key=lambda x: (float(x["val_mle_m"]), float(x.get("val_p90_m", 1e30))))
    nk = payload.get("validation_no_kalman_diagnostic", {})
    audit = {
        "source": str(p),
        "selection_source": payload.get("selection_source"),
        "validation_range": payload.get("validation_range"),
        "selected": selected,
        "validation_no_kalman_diagnostic": nk,
        "held_out_used_for_selection": False,
    }
    return selected, audit


CAL_ATTRS = {
    "fixed_variance_m2": ("EXPERIMENT_FIXED_VARIANCE_M2", "KALMAN_FIXED_VARIANCE_M2"),
    "q_progress": ("KALMAN_Q_PROGRESS",),
    "q_cross": ("KALMAN_Q_CROSS",),
    "q_velocity": ("KALMAN_Q_VELOCITY",),
    "confidence_power": ("KALMAN_CONFIDENCE_POWER",),
    "temporal_3frame_scale": ("TEMPORAL_ADAPTER_3FRAME_SCALE", "TEMPORAL_3FRAME_SCALE"),
    "delta2_scale": ("TEMPORAL_DELTA2_SCALE", "TEMPORAL_DIRECT_DELTA2_SCALE"),
    "prior_blend_base": ("KALMAN_PRIOR_BLEND_BASE",),
    "prior_blend_lowconf_gain": ("KALMAN_PRIOR_BLEND_LOWCONF_GAIN",),
    "prior_blend_max": ("KALMAN_PRIOR_BLEND_MAX",),
    "prior_blend_cutoff": ("KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF",),
    "step_relax_confidence": ("KALMAN_STEP_RELAX_CONFIDENCE",),
    "step_relax_width": ("KALMAN_STEP_RELAX_WIDTH",),
    "step_visual_slack_m": ("KALMAN_STEP_VISUAL_SLACK_M",),
}


def _patch_paths(config, args, prepared_root):
    _ORIG_PATCH_PATHS(config, args, prepared_root)
    # Explicitly restore V5 estimator-side attributes after the generic Smooth
    # config adapter has run.
    fixed = {
        "MAX_FORWARD_SPEED_M_PER_FRAME": 14.0,
        "MAX_CROSS_SPEED_M_PER_FRAME": 5.0,
        "MAX_CROSS_ACCEL_M_PER_FRAME2": 4.0,
        "MAX_POLYNOMIAL_STEP_M_PER_FRAME": 14.0,
        "HEADING_STATE_EMA_ALPHA": 0.35,
        "TURN_RATE_EMA_ALPHA": 0.30,
        "MAX_HEADING_DELTA_DEG_PER_FRAME": 5.0,
        "MAX_TURN_RATE_DELTA_DEG_PER_FRAME2": 5.0,
        "LOSS_CROSS_MOTION_REG": 0.0,
        "KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M": 1.75,
        "KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME": 1.25,
        "KALMAN_FINAL_STEP_MAX_M": 7.0,
        "ROUTE_FRAME_SMOOTH_RADIUS_M": 24.0,
        "MAX_MEASUREMENT_CORRECTION_PARALLEL_M": 0.75,
        "MAX_MEASUREMENT_CORRECTION_CROSS_M": 0.50,
    }
    applied = {}
    for k, v in fixed.items():
        if hasattr(config, k):
            setattr(config, k, v); applied[k] = v
    selected, audit = _select_calibration(args)
    if selected:
        for key, attrs in CAL_ATTRS.items():
            if key not in selected or not _finite(selected[key]):
                continue
            v = float(selected[key])
            for attr in attrs:
                if hasattr(config, attr):
                    setattr(config, attr, v); applied[attr] = v
    args.core_v5_restore_audit = {**audit, "restored_v5_config": applied}


def _reuse_visual_checkpoint(config, prepared_root):
    # Reuse the already-trained visual model; only temporal/estimator state is retrained.
    args = CURRENT_ARGS
    root = Path(args.suite_root).resolve() / args.city
    dest = Path(config.VISUAL_CHECKPOINT)
    for p in (
        root / "train_frames3" / "checkpoints" / dest.name,
        root / "train_full" / "checkpoints" / dest.name,
    ):
        if p.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink(): dest.unlink()
            dest.symlink_to(p.resolve())
            print(f"[CORE-V5 REUSE VISUAL] {p}", flush=True)
            return
    _ORIG_REUSE_VISUAL(config, prepared_root)


def _link_full_checkpoints(config, _ignored):
    args = CURRENT_ARGS
    srcdir = _train_root(args) / "checkpoints"
    for dest in (Path(config.VISUAL_CHECKPOINT), Path(config.TEMPORAL_CHECKPOINT)):
        src = srcdir / dest.name
        if not src.exists():
            raise FileNotFoundError(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink(): dest.unlink()
        dest.symlink_to(src.resolve())
    print(f"[CORE-V5 CKPT] reuse restored 3f Full checkpoint: {srcdir}", flush=True)


ab._train_root = _train_root
ab._variant_root = _variant_root
ab._set_environment = _set_environment
ab._patch_paths = _patch_paths
ab._reuse_visual_checkpoint = _reuse_visual_checkpoint
ab._link_full_checkpoints = _link_full_checkpoints


def _write_protocol(args, out: Path):
    payload = {
        "city": args.city,
        "variant": getattr(args, "variant", "full"),
        "architecture": "Forward-18 SoftMS -> 3-frame recurrent GRU -> constrained Kalman -> final MeanShift -> XY",
        "restored_profile": "pre-Smooth-V1 V5 estimator dynamics",
        "same_3frame_checkpoint_for_component_and_context_ablation": True,
        "held_out_used_for_selection": False,
        "validation_calibration": getattr(args, "core_v5_restore_audit", None),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "core_v5_restore_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--suite-root", required=True)
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--variant", default="corev5_full", choices=sorted(CORE_V5))
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--visual-epochs", type=int, default=30)
    p.add_argument("--temporal-epochs", type=int, default=100)
    p.add_argument("--epochs-per-route", type=int, default=100)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--jitter-m", type=float, default=8.0)
    p.add_argument("--step-m", type=float, default=None)
    p.add_argument("--max-sample-distance-m", type=float, default=15.0)
    p.add_argument("--heading-weight-px-per-deg", type=float, default=0.0)
    p.add_argument("--ms-bandwidth-m", type=float, default=7.0)
    p.add_argument("--latency-warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=2033)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--reprepare", action="store_true")
    return p


def main():
    global CURRENT_ARGS
    args = parser().parse_args()
    args.core_v5_restore = True
    CURRENT_ARGS = args
    if args.mode == "train":
        # Train one restored 3-frame Full temporal checkpoint per city.
        args.variant = "corev5_full"
        ab.train_full(args)
        _write_protocol(args, _train_root(args))
    else:
        args.jitter_m = float(CORE_V5[args.variant]["prior_jitter_m"])
        ab.evaluate(args)
        out = _variant_root(args)
        _write_protocol(args, out)
        sp = out / "bearing_v39_summary.json"
        data = json.loads(sp.read_text(encoding="utf-8"))
        for row in data.values():
            row["CoreV5RestoreProtocol"] = {
                "variant": args.variant,
                "same_restored_3frame_full_checkpoint": True,
                "held_out_used_for_selection": False,
                "validation_calibration": getattr(args, "core_v5_restore_audit", None),
            }
        sp.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
        print(f"[CORE-V5 RESTORE DONE] {args.city} {args.variant}", flush=True)

if __name__ == "__main__":
    main()
