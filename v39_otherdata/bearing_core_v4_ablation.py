#!/usr/bin/env python3
"""Core V4 paper ablations for the completed Bearing V5 Smooth suite.

Goals:
  * Core components: w/o GRU, w/o Kalman, w/o final MeanShift, Full.
  * Temporal context: 1f/2f/3f using the SAME trained 3-frame Full checkpoint.
    This restores the original context-truncation ablation protocol rather than
    comparing separately retrained 1f/2f models.
  * Kalman/temporal runtime hyperparameters are selected only from the existing
    train_01 validation calibration JSON. Held-out nav50/nav51 results are never
    used for profile selection.

Main architecture is unchanged:
  Forward-18 SoftMS -> 3-frame recurrent GRU -> constrained Kalman
  -> final MeanShift -> XY.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import bearing_paper_ablation as paper

ab = paper.ab

CORE_V4 = {
    "corev4_full": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                        forward_only=True, anchor="softms", prior_jitter_m=8.0,
                        heading_feedback=True),
    "corev4_no_gru": dict(frames=3, disable_gru=True, kalman="fixed", ms=True, grid=6,
                          forward_only=True, anchor="softms", prior_jitter_m=8.0,
                          heading_feedback=True),
    "corev4_no_kalman": dict(frames=3, disable_gru=False, kalman="none", ms=True, grid=6,
                             forward_only=True, anchor="softms", prior_jitter_m=8.0,
                             heading_feedback=True),
    "corev4_no_ms": dict(frames=3, disable_gru=False, kalman="fixed", ms=False, grid=6,
                         forward_only=True, anchor="softms", prior_jitter_m=8.0,
                         heading_feedback=True),
    # Same 3-frame trained checkpoint; only available explicit context is truncated.
    "corev4_ctx1": dict(frames=1, disable_gru=False, kalman="fixed", ms=True, grid=6,
                        forward_only=True, anchor="softms", prior_jitter_m=8.0,
                        heading_feedback=True),
    "corev4_ctx2": dict(frames=2, disable_gru=False, kalman="fixed", ms=True, grid=6,
                        forward_only=True, anchor="softms", prior_jitter_m=8.0,
                        heading_feedback=True),
}
for name, row in CORE_V4.items():
    ab.VARIANTS[name] = row

_ORIG_VARIANT_ROOT = ab._variant_root
_ORIG_LINK = ab._link_full_checkpoints
_ORIG_PATCH_PATHS = ab._patch_paths


def _variant_root(args):
    if str(args.variant).startswith("corev4_"):
        return Path(args.suite_root).resolve() / args.city / "variants_core_v4" / args.variant
    return _ORIG_VARIANT_ROOT(args)


def _find_full_checkpoint_dir(args, config) -> Path:
    root = Path(args.suite_root).resolve() / args.city
    candidates = [
        root / "train_frames3" / "checkpoints",
        root / "train_full" / "checkpoints",
    ]
    vname = Path(config.VISUAL_CHECKPOINT).name
    tname = Path(config.TEMPORAL_CHECKPOINT).name
    for d in candidates:
        if (d / vname).exists() and (d / tname).exists():
            return d
    raise FileNotFoundError(
        "No completed 3-frame Full checkpoint directory. Checked: "
        + ", ".join(str(x) for x in candidates)
    )


def _link_same_full_checkpoint(config, _ignored_train_root):
    # ab.evaluate has args only indirectly, so infer suite/city from checkpoint path
    # after _patch_paths has populated config. CURRENT_ARGS is set in main().
    args = CURRENT_ARGS
    srcdir = _find_full_checkpoint_dir(args, config)
    for dest in (Path(config.VISUAL_CHECKPOINT), Path(config.TEMPORAL_CHECKPOINT)):
        src = srcdir / dest.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())
    print(f"[CORE-V4 CKPT] all variants reuse 3f Full checkpoint: {srcdir}", flush=True)


def _calibration_path(args) -> Path:
    root = Path(args.suite_root).resolve() / args.city
    candidates = [
        root / "train_frames3" / "kalman_calibration.json",
        root / "train_full" / "kalman_calibration.json",
        root / "variants" / "full" / "kalman_calibration.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Smooth uploader may have retained calibration at the city root in a copied suite.
    p = root / "kalman_calibration.json"
    if p.is_file():
        return p
    raise FileNotFoundError("Missing train-validation Kalman calibration for " + args.city)


def _finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _select_validation_profile(args):
    path = _calibration_path(args)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_source") != "current_city_training_validation_only":
        raise RuntimeError(f"Refusing non-training-validation calibration: {path}")
    candidates = []
    if isinstance(payload.get("best"), dict):
        candidates.append(dict(payload["best"]))
    candidates += [dict(x) for x in payload.get("profiles", []) if isinstance(x, dict)]
    candidates = [x for x in candidates if _finite(x.get("val_mle_m"))]
    if not candidates:
        raise RuntimeError(f"No finite validation profiles in {path}")
    # This is selected only on train_01 validation: minimize localization error.
    selected = min(candidates, key=lambda x: (float(x["val_mle_m"]), float(x.get("val_p90_m", 1e30))))
    nk = payload.get("validation_no_kalman_diagnostic", {})
    audit = {
        "path": str(path),
        "selection_source": payload.get("selection_source"),
        "validation_range": payload.get("validation_range"),
        "selection_rule": "minimum train-validation MLE; P90 tie-break",
        "selected": selected,
        "validation_no_kalman_diagnostic": nk,
        "validation_kalman_gain_m": (
            float(nk["mle_m"]) - float(selected["val_mle_m"])
            if _finite(nk.get("mle_m")) else None
        ),
        "held_out_used_for_selection": False,
    }
    return selected, audit


CALIBRATION_ATTRS = {
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


def _apply_profile(config, selected):
    applied = {}
    for key, attrs in CALIBRATION_ATTRS.items():
        if key not in selected or not _finite(selected[key]):
            continue
        value = float(selected[key])
        for attr in attrs:
            if hasattr(config, attr):
                setattr(config, attr, value)
                applied[attr] = value
    return applied


def _patch_paths(config, args, prepared_root):
    _ORIG_PATCH_PATHS(config, args, prepared_root)
    selected, audit = _select_validation_profile(args)
    applied = _apply_profile(config, selected)
    args.core_v4_calibration_audit = {**audit, "applied_config": applied}
    print(
        "[CORE-V4 CAL] city=%s val_mle=%.6f no_kalman_val=%s gain=%s" % (
            args.city,
            float(selected["val_mle_m"]),
            str(audit["validation_no_kalman_diagnostic"].get("mle_m")),
            str(audit["validation_kalman_gain_m"]),
        ),
        flush=True,
    )


ab._variant_root = _variant_root
ab._link_full_checkpoints = _link_same_full_checkpoint
ab._patch_paths = _patch_paths

CURRENT_ARGS = None


def _postprocess(args):
    out = _variant_root(args)
    sp = out / "bearing_v39_summary.json"
    data = json.loads(sp.read_text(encoding="utf-8"))
    for row in data.values():
        row["CoreV4Protocol"] = {
            "variant": args.variant,
            "ablation_type": (
                "same_full_checkpoint_temporal_context_truncation"
                if args.variant in {"corev4_ctx1", "corev4_ctx2", "corev4_full"}
                else "same_full_checkpoint_single_component_ablation"
            ),
            "same_trained_3frame_full_checkpoint": True,
            "kalman_profile_selected_on": "train_01_validation_only",
            "held_out_used_for_selection": False,
            "calibration": args.core_v4_calibration_audit,
        }
    sp.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    manifest = {
        "city": args.city,
        "variant": args.variant,
        "architecture": "Forward18 SoftMS -> recurrent GRU -> constrained Kalman -> final MeanShift -> XY",
        "same_trained_3frame_full_checkpoint": True,
        "temporal_context_protocol": "context truncation at inference; no 1f/2f retraining",
        "kalman_profile_selection": args.core_v4_calibration_audit,
        "integrity": "No held-out nav50/nav51 metric is used for hyperparameter/profile selection.",
    }
    (out / "core_v4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--variant", required=True, choices=sorted(CORE_V4))
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
    CURRENT_ARGS = args
    # paper.variant() reads from ab.VARIANTS and therefore supports CORE_V4 names.
    args.jitter_m = float(CORE_V4[args.variant].get("prior_jitter_m", args.jitter_m))
    ab.evaluate(args)
    _postprocess(args)
    print(f"[CORE-V4 DONE] {args.city} {args.variant}", flush=True)


if __name__ == "__main__":
    main()
