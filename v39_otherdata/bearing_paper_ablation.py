#!/usr/bin/env python3
"""Paper-facing Bearing V5 ablations.

Temporal-context rows (1f/2f) use separately trained temporal checkpoints.
All other rows are one-factor inference/component or sensitivity ablations of
one completed Full checkpoint. The formal Full front decoder is SoftMS; all
non-decoder rows preserve that decoder so only one factor changes at a time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import bearing_iclr_ablation as ab

EXTRA = {
    "full36": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                   forward_only=False, anchor="softms", prior_jitter_m=8.0,
                   heading_feedback=True),
    "front_top1": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                       forward_only=True, anchor="top1", prior_jitter_m=8.0,
                       heading_feedback=True),
    "front_weighted": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                           forward_only=True, anchor="weighted_centroid", prior_jitter_m=8.0,
                           heading_feedback=True),
    "no_heading_feedback": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                                forward_only=True, anchor="softms", prior_jitter_m=8.0,
                                heading_feedback=False),
    "jitter0": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                    forward_only=True, anchor="softms", prior_jitter_m=0.0,
                    heading_feedback=True),
    "jitter4": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                    forward_only=True, anchor="softms", prior_jitter_m=4.0,
                    heading_feedback=True),
    "jitter12": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                     forward_only=True, anchor="softms", prior_jitter_m=12.0,
                     heading_feedback=True),
    "jitter16": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                     forward_only=True, anchor="softms", prior_jitter_m=16.0,
                     heading_feedback=True),
}

for _, row in list(ab.VARIANTS.items()):
    row.setdefault("forward_only", True)
    row.setdefault("anchor", "softms")
    row.setdefault("prior_jitter_m", 8.0)
    row.setdefault("heading_feedback", True)
for name, row in EXTRA.items():
    ab.VARIANTS[name] = row

_ORIG_SET_ENV = ab._set_environment
_ORIG_PATCH_PATHS = ab._patch_paths
_ORIG_MAKE_RUNTIME = ab._make_runtime


def variant(args):
    return ab.VARIANTS[str(args.variant)]


def set_environment(args, prepared_root, output, v, *, training):
    _ORIG_SET_ENV(args, prepared_root, output, v, training=training)
    os.environ["UAVSAT_EXPERIMENT_FORWARD_ONLY"] = "1" if v.get("forward_only", True) else "0"
    os.environ["UAVSAT_EXPERIMENT_ANCHOR"] = str(v.get("anchor", "softms"))


def patch_paths(config, args, prepared_root):
    _ORIG_PATCH_PATHS(config, args, prepared_root)
    v = variant(args)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(v.get("prior_jitter_m", 8.0))
    if not bool(v.get("heading_feedback", True)):
        # Disable only the learned recurrent angular residual. The predefined
        # route tangent remains because it is part of the route-coordinate model.
        config.HEADING_STATE_EMA_ALPHA = 0.0
        config.TURN_RATE_EMA_ALPHA = 0.0
        config.MAX_HEADING_DELTA_DEG_PER_FRAME = 0.0
        config.MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = 0.0


def patch_front_decoder(runtime: Path) -> None:
    """Re-introduce a controlled decoder switch only for this ablation runtime."""
    p = runtime / "robust_tracker.py"
    s = p.read_text(encoding="utf-8")
    if "PAPER_FRONT_DECODER_SWITCH" in s:
        return

    old_anchor = '''    # Forward-18 decoder is always Soft MeanShift.
    # No alternate centroid decoder exists in this runtime.
    anchor_xy_all = candidate.softms_xy
'''
    new_anchor = '''    # PAPER_FRONT_DECODER_SWITCH: ablation-only controlled switch.
    _paper_anchor = str(getattr(config, "EXPERIMENT_ANCHOR", "softms"))
    if _paper_anchor == "top1":
        _paper_best = posterior.argmax(dim=1)
        _paper_batch = torch.arange(
            candidate.centers.shape[0], device=candidate.centers.device
        )
        anchor_xy_all = candidate.centers[_paper_batch, _paper_best]
    elif _paper_anchor == "weighted_centroid":
        anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)
    elif _paper_anchor == "softms":
        anchor_xy_all = candidate.softms_xy
    else:
        raise ValueError("unknown PAPER front decoder: %s" % _paper_anchor)
'''
    if s.count(old_anchor) != 1:
        raise RuntimeError(f"front decoder anchor patch matches={s.count(old_anchor)}")
    s = s.replace(old_anchor, new_anchor, 1)

    old_variance = '''        variance_points = softms_modes_all[h]
        variance_weights = softms_mode_weights_all[h]
'''
    new_variance = '''        if _paper_anchor == "softms":
            variance_points = softms_modes_all[h]
            variance_weights = softms_mode_weights_all[h]
        else:
            # Top-1 / weighted-centroid decoder uncertainty is measured in the
            # original candidate space using the same posterior.
            variance_points = candidate.centers[h]
            variance_weights = posterior[h]
'''
    if s.count(old_variance) != 1:
        raise RuntimeError(f"front decoder variance patch matches={s.count(old_variance)}")
    s = s.replace(old_variance, new_variance, 1)
    compile(s, str(p), "exec")
    p.write_text(s, encoding="utf-8")


def make_runtime(prepared_root, runtime_root):
    runtime = _ORIG_MAKE_RUNTIME(prepared_root, runtime_root)
    patch_front_decoder(runtime)
    return runtime


def audit_runtime(config, runtime, v, training):
    tracker = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    model = (runtime / "visual_model.py").read_text(encoding="utf-8")
    checks = {
        "base_geometry_6x6": int(config.ACQ_LOCAL_GRID_SIZE) == 6,
        "forward_policy": bool(config.FORWARD_ONLY_LOCAL_SEARCH) == bool(v.get("forward_only", True)),
        "frame_count": int(config.EXPERIMENT_FRAME_COUNT) == int(v["frames"]),
        "gru_flag": bool(config.EXPERIMENT_DISABLE_GRU) == bool(v["disable_gru"]),
        "kalman_flag": str(config.EXPERIMENT_KALMAN) == str(v["kalman"]),
        "front_decoder": str(config.EXPERIMENT_ANCHOR) == str(v.get("anchor", "softms")),
        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",
        "controlled_gt_reference_enabled": not bool(config.NO_GT_INFERENCE),
        "ablation_decoder_switch": "PAPER_FRONT_DECODER_SWITCH" in tracker,
        "v5_temporal_source": ("delta2_direct" in model or "TEMPORAL_DIRECT_STEP_FORWARD_M" in model),
        "final_ms_source": "exactly one final local Soft MeanShift after the Kalman estimator" in tracker,
    }
    failed = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(f"[PAPER-ABLATION-AUDIT] {k}: {'PASS' if ok else 'FAIL'}", flush=True)
    if failed:
        raise RuntimeError("paper ablation audit failed: " + ", ".join(failed))
    return checks


ab._set_environment = set_environment
ab._patch_paths = patch_paths
ab._make_runtime = make_runtime
ab._audit_runtime = audit_runtime


def postprocess(args):
    p = Path(args.suite_root).resolve() / args.city / "variants" / args.variant / "bearing_v39_summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    v = variant(args)
    retrained = args.variant in {"frames1", "frames2"}
    for row in data.values():
        row.setdefault("PaperAblationProtocol", {}).update({
            "variant": args.variant,
            "ablation_type": "retrained_temporal_context_ablation" if retrained else "inference_component_or_sensitivity_ablation",
            "retrained_for_variant": retrained,
            "base_candidate_geometry": "6x6",
            "forward_only": bool(v.get("forward_only", True)),
            "scored_candidates": 18 if v.get("forward_only", True) else 36,
            "front_anchor": str(v.get("anchor", "softms")),
            "controlled_prior_jitter_m": float(v.get("prior_jitter_m", 8.0)),
            "heading_feedback": bool(v.get("heading_feedback", True)),
            "held_out_used_for_selection": False,
        })
    p.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--variant", required=True, choices=sorted(ab.VARIANTS))
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
    args = parser().parse_args()
    args.jitter_m = float(variant(args).get("prior_jitter_m", args.jitter_m))
    ab.evaluate(args)
    postprocess(args)
    print(f"[PAPER-ABLATION DONE] {args.city} {args.variant}", flush=True)

if __name__ == "__main__": main()
