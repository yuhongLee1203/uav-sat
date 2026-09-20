#!/usr/bin/env python3
"""Paper-required inference/component ablations for the completed Bearing V5 suite.

This runner deliberately reuses the already trained full checkpoint.  Therefore
these rows are *inference/component ablations*, not retrained architectural
ablations.  The distinction is written into every output manifest/table.

Added experiments:
  - full36: score the complete 6x6 local bank instead of heading-guided 3x6
  - front_top1 / front_softms: visual anchor decoder sensitivity
  - no_heading_feedback: remove learned recurrent heading correction at inference
  - jitter{0,4,12,16}: controlled local-prior sensitivity

Existing bearing_iclr_ablation variants remain available:
  no_gru, no_kalman, no_ms, frames1, frames2, grid4/5/7/8, full.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import bearing_iclr_ablation as ab

EXTRA = {
    "full36": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                   forward_only=False, anchor="weighted_centroid", prior_jitter_m=8.0,
                   heading_feedback=True),
    "front_top1": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                       forward_only=True, anchor="top1", prior_jitter_m=8.0,
                       heading_feedback=True),
    "front_softms": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                         forward_only=True, anchor="softms", prior_jitter_m=8.0,
                         heading_feedback=True),
    "no_heading_feedback": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                                forward_only=True, anchor="weighted_centroid", prior_jitter_m=8.0,
                                heading_feedback=False),
    "jitter0": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                    forward_only=True, anchor="weighted_centroid", prior_jitter_m=0.0,
                    heading_feedback=True),
    "jitter4": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                    forward_only=True, anchor="weighted_centroid", prior_jitter_m=4.0,
                    heading_feedback=True),
    "jitter12": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                     forward_only=True, anchor="weighted_centroid", prior_jitter_m=12.0,
                     heading_feedback=True),
    "jitter16": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6,
                     forward_only=True, anchor="weighted_centroid", prior_jitter_m=16.0,
                     heading_feedback=True),
}

# Enrich every canonical variant with explicit values used by this wrapper.
for name, row in list(ab.VARIANTS.items()):
    row.setdefault("forward_only", True)
    row.setdefault("anchor", "weighted_centroid")
    row.setdefault("prior_jitter_m", 8.0)
    row.setdefault("heading_feedback", True)
for name, row in EXTRA.items():
    ab.VARIANTS[name] = row

_ORIG_SET_ENV = ab._set_environment
_ORIG_PATCH_PATHS = ab._patch_paths
_ORIG_MAKE_RUNTIME = ab._make_runtime


def _variant(args):
    return ab.VARIANTS[str(args.variant)]


def _set_environment(args, prepared_root, output, variant, *, training):
    _ORIG_SET_ENV(args, prepared_root, output, variant, training=training)
    os.environ["UAVSAT_EXPERIMENT_FORWARD_ONLY"] = "1" if variant.get("forward_only", True) else "0"
    os.environ["UAVSAT_EXPERIMENT_ANCHOR"] = str(variant.get("anchor", "weighted_centroid"))


def _patch_paths(config, args, prepared_root):
    _ORIG_PATCH_PATHS(config, args, prepared_root)
    v = _variant(args)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(v.get("prior_jitter_m", 8.0))
    if not bool(v.get("heading_feedback", True)):
        # Remove the learned recurrent heading correction while leaving the
        # waypoint route tangent available as the geometric route frame.
        config.HEADING_STATE_EMA_ALPHA = 0.0
        config.TURN_RATE_EMA_ALPHA = 0.0
        config.MAX_HEADING_DELTA_DEG_PER_FRAME = 0.0
        config.MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = 0.0


def _patch_top1_anchor(runtime: Path) -> None:
    path = runtime / "robust_tracker.py"
    s = path.read_text(encoding="utf-8")
    marker = "PAPER_TOP1_ANCHOR_PATCH"
    if marker in s:
        return
    pattern = re.compile(
        r'(?P<indent>\s*)if str\(getattr\(config, "EXPERIMENT_ANCHOR", "softms"\)\) == "weighted_centroid":\n'
        r'(?P=indent)    anchor_xy_all = \(posterior\.unsqueeze\(-1\) \* candidate\.centers\)\.sum\(dim=1\)\n'
        r'(?P=indent)else:\n'
        r'(?P=indent)    anchor_xy_all = candidate\.softms_xy\n'
    )
    m = pattern.search(s)
    if not m:
        raise RuntimeError("Could not patch top1 visual anchor in runtime")
    ind = m.group("indent")
    repl = (
        f'{ind}# PAPER_TOP1_ANCHOR_PATCH\n'
        f'{ind}_anchor_mode = str(getattr(config, "EXPERIMENT_ANCHOR", "softms"))\n'
        f'{ind}if _anchor_mode == "top1":\n'
        f'{ind}    _best = posterior.argmax(dim=1)\n'
        f'{ind}    _batch = torch.arange(candidate.centers.shape[0], device=candidate.centers.device)\n'
        f'{ind}    anchor_xy_all = candidate.centers[_batch, _best]\n'
        f'{ind}elif _anchor_mode == "weighted_centroid":\n'
        f'{ind}    anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)\n'
        f'{ind}else:\n'
        f'{ind}    anchor_xy_all = candidate.softms_xy\n'
    )
    s = s[:m.start()] + repl + s[m.end():]
    compile(s, str(path), "exec")
    path.write_text(s, encoding="utf-8")


def _make_runtime(prepared_root, runtime_root):
    runtime = _ORIG_MAKE_RUNTIME(prepared_root, runtime_root)
    _patch_top1_anchor(runtime)
    return runtime


def _audit_runtime(config, runtime, variant, training):
    tracker_text = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    model_text = (runtime / "visual_model.py").read_text(encoding="utf-8")
    expected_forward = bool(variant.get("forward_only", True))
    checks = {
        "base_geometry_6x6": int(config.ACQ_LOCAL_GRID_SIZE) == 6,
        "forward_policy": bool(config.FORWARD_ONLY_LOCAL_SEARCH) == expected_forward,
        "frame_count": int(config.EXPERIMENT_FRAME_COUNT) == int(variant["frames"]),
        "gru_flag": bool(config.EXPERIMENT_DISABLE_GRU) == bool(variant["disable_gru"]),
        "kalman_flag": str(config.EXPERIMENT_KALMAN) == str(variant["kalman"]),
        "anchor_mode": str(config.EXPERIMENT_ANCHOR) == str(variant.get("anchor", "weighted_centroid")),
        "controlled_gt_reference": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",
        "final_ms_source": "online final path contains exactly one MeanShift" in tracker_text,
        "v5_temporal_source": ("delta2_direct" in model_text or "TEMPORAL_DIRECT_STEP_FORWARD_M" in model_text),
        "top1_runtime_support": "PAPER_TOP1_ANCHOR_PATCH" in tracker_text,
    }
    failed = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(f"[PAPER-ABLATION-AUDIT] {k}: {'PASS' if ok else 'FAIL'}", flush=True)
    if failed:
        raise RuntimeError("paper ablation architecture audit failed: " + ", ".join(failed))
    return checks


ab._set_environment = _set_environment
ab._patch_paths = _patch_paths
ab._make_runtime = _make_runtime
ab._audit_runtime = _audit_runtime


def _postprocess(args) -> None:
    out = Path(args.suite_root).resolve() / args.city / "variants" / args.variant
    p = out / "bearing_v39_summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    v = _variant(args)
    for _, row in data.items():
        proto = row.setdefault("PaperAblationProtocol", {})
        proto.update({
            "ablation_type": "inference_component_ablation_reusing_full_checkpoint",
            "variant": args.variant,
            "base_candidate_geometry": "6x6",
            "forward_only": bool(v.get("forward_only", True)),
            "scored_candidates": 18 if v.get("forward_only", True) else 36,
            "front_anchor": str(v.get("anchor", "weighted_centroid")),
            "controlled_prior_jitter_m": float(v.get("prior_jitter_m", 8.0)),
            "heading_feedback": bool(v.get("heading_feedback", True)),
            "retrained_for_variant": False,
        })
    p.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def build_parser():
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
    args = build_parser().parse_args()
    # The jitter value used by the variant is explicit and independent of the
    # visual-retrieval training jitter argument.
    args.jitter_m = float(_variant(args).get("prior_jitter_m", args.jitter_m))
    ab.evaluate(args)
    _postprocess(args)
    print(f"[PAPER-ABLATION DONE] {args.city} {args.variant}", flush=True)


if __name__ == "__main__":
    main()
