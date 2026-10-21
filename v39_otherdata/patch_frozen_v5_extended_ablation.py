#!/usr/bin/env python3
"""Add fair search-geometry and visual-decoder evaluations to frozen V5.

Apply this after patch_bearing_iclr_main_alignment.py.  The two added variants
reuse the trained Full checkpoint; only the candidate geometry or front visual
aggregation changes during held-out evaluation.
"""
from pathlib import Path
import sys


path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str, label: str) -> None:
    global s
    if new in s:
        return
    if s.count(old) != 1:
        raise SystemExit(f"PATCH FAILED [{label}]: found {s.count(old)} targets")
    s = s.replace(old, new, 1)


once(
    '    "grid8": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=8),\n}',
    '    "grid8": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=8),\n'
    '    # Same trained Full model; evaluation-only geometry/decoder controls.\n'
    '    "search_full6x6": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6, forward_only=False, decoder="softms"),\n'
    '    "decoder_weighted": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6, forward_only=False, decoder="weighted_centroid"),\n'
    '}',
    "variants",
)

once(
    '        "UAVSAT_EXPERIMENT_ANCHOR": "softms",',
    '        "UAVSAT_EXPERIMENT_ANCHOR": str(variant.get("decoder", "softms")),',
    "decoder environment",
)
once(
    '        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1",',
    '        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1" if variant.get("forward_only", True) else "0",',
    "search environment",
)

# Frozen V5 intentionally hard-wires SoftMS in the generated runtime.  Restore
# the old *evaluation-only* switch for decoder_weighted.  MeanShift may still be
# computed while constructing CandidateBatch, but its output and uncertainty
# are not used by the weighted path; aggregation latency is measured separately.
helper = r"""

def _enable_weighted_decoder(runtime: Path) -> None:
    tracker_path = runtime / "robust_tracker.py"
    text = tracker_path.read_text(encoding="utf-8")
    fixed = '''    # Forward-18 decoder is always Soft MeanShift.
    # No alternate centroid decoder exists in this runtime.
    anchor_xy_all = candidate.softms_xy
'''
    switched = '''    # Evaluation-only visual decoder ablation.
    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "weighted_centroid":
        anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)
    else:
        anchor_xy_all = candidate.softms_xy
'''
    if fixed not in text and switched not in text:
        raise RuntimeError("weighted decoder patch target missing")
    text = text.replace(fixed, switched, 1)
    # The direct-final-MS patch also hard-wires mode-space uncertainty.  Restore
    # the corresponding posterior-space branch so this is a coherent decoder.
    fixed_u = '''    _, _, softms_modes_all, _, softms_mode_weights_all, _ = soft_mean_shift(
        candidate.raw_logits,
        candidate.centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
'''
    switched_u = '''    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "softms":
        _, _, softms_modes_all, _, softms_mode_weights_all, _ = soft_mean_shift(
            candidate.raw_logits,
            candidate.centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )
'''
    text = text.replace(fixed_u, switched_u, 1)
    fixed_v = '''        variance_points = softms_modes_all[h]
        variance_weights = softms_mode_weights_all[h]
'''
    switched_v = '''        if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "softms":
            variance_points = softms_modes_all[h]
            variance_weights = softms_mode_weights_all[h]
        else:
            variance_points = candidate.centers[h]
            variance_weights = posterior[h]
'''
    text = text.replace(fixed_v, switched_v, 1)
    compile(text, str(tracker_path), "exec")
    tracker_path.write_text(text, encoding="utf-8")
"""
marker = "\ndef evaluate(args: argparse.Namespace) -> None:\n"
if "def _enable_weighted_decoder" not in s:
    if s.count(marker) != 1:
        raise SystemExit("PATCH FAILED [helper insertion]")
    s = s.replace(marker, helper + marker, 1)

once(
    '    runtime = _make_runtime(prepared, output / "runtime")\n'
    '    _set_environment(args, prepared, output, variant, training=False)',
    '    runtime = _make_runtime(prepared, output / "runtime")\n'
    '    if variant.get("decoder") == "weighted_centroid":\n'
    '        _enable_weighted_decoder(runtime)\n'
    '    _set_environment(args, prepared, output, variant, training=False)',
    "runtime decoder activation",
)

once(
    '        "forward_18": int(config.FORWARD_SEARCH_CANDIDATE_COUNT) == 18,',
    '        "candidate_search": (\n'
    '            int(config.FORWARD_SEARCH_CANDIDATE_COUNT) == 18\n'
    '            and bool(config.FORWARD_ONLY_LOCAL_SEARCH) == bool(variant.get("forward_only", True))\n'
    '        ),',
    "search audit",
)
once(
    '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",',
    '        "front_decoder_selected": str(config.EXPERIMENT_ANCHOR) == str(variant.get("decoder", "softms")),',
    "decoder audit",
)
once(
    '        "front_softms_source": (\n'
    '            "anchor_xy_all = candidate.softms_xy" in tracker_text\n'
    '            and tracker_text.count("soft_mean_shift(") == 3\n'
    '            and \'getattr(config, "EXPERIMENT_ANCHOR"\' not in tracker_text\n'
    '        ),',
    '        "front_decoder_source": (\n'
    '            (variant.get("decoder", "softms") == "softms" and "anchor_xy_all = candidate.softms_xy" in tracker_text)\n'
    '            or (variant.get("decoder") == "weighted_centroid" and "posterior.unsqueeze(-1) * candidate.centers" in tracker_text)\n'
    '        ),',
    "decoder source audit",
)

once(
    '            "scored_forward_candidates": 18,',
    '            "scored_candidates": 18 if variant.get("forward_only", True) else 36,',
    "candidate metadata",
)
once(
    '            "front_decoder": "forward18_softms",',
    '            "front_decoder": str(variant.get("decoder", "softms")),',
    "decoder metadata",
)
once(
    '        "candidate_geometry": "6x6 local geometry; heading-guided forward 3x6 = 18 scored patches",',
    '        "candidate_geometry": ("heading-guided forward 3x6 = 18" if variant.get("forward_only", True) else "full 6x6 = 36"),',
    "manifest geometry",
)

required = ("search_full6x6", "decoder_weighted", "_enable_weighted_decoder", "front_decoder_selected")
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit(f"PATCH AUDIT FAILED: {missing}")
compile(s, str(path), "exec")
path.write_text(s, encoding="utf-8")
print(f"[EXTENDED ABLATION PATCH OK] {path}")
