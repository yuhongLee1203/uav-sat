#!/usr/bin/env python3
"""Patch Bearing ICLR runner to the SoftMS-only fair temporal experiment.

Active chain:
    6x6 geometry -> heading-forward 3x6 (18 scored patches)
    -> front Soft MeanShift -> temporal GRU
    -> fixed-R Kalman -> final local MeanShift -> XY

1/2/3-frame variants are trained separately on Route A.  B/C are evaluation
only and are never read by this patch.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else "v39_otherdata/bearing_iclr_ablation.py")
s = path.read_text(encoding="utf-8")


def replace_once_or_already(old: str, new: str, label: str) -> None:
    global s
    if new in s:
        return
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"PATCH FAILED [{label}]: expected 1 old block, found {count}")
    s = s.replace(old, new, 1)


replace_once_or_already(
    'The paper-facing chain contains exactly one MeanShift decoder:\n\n'
    '    6x6 geometry -> forward 3x6 visual scores -> 3-frame GRU\n'
    '    -> fixed-R Kalman -> one final local MeanShift -> XY',
    'The paper-facing chain uses two explicit SoftMS stages:\n\n'
    '    6x6 geometry -> forward 3x6 = 18 visual scores -> front SoftMS\n'
    '    -> temporal GRU -> fixed-R Kalman -> final local MeanShift -> XY',
    'docstring chain',
)

replace_once_or_already(
    'ARCH = "ICLR_Forward18_Simple3FrameGRU_FixedKalman_OneFinalMS"',
    'ARCH = "ICLR_Forward18SoftMS_SimpleTemporalGRU_FixedKalman_FinalMS"',
    'architecture name',
)

replace_once_or_already(
    'def _train_root(args: argparse.Namespace) -> Path:\n'
    '    return Path(args.suite_root).resolve() / args.city / "train_full"',
    'def _train_root(args: argparse.Namespace, frames=None) -> Path:\n'
    '    frame_count = int(frames if frames is not None else getattr(args, "train_frames", 3))\n'
    '    return Path(args.suite_root).resolve() / args.city / f"train_frames{frame_count}"',
    'frame-specific train root',
)

replace_once_or_already(
    '        "max_forward_speed_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),\n'
    '        "max_polynomial_step_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),\n'
    '        "max_measurement_correction_parallel_m": max(4.0, min(8.0, 0.55 * p90)),\n'
    '        "kalman_max_measurement_innovation_progress_m": max(5.0, min(10.0, 0.65 * p90)),\n'
    '        "kalman_max_posterior_correction_progress_m": max(3.0, min(6.0, 0.45 * p90)),\n'
    '        "kalman_max_velocity_correction_m_per_frame": max(1.25, min(2.5, 0.18 * p90)),\n'
    '        "kalman_final_step_max_m": max(7.0, min(14.0, 1.10 * p90)),',
    '        "max_forward_speed_m_per_frame": max(14.0, min(30.0, 1.10 * p95)),\n'
    '        "max_polynomial_step_m_per_frame": max(14.0, min(30.0, 1.10 * p95)),\n'
    '        "max_measurement_correction_parallel_m": max(4.0, min(12.0, 0.60 * p90)),\n'
    '        "kalman_max_measurement_innovation_progress_m": max(5.0, min(14.0, 0.75 * p90)),\n'
    '        "kalman_max_posterior_correction_progress_m": max(3.0, min(9.0, 0.50 * p90)),\n'
    '        "kalman_max_velocity_correction_m_per_frame": max(1.25, min(3.5, 0.20 * p90)),\n'
    '        "kalman_final_step_max_m": max(7.0, min(30.0, 1.10 * p95)),',
    'Route-A cadence bounds',
)

# Remove the old front-decoder choice from the experiment runner.  Construct the
# old token so the patched runner itself contains no legacy decoder identifier.
legacy_token = "weighted" + "_" + "centroid"
old_env = (
    '        # Forward-18 posterior is summarized without a front MeanShift.  The\n'
    '        # only MeanShift in the paper chain is the post-Kalman decoder.\n'
    f'        "UAVSAT_EXPERIMENT_ANCHOR": "{legacy_token}",'
)
new_env = (
    '        # Forward-18 is always decoded by Soft MeanShift.\n'
    '        "UAVSAT_EXPERIMENT_ANCHOR": "softms",'
)
replace_once_or_already(old_env, new_env, 'front SoftMS environment')

replace_once_or_already(
    '    exact._patch_paths_and_scale(config, args, prepared_root)\n'
    '    config.ARCHITECTURE_NAME = ARCH',
    '    exact._patch_paths_and_scale(config, args, prepared_root)\n'
    '    geometry = getattr(config, "BEARING_PHYSICAL_SAT_GEOMETRY", None)\n'
    '    if not isinstance(geometry, dict) or "sat_stride_m" not in geometry:\n'
    '        raise RuntimeError("missing audited Bearing physical SAT geometry")\n'
    '    # Cover the full bounded local-prior jitter before retaining only the\n'
    '    # heading-forward 18 cells.  Uses protocol constants, never B/C metrics.\n'
    '    config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (\n'
    '        float(config.CONTROLLED_GT_PRIOR_JITTER_M)\n'
    '        + 0.5 * float(geometry["sat_stride_m"])\n'
    '    )\n'
    '    config.ARCHITECTURE_NAME = ARCH',
    'forward-origin backshift',
)

old_audit = (
    '        "one_final_ms_source": "online final path contains exactly one MeanShift" in tracker_text,\n'
    f'        "front_decoder_not_ms": str(config.EXPERIMENT_ANCHOR) == "{legacy_token}",'
)
new_audit = (
    '        "front_softms_source": (\n'
    '            "anchor_xy_all = candidate.softms_xy" in tracker_text\n'
    '            and tracker_text.count("soft_mean_shift(") == 3\n'
    '            and "anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)" not in tracker_text\n'
    '        ),\n'
    '        "one_final_ms_source": "exactly one final local Soft MeanShift after the Kalman estimator" in tracker_text,\n'
    '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",'
)
replace_once_or_already(old_audit, new_audit, 'runtime SoftMS audit')

replace_once_or_already(
    '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",\n'
    '        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",',
    '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",\n'
    '        "forward_origin_backshift_covers_jitter": (\n'
    '            float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M)\n'
    '            >= float(config.CONTROLLED_GT_PRIOR_JITTER_M)\n'
    '        ),\n'
    '        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",',
    'backshift audit',
)

replace_once_or_already(
    '    output = _train_root(args)\n'
    '    runtime = _make_runtime(prepared, output / "runtime")\n'
    '    _set_environment(args, prepared, output, VARIANTS["full"], training=True)\n'
    '    config, tracker, visual_localizer = base._load_runtime_modules(runtime)\n'
    '    _patch_paths(config, args, prepared)\n'
    '    audit = _audit_runtime(config, runtime, VARIANTS["full"], training=True)',
    '    train_variant = dict(VARIANTS["full"])\n'
    '    train_variant["frames"] = int(args.train_frames)\n'
    '    output = _train_root(args, args.train_frames)\n'
    '    runtime = _make_runtime(prepared, output / "runtime")\n'
    '    _set_environment(args, prepared, output, train_variant, training=True)\n'
    '    config, tracker, visual_localizer = base._load_runtime_modules(runtime)\n'
    '    _patch_paths(config, args, prepared)\n'
    '    audit = _audit_runtime(config, runtime, train_variant, training=True)',
    'frame-specific training variant',
)

s = s.replace(
    '_write_manifest(args, output, VARIANTS["full"], audit, training=True)',
    '_write_manifest(args, output, train_variant, audit, training=True)',
)

replace_once_or_already(
    '    _link_full_checkpoints(config, _train_root(args))',
    '    checkpoint_frames = int(variant["frames"]) if args.variant in {"frames1", "frames2"} else 3\n'
    '    _link_full_checkpoints(config, _train_root(args, checkpoint_frames))',
    'frame-specific evaluation checkpoint',
)

old_summary = (
    f'            "front_decoder": "posterior_{legacy_token}",\n'
    '            "online_meanshift_count": 1 if variant["ms"] else 0,'
)
new_summary = (
    '            "front_decoder": "forward18_softms",\n'
    '            "front_meanshift_count": 1,\n'
    '            "final_meanshift_count": 1 if variant["ms"] else 0,\n'
    '            "online_meanshift_count": 2 if variant["ms"] else 1,\n'
    '            "forward_origin_backshift_m": float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M),\n'
    '            "controlled_prior_jitter_m": float(config.CONTROLLED_GT_PRIOR_JITTER_M),'
)
replace_once_or_already(old_summary, new_summary, 'result protocol summary')

replace_once_or_already(
    '        "paper_chain": "Forward18 posterior -> 3-frame GRU -> fixed-R Kalman -> one final MeanShift -> XY",',
    '        "paper_chain": "Forward18 SoftMS -> temporal GRU -> fixed-R Kalman -> final MeanShift -> XY",',
    'manifest chain',
)

replace_once_or_already(
    '    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))',
    '    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))\n'
    '    p.add_argument("--train-frames", type=int, default=3, choices=[1, 2, 3])',
    'train-frames CLI',
)

required = [
    'UAVSAT_EXPERIMENT_ANCHOR": "softms"',
    'front_softms_source',
    'tracker_text.count("soft_mean_shift(") == 3',
    'anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)',
    'forward_origin_backshift_covers_jitter',
    'config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (',
    'checkpoint_frames = int(variant["frames"])',
    'train_variant["frames"] = int(args.train_frames)',
    'p.add_argument("--train-frames"',
    '"front_decoder": "forward18_softms"',
    'Forward18 SoftMS -> temporal GRU -> fixed-R Kalman -> final MeanShift -> XY',
]
missing = [item for item in required if item not in s]
if missing:
    raise SystemExit("PATCH AUDIT FAILED: missing " + repr(missing))

# The patched experiment runner itself must not carry the legacy decoder token.
# The runtime audit above detects the old centroid formula structurally instead.
if legacy_token in s.lower():
    raise SystemExit("PATCH AUDIT FAILED: legacy decoder identifier remains in active runner")

compile(s, str(path), "exec")
path.write_text(s, encoding="utf-8")
print(f"[PATCH OK] {path}")
print("[PATCH OK] active runner = Forward-18 SoftMS -> temporal GRU -> fixed-R Kalman -> final MeanShift")
print("[PATCH OK] runtime source audit checks the real SoftMS code path")
print("[PATCH OK] 1/2/3-frame rows use separately Route-A-trained temporal checkpoints")
print("[PATCH OK] forward 3x6 backshift = jitter bound + 0.5 SAT stride")
print("[PATCH OK] no B/C metric was read or modified")
