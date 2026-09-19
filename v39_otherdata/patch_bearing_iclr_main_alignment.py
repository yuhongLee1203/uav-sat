#!/usr/bin/env python3
"""Align bearing_iclr_ablation.py with the current paper-ready main architecture.

This patch only fixes method/protocol mismatches, Route-A-only cadence limits,
and the forward-only search geometry. It never reads held-out B/C outputs and
never edits measured results.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else "v39_otherdata/bearing_iclr_ablation.py")
s = path.read_text(encoding="utf-8")

repls = [
    (
        'The paper-facing chain contains exactly one MeanShift decoder:\n\n    6x6 geometry -> forward 3x6 visual scores -> 3-frame GRU\n    -> fixed-R Kalman -> one final local MeanShift -> XY',
        'The paper-facing chain matches the current paper-ready main model:\n\n    6x6 geometry -> forward 3x6 visual scores -> front SoftMS\n    -> 3-frame GRU -> fixed-R Kalman -> one final local MeanShift -> XY',
    ),
    (
        'ARCH = "ICLR_Forward18_Simple3FrameGRU_FixedKalman_OneFinalMS"',
        'ARCH = "ICLR_Forward18SoftMS_Simple3FrameGRU_FixedKalman_FinalMS"',
    ),
    (
        '"max_forward_speed_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),\n'
        '        "max_polynomial_step_m_per_frame": max(14.0, min(20.0, 1.05 * p95)),\n'
        '        "max_measurement_correction_parallel_m": max(4.0, min(8.0, 0.55 * p90)),\n'
        '        "kalman_max_measurement_innovation_progress_m": max(5.0, min(10.0, 0.65 * p90)),\n'
        '        "kalman_max_posterior_correction_progress_m": max(3.0, min(6.0, 0.45 * p90)),\n'
        '        "kalman_max_velocity_correction_m_per_frame": max(1.25, min(2.5, 0.18 * p90)),\n'
        '        "kalman_final_step_max_m": max(7.0, min(14.0, 1.10 * p90)),',
        '"max_forward_speed_m_per_frame": max(14.0, min(30.0, 1.10 * p95)),\n'
        '        "max_polynomial_step_m_per_frame": max(14.0, min(30.0, 1.10 * p95)),\n'
        '        "max_measurement_correction_parallel_m": max(4.0, min(12.0, 0.60 * p90)),\n'
        '        "kalman_max_measurement_innovation_progress_m": max(5.0, min(14.0, 0.75 * p90)),\n'
        '        "kalman_max_posterior_correction_progress_m": max(3.0, min(9.0, 0.50 * p90)),\n'
        '        "kalman_max_velocity_correction_m_per_frame": max(1.25, min(3.5, 0.20 * p90)),\n'
        '        "kalman_final_step_max_m": max(7.0, min(30.0, 1.10 * p95)),',
    ),
    (
        '        # Forward-18 posterior is summarized without a front MeanShift.  The\n'
        '        # only MeanShift in the paper chain is the post-Kalman decoder.\n'
        '        "UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid",',
        '        # Paper-ready main architecture: Forward-18 is decoded by front SoftMS.\n'
        '        # A separate final MeanShift remains after the fixed-R Kalman.\n'
        '        "UAVSAT_EXPERIMENT_ANCHOR": "softms",',
    ),
    (
        '    exact._patch_paths_and_scale(config, args, prepared_root)\n'
        '    config.ARCHITECTURE_NAME = ARCH',
        '    exact._patch_paths_and_scale(config, args, prepared_root)\n'
        '    # The controlled prior is GT + bounded jitter. A forward-only 3x6\n'
        '    # selector must not discard the true location simply because the\n'
        '    # jittered prior happens to lie ahead of it. Shift the 6x6 lattice\n'
        '    # backward by the known jitter bound plus half one physical SAT\n'
        '    # stride (quantization margin). This uses protocol constants only;\n'
        '    # no Route-B/C metric or label is inspected.\n'
        '    geometry = getattr(config, "BEARING_PHYSICAL_SAT_GEOMETRY", None)\n'
        '    if not isinstance(geometry, dict) or "sat_stride_m" not in geometry:\n'
        '        raise RuntimeError("missing audited Bearing physical SAT geometry")\n'
        '    config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (\n'
        '        float(config.CONTROLLED_GT_PRIOR_JITTER_M)\n'
        '        + 0.5 * float(geometry["sat_stride_m"])\n'
        '    )\n'
        '    config.ARCHITECTURE_NAME = ARCH',
    ),
    (
        '        "front_decoder_not_ms": str(config.EXPERIMENT_ANCHOR) == "weighted_centroid",',
        '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",',
    ),
    (
        '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",\n'
        '        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",',
        '        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",\n'
        '        "forward_origin_backshift_covers_jitter": (\n'
        '            float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M)\n'
        '            >= float(config.CONTROLLED_GT_PRIOR_JITTER_M)\n'
        '        ),\n'
        '        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",',
    ),
    (
        '            "front_decoder": "posterior_weighted_centroid",',
        '            "front_decoder": "forward18_softms",',
    ),
    (
        '            "front_decoder": "forward18_softms",\n'
        '            "online_meanshift_count": 1 if variant["ms"] else 0,',
        '            "front_decoder": "forward18_softms",\n'
        '            "forward_origin_backshift_m": float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M),\n'
        '            "controlled_prior_jitter_m": float(config.CONTROLLED_GT_PRIOR_JITTER_M),\n'
        '            "online_meanshift_count": 1 if variant["ms"] else 0,',
    ),
    (
        '        "paper_chain": "Forward18 posterior -> 3-frame GRU -> fixed-R Kalman -> one final MeanShift -> XY",',
        '        "paper_chain": "Forward18 SoftMS -> 3-frame GRU -> fixed-R Kalman -> final MeanShift -> XY",',
    ),
]

changed = 0
for old, new in repls:
    if old in s:
        s = s.replace(old, new, 1)
        changed += 1
    elif new in s:
        pass
    else:
        raise SystemExit("PATCH FAILED: expected block not found:\n" + old[:240])

required = [
    'UAVSAT_EXPERIMENT_ANCHOR": "softms"',
    'front_decoder_softms',
    'forward_origin_backshift_covers_jitter',
    'config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (',
    '0.5 * float(geometry["sat_stride_m"])',
    '"forward_origin_backshift_m": float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M)',
    '1.10 * p95',
    '"kalman_final_step_max_m": max(7.0, min(30.0, 1.10 * p95))',
    'Forward18 SoftMS -> 3-frame GRU -> fixed-R Kalman -> final MeanShift -> XY',
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit("PATCH AUDIT FAILED: " + repr(missing))

compile(s, str(path), "exec")
path.write_text(s, encoding="utf-8")
print(f"[PATCH OK] {path}")
print("[PATCH OK] front decoder aligned to Forward-18 SoftMS")
print("[PATCH OK] forward 3x6 origin backshift = jitter bound + 0.5 SAT stride")
print("[PATCH OK] longitudinal Kalman/motion limits follow Route-A p90/p95 without the stale 14/20 m bottleneck")
print("[PATCH OK] no B/C metric was read or modified")
