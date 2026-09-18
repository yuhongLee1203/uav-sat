#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_front_softms.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

weighted = '''    # Front visual observation: Weighted Centroid, no MeanShift.
    # visual_observation() uses the local posterior again for the actual anchor
    # and computes uncertainty from posterior-weighted candidate dispersion.
    weighted_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)
    posterior_support = raw_prob.max(dim=1).values
    posterior_mode_count = torch.ones(
        raw_prob.shape[0], dtype=torch.long, device=raw_prob.device
    )
    return CandidateBatch(
        indices=selected_indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=weighted_xy,
        softms_support=posterior_support,
        softms_mode_count=posterior_mode_count,
    )
'''
softms = '''    # Front visual observation: Soft MeanShift over the scored Forward 3x6
    # candidates.  Only the selected 18 forward candidates participate.
    softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(
        raw_logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    return CandidateBatch(
        indices=selected_indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=softms_xy,
        softms_support=softms_support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
    )
'''
if s.count(weighted) != 1:
    raise SystemExit(f"ERROR: patched Weighted-Centroid front block count={s.count(weighted)}")
s = s.replace(weighted, softms, 1)
s = s.replace(
    '# v39 selected chain, with ONLY the front decoder changed:\n        # Weighted Centroid -> GRU -> Kalman -> ONE final MS -> Final',
    '# v39 selected chain:\n        # Forward 3x6 SoftMS -> GRU -> Kalman -> ONE final 6x6 MS -> Final',
    1,
)
s = s.replace(
    'summary["VisualObservationDecoder"] = "posterior weighted centroid"',
    'summary["VisualObservationDecoder"] = "forward 3x6 soft mean shift"',
    1,
)
s = s.replace(
    'summary["VisualObservationUncertainty"] = "posterior-weighted spatial variance projected to route parallel/cross coordinates"',
    'summary["VisualObservationUncertainty"] = "SoftMS converged-mode spatial variance projected to route parallel/cross coordinates"',
    1,
)
s = s.replace(
    'summary["OnlineMeanShiftCount"] = 1 if summary["MS_Enabled"] else 0',
    'summary["OnlineMeanShiftCount"] = 2 if summary["MS_Enabled"] else 1',
    1,
)
s = s.replace(
    '"weighted-centroid visual observation -> GRU -> robust constrained route-coordinate Kalman -> one final MeanShift -> final XY.",',
    '"Forward-3x6 Soft MeanShift visual observation -> GRU -> robust constrained route-coordinate Kalman -> one final 6x6 MeanShift -> final XY.",',
    1,
)
if s.count("soft_mean_shift(") != 3:
    raise SystemExit(f"ERROR: expected 3 SoftMS call sites after front restoration, got {s.count('soft_mean_shift(')}")
if "weighted_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)" in s:
    raise SystemExit("ERROR: Weighted Centroid front decoder still remains")
compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] V39 front decoder: Forward 3x6 SoftMS -> GRU -> fixed Kalman -> final 6x6 SoftMS")
