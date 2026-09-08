#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_weighted_front.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old = '''    softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(\n        raw_logits,\n        centers,\n        config.MEANSHIFT_SCORE_TAU,\n        config.MEANSHIFT_BANDWIDTH_M,\n        config.MEANSHIFT_ITERATIONS,\n        config.MEANSHIFT_MODE_BETA,\n    )\n    return CandidateBatch(\n        indices=selected_indices,\n        centers=centers,\n        z_uav=z_uav,\n        z_sat=z_sat,\n        raw_logits=raw_logits,\n        raw_prob=raw_prob,\n        raw_top1_xy=raw_top1_xy,\n        softms_xy=softms_xy,\n        softms_support=softms_support,\n        softms_mode_count=(mode_weights > 0).sum(dim=1),\n    )'''

new = '''    # Front-stage decoder switch. For the Weighted-Centroid experiment, do\n    # NOT run MeanShift iterations at all. The final visual anchor is computed\n    # later from the same local posterior in visual_observation(). These fields\n    # remain populated only for API compatibility with the acquisition scorer.\n    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "weighted_centroid":\n        softms_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)\n        softms_support = raw_prob.max(dim=1).values\n        softms_mode_count = torch.ones(\n            raw_prob.shape[0], dtype=torch.long, device=raw_prob.device\n        )\n    else:\n        softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(\n            raw_logits,\n            centers,\n            config.MEANSHIFT_SCORE_TAU,\n            config.MEANSHIFT_BANDWIDTH_M,\n            config.MEANSHIFT_ITERATIONS,\n            config.MEANSHIFT_MODE_BETA,\n        )\n        softms_mode_count = (mode_weights > 0).sum(dim=1)\n\n    return CandidateBatch(\n        indices=selected_indices,\n        centers=centers,\n        z_uav=z_uav,\n        z_sat=z_sat,\n        raw_logits=raw_logits,\n        raw_prob=raw_prob,\n        raw_top1_xy=raw_top1_xy,\n        softms_xy=softms_xy,\n        softms_support=softms_support,\n        softms_mode_count=softms_mode_count,\n    )'''

count = s.count(old)
if count != 1:
    raise SystemExit(f"ERROR: weighted-front patch pattern count={count}")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print(f"[PATCHED] front 3x6 decoder: Weighted Centroid with no MeanShift iterations -> {p}")
