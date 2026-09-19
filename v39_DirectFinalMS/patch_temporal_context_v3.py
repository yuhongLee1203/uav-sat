#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_temporal_context_v3.py <runtime-dir>")

root = Path(sys.argv[1])
visual = root / "visual_model.py"
config = root / "config.py"
tracker = root / "robust_tracker.py"
for p in (visual, config, tracker):
    if not p.is_file():
        raise SystemExit(f"missing {p}")

s = visual.read_text(encoding="utf-8")
old = """        self.sat_projection = projection(config.EMBED_DIM)\n        self.previous_state_projection = nn.Sequential(\n"""
new = """        self.sat_projection = projection(config.EMBED_DIM)\n        # Main recurrent state now also receives visual uncertainty + innovation.\n        # [log response var_s, log response var_e, innovation_s, innovation_e]\n        self.innovation_projection = projection(4)\n        self.previous_state_projection = nn.Sequential(\n"""
if s.count(old) != 1:
    raise SystemExit(f"visual_model sat-projection anchor count={s.count(old)}")
s = s.replace(old, new, 1)

old = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
new = "        self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)\n"
if s.count(old) != 1:
    raise SystemExit(f"visual_model GRU input anchor count={s.count(old)}")
s = s.replace(old, new, 1)

old = """        recurrent_input = torch.cat(\n            [\n                self.clip_mean_projection(clip_mean),\n                self.delta_recent_projection(delta_recent),\n                self.delta_accel_projection(delta_accel),\n                self.previous_state_projection(previous_state),\n            ],\n            dim=1,\n        )\n"""
new = """        # Satellite-aware temporal fusion.  This restores the intended five-part\n        # state: three-frame appearance dynamics, SAT context, recurrent numeric\n        # state, and current visual innovation/uncertainty.\n        innovation = visual_anchor_se - predicted_se\n        innovation_numeric = torch.cat(\n            [\n                torch.log1p(response_variance_se.clamp_min(0.0)) / 7.0,\n                innovation[:, 0:1] / float(config.ROUTE_PROGRESS_SCALE_M),\n                innovation[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),\n            ],\n            dim=1,\n        )\n        recurrent_input = torch.cat(\n            [\n                self.clip_mean_projection(clip_mean),\n                self.delta_recent_projection(delta_recent),\n                self.delta_accel_projection(delta_accel),\n                self.sat_projection(sat_context),\n                self.previous_state_projection(previous_state),\n                self.innovation_projection(innovation_numeric),\n            ],\n            dim=1,\n        )\n"""
if s.count(old) != 1:
    raise SystemExit(f"visual_model recurrent-input anchor count={s.count(old)}")
s = s.replace(old, new, 1)
compile(s, str(visual), "exec")
visual.write_text(s, encoding="utf-8")

c = config.read_text(encoding="utf-8")
repls = [
    ("MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 4.0", "MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 2.0"),
    ("MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0", "MAX_MEASUREMENT_CORRECTION_CROSS_M = 2.0"),
    ("LOSS_VELOCITY = 0.25", "LOSS_VELOCITY = 0.50"),
    ("EARLY_STOP_MIN_DELTA = 0.05", "EARLY_STOP_MIN_DELTA = 0.02"),
]
for old, new in repls:
    if c.count(old) != 1:
        raise SystemExit(f"config token count for {old!r} = {c.count(old)}")
    c = c.replace(old, new, 1)
compile(c, str(config), "exec")
config.write_text(c, encoding="utf-8")

# Add P90 to checkpoint selection so the temporal model is selected for both
# average localization and tail robustness, rather than motion diagnostics alone.
t = tracker.read_text(encoding="utf-8")
old = """    score = (\n        mle\n        + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed_mae\n"""
new = """    p90 = float(np.quantile(errors, 0.90))\n    score = (\n        mle\n        + 0.20 * p90\n        + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed_mae\n"""
if t.count(old) != 1:
    raise SystemExit(f"tracker early-score anchor count={t.count(old)}")
t = t.replace(old, new, 1)
old = '        "p90": float(np.quantile(errors, 0.90)),\n'
new = '        "p90": p90,\n'
if t.count(old) != 1:
    raise SystemExit(f"tracker p90 return anchor count={t.count(old)}")
t = t.replace(old, new, 1)
compile(t, str(tracker), "exec")
tracker.write_text(t, encoding="utf-8")

print("[OK] temporal-context v3 installed: SAT context + uncertainty/innovation + conservative residual heads")
