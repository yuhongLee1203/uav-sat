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

# Main GRU inputs become exactly five branches:
#   1) 3-frame temporal mean
#   2) recent first difference
#   3) temporal second difference
#   4) current posterior-weighted satellite context
#   5) previous recurrent numeric state
# There is deliberately NO visual_anchor-predicted_position innovation input,
# NO Kalman position input, and NO direct position-difference feature.
old = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
new = "        self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)\n"
if s.count(old) != 1:
    raise SystemExit(f"visual_model GRU input anchor count={s.count(old)}")
s = s.replace(old, new, 1)

old = """        recurrent_input = torch.cat(\n            [\n                self.clip_mean_projection(clip_mean),\n                self.delta_recent_projection(delta_recent),\n                self.delta_accel_projection(delta_accel),\n                self.previous_state_projection(previous_state),\n            ],\n            dim=1,\n        )\n"""
new = """        # Temporal-context GRU: no position innovation is fed back.\n        # The only recurrent numeric feedback is previous_state. Satellite\n        # context is a current-frame visual/context input, not a position state.\n        recurrent_input = torch.cat(\n            [\n                self.clip_mean_projection(clip_mean),\n                self.delta_recent_projection(delta_recent),\n                self.delta_accel_projection(delta_accel),\n                self.sat_projection(sat_context),\n                self.previous_state_projection(previous_state),\n            ],\n            dim=1,\n        )\n"""
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

# Select the temporal checkpoint using Route-A validation localization quality
# plus the existing motion diagnostics. This does not use Route B/C.
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

print("[OK] temporal-context v3 installed: mean + delta + delta2 + SAT context + previous state; NO position innovation")
