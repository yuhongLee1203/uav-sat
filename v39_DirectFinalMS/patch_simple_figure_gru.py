#!/usr/bin/env python3
"""Patch V39 GRU to match the paper figure without any split/residual gates.

Main GRU inputs after this patch:
  temporal mean + first difference + second difference
  + satellite context
  + current Forward-18 SoftMS visual position
  + previous recurrent state

The visual position is used directly; no position innovation/difference to a
motion/Kalman prediction is constructed. Output heads remain the original V39
correction/variance/motion/heading heads.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_simple_figure_gru.py VISUAL_MODEL.py")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

if "innovation_projection" in s or "visual_anchor_se - predicted_se" in s:
    raise SystemExit("refusing runtime containing position-innovation GRU input")

# Add a direct visual-position projector once.
needle = "        self.sat_projection = projection(config.EMBED_DIM)\n"
addition = needle + "        self.visual_position_projection = projection(2)\n"
if "self.visual_position_projection = projection(2)" not in s:
    if s.count(needle) != 1:
        raise SystemExit("expected exactly one sat_projection declaration")
    s = s.replace(needle, addition, 1)

# The canonical source has 4 blocks. Bearing's existing context patch may have
# already made it 5. Both are promoted to the requested six direct input blocks.
if "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)" not in s:
    if s.count("self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)") == 1:
        s = s.replace(
            "self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)",
            "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)",
            1,
        )
    elif s.count("self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)") == 1:
        s = s.replace(
            "self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)",
            "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)",
            1,
        )
    else:
        raise SystemExit("could not identify canonical GRUCell declaration")

old4 = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
old5 = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
new = '''        # Direct current visual position from the Forward-18 SoftMS decoder.
        # This is NOT an innovation: no predicted/Kalman position is subtracted.
        visual_position = torch.cat(
            [
                visual_anchor_se[:, 0:1] / float(config.ROUTE_PROGRESS_SCALE_M),
                visual_anchor_se[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),
            ],
            dim=1,
        )
        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.visual_position_projection(visual_position),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
if new not in s:
    if s.count(old5) == 1:
        s = s.replace(old5, new, 1)
    elif s.count(old4) == 1:
        s = s.replace(old4, new, 1)
    else:
        raise SystemExit("could not identify canonical recurrent-input block")

# Final structural audit.
required = [
    "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)",
    "self.sat_projection(sat_context)",
    "self.visual_position_projection(visual_position)",
    "self.previous_state_projection(previous_state)",
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit("GRU patch audit failed: missing " + repr(missing))
if "visual_anchor_se - predicted_se" in s or "innovation_projection" in s:
    raise SystemExit("GRU patch audit failed: forbidden innovation input present")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[PATCH OK] GRU = mean + delta + delta2 + SAT context + direct visual position + previous state")
print("[PATCH OK] no split/dual/residual gate; no position innovation")
