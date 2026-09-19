#!/usr/bin/env python3
"""Patch V39 to the simple paper-figure GRU without inference gates.

GRU inputs:
  temporal mean + first difference + second difference
  + satellite context
  + current Forward-18 SoftMS visual position
  + previous recurrent state

No split gate / dual gate / inference gain / position innovation is added.
Training-side constants are configurable through environment variables so a
Route-A-only hyperparameter search can be performed without touching B/C.
"""
from pathlib import Path
import re
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_simple_figure_gru.py VISUAL_MODEL.py")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

if "innovation_projection" in s or "visual_anchor_se - predicted_se" in s:
    raise SystemExit("refusing runtime containing position-innovation GRU input")

needle = "        self.sat_projection = projection(config.EMBED_DIM)\n"
addition = needle + "        self.visual_position_projection = projection(2)\n"
if "self.visual_position_projection = projection(2)" not in s:
    if s.count(needle) != 1:
        raise SystemExit("expected exactly one sat_projection declaration")
    s = s.replace(needle, addition, 1)

if "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)" not in s:
    if s.count("self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)") == 1:
        s = s.replace(
            "self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)",
            "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)", 1,
        )
    elif s.count("self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)") == 1:
        s = s.replace(
            "self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)",
            "self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)", 1,
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
new = '''        # Direct current visual position from Forward-18 SoftMS.
        # This is NOT an innovation: no motion/Kalman position is subtracted.
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

cfg = p.with_name("config.py")
if not cfg.exists():
    raise SystemExit(f"missing sibling config.py: {cfg}")
c = cfg.read_text(encoding="utf-8")

def sub1(pattern, replacement, label):
    global c
    c2, n = re.subn(pattern, replacement, c, count=1, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"config patch failed for {label}: matches={n}")
    c = c2

sub1(r'^MAX_MEASUREMENT_CORRECTION_PARALLEL_M\s*=\s*[0-9.]+\s*$',
     'MAX_MEASUREMENT_CORRECTION_PARALLEL_M = float(os.environ.get("UAVSAT_CORR_PARALLEL_M", "0.75"))', 'parallel correction bound')
sub1(r'^MAX_MEASUREMENT_CORRECTION_CROSS_M\s*=\s*[0-9.]+\s*$',
     'MAX_MEASUREMENT_CORRECTION_CROSS_M = float(os.environ.get("UAVSAT_CORR_CROSS_M", "0.50"))', 'cross correction bound')
sub1(r'^LOSS_MEASUREMENT\s*=\s*[0-9.]+\s*$',
     'LOSS_MEASUREMENT = float(os.environ.get("UAVSAT_LOSS_MEASUREMENT", "2.50"))', 'measurement loss')
sub1(r'^LOSS_NEXT_STEP\s*=\s*[0-9.]+\s*$',
     'LOSS_NEXT_STEP = float(os.environ.get("UAVSAT_LOSS_NEXT_STEP", "1.50"))', 'next-step loss')
sub1(r'^LOSS_VELOCITY\s*=\s*[0-9.]+\s*$',
     'LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "0.10"))', 'velocity loss')
sub1(r'^TEMPORAL_LR\s*=\s*[0-9.eE+-]+\s*$',
     'TEMPORAL_LR = float(os.environ.get("UAVSAT_TEMPORAL_LR", "2e-4"))', 'temporal lr')
sub1(r'^RNN_DROPOUT\s*=\s*[0-9.]+\s*$',
     'RNN_DROPOUT = float(os.environ.get("UAVSAT_RNN_DROPOUT", "0.10"))', 'gru dropout')
sub1(r'^MOTION_VELOCITY_EMA_ALPHA\s*=\s*[0-9.]+\s*$',
     'MOTION_VELOCITY_EMA_ALPHA = float(os.environ.get("UAVSAT_MOTION_VEL_ALPHA", "0.35"))', 'motion velocity alpha')
sub1(r'^MOTION_POLYNOMIAL_STEP_EMA_ALPHA\s*=\s*[0-9.]+\s*$',
     'MOTION_POLYNOMIAL_STEP_EMA_ALPHA = float(os.environ.get("UAVSAT_MOTION_STEP_ALPHA", "0.40"))', 'motion step alpha')
sub1(r'^EARLY_STOP_MIN_DELTA\s*=\s*[0-9.]+\s*$',
     'EARLY_STOP_MIN_DELTA = float(os.environ.get("UAVSAT_EARLY_MIN_DELTA", "0.01"))', 'early stop delta')
sub1(r'^EARLY_STOP_MIN_EPOCH\s*=\s*[0-9]+\s*$',
     'EARLY_STOP_MIN_EPOCH = int(os.environ.get("UAVSAT_EARLY_MIN_EPOCH", "10"))', 'early stop min epoch')
sub1(r'^SEED\s*=\s*2033\s*$',
     'SEED = int(os.environ.get("UAVSAT_SEED", "2033"))', 'seed env')

compile(c, str(cfg), "exec")
cfg.write_text(c, encoding="utf-8")

print("[PATCH OK] simple GRU = mean + delta + delta2 + SAT context + direct SoftMS position + previous state")
print("[PATCH OK] no split/dual/inference gate; no position innovation")
print("[PATCH OK] Route-A-only search knobs exposed through UAVSAT_* environment variables")
