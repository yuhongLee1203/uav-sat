#!/usr/bin/env python3
"""Patch V39 to the compact temporal GRU used by the Bearing-UAV ablation.

The paper architecture stays:
  Forward-18 SoftMS -> temporal GRU -> constrained Kalman -> final MeanShift.

This patch only changes the temporal motion parameterization:
  * GRU sees mean, first difference, second difference, SAT context,
    current SoftMS position, causal visual displacement and previous state.
  * Motion is predicted as a bounded residual around the previous stable
    motion state instead of as a new absolute velocity every frame.
  * The recurrent/Kalman motion state starts from the training-city cadence,
    not from zero.
  * Acceleration supervision is exposed so the 3-frame second difference has
    a direct training target.

All knobs are training/validation-side environment variables. Held-out city
navigation outputs are never read by this patch.
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

# -----------------------------------------------------------------------------
# Seven compact GRU blocks.  The added visual-motion term is a causal
# frame-to-frame SoftMS displacement, not a motion-prediction innovation.
# -----------------------------------------------------------------------------
needle = "        self.sat_projection = projection(config.EMBED_DIM)\n"
addition = (
    needle
    + "        self.visual_position_projection = projection(2)\n"
    + "        self.visual_motion_projection = projection(2)\n"
)
if "self.visual_motion_projection = projection(2)" not in s:
    if s.count(needle) != 1:
        raise SystemExit("expected exactly one sat_projection declaration")
    s = s.replace(needle, addition, 1)

for width in (4, 5, 6):
    old = f"self.gru = nn.GRUCell(feature_dim * {width}, hidden_dim)"
    if old in s:
        s = s.replace(old, "self.gru = nn.GRUCell(feature_dim * 7, hidden_dim)", 1)
        break
if "self.gru = nn.GRUCell(feature_dim * 7, hidden_dim)" not in s:
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
old6 = '''        # Direct current visual position from Forward-18 SoftMS.
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
new = '''        # Direct current visual position from Forward-18 SoftMS.
        visual_position = torch.cat(
            [
                visual_anchor_se[:, 0:1] / float(config.ROUTE_PROGRESS_SCALE_M),
                visual_anchor_se[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),
            ],
            dim=1,
        )
        # Causal visual displacement.  This is deliberately not
        # visual_anchor - motion_prediction; it is simply the displacement
        # between consecutive visual measurements.
        if previous_measurement_se is None:
            visual_motion = torch.zeros_like(visual_anchor_se)
        else:
            visual_motion = (
                visual_anchor_se - previous_measurement_se
            ) / float(config.ROUTE_STEP_SCALE_M)
        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.visual_position_projection(visual_position),
                self.visual_motion_projection(visual_motion),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
if new not in s:
    replaced = False
    for old in (old6, old5, old4):
        if s.count(old) == 1:
            s = s.replace(old, new, 1)
            replaced = True
            break
    if not replaced:
        raise SystemExit("could not identify canonical recurrent-input block")

# -----------------------------------------------------------------------------
# Residual motion head.  The previous stable motion is the baseline; the GRU
# only predicts a bounded correction.  This preserves inertia and makes the
# second temporal difference useful for acceleration refinement.
# -----------------------------------------------------------------------------
old_motion = '''        raw_motion = self.motion_head(h)
        v_parallel = F.softplus(raw_motion[:, 0:1]).clamp(
            max=float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        )
        v_cross = torch.tanh(raw_motion[:, 1:2]) * float(
            config.MAX_CROSS_SPEED_M_PER_FRAME
        )
        a_parallel = torch.tanh(raw_motion[:, 2:3]) * float(
            config.MAX_FORWARD_ACCEL_M_PER_FRAME2
        )
        a_cross = torch.tanh(raw_motion[:, 3:4]) * float(
            config.MAX_CROSS_ACCEL_M_PER_FRAME2
        )
        velocity = torch.cat([v_parallel, v_cross], dim=1)
        acceleration = torch.cat([a_parallel, a_cross], dim=1)
'''
new_motion = '''        raw_motion = self.motion_head(h)
        dv_parallel = torch.tanh(raw_motion[:, 0:1]) * float(
            config.MOTION_RESIDUAL_FORWARD_M
        )
        dv_cross = torch.tanh(raw_motion[:, 1:2]) * float(
            config.MOTION_RESIDUAL_CROSS_M
        )
        da_parallel = torch.tanh(raw_motion[:, 2:3]) * float(
            config.MOTION_RESIDUAL_ACCEL_FORWARD_M
        )
        da_cross = torch.tanh(raw_motion[:, 3:4]) * float(
            config.MOTION_RESIDUAL_ACCEL_CROSS_M
        )
        v_parallel = (previous_velocity_se[:, 0:1] + dv_parallel).clamp(
            min=0.0, max=float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        )
        v_cross = (previous_velocity_se[:, 1:2] + dv_cross).clamp(
            min=-float(config.MAX_CROSS_SPEED_M_PER_FRAME),
            max=float(config.MAX_CROSS_SPEED_M_PER_FRAME),
        )
        a_parallel = (previous_acceleration_se[:, 0:1] + da_parallel).clamp(
            min=-float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
            max=float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
        )
        a_cross = (previous_acceleration_se[:, 1:2] + da_cross).clamp(
            min=-float(config.MAX_CROSS_ACCEL_M_PER_FRAME2),
            max=float(config.MAX_CROSS_ACCEL_M_PER_FRAME2),
        )
        velocity = torch.cat([v_parallel, v_cross], dim=1)
        acceleration = torch.cat([a_parallel, a_cross], dim=1)
'''
if new_motion not in s:
    if s.count(old_motion) != 1:
        raise SystemExit("could not identify canonical absolute-motion block")
    s = s.replace(old_motion, new_motion, 1)

# Residual head starts at zero correction.  The cadence baseline is installed in
# the recurrent/Kalman state below, so there is no large arbitrary output bias.
old_init = '''        init_speed = 0.75
        self.motion_head[-1].bias.data[0] = math.log(math.exp(init_speed) - 1.0)
'''
old_init_route = '''        init_speed = float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.75))
        init_speed = max(
            1e-3,
            min(init_speed, float(config.MAX_FORWARD_SPEED_M_PER_FRAME) - 1e-3),
        )
        self.motion_head[-1].bias.data[0] = math.log(math.expm1(init_speed))
'''
new_init = '''        # Residual motion starts from zero correction around the cadence prior.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
'''
if new_init not in s:
    if s.count(old_init) == 1:
        s = s.replace(old_init, new_init, 1)
    elif s.count(old_init_route) == 1:
        s = s.replace(old_init_route, new_init, 1)
    else:
        raise SystemExit("could not identify canonical motion-head initialization")

required = [
    "self.gru = nn.GRUCell(feature_dim * 7, hidden_dim)",
    "self.sat_projection(sat_context)",
    "self.visual_position_projection(visual_position)",
    "self.visual_motion_projection(visual_motion)",
    "self.previous_state_projection(previous_state)",
    "MOTION_RESIDUAL_FORWARD_M",
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit("GRU patch audit failed: missing " + repr(missing))
if "visual_anchor_se - predicted_se" in s or "innovation_projection" in s:
    raise SystemExit("GRU patch audit failed: forbidden prediction-innovation input present")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")

# -----------------------------------------------------------------------------
# Runtime configuration knobs.
# -----------------------------------------------------------------------------
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

sub1(r'^MAX_MEASUREMENT_CORRECTION_PARALLEL_M\s*=\s*[0-9.]+\s*$', 'MAX_MEASUREMENT_CORRECTION_PARALLEL_M = float(os.environ.get("UAVSAT_CORR_PARALLEL_M", "0.75"))', 'parallel correction bound')
sub1(r'^MAX_MEASUREMENT_CORRECTION_CROSS_M\s*=\s*[0-9.]+\s*$', 'MAX_MEASUREMENT_CORRECTION_CROSS_M = float(os.environ.get("UAVSAT_CORR_CROSS_M", "0.50"))', 'cross correction bound')
sub1(r'^LOSS_MEASUREMENT\s*=\s*[0-9.]+\s*$', 'LOSS_MEASUREMENT = float(os.environ.get("UAVSAT_LOSS_MEASUREMENT", "2.00"))', 'measurement loss')
sub1(r'^LOSS_NEXT_STEP\s*=\s*[0-9.]+\s*$', 'LOSS_NEXT_STEP = float(os.environ.get("UAVSAT_LOSS_NEXT_STEP", "2.00"))', 'next-step loss')
sub1(r'^LOSS_VELOCITY\s*=\s*[0-9.]+\s*$', 'LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "0.50"))', 'velocity loss')
sub1(r'^LOSS_ACCELERATION\s*=\s*[0-9.]+\s*$', 'LOSS_ACCELERATION = float(os.environ.get("UAVSAT_LOSS_ACCELERATION", "0.20"))', 'acceleration loss')
sub1(r'^TEMPORAL_LR\s*=\s*[0-9.eE+-]+\s*$', 'TEMPORAL_LR = float(os.environ.get("UAVSAT_TEMPORAL_LR", "1e-4"))', 'temporal lr')
sub1(r'^RNN_DROPOUT\s*=\s*[0-9.]+\s*$', 'RNN_DROPOUT = float(os.environ.get("UAVSAT_RNN_DROPOUT", "0.08"))', 'gru dropout')
sub1(r'^MOTION_VELOCITY_EMA_ALPHA\s*=\s*[0-9.]+\s*$', 'MOTION_VELOCITY_EMA_ALPHA = float(os.environ.get("UAVSAT_MOTION_VEL_ALPHA", "0.60"))', 'motion velocity alpha')
sub1(r'^MOTION_POLYNOMIAL_STEP_EMA_ALPHA\s*=\s*[0-9.]+\s*$', 'MOTION_POLYNOMIAL_STEP_EMA_ALPHA = float(os.environ.get("UAVSAT_MOTION_STEP_ALPHA", "0.65"))', 'motion step alpha')
sub1(r'^KALMAN_Q_PROGRESS\s*=\s*[0-9.]+\s*$', 'KALMAN_Q_PROGRESS = float(os.environ.get("UAVSAT_KALMAN_Q_PROGRESS", "1.50"))', 'kalman q progress')
sub1(r'^KALMAN_Q_CROSS\s*=\s*[0-9.]+\s*$', 'KALMAN_Q_CROSS = float(os.environ.get("UAVSAT_KALMAN_Q_CROSS", "0.40"))', 'kalman q cross')
sub1(r'^KALMAN_Q_VELOCITY\s*=\s*[0-9.]+\s*$', 'KALMAN_Q_VELOCITY = float(os.environ.get("UAVSAT_KALMAN_Q_VELOCITY", "1.00"))', 'kalman q velocity')
sub1(r'^EARLY_SCORE_SPEED_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_SPEED_WEIGHT = float(os.environ.get("UAVSAT_EARLY_SPEED_WEIGHT", "0.10"))', 'early speed weight')
sub1(r'^EARLY_SCORE_PROGRESS_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_PROGRESS_WEIGHT = float(os.environ.get("UAVSAT_EARLY_PROGRESS_WEIGHT", "0.05"))', 'early progress weight')
sub1(r'^EARLY_SCORE_HEADING_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_HEADING_WEIGHT = float(os.environ.get("UAVSAT_EARLY_HEADING_WEIGHT", "0.005"))', 'early heading weight')
sub1(r'^EARLY_SCORE_MISS_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_MISS_WEIGHT = float(os.environ.get("UAVSAT_EARLY_MISS_WEIGHT", "0.02"))', 'early miss weight')
sub1(r'^EARLY_STOP_MIN_DELTA\s*=\s*[0-9.]+\s*$', 'EARLY_STOP_MIN_DELTA = float(os.environ.get("UAVSAT_EARLY_MIN_DELTA", "0.005"))', 'early stop delta')
sub1(r'^EARLY_STOP_MIN_EPOCH\s*=\s*[0-9]+\s*$', 'EARLY_STOP_MIN_EPOCH = int(os.environ.get("UAVSAT_EARLY_MIN_EPOCH", "14"))', 'early stop min epoch')
sub1(r'^SEED\s*=\s*2033\s*$', 'SEED = int(os.environ.get("UAVSAT_SEED", "2033"))', 'seed env')

if "MOTION_RESIDUAL_FORWARD_M" not in c:
    c += '''\n# Residual temporal-motion refinement.\nMOTION_RESIDUAL_FORWARD_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_FORWARD_M", "3.0"))\nMOTION_RESIDUAL_CROSS_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_CROSS_M", "1.5"))\nMOTION_RESIDUAL_ACCEL_FORWARD_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M", "1.5"))\nMOTION_RESIDUAL_ACCEL_CROSS_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M", "1.0"))\nKALMAN_CONFIDENCE_POWER = float(os.environ.get("UAVSAT_KALMAN_CONFIDENCE_POWER", "0.5"))\n'''

compile(c, str(cfg), "exec")
cfg.write_text(c, encoding="utf-8")

# -----------------------------------------------------------------------------
# The tracker previously initialized its stable motion state at zero.  That
# nullified the cadence-aware model initialization because the rate limiter then
# had to climb from 0 m/frame.  Start both the external Kalman velocity and the
# recurrent stable motion from the training-city cadence instead.
# -----------------------------------------------------------------------------
tracker = p.with_name("robust_tracker.py")
if not tracker.exists():
    raise SystemExit(f"missing sibling robust_tracker.py: {tracker}")
t = tracker.read_text(encoding="utf-8")

old_kf_x = '        self.x = np.asarray([initial_s, initial_e, 0.0, 0.0], dtype=np.float64)\n'
new_kf_x = '''        initial_forward_speed = float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.0))
        self.x = np.asarray(
            [initial_s, initial_e, initial_forward_speed, 0.0], dtype=np.float64
        )
'''
if new_kf_x not in t:
    if t.count(old_kf_x) != 1:
        raise SystemExit("tracker patch failed: RouteKalman initial-state block")
    t = t.replace(old_kf_x, new_kf_x, 1)

old_state = '''    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)
'''
new_state = '''    _init_speed = float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.0))
    previous_velocity = torch.tensor([[_init_speed, 0.0]], dtype=torch.float32, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = previous_velocity.clone()
'''
if new_state not in t:
    count = t.count(old_state)
    if count < 2:
        raise SystemExit(f"tracker patch failed: motion-state init matches={count}")
    t = t.replace(old_state, new_state)

old_conf = '        confidence_scale = 1.0 / max(confidence * confidence, 0.05)\n'
new_conf = '''        confidence_power = float(getattr(config, "KALMAN_CONFIDENCE_POWER", 0.5))
        confidence_scale = 1.0 / max(confidence ** confidence_power, 0.20)
'''
if new_conf not in t:
    if t.count(old_conf) != 1:
        raise SystemExit("tracker patch failed: confidence scaling")
    t = t.replace(old_conf, new_conf, 1)

# A no-GRU ablation keeps the same cadence prior and external Kalman, but has no
# learned residual correction.  Do not artificially decay its motion to zero.
needle_stab = '''    pv = previous_velocity.detach()
    pa = previous_acceleration.detach()
    ps = previous_polynomial_step.detach()
    rv = raw_velocity.detach()
'''
replacement_stab = '''    pv = previous_velocity.detach()
    pa = previous_acceleration.detach()
    ps = previous_polynomial_step.detach()
    if bool(getattr(config, "EXPERIMENT_DISABLE_GRU", False)):
        return pv, pa, ps
    rv = raw_velocity.detach()
'''
if replacement_stab not in t:
    if t.count(needle_stab) != 1:
        raise SystemExit("tracker patch failed: stabilize_motion_state")
    t = t.replace(needle_stab, replacement_stab, 1)

compile(t, str(tracker), "exec")
tracker.write_text(t, encoding="utf-8")

print("[PATCH OK] temporal GRU = mean + delta + delta2 + SAT + SoftMS position + visual displacement + previous state")
print("[PATCH OK] motion head = bounded residual around previous stable motion")
print("[PATCH OK] recurrent/Kalman motion state starts from training-city cadence")
print("[PATCH OK] acceleration supervision and measurement-trusting Kalman knobs enabled")
print("[PATCH OK] no split/dual/inference gate; no prediction-position innovation")
