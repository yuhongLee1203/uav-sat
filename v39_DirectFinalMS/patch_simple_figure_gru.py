#!/usr/bin/env python3
'''Patch V39 for single-city Bearing-UAV temporal ablation.

Main chain remains:
  Forward-18 SoftMS -> temporal GRU -> constrained Kalman -> final MeanShift.

Changes:
  1) keep current-frame feature explicitly;
  2) add a temporal residual adapter plus a dedicated 3-frame delta2 head;
  3) initialize recurrent/Kalman motion from current-city training cadence;
  4) use confidence-adaptive measurement-preserving residual Kalman fusion;
  5) continuously relax the final step corridor as visual confidence rises.

No held-out nav50/nav51 metric is read here.
'''
from pathlib import Path
import re
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_simple_figure_gru.py VISUAL_MODEL.py")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

if "innovation_projection" in s or "visual_anchor_se - predicted_se" in s:
    raise SystemExit("refusing runtime containing prediction-position innovation input")

# -----------------------------------------------------------------------------
# GRU inputs:
# current + delta + delta2 + SAT + SoftMS position
# + causal visual displacement + previous recurrent state = 7 blocks.
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

old_gru = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
new_gru = "        self.gru = nn.GRUCell(feature_dim * 7, hidden_dim)\n"
if new_gru not in s:
    if s.count(old_gru) != 1:
        raise SystemExit("could not identify canonical GRUCell declaration")
    s = s.replace(old_gru, new_gru, 1)

old_motion_head = "        self.motion_head = head(4)\n"
new_motion_head = '''        self.motion_head = head(4)
        # Shared temporal residual uses first + second difference.
        self.temporal_motion_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )
        # 3-frame-only residual: this branch receives delta2 alone and therefore
        # cannot help the 1-frame or 2-frame ablations.
        self.delta2_motion_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )
        # Reliability gate for the noisy second difference. Bearing-UAV routes
        # are ordered independent observations rather than native video, so a
        # third-frame residual must be allowed to fall back to first-order motion.
        self.temporal_reliability_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
'''
if new_motion_head not in s:
    if s.count(old_motion_head) != 1:
        raise SystemExit("could not identify motion_head declaration")
    s = s.replace(old_motion_head, new_motion_head, 1)

old_recurrent = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
'''
new_recurrent = '''        # Current-frame feature is the base. Temporal history enters as
        # residual first/second differences, so 3-frame context cannot blur the
        # strongest current visual evidence.
        current_h = self.uav_projection(z_uav)
        recent_h = self.delta_recent_projection(delta_recent)
        accel_h = self.delta_accel_projection(delta_accel)
        frame_count = int(getattr(config, "EXPERIMENT_FRAME_COUNT", 3))
        temporal_gate_input = torch.cat([recent_h, accel_h], dim=1)
        if frame_count >= 3:
            temporal_reliability = torch.sigmoid(
                self.temporal_reliability_head(temporal_gate_input)
            )
        else:
            temporal_reliability = torch.ones(
                (z_uav.shape[0], 1), device=z_uav.device, dtype=z_uav.dtype
            )
        gated_accel_h = accel_h * temporal_reliability

        visual_position = torch.cat(
            [
                visual_anchor_se[:, 0:1] / float(config.ROUTE_PROGRESS_SCALE_M),
                visual_anchor_se[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),
            ],
            dim=1,
        )
        if previous_measurement_se is None:
            visual_motion = torch.zeros_like(visual_anchor_se)
        else:
            visual_motion = (
                visual_anchor_se - previous_measurement_se
            ) / float(config.ROUTE_STEP_SCALE_M)

        recurrent_input = torch.cat(
            [
                current_h,
                recent_h,
                gated_accel_h,
                self.sat_projection(sat_context),
                self.visual_position_projection(visual_position),
                self.visual_motion_projection(visual_motion),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
'''
if new_recurrent not in s:
    if s.count(old_recurrent) != 1:
        raise SystemExit("could not identify canonical recurrent-input block")
    s = s.replace(old_recurrent, new_recurrent, 1)

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

        # Shared temporal residual. Frame-1 has no temporal correction;
        # Frame-2 gets first difference; Frame-3 gets first + second difference.
        temporal_input = torch.cat([recent_h, accel_h], dim=1)
        temporal_raw = self.temporal_motion_head(temporal_input)
        if frame_count <= 1:
            temporal_scale = 0.0
        elif frame_count == 2:
            temporal_scale = float(config.TEMPORAL_ADAPTER_2FRAME_SCALE)
        else:
            temporal_scale = float(config.TEMPORAL_ADAPTER_3FRAME_SCALE)
        raw_motion = raw_motion + temporal_scale * temporal_reliability * temporal_raw

        # V5 third-frame-only direct second-order state.
        if frame_count >= 3:
            delta2_direct = (
                temporal_reliability
                * torch.tanh(self.delta2_motion_head(accel_h))
            )
        else:
            delta2_direct = torch.zeros(
                accel_h.shape[0], 4, device=accel_h.device, dtype=accel_h.dtype
            )

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
        if frame_count >= 3:
            d2_scale = float(config.TEMPORAL_DELTA2_SCALE)
            a_parallel = (
                a_parallel
                + d2_scale * delta2_direct[:, 0:1]
                * float(config.TEMPORAL_DIRECT_ACCEL_FORWARD_M)
            ).clamp(
                min=-float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
                max=float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
            )
            a_cross = (
                a_cross
                + d2_scale * delta2_direct[:, 1:2]
                * float(config.TEMPORAL_DIRECT_ACCEL_CROSS_M)
            ).clamp(
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

# V5_DIRECT_NEXT_STEP_PATCH
old_next_step = '''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
        effective_heading = heading_residual
        cos_h = torch.cos(effective_heading)
        sin_h = torch.sin(effective_heading)
        next_parallel = base_forward * cos_h - base_cross * sin_h
        next_cross = base_forward * sin_h + base_cross * cos_h
        next_parallel = next_parallel.clamp(min=0.0)
        next_step = torch.cat([next_parallel, next_cross], dim=1)
        norm = torch.linalg.norm(next_step, dim=1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME) / norm, max=1.0
        )
        next_step = next_step * scale
'''
new_next_step = '''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
        effective_heading = heading_residual
        cos_h = torch.cos(effective_heading)
        sin_h = torch.sin(effective_heading)
        next_parallel = base_forward * cos_h - base_cross * sin_h
        next_cross = base_forward * sin_h + base_cross * cos_h
        if frame_count >= 3:
            d2_scale = float(config.TEMPORAL_DELTA2_SCALE)
            next_parallel = next_parallel + (
                d2_scale * delta2_direct[:, 2:3]
                * float(config.TEMPORAL_DIRECT_STEP_FORWARD_M)
            )
            next_cross = next_cross + (
                d2_scale * delta2_direct[:, 3:4]
                * float(config.TEMPORAL_DIRECT_STEP_CROSS_M)
            )
        next_parallel = next_parallel.clamp(min=0.0)
        next_step = torch.cat([next_parallel, next_cross], dim=1)
        norm = torch.linalg.norm(next_step, dim=1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME) / norm, max=1.0
        )
        next_step = next_step * scale
'''
if new_next_step not in s:
    if s.count(old_next_step) != 1:
        raise SystemExit(f"V5 patch failed: next-step block matches={s.count(old_next_step)}")
    s = s.replace(old_next_step, new_next_step, 1)

old_init = '''        # Avoid a large arbitrary initial speed. A small positive bias makes the
        # state numerically well behaved while acquisition provides the initial
        # displacement evidence.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        init_speed = 0.75
        self.motion_head[-1].bias.data[0] = math.log(math.exp(init_speed) - 1.0)
'''
new_init = '''        # All motion heads start as zero residual around the city-cadence state.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        nn.init.zeros_(self.temporal_motion_head[-1].weight)
        nn.init.zeros_(self.temporal_motion_head[-1].bias)
        nn.init.zeros_(self.delta2_motion_head[-1].weight)
        nn.init.zeros_(self.delta2_motion_head[-1].bias)
        nn.init.zeros_(self.temporal_reliability_head[-1].weight)
        nn.init.constant_(self.temporal_reliability_head[-1].bias, -1.0986122886681098)
'''
if new_init not in s:
    if s.count(old_init) != 1:
        raise SystemExit("could not identify canonical motion-head initialization")
    s = s.replace(old_init, new_init, 1)

required_model = [
    "self.gru = nn.GRUCell(feature_dim * 7, hidden_dim)",
    "current_h = self.uav_projection(z_uav)",
    "self.visual_motion_projection(visual_motion)",
    "self.temporal_motion_head",
    "self.delta2_motion_head",
    "self.temporal_reliability_head",
    "TEMPORAL_DELTA2_SCALE",
    "MOTION_RESIDUAL_FORWARD_M",
]
missing = [x for x in required_model if x not in s]
if missing:
    raise SystemExit("GRU patch audit failed: missing " + repr(missing))

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")

# -----------------------------------------------------------------------------
# Runtime configuration knobs.
# -----------------------------------------------------------------------------
cfg = p.with_name("config.py")
c = cfg.read_text(encoding="utf-8")

def sub1(pattern, replacement, label):
    global c
    c2, n = re.subn(pattern, replacement, c, count=1, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"config patch failed for {label}: matches={n}")
    c = c2

sub1(r'^MAX_MEASUREMENT_CORRECTION_PARALLEL_M\s*=\s*[0-9.]+\s*$', 'MAX_MEASUREMENT_CORRECTION_PARALLEL_M = float(os.environ.get("UAVSAT_CORR_PARALLEL_M", "0.75"))', 'parallel correction bound')
sub1(r'^MAX_MEASUREMENT_CORRECTION_CROSS_M\s*=\s*[0-9.]+\s*$', 'MAX_MEASUREMENT_CORRECTION_CROSS_M = float(os.environ.get("UAVSAT_CORR_CROSS_M", "0.50"))', 'cross correction bound')
sub1(r'^LOSS_MEASUREMENT\s*=\s*[0-9.]+\s*$', 'LOSS_MEASUREMENT = float(os.environ.get("UAVSAT_LOSS_MEASUREMENT", "2.0"))', 'measurement loss')
sub1(r'^LOSS_NEXT_STEP\s*=\s*[0-9.]+\s*$', 'LOSS_NEXT_STEP = float(os.environ.get("UAVSAT_LOSS_NEXT_STEP", "2.5"))', 'next-step loss')
sub1(r'^LOSS_VELOCITY\s*=\s*[0-9.]+\s*$', 'LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "0.0"))', 'velocity loss')
sub1(r'^LOSS_ACCELERATION\s*=\s*[0-9.]+\s*$', 'LOSS_ACCELERATION = float(os.environ.get("UAVSAT_LOSS_ACCELERATION", "0.0"))', 'acceleration loss')
sub1(r'^TEMPORAL_LR\s*=\s*[0-9.eE+-]+\s*$', 'TEMPORAL_LR = float(os.environ.get("UAVSAT_TEMPORAL_LR", "8e-5"))', 'temporal lr')
sub1(r'^RNN_DROPOUT\s*=\s*[0-9.]+\s*$', 'RNN_DROPOUT = float(os.environ.get("UAVSAT_RNN_DROPOUT", "0.05"))', 'gru dropout')
sub1(r'^MOTION_VELOCITY_EMA_ALPHA\s*=\s*[0-9.]+\s*$', 'MOTION_VELOCITY_EMA_ALPHA = float(os.environ.get("UAVSAT_MOTION_VEL_ALPHA", "0.65"))', 'motion velocity alpha')
sub1(r'^MOTION_POLYNOMIAL_STEP_EMA_ALPHA\s*=\s*[0-9.]+\s*$', 'MOTION_POLYNOMIAL_STEP_EMA_ALPHA = float(os.environ.get("UAVSAT_MOTION_STEP_ALPHA", "0.70"))', 'motion step alpha')
sub1(r'^KALMAN_Q_PROGRESS\s*=\s*[0-9.]+\s*$', 'KALMAN_Q_PROGRESS = float(os.environ.get("UAVSAT_KALMAN_Q_PROGRESS", "1.50"))', 'kalman q progress')
sub1(r'^KALMAN_Q_CROSS\s*=\s*[0-9.]+\s*$', 'KALMAN_Q_CROSS = float(os.environ.get("UAVSAT_KALMAN_Q_CROSS", "0.40"))', 'kalman q cross')
sub1(r'^KALMAN_Q_VELOCITY\s*=\s*[0-9.]+\s*$', 'KALMAN_Q_VELOCITY = float(os.environ.get("UAVSAT_KALMAN_Q_VELOCITY", "1.00"))', 'kalman q velocity')
sub1(r'^EARLY_SCORE_SPEED_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_SPEED_WEIGHT = float(os.environ.get("UAVSAT_EARLY_SPEED_WEIGHT", "0.05"))', 'early speed weight')
sub1(r'^EARLY_SCORE_PROGRESS_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_PROGRESS_WEIGHT = float(os.environ.get("UAVSAT_EARLY_PROGRESS_WEIGHT", "0.03"))', 'early progress weight')
sub1(r'^EARLY_SCORE_HEADING_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_HEADING_WEIGHT = float(os.environ.get("UAVSAT_EARLY_HEADING_WEIGHT", "0.002"))', 'early heading weight')
sub1(r'^EARLY_SCORE_MISS_WEIGHT\s*=\s*[0-9.]+\s*$', 'EARLY_SCORE_MISS_WEIGHT = float(os.environ.get("UAVSAT_EARLY_MISS_WEIGHT", "0.02"))', 'early miss weight')
sub1(r'^EARLY_STOP_MIN_DELTA\s*=\s*[0-9.]+\s*$', 'EARLY_STOP_MIN_DELTA = float(os.environ.get("UAVSAT_EARLY_MIN_DELTA", "0.003"))', 'early stop delta')
sub1(r'^EARLY_STOP_MIN_EPOCH\s*=\s*[0-9]+\s*$', 'EARLY_STOP_MIN_EPOCH = int(os.environ.get("UAVSAT_EARLY_MIN_EPOCH", "10"))', 'early stop min epoch')
sub1(r'^SEED\s*=\s*2033\s*$', 'SEED = int(os.environ.get("UAVSAT_SEED", "2033"))', 'seed env')

c += '''
# Minimal, identifiable temporal objective.  Position and one-step displacement
# are the two geometric labels available from the ordered Bearing-UAV samples.
# Variance NLL is retained only to calibrate the learned variance consumed by
# the Kalman update.  Derived velocity/acceleration/heading pseudo-labels are
# deliberately not separate objectives.
LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "0.0"))
LOSS_ACCELERATION = float(os.environ.get("UAVSAT_LOSS_ACCELERATION", "0.0"))
LOSS_HEADING = float(os.environ.get("UAVSAT_LOSS_HEADING", "0.0"))
LOSS_TURN_RATE = 0.0
LOSS_SPEED = 0.0
LOSS_CROSS_MOTION_REG = 0.0
LOSS_PROGRESS = 0.0
LOSS_ACQUISITION = 0.0
LOSS_VARIANCE_NLL = float(os.environ.get("UAVSAT_LOSS_VARIANCE_NLL", "0.05"))

# Single-city residual temporal refinement.
MOTION_RESIDUAL_FORWARD_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_FORWARD_M", "2.5"))
MOTION_RESIDUAL_CROSS_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_CROSS_M", "1.25"))
MOTION_RESIDUAL_ACCEL_FORWARD_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M", "1.25"))
MOTION_RESIDUAL_ACCEL_CROSS_M = float(os.environ.get("UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M", "0.75"))
TEMPORAL_ADAPTER_2FRAME_SCALE = float(os.environ.get("UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE", "0.45"))
TEMPORAL_ADAPTER_3FRAME_SCALE = float(os.environ.get("UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE", "1.00"))
TEMPORAL_DELTA2_SCALE = float(os.environ.get("UAVSAT_TEMPORAL_DELTA2_SCALE", "1.00"))
TEMPORAL_DIRECT_ACCEL_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M", "1.25"))
TEMPORAL_DIRECT_ACCEL_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M", "0.75"))
TEMPORAL_DIRECT_STEP_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M", "2.00"))
TEMPORAL_DIRECT_STEP_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M", "1.00"))

# Confidence-adaptive measurement-preserving residual Kalman.
KALMAN_CONFIDENCE_POWER = float(os.environ.get("UAVSAT_KALMAN_CONFIDENCE_POWER", "0.50"))
KALMAN_PRIOR_BLEND_BASE = float(os.environ.get("UAVSAT_KALMAN_PRIOR_BLEND_BASE", "0.00"))
KALMAN_PRIOR_BLEND_LOWCONF_GAIN = float(os.environ.get("UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN", "0.18"))
KALMAN_PRIOR_BLEND_MAX = float(os.environ.get("UAVSAT_KALMAN_PRIOR_BLEND_MAX", "0.30"))
KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF = float(os.environ.get("UAVSAT_KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF", "0.60"))
KALMAN_STEP_RELAX_CONFIDENCE = float(os.environ.get("UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE", "0.55"))
KALMAN_STEP_RELAX_WIDTH = float(os.environ.get("UAVSAT_KALMAN_STEP_RELAX_WIDTH", "0.08"))
KALMAN_STEP_VISUAL_SLACK_M = float(os.environ.get("UAVSAT_KALMAN_STEP_VISUAL_SLACK_M", "3.0"))
'''

compile(c, str(cfg), "exec")
cfg.write_text(c, encoding="utf-8")

# -----------------------------------------------------------------------------
# Tracker: cadence initialization + confidence-adaptive residual Kalman.
# -----------------------------------------------------------------------------
tracker = p.with_name("robust_tracker.py")
t = tracker.read_text(encoding="utf-8")

old_kf_x = '        self.x = np.asarray([initial_s, initial_e, 0.0, 0.0], dtype=np.float64)\n'
new_kf_x = '''        initial_forward_speed = float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.0))
        self.x = np.asarray(
            [initial_s, initial_e, initial_forward_speed, 0.0], dtype=np.float64
        )
'''
if new_kf_x not in t:
    if t.count(old_kf_x) != 1:
        raise SystemExit("tracker patch failed: RouteKalman initial state")
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
        raise SystemExit(f"tracker patch failed: top-level motion-state matches={count}")
    t = t.replace(old_state, new_state)

old_train_state = '''        previous_velocity = torch.zeros(1, 2, device=device)
        previous_acceleration = torch.zeros(1, 2, device=device)
        previous_heading_state = torch.zeros(1, 2, device=device)
        previous_poly_step = torch.zeros(1, 2, device=device)
'''
new_train_state = '''        _init_speed = float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.0))
        previous_velocity = torch.tensor([[_init_speed, 0.0]], dtype=torch.float32, device=device)
        previous_acceleration = torch.zeros(1, 2, device=device)
        previous_heading_state = torch.zeros(1, 2, device=device)
        previous_poly_step = previous_velocity.clone()
'''
if new_train_state not in t:
    if t.count(old_train_state) != 1:
        raise SystemExit("tracker patch failed: training motion-state init")
    t = t.replace(old_train_state, new_train_state, 1)

old_conf = '        confidence_scale = 1.0 / max(confidence * confidence, 0.05)\n'
new_conf = '''        confidence_power = float(getattr(config, "KALMAN_CONFIDENCE_POWER", 0.50))
        confidence_scale = 1.0 / max(confidence ** confidence_power, 0.25)
'''
if new_conf not in t:
    if t.count(old_conf) != 1:
        raise SystemExit("tracker patch failed: confidence scaling")
    t = t.replace(old_conf, new_conf, 1)

old_used = '        self.last_used_measurement = z.copy()\n'
new_used = '        self.last_used_measurement = raw_z.copy()\n'
if new_used not in t:
    if t.count(old_used) != 1:
        raise SystemExit("tracker patch failed: raw previous measurement")
    t = t.replace(old_used, new_used, 1)

old_post = '''        candidate_x[:2] = prior_position + bounded_correction
        self.last_posterior_projection_m = float(
            np.linalg.norm(posterior_correction - bounded_correction)
        )
'''
new_post = '''        candidate_x[:2] = prior_position + bounded_correction

        # Smooth confidence-adaptive residual fusion. High-confidence visual
        # measurements stay almost untouched; the prior contributes gradually
        # only as confidence falls below the validation-selected cutoff.
        blend_cutoff = max(
            float(config.KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF), 1e-3
        )
        lowconf = float(np.clip(
            (blend_cutoff - confidence) / blend_cutoff, 0.0, 1.0
        ))
        prior_blend = float(config.KALMAN_PRIOR_BLEND_BASE) + (
            lowconf * lowconf
        ) * float(config.KALMAN_PRIOR_BLEND_LOWCONF_GAIN)
        prior_blend = float(np.clip(
            prior_blend, 0.0, float(config.KALMAN_PRIOR_BLEND_MAX)
        ))
        candidate_x[:2] = raw_z + prior_blend * (candidate_x[:2] - raw_z)

        self.last_posterior_projection_m = float(
            np.linalg.norm(posterior_correction - bounded_correction)
        )
'''
if new_post not in t:
    if t.count(old_post) != 1:
        raise SystemExit("tracker patch failed: posterior residual fusion")
    t = t.replace(old_post, new_post, 1)

old_allowed = '''        step_norm = float(np.linalg.norm(total_step))
        self.last_step_limited = bool(step_norm > allowed_step + 1e-9)
'''
new_allowed = '''        # Continuous confidence relaxation instead of a hard inference gate.
        # At high confidence the allowed corridor approaches the observed visual
        # step; at low confidence the original motion constraint remains active.
        visual_step = float(np.linalg.norm(raw_z - self.last_previous_position))
        relax_center = float(config.KALMAN_STEP_RELAX_CONFIDENCE)
        relax_width = max(float(config.KALMAN_STEP_RELAX_WIDTH), 1e-3)
        visual_weight = 1.0 / (
            1.0 + math.exp(-(confidence - relax_center) / relax_width)
        )
        visual_allowed = visual_step + float(config.KALMAN_STEP_VISUAL_SLACK_M)
        if visual_allowed > allowed_step:
            allowed_step = allowed_step + visual_weight * (
                visual_allowed - allowed_step
            )

        step_norm = float(np.linalg.norm(total_step))
        self.last_step_limited = bool(step_norm > allowed_step + 1e-9)
'''
if new_allowed not in t:
    if t.count(old_allowed) != 1:
        raise SystemExit("tracker patch failed: confidence-aware step corridor")
    t = t.replace(old_allowed, new_allowed, 1)

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

t = t.replace(
    '"Route-A GT mean forward step=%.3fm/frame p90=%.3fm/frame"',
    '"City training-sequence mean forward step=%.3fm/frame p90=%.3fm/frame"',
)

required_tracker = [
    "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF",
    "KALMAN_STEP_RELAX_WIDTH",
    "self.last_used_measurement = raw_z.copy()",
    "INIT_FORWARD_SPEED_M_PER_FRAME",
]
missing = [x for x in required_tracker if x not in t]
if missing:
    raise SystemExit("tracker patch audit failed: " + repr(missing))

compile(t, str(tracker), "exec")
tracker.write_text(t, encoding="utf-8")

print("[PATCH OK] GRU = current + delta + delta2 + SAT + SoftMS position + visual displacement + previous state")
print("[PATCH OK] V5 3-frame direct acceleration + next-step correction")
print("[PATCH OK] training/validation/inference motion starts from current-city cadence")
print("[PATCH OK] Kalman = smooth confidence-adaptive measurement-preserving residual fusion")
print("[PATCH OK] step corridor = continuous confidence relaxation")
print("[PATCH OK] no held-out navigation result is read by this patch")
