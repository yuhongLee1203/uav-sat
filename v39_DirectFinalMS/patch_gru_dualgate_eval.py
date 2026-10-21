#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_gru_dualgate_eval.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old = '''    # Validation-calibrated residual gate. gain=1 is exactly the original GRU.
    # For 0<gain<1 the recurrent branch remains active, but it only contributes
    # a bounded residual around the visual measurement and previous temporal state.
    gain = float(__import__("os").environ.get("UAVSAT_GRU_FUSION_GAIN", "1.0"))
    gain = max(0.0, min(1.0, gain))
    if gain >= 0.999999:
        return raw

    correction = gain * raw.correction_se
    measurement = observation.anchor_se + correction
    velocity = previous_velocity_se + gain * (raw.velocity_se - previous_velocity_se)
    acceleration = previous_acceleration_se + gain * (raw.acceleration_se - previous_acceleration_se)
    next_step = previous_polynomial_step_se + gain * (raw.next_step_se - previous_polynomial_step_se)
    heading = previous_heading_state[:, 0:1] + gain * (
        raw.heading_residual_rad - previous_heading_state[:, 0:1]
    )
    turn = previous_heading_state[:, 1:2] + gain * (
        raw.turn_rate_rad - previous_heading_state[:, 1:2]
    )
    variance = raw.measurement_variance_se
    state = torch.cat(
        [velocity, acceleration, heading, turn, correction, variance], dim=1
    )
    return RouteProgressGRUOutput(
        measurement_se=measurement,
        correction_se=correction,
        measurement_variance_se=variance,
        velocity_se=velocity,
        acceleration_se=acceleration,
        next_step_se=next_step,
        heading_residual_rad=heading,
        turn_rate_rad=turn,
        hidden=raw.hidden,
        state=state,
    )
'''

new = '''    # Route-A-validation calibrated split residual gates.
    # Keep the GRU active, but independently control (a) visual-position
    # correction and (b) temporal motion/heading.  This avoids one scalar gain
    # forcing the position and motion branches to have the same strength.
    # No position innovation is introduced here: measurement starts from the
    # current Forward18 SoftMS visual anchor and only adds the GRU correction.
    env = __import__("os").environ
    correction_gain = float(env.get("UAVSAT_GRU_CORRECTION_GAIN", "1.0"))
    motion_gain = float(env.get("UAVSAT_GRU_MOTION_GAIN", "1.0"))
    variance_gain = float(env.get("UAVSAT_GRU_VARIANCE_GAIN", "1.0"))
    correction_gain = max(0.0, min(1.0, correction_gain))
    motion_gain = max(0.0, min(1.0, motion_gain))
    variance_gain = max(0.0, min(1.0, variance_gain))

    correction = correction_gain * raw.correction_se
    measurement = observation.anchor_se + correction

    velocity = previous_velocity_se + motion_gain * (
        raw.velocity_se - previous_velocity_se
    )
    acceleration = previous_acceleration_se + motion_gain * (
        raw.acceleration_se - previous_acceleration_se
    )
    next_step = previous_polynomial_step_se + motion_gain * (
        raw.next_step_se - previous_polynomial_step_se
    )
    heading = previous_heading_state[:, 0:1] + motion_gain * (
        raw.heading_residual_rad - previous_heading_state[:, 0:1]
    )
    turn = previous_heading_state[:, 1:2] + motion_gain * (
        raw.turn_rate_rad - previous_heading_state[:, 1:2]
    )

    base_variance = observation.response_variance_se.clamp(
        min=float(config.KALMAN_R_MIN_VAR),
        max=float(config.KALMAN_R_MAX_VAR),
    )
    variance = base_variance + variance_gain * (
        raw.measurement_variance_se - base_variance
    )
    variance = variance.clamp(
        min=float(config.KALMAN_R_MIN_VAR),
        max=float(config.KALMAN_R_MAX_VAR),
    )

    state = torch.cat(
        [velocity, acceleration, heading, turn, correction, variance], dim=1
    )
    return RouteProgressGRUOutput(
        measurement_se=measurement,
        correction_se=correction,
        measurement_variance_se=variance,
        velocity_se=velocity,
        acceleration_se=acceleration,
        next_step_se=next_step,
        heading_residual_rad=heading,
        turn_rate_rad=turn,
        hidden=raw.hidden,
        state=state,
    )
'''

if s.count(old) != 1:
    raise SystemExit(f"ERROR: single-gain GRU gate block count={s.count(old)}")
s = s.replace(old, new, 1)

# Safety checks: this patch must not add subtraction-based position innovation.
if "visual_anchor_se - predicted_se" in s or "innovation_projection" in s:
    raise SystemExit("ERROR: forbidden position-innovation feature detected")
if "observation.anchor_se + correction" not in s:
    raise SystemExit("ERROR: direct Forward18 SoftMS visual-position path missing")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] split GRU residual gates installed: correction / motion / variance; no position innovation")
