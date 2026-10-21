#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_gru_gate_eval.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old = '''    return model.forward_step(
        z_uav=observation.candidate.z_uav,
        previous_z_uav=previous_z_uav,
        previous2_z_uav=previous2_z_uav,
        sat_context=observation.sat_context,
        posterior_probability=observation.posterior,
        visual_anchor_se=observation.anchor_se,
        response_variance_se=observation.response_variance_se,
        predicted_se=tensor2(predicted_se, device),
        previous_measurement_se=previous_measurement_se,
        previous_velocity_se=previous_velocity_se,
        previous_acceleration_se=previous_acceleration_se,
        previous_heading_state=previous_heading_state,
        polynomial_step_se=previous_polynomial_step_se,
        route_remaining_m=remaining,
        predicted_cross_m=predicted_cross,
        total_progress_fraction=total_fraction,
        leg_progress_fraction=leg_fraction,
        top1_distance_m=observation.top1_distance_m.reshape(-1, 1),
        softms_support=observation.candidate.softms_support,
        hidden=hidden,
    )
'''
new = '''    raw = model.forward_step(
        z_uav=observation.candidate.z_uav,
        previous_z_uav=previous_z_uav,
        previous2_z_uav=previous2_z_uav,
        sat_context=observation.sat_context,
        posterior_probability=observation.posterior,
        visual_anchor_se=observation.anchor_se,
        response_variance_se=observation.response_variance_se,
        predicted_se=tensor2(predicted_se, device),
        previous_measurement_se=previous_measurement_se,
        previous_velocity_se=previous_velocity_se,
        previous_acceleration_se=previous_acceleration_se,
        previous_heading_state=previous_heading_state,
        polynomial_step_se=previous_polynomial_step_se,
        route_remaining_m=remaining,
        predicted_cross_m=predicted_cross,
        total_progress_fraction=total_fraction,
        leg_progress_fraction=leg_fraction,
        top1_distance_m=observation.top1_distance_m.reshape(-1, 1),
        softms_support=observation.candidate.softms_support,
        hidden=hidden,
    )

    # Validation-calibrated residual gate. gain=1 is exactly the original GRU.
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
if s.count(old) != 1:
    raise SystemExit(f"ERROR: model_forward return block count={s.count(old)}")
s = s.replace(old, new, 1)

old_routes = '''    for route_name in ["route_B", "route_C"]:
'''
new_routes = '''    _route_env = __import__("os").environ.get("UAVSAT_EVAL_ROUTES", "route_B,route_C")
    _eval_routes = [x.strip() for x in _route_env.split(",") if x.strip()]
    for _name in _eval_routes:
        if _name not in config.ROUTE_NAMES:
            raise ValueError("Unknown UAVSAT_EVAL_ROUTES entry: %s" % _name)
    all_summary["eval_routes"] = list(_eval_routes)
    for route_name in _eval_routes:
'''
if s.count(old_routes) != 1:
    raise SystemExit(f"ERROR: eval-route loop count={s.count(old_routes)}")
s = s.replace(old_routes, new_routes, 1)

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] GRU residual gate + selectable eval routes installed")
