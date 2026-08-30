"""Autonomous v36_byTeacher tracker, stable v4.

Architecture per frame t
------------------------
previous final X_(t-1) + previous learned polynomial Delta_(t-1)
    -> predicted search center X_pre(t)
    -> MS #1: nearest forward 3x6 -> X_ms(t)

In parallel:
    Kalman(previous final X_(t-1), measurement X_ms(t)) -> X'_t
    GRU(current X_ms(t), temporal state) -> v_t, a_t, theta_t
    polynomial(v_t, a_t, theta_t) -> Delta_t

X'_t -> MS #2 full centered 6x6 -> final X_t
next search center = X_t + Delta_t

The current-frame reference coordinate is never read before final X_t exists.
Reference coordinates are used only for Route-A supervision and post-prediction
validation/test metrics.
"""

import argparse
import csv
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker_base as b
from visual_localizer import (
    CandidateBatch,
    FrozenVisualLocalizer,
    regular_grid_indices,
    soft_mean_shift,
    train_visual_retrieval_a_only,
)
from visual_model import ThreeFrameRouteStateGRU

ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)

RouteCache = b.RouteCache
build_route_cache = b.build_route_cache
load_waypoint_xy = b.load_waypoint_xy
metric_summary = b.metric_summary
resolve_device = b.resolve_device
set_seed = b.set_seed


def _tensor_xy(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, 2)


def _tensor_scalar(value, device):
    return torch.as_tensor([[float(value)]], dtype=torch.float32, device=device)


def _wrap_angle_np(angle):
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def planned_route_start(route_name, origin_lat, origin_lon):
    points = load_waypoint_xy(route_name, origin_lat, origin_lon)
    start_xy = points[0].astype(np.float64)
    direction = points[1] - points[0]
    if float(np.linalg.norm(direction)) < 1e-9:
        heading = 0.0
    else:
        heading = float(math.atan2(direction[1], direction[0]))
    return points, start_xy, heading


class XYKalman:
    """Linear XY Kalman whose positional prior is the previous FINAL position.

    This matches the requested role:
        measurement = current MS1 position
        second positional input = previous final position

    Previous polynomial displacement is kept only as the velocity state. It is
    NOT added a second time to the positional state before the visual update.
    There are no motion caps, innovation gates, or reference-dependent limits.
    """

    def __init__(self, initial_xy):
        p = np.asarray(initial_xy, dtype=np.float64).reshape(2)
        self.x = np.asarray([p[0], p[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
            ]
        )
        self.F = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.Q = np.diag(
            [
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_VELOCITY),
                float(config.KALMAN_Q_VELOCITY),
            ]
        )
        self.R = np.diag(
            [float(config.KALMAN_R_POSITION), float(config.KALMAN_R_POSITION)]
        )

    def prepare(self, previous_final_xy, previous_delta_xy):
        previous_final_xy = np.asarray(previous_final_xy, dtype=np.float64).reshape(2)
        previous_delta_xy = np.asarray(previous_delta_xy, dtype=np.float64).reshape(2)
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[:2] = previous_final_xy
        self.x[2:] = previous_delta_xy
        return self.x[:2].copy()

    def update(self, measurement_xy):
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ innovation
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x[:2].copy()


def _nearest_forward_3x6(full_centers, center_xy, heading_rad):
    """Select center-adjacent forward three rows/columns from the 6x6 grid.

    The old code selected the 18 largest forward projections. Because the 6x6
    offsets are [-3,-2,-1,0,1,2], west/south headings could select
    [-3,-2,-1] and exclude the center row entirely. Here the dominant heading
    axis is used as the longitudinal axis, and the nearest 18 candidates on or
    in front of the center plane are selected. Cardinal cases become exactly:
      east/north: 0,+1,+2
      west/south: 0,-1,-2
    times all six lateral columns.
    """

    relative = full_centers - center_xy[:, None, :]
    headings = torch.as_tensor(
        heading_rad, dtype=relative.dtype, device=relative.device
    ).reshape(-1)
    if headings.numel() == 1 and relative.shape[0] > 1:
        headings = headings.expand(relative.shape[0])
    if headings.numel() != relative.shape[0]:
        raise ValueError("heading count must match center batch size")

    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    use_x = cos_h.abs() >= sin_h.abs()
    sign_x = torch.where(cos_h >= 0, torch.ones_like(cos_h), -torch.ones_like(cos_h))
    sign_y = torch.where(sin_h >= 0, torch.ones_like(sin_h), -torch.ones_like(sin_h))

    primary = torch.where(
        use_x[:, None],
        relative[:, :, 0] * sign_x[:, None],
        relative[:, :, 1] * sign_y[:, None],
    )
    secondary = torch.where(
        use_x[:, None],
        relative[:, :, 1],
        relative[:, :, 0],
    )

    forward_mask = primary >= -1e-4
    if not bool(torch.all(forward_mask.sum(dim=1) >= int(config.MS1_CANDIDATE_COUNT))):
        raise RuntimeError("MS1 grid does not contain 18 forward candidates")

    huge = torch.full_like(primary, 1e9)
    forward_cost = torch.where(forward_mask, primary.abs(), huge)
    selected_local = torch.topk(
        forward_cost,
        k=int(config.MS1_CANDIDATE_COUNT),
        dim=1,
        largest=False,
        sorted=False,
    ).indices

    chosen_primary = torch.gather(primary, 1, selected_local)
    chosen_secondary = torch.gather(secondary, 1, selected_local)
    ordering_key = chosen_primary * 1000.0 + chosen_secondary
    order = torch.argsort(ordering_key, dim=1)
    return torch.gather(selected_local, 1, order)


def forward_3x6_candidate_batch(visual, uav_clip, center_xy, heading_rad):
    """MS #1: nearest forward 3x6 around the autonomous predicted center."""

    grid_size = int(config.MS1_BASE_GRID_SIZE)
    keep_count = int(config.MS1_CANDIDATE_COUNT)
    if grid_size != 6 or keep_count != 18:
        raise RuntimeError("MS1 must be forward 3x6 selected from a 6x6 lattice")

    full_indices = regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        center_xy,
        grid_size,
        config.SAT_STRIDE,
        visual.device,
    )
    full_centers = visual.gallery["xy"][full_indices]
    selected_local = _nearest_forward_3x6(
        full_centers=full_centers,
        center_xy=center_xy,
        heading_rad=heading_rad,
    )
    selected_indices = torch.gather(full_indices, 1, selected_local)

    centers = visual.gallery["xy"][selected_indices]
    satellite_clip = visual.gallery["clip_feat"][selected_indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)

    raw_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    raw_prob = torch.softmax(
        raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    raw_index = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=visual.device), raw_index
    ]
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


@torch.no_grad()
def ms1_forward_search(visual, uav_clip, prior_xy, heading_rad, device):
    return forward_3x6_candidate_batch(
        visual=visual,
        uav_clip=uav_clip,
        center_xy=_tensor_xy(prior_xy, device),
        heading_rad=_tensor_scalar(heading_rad, device).reshape(-1),
    )


@torch.no_grad()
def ms2_center_search(visual, uav_clip, kalman_xy, device):
    return visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=_tensor_xy(kalman_xy, device),
        grid_size=int(config.MS2_GRID_SIZE),
    )


def motion_targets_from_xy(xy, fallback_heading=0.0):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    delta = np.zeros((n, 2), dtype=np.float64)
    if n > 1:
        delta[:-1] = xy[1:] - xy[:-1]
        delta[-1] = delta[-2]

    distance = np.linalg.norm(delta, axis=1)
    previous_distance = distance.copy()
    if n > 1:
        previous_distance[1:] = distance[:-1]
        previous_distance[0] = distance[0]

    speed = 0.5 * (previous_distance + distance)
    acceleration = distance - previous_distance

    heading = np.zeros(n, dtype=np.float64)
    heading_delta = np.zeros(n, dtype=np.float64)
    last_heading = float(fallback_heading)
    for i in range(n):
        if float(distance[i]) > 1e-8:
            current_heading = float(math.atan2(delta[i, 1], delta[i, 0]))
        else:
            current_heading = last_heading
        heading[i] = current_heading
        heading_delta[i] = _wrap_angle_np(current_heading - last_heading)
        last_heading = current_heading

    if n > 1:
        polynomial_distance = speed + 0.5 * acceleration
        if not np.allclose(
            polynomial_distance[:-1], distance[:-1], rtol=1e-6, atol=1e-6
        ):
            raise RuntimeError("motion target consistency failed")

    return {
        "delta_xy": delta,
        "distance": distance,
        "speed": speed,
        "acceleration": acceleration,
        "heading": heading,
        "heading_delta": heading_delta,
    }


def temporal_loss(output, target, device):
    target_delta = _tensor_xy(target["delta_xy"], device)
    target_speed = _tensor_scalar(target["speed"], device)
    target_accel = _tensor_scalar(target["acceleration"], device)
    target_heading = _tensor_scalar(target["heading"], device)
    target_heading_delta = _tensor_scalar(target["heading_delta"], device)

    delta_loss = F.smooth_l1_loss(output.delta_xy, target_delta)
    speed_loss = F.smooth_l1_loss(output.speed_m_per_frame, target_speed)
    accel_loss = F.smooth_l1_loss(output.acceleration_m_per_frame2, target_accel)
    heading_loss = (1.0 - torch.cos(output.heading_rad - target_heading)).mean()
    heading_delta_loss = (
        1.0 - torch.cos(output.heading_delta_rad - target_heading_delta)
    ).mean()

    total = (
        float(config.LOSS_DELTA) * delta_loss
        + float(config.LOSS_SPEED) * speed_loss
        + float(config.LOSS_ACCELERATION) * accel_loss
        + float(config.LOSS_HEADING) * heading_loss
        + float(config.LOSS_HEADING_DELTA) * heading_delta_loss
    )
    return total, {
        "delta": float(delta_loss.detach().cpu()),
        "speed": float(speed_loss.detach().cpu()),
        "acceleration": float(accel_loss.detach().cpu()),
        "heading": float(heading_loss.detach().cpu()),
        "heading_delta": float(heading_delta_loss.detach().cpu()),
    }


def _initial_temporal_state(route_name, visual, device):
    _, start_xy, initial_heading = planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "prior_xy": start_xy.copy(),
        "previous_final_xy": start_xy.copy(),
        "previous_delta_xy": torch.zeros(1, 2, device=device),
        "previous_speed": torch.zeros(1, 1, device=device),
        "previous_acceleration": torch.zeros(1, 1, device=device),
        "previous_heading": _tensor_scalar(initial_heading, device),
        "previous_z": None,
        "previous_ms1_xy": None,
        "hidden": None,
        "motion_ready": False,
        "kalman": XYKalman(start_xy),
    }


def _forward_frame(model, visual, uav_clip, state, device):
    search_prior_xy = np.asarray(state["prior_xy"], dtype=np.float64).reshape(2)
    previous_final_xy = np.asarray(state["previous_final_xy"], dtype=np.float64).reshape(2)
    heading_value = float(state["previous_heading"][0, 0].detach().cpu())

    ms1 = ms1_forward_search(
        visual=visual,
        uav_clip=uav_clip,
        prior_xy=search_prior_xy,
        heading_rad=heading_value,
        device=device,
    )
    ms1_xy = ms1.softms_xy

    state["kalman"].prepare(
        previous_final_xy,
        state["previous_delta_xy"][0].detach().cpu().numpy(),
    )
    kalman_xy = state["kalman"].update(
        ms1_xy[0].detach().cpu().numpy()
    )

    output = model.forward_step(
        z_uav=ms1.z_uav,
        previous_z_uav=state["previous_z"],
        ms1_xy=ms1_xy,
        prior_xy=_tensor_xy(search_prior_xy, device),
        previous_ms1_xy=state["previous_ms1_xy"],
        previous_delta_xy=state["previous_delta_xy"],
        previous_speed=state["previous_speed"],
        previous_acceleration=state["previous_acceleration"],
        previous_heading_rad=state["previous_heading"],
        hidden=state["hidden"],
    )

    ms2 = ms2_center_search(
        visual=visual,
        uav_clip=uav_clip,
        kalman_xy=kalman_xy,
        device=device,
    )
    final_xy = ms2.softms_xy[0].detach().cpu().numpy().astype(np.float64)

    return {
        "prior_xy": search_prior_xy,
        "previous_final_xy": previous_final_xy,
        "ms1": ms1,
        "ms1_xy": ms1_xy,
        "kalman_xy": kalman_xy,
        "output": output,
        "ms2": ms2,
        "final_xy": final_xy,
    }


def _advance_state(state, frame):
    output = frame["output"]
    final_xy = frame["final_xy"]

    state["previous_z"] = frame["ms1"].z_uav
    state["previous_ms1_xy"] = frame["ms1_xy"]
    state["hidden"] = output.hidden
    state["previous_final_xy"] = final_xy.copy()

    if not bool(state["motion_ready"]):
        state["prior_xy"] = final_xy.copy()
        state["previous_delta_xy"] = torch.zeros_like(output.delta_xy)
        state["previous_speed"] = torch.zeros_like(output.speed_m_per_frame)
        state["previous_acceleration"] = torch.zeros_like(
            output.acceleration_m_per_frame2
        )
        state["motion_ready"] = True
        return

    delta_xy = output.delta_xy
    state["prior_xy"] = (
        final_xy + delta_xy[0].detach().cpu().numpy().astype(np.float64)
    )
    state["previous_delta_xy"] = delta_xy
    state["previous_speed"] = output.speed_m_per_frame
    state["previous_acceleration"] = output.acceleration_m_per_frame2
    state["previous_heading"] = output.heading_rad


def _detach_state(state):
    for key in [
        "previous_delta_xy",
        "previous_speed",
        "previous_acceleration",
        "previous_heading",
        "previous_z",
        "previous_ms1_xy",
        "hidden",
    ]:
        value = state.get(key)
        if torch.is_tensor(value):
            state[key] = value.detach()


def _target_stats(cache, fallback_heading):
    targets = motion_targets_from_xy(
        cache.gt_xy.detach().cpu().numpy(), fallback_heading=fallback_heading
    )
    d = targets["distance"][:-1]
    a = targets["acceleration"][:-1]
    return {
        "mean_step": float(np.mean(d)) if len(d) else 0.0,
        "median_step": float(np.median(d)) if len(d) else 0.0,
        "p90_step": float(np.quantile(d, 0.90)) if len(d) else 0.0,
        "mean_abs_accel": float(np.mean(np.abs(a))) if len(a) else 0.0,
    }


def _training_sequence_loss(model, optimizer, visual, cache, device):
    if len(cache) < 3:
        return float("nan"), {}

    _, _, initial_heading = planned_route_start(
        "route_A", visual.origin_lat, visual.origin_lon
    )
    targets = motion_targets_from_xy(
        cache.gt_xy.detach().cpu().numpy(), fallback_heading=initial_heading
    )
    state = _initial_temporal_state("route_A", visual, device)
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        frame0 = _forward_frame(
            model,
            visual,
            cache.uav_clip[0:1].to(device).float(),
            state,
            device,
        )
        _advance_state(state, frame0)
        _detach_state(state)

    chunk_loss = None
    chunk_count = 0
    total_values = []
    component_values = {
        key: []
        for key in ["delta", "speed", "acceleration", "heading", "heading_delta"]
    }

    for index in range(1, len(cache) - 1):
        frame = _forward_frame(
            model,
            visual,
            cache.uav_clip[index : index + 1].to(device).float(),
            state,
            device,
        )
        target = {
            "delta_xy": targets["delta_xy"][index],
            "speed": targets["speed"][index],
            "acceleration": targets["acceleration"][index],
            "heading": targets["heading"][index],
            "heading_delta": targets["heading_delta"][index],
        }
        loss, components = temporal_loss(frame["output"], target, device)
        for key, value in components.items():
            component_values[key].append(value)

        chunk_loss = loss if chunk_loss is None else chunk_loss + loss
        chunk_count += 1
        _advance_state(state, frame)

        is_last = index + 1 >= len(cache) - 1
        if chunk_count >= int(config.TBPTT_STEPS) or is_last:
            normalized = chunk_loss / float(chunk_count)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(config.GRAD_CLIP_NORM),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_values.append(float(normalized.detach().cpu()))
            _detach_state(state)
            chunk_loss = None
            chunk_count = 0

    components = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in component_values.items()
    }
    return float(np.mean(total_values)) if total_values else float("nan"), components


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else float("inf")


@torch.no_grad()
def run_route_inference(
    route_name,
    visual,
    model,
    cache,
    device,
    save_csv=True,
    measure_latency=False,
):
    model.eval()
    state = _initial_temporal_state(route_name, visual, device)

    rows = []
    final_errors = []
    prior_errors = []
    previous_final_errors = []
    ms1_errors = []
    kalman_errors = []
    timing_ms = []
    first_over_20 = None
    first_over_50 = None

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()

        if measure_latency and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        frame = _forward_frame(model, visual, uav_clip, state, device)
        if measure_latency and device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if measure_latency:
            timing_ms.append(elapsed_ms)

        reference_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        final_xy = frame["final_xy"]
        ms1_xy = frame["ms1_xy"][0].detach().cpu().numpy().astype(np.float64)

        prior_error = float(np.linalg.norm(frame["prior_xy"] - reference_xy))
        previous_final_error = float(
            np.linalg.norm(frame["previous_final_xy"] - reference_xy)
        )
        ms1_error = float(np.linalg.norm(ms1_xy - reference_xy))
        kalman_error = float(np.linalg.norm(frame["kalman_xy"] - reference_xy))
        final_error = float(np.linalg.norm(final_xy - reference_xy))

        prior_errors.append(prior_error)
        previous_final_errors.append(previous_final_error)
        ms1_errors.append(ms1_error)
        kalman_errors.append(kalman_error)
        final_errors.append(final_error)

        if first_over_20 is None and final_error > 20.0:
            first_over_20 = int(index)
        if first_over_50 is None and final_error > 50.0:
            first_over_50 = int(index)

        output = frame["output"]
        if bool(state["motion_ready"]):
            next_prior_xy = (
                final_xy + output.delta_xy[0].detach().cpu().numpy().astype(np.float64)
            )
        else:
            next_prior_xy = final_xy.copy()

        rows.append(
            {
                "frame_id": int(cache.frame_ids[index]),
                "image_path": cache.image_paths[index],
                "reference_x": float(reference_xy[0]),
                "reference_y": float(reference_xy[1]),
                "previous_final_x": float(frame["previous_final_xy"][0]),
                "previous_final_y": float(frame["previous_final_xy"][1]),
                "previous_final_error_to_current_ref_m": previous_final_error,
                "prior_x": float(frame["prior_xy"][0]),
                "prior_y": float(frame["prior_xy"][1]),
                "prior_error_m": prior_error,
                "ms1_x": float(ms1_xy[0]),
                "ms1_y": float(ms1_xy[1]),
                "ms1_error_m": ms1_error,
                "ms1_support": float(frame["ms1"].softms_support[0]),
                "kalman_x_prime_x": float(frame["kalman_xy"][0]),
                "kalman_x_prime_y": float(frame["kalman_xy"][1]),
                "kalman_error_m": kalman_error,
                "pred_speed_m_per_frame": float(output.speed_m_per_frame[0, 0]),
                "pred_acceleration_m_per_frame2": float(
                    output.acceleration_m_per_frame2[0, 0]
                ),
                "pred_heading_deg": float(
                    math.degrees(float(output.heading_rad[0, 0]))
                ),
                "pred_heading_delta_deg": float(
                    math.degrees(float(output.heading_delta_rad[0, 0]))
                ),
                "delta_x": float(output.delta_xy[0, 0]),
                "delta_y": float(output.delta_xy[0, 1]),
                "ms2_x": float(frame["ms2"].softms_xy[0, 0]),
                "ms2_y": float(frame["ms2"].softms_xy[0, 1]),
                "ms2_support": float(frame["ms2"].softms_support[0]),
                "final_x": float(final_xy[0]),
                "final_y": float(final_xy[1]),
                "next_prior_x": float(next_prior_xy[0]),
                "next_prior_y": float(next_prior_xy[1]),
                "error_final_m": final_error,
                "tracking_core_latency_ms": float(elapsed_ms),
            }
        )

        _advance_state(state, frame)
        _detach_state(state)

    summary = metric_summary(final_errors)
    summary.update(
        {
            "Architecture": ARCHITECTURE_NAME,
            "Route": route_name,
            "PreviousFinalToCurrentRef_MLE_m": _mean(previous_final_errors),
            "Prior_MLE_m": _mean(prior_errors),
            "MS1_MLE_m": _mean(ms1_errors),
            "Kalman_MLE_m": _mean(kalman_errors),
            "MS2_Final_MLE_m": _mean(final_errors),
            "FirstFinalErrorOver20mFrame": first_over_20,
            "FirstFinalErrorOver50mFrame": first_over_50,
            "MS1": "nearest forward 3x6; center-adjacent, no far-forward bias",
            "Kalman": "previous final position + current MS1 measurement",
            "GRU": "MS1/current temporal state -> v,a,theta; no uncertainty input",
            "MS2": "full centered 6x6 around Kalman X_prime",
            "ReferenceUsage": "supervision/metrics only; never runtime search center",
        }
    )

    if measure_latency and timing_ms:
        warmup = int(config.LATENCY_WARMUP_FRAMES)
        steady = np.asarray(timing_ms[warmup:], dtype=np.float64)
        if steady.size == 0:
            steady = np.asarray(timing_ms, dtype=np.float64)
        mean_ms = float(np.mean(steady))
        summary["TrackingCoreTiming"] = {
            "mean_ms": mean_ms,
            "median_ms": float(np.median(steady)),
            "p90_ms": float(np.quantile(steady, 0.90)),
            "fps": float(1000.0 / max(mean_ms, 1e-12)),
            "warmup_frames": int(min(warmup, len(timing_ms))),
        }

    if save_csv and rows:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = (
            config.OUTPUT_DIR
            / f"{route_name}_autonomous_ms1_kf_gru_ms2_frames.csv"
        )
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["CSV"] = str(csv_path)

    return summary


@torch.no_grad()
def evaluate_route(route_name, visual, model, cache, device):
    return run_route_inference(
        route_name,
        visual,
        model,
        cache,
        device,
        save_csv=False,
        measure_latency=False,
    )


def train_temporal_route_a(
    visual,
    route_a_cache,
    route_c_cache,
    device,
    epochs,
    patience_limit,
    resume=False,
):
    model = ThreeFrameRouteStateGRU().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    start_epoch = 1
    best_score = float("inf")
    best_state = None
    patience = 0

    if resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        if payload.get("architecture") != ARCHITECTURE_NAME:
            raise RuntimeError(
                "resume architecture mismatch: %r" % payload.get("architecture")
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    _, _, a_initial_heading = planned_route_start(
        "route_A", visual.origin_lat, visual.origin_lon
    )
    stats = _target_stats(route_a_cache, a_initial_heading)

    _, a_start, _ = planned_route_start(
        "route_A", visual.origin_lat, visual.origin_lon
    )
    _, c_start, _ = planned_route_start(
        "route_C", visual.origin_lat, visual.origin_lon
    )
    a_ref0 = route_a_cache.gt_xy[0].cpu().numpy().astype(np.float64)
    c_ref0 = route_c_cache.gt_xy[0].cpu().numpy().astype(np.float64)

    print(
        "Temporal training v4: Route-A ORIGINAL SPEED ONLY; C=validation; B=test",
        flush=True,
    )
    print(
        "Route-A target step mean=%.3fm median=%.3fm p90=%.3fm mean|a|=%.3f"
        % (
            stats["mean_step"],
            stats["median_step"],
            stats["p90_step"],
            stats["mean_abs_accel"],
        ),
        flush=True,
    )
    print(
        "planned-start alignment: A=%.3fm C=%.3fm"
        % (
            float(np.linalg.norm(a_start - a_ref0)),
            float(np.linalg.norm(c_start - c_ref0)),
        ),
        flush=True,
    )
    print(
        "MS1 v4 = center-adjacent forward 3x6; Kalman positional prior = previous final; frame0 = temporal warm-up",
        flush=True,
    )

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        train_loss, comp = _training_sequence_loss(
            model, optimizer, visual, route_a_cache, device
        )

        val = evaluate_route("route_C", visual, model, route_c_cache, device)
        score = float(val["MLE_m"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)

        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "model": best_state,
                    "epoch": epoch,
                    "validation_route": "route_C",
                    "validation": val,
                    "training_routes": ["route_A"],
                    "reference_usage": "Route-A motion supervision and post-prediction metrics only",
                    "training_fix": (
                        "nearest-forward 3x6; previous-final Kalman prior; "
                        "single-frame warm-up; no MS1-to-MS1 speed feedback"
                    ),
                },
                config.TEMPORAL_CHECKPOINT,
            )
        else:
            patience += 1

        torch.save(
            {
                "architecture": ARCHITECTURE_NAME,
                "model": model.state_dict(),
                "best_model": best_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "patience": patience,
            },
            config.LATEST_TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch:03d}/{epochs} A_loss={train_loss:.5f} "
            f"[d={comp.get('delta', float('nan')):.3f} "
            f"v={comp.get('speed', float('nan')):.3f} "
            f"a={comp.get('acceleration', float('nan')):.3f} "
            f"h={comp.get('heading', float('nan')):.3f}] "
            f"C: prevFinal={val['PreviousFinalToCurrentRef_MLE_m']:.1f} "
            f"prior={val['Prior_MLE_m']:.1f} "
            f"MS1={val['MS1_MLE_m']:.1f} "
            f"KF={val['Kalman_MLE_m']:.1f} "
            f"MS2={val['MLE_m']:.1f}m "
            f"P90={val['P90_m']:.1f}m "
            f"fail20={val['FirstFinalErrorOver20mFrame']} "
            f"best={best_score:.3f} patience={patience}/{patience_limit}",
            flush=True,
        )

        if epoch >= int(config.EARLY_STOP_MIN_EPOCH) and patience >= int(patience_limit):
            break

    if best_state is None:
        raise RuntimeError("temporal training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


def load_temporal_model(device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.TEMPORAL_CHECKPOINT)
    payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            "checkpoint architecture mismatch: %r != %r"
            % (payload.get("architecture"), ARCHITECTURE_NAME)
        )
    model = ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def train_pipeline(args, device):
    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume_visual),
        )

    visual = FrozenVisualLocalizer(device)
    route_a = build_route_cache(
        "route_A", config.ROUTE_ROOTS[0], visual, device
    )
    c_index = config.ROUTE_NAMES.index("route_C")
    route_c = build_route_cache(
        "route_C", config.ROUTE_ROOTS[c_index], visual, device
    )
    _, score = train_temporal_route_a(
        visual=visual,
        route_a_cache=route_a,
        route_c_cache=route_c,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume_temporal),
    )
    print(f"best Route-C validation MLE={score:.3f}m", flush=True)


def eval_pipeline(args, device):
    visual = FrozenVisualLocalizer(device)
    model = load_temporal_model(device)
    results = {
        "architecture": ARCHITECTURE_NAME,
        "train": ["route_A"],
        "validation": "route_C",
        "test": "route_B",
        "results": {},
    }

    for route_name in args.eval_routes:
        route_index = config.ROUTE_NAMES.index(route_name)
        cache = build_route_cache(
            route_name, config.ROUTE_ROOTS[route_index], visual, device
        )
        summary = run_route_inference(
            route_name,
            visual,
            model,
            cache,
            device,
            save_csv=True,
            measure_latency=bool(args.measure_latency),
        )
        results["results"][route_name] = summary
        role = (
            "validation"
            if route_name == "route_C"
            else ("test" if route_name == "route_B" else "evaluation")
        )
        print(
            f"{route_name} ({role}): "
            f"prevFinal={summary['PreviousFinalToCurrentRef_MLE_m']:.3f}m "
            f"prior={summary['Prior_MLE_m']:.3f}m "
            f"MS1={summary['MS1_MLE_m']:.3f}m "
            f"KF={summary['Kalman_MLE_m']:.3f}m "
            f"MS2/final={summary['MLE_m']:.3f}m "
            f"P90={summary['P90_m']:.3f}m "
            f"LSR@15={summary['LSR@15_pct']:.2f}%",
            flush=True,
        )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "autonomous_ms1_kf_gru_ms2_summary.json"
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary: {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "eval", "train_eval"], default="train_eval"
    )
    parser.add_argument(
        "--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS)
    )
    parser.add_argument(
        "--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M)
    )
    parser.add_argument(
        "--patience", type=int, default=int(config.EARLY_STOP_PATIENCE)
    )
    parser.add_argument("--reuse-visual", action="store_true")
    parser.add_argument("--resume-visual", action="store_true")
    parser.add_argument("--resume-temporal", action="store_true")
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument(
        "--eval-routes",
        nargs="+",
        choices=config.ROUTE_NAMES,
        default=["route_C", "route_B"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = resolve_device()
    print(f"device={device} architecture={ARCHITECTURE_NAME}", flush=True)

    if args.mode in {"train", "train_eval"}:
        train_pipeline(args, device)
    if args.mode in {"eval", "train_eval"}:
        eval_pipeline(args, device)


if __name__ == "__main__":
    main()
