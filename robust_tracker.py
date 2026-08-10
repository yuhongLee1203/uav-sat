import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import RouteDataset, meters_from_latlon
from visual_localizer import (
    CandidateBatch,
    FrozenVisualLocalizer,
    hard_mean_shift,
    train_visual_retrieval_a_only,
)
from visual_model import (
    RouteCoordinateGRU,
    initial_covariance_torch,
    kalman_predict_torch,
    kalman_update_torch,
    constrain_route_state_torch,
)

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "TimestampAwareRouteCoordinateGRU_FilterPyKalman"


@dataclass
class RouteWaypoint:
    order: int
    role: str
    frame_index: int
    timestamp_ns: int
    xy: torch.Tensor


@dataclass
class RouteLeg:
    index: int
    start_waypoint: RouteWaypoint
    end_waypoint: RouteWaypoint

    @property
    def start_frame(self):
        return int(self.start_waypoint.frame_index)

    @property
    def end_frame(self):
        return int(self.end_waypoint.frame_index)

    @property
    def start_xy(self):
        return self.start_waypoint.xy

    @property
    def end_xy(self):
        return self.end_waypoint.xy

    @property
    def vector(self):
        return self.end_xy - self.start_xy

    @property
    def length(self):
        return float(torch.linalg.norm(self.vector).item())

    @property
    def unit(self):
        return self.vector / max(self.length, 1e-8)

    @property
    def normal(self):
        unit = self.unit
        return torch.tensor(
            [-float(unit[1]), float(unit[0])], dtype=torch.float32
        )

    @property
    def duration_seconds(self):
        delta_ns = (
            int(self.end_waypoint.timestamp_ns)
            - int(self.start_waypoint.timestamp_ns)
        )
        return max(float(delta_ns) / 1e9, float(config.MIN_DT_SECONDS))


@dataclass
class WaypointManifest:
    route_name: str
    waypoints: list
    legs: list


@dataclass
class BackboneCache:
    route_name: str
    frame_ids: torch.Tensor
    timestamp_ns: torch.Tensor
    gt_xy: torch.Tensor
    uav_clip: torch.Tensor

    def __len__(self):
        return int(self.gt_xy.shape[0])


@dataclass
class MotionEnvelope:
    nominal_forward_speed_mps: float
    forward_speed_limit_mps: float
    cross_speed_limit_mps: float
    training_leg_speeds_mps: list

    def as_dict(self):
        return {
            "nominal_forward_speed_mps": float(self.nominal_forward_speed_mps),
            "forward_speed_limit_mps": float(self.forward_speed_limit_mps),
            "cross_speed_limit_mps": float(self.cross_speed_limit_mps),
            "source": (
                "Route-A temporal-training waypoint legs only; "
                "speed=leg_length/timestamp_duration; upper=median+3*MAD"
            ),
            "training_leg_speeds_mps": [
                float(value) for value in self.training_leg_speeds_mps
            ],
        }


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cache_dtype():
    return torch.float16 if config.FEATURE_CACHE_DTYPE == "float16" else torch.float32


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def load_waypoint_manifest(route_name, origin_lat, origin_lon):
    """Load EVERY waypoint and rebuild every adjacent leg.

    straight_legs is intentionally not the source of truth. If the user adds a
    waypoint to the JSON but forgets to regenerate straight_legs, the waypoint
    is still used by this code.
    """
    path = Path(config.WAYPOINT_FILES[route_name])
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = sorted(
        payload["waypoints"], key=lambda item: int(item["waypoint_order"])
    )
    if len(raw) < 2:
        raise RuntimeError(f"{route_name}: fewer than two waypoints")

    waypoints = []
    for item in raw:
        if item.get("timestamp_ns") is None:
            raise RuntimeError(
                f"{route_name} W{item['waypoint_order']} has no timestamp_ns"
            )
        xy = torch.tensor(
            meters_from_latlon(
                item["latitude"],
                item["longitude"],
                origin_lat,
                origin_lon,
            ),
            dtype=torch.float32,
        )
        waypoints.append(
            RouteWaypoint(
                order=int(item["waypoint_order"]),
                role=str(item.get("role", "waypoint")),
                frame_index=int(item["frame_index"]),
                timestamp_ns=int(item["timestamp_ns"]),
                xy=xy,
            )
        )

    legs = [
        RouteLeg(index=i, start_waypoint=waypoints[i], end_waypoint=waypoints[i + 1])
        for i in range(len(waypoints) - 1)
    ]

    print(
        f"{route_name}: loaded ALL {len(waypoints)} waypoints -> {len(legs)} legs",
        flush=True,
    )
    print(
        "  "
        + " -> ".join(
            f"W{wp.order}[f{wp.frame_index}]" for wp in waypoints
        ),
        flush=True,
    )

    stored = payload.get("straight_legs")
    if isinstance(stored, list) and len(stored) != len(legs):
        print(
            f"WARNING: {route_name} straight_legs={len(stored)} but "
            f"waypoints imply {len(legs)}. Using ALL waypoints and rebuilding legs.",
            flush=True,
        )

    return WaypointManifest(route_name=route_name, waypoints=waypoints, legs=legs)


def active_leg_for_frame(manifest, frame_id):
    frame_id = int(frame_id)
    for index, leg in enumerate(manifest.legs):
        is_last = index == len(manifest.legs) - 1
        if (
            leg.start_frame <= frame_id < leg.end_frame
            or (is_last and frame_id <= leg.end_frame)
        ):
            return leg
    if frame_id < manifest.legs[0].start_frame:
        return manifest.legs[0]
    return manifest.legs[-1]


def interpolate_timestamp_ns(manifest, frame_id):
    """Get per-frame time from waypoint sensor timestamps.

    The waypoint timestamps come from the same source sensor timeline. This is
    only used when the RouteDataset item does not directly expose a timestamp.
    """
    frame_id = int(frame_id)
    if frame_id <= manifest.waypoints[0].frame_index:
        return int(manifest.waypoints[0].timestamp_ns)
    if frame_id >= manifest.waypoints[-1].frame_index:
        return int(manifest.waypoints[-1].timestamp_ns)

    leg = active_leg_for_frame(manifest, frame_id)
    span = max(leg.end_frame - leg.start_frame, 1)
    ratio = float(frame_id - leg.start_frame) / float(span)
    return int(
        round(
            leg.start_waypoint.timestamp_ns
            + ratio
            * (
                leg.end_waypoint.timestamp_ns
                - leg.start_waypoint.timestamp_ns
            )
        )
    )


def split_route_a_legs(manifest):
    count = len(manifest.legs)
    train_count = max(
        1, int(count * float(config.TEMPORAL_TRAIN_LEG_FRACTION))
    )
    val_count = max(
        1, int(count * float(config.TEMPORAL_VAL_LEG_FRACTION))
    )
    if train_count + val_count >= count:
        val_count = max(1, count - train_count - 1)

    train_legs = manifest.legs[:train_count]
    val_legs = manifest.legs[train_count : train_count + val_count]
    test_legs = manifest.legs[train_count + val_count :]

    return {
        "train": (train_legs[0].start_frame, train_legs[-1].end_frame),
        "val": (val_legs[0].start_frame, val_legs[-1].end_frame),
        "test": (
            (test_legs[0].start_frame, test_legs[-1].end_frame)
            if test_legs
            else (val_legs[-1].end_frame, val_legs[-1].end_frame)
        ),
        "train_leg_count": len(train_legs),
        "val_leg_count": len(val_legs),
        "test_leg_count": len(test_legs),
    }


def route_frame_indices(cache, start_frame, end_frame):
    values = cache.frame_ids.numpy()
    mask = (values >= int(start_frame)) & (values <= int(end_frame))
    return np.nonzero(mask)[0].tolist()


def xy_to_route(xy, leg):
    xy = torch.as_tensor(xy, dtype=torch.float32)
    relative = xy - leg.start_xy
    return torch.stack(
        [torch.dot(relative, leg.unit), torch.dot(relative, leg.normal)]
    )


def xy_batch_to_route(xy, leg):
    start = leg.start_xy.to(xy.device).reshape(1, 2)
    unit = leg.unit.to(xy.device).reshape(1, 2)
    normal = leg.normal.to(xy.device).reshape(1, 2)
    relative = xy - start
    return torch.stack(
        [
            (relative * unit).sum(dim=1),
            (relative * normal).sum(dim=1),
        ],
        dim=1,
    )


def route_to_xy_torch(progress, cross_track, leg, device, dtype):
    start = leg.start_xy.to(device=device, dtype=dtype).reshape(1, 2)
    unit = leg.unit.to(device=device, dtype=dtype).reshape(1, 2)
    normal = leg.normal.to(device=device, dtype=dtype).reshape(1, 2)
    return (
        start
        + progress.reshape(-1, 1) * unit
        + cross_track.reshape(-1, 1) * normal
    )


def route_to_xy_numpy(progress, cross_track, leg):
    return (
        leg.start_xy.numpy().astype(np.float64)
        + float(progress) * leg.unit.numpy().astype(np.float64)
        + float(cross_track) * leg.normal.numpy().astype(np.float64)
    )


@torch.no_grad()
def build_backbone_cache(route_name, root, visual, manifest, device):
    dataset = RouteDataset(
        Path(root),
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )
    frame_rows = []
    timestamp_rows = []
    gt_rows = []
    clip_rows = []
    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        clip_rows.append(
            visual.encode_uav_clip(uav).detach().cpu().to(cache_dtype())
        )
        gt_rows.append(torch.stack([item["xy"].float() for item in items]))
        for item in items:
            frame_id = parse_frame_id(item["frame_id"])
            frame_rows.append(frame_id)
            timestamp_rows.append(interpolate_timestamp_ns(manifest, frame_id))

        if start == 0 or end == len(dataset) or (start // batch_size) % 20 == 0:
            print(
                f"{route_name} backbone cache: {end}/{len(dataset)}",
                flush=True,
            )

    return BackboneCache(
        route_name=route_name,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        timestamp_ns=torch.tensor(timestamp_rows, dtype=torch.long),
        gt_xy=torch.cat(gt_rows).float(),
        uav_clip=torch.cat(clip_rows),
    )


def robust_upper(values, minimum):
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_sigma = 1.4826 * mad
    upper = median + 3.0 * robust_sigma
    if robust_sigma < 1e-6:
        upper = max(upper, float(array.max()))
    return max(float(minimum), upper), median


def derive_motion_envelope(cache, manifest, split):
    train_start, train_end = split["train"]
    train_legs = [
        leg
        for leg in manifest.legs
        if leg.start_frame >= train_start and leg.end_frame <= train_end
    ]
    leg_speeds = [
        leg.length / leg.duration_seconds
        for leg in train_legs
        if leg.duration_seconds > 0 and leg.length > 0
    ]
    if not leg_speeds:
        raise RuntimeError("No Route-A training leg speeds")

    forward_limit, nominal = robust_upper(
        leg_speeds, config.MIN_FORWARD_SPEED_MPS
    )

    # Estimate cross-track speed over windows >= 1 second to avoid treating
    # quantized one-frame GPS jumps as physical velocity.
    cross_speeds = []
    last_by_leg = {}
    for index in route_frame_indices(cache, train_start, train_end):
        frame_id = int(cache.frame_ids[index].item())
        timestamp_ns = int(cache.timestamp_ns[index].item())
        leg = active_leg_for_frame(manifest, frame_id)
        d_value = float(xy_to_route(cache.gt_xy[index], leg)[1])
        previous = last_by_leg.get(leg.index)
        if previous is None:
            last_by_leg[leg.index] = (timestamp_ns, d_value)
            continue
        old_time, old_d = previous
        elapsed = (timestamp_ns - old_time) / 1e9
        if elapsed >= 1.0:
            cross_speeds.append(abs(d_value - old_d) / elapsed)
            last_by_leg[leg.index] = (timestamp_ns, d_value)

    if cross_speeds:
        cross_limit, _ = robust_upper(
            cross_speeds, config.MIN_CROSS_SPEED_MPS
        )
    else:
        cross_limit = float(config.MIN_CROSS_SPEED_MPS)

    envelope = MotionEnvelope(
        nominal_forward_speed_mps=nominal,
        forward_speed_limit_mps=forward_limit,
        cross_speed_limit_mps=cross_limit,
        training_leg_speeds_mps=leg_speeds,
    )

    print("Route-A timestamp-aware motion envelope:", flush=True)
    print(f"  nominal forward = {nominal:.3f} m/s", flush=True)
    print(f"  robust forward max = {forward_limit:.3f} m/s", flush=True)
    print(f"  robust cross max = {cross_limit:.3f} m/s", flush=True)
    print(
        "  training leg speeds = "
        + ", ".join(f"{value:.2f}" for value in leg_speeds),
        flush=True,
    )
    return envelope


def gallery_route_coordinates(visual, leg):
    gallery_xy = visual.gallery["xy"]
    start = leg.start_xy.to(gallery_xy.device)
    unit = leg.unit.to(gallery_xy.device)
    normal = leg.normal.to(gallery_xy.device)
    relative = gallery_xy - start[None, :]
    return (
        (relative * unit[None, :]).sum(dim=1),
        (relative * normal[None, :]).sum(dim=1),
    )


def candidate_indices_forward(
    visual, predicted_progress, previous_progress, leg, motion_envelope
):
    progress, cross_track = gallery_route_coordinates(visual, leg)
    count = int(config.ROUTE_CANDIDATE_COUNT)
    previous_progress = min(max(0.0, float(previous_progress)), leg.length)
    predicted_progress = min(
        max(previous_progress, float(predicted_progress)), leg.length
    )
    search_upper = min(
        leg.length + float(config.ROUTE_ENDPOINT_PADDING_M),
        previous_progress
        + motion_envelope.forward_speed_limit_mps
        * float(config.SEARCH_LOOKAHEAD_SECONDS),
    )

    chosen = None
    for width_scale in (1.0, 1.5, 2.0, 4.0):
        mask = (
            (progress >= previous_progress)
            & (progress <= search_upper)
            & (
                cross_track.abs()
                <= float(config.ROUTE_CORRIDOR_HALF_WIDTH_M) * width_scale
            )
        )
        if int(mask.sum().item()) >= count:
            chosen = mask
            break
    if chosen is None:
        chosen = (progress >= previous_progress) & (progress <= search_upper)

    valid = torch.nonzero(chosen, as_tuple=False).flatten()
    if valid.numel() == 0:
        terminal_start = max(
            0.0, leg.length - float(config.ROUTE_ENDPOINT_PADDING_M)
        )
        terminal_end = leg.length + float(config.ROUTE_ENDPOINT_PADDING_M)
        terminal = (
            (progress >= terminal_start)
            & (progress <= terminal_end)
            & (
                cross_track.abs()
                <= float(config.ROUTE_CORRIDOR_HALF_WIDTH_M) * 4.0
            )
        )
        valid = torch.nonzero(terminal, as_tuple=False).flatten()
    if valid.numel() == 0:
        raise RuntimeError(
            "No legal SAT patch in forward route support: "
            f"leg={leg.index}, progress={previous_progress:.2f}, "
            f"search_upper={search_upper:.2f}"
        )

    valid_progress = progress[valid]
    valid_cross = cross_track[valid]
    along_scale = max(motion_envelope.nominal_forward_speed_mps, 1.0)
    cross_scale = max(motion_envelope.cross_speed_limit_mps, 1.0)
    cost = (
        ((valid_progress - predicted_progress) / along_scale).square()
        + (valid_cross / cross_scale).square()
    )
    actual = min(count, int(valid.numel()))
    order = torch.topk(cost, k=actual, largest=False).indices
    selected = valid[order]
    if selected.numel() < count:
        selected = torch.cat(
            [selected, selected[-1].repeat(count - selected.numel())], dim=0
        )
    return selected.reshape(1, -1)


@torch.no_grad()
def candidate_batch_from_indices(visual, uav_clip, indices):
    device = visual.device
    indices = indices.to(device)
    centers = visual.gallery["xy"][indices]
    satellite_clip = visual.gallery["clip_feat"][indices]
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
        torch.arange(centers.shape[0], device=device), raw_index
    ]
    hardms_xy, hardms_support = hard_mean_shift(
        raw_logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
    )
    return CandidateBatch(
        indices=indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        hardms_xy=hardms_xy,
        hardms_support=hardms_support,
    )


def candidate_centers_to_route(centers, leg):
    start = leg.start_xy.to(centers.device).reshape(1, 1, 2)
    unit = leg.unit.to(centers.device).reshape(1, 1, 2)
    normal = leg.normal.to(centers.device).reshape(1, 1, 2)
    relative = centers - start
    return torch.stack(
        [
            (relative * unit).sum(dim=2),
            (relative * normal).sum(dim=2),
        ],
        dim=2,
    )


def clamp_dt_seconds(value):
    return min(
        max(float(value), float(config.MIN_DT_SECONDS)),
        float(config.MAX_DT_SECONDS),
    )


def initial_route_state_torch(device):
    # Start stationary. Visual evidence must establish motion; this prevents the
    # old failure mode where a wrong prior speed marched ahead by itself.
    return torch.zeros(1, 4, device=device, dtype=torch.float32)


def reset_route_state_torch(state, forward_limit):
    carried_speed = torch.clamp(
        state[:, 1], min=0.0, max=float(forward_limit)
    )
    # Turning should not create acceleration. Carry current speed and reset the
    # coordinate origin/cross-track state to the new known leg.
    zero = torch.zeros_like(carried_speed)
    return torch.stack([zero, carried_speed, zero, zero], dim=1)


def filterpy_transition(dt_seconds):
    dt = clamp_dt_seconds(dt_seconds)
    return np.asarray(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def filterpy_process_covariance(dt_seconds):
    dt = clamp_dt_seconds(dt_seconds)
    return np.diag(
        np.asarray(
            [
                config.KALMAN_Q_PROGRESS,
                config.KALMAN_Q_FORWARD_SPEED,
                config.KALMAN_Q_CROSS_TRACK,
                config.KALMAN_Q_CROSS_SPEED,
            ],
            dtype=np.float64,
        )
        * dt
    )


def make_filterpy_filter():
    if KalmanFilter is None:
        raise ImportError("FilterPy is required: pip install filterpy")
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.zeros(4, dtype=np.float64)
    kf.H = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    kf.P = np.diag(
        [
            config.KALMAN_INIT_PROGRESS_VAR,
            config.KALMAN_INIT_FORWARD_SPEED_VAR,
            config.KALMAN_INIT_CROSS_TRACK_VAR,
            config.KALMAN_INIT_CROSS_SPEED_VAR,
        ]
    ).astype(np.float64)
    return kf


def reset_filterpy_for_new_leg(kf, motion_envelope):
    speed = min(
        max(float(kf.x[1]), 0.0),
        float(motion_envelope.forward_speed_limit_mps),
    )
    kf.x = np.asarray([0.0, speed, 0.0, 0.0], dtype=np.float64)
    kf.P = np.diag(
        [
            config.KALMAN_INIT_PROGRESS_VAR,
            config.KALMAN_INIT_FORWARD_SPEED_VAR,
            config.KALMAN_INIT_CROSS_TRACK_VAR,
            config.KALMAN_INIT_CROSS_SPEED_VAR,
        ]
    ).astype(np.float64)


def constrain_route_state_numpy(
    state, previous_progress, previous_cross, leg, motion_envelope, dt_seconds
):
    state = np.asarray(state, dtype=np.float64).reshape(4)
    dt = clamp_dt_seconds(dt_seconds)
    progress_upper = min(
        leg.length,
        float(previous_progress)
        + float(motion_envelope.forward_speed_limit_mps) * dt,
    )
    progress = min(
        max(float(state[0]), float(previous_progress)), progress_upper
    )
    forward_speed = min(
        max(float(state[1]), 0.0),
        float(motion_envelope.forward_speed_limit_mps),
    )
    cross_delta_limit = float(motion_envelope.cross_speed_limit_mps) * dt
    cross_delta = min(
        max(float(state[2]) - float(previous_cross), -cross_delta_limit),
        cross_delta_limit,
    )
    cross = min(
        max(
            float(previous_cross) + cross_delta,
            -float(config.ROUTE_CORRIDOR_HALF_WIDTH_M),
        ),
        float(config.ROUTE_CORRIDOR_HALF_WIDTH_M),
    )
    cross_speed = min(
        max(
            float(state[3]), -float(motion_envelope.cross_speed_limit_mps)
        ),
        float(motion_envelope.cross_speed_limit_mps),
    )
    return np.asarray(
        [progress, forward_speed, cross, cross_speed], dtype=np.float64
    )


def metric_block(prediction, gt):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    error = np.linalg.norm(prediction - gt, axis=1)
    if len(prediction) > 1:
        predicted_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)
        rpe = np.linalg.norm(predicted_step - gt_step, axis=1)
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = float(np.percentile(gt_step_length, 99)) + float(
            config.JUMP_TOLERANCE_M
        )
        predicted_step_length = np.linalg.norm(predicted_step, axis=1)
        jump_rate = float(
            (predicted_step_length > jump_threshold).mean() * 100.0
        )
    else:
        rpe = np.zeros(1)
        jump_threshold = 0.0
        jump_rate = 0.0
    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "ATE_RMSE_m": float(np.sqrt(np.mean(error ** 2))),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": jump_rate,
        "JumpThreshold_m": jump_threshold,
        "LSR@5_pct": float((error <= 5.0).mean() * 100.0),
        "LSR@10_pct": float((error <= 10.0).mean() * 100.0),
        "LSR@15_pct": float((error <= 15.0).mean() * 100.0),
        "LSR@20_pct": float((error <= 20.0).mean() * 100.0),
        "MaxLE_m": float(error.max()),
    }


def training_dt(cache, index, previous_timestamp_ns, leg):
    current_ns = int(cache.timestamp_ns[index].item())
    if previous_timestamp_ns is None:
        dt = leg.duration_seconds / max(leg.end_frame - leg.start_frame, 1)
    else:
        dt = (current_ns - int(previous_timestamp_ns)) / 1e9
    return clamp_dt_seconds(dt), current_ns


def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    manifest,
    train_range,
    envelope,
    device,
):
    model.train()
    indices = route_frame_indices(cache, train_range[0], train_range[1])
    if not indices:
        raise RuntimeError("Empty Route-A temporal train split")

    hidden = model.initial_hidden(1, device, torch.float32)
    state = initial_route_state_torch(device)
    covariance = initial_covariance_torch(1, device, torch.float32)
    first_frame = int(cache.frame_ids[indices[0]].item())
    previous_leg_index = active_leg_for_frame(manifest, first_frame).index
    previous_timestamp_ns = None

    optimizer.zero_grad(set_to_none=True)
    accumulated = None
    chunk_steps = 0
    losses = []
    capture_rows = []

    for row_number, index in enumerate(indices):
        frame_id = int(cache.frame_ids[index].item())
        leg = active_leg_for_frame(manifest, frame_id)
        leg_changed = leg.index != previous_leg_index
        if leg_changed:
            state = reset_route_state_torch(
                state, envelope.forward_speed_limit_mps
            )
            covariance = initial_covariance_torch(1, device, torch.float32)
            previous_timestamp_ns = None

        dt_seconds, current_timestamp_ns = training_dt(
            cache, index, previous_timestamp_ns, leg
        )
        dt_tensor = torch.tensor([dt_seconds], device=device, dtype=torch.float32)
        previous_progress = state[:, 0]
        previous_cross = state[:, 2]

        predicted_state, predicted_covariance = kalman_predict_torch(
            state, covariance, dt_tensor
        )
        predicted_state = constrain_route_state_torch(
            predicted_state,
            previous_progress,
            previous_cross,
            torch.tensor([leg.length], device=device, dtype=torch.float32),
            torch.tensor(
                [envelope.forward_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [envelope.cross_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            dt_tensor,
        )

        candidate_indices = candidate_indices_forward(
            visual,
            float(predicted_state[0, 0].detach().cpu()),
            float(previous_progress[0].detach().cpu()),
            leg,
            envelope,
        )
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        candidate = candidate_batch_from_indices(
            visual, uav_clip, candidate_indices
        )
        candidate_sd = candidate_centers_to_route(candidate.centers, leg)
        hardms_sd = xy_batch_to_route(candidate.hardms_xy, leg)

        measurement = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_prob,
            candidate_sd,
            hardms_sd,
            candidate.hardms_support,
            predicted_state,
            previous_progress,
            torch.tensor([leg.length], device=device, dtype=torch.float32),
            torch.tensor(
                [envelope.nominal_forward_speed_mps],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [envelope.forward_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [envelope.cross_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            dt_tensor,
            torch.tensor([leg_changed], device=device),
            hidden,
        )

        updated_state, updated_covariance, _ = kalman_update_torch(
            predicted_state,
            predicted_covariance,
            measurement.measurement_sd,
            measurement.measurement_variance,
        )
        updated_state = constrain_route_state_torch(
            updated_state,
            previous_progress,
            previous_cross,
            torch.tensor([leg.length], device=device, dtype=torch.float32),
            torch.tensor(
                [envelope.forward_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [envelope.cross_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            dt_tensor,
        )

        gt_xy = cache.gt_xy[index].to(device).float()
        gt_sd = xy_batch_to_route(gt_xy.reshape(1, 2), leg)[0]
        prediction_xy = route_to_xy_torch(
            predicted_state[:, 0], predicted_state[:, 2], leg, device, torch.float32
        )
        final_xy = route_to_xy_torch(
            updated_state[:, 0], updated_state[:, 2], leg, device, torch.float32
        )

        final_loss = F.smooth_l1_loss(final_xy, gt_xy.reshape(1, 2))
        measurement_nll = F.gaussian_nll_loss(
            measurement.measurement_sd,
            gt_sd.reshape(1, 2),
            measurement.measurement_variance,
            full=False,
            reduction="mean",
        )
        prediction_loss = F.smooth_l1_loss(
            prediction_xy, gt_xy.reshape(1, 2)
        )
        # Smooth physical velocity target from the current training waypoint leg,
        # not the quantized one-frame GPS difference.
        target_v = min(
            leg.length / leg.duration_seconds,
            envelope.forward_speed_limit_mps,
        )
        velocity_target = torch.tensor(
            [[target_v, 0.0]], device=device, dtype=torch.float32
        )
        velocity_loss = F.smooth_l1_loss(
            torch.stack([updated_state[:, 1], updated_state[:, 3]], dim=1),
            velocity_target,
        )

        loss = (
            float(config.LOSS_FINAL_SMOOTH_L1) * final_loss
            + float(config.LOSS_MEASUREMENT_GAUSSIAN_NLL) * measurement_nll
            + float(config.LOSS_PREDICTION_SMOOTH_L1) * prediction_loss
            + float(config.LOSS_VELOCITY_SMOOTH_L1) * velocity_loss
        )
        accumulated = loss if accumulated is None else accumulated + loss
        chunk_steps += 1
        losses.append(
            [
                float(loss.detach().cpu()),
                float(final_loss.detach().cpu()),
                float(measurement_nll.detach().cpu()),
                float(prediction_loss.detach().cpu()),
                float(velocity_loss.detach().cpu()),
            ]
        )
        distance = torch.linalg.norm(
            candidate.centers[0] - gt_xy.reshape(1, 2), dim=1
        )
        capture_rows.append(
            bool(
                distance.min().item()
                <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
            )
        )

        state = updated_state
        covariance = updated_covariance
        hidden = measurement.hidden
        previous_timestamp_ns = current_timestamp_ns
        previous_leg_index = leg.index

        end_chunk = (
            chunk_steps >= int(config.TBPTT_STEPS)
            or row_number == len(indices) - 1
        )
        if end_chunk:
            normalized = accumulated / float(chunk_steps)
            if not torch.isfinite(normalized):
                raise FloatingPointError("non-finite temporal loss")
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state = state.detach()
            covariance = covariance.detach()
            hidden = hidden.detach()
            accumulated = None
            chunk_steps = 0

    loss_array = np.asarray(losses, dtype=np.float64)
    return {
        "total": float(loss_array[:, 0].mean()),
        "final": float(loss_array[:, 1].mean()),
        "nll": float(loss_array[:, 2].mean()),
        "prediction": float(loss_array[:, 3].mean()),
        "velocity": float(loss_array[:, 4].mean()),
        "capture_pct": float(np.mean(capture_rows) * 100.0),
    }


@torch.no_grad()
def evaluate_filterpy(
    model,
    visual,
    cache,
    manifest,
    envelope,
    device,
    start_frame,
    end_frame,
    csv_path=None,
):
    model.eval()
    indices = route_frame_indices(cache, start_frame, end_frame)
    if not indices:
        raise RuntimeError("Empty evaluation range")

    kf = make_filterpy_filter()
    hidden = model.initial_hidden(1, device, torch.float32)
    first_frame = int(cache.frame_ids[indices[0]].item())
    first_timestamp_ns = int(cache.timestamp_ns[indices[0]].item())
    previous_leg_index = active_leg_for_frame(manifest, first_frame).index
    previous_timestamp_ns = None
    rows = []

    for index in indices:
        frame_id = int(cache.frame_ids[index].item())
        timestamp_ns = int(cache.timestamp_ns[index].item())
        leg = active_leg_for_frame(manifest, frame_id)
        leg_changed = leg.index != previous_leg_index
        if leg_changed:
            reset_filterpy_for_new_leg(kf, envelope)
            previous_timestamp_ns = None

        if previous_timestamp_ns is None:
            dt_seconds = leg.duration_seconds / max(
                leg.end_frame - leg.start_frame, 1
            )
        else:
            dt_seconds = (timestamp_ns - previous_timestamp_ns) / 1e9
        dt_seconds = clamp_dt_seconds(dt_seconds)

        previous_progress = float(kf.x[0])
        previous_cross = float(kf.x[2])
        kf.F = filterpy_transition(dt_seconds)
        kf.Q = filterpy_process_covariance(dt_seconds)
        kf.predict()
        kf.x = constrain_route_state_numpy(
            kf.x,
            previous_progress,
            previous_cross,
            leg,
            envelope,
            dt_seconds,
        )
        predicted_state_np = np.asarray(kf.x, dtype=np.float64).reshape(4)

        candidate_indices = candidate_indices_forward(
            visual,
            predicted_state_np[0],
            previous_progress,
            leg,
            envelope,
        )
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        candidate = candidate_batch_from_indices(
            visual, uav_clip, candidate_indices
        )
        candidate_sd = candidate_centers_to_route(candidate.centers, leg)
        hardms_sd = xy_batch_to_route(candidate.hardms_xy, leg)
        predicted_state = torch.tensor(
            predicted_state_np, device=device, dtype=torch.float32
        ).reshape(1, 4)
        dt_tensor = torch.tensor([dt_seconds], device=device, dtype=torch.float32)

        measurement = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_prob,
            candidate_sd,
            hardms_sd,
            candidate.hardms_support,
            predicted_state,
            torch.tensor([previous_progress], device=device, dtype=torch.float32),
            torch.tensor([leg.length], device=device, dtype=torch.float32),
            torch.tensor(
                [envelope.nominal_forward_speed_mps],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [envelope.forward_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [envelope.cross_speed_limit_mps],
                device=device,
                dtype=torch.float32,
            ),
            dt_tensor,
            torch.tensor([leg_changed], device=device),
            hidden,
        )
        hidden = measurement.hidden
        measurement_sd = measurement.measurement_sd[0].cpu().numpy().astype(
            np.float64
        )
        measurement_variance = (
            measurement.measurement_variance[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        kf.R = np.diag(measurement_variance)
        kf.update(measurement_sd)
        kf.x = constrain_route_state_numpy(
            kf.x,
            previous_progress,
            previous_cross,
            leg,
            envelope,
            dt_seconds,
        )
        final_state = np.asarray(kf.x, dtype=np.float64).reshape(4)

        prediction_xy = route_to_xy_numpy(
            predicted_state_np[0], predicted_state_np[2], leg
        )
        final_xy = route_to_xy_numpy(final_state[0], final_state[2], leg)
        gt_xy = cache.gt_xy[index].numpy()
        gt_sd = xy_to_route(cache.gt_xy[index], leg).numpy()
        raw_top1 = candidate.raw_top1_xy[0].cpu().numpy()
        hardms = candidate.hardms_xy[0].cpu().numpy()
        candidate_distance = np.linalg.norm(
            candidate.centers[0].cpu().numpy() - gt_xy[None, :], axis=1
        )
        error = float(np.linalg.norm(final_xy - gt_xy))

        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": timestamp_ns,
                "elapsed_time_s": (timestamp_ns - first_timestamp_ns) / 1e9,
                "dt_seconds": dt_seconds,
                "waypoint_from": leg.start_waypoint.order,
                "waypoint_to": leg.end_waypoint.order,
                "waypoint_from_frame": leg.start_waypoint.frame_index,
                "waypoint_to_frame": leg.end_waypoint.frame_index,
                "leg_index": leg.index,
                "leg_changed": int(leg_changed),
                "gt_x": float(gt_xy[0]),
                "gt_y": float(gt_xy[1]),
                "gt_progress_s": float(gt_sd[0]),
                "gt_cross_d": float(gt_sd[1]),
                "prediction_x": float(prediction_xy[0]),
                "prediction_y": float(prediction_xy[1]),
                "prediction_s": float(predicted_state_np[0]),
                "prediction_v_mps": float(predicted_state_np[1]),
                "prediction_d": float(predicted_state_np[2]),
                "prediction_vd_mps": float(predicted_state_np[3]),
                "raw_top1_x": float(raw_top1[0]),
                "raw_top1_y": float(raw_top1[1]),
                "hardms_x": float(hardms[0]),
                "hardms_y": float(hardms[1]),
                "measurement_s": float(measurement_sd[0]),
                "measurement_d": float(measurement_sd[1]),
                "measurement_var_s": float(measurement_variance[0]),
                "measurement_var_d": float(measurement_variance[1]),
                "final_x": float(final_xy[0]),
                "final_y": float(final_xy[1]),
                "final_s": float(final_state[0]),
                "final_v_mps": float(final_state[1]),
                "final_d": float(final_state[2]),
                "final_vd_mps": float(final_state[3]),
                "error_m": error,
                "candidate_capture": int(
                    candidate_distance.min()
                    <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
                ),
            }
        )
        previous_timestamp_ns = timestamp_ns
        previous_leg_index = leg.index

    gt = [[row["gt_x"], row["gt_y"]] for row in rows]
    summary = {
        "MotionPrediction": metric_block(
            [[row["prediction_x"], row["prediction_y"]] for row in rows], gt
        ),
        "RawTop1": metric_block(
            [[row["raw_top1_x"], row["raw_top1_y"]] for row in rows], gt
        ),
        "FixedHardMS": metric_block(
            [[row["hardms_x"], row["hardms_y"]] for row in rows], gt
        ),
        "RouteCoordinateKalmanFinal": metric_block(
            [[row["final_x"], row["final_y"]] for row in rows], gt
        ),
        "CandidateCaptureRate_pct": float(
            np.mean([row["candidate_capture"] for row in rows]) * 100.0
        ),
        "MeanDtSeconds": float(np.mean([row["dt_seconds"] for row in rows])),
        "MeanMeasurementVarS": float(
            np.mean([row["measurement_var_s"] for row in rows])
        ),
        "MeanMeasurementVarD": float(
            np.mean([row["measurement_var_d"] for row in rows])
        ),
        "BackwardProgressCount": int(
            sum(
                1
                for i in range(1, len(rows))
                if rows[i]["leg_index"] == rows[i - 1]["leg_index"]
                and rows[i]["final_s"] < rows[i - 1]["final_s"] - 1e-8
            )
        ),
    }

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return summary, rows


def validation_score(summary):
    metric = summary["RouteCoordinateKalmanFinal"]
    return (
        metric["MLE_m"]
        + float(config.VAL_RPE_WEIGHT) * metric["RPE_m"]
        + float(config.VAL_JUMP_WEIGHT) * metric["JumpRate_pct"]
    )


def train_temporal(model, visual, cache, manifest, device, epochs):
    split = split_route_a_legs(manifest)
    envelope = derive_motion_envelope(cache, manifest, split)
    print("Route A temporal split:", split, flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LR),
        weight_decay=float(config.WEIGHT_DECAY),
    )
    best_score = float("inf")
    best_state = None
    patience = 0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(epochs)):
        training = train_one_epoch(
            model,
            optimizer,
            visual,
            cache,
            manifest,
            split["train"],
            envelope,
            device,
        )
        validation, _ = evaluate_filterpy(
            model,
            visual,
            cache,
            manifest,
            envelope,
            device,
            split["val"][0],
            split["val"][1],
        )
        score = validation_score(validation)
        if score < best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        torch.save(
            {
                "architecture": ARCHITECTURE_NAME,
                "model": model.state_dict(),
                "best_model": best_state,
                "epoch": epoch + 1,
                "best_score": best_score,
                "motion_envelope": envelope.as_dict(),
                "waypoint_count": len(manifest.waypoints),
                "leg_count": len(manifest.legs),
                "temporal_train_routes": ["route_A"],
                "temporal_validation_routes": ["route_A"],
                "temporal_eval_routes": ["route_B", "route_C"],
                "state": [
                    "route_progress_s_m",
                    "forward_velocity_mps",
                    "cross_track_d_m",
                    "cross_velocity_mps",
                ],
                "timestamp_aware": True,
                "all_waypoints_used": True,
                "filterpy_inference": True,
                "monotonic_progress": True,
            },
            config.TEMPORAL_CHECKPOINT,
        )

        metric = validation["RouteCoordinateKalmanFinal"]
        print(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"loss={training['total']:.5f} "
            f"final={training['final']:.4f} "
            f"nll={training['nll']:.4f} "
            f"pred={training['prediction']:.4f} "
            f"vel={training['velocity']:.4f} "
            f"capture={training['capture_pct']:.2f}% "
            f"val_mle={metric['MLE_m']:.3f}m "
            f"val_rpe={metric['RPE_m']:.3f}m "
            f"val_jump={metric['JumpRate_pct']:.3f}% "
            f"backward={validation['BackwardProgressCount']} "
            f"mean_Rs={validation['MeanMeasurementVarS']:.2f}",
            flush=True,
        )
        if patience >= int(config.EARLY_STOPPING_PATIENCE):
            print("temporal early stopping", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Temporal training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    checkpoint["model"] = best_state
    checkpoint["best_model"] = best_state
    torch.save(checkpoint, config.TEMPORAL_CHECKPOINT)
    return split, envelope


def load_temporal_checkpoint(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.TEMPORAL_CHECKPOINT)
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError("Temporal checkpoint architecture mismatch")
    model.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"], strict=True
    )
    payload = checkpoint["motion_envelope"]
    envelope = MotionEnvelope(
        nominal_forward_speed_mps=float(payload["nominal_forward_speed_mps"]),
        forward_speed_limit_mps=float(payload["forward_speed_limit_mps"]),
        cross_speed_limit_mps=float(payload["cross_speed_limit_mps"]),
        training_leg_speeds_mps=[
            float(value) for value in payload.get("training_leg_speeds_mps", [])
        ],
    )
    return checkpoint, envelope


def route_catalog():
    return {
        name: Path(root)
        for name, root in zip(config.ROUTE_NAMES, config.ROUTE_ROOTS)
    }


def main():
    print("[PYTHON] v3 tracker main entered", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("train", "eval", "train_eval"), default="train_eval"
    )
    parser.add_argument("--visual-epochs", type=int, default=config.VISUAL_EPOCHS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--jitter-m", type=float, default=config.LOCAL_PRIOR_JITTER_M)
    parser.add_argument("--reuse-visual", action="store_true")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print("=" * 88, flush=True)
    print("TIMESTAMP-AWARE ROUTE-COORDINATE GRU + FILTERPY KALMAN", flush=True)
    print("All JSON waypoints are used; straight_legs is not trusted.", flush=True)
    print("Time unit=seconds; velocity unit=m/s.", flush=True)
    print("=" * 88, flush=True)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "train_eval"):
        if config.TEMPORAL_CHECKPOINT.exists():
            config.TEMPORAL_CHECKPOINT.unlink()
        if args.reuse_visual:
            if not config.VISUAL_CHECKPOINT.exists():
                raise FileNotFoundError(
                    f"--reuse-visual requested but missing {config.VISUAL_CHECKPOINT}"
                )
            print("[STAGE 1/4] reuse Route-A visual checkpoint", flush=True)
        else:
            print("[STAGE 1/4] train visual retrieval from scratch", flush=True)
            if config.VISUAL_CHECKPOINT.exists():
                config.VISUAL_CHECKPOINT.unlink()
            train_visual_retrieval_a_only(
                device=device,
                epochs=int(args.visual_epochs),
                jitter_m=float(args.jitter_m),
                resume=False,
            )

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)

    print("[STAGE 2/4] loading frozen visual localizer/gallery", flush=True)
    visual = FrozenVisualLocalizer(device)
    model = RouteCoordinateGRU().to(device)
    catalog = route_catalog()
    manifests = {
        route_name: load_waypoint_manifest(
            route_name, visual.origin_lat, visual.origin_lon
        )
        for route_name in ("route_A", "route_B", "route_C")
    }

    if args.mode in ("train", "train_eval"):
        print("[STAGE 3/4] Route-A temporal training", flush=True)
        route_a_cache = build_backbone_cache(
            "route_A",
            catalog["route_A"],
            visual,
            manifests["route_A"],
            device,
        )
        split, envelope = train_temporal(
            model,
            visual,
            route_a_cache,
            manifests["route_A"],
            device,
            int(args.epochs),
        )
    else:
        _, envelope = load_temporal_checkpoint(model, device)
        split = split_route_a_legs(manifests["route_A"])

    if args.mode in ("eval", "train_eval"):
        if args.mode == "eval":
            _, envelope = load_temporal_checkpoint(model, device)
        print("[STAGE 4/4] Route-B / Route-C inference", flush=True)
        route_results = {}
        for route_name in ("route_B", "route_C"):
            manifest = manifests[route_name]
            cache = build_backbone_cache(
                route_name, catalog[route_name], visual, manifest, device
            )
            csv_path = (
                config.OUTPUT_DIR / f"{route_name}_route_coordinate_frames.csv"
            )
            summary, _ = evaluate_filterpy(
                model,
                visual,
                cache,
                manifest,
                envelope,
                device,
                manifest.legs[0].start_frame,
                manifest.legs[-1].end_frame,
                csv_path=csv_path,
            )
            route_results[route_name] = summary
            metric = summary["RouteCoordinateKalmanFinal"]
            print(
                f"{route_name}: MLE={metric['MLE_m']:.3f}m "
                f"P90={metric['P90_m']:.3f}m "
                f"RPE={metric['RPE_m']:.3f}m "
                f"Jump={metric['JumpRate_pct']:.3f}% "
                f"Capture={summary['CandidateCaptureRate_pct']:.2f}% "
                f"Backward={summary['BackwardProgressCount']} "
                f"MeanRs={summary['MeanMeasurementVarS']:.2f}",
                flush=True,
            )

        payload = {
            "architecture": ARCHITECTURE_NAME,
            "protocol": {
                "single_frame_streaming": True,
                "timestamp_aware": True,
                "time_unit": "seconds",
                "velocity_unit": "m/s",
                "all_waypoints_used": True,
                "straight_legs_ignored_and_rebuilt": True,
                "visual_train": ["route_A"],
                "temporal_train": ["route_A"],
                "eval": ["route_B", "route_C"],
                "filter": "filterpy.kalman.KalmanFilter",
            },
            "waypoint_counts": {
                name: len(manifest.waypoints)
                for name, manifest in manifests.items()
            },
            "leg_counts": {
                name: len(manifest.legs)
                for name, manifest in manifests.items()
            },
            "route_A_temporal_split": split,
            "motion_envelope": envelope.as_dict(),
            "routes": route_results,
        }
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = config.OUTPUT_DIR / "robust_tracker_summary.json"
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("summary saved:", summary_path, flush=True)


if __name__ == "__main__":
    main()
