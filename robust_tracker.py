import argparse
import csv
import json
import math
import random
import shutil
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


ARCHITECTURE_NAME = "RouteCoordinateGRU_ConstrainedFilterPyKalman"


@dataclass
class BackboneCache:
    route_name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    uav_clip: torch.Tensor

    def __len__(self):
        return int(self.gt_xy.shape[0])


@dataclass
class RouteLeg:
    index: int
    start_frame: int
    end_frame: int
    start_xy: torch.Tensor
    end_xy: torch.Tensor

    @property
    def vector(self):
        return self.end_xy - self.start_xy

    @property
    def length(self):
        return float(
            torch.linalg.norm(self.vector).item()
        )

    @property
    def unit(self):
        return self.vector / max(
            self.length,
            1e-8,
        )

    @property
    def normal(self):
        unit = self.unit
        return torch.tensor(
            [-float(unit[1]), float(unit[0])],
            dtype=torch.float32,
        )


@dataclass
class WaypointManifest:
    route_name: str
    legs: list


@dataclass
class MotionEnvelope:
    forward_speed_limit: float
    cross_speed_limit: float
    percentile: float

    def as_dict(self):
        return {
            "forward_speed_limit_m_per_frame": float(
                self.forward_speed_limit
            ),
            "cross_speed_limit_m_per_frame": float(
                self.cross_speed_limit
            ),
            "percentile": float(
                self.percentile
            ),
            "source": (
                "Route-A temporal training legs only"
            ),
        }


# =============================================================================
# General utilities
# =============================================================================

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cache_dtype():
    if config.FEATURE_CACHE_DTYPE == "float16":
        return torch.float16
    return torch.float32


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def load_waypoint_manifest(
    route_name,
    origin_lat,
    origin_lon,
):
    path = Path(
        config.WAYPOINT_FILES[route_name]
    )

    if not path.exists():
        raise FileNotFoundError(
            "waypoint file not found: "
            f"{path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    waypoints = {
        int(item["waypoint_order"]): item
        for item in payload["waypoints"]
    }

    legs = []

    for row in payload["straight_legs"]:
        start_item = waypoints[
            int(row["start_waypoint_order"])
        ]
        end_item = waypoints[
            int(row["end_waypoint_order"])
        ]

        start_xy = torch.tensor(
            meters_from_latlon(
                start_item["latitude"],
                start_item["longitude"],
                origin_lat,
                origin_lon,
            ),
            dtype=torch.float32,
        )

        end_xy = torch.tensor(
            meters_from_latlon(
                end_item["latitude"],
                end_item["longitude"],
                origin_lat,
                origin_lon,
            ),
            dtype=torch.float32,
        )

        legs.append(
            RouteLeg(
                index=int(row["leg"]) - 1,
                start_frame=int(row["start_frame_index"]),
                end_frame=int(row["end_frame_index"]),
                start_xy=start_xy,
                end_xy=end_xy,
            )
        )

    return WaypointManifest(
        route_name=route_name,
        legs=legs,
    )


def active_leg_for_frame(
    manifest,
    frame_id,
):
    frame_id = int(frame_id)

    for leg_index, leg in enumerate(
        manifest.legs
    ):
        is_last = (
            leg_index
            == len(manifest.legs) - 1
        )

        if (
            leg.start_frame
            <= frame_id
            < leg.end_frame
            or (
                is_last
                and frame_id
                <= leg.end_frame
            )
        ):
            return leg

    if frame_id < manifest.legs[0].start_frame:
        return manifest.legs[0]

    return manifest.legs[-1]


def xy_to_route(
    xy,
    leg,
):
    xy = torch.as_tensor(
        xy,
        dtype=torch.float32,
    )

    relative = (
        xy - leg.start_xy
    )

    progress = torch.dot(
        relative,
        leg.unit,
    )

    cross_track = torch.dot(
        relative,
        leg.normal,
    )

    return torch.stack(
        [progress, cross_track]
    )


def route_to_xy_torch(
    progress,
    cross_track,
    leg,
    device,
    dtype,
):
    start = leg.start_xy.to(
        device=device,
        dtype=dtype,
    ).reshape(1, 2)

    unit = leg.unit.to(
        device=device,
        dtype=dtype,
    ).reshape(1, 2)

    normal = leg.normal.to(
        device=device,
        dtype=dtype,
    ).reshape(1, 2)

    return (
        start
        + progress.reshape(-1, 1)
        * unit
        + cross_track.reshape(-1, 1)
        * normal
    )


def route_to_xy_numpy(
    progress,
    cross_track,
    leg,
):
    return (
        leg.start_xy.numpy().astype(
            np.float64
        )
        + float(progress)
        * leg.unit.numpy().astype(
            np.float64
        )
        + float(cross_track)
        * leg.normal.numpy().astype(
            np.float64
        )
    )


def split_route_a_legs(
    manifest,
):
    count = len(manifest.legs)

    train_count = max(
        1,
        int(
            count
            * float(
                config.TEMPORAL_TRAIN_LEG_FRACTION
            )
        ),
    )

    val_count = max(
        1,
        int(
            count
            * float(
                config.TEMPORAL_VAL_LEG_FRACTION
            )
        ),
    )

    if train_count + val_count >= count:
        val_count = max(
            1,
            count - train_count - 1,
        )

    train_legs = manifest.legs[
        :train_count
    ]
    val_legs = manifest.legs[
        train_count:train_count + val_count
    ]
    test_legs = manifest.legs[
        train_count + val_count:
    ]

    return {
        "train": (
            train_legs[0].start_frame,
            train_legs[-1].end_frame,
        ),
        "val": (
            val_legs[0].start_frame,
            val_legs[-1].end_frame,
        ),
        "test": (
            (
                test_legs[0].start_frame,
                test_legs[-1].end_frame,
            )
            if test_legs
            else (
                val_legs[-1].end_frame,
                val_legs[-1].end_frame,
            )
        ),
    }


def route_frame_indices(
    cache,
    start_frame,
    end_frame,
):
    values = cache.frame_ids.numpy()

    mask = (
        (values >= int(start_frame))
        & (values <= int(end_frame))
    )

    return np.nonzero(mask)[0].tolist()


# =============================================================================
# Single-frame backbone cache
# =============================================================================

@torch.no_grad()
def build_backbone_cache(
    route_name,
    root,
    visual,
    device,
):
    dataset = RouteDataset(
        Path(root),
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )

    frame_rows = []
    gt_rows = []
    clip_rows = []

    batch_size = int(
        config.VISUAL_CACHE_BATCH_SIZE
    )

    for start in range(
        0,
        len(dataset),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(dataset),
        )

        items = [
            dataset[index]
            for index in range(
                start,
                end,
            )
        ]

        # Batching is only a compute optimization.
        # Every temporal step still consumes exactly ONE cached frame descriptor.
        uav = torch.stack(
            [
                item["uav"]
                for item in items
            ]
        ).to(device)

        clip = visual.encode_uav_clip(
            uav
        )

        clip_rows.append(
            clip.detach().cpu().to(
                cache_dtype()
            )
        )

        gt_rows.append(
            torch.stack(
                [
                    item["xy"].float()
                    for item in items
                ]
            )
        )

        frame_rows.extend(
            parse_frame_id(
                item["frame_id"]
            )
            for item in items
        )

        if (
            start == 0
            or end == len(dataset)
            or (
                start // batch_size
            ) % 20 == 0
        ):
            print(
                f"{route_name} backbone cache: "
                f"{end}/{len(dataset)}",
                flush=True,
            )

    return BackboneCache(
        route_name=route_name,
        frame_ids=torch.tensor(
            frame_rows,
            dtype=torch.long,
        ),
        gt_xy=torch.cat(
            gt_rows
        ).float(),
        uav_clip=torch.cat(
            clip_rows
        ),
    )


# =============================================================================
# Derive physical speed envelope ONLY from Route-A temporal training split
# =============================================================================

def derive_motion_envelope(
    cache,
    manifest,
    train_range,
):
    indices = route_frame_indices(
        cache,
        train_range[0],
        train_range[1],
    )

    forward_speeds = []
    cross_speeds = []

    previous_frame = None
    previous_leg_index = None
    previous_sd = None

    for index in indices:
        frame_id = int(
            cache.frame_ids[index].item()
        )
        leg = active_leg_for_frame(
            manifest,
            frame_id,
        )

        current_sd = xy_to_route(
            cache.gt_xy[index],
            leg,
        )

        if (
            previous_frame is not None
            and previous_leg_index
            == leg.index
        ):
            dt = max(
                frame_id - previous_frame,
                1,
            )

            delta = (
                current_sd
                - previous_sd
            )

            # Forward mission speed is nonnegative by definition.
            forward_speeds.append(
                max(
                    0.0,
                    float(delta[0]) / dt,
                )
            )

            cross_speeds.append(
                abs(
                    float(delta[1]) / dt
                )
            )

        previous_frame = frame_id
        previous_leg_index = leg.index
        previous_sd = current_sd

    if not forward_speeds:
        raise RuntimeError(
            "Cannot derive Route-A motion envelope"
        )

    percentile = float(
        config.MOTION_ENVELOPE_PERCENTILE
    )

    forward_limit = max(
        float(
            np.percentile(
                np.asarray(
                    forward_speeds,
                    dtype=np.float64,
                ),
                percentile,
            )
        ),
        float(
            config.MIN_FORWARD_SPEED_LIMIT_M_PER_FRAME
        ),
    )

    cross_limit = max(
        float(
            np.percentile(
                np.asarray(
                    cross_speeds,
                    dtype=np.float64,
                ),
                percentile,
            )
        ),
        float(
            config.MIN_CROSS_SPEED_LIMIT_M_PER_FRAME
        ),
    )

    envelope = MotionEnvelope(
        forward_speed_limit=forward_limit,
        cross_speed_limit=cross_limit,
        percentile=percentile,
    )

    print(
        "Route-A learned motion envelope: "
        f"forward <= {forward_limit:.3f} m/frame, "
        f"|cross| <= {cross_limit:.3f} m/frame "
        f"(p{percentile})",
        flush=True,
    )

    return envelope


# =============================================================================
# Route-coordinate forward candidate retrieval
# =============================================================================

def gallery_route_coordinates(
    visual,
    leg,
):
    gallery_xy = visual.gallery["xy"]

    start = leg.start_xy.to(
        gallery_xy.device
    )
    unit = leg.unit.to(
        gallery_xy.device
    )
    normal = leg.normal.to(
        gallery_xy.device
    )

    relative = (
        gallery_xy
        - start[None, :]
    )

    progress = (
        relative
        * unit[None, :]
    ).sum(dim=1)

    cross_track = (
        relative
        * normal[None, :]
    ).sum(dim=1)

    return (
        progress,
        cross_track,
    )


def candidate_indices_forward(
    visual,
    predicted_progress,
    previous_progress,
    leg,
    motion_envelope,
):
    progress, cross_track = (
        gallery_route_coordinates(
            visual,
            leg,
        )
    )

    count = int(
        config.ROUTE_CANDIDATE_COUNT
    )

    previous_progress = min(
        max(
            0.0,
            float(previous_progress),
        ),
        float(leg.length),
    )

    predicted_progress = min(
        max(
            previous_progress,
            float(predicted_progress),
        ),
        float(leg.length),
    )

    search_upper = min(
        float(leg.length)
        + float(
            config.ROUTE_ENDPOINT_PADDING_M
        ),
        previous_progress
        + float(
            motion_envelope.forward_speed_limit
        )
        * float(
            config.SEARCH_LOOKAHEAD_FRAMES
        ),
    )

    chosen_mask = None

    for width_scale in (
        1.0,
        1.5,
        2.0,
        4.0,
    ):
        mask = (
            (progress >= previous_progress)
            & (progress <= search_upper)
            & (
                cross_track.abs()
                <= float(
                    config.ROUTE_CORRIDOR_HALF_WIDTH_M
                )
                * width_scale
            )
        )

        if int(
            mask.sum().item()
        ) >= count:
            chosen_mask = mask
            break

    if chosen_mask is None:
        chosen_mask = (
            (progress >= previous_progress)
            & (progress <= search_upper)
        )

    valid_indices = torch.nonzero(
        chosen_mask,
        as_tuple=False,
    ).flatten()

    # At the exact endpoint, use only a terminal support band.
    if valid_indices.numel() == 0:
        terminal_start = max(
            0.0,
            float(leg.length)
            - float(
                config.ROUTE_ENDPOINT_PADDING_M
            ),
        )

        terminal_end = (
            float(leg.length)
            + float(
                config.ROUTE_ENDPOINT_PADDING_M
            )
        )

        terminal_mask = (
            (progress >= terminal_start)
            & (progress <= terminal_end)
            & (
                cross_track.abs()
                <= float(
                    config.ROUTE_CORRIDOR_HALF_WIDTH_M
                )
                * 4.0
            )
        )

        valid_indices = torch.nonzero(
            terminal_mask,
            as_tuple=False,
        ).flatten()

    if valid_indices.numel() == 0:
        raise RuntimeError(
            "No legal SAT patch in forward route support: "
            f"leg={leg.index}, "
            f"previous_progress={previous_progress:.2f}m, "
            f"search_upper={search_upper:.2f}m"
        )

    valid_progress = progress[
        valid_indices
    ]

    valid_cross = cross_track[
        valid_indices
    ]

    # Rank in ROUTE coordinates, not free XY.
    scale_s = max(
        motion_envelope.forward_speed_limit
        * 2.0,
        1.0,
    )

    scale_d = max(
        motion_envelope.cross_speed_limit
        * 4.0,
        1.0,
    )

    cost = (
        (
            (
                valid_progress
                - predicted_progress
            )
            / scale_s
        ).square()
        + (
            valid_cross
            / scale_d
        ).square()
    )

    actual_count = min(
        count,
        int(valid_indices.numel()),
    )

    order = torch.topk(
        cost,
        k=actual_count,
        largest=False,
    ).indices

    selected = valid_indices[
        order
    ]

    if selected.numel() < count:
        selected = torch.cat(
            [
                selected,
                selected[-1].repeat(
                    count
                    - selected.numel()
                ),
            ],
            dim=0,
        )

    return selected.reshape(
        1,
        -1,
    )


@torch.no_grad()
def candidate_batch_from_indices(
    visual,
    uav_clip,
    indices,
):
    device = visual.device
    indices = indices.to(device)

    centers = visual.gallery["xy"][
        indices
    ]

    satellite_clip = visual.gallery[
        "clip_feat"
    ][indices]

    z_uav = (
        visual.model.encode_uav_from_clip(
            uav_clip
        )
    )

    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(
            -1,
            satellite_clip.shape[-1],
        ),
        centers.reshape(
            -1,
            2,
        ),
    ).reshape(
        centers.shape[0],
        centers.shape[1],
        -1,
    )

    raw_logits = (
        visual.model.logit_scale.exp().clamp(
            max=100.0
        )
        * (
            z_uav[:, None]
            * z_sat
        ).sum(dim=2)
    )

    raw_prob = torch.softmax(
        raw_logits
        / float(
            config.MEANSHIFT_SCORE_TAU
        ),
        dim=1,
    )

    raw_index = raw_logits.argmax(
        dim=1
    )

    raw_top1_xy = centers[
        torch.arange(
            centers.shape[0],
            device=device,
        ),
        raw_index,
    ]

    hardms_xy, hardms_support = (
        hard_mean_shift(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
        )
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


def candidate_centers_to_route(
    centers,
    leg,
):
    start = leg.start_xy.to(
        centers.device
    ).reshape(1, 1, 2)

    unit = leg.unit.to(
        centers.device
    ).reshape(1, 1, 2)

    normal = leg.normal.to(
        centers.device
    ).reshape(1, 1, 2)

    relative = (
        centers - start
    )

    progress = (
        relative * unit
    ).sum(dim=2)

    cross = (
        relative * normal
    ).sum(dim=2)

    return torch.stack(
        [progress, cross],
        dim=2,
    )


def xy_batch_to_route(
    xy,
    leg,
):
    start = leg.start_xy.to(
        xy.device
    ).reshape(1, 2)

    unit = leg.unit.to(
        xy.device
    ).reshape(1, 2)

    normal = leg.normal.to(
        xy.device
    ).reshape(1, 2)

    relative = (
        xy - start
    )

    return torch.stack(
        [
            (relative * unit).sum(dim=1),
            (relative * normal).sum(dim=1),
        ],
        dim=1,
    )


# =============================================================================
# Route-coordinate Kalman helpers
# =============================================================================

def initial_route_state_torch(
    device,
):
    return torch.zeros(
        1,
        4,
        device=device,
        dtype=torch.float32,
    )


def reset_route_state_torch(
    old_state,
):
    # At a controller-confirmed waypoint switch:
    # progress=0 and cross-track=0 in the NEW leg.
    # Carry only the nonnegative forward speed magnitude.
    forward_speed = torch.clamp(
        old_state[:, 1],
        min=0.0,
    )

    zero = torch.zeros_like(
        forward_speed
    )

    return torch.stack(
        [
            zero,
            forward_speed,
            zero,
            zero,
        ],
        dim=1,
    )


def filterpy_transition(
    dt,
):
    dt = float(
        max(dt, 1.0)
    )

    return np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def filterpy_process_covariance(
    dt,
):
    dt = float(
        max(dt, 1.0)
    )

    return np.diag(
        np.array(
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
        raise ImportError(
            "FilterPy is required: pip install filterpy"
        )

    kf = KalmanFilter(
        dim_x=4,
        dim_z=2,
    )

    kf.x = np.zeros(
        4,
        dtype=np.float64,
    )

    # measurement z=[s,d]
    kf.H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
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


def reset_filterpy_for_new_leg(
    kf,
    motion_envelope,
):
    carried_speed = min(
        max(
            0.0,
            float(kf.x[1]),
        ),
        float(
            motion_envelope.forward_speed_limit
        ),
    )

    kf.x = np.array(
        [
            0.0,
            carried_speed,
            0.0,
            0.0,
        ],
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


def constrain_route_state_numpy(
    state,
    previous_progress,
    previous_cross,
    leg,
    motion_envelope,
    dt,
):
    state = np.asarray(
        state,
        dtype=np.float64,
    ).reshape(4)

    dt = float(
        max(dt, 1.0)
    )

    progress_upper = min(
        float(leg.length),
        float(previous_progress)
        + float(
            motion_envelope.forward_speed_limit
        )
        * dt,
    )

    progress = min(
        max(
            float(state[0]),
            float(previous_progress),
        ),
        progress_upper,
    )

    forward_speed = min(
        max(
            float(state[1]),
            0.0,
        ),
        float(
            motion_envelope.forward_speed_limit
        ),
    )

    cross_delta_limit = (
        float(
            motion_envelope.cross_speed_limit
        )
        * dt
    )

    cross_delta = min(
        max(
            float(state[2])
            - float(previous_cross),
            -cross_delta_limit,
        ),
        cross_delta_limit,
    )

    cross = min(
        max(
            float(previous_cross)
            + cross_delta,
            -float(
                config.ROUTE_CORRIDOR_HALF_WIDTH_M
            ),
        ),
        float(
            config.ROUTE_CORRIDOR_HALF_WIDTH_M
        ),
    )

    cross_speed = min(
        max(
            float(state[3]),
            -float(
                motion_envelope.cross_speed_limit
            ),
        ),
        float(
            motion_envelope.cross_speed_limit
        ),
    )

    return np.array(
        [
            progress,
            forward_speed,
            cross,
            cross_speed,
        ],
        dtype=np.float64,
    )


# =============================================================================
# Metrics
# =============================================================================

def metric_block(
    prediction,
    gt,
):
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    error = np.linalg.norm(
        prediction - gt,
        axis=1,
    )

    if len(prediction) > 1:
        predicted_step = np.diff(
            prediction,
            axis=0,
        )
        gt_step = np.diff(
            gt,
            axis=0,
        )

        rpe = np.linalg.norm(
            predicted_step
            - gt_step,
            axis=1,
        )

        gt_step_length = np.linalg.norm(
            gt_step,
            axis=1,
        )

        jump_threshold = (
            float(
                np.percentile(
                    gt_step_length,
                    99,
                )
            )
            + float(
                config.JUMP_TOLERANCE_M
            )
        )

        predicted_step_length = np.linalg.norm(
            predicted_step,
            axis=1,
        )

        jump_rate = float(
            (
                predicted_step_length
                > jump_threshold
            ).mean()
            * 100.0
        )
    else:
        rpe = np.zeros(
            1,
            dtype=np.float64,
        )
        jump_threshold = 0.0
        jump_rate = 0.0

    return {
        "MLE_m": float(
            error.mean()
        ),
        "MedLE_m": float(
            np.median(error)
        ),
        "P90_m": float(
            np.percentile(error, 90)
        ),
        "P95_m": float(
            np.percentile(error, 95)
        ),
        "ATE_RMSE_m": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "RPE_m": float(
            rpe.mean()
        ),
        "JumpRate_pct": float(
            jump_rate
        ),
        "JumpThreshold_m": float(
            jump_threshold
        ),
        "LSR@5_pct": float(
            (error <= 5.0).mean()
            * 100.0
        ),
        "LSR@10_pct": float(
            (error <= 10.0).mean()
            * 100.0
        ),
        "LSR@15_pct": float(
            (error <= 15.0).mean()
            * 100.0
        ),
        "LSR@20_pct": float(
            (error <= 20.0).mean()
            * 100.0
        ),
        "MaxLE_m": float(
            error.max()
        ),
    }


# =============================================================================
# Temporal training
# =============================================================================

def gt_velocity_target(
    current_sd,
    previous_sd,
    dt,
):
    if previous_sd is None:
        return torch.zeros(
            2,
            device=current_sd.device,
            dtype=current_sd.dtype,
        )

    dt = float(
        max(dt, 1.0)
    )

    velocity = (
        current_sd
        - previous_sd
    ) / dt

    # Route state is explicitly forward-only.
    return torch.stack(
        [
            torch.clamp(
                velocity[0],
                min=0.0,
            ),
            velocity[1],
        ]
    )


def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    manifest,
    train_range,
    motion_envelope,
    device,
):
    model.train()

    indices = route_frame_indices(
        cache,
        train_range[0],
        train_range[1],
    )

    if not indices:
        raise RuntimeError(
            "Empty Route-A temporal training split"
        )

    hidden = model.initial_hidden(
        1,
        device,
        torch.float32,
    )

    state = initial_route_state_torch(
        device
    )

    covariance = initial_covariance_torch(
        1,
        device,
        torch.float32,
    )

    first_frame = int(
        cache.frame_ids[
            indices[0]
        ].item()
    )

    current_leg = active_leg_for_frame(
        manifest,
        first_frame,
    )

    previous_leg_index = (
        current_leg.index
    )
    previous_frame = first_frame - 1
    previous_gt_sd = None

    loss_accumulator = None
    chunk_steps = 0
    term_rows = []
    capture_rows = []

    optimizer.zero_grad(
        set_to_none=True
    )

    for row_number, index in enumerate(
        indices
    ):
        frame_id = int(
            cache.frame_ids[index].item()
        )

        leg = active_leg_for_frame(
            manifest,
            frame_id,
        )

        leg_changed = (
            leg.index
            != previous_leg_index
        )

        if leg_changed:
            state = reset_route_state_torch(
                state
            )
            covariance = initial_covariance_torch(
                1,
                device,
                torch.float32,
            )
            previous_gt_sd = None

        dt = max(
            frame_id
            - previous_frame,
            1,
        )

        previous_progress = state[
            :,
            0
        ]
        previous_cross = state[
            :,
            2
        ]

        predicted_state, predicted_covariance = (
            kalman_predict_torch(
                state,
                covariance,
                torch.tensor(
                    [dt],
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

        # Constrain prediction BEFORE it controls candidate search.
        predicted_state = (
            constrain_route_state_torch(
                predicted_state,
                previous_progress,
                previous_cross,
                torch.tensor(
                    [leg.length],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        motion_envelope.forward_speed_limit
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        motion_envelope.cross_speed_limit
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [dt],
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

        candidate_indices = (
            candidate_indices_forward(
                visual,
                float(
                    predicted_state[
                        0,
                        0,
                    ].detach().cpu()
                ),
                float(
                    previous_progress[
                        0
                    ].detach().cpu()
                ),
                leg,
                motion_envelope,
            )
        )

        uav_clip = cache.uav_clip[
            index:index + 1
        ].to(
            device
        ).float()

        candidate = candidate_batch_from_indices(
            visual,
            uav_clip,
            candidate_indices,
        )

        candidate_sd = (
            candidate_centers_to_route(
                candidate.centers,
                leg,
            )
        )

        hardms_sd = xy_batch_to_route(
            candidate.hardms_xy,
            leg,
        )

        measurement = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_prob,
            candidate_sd,
            hardms_sd,
            candidate.hardms_support,
            predicted_state,
            previous_progress,
            torch.tensor(
                [leg.length],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    motion_envelope.forward_speed_limit
                ],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    motion_envelope.cross_speed_limit
                ],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [leg_changed],
                device=device,
            ),
            hidden,
        )

        updated_state, updated_covariance, _ = (
            kalman_update_torch(
                predicted_state,
                predicted_covariance,
                measurement.measurement_sd,
                measurement.measurement_variance,
            )
        )

        updated_state = (
            constrain_route_state_torch(
                updated_state,
                previous_progress,
                previous_cross,
                torch.tensor(
                    [leg.length],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        motion_envelope.forward_speed_limit
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        motion_envelope.cross_speed_limit
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [dt],
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

        gt_xy = cache.gt_xy[
            index
        ].to(
            device
        ).float()

        gt_sd = xy_batch_to_route(
            gt_xy.reshape(
                1,
                2,
            ),
            leg,
        )[0]

        prediction_xy = route_to_xy_torch(
            predicted_state[:, 0],
            predicted_state[:, 2],
            leg,
            device,
            torch.float32,
        )

        final_xy = route_to_xy_torch(
            updated_state[:, 0],
            updated_state[:, 2],
            leg,
            device,
            torch.float32,
        )

        target_velocity = gt_velocity_target(
            gt_sd,
            previous_gt_sd,
            dt,
        )

        final_loss = F.smooth_l1_loss(
            final_xy,
            gt_xy.reshape(
                1,
                2,
            ),
        )

        measurement_nll = F.gaussian_nll_loss(
            measurement.measurement_sd,
            gt_sd.reshape(
                1,
                2,
            ),
            measurement.measurement_variance,
            full=False,
            reduction="mean",
        )

        prediction_loss = F.smooth_l1_loss(
            prediction_xy,
            gt_xy.reshape(
                1,
                2,
            ),
        )

        velocity_loss = F.smooth_l1_loss(
            torch.stack(
                [
                    updated_state[:, 1],
                    updated_state[:, 3],
                ],
                dim=1,
            ),
            target_velocity.reshape(
                1,
                2,
            ),
        )

        loss = (
            float(
                config.LOSS_FINAL_SMOOTH_L1
            )
            * final_loss
            + float(
                config.LOSS_MEASUREMENT_GAUSSIAN_NLL
            )
            * measurement_nll
            + float(
                config.LOSS_PREDICTION_SMOOTH_L1
            )
            * prediction_loss
            + float(
                config.LOSS_VELOCITY_SMOOTH_L1
            )
            * velocity_loss
        )

        if loss_accumulator is None:
            loss_accumulator = loss
        else:
            loss_accumulator = (
                loss_accumulator + loss
            )

        chunk_steps += 1

        term_rows.append(
            {
                "total": float(
                    loss.detach().cpu()
                ),
                "final": float(
                    final_loss.detach().cpu()
                ),
                "nll": float(
                    measurement_nll.detach().cpu()
                ),
                "prediction": float(
                    prediction_loss.detach().cpu()
                ),
                "velocity": float(
                    velocity_loss.detach().cpu()
                ),
            }
        )

        # Capture diagnostic only.
        distance = torch.linalg.norm(
            candidate.centers[0]
            - gt_xy.reshape(
                1,
                2,
            ),
            dim=1,
        )

        capture_rows.append(
            bool(
                distance.min().item()
                <= float(
                    config.CANDIDATE_CAPTURE_RADIUS_M
                )
            )
        )

        state = updated_state
        covariance = updated_covariance
        hidden = measurement.hidden

        previous_gt_sd = gt_sd.detach()
        previous_frame = frame_id
        previous_leg_index = leg.index

        end_of_chunk = (
            chunk_steps
            >= int(
                config.TBPTT_STEPS
            )
            or row_number
            == len(indices) - 1
        )

        if end_of_chunk:
            normalized = (
                loss_accumulator
                / float(
                    chunk_steps
                )
            )

            if not torch.isfinite(
                normalized
            ):
                raise FloatingPointError(
                    "non-finite temporal loss"
                )

            normalized.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    config.GRAD_CLIP_NORM
                ),
            )

            optimizer.step()
            optimizer.zero_grad(
                set_to_none=True
            )

            # Carry values, cut graph.
            state = state.detach()
            covariance = covariance.detach()
            hidden = hidden.detach()

            loss_accumulator = None
            chunk_steps = 0

    means = {
        key: float(
            np.mean(
                [
                    row[key]
                    for row in term_rows
                ]
            )
        )
        for key in term_rows[0]
    }

    means["capture_pct"] = float(
        np.mean(
            capture_rows
        )
        * 100.0
    )

    return means


# =============================================================================
# FilterPy evaluation / inference
# =============================================================================

@torch.no_grad()
def evaluate_filterpy(
    model,
    visual,
    cache,
    manifest,
    motion_envelope,
    device,
    start_frame,
    end_frame,
    csv_path=None,
):
    if KalmanFilter is None:
        raise ImportError(
            "FilterPy is required: pip install filterpy"
        )

    model.eval()

    indices = route_frame_indices(
        cache,
        start_frame,
        end_frame,
    )

    if not indices:
        raise RuntimeError(
            "Empty evaluation range"
        )

    kf = make_filterpy_filter()

    hidden = model.initial_hidden(
        1,
        device,
        torch.float32,
    )

    first_frame = int(
        cache.frame_ids[
            indices[0]
        ].item()
    )

    current_leg = active_leg_for_frame(
        manifest,
        first_frame,
    )

    previous_leg_index = current_leg.index
    previous_frame = first_frame - 1

    rows = []

    for index in indices:
        frame_id = int(
            cache.frame_ids[index].item()
        )

        leg = active_leg_for_frame(
            manifest,
            frame_id,
        )

        leg_changed = (
            leg.index
            != previous_leg_index
        )

        if leg_changed:
            reset_filterpy_for_new_leg(
                kf,
                motion_envelope,
            )

        dt = max(
            frame_id
            - previous_frame,
            1,
        )

        previous_progress = float(
            kf.x[0]
        )
        previous_cross = float(
            kf.x[2]
        )

        # Standard FilterPy predict().
        kf.F = filterpy_transition(
            dt
        )
        kf.Q = filterpy_process_covariance(
            dt
        )
        kf.predict()

        kf.x = constrain_route_state_numpy(
            kf.x,
            previous_progress,
            previous_cross,
            leg,
            motion_envelope,
            dt,
        )

        predicted_state_np = (
            np.asarray(
                kf.x,
                dtype=np.float64,
            ).reshape(4)
        )

        candidate_indices = (
            candidate_indices_forward(
                visual,
                predicted_state_np[0],
                previous_progress,
                leg,
                motion_envelope,
            )
        )

        uav_clip = cache.uav_clip[
            index:index + 1
        ].to(
            device
        ).float()

        candidate = (
            candidate_batch_from_indices(
                visual,
                uav_clip,
                candidate_indices,
            )
        )

        candidate_sd = (
            candidate_centers_to_route(
                candidate.centers,
                leg,
            )
        )

        hardms_sd = xy_batch_to_route(
            candidate.hardms_xy,
            leg,
        )

        predicted_state = torch.tensor(
            predicted_state_np,
            device=device,
            dtype=torch.float32,
        ).reshape(
            1,
            4,
        )

        measurement = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_prob,
            candidate_sd,
            hardms_sd,
            candidate.hardms_support,
            predicted_state,
            torch.tensor(
                [previous_progress],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [leg.length],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    motion_envelope.forward_speed_limit
                ],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    motion_envelope.cross_speed_limit
                ],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [leg_changed],
                device=device,
            ),
            hidden,
        )

        hidden = measurement.hidden

        measurement_sd = (
            measurement.measurement_sd[
                0
            ].cpu().numpy().astype(
                np.float64
            )
        )

        measurement_variance = (
            measurement.measurement_variance[
                0
            ].cpu().numpy().astype(
                np.float64
            )
        )

        # Standard FilterPy update().
        kf.R = np.diag(
            measurement_variance
        )
        kf.update(
            measurement_sd
        )

        kf.x = constrain_route_state_numpy(
            kf.x,
            previous_progress,
            previous_cross,
            leg,
            motion_envelope,
            dt,
        )

        final_state = np.asarray(
            kf.x,
            dtype=np.float64,
        ).reshape(4)

        prediction_xy = route_to_xy_numpy(
            predicted_state_np[0],
            predicted_state_np[2],
            leg,
        )

        final_xy = route_to_xy_numpy(
            final_state[0],
            final_state[2],
            leg,
        )

        raw_top1 = candidate.raw_top1_xy[
            0
        ].cpu().numpy()

        hardms = candidate.hardms_xy[
            0
        ].cpu().numpy()

        gt_xy = cache.gt_xy[
            index
        ].numpy()

        gt_sd = xy_to_route(
            cache.gt_xy[index],
            leg,
        ).numpy()

        candidate_distance = np.linalg.norm(
            candidate.centers[
                0
            ].cpu().numpy()
            - gt_xy[None, :],
            axis=1,
        )

        rows.append(
            {
                "frame_id": frame_id,
                "leg_index": int(
                    leg.index
                ),
                "leg_changed": int(
                    leg_changed
                ),
                "gt_x": float(
                    gt_xy[0]
                ),
                "gt_y": float(
                    gt_xy[1]
                ),
                "gt_progress_s": float(
                    gt_sd[0]
                ),
                "gt_cross_d": float(
                    gt_sd[1]
                ),
                "prediction_x": float(
                    prediction_xy[0]
                ),
                "prediction_y": float(
                    prediction_xy[1]
                ),
                "prediction_s": float(
                    predicted_state_np[0]
                ),
                "prediction_v": float(
                    predicted_state_np[1]
                ),
                "prediction_d": float(
                    predicted_state_np[2]
                ),
                "prediction_vd": float(
                    predicted_state_np[3]
                ),
                "raw_top1_x": float(
                    raw_top1[0]
                ),
                "raw_top1_y": float(
                    raw_top1[1]
                ),
                "hardms_x": float(
                    hardms[0]
                ),
                "hardms_y": float(
                    hardms[1]
                ),
                "measurement_s": float(
                    measurement_sd[0]
                ),
                "measurement_d": float(
                    measurement_sd[1]
                ),
                "measurement_var_s": float(
                    measurement_variance[0]
                ),
                "measurement_var_d": float(
                    measurement_variance[1]
                ),
                "final_x": float(
                    final_xy[0]
                ),
                "final_y": float(
                    final_xy[1]
                ),
                "final_s": float(
                    final_state[0]
                ),
                "final_v": float(
                    final_state[1]
                ),
                "final_d": float(
                    final_state[2]
                ),
                "final_vd": float(
                    final_state[3]
                ),
                "candidate_capture": int(
                    candidate_distance.min()
                    <= float(
                        config.CANDIDATE_CAPTURE_RADIUS_M
                    )
                ),
            }
        )

        previous_frame = frame_id
        previous_leg_index = leg.index

    gt = [
        [
            row["gt_x"],
            row["gt_y"],
        ]
        for row in rows
    ]

    summary = {
        "MotionPrediction": metric_block(
            [
                [
                    row["prediction_x"],
                    row["prediction_y"],
                ]
                for row in rows
            ],
            gt,
        ),
        "RawTop1": metric_block(
            [
                [
                    row["raw_top1_x"],
                    row["raw_top1_y"],
                ]
                for row in rows
            ],
            gt,
        ),
        "FixedHardMS": metric_block(
            [
                [
                    row["hardms_x"],
                    row["hardms_y"],
                ]
                for row in rows
            ],
            gt,
        ),
        "RouteCoordinateKalmanFinal": metric_block(
            [
                [
                    row["final_x"],
                    row["final_y"],
                ]
                for row in rows
            ],
            gt,
        ),
        "CandidateCaptureRate_pct": float(
            np.mean(
                [
                    row[
                        "candidate_capture"
                    ]
                    for row in rows
                ]
            )
            * 100.0
        ),
        "MaxForwardStep_m": float(
            max(
                [
                    max(
                        0.0,
                        rows[index][
                            "final_s"
                        ]
                        - rows[index - 1][
                            "final_s"
                        ],
                    )
                    for index in range(
                        1,
                        len(rows),
                    )
                    if rows[index][
                        "leg_index"
                    ]
                    == rows[index - 1][
                        "leg_index"
                    ]
                ]
                or [0.0]
            )
        ),
        "BackwardProgressCount": int(
            sum(
                1
                for index in range(
                    1,
                    len(rows),
                )
                if (
                    rows[index][
                        "leg_index"
                    ]
                    == rows[index - 1][
                        "leg_index"
                    ]
                    and rows[index][
                        "final_s"
                    ]
                    < rows[index - 1][
                        "final_s"
                    ]
                    - 1e-8
                )
            )
        ),
    }

    if csv_path is not None:
        csv_path = Path(
            csv_path
        )
        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                rows
            )

    return (
        summary,
        rows,
    )


def validation_score(
    summary,
):
    metric = summary[
        "RouteCoordinateKalmanFinal"
    ]

    return (
        metric["MLE_m"]
        + float(
            config.VAL_RPE_WEIGHT
        )
        * metric["RPE_m"]
        + float(
            config.VAL_JUMP_WEIGHT
        )
        * metric["JumpRate_pct"]
    )


def train_temporal(
    model,
    visual,
    cache,
    manifest,
    device,
    epochs,
):
    split = split_route_a_legs(
        manifest
    )

    motion_envelope = (
        derive_motion_envelope(
            cache,
            manifest,
            split["train"],
        )
    )

    print(
        "Route A temporal leg split:",
        split,
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config.LR
        ),
        weight_decay=float(
            config.WEIGHT_DECAY
        ),
    )

    best_score = float(
        "inf"
    )
    best_state = None
    patience = 0

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        int(epochs)
    ):
        training = train_one_epoch(
            model,
            optimizer,
            visual,
            cache,
            manifest,
            split["train"],
            motion_envelope,
            device,
        )

        validation, _ = (
            evaluate_filterpy(
                model,
                visual,
                cache,
                manifest,
                motion_envelope,
                device,
                split["val"][0],
                split["val"][1],
                csv_path=None,
            )
        )

        score = validation_score(
            validation
        )

        improved = (
            score
            < best_score
        )

        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
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
                "motion_envelope": motion_envelope.as_dict(),
                "temporal_train_routes": [
                    "route_A"
                ],
                "temporal_validation_routes": [
                    "route_A"
                ],
                "temporal_eval_routes": [
                    "route_B",
                    "route_C",
                ],
                "state": [
                    "route_progress_s",
                    "forward_velocity_v",
                    "cross_track_d",
                    "cross_velocity_vd",
                ],
                "single_frame_streaming": True,
                "filterpy_inference": True,
                "monotonic_progress_constraint": True,
            },
            config.TEMPORAL_CHECKPOINT,
        )

        metric = validation[
            "RouteCoordinateKalmanFinal"
        ]

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
            f"score={score:.3f}",
            flush=True,
        )

        if patience >= int(
            config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "temporal early stopping",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Temporal training produced no best state"
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )
    checkpoint["model"] = best_state
    checkpoint["best_model"] = best_state

    torch.save(
        checkpoint,
        config.TEMPORAL_CHECKPOINT,
    )

    return (
        split,
        motion_envelope,
    )


def load_temporal_checkpoint(
    model,
    device,
):
    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location=device,
    )

    if checkpoint.get(
        "architecture"
    ) != ARCHITECTURE_NAME:
        raise RuntimeError(
            "Checkpoint architecture mismatch"
        )

    state = (
        checkpoint.get(
            "best_model"
        )
        or checkpoint[
            "model"
        ]
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    envelope_dict = checkpoint[
        "motion_envelope"
    ]

    envelope = MotionEnvelope(
        forward_speed_limit=float(
            envelope_dict[
                "forward_speed_limit_m_per_frame"
            ]
        ),
        cross_speed_limit=float(
            envelope_dict[
                "cross_speed_limit_m_per_frame"
            ]
        ),
        percentile=float(
            envelope_dict[
                "percentile"
            ]
        ),
    )

    return (
        checkpoint,
        envelope,
    )


# =============================================================================
# Automatic visualization after inference
# =============================================================================

def render_route_outputs(
    route_name,
    rows,
    output_dir,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import (
            FuncAnimation,
            FFMpegWriter,
        )
    except Exception as exc:
        print(
            "Visualization skipped because matplotlib "
            f"is unavailable: {exc}",
            flush=True,
        )
        return

    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt = np.asarray(
        [
            [
                row["gt_x"],
                row["gt_y"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    final = np.asarray(
        [
            [
                row["final_x"],
                row["final_y"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    frame_ids = np.asarray(
        [
            row["frame_id"]
            for row in rows
        ]
    )

    error = np.linalg.norm(
        final - gt,
        axis=1,
    )

    # 1) trajectory
    fig, ax = plt.subplots(
        figsize=(11, 8)
    )
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        linewidth=2.0,
        label="GT",
    )
    ax.plot(
        final[:, 0],
        final[:, 1],
        linewidth=1.8,
        label="Route-GRU + Kalman Final",
    )
    ax.scatter(
        [gt[0, 0]],
        [gt[0, 1]],
        s=70,
        label="Start",
    )
    ax.set_title(
        f"{route_name}: GT vs Final Route-Coordinate Kalman"
    )
    ax.set_xlabel(
        "Local X (m)"
    )
    ax.set_ylabel(
        "Local Y (m)"
    )
    ax.axis(
        "equal"
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{route_name}_trajectory.png",
        dpi=200,
    )
    plt.close(fig)

    # 2) localization error
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )
    ax.plot(
        frame_ids,
        error,
        label="Final localization error",
    )
    ax.set_title(
        f"{route_name}: localization error"
    )
    ax.set_xlabel(
        "Frame"
    )
    ax.set_ylabel(
        "Error (m)"
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{route_name}_error.png",
        dpi=200,
    )
    plt.close(fig)

    # 3) frame displacement -- direct smoothness view
    if len(rows) > 1:
        gt_step = np.linalg.norm(
            np.diff(
                gt,
                axis=0,
            ),
            axis=1,
        )
        final_step = np.linalg.norm(
            np.diff(
                final,
                axis=0,
            ),
            axis=1,
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )
        ax.plot(
            frame_ids[1:],
            gt_step,
            label="GT displacement/frame",
        )
        ax.plot(
            frame_ids[1:],
            final_step,
            label="Final displacement/frame",
        )
        ax.set_title(
            f"{route_name}: frame-to-frame displacement"
        )
        ax.set_xlabel(
            "Frame"
        )
        ax.set_ylabel(
            "Displacement (m)"
        )
        ax.grid(
            True,
            alpha=0.25,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"{route_name}_displacement.png",
            dpi=200,
        )
        plt.close(fig)

    # 4) animation
    try:
        all_xy = np.vstack(
            [gt, final]
        )

        x_span = max(
            float(
                all_xy[:, 0].max()
                - all_xy[:, 0].min()
            ),
            1.0,
        )
        y_span = max(
            float(
                all_xy[:, 1].max()
                - all_xy[:, 1].min()
            ),
            1.0,
        )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        ax.set_xlim(
            all_xy[:, 0].min()
            - max(
                20.0,
                0.05 * x_span,
            ),
            all_xy[:, 0].max()
            + max(
                20.0,
                0.05 * x_span,
            ),
        )

        ax.set_ylim(
            all_xy[:, 1].min()
            - max(
                20.0,
                0.05 * y_span,
            ),
            all_xy[:, 1].max()
            + max(
                20.0,
                0.05 * y_span,
            ),
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.grid(
            True,
            alpha=0.25,
        )

        gt_line, = ax.plot(
            [],
            [],
            linewidth=2.0,
            label="GT",
        )
        final_line, = ax.plot(
            [],
            [],
            linewidth=1.8,
            label="Final",
        )
        gt_dot, = ax.plot(
            [],
            [],
            marker="o",
            linestyle="None",
        )
        final_dot, = ax.plot(
            [],
            [],
            marker="o",
            linestyle="None",
        )
        title = ax.set_title("")
        ax.legend()

        stride = max(
            1,
            int(
                math.ceil(
                    len(rows)
                    / 500.0
                )
            ),
        )

        animation_indices = list(
            range(
                0,
                len(rows),
                stride,
            )
        )

        if (
            animation_indices[-1]
            != len(rows) - 1
        ):
            animation_indices.append(
                len(rows) - 1
            )

        def update(animation_index):
            index = animation_indices[
                animation_index
            ]

            gt_line.set_data(
                gt[:index + 1, 0],
                gt[:index + 1, 1],
            )
            final_line.set_data(
                final[:index + 1, 0],
                final[:index + 1, 1],
            )
            gt_dot.set_data(
                [gt[index, 0]],
                [gt[index, 1]],
            )
            final_dot.set_data(
                [final[index, 0]],
                [final[index, 1]],
            )
            title.set_text(
                f"{route_name} | "
                f"frame={frame_ids[index]} | "
                f"error={error[index]:.2f}m"
            )

            return (
                gt_line,
                final_line,
                gt_dot,
                final_dot,
                title,
            )

        animation = FuncAnimation(
            fig,
            update,
            frames=len(
                animation_indices
            ),
            interval=50,
            blit=False,
        )

        writer = FFMpegWriter(
            fps=20,
            bitrate=2200,
        )

        animation.save(
            output_dir
            / f"{route_name}_trajectory.mp4",
            writer=writer,
            dpi=120,
        )

        plt.close(fig)

    except Exception as exc:
        print(
            f"{route_name} MP4 skipped: {exc}",
            flush=True,
        )


# =============================================================================
# Main
# =============================================================================

def route_catalog():
    return {
        name: Path(root)
        for name, root
        in zip(
            config.ROUTE_NAMES,
            config.ROUTE_ROOTS,
        )
    }


def main():
    print(
        "[PYTHON] robust_tracker.py main() entered",
        flush=True,
    )
    print(
        "[PYTHON] file:",
        Path(__file__).resolve(),
        flush=True,
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "train",
            "eval",
            "train_eval",
        ),
        default="train_eval",
    )

    parser.add_argument(
        "--visual-epochs",
        type=int,
        default=config.VISUAL_EPOCHS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
    )

    parser.add_argument(
        "--jitter-m",
        type=float,
        default=config.LOCAL_PRIOR_JITTER_M,
    )

    parser.add_argument(
        "--reuse-visual",
        action="store_true",
        help=(
            "Reuse this experiment's visual checkpoint; "
            "restart only the new temporal model."
        ),
    )

    args = parser.parse_args()

    print(
        "[PYTHON] parsed args:",
        {
            "mode": args.mode,
            "visual_epochs": args.visual_epochs,
            "epochs": args.epochs,
            "jitter_m": args.jitter_m,
            "reuse_visual": args.reuse_visual,
        },
        flush=True,
    )

    set_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 88, flush=True)
    print(
        "ROUTE-COORDINATE GRU + CONSTRAINED FILTERPY KALMAN",
        flush=True,
    )
    print("=" * 88, flush=True)
    print("One UAV image per temporal step", flush=True)
    print(
        "State: [route progress s, forward v, cross-track d, cross velocity vd]",
        flush=True,
    )
    print("Inertia: constant-velocity prediction", flush=True)
    print(
        "Progress is structurally monotonic within each active mission leg",
        flush=True,
    )
    print(
        "Per-frame speed envelope is learned only from Route-A training legs",
        flush=True,
    )
    print(
        "Inference filter: filterpy.kalman.KalmanFilter",
        flush=True,
    )
    print(
        "Automatic B/C PNG + MP4 visualization: enabled",
        flush=True,
    )
    print("=" * 88, flush=True)

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode in (
        "train",
        "train_eval",
    ):
        # New temporal model ALWAYS starts from scratch.
        if config.TEMPORAL_CHECKPOINT.exists():
            config.TEMPORAL_CHECKPOINT.unlink()

        if args.reuse_visual:
            print(
                "[STAGE 1/4] reuse visual retrieval checkpoint",
                flush=True,
            )

            if not config.VISUAL_CHECKPOINT.exists():
                raise FileNotFoundError(
                    "--reuse-visual requested but missing: "
                    f"{config.VISUAL_CHECKPOINT}"
                )

            print(
                "Reusing visual checkpoint:",
                config.VISUAL_CHECKPOINT,
                flush=True,
            )
        else:
            print(
                "[STAGE 1/4] train visual retrieval FROM SCRATCH on Route A",
                flush=True,
            )

            if config.VISUAL_CHECKPOINT.exists():
                config.VISUAL_CHECKPOINT.unlink()

            train_visual_retrieval_a_only(
                device=device,
                epochs=int(
                    args.visual_epochs
                ),
                jitter_m=float(
                    args.jitter_m
                ),
                resume=False,
            )

            print(
                "[STAGE 1/4] visual training function returned",
                flush=True,
            )

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Visual checkpoint missing. "
            "Run train_eval without --reuse-visual."
        )

    print(
        "[STAGE 2/4] loading frozen visual localizer/gallery",
        flush=True,
    )

    visual = FrozenVisualLocalizer(
        device
    )

    print(
        "[STAGE 2/4] visual localizer/gallery ready",
        flush=True,
    )

    model = RouteCoordinateGRU().to(
        device
    )

    catalog = route_catalog()

    route_a_manifest = (
        load_waypoint_manifest(
            "route_A",
            visual.origin_lat,
            visual.origin_lon,
        )
    )

    if args.mode in (
        "train",
        "train_eval",
    ):
        print(
            "[STAGE 3/4] building Route-A single-frame backbone cache",
            flush=True,
        )

        route_a_cache = build_backbone_cache(
            "route_A",
            catalog["route_A"],
            visual,
            device,
        )

        print(
            "[STAGE 3/4] Route-A cache ready; starting NEW GRU/Kalman training",
            flush=True,
        )

        split, motion_envelope = (
            train_temporal(
                model,
                visual,
                route_a_cache,
                route_a_manifest,
                device,
                int(
                    args.epochs
                ),
            )
        )
    else:
        _, motion_envelope = (
            load_temporal_checkpoint(
                model,
                device,
            )
        )
        split = split_route_a_legs(
            route_a_manifest
        )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        print(
            "[STAGE 4/4] starting Route-B / Route-C inference",
            flush=True,
        )
        if args.mode == "eval":
            _, motion_envelope = (
                load_temporal_checkpoint(
                    model,
                    device,
                )
            )

        route_results = {}
        visualization_dir = (
            config.OUTPUT_DIR
            / "visualizations"
        )

        for route_name in (
            "route_B",
            "route_C",
        ):
            manifest = load_waypoint_manifest(
                route_name,
                visual.origin_lat,
                visual.origin_lon,
            )

            cache = build_backbone_cache(
                route_name,
                catalog[route_name],
                visual,
                device,
            )

            csv_path = (
                config.OUTPUT_DIR
                / f"{route_name}_route_coordinate_frames.csv"
            )

            summary, rows = evaluate_filterpy(
                model,
                visual,
                cache,
                manifest,
                motion_envelope,
                device,
                manifest.legs[0].start_frame,
                manifest.legs[-1].end_frame,
                csv_path=csv_path,
            )

            route_results[
                route_name
            ] = summary

            render_route_outputs(
                route_name,
                rows,
                visualization_dir,
            )

            metric = summary[
                "RouteCoordinateKalmanFinal"
            ]

            print(
                f"{route_name}: "
                f"MLE={metric['MLE_m']:.3f}m "
                f"P90={metric['P90_m']:.3f}m "
                f"RPE={metric['RPE_m']:.3f}m "
                f"Jump={metric['JumpRate_pct']:.3f}% "
                f"BackwardProgress="
                f"{summary['BackwardProgressCount']} "
                f"Capture="
                f"{summary['CandidateCaptureRate_pct']:.2f}%",
                flush=True,
            )

        payload = {
            "architecture": ARCHITECTURE_NAME,
            "state": [
                "route_progress_s",
                "forward_velocity_v",
                "cross_track_d",
                "cross_velocity_vd",
            ],
            "protocol": {
                "single_frame_streaming": True,
                "visual_train": [
                    "route_A"
                ],
                "temporal_train": [
                    "route_A"
                ],
                "eval": [
                    "route_B",
                    "route_C",
                ],
                "mission_waypoint_prior": True,
                "monotonic_progress": True,
                "motion_envelope_source": (
                    "Route-A temporal training split only"
                ),
                "filter": (
                    "filterpy.kalman.KalmanFilter"
                ),
            },
            "route_A_temporal_split": split,
            "motion_envelope": (
                motion_envelope.as_dict()
            ),
            "routes": route_results,
        }

        config.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = (
            config.OUTPUT_DIR
            / "robust_tracker_summary.json"
        )

        summary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        print(
            "Summary:",
            summary_path,
            flush=True,
        )
        print(
            "Visualizations:",
            visualization_dir,
            flush=True,
        )


if __name__ == "__main__":
    main()
