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
    FrozenVisualLocalizer,
    hard_mean_shift,
    train_visual_retrieval_a_only,
)
from visual_model import RouteInertialLSTM

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "RouteConditionedInertialLSTM_v4"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RouteCache:
    route_name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    uav_clip: torch.Tensor
    image_paths: list

    def __len__(self):
        return int(
            self.gt_xy.shape[0]
        )


@dataclass
class MissionWaypoint:
    order: int
    frame_index: int
    xy: torch.Tensor


@dataclass
class MissionLeg:
    index: int
    start: MissionWaypoint
    end: MissionWaypoint

    @property
    def start_frame(self):
        return int(
            self.start.frame_index
        )

    @property
    def end_frame(self):
        return int(
            self.end.frame_index
        )

    @property
    def start_xy(self):
        return self.start.xy

    @property
    def end_xy(self):
        return self.end.xy


@dataclass
class MissionRoute:
    route_name: str
    waypoints: list
    legs: list


# =============================================================================
# General utilities
# =============================================================================

def set_seed(seed):
    random.seed(
        int(seed)
    )

    np.random.seed(
        int(seed)
    )

    torch.manual_seed(
        int(seed)
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )


def cache_dtype():
    if (
        config.FEATURE_CACHE_DTYPE
        == "float16"
    ):
        return torch.float16

    return torch.float32


def parse_frame_id(value):
    if isinstance(
        value,
        torch.Tensor,
    ):
        return int(
            value.item()
        )

    return int(
        str(value)
    )


# =============================================================================
# Waypoint / route loader
#
# TRAINING:
#   Route A uses waypoint frame boundaries only to know which images belong to
#   a start->end training leg. The NETWORK receives only a translation-invariant
#   route context derived from that start/end pair.
#
# INFERENCE:
#   B/C uses waypoint coordinate/order. frame_index is loaded for diagnostics
#   but is NEVER used to switch legs.
# =============================================================================

def load_mission_route(
    route_name,
    origin_lat,
    origin_lon,
):
    path = Path(
        config.WAYPOINT_FILES[
            route_name
        ]
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    raw_waypoints = sorted(
        payload[
            "waypoints"
        ],
        key=lambda item: int(
            item[
                "waypoint_order"
            ]
        ),
    )

    waypoints = []

    for item in raw_waypoints:
        x_meter, y_meter = meters_from_latlon(
            item[
                "latitude"
            ],
            item[
                "longitude"
            ],
            origin_lat,
            origin_lon,
        )

        waypoints.append(
            MissionWaypoint(
                order=int(
                    item[
                        "waypoint_order"
                    ]
                ),
                frame_index=int(
                    item.get(
                        "frame_index",
                        -1,
                    )
                ),
                xy=torch.tensor(
                    [
                        x_meter,
                        y_meter,
                    ],
                    dtype=torch.float32,
                ),
            )
        )

    if len(
        waypoints
    ) < 2:
        raise RuntimeError(
            f"{route_name}: fewer than two waypoints"
        )

    # Use every adjacent waypoint pair. This avoids stale straight_legs metadata.
    legs = []

    for index in range(
        len(waypoints) - 1
    ):
        legs.append(
            MissionLeg(
                index=index,
                start=waypoints[
                    index
                ],
                end=waypoints[
                    index + 1
                ],
            )
        )

    print(
        f"{route_name}: "
        f"{len(waypoints)} waypoints -> "
        f"{len(legs)} start/end legs",
        flush=True,
    )

    print(
        "  "
        + " -> ".join(
            [
                (
                    f"W{waypoint.order}"
                    f"[f{waypoint.frame_index}]"
                )
                for waypoint
                in waypoints
            ]
        ),
        flush=True,
    )

    return MissionRoute(
        route_name=route_name,
        waypoints=waypoints,
        legs=legs,
    )


# =============================================================================
# Frozen UAV feature cache
# =============================================================================

@torch.no_grad()
def build_route_cache(
    route_name,
    root,
    visual,
    device,
):
    checkpoint_stat = (
        config.VISUAL_CHECKPOINT.stat()
    )

    visual_signature = {
        "path": str(
            config.VISUAL_CHECKPOINT
        ),
        "size": int(
            checkpoint_stat.st_size
        ),
        "mtime_ns": int(
            checkpoint_stat.st_mtime_ns
        ),
    }

    cache_path = (
        config.OUTPUT_DIR
        / "feature_cache"
        / (
            f"{route_name}_"
            "uav_clip.pt"
        )
    )

    if cache_path.exists():
        payload = torch.load(
            cache_path,
            map_location="cpu",
        )

        if (
            payload.get(
                "visual_signature"
            )
            == visual_signature
        ):
            print(
                f"{route_name}: "
                "reuse cached frozen UAV embeddings",
                flush=True,
            )

            return RouteCache(
                route_name=route_name,
                frame_ids=payload[
                    "frame_ids"
                ],
                gt_xy=payload[
                    "gt_xy"
                ],
                uav_clip=payload[
                    "uav_clip"
                ],
                image_paths=payload[
                    "image_paths"
                ],
            )

    dataset = RouteDataset(
        Path(root),
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )

    frame_rows = []
    gt_rows = []
    clip_rows = []
    image_paths = []

    batch_size = int(
        config.VISUAL_CACHE_BATCH_SIZE
    )

    for start in range(
        0,
        len(dataset),
        batch_size,
    ):
        end = min(
            start
            + batch_size,
            len(dataset),
        )

        items = [
            dataset[index]
            for index
            in range(
                start,
                end,
            )
        ]

        uav = torch.stack(
            [
                item[
                    "uav"
                ]
                for item
                in items
            ]
        ).to(
            device
        )

        clip = (
            visual.encode_uav_clip(
                uav
            )
        )

        clip_rows.append(
            clip.detach()
            .cpu()
            .to(
                cache_dtype()
            )
        )

        gt_rows.append(
            torch.stack(
                [
                    item[
                        "xy"
                    ].float()
                    for item
                    in items
                ]
            )
        )

        for item in items:
            frame_rows.append(
                parse_frame_id(
                    item[
                        "frame_id"
                    ]
                )
            )

            image_paths.append(
                str(
                    item[
                        "image_path"
                    ]
                )
            )

        if (
            start == 0
            or end
            == len(dataset)
            or (
                start
                // batch_size
            )
            % 20
            == 0
        ):
            print(
                f"{route_name} visual cache: "
                f"{end}/{len(dataset)}",
                flush=True,
            )

    result = RouteCache(
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
        image_paths=image_paths,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "visual_signature": (
                visual_signature
            ),
            "frame_ids": (
                result.frame_ids
            ),
            "gt_xy": (
                result.gt_xy
            ),
            "uav_clip": (
                result.uav_clip
            ),
            "image_paths": (
                result.image_paths
            ),
        },
        cache_path,
    )

    print(
        f"{route_name}: "
        f"cached frozen UAV embeddings -> {cache_path}",
        flush=True,
    )

    return result


# =============================================================================
# Route geometry.
#
# Absolute waypoint coordinates stay outside the neural network.
# The model receives only route-relative quantities.
# =============================================================================

def leg_geometry(
    leg,
    device=None,
    dtype=torch.float32,
):
    start = leg.start_xy

    end = leg.end_xy

    if device is not None:
        start = start.to(
            device=device,
            dtype=dtype,
        )

        end = end.to(
            device=device,
            dtype=dtype,
        )

    vector = (
        end - start
    )

    length = torch.linalg.norm(
        vector
    ).clamp_min(
        1e-6
    )

    unit = (
        vector
        / length
    )

    normal = torch.stack(
        [
            -unit[1],
            unit[0],
        ]
    )

    return (
        start,
        end,
        unit,
        normal,
        length,
    )


def xy_vector_to_leg_frame(
    vector_xy,
    unit,
    normal,
):
    parallel = (
        vector_xy
        * unit
    ).sum(
        dim=-1
    )

    cross = (
        vector_xy
        * normal
    ).sum(
        dim=-1
    )

    return torch.stack(
        [
            parallel,
            cross,
        ],
        dim=-1,
    )


def leg_frame_vector_to_xy(
    vector_route,
    unit,
    normal,
):
    return (
        vector_route[
            ...,
            0:1
        ]
        * unit
        + vector_route[
            ...,
            1:2
        ]
        * normal
    )


def candidate_offsets_in_leg_frame(
    centers,
    search_center,
    leg,
):
    (
        _,
        _,
        unit,
        normal,
        _,
    ) = leg_geometry(
        leg,
        device=centers.device,
        dtype=centers.dtype,
    )

    relative_xy = (
        centers
        - search_center[
            :,
            None,
            :
        ]
    )

    parallel = (
        relative_xy
        * unit.reshape(
            1,
            1,
            2,
        )
    ).sum(
        dim=2
    )

    cross = (
        relative_xy
        * normal.reshape(
            1,
            1,
            2,
        )
    ).sum(
        dim=2
    )

    return torch.stack(
        [
            parallel,
            cross,
        ],
        dim=2,
    )


def route_context_tensor(
    search_center,
    leg,
):
    (
        start,
        _,
        unit,
        normal,
        length,
    ) = leg_geometry(
        leg,
        device=search_center.device,
        dtype=search_center.dtype,
    )

    relative = (
        search_center
        - start.reshape(
            1,
            2,
        )
    )

    along = (
        relative
        * unit.reshape(
            1,
            2,
        )
    ).sum(
        dim=1
    )

    cross = (
        relative
        * normal.reshape(
            1,
            2,
        )
    ).sum(
        dim=1
    )

    remaining_ratio = (
        (
            length
            - along
        )
        / length
    ).clamp(
        -0.25,
        1.50,
    )

    normalized_cross = (
        cross
        / float(
            config.ROUTE_CROSS_TRACK_SCALE_M
        )
    ).clamp(
        -2.0,
        2.0,
    )

    normalized_log_length = (
        torch.log1p(
            length
        )
        / math.log1p(
            float(
                config.ROUTE_LENGTH_LOG_SCALE_M
            )
        )
    ).reshape(
        1
    ).expand_as(
        remaining_ratio
    )

    return torch.stack(
        [
            remaining_ratio,
            normalized_cross,
            normalized_log_length,
        ],
        dim=1,
    )


def rotate_motion_state(
    motion_state,
    old_leg,
    new_leg,
):
    """
    Rotate [v_parallel, v_cross, a_parallel, a_cross] from the old leg frame
    into the new leg frame. No GPS/GT is involved.
    """
    device = motion_state.device
    dtype = motion_state.dtype

    (
        _,
        _,
        old_unit,
        old_normal,
        _,
    ) = leg_geometry(
        old_leg,
        device=device,
        dtype=dtype,
    )

    (
        _,
        _,
        new_unit,
        new_normal,
        _,
    ) = leg_geometry(
        new_leg,
        device=device,
        dtype=dtype,
    )

    velocity_xy = (
        motion_state[
            :,
            0:1
        ]
        * old_unit.reshape(
            1,
            2,
        )
        + motion_state[
            :,
            1:2
        ]
        * old_normal.reshape(
            1,
            2,
        )
    )

    acceleration_xy = (
        motion_state[
            :,
            2:3
        ]
        * old_unit.reshape(
            1,
            2,
        )
        + motion_state[
            :,
            3:4
        ]
        * old_normal.reshape(
            1,
            2,
        )
    )

    new_velocity = torch.stack(
        [
            (
                velocity_xy
                * new_unit.reshape(
                    1,
                    2,
                )
            ).sum(
                dim=1
            ),
            (
                velocity_xy
                * new_normal.reshape(
                    1,
                    2,
                )
            ).sum(
                dim=1
            ),
        ],
        dim=1,
    )

    new_acceleration = torch.stack(
        [
            (
                acceleration_xy
                * new_unit.reshape(
                    1,
                    2,
                )
            ).sum(
                dim=1
            ),
            (
                acceleration_xy
                * new_normal.reshape(
                    1,
                    2,
                )
            ).sum(
                dim=1
            ),
        ],
        dim=1,
    )

    return torch.cat(
        [
            new_velocity,
            new_acceleration,
        ],
        dim=1,
    )


# =============================================================================
# Visual decoding.
#
# CRITICAL DIFFERENCE FROM THE FAILED VERSION:
# The recurrent state is NOT snapped to a Fixed-HardMS anchor.
# The current synchronized visual state is a continuous expectation over the
# current image-conditioned candidate distribution.
# =============================================================================

def decode_continuous_measurement(
    refined_logits,
    centers,
):
    probability = torch.softmax(
        refined_logits,
        dim=1,
    )

    measurement_xy = (
        probability.unsqueeze(
            -1
        )
        * centers
    ).sum(
        dim=1
    )

    return (
        measurement_xy,
        probability,
    )


def nearest_candidate_label(
    centers,
    gt_xy,
):
    distance = torch.linalg.norm(
        centers
        - gt_xy[
            :,
            None,
            :
        ],
        dim=2,
    )

    return distance.argmin(
        dim=1
    )


# =============================================================================
# Route-A leg split.
# =============================================================================

def split_training_legs(
    route,
):
    count = len(
        route.legs
    )

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

    if (
        train_count
        + val_count
        >= count
    ):
        val_count = max(
            1,
            count
            - train_count
            - 1,
        )

    return {
        "train": route.legs[
            :train_count
        ],
        "val": route.legs[
            train_count:
            train_count
            + val_count
        ],
        "test": route.legs[
            train_count
            + val_count:
        ],
    }


def frame_to_cache_index(
    cache,
):
    return {
        int(
            frame_id
        ): index
        for index, frame_id
        in enumerate(
            cache.frame_ids.tolist()
        )
    }


def indices_for_leg(
    cache,
    leg,
    is_last_route_leg,
):
    index_map = frame_to_cache_index(
        cache
    )

    start_frame = int(
        leg.start_frame
    )

    end_frame_exclusive = int(
        leg.end_frame
    )

    if is_last_route_leg:
        end_frame_exclusive += 1

    indices = []

    for frame_id in range(
        start_frame,
        end_frame_exclusive,
    ):
        if frame_id in index_map:
            indices.append(
                index_map[
                    frame_id
                ]
            )

    if not indices:
        raise RuntimeError(
            "No frames found for "
            f"leg W{leg.start.order}->W{leg.end.order}"
        )

    return indices


def teacher_center_ratio(
    epoch_index,
):
    end_epoch = int(
        config.TEACHER_CENTER_END_EPOCH
    )

    if end_epoch <= 0:
        return 0.0

    if epoch_index >= end_epoch:
        return 0.0

    return max(
        0.0,
        1.0
        - float(
            epoch_index
        )
        / float(
            end_epoch
        ),
    )


# =============================================================================
# Training target construction.
#
# GT is used as SUPERVISION, never as a network input.
#
# Why GT exists:
#   1. tell CE which current satellite patch is correct;
#   2. supervise relative current visual offset;
#   3. supervise next-frame velocity/acceleration state.
#
# No loss asks the LSTM to memorize absolute global coordinates.
# =============================================================================

def motion_targets(
    cache,
    current_index,
    next_index,
    previous_velocity_target,
    leg,
    device,
):
    if next_index is None:
        velocity_target = torch.zeros(
            1,
            2,
            device=device,
            dtype=torch.float32,
        )
    else:
        current_gt = cache.gt_xy[
            current_index
        ].to(
            device
        )

        next_gt = cache.gt_xy[
            next_index
        ].to(
            device
        )

        delta_xy = (
            next_gt
            - current_gt
        )

        (
            _,
            _,
            unit,
            normal,
            _,
        ) = leg_geometry(
            leg,
            device=device,
            dtype=torch.float32,
        )

        velocity_target = (
            xy_vector_to_leg_frame(
                delta_xy.reshape(
                    1,
                    2,
                ),
                unit,
                normal,
            )
        )

    if previous_velocity_target is None:
        acceleration_target = torch.zeros_like(
            velocity_target
        )
    else:
        acceleration_target = (
            velocity_target
            - previous_velocity_target
        )

    return (
        velocity_target,
        acceleration_target,
    )


# =============================================================================
# One epoch of Route-A recurrent training.
# =============================================================================

def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    train_legs,
    all_route_legs,
    device,
    epoch_index,
):
    model.train()

    (
        hidden,
        cell,
        motion_state,
    ) = model.initial_state(
        1,
        device,
        torch.float32,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulated_loss = None
    accumulated_steps = 0

    logs = []

    teacher_ratio = (
        teacher_center_ratio(
            epoch_index
        )
    )

    previous_leg = None
    model_search_center = None
    previous_teacher_gt = None
    previous_velocity_target = None

    for leg_position, leg in enumerate(
        train_legs
    ):
        is_last_route_leg = (
            leg.index
            == all_route_legs[
                -1
            ].index
        )

        leg_indices = indices_for_leg(
            cache,
            leg,
            is_last_route_leg,
        )

        if previous_leg is None:
            model_search_center = (
                leg.start_xy.to(
                    device
                ).reshape(
                    1,
                    2,
                )
            )

            previous_teacher_gt = (
                model_search_center.detach()
            )
        else:
            motion_state = rotate_motion_state(
                motion_state,
                previous_leg,
                leg,
            )

            # Do not teleport the model state to the waypoint.
            # Keep the previous model visual estimate.
            # Teacher scheduled sampling can still stabilize early epochs.
            previous_teacher_gt = (
                leg.start_xy.to(
                    device
                ).reshape(
                    1,
                    2,
                )
            )

        previous_velocity_target = None

        for local_index, cache_index in enumerate(
            leg_indices
        ):
            gt_xy = cache.gt_xy[
                cache_index
            ].to(
                device
            ).reshape(
                1,
                2,
            )

            # Causal teacher center:
            # teacher uses PREVIOUS-frame GT, never current-frame GT.
            candidate_center = (
                float(
                    teacher_ratio
                )
                * previous_teacher_gt
                + (
                    1.0
                    - float(
                        teacher_ratio
                    )
                )
                * model_search_center
            )

            uav_clip = cache.uav_clip[
                cache_index:
                cache_index
                + 1
            ].to(
                device
            ).float()

            candidate = (
                visual.candidate_batch(
                    uav_clip,
                    candidate_center,
                    grid_size=(
                        config.GRID_SIZE
                    ),
                )
            )

            candidate_offsets_route = (
                candidate_offsets_in_leg_frame(
                    candidate.centers,
                    candidate_center,
                    leg,
                )
            )

            route_context = (
                route_context_tensor(
                    candidate_center,
                    leg,
                )
            )

            output = model.forward_step(
                candidate.z_uav,
                candidate.z_sat,
                candidate.raw_logits,
                candidate.raw_prob,
                candidate_offsets_route,
                route_context,
                motion_state,
                hidden,
                cell,
            )

            (
                measurement_xy,
                refined_probability,
            ) = decode_continuous_measurement(
                output.refined_logits,
                candidate.centers,
            )

            target_index = (
                nearest_candidate_label(
                    candidate.centers,
                    gt_xy,
                )
            )

            ce_loss = F.cross_entropy(
                output.refined_logits,
                target_index,
                label_smoothing=float(
                    config.VISUAL_LABEL_SMOOTHING
                ),
            )

            # Relative current localization target.
            (
                _,
                _,
                unit,
                normal,
                _,
            ) = leg_geometry(
                leg,
                device=device,
                dtype=torch.float32,
            )

            target_offset_route = (
                xy_vector_to_leg_frame(
                    gt_xy
                    - candidate_center,
                    unit,
                    normal,
                )
            )

            predicted_offset_route = (
                refined_probability.unsqueeze(
                    -1
                )
                * candidate_offsets_route
            ).sum(
                dim=1
            )

            offset_loss = (
                F.smooth_l1_loss(
                    predicted_offset_route,
                    target_offset_route,
                )
            )

            next_cache_index = None

            if (
                local_index
                + 1
                < len(
                    leg_indices
                )
            ):
                next_cache_index = (
                    leg_indices[
                        local_index
                        + 1
                    ]
                )

            (
                velocity_target,
                acceleration_target,
            ) = motion_targets(
                cache,
                cache_index,
                next_cache_index,
                previous_velocity_target,
                leg,
                device,
            )

            velocity_loss = (
                F.smooth_l1_loss(
                    output.next_motion_state[
                        :,
                        0:2
                    ],
                    velocity_target,
                )
            )

            acceleration_loss = (
                F.smooth_l1_loss(
                    output.next_motion_state[
                        :,
                        2:4
                    ],
                    acceleration_target,
                )
            )

            loss = (
                float(
                    config.LOSS_RETRIEVAL_CE
                )
                * ce_loss
                + float(
                    config.LOSS_RELATIVE_OFFSET
                )
                * offset_loss
                + float(
                    config.LOSS_VELOCITY
                )
                * velocity_loss
                + float(
                    config.LOSS_ACCELERATION
                )
                * acceleration_loss
            )

            if accumulated_loss is None:
                accumulated_loss = loss
            else:
                accumulated_loss = (
                    accumulated_loss
                    + loss
                )

            accumulated_steps += 1

            minimum_gt_distance = (
                torch.linalg.norm(
                    candidate.centers[
                        0
                    ]
                    - gt_xy[
                        0
                    ].reshape(
                        1,
                        2,
                    ),
                    dim=1,
                ).min()
            )

            logs.append(
                {
                    "loss": float(
                        loss.detach()
                        .cpu()
                    ),
                    "ce": float(
                        ce_loss.detach()
                        .cpu()
                    ),
                    "offset": float(
                        offset_loss.detach()
                        .cpu()
                    ),
                    "velocity": float(
                        velocity_loss.detach()
                        .cpu()
                    ),
                    "acceleration": float(
                        acceleration_loss.detach()
                        .cpu()
                    ),
                    "capture": float(
                        minimum_gt_distance.item()
                        <= float(
                            config.CANDIDATE_CAPTURE_RADIUS_M
                        )
                    ),
                    "inertia": float(
                        output.inertia_strength.mean()
                        .detach()
                        .cpu()
                    ),
                    "sigma": float(
                        output.polynomial_sigma.mean()
                        .detach()
                        .cpu()
                    ),
                }
            )

            # ----------------------------------------------------------
            # Recurrent state carried to NEXT frame.
            #
            # IMPORTANT:
            # The search center is the CURRENT VISUAL MEASUREMENT.
            # Polynomial/Kalman are not allowed to advance it.
            # ----------------------------------------------------------
            model_search_center = (
                measurement_xy.detach()
            )

            previous_teacher_gt = (
                gt_xy.detach()
            )

            previous_velocity_target = (
                velocity_target.detach()
            )

            hidden = output.hidden
            cell = output.cell
            motion_state = (
                output.next_motion_state
            )

            chunk_end = (
                accumulated_steps
                >= int(
                    config.TBPTT_STEPS
                )
            )

            final_training_step = (
                leg_position
                == len(
                    train_legs
                )
                - 1
                and local_index
                == len(
                    leg_indices
                )
                - 1
            )

            if (
                chunk_end
                or final_training_step
            ):
                normalized_loss = (
                    accumulated_loss
                    / float(
                        accumulated_steps
                    )
                )

                if not torch.isfinite(
                    normalized_loss
                ):
                    raise FloatingPointError(
                        "non-finite temporal loss"
                    )

                normalized_loss.backward()

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

                hidden = (
                    hidden.detach()
                )

                cell = (
                    cell.detach()
                )

                motion_state = (
                    motion_state.detach()
                )

                model_search_center = (
                    model_search_center.detach()
                )

                accumulated_loss = None
                accumulated_steps = 0

        previous_leg = leg

    if not logs:
        raise RuntimeError(
            "Temporal training produced no steps"
        )

    means = {}

    for key in (
        "loss",
        "ce",
        "offset",
        "velocity",
        "acceleration",
        "capture",
        "inertia",
        "sigma",
    ):
        means[
            key
        ] = float(
            np.mean(
                [
                    row[
                        key
                    ]
                    for row
                    in logs
                ]
            )
        )

    means[
        "capture_pct"
    ] = (
        means[
            "capture"
        ]
        * 100.0
    )

    means[
        "teacher_ratio"
    ] = float(
        teacher_ratio
    )

    return means


# =============================================================================
# Closed-loop validation.
#
# No teacher center is used.
# Each validation leg begins from its known start waypoint and then advances
# only through current image evidence.
# =============================================================================

@torch.no_grad()
def evaluate_closed_loop_legs(
    model,
    visual,
    cache,
    legs,
    all_route_legs,
    device,
):
    model.eval()

    prediction_rows = []
    gt_rows = []

    for leg in legs:
        (
            hidden,
            cell,
            motion_state,
        ) = model.initial_state(
            1,
            device,
            torch.float32,
        )

        search_center = (
            leg.start_xy.to(
                device
            ).reshape(
                1,
                2,
            )
        )

        leg_indices = indices_for_leg(
            cache,
            leg,
            (
                leg.index
                == all_route_legs[
                    -1
                ].index
            ),
        )

        for cache_index in leg_indices:
            uav_clip = cache.uav_clip[
                cache_index:
                cache_index
                + 1
            ].to(
                device
            ).float()

            candidate = (
                visual.candidate_batch(
                    uav_clip,
                    search_center,
                    grid_size=(
                        config.GRID_SIZE
                    ),
                )
            )

            offsets_route = (
                candidate_offsets_in_leg_frame(
                    candidate.centers,
                    search_center,
                    leg,
                )
            )

            context = route_context_tensor(
                search_center,
                leg,
            )

            output = model.forward_step(
                candidate.z_uav,
                candidate.z_sat,
                candidate.raw_logits,
                candidate.raw_prob,
                offsets_route,
                context,
                motion_state,
                hidden,
                cell,
            )

            (
                measurement_xy,
                _,
            ) = decode_continuous_measurement(
                output.refined_logits,
                candidate.centers,
            )

            prediction_rows.append(
                measurement_xy[
                    0
                ].cpu()
                .numpy()
            )

            gt_rows.append(
                cache.gt_xy[
                    cache_index
                ].numpy()
            )

            search_center = (
                measurement_xy
            )

            hidden = output.hidden
            cell = output.cell
            motion_state = (
                output.next_motion_state
            )

    prediction = np.asarray(
        prediction_rows,
        dtype=np.float64,
    )

    gt = np.asarray(
        gt_rows,
        dtype=np.float64,
    )

    error = np.linalg.norm(
        prediction
        - gt,
        axis=1,
    )

    if len(
        prediction
    ) > 1:
        rpe = np.linalg.norm(
            np.diff(
                prediction,
                axis=0,
            )
            - np.diff(
                gt,
                axis=0,
            ),
            axis=1,
        ).mean()
    else:
        rpe = 0.0

    return {
        "MLE_m": float(
            error.mean()
        ),
        "P90_m": float(
            np.percentile(
                error,
                90,
            )
        ),
        "RPE_m": float(
            rpe
        ),
    }


# =============================================================================
# Temporal training driver
# =============================================================================

def train_temporal_model(
    model,
    visual,
    cache,
    training_route,
    device,
    epochs,
):
    split = split_training_legs(
        training_route
    )

    print(
        "Route-A leg split:",
        {
            key: [
                (
                    leg.start.order,
                    leg.end.order,
                )
                for leg
                in value
            ]
            for key, value
            in split.items()
        },
        flush=True,
    )

    print(
        "TRAINING INPUT AUDIT:",
        flush=True,
    )

    print(
        "  network gets current UAV/SAT visual evidence",
        flush=True,
    )

    print(
        "  network gets START/END only as route-relative context",
        flush=True,
    )

    print(
        "  network gets previous recurrent motion "
        "[v_parallel,v_cross,a_parallel,a_cross]",
        flush=True,
    )

    print(
        "  polynomial = v + 0.5*a, used ONLY as a soft candidate-score prior",
        flush=True,
    )

    print(
        "  NO absolute current GT/GPS coordinate is a network input",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config.TEMPORAL_LR
        ),
        weight_decay=float(
            config.TEMPORAL_WEIGHT_DECAY
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

    for epoch_index in range(
        int(
            epochs
        )
    ):
        training = train_one_epoch(
            model,
            optimizer,
            visual,
            cache,
            split[
                "train"
            ],
            training_route.legs,
            device,
            epoch_index,
        )

        validation = (
            evaluate_closed_loop_legs(
                model,
                visual,
                cache,
                split[
                    "val"
                ],
                training_route.legs,
                device,
            )
        )

        score = (
            validation[
                "MLE_m"
            ]
            + 0.25
            * validation[
                "RPE_m"
            ]
        )

        if score < best_score:
            best_score = float(
                score
            )

            best_state = {
                key: (
                    value.detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in model.state_dict().items()
            }

            patience = 0
        else:
            patience += 1

        torch.save(
            {
                "architecture": (
                    ARCHITECTURE_NAME
                ),
                "model": (
                    model.state_dict()
                ),
                "best_model": (
                    best_state
                ),
                "epoch": (
                    epoch_index
                    + 1
                ),
                "best_score": (
                    best_score
                ),
                "training_route": (
                    "route_A"
                ),
                "network_inputs": [
                    "uav_visual_embedding",
                    "sat_visual_embeddings",
                    "visual_similarity",
                    "candidate_relative_offsets_in_leg_frame",
                    "start_end_route_relative_context",
                    "previous_model_motion_state",
                    "previous_lstm_hidden",
                    "previous_lstm_cell",
                ],
                "network_not_given": [
                    "absolute_current_gt_xy",
                    "absolute_current_gps",
                    "waypoint_frame_index",
                    "waypoint_timestamp",
                    "kalman_state",
                ],
                "gt_supervision_only": [
                    "current_candidate_class_label",
                    "relative_visual_offset_target",
                    "next_relative_velocity_target",
                    "next_relative_acceleration_target",
                    "early_scheduled_previous_gt_center",
                ],
                "teacher_center_end_epoch": int(
                    config.TEACHER_CENTER_END_EPOCH
                ),
                "final_training_is_closed_loop": True,
                "polynomial_role": (
                    "soft candidate-score prior only; "
                    "never advances localization state directly"
                ),
            },
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch_index + 1:03d}/{epochs} "
            f"loss={training['loss']:.4f} "
            f"ce={training['ce']:.4f} "
            f"offset={training['offset']:.4f} "
            f"vel={training['velocity']:.4f} "
            f"acc={training['acceleration']:.4f} "
            f"capture={training['capture_pct']:.2f}% "
            f"teacher={training['teacher_ratio']:.3f} "
            f"inertia={training['inertia']:.3f} "
            f"sigma={training['sigma']:.2f}m "
            f"val_mle={validation['MLE_m']:.3f}m "
            f"val_rpe={validation['RPE_m']:.3f}m",
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
            "Temporal training produced no best checkpoint"
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )

    checkpoint[
        "model"
    ] = best_state

    checkpoint[
        "best_model"
    ] = best_state

    torch.save(
        checkpoint,
        config.TEMPORAL_CHECKPOINT,
    )

    return split


def load_temporal_model(
    model,
    device,
):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            config.TEMPORAL_CHECKPOINT
        )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location=device,
    )

    if (
        checkpoint.get(
            "architecture"
        )
        != ARCHITECTURE_NAME
    ):
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch: "
            f"{checkpoint.get('architecture')}"
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

    return checkpoint


# =============================================================================
# Inference leg switching.
#
# frame_index is NOT consulted.
# The current synchronized VISUAL state decides when the endpoint is reached.
# =============================================================================

def reached_leg_endpoint(
    visual_xy,
    leg,
):
    (
        start,
        end,
        unit,
        _,
        length,
    ) = leg_geometry(
        leg,
        device=visual_xy.device,
        dtype=visual_xy.dtype,
    )

    position = visual_xy[
        0
    ]

    distance = torch.linalg.norm(
        position
        - end
    )

    progress = torch.dot(
        position
        - start,
        unit,
    )

    radius = float(
        config.INFER_WAYPOINT_REACHED_RADIUS_M
    )

    return bool(
        (
            distance
            <= radius
        ).item()
        or (
            progress
            >= (
                length
                - radius
            )
        ).item()
    )


# =============================================================================
# Final-output Kalman.
#
# It smooths the RNN visual measurement.
# Its prediction is NEVER used to center the next candidate lattice.
# =============================================================================

def make_kalman_filter(
    initial_xy,
):
    if KalmanFilter is None:
        raise ImportError(
            "FilterPy is required: pip install filterpy"
        )

    kf = KalmanFilter(
        dim_x=4,
        dim_z=2,
    )

    kf.x = np.asarray(
        [
            float(
                initial_xy[
                    0
                ]
            ),
            float(
                initial_xy[
                    1
                ]
            ),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    kf.F = np.asarray(
        [
            [
                1.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    kf.H = np.asarray(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    kf.P = np.diag(
        [
            config.KALMAN_INIT_POSITION_VAR,
            config.KALMAN_INIT_POSITION_VAR,
            config.KALMAN_INIT_VELOCITY_VAR,
            config.KALMAN_INIT_VELOCITY_VAR,
        ]
    ).astype(
        np.float64
    )

    kf.Q = np.diag(
        [
            config.KALMAN_Q_POSITION,
            config.KALMAN_Q_POSITION,
            config.KALMAN_Q_VELOCITY,
            config.KALMAN_Q_VELOCITY,
        ]
    ).astype(
        np.float64
    )

    return kf


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
        prediction
        - gt,
        axis=1,
    )

    if len(
        prediction
    ) > 1:
        prediction_step = np.diff(
            prediction,
            axis=0,
        )

        gt_step = np.diff(
            gt,
            axis=0,
        )

        rpe = np.linalg.norm(
            prediction_step
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

        prediction_step_length = (
            np.linalg.norm(
                prediction_step,
                axis=1,
            )
        )

        jump_rate = float(
            (
                prediction_step_length
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
            np.median(
                error
            )
        ),
        "P90_m": float(
            np.percentile(
                error,
                90,
            )
        ),
        "P95_m": float(
            np.percentile(
                error,
                95,
            )
        ),
        "ATE_RMSE_m": float(
            np.sqrt(
                np.mean(
                    error
                    ** 2
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
            (
                error
                <= 5.0
            ).mean()
            * 100.0
        ),
        "LSR@10_pct": float(
            (
                error
                <= 10.0
            ).mean()
            * 100.0
        ),
        "LSR@15_pct": float(
            (
                error
                <= 15.0
            ).mean()
            * 100.0
        ),
        "LSR@20_pct": float(
            (
                error
                <= 20.0
            ).mean()
            * 100.0
        ),
        "MaxLE_m": float(
            error.max()
        ),
    }


# =============================================================================
# B/C inference.
# =============================================================================

@torch.no_grad()
def run_inference(
    model,
    visual,
    cache,
    route,
    device,
    csv_path,
):
    model.eval()

    (
        hidden,
        cell,
        motion_state,
    ) = model.initial_state(
        1,
        device,
        torch.float32,
    )

    active_leg_index = 0

    active_leg = route.legs[
        active_leg_index
    ]

    # Known mission start.
    visual_state_xy = (
        active_leg.start_xy.to(
            device
        ).reshape(
            1,
            2,
        )
    )

    kf = make_kalman_filter(
        visual_state_xy[
            0
        ].cpu()
        .numpy()
    )

    rows = []

    previous_leg = active_leg

    for sequence_index in range(
        len(cache)
    ):
        frame_id = int(
            cache.frame_ids[
                sequence_index
            ].item()
        )

        active_leg = route.legs[
            active_leg_index
        ]

        search_center = (
            visual_state_xy
        )

        uav_clip = cache.uav_clip[
            sequence_index:
            sequence_index
            + 1
        ].to(
            device
        ).float()

        candidate = (
            visual.candidate_batch(
                uav_clip,
                search_center,
                grid_size=(
                    config.GRID_SIZE
                ),
            )
        )

        offsets_route = (
            candidate_offsets_in_leg_frame(
                candidate.centers,
                search_center,
                active_leg,
            )
        )

        context = route_context_tensor(
            search_center,
            active_leg,
        )

        output = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_logits,
            candidate.raw_prob,
            offsets_route,
            context,
            motion_state,
            hidden,
            cell,
        )

        (
            visual_measurement_xy,
            refined_probability,
        ) = decode_continuous_measurement(
            output.refined_logits,
            candidate.centers,
        )

        # Baselines / diagnostics only.
        raw_top1 = candidate.raw_top1_xy[
            0
        ]

        raw_hardms = candidate.hardms_xy[
            0
        ]

        refined_hardms, refined_support = (
            hard_mean_shift(
                output.refined_logits,
                candidate.centers,
                1.0,
                config.MEANSHIFT_BANDWIDTH_M,
                config.MEANSHIFT_ITERATIONS,
            )
        )

        # Polynomial visualization only.
        (
            _,
            _,
            unit,
            normal,
            _,
        ) = leg_geometry(
            active_leg,
            device=device,
            dtype=torch.float32,
        )

        polynomial_delta_xy = (
            leg_frame_vector_to_xy(
                output.polynomial_delta,
                unit,
                normal,
            )
        )

        polynomial_xy = (
            search_center
            + polynomial_delta_xy
        )

        # --------------------------------------------------------------
        # Synchronization guarantee by architecture:
        # NEXT search center is the current IMAGE-derived measurement.
        # Neither waypoint, polynomial nor Kalman prediction moves it.
        # --------------------------------------------------------------
        visual_state_xy = (
            visual_measurement_xy
        )

        # Final smoother only.
        if sequence_index > 0:
            kf.predict()

        measurement_variance = (
            output.measurement_variance[
                0
            ].cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        kf.R = np.diag(
            measurement_variance
        )

        kf.update(
            visual_measurement_xy[
                0
            ].cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        final_xy = np.asarray(
            kf.x[
                0:2
            ],
            dtype=np.float64,
        )

        # Update recurrent state.
        hidden = output.hidden
        cell = output.cell
        motion_state = (
            output.next_motion_state
        )

        switched = False

        if (
            active_leg_index
            < len(
                route.legs
            )
            - 1
            and reached_leg_endpoint(
                visual_state_xy,
                active_leg,
            )
        ):
            old_leg = (
                active_leg
            )

            active_leg_index += 1

            new_leg = route.legs[
                active_leg_index
            ]

            motion_state = rotate_motion_state(
                motion_state,
                old_leg,
                new_leg,
            )

            previous_leg = old_leg
            switched = True

        gt_xy = cache.gt_xy[
            sequence_index
        ].numpy()

        visual_np = (
            visual_measurement_xy[
                0
            ].cpu()
            .numpy()
        )

        polynomial_np = (
            polynomial_xy[
                0
            ].cpu()
            .numpy()
        )

        raw_top1_np = (
            raw_top1.cpu()
            .numpy()
        )

        raw_hardms_np = (
            raw_hardms.cpu()
            .numpy()
        )

        refined_hardms_np = (
            refined_hardms[
                0
            ].cpu()
            .numpy()
        )

        current_leg_for_log = route.legs[
            active_leg_index
            - 1
            if switched
            else active_leg_index
        ]

        rows.append(
            {
                "sequence_index": int(
                    sequence_index
                ),
                "frame_id": int(
                    frame_id
                ),
                "image_path": (
                    cache.image_paths[
                        sequence_index
                    ]
                ),
                "active_waypoint_from": int(
                    current_leg_for_log.start.order
                ),
                "active_waypoint_to": int(
                    current_leg_for_log.end.order
                ),
                "waypoint_switched_after_frame": int(
                    switched
                ),
                "gt_x": float(
                    gt_xy[
                        0
                    ]
                ),
                "gt_y": float(
                    gt_xy[
                        1
                    ]
                ),
                "search_center_x": float(
                    search_center[
                        0,
                        0
                    ].cpu()
                    .item()
                ),
                "search_center_y": float(
                    search_center[
                        0,
                        1
                    ].cpu()
                    .item()
                ),
                "raw_top1_x": float(
                    raw_top1_np[
                        0
                    ]
                ),
                "raw_top1_y": float(
                    raw_top1_np[
                        1
                    ]
                ),
                "raw_hardms_x": float(
                    raw_hardms_np[
                        0
                    ]
                ),
                "raw_hardms_y": float(
                    raw_hardms_np[
                        1
                    ]
                ),
                "refined_hardms_x": float(
                    refined_hardms_np[
                        0
                    ]
                ),
                "refined_hardms_y": float(
                    refined_hardms_np[
                        1
                    ]
                ),
                "visual_measurement_x": float(
                    visual_np[
                        0
                    ]
                ),
                "visual_measurement_y": float(
                    visual_np[
                        1
                    ]
                ),
                "polynomial_x": float(
                    polynomial_np[
                        0
                    ]
                ),
                "polynomial_y": float(
                    polynomial_np[
                        1
                    ]
                ),
                "poly_v_parallel": float(
                    motion_state[
                        0,
                        0
                    ].cpu()
                    .item()
                ),
                "poly_v_cross": float(
                    motion_state[
                        0,
                        1
                    ].cpu()
                    .item()
                ),
                "poly_a_parallel": float(
                    motion_state[
                        0,
                        2
                    ].cpu()
                    .item()
                ),
                "poly_a_cross": float(
                    motion_state[
                        0,
                        3
                    ].cpu()
                    .item()
                ),
                "inertia_strength": float(
                    output.inertia_strength[
                        0,
                        0
                    ].cpu()
                    .item()
                ),
                "polynomial_sigma_m": float(
                    output.polynomial_sigma[
                        0,
                        0
                    ].cpu()
                    .item()
                ),
                "measurement_var_x": float(
                    measurement_variance[
                        0
                    ]
                ),
                "measurement_var_y": float(
                    measurement_variance[
                        1
                    ]
                ),
                "refined_hardms_support": float(
                    refined_support[
                        0
                    ].cpu()
                    .item()
                ),
                "final_x": float(
                    final_xy[
                        0
                    ]
                ),
                "final_y": float(
                    final_xy[
                        1
                    ]
                ),
                "kf_vx": float(
                    kf.x[
                        2
                    ]
                ),
                "kf_vy": float(
                    kf.x[
                        3
                    ]
                ),
                "error_visual_m": float(
                    np.linalg.norm(
                        visual_np
                        - gt_xy
                    )
                ),
                "error_final_m": float(
                    np.linalg.norm(
                        final_xy
                        - gt_xy
                    )
                ),
            }
        )

    gt = np.asarray(
        [
            [
                row[
                    "gt_x"
                ],
                row[
                    "gt_y"
                ],
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    summary = {
        "RawTop1": metric_block(
            [
                [
                    row[
                        "raw_top1_x"
                    ],
                    row[
                        "raw_top1_y"
                    ],
                ]
                for row
                in rows
            ],
            gt,
        ),
        "RawFixedHardMS": metric_block(
            [
                [
                    row[
                        "raw_hardms_x"
                    ],
                    row[
                        "raw_hardms_y"
                    ],
                ]
                for row
                in rows
            ],
            gt,
        ),
        "RecurrentVisualMeasurement": metric_block(
            [
                [
                    row[
                        "visual_measurement_x"
                    ],
                    row[
                        "visual_measurement_y"
                    ],
                ]
                for row
                in rows
            ],
            gt,
        ),
        "FinalKalman": metric_block(
            [
                [
                    row[
                        "final_x"
                    ],
                    row[
                        "final_y"
                    ],
                ]
                for row
                in rows
            ],
            gt,
        ),
        "WaypointSwitchCount": int(
            sum(
                row[
                    "waypoint_switched_after_frame"
                ]
                for row
                in rows
            )
        ),
        "MeanInertiaStrength": float(
            np.mean(
                [
                    row[
                        "inertia_strength"
                    ]
                    for row
                    in rows
                ]
            )
        ),
        "MeanPolynomialSigma_m": float(
            np.mean(
                [
                    row[
                        "polynomial_sigma_m"
                    ]
                    for row
                    in rows
                ]
            )
        ),
    }

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
                rows[
                    0
                ].keys()
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


# =============================================================================
# Main
# =============================================================================

def route_catalog():
    return {
        name: Path(
            root
        )
        for name, root
        in zip(
            config.ROUTE_NAMES,
            config.ROUTE_ROOTS,
        )
    }


def main():
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
        default=(
            config.VISUAL_EPOCHS
        ),
    )

    parser.add_argument(
        "--temporal-epochs",
        type=int,
        default=(
            config.TEMPORAL_EPOCHS
        ),
    )

    parser.add_argument(
        "--reuse-visual",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "="
        * 96,
        flush=True,
    )

    print(
        "ROUTE-CONDITIONED INERTIAL LSTM",
        flush=True,
    )

    print(
        "="
        * 96,
        flush=True,
    )

    print(
        "CURRENT IMAGE determines CURRENT visual position.",
        flush=True,
    )

    print(
        "Previous RNN motion state supplies a second-order polynomial SOFT prior.",
        flush=True,
    )

    print(
        "Waypoint START/END is converted to relative route context; "
        "raw absolute current GT/GPS is not a network input.",
        flush=True,
    )

    print(
        "Next SAT grid center = previous IMAGE-derived visual state, "
        "never polynomial/Kalman prediction.",
        flush=True,
    )

    print(
        "FilterPy Kalman is final-output smoothing only.",
        flush=True,
    )

    print(
        "="
        * 96,
        flush=True,
    )

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode in (
        "train",
        "train_eval",
    ):
        if args.reuse_visual:
            if not config.VISUAL_CHECKPOINT.exists():
                raise FileNotFoundError(
                    "--reuse-visual requested but visual checkpoint is missing: "
                    f"{config.VISUAL_CHECKPOINT}"
                )

            print(
                "[STAGE 1/4] reuse Route-A visual checkpoint",
                flush=True,
            )
        else:
            print(
                "[STAGE 1/4] train Route-A single-frame visual retrieval",
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
                    config.LOCAL_PRIOR_JITTER_M
                ),
                resume=False,
            )

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            config.VISUAL_CHECKPOINT
        )

    print(
        "[STAGE 2/4] load frozen visual localizer/gallery",
        flush=True,
    )

    visual = FrozenVisualLocalizer(
        device
    )

    model = RouteInertialLSTM().to(
        device
    )

    catalog = route_catalog()

    route_a = load_mission_route(
        "route_A",
        visual.origin_lat,
        visual.origin_lon,
    )

    split_description = None

    if args.mode in (
        "train",
        "train_eval",
    ):
        if config.TEMPORAL_CHECKPOINT.exists():
            config.TEMPORAL_CHECKPOINT.unlink()

        print(
            "[STAGE 3/4] train recurrent model on Route-A start/end legs",
            flush=True,
        )

        route_a_cache = build_route_cache(
            "route_A",
            catalog[
                "route_A"
            ],
            visual,
            device,
        )

        split = train_temporal_model(
            model,
            visual,
            route_a_cache,
            route_a,
            device,
            int(
                args.temporal_epochs
            ),
        )

        split_description = {
            key: [
                [
                    leg.start.order,
                    leg.end.order,
                ]
                for leg
                in value
            ]
            for key, value
            in split.items()
        }
    else:
        load_temporal_model(
            model,
            device,
        )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        if args.mode == "eval":
            load_temporal_model(
                model,
                device,
            )

        print(
            "[STAGE 4/4] Route-B / Route-C closed-loop inference",
            flush=True,
        )

        route_results = {}
        waypoint_counts = {}

        for route_name in (
            "route_B",
            "route_C",
        ):
            route = load_mission_route(
                route_name,
                visual.origin_lat,
                visual.origin_lon,
            )

            waypoint_counts[
                route_name
            ] = len(
                route.waypoints
            )

            cache = build_route_cache(
                route_name,
                catalog[
                    route_name
                ],
                visual,
                device,
            )

            csv_path = (
                config.OUTPUT_DIR
                / (
                    f"{route_name}_"
                    "route_inertial_lstm_frames.csv"
                )
            )

            (
                summary,
                _,
            ) = run_inference(
                model,
                visual,
                cache,
                route,
                device,
                csv_path,
            )

            route_results[
                route_name
            ] = summary

            visual_metric = summary[
                "RecurrentVisualMeasurement"
            ]

            final_metric = summary[
                "FinalKalman"
            ]

            print(
                f"{route_name}: "
                f"Visual MLE={visual_metric['MLE_m']:.3f}m "
                f"Visual RPE={visual_metric['RPE_m']:.3f}m "
                f"| Final MLE={final_metric['MLE_m']:.3f}m "
                f"Final P90={final_metric['P90_m']:.3f}m "
                f"Final Jump={final_metric['JumpRate_pct']:.3f}% "
                f"| waypoint switches={summary['WaypointSwitchCount']}",
                flush=True,
            )

        payload = {
            "architecture": (
                ARCHITECTURE_NAME
            ),
            "training": {
                "route": (
                    "route_A"
                ),
                "waypoint_start_end_used": True,
                "raw_absolute_current_gt_is_network_input": False,
                "raw_gps_is_network_input": False,
                "network_position_representation": (
                    "translation-invariant route-relative context + "
                    "local candidate offsets"
                ),
                "gt_role": (
                    "supervised labels/relative motion targets only"
                ),
                "scheduled_sampling": (
                    "candidate center uses previous-frame GT only during "
                    "early training; final epochs are fully closed-loop"
                ),
                "leg_split": (
                    split_description
                ),
            },
            "model": {
                "recurrent_state": (
                    "LSTM hidden/cell"
                ),
                "explicit_motion_state": [
                    "v_parallel",
                    "v_cross",
                    "a_parallel",
                    "a_cross",
                ],
                "polynomial": (
                    "delta = v + 0.5*a"
                ),
                "polynomial_role": (
                    "soft candidate-score prior only"
                ),
                "current_position_source": (
                    "continuous current-image candidate probability expectation"
                ),
            },
            "inference": {
                "waypoint_coordinates_order_used": True,
                "waypoint_frame_index_used_for_switching": False,
                "test_gt_used_by_inference": False,
                "next_search_center": (
                    "previous image-derived visual measurement"
                ),
                "kalman_controls_search": False,
                "kalman_role": (
                    "final output smoother only"
                ),
            },
            "waypoint_counts": (
                waypoint_counts
            ),
            "routes": (
                route_results
            ),
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
            "summary:",
            summary_path,
            flush=True,
        )


if __name__ == "__main__":
    main()
