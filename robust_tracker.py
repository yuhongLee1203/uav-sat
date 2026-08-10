import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
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
from visual_model import VisualMotionRouteLSTM

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "VisualMotionGatedRouteLSTM_v5"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RouteCache:
    route_name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    raw_gt_xy: torch.Tensor
    uav_clip: torch.Tensor
    image_motion_cues: torch.Tensor
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
# General
# =============================================================================

def set_seed(seed):
    random.seed(
        int(
            seed
        )
    )

    np.random.seed(
        int(
            seed
        )
    )

    torch.manual_seed(
        int(
            seed
        )
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(
                seed
            )
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
        str(
            value
        )
    )


# =============================================================================
# Mission route
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
        x_meter, y_meter = (
            meters_from_latlon(
                item[
                    "latitude"
                ],
                item[
                    "longitude"
                ],
                origin_lat,
                origin_lon,
            )
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

    legs = []

    for index in range(
        len(
            waypoints
        )
        - 1
    ):
        legs.append(
            MissionLeg(
                index=index,
                start=waypoints[
                    index
                ],
                end=waypoints[
                    index
                    + 1
                ],
            )
        )

    print(
        f"{route_name}: "
        f"{len(waypoints)} waypoints -> "
        f"{len(legs)} adjacent start/end legs",
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
# Pure-image pair motion cue
# =============================================================================

def _read_motion_gray(path):
    image = cv2.imread(
        str(
            path
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            path
        )

    size = int(
        config.ECC_IMAGE_SIZE
    )

    image = cv2.resize(
        image,
        (
            size,
            size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return (
        image.astype(
            np.float32
        )
        / 255.0
    )


def estimate_image_pair_motion(
    previous_path,
    current_path,
):
    """
    Image-only motion evidence.

    The cue is NOT converted to metres and does not move the localization.
    It only helps the LSTM distinguish:
      stationary / translation / in-place rotation.
    """
    if previous_path is None:
        return np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    previous = _read_motion_gray(
        previous_path
    )

    current = _read_motion_gray(
        current_path
    )

    size = float(
        config.ECC_IMAGE_SIZE
    )

    warp = np.eye(
        2,
        3,
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        int(
            config.ECC_ITERATIONS
        ),
        float(
            config.ECC_EPSILON
        ),
    )

    correlation = 0.0

    try:
        correlation, warp = (
            cv2.findTransformECC(
                previous,
                current,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                1,
            )
        )

        center = np.asarray(
            [
                size
                * 0.5,
                size
                * 0.5,
                1.0,
            ],
            dtype=np.float32,
        )

        mapped_center = (
            warp
            @ center
        )

        center_shift = (
            mapped_center
            - center[
                :2
            ]
        )

        theta = math.atan2(
            float(
                warp[
                    1,
                    0
                ]
            ),
            float(
                warp[
                    0,
                    0
                ]
            ),
        )

        dx_norm = float(
            center_shift[
                0
            ]
            / size
        )

        dy_norm = float(
            center_shift[
                1
            ]
            / size
        )

        cue = np.asarray(
            [
                dx_norm,
                dy_norm,
                math.sin(
                    theta
                ),
                1.0
                - math.cos(
                    theta
                ),
                float(
                    np.clip(
                        correlation,
                        0.0,
                        1.0,
                    )
                ),
            ],
            dtype=np.float32,
        )

        if np.all(
            np.isfinite(
                cue
            )
        ):
            return cue

    except cv2.error:
        pass

    # Fallback: translation-only phase correlation.
    try:
        (
            shift_x_y,
            response,
        ) = cv2.phaseCorrelate(
            previous,
            current,
        )

        cue = np.asarray(
            [
                float(
                    shift_x_y[
                        0
                    ]
                    / size
                ),
                float(
                    shift_x_y[
                        1
                    ]
                    / size
                ),
                0.0,
                0.0,
                float(
                    np.clip(
                        response,
                        0.0,
                        1.0,
                    )
                ),
            ],
            dtype=np.float32,
        )

        if np.all(
            np.isfinite(
                cue
            )
        ):
            return cue

    except cv2.error:
        pass

    return np.zeros(
        int(
            config.IMAGE_MOTION_CUE_DIM
        ),
        dtype=np.float32,
    )


# =============================================================================
# Route cache
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
        "ecc_size": int(
            config.ECC_IMAGE_SIZE
        ),
        "ecc_iterations": int(
            config.ECC_ITERATIONS
        ),
    }

    cache_path = (
        config.OUTPUT_DIR
        / "feature_cache"
        / (
            f"{route_name}_"
            "visual_motion_cache.pt"
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
                "reuse cached UAV embeddings + image-pair motion cues",
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
                raw_gt_xy=payload[
                    "raw_gt_xy"
                ],
                uav_clip=payload[
                    "uav_clip"
                ],
                image_motion_cues=payload[
                    "image_motion_cues"
                ],
                image_paths=payload[
                    "image_paths"
                ],
            )

    dataset = RouteDataset(
        Path(
            root
        ),
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )

    frame_rows = []
    gt_rows = []
    raw_gt_rows = []
    clip_rows = []
    image_paths = []

    batch_size = int(
        config.VISUAL_CACHE_BATCH_SIZE
    )

    for start in range(
        0,
        len(
            dataset
        ),
        batch_size,
    ):
        end = min(
            start
            + batch_size,
            len(
                dataset
            ),
        )

        items = [
            dataset[
                index
            ]
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

        clip = visual.encode_uav_clip(
            uav
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

        raw_gt_rows.append(
            torch.stack(
                [
                    item[
                        "raw_xy"
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
            start
            == 0
            or end
            == len(
                dataset
            )
            or (
                start
                // batch_size
            )
            % 10
            == 0
        ):
            print(
                f"{route_name} backbone cache: "
                f"{end}/{len(dataset)}",
                flush=True,
            )

    motion_cues = []

    previous_path = None

    for index, image_path in enumerate(
        image_paths
    ):
        cue = estimate_image_pair_motion(
            previous_path,
            image_path,
        )

        motion_cues.append(
            cue
        )

        previous_path = (
            image_path
        )

        if (
            index
            == 0
            or index
            + 1
            == len(
                image_paths
            )
            or (
                index
                + 1
            )
            % 250
            == 0
        ):
            print(
                f"{route_name} image-pair motion cue: "
                f"{index + 1}/{len(image_paths)}",
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
        raw_gt_xy=torch.cat(
            raw_gt_rows
        ).float(),
        uav_clip=torch.cat(
            clip_rows
        ),
        image_motion_cues=torch.tensor(
            np.asarray(
                motion_cues
            ),
            dtype=torch.float32,
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
            "raw_gt_xy": (
                result.raw_gt_xy
            ),
            "uav_clip": (
                result.uav_clip
            ),
            "image_motion_cues": (
                result.image_motion_cues
            ),
            "image_paths": (
                result.image_paths
            ),
        },
        cache_path,
    )

    print(
        f"{route_name}: cache saved -> "
        f"{cache_path}",
        flush=True,
    )

    return result


# =============================================================================
# Route geometry
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
        end
        - start
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
            -unit[
                1
            ],
            unit[
                0
            ],
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
    final_leg,
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

    final_leg_tensor = torch.full_like(
        remaining_ratio,
        1.0
        if final_leg
        else 0.0,
    )

    return torch.stack(
        [
            remaining_ratio,
            normalized_cross,
            normalized_log_length,
            final_leg_tensor,
        ],
        dim=1,
    )


def rotate_observed_motion_state(
    state,
    old_leg,
    new_leg,
):
    device = state.device
    dtype = state.dtype

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
        state[
            :,
            0:1
        ]
        * old_unit.reshape(
            1,
            2,
        )
        + state[
            :,
            1:2
        ]
        * old_normal.reshape(
            1,
            2,
        )
    )

    acceleration_xy = (
        state[
            :,
            2:3
        ]
        * old_unit.reshape(
            1,
            2,
        )
        + state[
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
# Visual proposal + gated synchronized state
# =============================================================================

def decode_visual_proposal(
    refined_logits,
    centers,
):
    probability = torch.softmax(
        refined_logits,
        dim=1,
    )

    proposal_xy = (
        probability.unsqueeze(
            -1
        )
        * centers
    ).sum(
        dim=1
    )

    return (
        proposal_xy,
        probability,
    )


def apply_translation_gate(
    previous_visual_xy,
    proposal_xy,
    translation_gate,
):
    return (
        previous_visual_xy
        + translation_gate
        * (
            proposal_xy
            - previous_visual_xy
        )
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


def update_observed_motion_state(
    previous_state,
    previous_visual_xy,
    current_visual_xy,
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
        device=current_visual_xy.device,
        dtype=current_visual_xy.dtype,
    )

    delta_xy = (
        current_visual_xy
        - previous_visual_xy
    )

    current_velocity = (
        xy_vector_to_leg_frame(
            delta_xy,
            unit,
            normal,
        )
    )

    previous_velocity = (
        previous_state[
            :,
            0:2
        ]
    )

    current_acceleration = (
        current_velocity
        - previous_velocity
    )

    return torch.cat(
        [
            current_velocity,
            current_acceleration,
        ],
        dim=1,
    )


# =============================================================================
# Image-motion phase soft supervision
# =============================================================================

def phase_soft_target(
    image_motion_cue,
    current_gt,
    previous_gt,
):
    """
    Soft 3-state target:
      stationary / translation / rotation

    This is training supervision only.

    High-quality ECC alignment dominates when correlation is high.
    If ECC is unreliable, the target falls back toward supervised GT step
    magnitude. Neither quantity is ever a network input as a position/speed.
    """
    cue = image_motion_cue

    shift_norm = torch.sqrt(
        cue[
            :,
            0
        ].square()
        + cue[
            :,
            1
        ].square()
        + 1e-12
    )

    ecc_translation = (
        1.0
        - torch.exp(
            -float(
                config.ECC_TRANSLATION_GAIN
            )
            * shift_norm
        )
    ).clamp(
        0.0,
        1.0,
    )

    sin_theta = cue[
        :,
        2
    ]

    cos_theta = (
        1.0
        - cue[
            :,
            3
        ]
    )

    rotation_angle = torch.atan2(
        sin_theta.abs(),
        cos_theta.clamp(
            min=-1.0,
            max=1.0,
        ),
    )

    rotation_scale = math.radians(
        float(
            config.ECC_ROTATION_SCALE_DEG
        )
    )

    rotation_strength = (
        rotation_angle
        / max(
            rotation_scale,
            1e-6,
        )
    ).clamp(
        0.0,
        1.0,
    )

    if previous_gt is None:
        gt_translation = torch.zeros_like(
            ecc_translation
        )
    else:
        gt_step = torch.linalg.norm(
            current_gt
            - previous_gt,
            dim=1,
        )

        gt_translation = (
            1.0
            - torch.exp(
                -gt_step
                / float(
                    config.GT_TRANSLATION_SOFT_SCALE_M
                )
            )
        ).clamp(
            0.0,
            1.0,
        )

    reliability = cue[
        :,
        4
    ].clamp(
        0.0,
        1.0,
    )

    translation = (
        reliability
        * ecc_translation
        + (
            1.0
            - reliability
        )
        * gt_translation
    ).clamp(
        0.0,
        1.0,
    )

    rotation = (
        (
            1.0
            - translation
        )
        * reliability
        * rotation_strength
    ).clamp(
        0.0,
        1.0,
    )

    stationary = (
        1.0
        - translation
        - rotation
    ).clamp(
        0.0,
        1.0,
    )

    target = torch.stack(
        [
            stationary,
            translation,
            rotation,
        ],
        dim=1,
    )

    return (
        target
        / target.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(
            1e-6
        )
    )


def soft_cross_entropy(
    logits,
    target_probability,
):
    return -(
        target_probability
        * torch.log_softmax(
            logits,
            dim=1,
        )
    ).sum(
        dim=1
    ).mean()


# =============================================================================
# Route-A leg split
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
            f"W{leg.start.order}->W{leg.end.order}"
        )

    return indices


def teacher_center_ratio(
    epoch_index,
):
    end_epoch = int(
        config.TEACHER_CENTER_END_EPOCH
    )

    if (
        end_epoch
        <= 0
        or epoch_index
        >= end_epoch
    ):
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
# Training
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
        observed_motion,
    ) = model.initial_state(
        1,
        device,
        torch.float32,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    teacher_ratio = (
        teacher_center_ratio(
            epoch_index
        )
    )

    accumulated_loss = None
    accumulated_steps = 0

    logs = []

    previous_leg = None
    previous_visual_xy = None
    previous_gt = None
    previous_z_uav = None

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

        if previous_visual_xy is None:
            previous_visual_xy = (
                leg.start_xy.to(
                    device
                ).reshape(
                    1,
                    2,
                )
            )

            previous_gt = (
                previous_visual_xy.detach()
            )
        elif previous_leg is not None:
            observed_motion = (
                rotate_observed_motion_state(
                    observed_motion,
                    previous_leg,
                    leg,
                )
            )

        final_leg = (
            leg.index
            == all_route_legs[
                -1
            ].index
        )

        for local_index, cache_index in enumerate(
            leg_indices
        ):
            current_gt = cache.gt_xy[
                cache_index
            ].to(
                device
            ).reshape(
                1,
                2,
            )

            # Causal teacher center only.
            # It uses PREVIOUS GT, never current GT.
            search_center = (
                float(
                    teacher_ratio
                )
                * previous_gt
                + (
                    1.0
                    - float(
                        teacher_ratio
                    )
                )
                * previous_visual_xy
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

            route_context = (
                route_context_tensor(
                    search_center,
                    leg,
                    final_leg,
                )
            )

            image_motion_cue = (
                cache.image_motion_cues[
                    cache_index:
                    cache_index
                    + 1
                ].to(
                    device
                ).float()
            )

            output = model.forward_step(
                candidate.z_uav,
                previous_z_uav,
                candidate.z_sat,
                candidate.raw_logits,
                candidate.raw_prob,
                offsets_route,
                route_context,
                image_motion_cue,
                observed_motion,
                hidden,
                cell,
            )

            (
                proposal_xy,
                refined_probability,
            ) = decode_visual_proposal(
                output.refined_logits,
                candidate.centers,
            )

            current_visual_xy = (
                apply_translation_gate(
                    previous_visual_xy,
                    proposal_xy,
                    output.translation_gate,
                )
            )

            target_index = (
                nearest_candidate_label(
                    candidate.centers,
                    current_gt,
                )
            )

            ce_loss = F.cross_entropy(
                output.refined_logits,
                target_index,
                label_smoothing=float(
                    config.VISUAL_LABEL_SMOOTHING
                ),
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

            target_relative_route = (
                xy_vector_to_leg_frame(
                    current_gt
                    - search_center,
                    unit,
                    normal,
                )
            )

            predicted_relative_route = (
                xy_vector_to_leg_frame(
                    current_visual_xy
                    - search_center,
                    unit,
                    normal,
                )
            )

            relative_position_loss = (
                F.smooth_l1_loss(
                    predicted_relative_route,
                    target_relative_route,
                )
            )

            phase_target = phase_soft_target(
                image_motion_cue,
                current_gt,
                previous_gt,
            )

            phase_loss = soft_cross_entropy(
                output.phase_logits,
                phase_target,
            )

            if previous_gt is None:
                target_step_route = torch.zeros_like(
                    predicted_relative_route
                )
            else:
                target_step_route = (
                    xy_vector_to_leg_frame(
                        current_gt
                        - previous_gt,
                        unit,
                        normal,
                    )
                )

            predicted_step_route = (
                xy_vector_to_leg_frame(
                    current_visual_xy
                    - previous_visual_xy,
                    unit,
                    normal,
                )
            )

            step_loss = F.smooth_l1_loss(
                predicted_step_route,
                target_step_route,
            )

            loss = (
                float(
                    config.LOSS_RETRIEVAL_CE
                )
                * ce_loss
                + float(
                    config.LOSS_CURRENT_RELATIVE_POSITION
                )
                * relative_position_loss
                + float(
                    config.LOSS_PHASE_SOFT_CE
                )
                * phase_loss
                + float(
                    config.LOSS_STEP_DISPLACEMENT
                )
                * step_loss
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
                    - current_gt[
                        0
                    ].reshape(
                        1,
                        2,
                    ),
                    dim=1,
                ).min()
            )

            predicted_phase = int(
                output.phase_probability[
                    0
                ].argmax()
                .detach()
                .cpu()
                .item()
            )

            cue_shift = float(
                torch.sqrt(
                    image_motion_cue[
                        0,
                        0
                    ].square()
                    + image_motion_cue[
                        0,
                        1
                    ].square()
                ).detach()
                .cpu()
                .item()
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
                    "relative": float(
                        relative_position_loss.detach()
                        .cpu()
                    ),
                    "phase": float(
                        phase_loss.detach()
                        .cpu()
                    ),
                    "step": float(
                        step_loss.detach()
                        .cpu()
                    ),
                    "capture": float(
                        minimum_gt_distance.item()
                        <= float(
                            config.CANDIDATE_CAPTURE_RADIUS_M
                        )
                    ),
                    "gate": float(
                        output.translation_gate.mean()
                        .detach()
                        .cpu()
                    ),
                    "stationary": float(
                        output.stationary_probability.mean()
                        .detach()
                        .cpu()
                    ),
                    "rotation": float(
                        output.rotation_probability.mean()
                        .detach()
                        .cpu()
                    ),
                    "phase_class": float(
                        predicted_phase
                    ),
                    "ecc_shift": cue_shift,
                }
            )

            new_observed_motion = (
                update_observed_motion_state(
                    observed_motion,
                    previous_visual_xy,
                    current_visual_xy,
                    leg,
                )
            )

            hidden = (
                output.hidden
            )

            cell = (
                output.cell
            )

            observed_motion = (
                new_observed_motion
            )

            previous_z_uav = (
                candidate.z_uav.detach()
            )

            previous_visual_xy = (
                current_visual_xy
            )

            previous_gt = (
                current_gt.detach()
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
                accumulated_steps
                >= int(
                    config.TBPTT_STEPS
                )
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

                observed_motion = (
                    observed_motion.detach()
                )

                previous_visual_xy = (
                    previous_visual_xy.detach()
                )

                previous_z_uav = (
                    previous_z_uav.detach()
                )

                accumulated_loss = None
                accumulated_steps = 0

        previous_leg = leg

    if not logs:
        raise RuntimeError(
            "Temporal training produced no steps"
        )

    result = {}

    for key in (
        "loss",
        "ce",
        "relative",
        "phase",
        "step",
        "capture",
        "gate",
        "stationary",
        "rotation",
        "ecc_shift",
    ):
        result[
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

    result[
        "capture_pct"
    ] = (
        result[
            "capture"
        ]
        * 100.0
    )

    result[
        "teacher_ratio"
    ] = float(
        teacher_ratio
    )

    return result


# =============================================================================
# Closed-loop Route-A validation
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
            observed_motion,
        ) = model.initial_state(
            1,
            device,
            torch.float32,
        )

        visual_xy = (
            leg.start_xy.to(
                device
            ).reshape(
                1,
                2,
            )
        )

        previous_z_uav = None

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

        final_leg = (
            leg.index
            == all_route_legs[
                -1
            ].index
        )

        for cache_index in leg_indices:
            search_center = (
                visual_xy
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
                final_leg,
            )

            image_motion_cue = (
                cache.image_motion_cues[
                    cache_index:
                    cache_index
                    + 1
                ].to(
                    device
                ).float()
            )

            output = model.forward_step(
                candidate.z_uav,
                previous_z_uav,
                candidate.z_sat,
                candidate.raw_logits,
                candidate.raw_prob,
                offsets_route,
                context,
                image_motion_cue,
                observed_motion,
                hidden,
                cell,
            )

            (
                proposal_xy,
                _,
            ) = decode_visual_proposal(
                output.refined_logits,
                candidate.centers,
            )

            new_visual_xy = (
                apply_translation_gate(
                    visual_xy,
                    proposal_xy,
                    output.translation_gate,
                )
            )

            prediction_rows.append(
                new_visual_xy[
                    0
                ].cpu()
                .numpy()
            )

            gt_rows.append(
                cache.gt_xy[
                    cache_index
                ].numpy()
            )

            observed_motion = (
                update_observed_motion_state(
                    observed_motion,
                    visual_xy,
                    new_visual_xy,
                    leg,
                )
            )

            visual_xy = (
                new_visual_xy
            )

            previous_z_uav = (
                candidate.z_uav
            )

            hidden = output.hidden
            cell = output.cell

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
# Temporal driver
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
        "v5 MOTION RULE:",
        flush=True,
    )

    print(
        "  NO fixed speed / nominal speed / learned free-running velocity head",
        flush=True,
    )

    print(
        "  previous motion = displacement actually observed from previous "
        "IMAGE-derived localizations",
        flush=True,
    )

    print(
        "  current-vs-previous UAV image pair predicts "
        "stationary / translation / rotation",
        flush=True,
    )

    print(
        "  polynomial is disabled automatically when translation probability is low",
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
                "fixed_speed_used": False,
                "nominal_speed_used": False,
                "free_running_velocity_head": False,
                "previous_motion_state_source": (
                    "previous image-derived localization differences"
                ),
                "image_pair_motion_cue": (
                    "ECC Euclidean registration + previous/current UAV embeddings"
                ),
                "phase_classes": [
                    "stationary",
                    "translation",
                    "rotation",
                ],
                "polynomial": (
                    "observed_delta + 0.5 * observed_acceleration"
                ),
                "polynomial_role": (
                    "translation-gated soft candidate-score prior only"
                ),
                "absolute_current_gt_network_input": False,
                "raw_gps_network_input": False,
                "waypoint_frame_index_inference_switch": False,
                "teacher_center_end_epoch": int(
                    config.TEACHER_CENTER_END_EPOCH
                ),
            },
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch_index + 1:03d}/{epochs} "
            f"loss={training['loss']:.4f} "
            f"ce={training['ce']:.4f} "
            f"rel={training['relative']:.4f} "
            f"phase={training['phase']:.4f} "
            f"step={training['step']:.4f} "
            f"capture={training['capture_pct']:.2f}% "
            f"teacher={training['teacher_ratio']:.2f} "
            f"move_gate={training['gate']:.3f} "
            f"stop={training['stationary']:.3f} "
            f"rotate={training['rotation']:.3f} "
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
# Endpoint / terminal handling
# =============================================================================

def endpoint_distance(
    visual_xy,
    leg,
):
    endpoint = leg.end_xy.to(
        device=visual_xy.device,
        dtype=visual_xy.dtype,
    ).reshape(
        1,
        2,
    )

    return float(
        torch.linalg.norm(
            visual_xy
            - endpoint,
            dim=1,
        )[
            0
        ].item()
    )


def endpoint_reached(
    visual_xy,
    leg,
):
    # v4's progress >= length-radius shortcut is deliberately removed.
    # The image-derived location must actually enter the endpoint neighborhood.
    return (
        endpoint_distance(
            visual_xy,
            leg,
        )
        <= float(
            config.INFER_WAYPOINT_REACHED_RADIUS_M
        )
    )


# =============================================================================
# Position-only FilterPy Kalman
# =============================================================================

def make_position_kalman(
    initial_xy,
):
    if KalmanFilter is None:
        raise ImportError(
            "FilterPy is required: pip install filterpy"
        )

    kf = KalmanFilter(
        dim_x=2,
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
        ],
        dtype=np.float64,
    )

    # Random-walk position state. NO velocity.
    kf.F = np.eye(
        2,
        dtype=np.float64,
    )

    kf.H = np.eye(
        2,
        dtype=np.float64,
    )

    kf.P = (
        np.eye(
            2,
            dtype=np.float64,
        )
        * float(
            config.KALMAN_INIT_POSITION_VAR
        )
    )

    kf.Q = (
        np.eye(
            2,
            dtype=np.float64,
        )
        * float(
            config.KALMAN_Q_POSITION
        )
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

        gt_step_length = (
            np.linalg.norm(
                gt_step,
                axis=1,
            )
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
# B/C inference
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
        observed_motion,
    ) = model.initial_state(
        1,
        device,
        torch.float32,
    )

    active_leg_index = 0

    active_leg = route.legs[
        active_leg_index
    ]

    visual_xy = (
        active_leg.start_xy.to(
            device
        ).reshape(
            1,
            2,
        )
    )

    previous_z_uav = None

    kf = make_position_kalman(
        visual_xy[
            0
        ].cpu()
        .numpy()
    )

    terminal_locked = False
    terminal_lock_frame = None

    rows = []

    for sequence_index in range(
        len(
            cache
        )
    ):
        frame_id = int(
            cache.frame_ids[
                sequence_index
            ].item()
        )

        active_leg = route.legs[
            active_leg_index
        ]

        final_leg = (
            active_leg_index
            == len(
                route.legs
            )
            - 1
        )

        search_center = (
            visual_xy
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

        route_context = (
            route_context_tensor(
                search_center,
                active_leg,
                final_leg,
            )
        )

        image_motion_cue = (
            cache.image_motion_cues[
                sequence_index:
                sequence_index
                + 1
            ].to(
                device
            ).float()
        )

        output = model.forward_step(
            candidate.z_uav,
            previous_z_uav,
            candidate.z_sat,
            candidate.raw_logits,
            candidate.raw_prob,
            offsets_route,
            route_context,
            image_motion_cue,
            observed_motion,
            hidden,
            cell,
        )

        (
            proposal_xy,
            refined_probability,
        ) = decode_visual_proposal(
            output.refined_logits,
            candidate.centers,
        )

        if terminal_locked:
            # Mission is finished. Keep the last IMAGE-derived terminal state.
            # No fixed speed / polynomial / Kalman velocity exists.
            current_visual_xy = (
                visual_xy
            )

            effective_translation_gate = torch.zeros_like(
                output.translation_gate
            )
        else:
            current_visual_xy = (
                apply_translation_gate(
                    visual_xy,
                    proposal_xy,
                    output.translation_gate,
                )
            )

            effective_translation_gate = (
                output.translation_gate
            )

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
                output.polynomial_delta_route,
                unit,
                normal,
            )
        )

        polynomial_xy = (
            visual_xy
            + polynomial_delta_xy
        )

        new_observed_motion = (
            update_observed_motion_state(
                observed_motion,
                visual_xy,
                current_visual_xy,
                active_leg,
            )
        )

        # Position-only Kalman: predict does not advance XY.
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
            current_visual_xy[
                0
            ].cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        final_xy = np.asarray(
            kf.x,
            dtype=np.float64,
        ).reshape(
            2
        )

        raw_top1 = (
            candidate.raw_top1_xy[
                0
            ].cpu()
            .numpy()
        )

        raw_hardms = (
            candidate.hardms_xy[
                0
            ].cpu()
            .numpy()
        )

        refined_hardms, refined_support = (
            hard_mean_shift(
                output.refined_logits,
                candidate.centers,
                1.0,
                config.MEANSHIFT_BANDWIDTH_M,
                config.MEANSHIFT_ITERATIONS,
            )
        )

        refined_hardms_np = (
            refined_hardms[
                0
            ].cpu()
            .numpy()
        )

        proposal_np = (
            proposal_xy[
                0
            ].cpu()
            .numpy()
        )

        current_visual_np = (
            current_visual_xy[
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

        gt_xy = cache.gt_xy[
            sequence_index
        ].numpy()

        phase_probability = (
            output.phase_probability[
                0
            ].cpu()
            .numpy()
        )

        predicted_phase = int(
            np.argmax(
                phase_probability
            )
        )

        cue_np = (
            image_motion_cue[
                0
            ].cpu()
            .numpy()
        )

        ecc_shift_norm = float(
            math.sqrt(
                float(
                    cue_np[
                        0
                    ]
                    ** 2
                )
                + float(
                    cue_np[
                        1
                    ]
                    ** 2
                )
            )
        )

        ecc_rotation_deg = math.degrees(
            math.atan2(
                abs(
                    float(
                        cue_np[
                            2
                        ]
                    )
                ),
                max(
                    -1.0,
                    min(
                        1.0,
                        1.0
                        - float(
                            cue_np[
                                3
                            ]
                        ),
                    ),
                ),
            )
        )

        switched = False
        reached_now = False

        if (
            not terminal_locked
            and endpoint_reached(
                current_visual_xy,
                active_leg,
            )
        ):
            reached_now = True

            if final_leg:
                if bool(
                    config.TERMINAL_LOCK_ENABLED
                ):
                    terminal_locked = True
                    terminal_lock_frame = (
                        frame_id
                    )

                    new_observed_motion = torch.zeros_like(
                        new_observed_motion
                    )
            else:
                old_leg = (
                    active_leg
                )

                active_leg_index += 1

                new_leg = route.legs[
                    active_leg_index
                ]

                new_observed_motion = (
                    rotate_observed_motion_state(
                        new_observed_motion,
                        old_leg,
                        new_leg,
                    )
                )

                switched = True

        current_leg_for_log = (
            active_leg
        )

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
                "waypoint_reached_this_frame": int(
                    reached_now
                ),
                "waypoint_switched_after_frame": int(
                    switched
                ),
                "terminal_locked": int(
                    terminal_locked
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
                    raw_top1[
                        0
                    ]
                ),
                "raw_top1_y": float(
                    raw_top1[
                        1
                    ]
                ),
                "raw_hardms_x": float(
                    raw_hardms[
                        0
                    ]
                ),
                "raw_hardms_y": float(
                    raw_hardms[
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
                "proposal_x": float(
                    proposal_np[
                        0
                    ]
                ),
                "proposal_y": float(
                    proposal_np[
                        1
                    ]
                ),
                "visual_measurement_x": float(
                    current_visual_np[
                        0
                    ]
                ),
                "visual_measurement_y": float(
                    current_visual_np[
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
                "translation_probability": float(
                    phase_probability[
                        config.PHASE_TRANSLATION
                    ]
                ),
                "stationary_probability": float(
                    phase_probability[
                        config.PHASE_STATIONARY
                    ]
                ),
                "rotation_probability": float(
                    phase_probability[
                        config.PHASE_ROTATION
                    ]
                ),
                "predicted_phase": int(
                    predicted_phase
                ),
                "effective_translation_gate": float(
                    effective_translation_gate[
                        0,
                        0
                    ].cpu()
                    .item()
                ),
                "ecc_center_shift_norm": float(
                    ecc_shift_norm
                ),
                "ecc_rotation_deg": float(
                    ecc_rotation_deg
                ),
                "ecc_correlation": float(
                    cue_np[
                        4
                    ]
                ),
                "observed_delta_parallel": float(
                    new_observed_motion[
                        0,
                        0
                    ].cpu()
                    .item()
                ),
                "observed_delta_cross": float(
                    new_observed_motion[
                        0,
                        1
                    ].cpu()
                    .item()
                ),
                "observed_acc_parallel": float(
                    new_observed_motion[
                        0,
                        2
                    ].cpu()
                    .item()
                ),
                "observed_acc_cross": float(
                    new_observed_motion[
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
                "error_visual_m": float(
                    np.linalg.norm(
                        current_visual_np
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

        hidden = (
            output.hidden
        )

        cell = (
            output.cell
        )

        observed_motion = (
            new_observed_motion
        )

        previous_z_uav = (
            candidate.z_uav
        )

        visual_xy = (
            current_visual_xy
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
        "RawHardMS": metric_block(
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
        "VisualMotionGatedMeasurement": metric_block(
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
        "FinalPositionKalman": metric_block(
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
        "TerminalLockFrame": (
            None
            if terminal_lock_frame
            is None
            else int(
                terminal_lock_frame
            )
        ),
        "MeanTranslationProbability": float(
            np.mean(
                [
                    row[
                        "translation_probability"
                    ]
                    for row
                    in rows
                ]
            )
        ),
        "MeanStationaryProbability": float(
            np.mean(
                [
                    row[
                        "stationary_probability"
                    ]
                    for row
                    in rows
                ]
            )
        ),
        "MeanRotationProbability": float(
            np.mean(
                [
                    row[
                        "rotation_probability"
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
        * 100,
        flush=True,
    )

    print(
        "VISUAL-MOTION-GATED ROUTE LSTM v5",
        flush=True,
    )

    print(
        "="
        * 100,
        flush=True,
    )

    print(
        "NO fixed speed. NO nominal speed. NO learned free-running velocity.",
        flush=True,
    )

    print(
        "Current+previous UAV images explicitly classify "
        "STATIONARY / TRANSLATION / ROTATION.",
        flush=True,
    )

    print(
        "Previous v/a state is reconstructed only from previous "
        "IMAGE-derived localization differences.",
        flush=True,
    )

    print(
        "Polynomial is translation-gated and can never move the state by itself.",
        flush=True,
    )

    print(
        "Final FilterPy Kalman is position-only; it has no vx/vy state.",
        flush=True,
    )

    print(
        "="
        * 100,
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

    model = VisualMotionRouteLSTM().to(
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
            "[STAGE 3/4] train image-motion-gated recurrent model on Route A",
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
            "[STAGE 4/4] B/C closed-loop inference",
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
                    "visual_motion_lstm_frames.csv"
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
                "VisualMotionGatedMeasurement"
            ]

            final_metric = summary[
                "FinalPositionKalman"
            ]

            print(
                f"{route_name}: "
                f"Visual MLE={visual_metric['MLE_m']:.3f}m "
                f"Visual RPE={visual_metric['RPE_m']:.3f}m "
                f"| Final MLE={final_metric['MLE_m']:.3f}m "
                f"Final P90={final_metric['P90_m']:.3f}m "
                f"Final Jump={final_metric['JumpRate_pct']:.3f}% "
                f"| switches={summary['WaypointSwitchCount']} "
                f"terminal={summary['TerminalLockFrame']}",
                flush=True,
            )

        payload = {
            "architecture": (
                ARCHITECTURE_NAME
            ),
            "why_v4_failed": {
                "mean_inertia_stayed_high": True,
                "free_running_motion_state_removed": True,
                "constant_velocity_kalman_removed": True,
                "early_progress_waypoint_switch_removed": True,
            },
            "training": {
                "route": (
                    "route_A"
                ),
                "waypoint_start_end_used": True,
                "absolute_current_gt_network_input": False,
                "raw_gps_network_input": False,
                "gt_role": (
                    "supervision only"
                ),
                "image_pair_motion": (
                    "previous/current UAV images + ECC image registration cue"
                ),
                "phase_classes": [
                    "stationary",
                    "translation",
                    "rotation",
                ],
                "leg_split": (
                    split_description
                ),
            },
            "model": {
                "fixed_speed": False,
                "nominal_speed": False,
                "free_running_velocity_prediction": False,
                "observed_motion_state": [
                    "previous visual delta parallel",
                    "previous visual delta cross",
                    "observed acceleration parallel",
                    "observed acceleration cross",
                ],
                "polynomial": (
                    "observed_delta + 0.5 * observed_acceleration"
                ),
                "polynomial_role": (
                    "translation-gated score prior only"
                ),
                "current_position_source": (
                    "current image proposal gated by image-derived translation probability"
                ),
            },
            "inference": {
                "waypoint_frame_index_switching": False,
                "endpoint_switch_rule": (
                    "actual image-derived position must be within endpoint radius"
                ),
                "terminal_lock": bool(
                    config.TERMINAL_LOCK_ENABLED
                ),
                "kalman_state": (
                    "[x,y] only"
                ),
                "kalman_constant_velocity": False,
                "test_gt_used_by_inference": False,
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
