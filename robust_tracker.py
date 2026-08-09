import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import (
    RouteDataset,
    meters_from_latlon,
)
from visual_localizer import (
    CandidateBatch,
    FrozenVisualLocalizer,
    hard_mean_shift,
    train_visual_retrieval_a_only,
)
from visual_model import (
    RouteGRUMeasurementModel,
    initial_covariance_torch,
    kalman_predict_torch,
    kalman_update_torch,
)

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "RouteConditionedGRU_FilterPyKalman"


# ============================================================================
# Data structures
# ============================================================================

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
        length = max(self.length, 1e-8)
        return self.vector / length


@dataclass
class WaypointManifest:
    route_name: str
    legs: list


# ============================================================================
# Utilities
# ============================================================================

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
            f"waypoint manifest not found: {path}"
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

    for leg_index, leg in enumerate(manifest.legs):
        is_last = (
            leg_index
            == len(manifest.legs) - 1
        )

        if (
            leg.start_frame <= frame_id < leg.end_frame
            or (
                is_last
                and frame_id <= leg.end_frame
            )
        ):
            return leg

    if frame_id < manifest.legs[0].start_frame:
        return manifest.legs[0]

    return manifest.legs[-1]


def project_to_leg(xy, leg):
    xy = torch.as_tensor(
        xy,
        dtype=torch.float32,
    )
    relative = xy - leg.start_xy
    unit = leg.unit
    normal = torch.tensor(
        [-unit[1], unit[0]],
        dtype=torch.float32,
    )

    along = float(
        torch.dot(relative, unit).item()
    )
    cross = float(
        torch.dot(relative, normal).item()
    )

    return along, cross


def split_route_a_legs(manifest):
    count = len(manifest.legs)

    train_count = max(
        1,
        int(
            count
            * float(config.TEMPORAL_TRAIN_LEG_FRACTION)
        ),
    )

    val_count = max(
        1,
        int(
            count
            * float(config.TEMPORAL_VAL_LEG_FRACTION)
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
            test_legs[0].start_frame,
            test_legs[-1].end_frame,
        )
        if test_legs
        else (
            val_legs[-1].end_frame,
            val_legs[-1].end_frame,
        ),
    }


# ============================================================================
# Current-frame backbone cache
# ============================================================================
# The temporal model never receives multiple images.  This cache stores one
# public-backbone descriptor for each independent UAV frame so recurrent
# training does not repeatedly decode JPEGs / rerun MobileCLIP.
# ============================================================================

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
            for index in range(start, end)
        ]

        # Batch is only a compute optimization.
        # Each row is still an independent single-frame descriptor.
        uav = torch.stack(
            [item["uav"] for item in items]
        ).to(device)

        clip = visual.encode_uav_clip(uav)

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
            parse_frame_id(item["frame_id"])
            for item in items
        )

        if (
            start == 0
            or end == len(dataset)
            or (start // batch_size) % 20 == 0
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
        gt_xy=torch.cat(gt_rows).float(),
        uav_clip=torch.cat(clip_rows),
    )


# ============================================================================
# Forward-only route candidate search
# ============================================================================

def candidate_indices_forward(
    visual,
    predicted_xy,
    leg,
    accepted_progress_m,
):
    gallery_xy = visual.gallery["xy"]

    start = leg.start_xy.to(
        gallery_xy.device
    )
    end = leg.end_xy.to(
        gallery_xy.device
    )

    vector = end - start
    length = torch.linalg.norm(
        vector
    ).clamp_min(1e-6)
    unit = vector / length

    normal = torch.stack(
        [-unit[1], unit[0]]
    )

    relative = gallery_xy - start[None, :]

    along = (
        relative * unit[None, :]
    ).sum(dim=1)

    cross = (
        relative * normal[None, :]
    ).sum(dim=1)

    predicted_xy = predicted_xy.to(
        gallery_xy.device
    ).reshape(2)

    predicted_along = (
        (predicted_xy - start) * unit
    ).sum()

    # Mission progress is a coordinate ON the currently active straight leg.
    # It must never grow beyond that leg's endpoint while the mission
    # controller still reports the same active leg.  The previous version
    # allowed Kalman overshoot (e.g. progress > leg_length) to become the hard
    # search lower bound, which could make the legal candidate set empty.
    leg_length_m = float(length.item())

    min_along = min(
        max(
            0.0,
            float(accepted_progress_m),
        ),
        leg_length_m,
    )

    max_along = min(
        leg_length_m
        + float(config.ROUTE_ENDPOINT_PADDING_M),
        max(
            min_along + 1.0,
            float(predicted_along.item())
            + float(config.ROUTE_FORWARD_HORIZON_M),
        ),
    )

    count = int(
        config.ROUTE_CANDIDATE_COUNT
    )

    chosen_mask = None

    # Widen only sideways / forward range. Never intentionally open the
    # accepted-progress floor toward already traversed route.
    for width_scale, forward_scale in (
        (1.0, 1.0),
        (1.5, 1.5),
        (2.0, 2.0),
        (4.0, 3.0),
    ):
        current_max = min(
            float(length.item())
            + float(config.ROUTE_ENDPOINT_PADDING_M),
            max(
                min_along + 1.0,
                float(predicted_along.item())
                + float(config.ROUTE_FORWARD_HORIZON_M)
                * forward_scale,
            ),
        )

        mask = (
            (along >= min_along)
            & (along <= current_max)
            & (
                cross.abs()
                <= float(
                    config.ROUTE_CORRIDOR_HALF_WIDTH_M
                )
                * width_scale
            )
        )

        if int(mask.sum().item()) >= count:
            chosen_mask = mask
            break

    if chosen_mask is None:
        chosen_mask = (
            (along >= min_along)
            & (
                along
                <= leg_length_m
                + float(config.ROUTE_ENDPOINT_PADDING_M)
            )
        )

    valid_indices = torch.nonzero(
        chosen_mask,
        as_tuple=False,
    ).flatten()

    if valid_indices.numel() == 0:
        # Terminal-cap fallback.
        #
        # A discrete SAT-patch gallery does not necessarily contain a patch
        # center whose along-track coordinate is >= the exact geometric
        # waypoint coordinate.  When the active leg has already reached its
        # endpoint, keep the search local to the endpoint support band instead
        # of reopening the whole traversed route or crashing.
        terminal_start = max(
            0.0,
            leg_length_m
            - float(config.ROUTE_ENDPOINT_PADDING_M),
        )

        terminal_end = (
            leg_length_m
            + float(config.ROUTE_ENDPOINT_PADDING_M)
        )

        terminal_mask = (
            (along >= terminal_start)
            & (along <= terminal_end)
            & (
                cross.abs()
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
            "No satellite gallery patch exists even in the "
            "terminal endpoint support band: "
            f"leg={leg.index}, "
            f"leg_length={leg_length_m:.2f}m, "
            f"accepted_progress={float(accepted_progress_m):.2f}m"
        )

    valid_xy = gallery_xy[
        valid_indices
    ]

    distance2 = (
        valid_xy - predicted_xy[None, :]
    ).square().sum(dim=1)

    valid_cross = cross[
        valid_indices
    ]

    ranking_cost = (
        distance2
        + float(config.ROUTE_CROSS_TRACK_COST)
        * valid_cross.square()
    )

    actual_count = min(
        count,
        int(valid_indices.numel()),
    )

    order = torch.topk(
        ranking_cost,
        k=actual_count,
        largest=False,
    ).indices

    selected = valid_indices[
        order
    ]

    # Near an endpoint the legal forward set can be smaller than 36.
    # Pad by repeating the furthest legal selected patch rather than opening
    # the search backward into already traversed route.
    if selected.numel() < count:
        pad_value = selected[-1]
        padding = pad_value.repeat(
            count - selected.numel()
        )
        selected = torch.cat(
            [selected, padding],
            dim=0,
        )

    return selected.reshape(1, -1)


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

    z_uav = visual.model.encode_uav_from_clip(
        uav_clip
    )

    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(
            -1,
            satellite_clip.shape[-1],
        ),
        centers.reshape(-1, 2),
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
        / float(config.MEANSHIFT_SCORE_TAU),
        dim=1,
    )

    raw_index = raw_logits.argmax(dim=1)

    raw_top1_xy = centers[
        torch.arange(
            centers.shape[0],
            device=device,
        ),
        raw_index,
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


def gt_candidate_captured(
    candidate,
    gt_xy,
):
    distance = torch.linalg.norm(
        candidate.centers[0]
        - gt_xy.to(
            candidate.centers.device
        )[None, :],
        dim=1,
    )

    return bool(
        distance.min().item()
        <= float(
            config.CANDIDATE_CAPTURE_RADIUS_M
        )
    )


# ============================================================================
# Kalman helpers
# ============================================================================

def initial_state_tensor(
    start_xy,
    device,
):
    # Build the initial state without any in-place slice assignment.
    position = start_xy.to(
        device=device,
        dtype=torch.float32,
    ).reshape(1, 2)

    motion = torch.zeros(
        1,
        4,
        device=device,
        dtype=torch.float32,
    )

    return torch.cat(
        [
            position,
            motion,
        ],
        dim=1,
    )


def retarget_torch_state(
    state,
    covariance,
    new_leg,
):
    """
    Align the carried velocity with the newly active mission leg.

    IMPORTANT:
    This function must be fully out-of-place because state/covariance can still
    belong to the current truncated-BPTT autograd graph.  In-place writes such
    as state[:, 2:4] = ... or covariance[:, 2, 2] += ... invalidate saved views
    that backward() still needs.
    """
    position = state[:, 0:2]

    old_velocity = state[:, 2:4]

    speed = torch.linalg.norm(
        old_velocity,
        dim=1,
        keepdim=True,
    )

    unit = new_leg.unit.to(
        device=state.device,
        dtype=state.dtype,
    ).reshape(1, 2)

    aligned_velocity = (
        speed * unit
    )

    zero_acceleration = torch.zeros_like(
        state[:, 4:6]
    )

    new_state = torch.cat(
        [
            position,
            aligned_velocity,
            zero_acceleration,
        ],
        dim=1,
    )

    covariance_boost_diagonal = torch.tensor(
        [
            0.0,
            0.0,
            float(
                config.LEG_CHANGE_VELOCITY_COVARIANCE_BOOST
            ),
            float(
                config.LEG_CHANGE_VELOCITY_COVARIANCE_BOOST
            ),
            float(
                config.LEG_CHANGE_ACCELERATION_COVARIANCE_BOOST
            ),
            float(
                config.LEG_CHANGE_ACCELERATION_COVARIANCE_BOOST
            ),
        ],
        device=covariance.device,
        dtype=covariance.dtype,
    )

    covariance_boost = torch.diag(
        covariance_boost_diagonal
    ).unsqueeze(0).expand(
        covariance.shape[0],
        -1,
        -1,
    )

    new_covariance = (
        covariance
        + covariance_boost
    )

    return (
        new_state,
        new_covariance,
    )


def filterpy_transition(dt):
    dt = float(max(dt, 1.0))
    half_dt2 = 0.5 * dt * dt

    return np.array(
        [
            [1, 0, dt, 0, half_dt2, 0],
            [0, 1, 0, dt, 0, half_dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def filterpy_process_covariance(dt):
    dt = float(max(dt, 1.0))

    diagonal = np.array(
        [
            config.KALMAN_Q_POSITION,
            config.KALMAN_Q_POSITION,
            config.KALMAN_Q_VELOCITY,
            config.KALMAN_Q_VELOCITY,
            config.KALMAN_Q_ACCELERATION,
            config.KALMAN_Q_ACCELERATION,
        ],
        dtype=np.float64,
    ) * dt

    return np.diag(diagonal)


def make_filterpy_filter(start_xy):
    if KalmanFilter is None:
        raise ImportError(
            "FilterPy is required for evaluation/inference. "
            "Install it with: pip install filterpy"
        )

    kf = KalmanFilter(
        dim_x=6,
        dim_z=2,
    )

    kf.x = np.array(
        [
            float(start_xy[0]),
            float(start_xy[1]),
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    kf.P = np.diag(
        [
            config.KALMAN_INIT_POSITION_VAR,
            config.KALMAN_INIT_POSITION_VAR,
            config.KALMAN_INIT_VELOCITY_VAR,
            config.KALMAN_INIT_VELOCITY_VAR,
            config.KALMAN_INIT_ACCELERATION_VAR,
            config.KALMAN_INIT_ACCELERATION_VAR,
        ]
    ).astype(np.float64)

    return kf


def retarget_filterpy_state(
    kf,
    new_leg,
):
    speed = math.hypot(
        float(kf.x[2]),
        float(kf.x[3]),
    )

    unit = new_leg.unit.numpy()

    kf.x[2] = speed * float(unit[0])
    kf.x[3] = speed * float(unit[1])
    kf.x[4] = 0.0
    kf.x[5] = 0.0

    kf.P[2, 2] += float(
        config.LEG_CHANGE_VELOCITY_COVARIANCE_BOOST
    )
    kf.P[3, 3] += float(
        config.LEG_CHANGE_VELOCITY_COVARIANCE_BOOST
    )
    kf.P[4, 4] += float(
        config.LEG_CHANGE_ACCELERATION_COVARIANCE_BOOST
    )
    kf.P[5, 5] += float(
        config.LEG_CHANGE_ACCELERATION_COVARIANCE_BOOST
    )


# ============================================================================
# Metrics
# ============================================================================

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
            predicted_step - gt_step,
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
            + float(config.JUMP_TOLERANCE_M)
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
        "MLE_m": float(error.mean()),
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
                np.mean(error ** 2)
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


# ============================================================================
# Training
# ============================================================================

def teacher_search_ratio(epoch):
    horizon = max(
        1,
        int(config.TEACHER_SEARCH_EPOCHS),
    )

    if epoch >= horizon:
        return 0.0

    return float(
        1.0
        - epoch / float(horizon)
    )


def frame_position_lookup(cache):
    return {
        int(frame_id): index
        for index, frame_id
        in enumerate(
            cache.frame_ids.tolist()
        )
    }


def route_frame_range_indices(
    cache,
    start_frame,
    end_frame,
):
    frame_ids = cache.frame_ids.numpy()

    mask = (
        (frame_ids >= int(start_frame))
        & (frame_ids <= int(end_frame))
    )

    return np.nonzero(mask)[0].tolist()


def compute_motion_targets(
    current_gt,
    previous_gt,
    previous_velocity,
    dt,
):
    if previous_gt is None:
        velocity = torch.zeros_like(
            current_gt
        )
        acceleration = torch.zeros_like(
            current_gt
        )
        return velocity, acceleration

    dt_value = float(max(dt, 1.0))

    velocity = (
        current_gt - previous_gt
    ) / dt_value

    if previous_velocity is None:
        acceleration = torch.zeros_like(
            velocity
        )
    else:
        acceleration = (
            velocity - previous_velocity
        ) / dt_value

    return velocity, acceleration


def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    manifest,
    train_range,
    device,
    epoch,
):
    model.train()

    indices = route_frame_range_indices(
        cache,
        train_range[0],
        train_range[1],
    )

    if not indices:
        raise RuntimeError(
            "Temporal Route-A training range is empty"
        )

    first_frame = int(
        cache.frame_ids[indices[0]].item()
    )
    current_leg = active_leg_for_frame(
        manifest,
        first_frame,
    )

    state = initial_state_tensor(
        current_leg.start_xy,
        device,
    )
    covariance = initial_covariance_torch(
        1,
        device,
        torch.float32,
    )
    hidden = model.initial_hidden(
        1,
        device,
        torch.float32,
    )

    accepted_progress = 0.0
    previous_frame = first_frame - 1
    previous_gt = None
    previous_gt_velocity = None
    previous_leg_index = current_leg.index

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_accumulator = None
    chunk_steps = 0
    loss_rows = []
    capture_rows = []

    ratio = teacher_search_ratio(
        epoch
    )

    for sequence_step, index in enumerate(
        indices
    ):
        frame_id = int(
            cache.frame_ids[index].item()
        )

        gt_xy = cache.gt_xy[
            index
        ].to(
            device
        ).float()

        leg = active_leg_for_frame(
            manifest,
            frame_id,
        )

        leg_changed = (
            leg.index
            != previous_leg_index
        )

        if leg_changed:
            state, covariance = retarget_torch_state(
                state,
                covariance,
                leg,
            )
            accepted_progress = 0.0

        dt = max(
            frame_id - previous_frame,
            1,
        )

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

        predicted_xy = predicted_state[
            0,
            0:2,
        ]

        # Training-only curriculum for candidate search center.
        # By the end of the curriculum ratio=0 and search is fully closed-loop.
        search_xy = (
            ratio * gt_xy.detach()
            + (1.0 - ratio)
            * predicted_xy.detach()
        )

        indices_forward = candidate_indices_forward(
            visual,
            search_xy,
            leg,
            accepted_progress,
        )

        uav_clip = cache.uav_clip[
            index:index + 1
        ].to(
            device
        ).float()

        candidate = candidate_batch_from_indices(
            visual,
            uav_clip,
            indices_forward,
        )

        capture_rows.append(
            gt_candidate_captured(
                candidate,
                gt_xy.detach().cpu(),
            )
        )

        measurement = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_prob,
            candidate.centers,
            candidate.hardms_xy,
            candidate.hardms_support,
            predicted_state,
            leg.start_xy.to(
                device
            ).reshape(1, 2),
            leg.end_xy.to(
                device
            ).reshape(1, 2),
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
                measurement.measurement_xy,
                measurement.measurement_variance,
            )
        )

        velocity_target, acceleration_target = (
            compute_motion_targets(
                gt_xy,
                previous_gt,
                previous_gt_velocity,
                dt,
            )
        )

        final_loss = F.smooth_l1_loss(
            updated_state[:, 0:2],
            gt_xy.reshape(1, 2),
        )

        measurement_nll = F.gaussian_nll_loss(
            measurement.measurement_xy,
            gt_xy.reshape(1, 2),
            measurement.measurement_variance,
            full=False,
            reduction="mean",
        )

        prediction_loss = F.smooth_l1_loss(
            predicted_state[:, 0:2],
            gt_xy.reshape(1, 2),
        )

        velocity_loss = F.smooth_l1_loss(
            updated_state[:, 2:4],
            velocity_target.reshape(1, 2),
        )

        acceleration_loss = F.smooth_l1_loss(
            updated_state[:, 4:6],
            acceleration_target.reshape(1, 2),
        )

        loss = (
            float(config.LOSS_FINAL_SMOOTH_L1)
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
            + float(
                config.LOSS_ACCELERATION_SMOOTH_L1
            )
            * acceleration_loss
        )

        if loss_accumulator is None:
            loss_accumulator = loss
        else:
            loss_accumulator = (
                loss_accumulator + loss
            )

        chunk_steps += 1

        loss_rows.append(
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
                "acceleration": float(
                    acceleration_loss.detach().cpu()
                ),
            }
        )

        state = updated_state
        covariance = updated_covariance
        hidden = measurement.hidden

        along, _ = project_to_leg(
            state[
                0,
                0:2,
            ].detach().cpu(),
            leg,
        )
        accepted_progress = min(
            max(
                accepted_progress,
                along,
                0.0,
            ),
            float(leg.length),
        )

        previous_frame = frame_id
        previous_gt = gt_xy.detach()
        previous_gt_velocity = (
            velocity_target.detach()
        )
        previous_leg_index = leg.index

        is_chunk_end = (
            chunk_steps
            >= int(config.TBPTT_STEPS)
            or sequence_step
            == len(indices) - 1
        )

        if is_chunk_end:
            normalized = (
                loss_accumulator
                / float(chunk_steps)
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
                float(config.GRAD_CLIP_NORM),
            )

            optimizer.step()
            optimizer.zero_grad(
                set_to_none=True
            )

            # Truncated BPTT: state values persist, graph does not.
            state = state.detach()
            covariance = covariance.detach()
            hidden = hidden.detach()

            loss_accumulator = None
            chunk_steps = 0

    means = {
        key: float(
            np.mean(
                [row[key] for row in loss_rows]
            )
        )
        for key in loss_rows[0]
    }

    means["capture_pct"] = (
        float(
            np.mean(capture_rows)
            * 100.0
        )
        if capture_rows
        else 0.0
    )
    means["teacher_search_ratio"] = ratio

    return means


# ============================================================================
# FilterPy evaluation / inference
# ============================================================================

@torch.no_grad()
def evaluate_segment_filterpy(
    model,
    visual,
    cache,
    manifest,
    device,
    start_frame,
    end_frame,
    save_csv_path=None,
):
    if KalmanFilter is None:
        raise ImportError(
            "FilterPy is required. "
            "Run: pip install filterpy"
        )

    model.eval()

    indices = route_frame_range_indices(
        cache,
        start_frame,
        end_frame,
    )

    if not indices:
        raise RuntimeError(
            f"empty evaluation segment "
            f"{start_frame}..{end_frame}"
        )

    first_frame_id = int(
        cache.frame_ids[
            indices[0]
        ].item()
    )

    current_leg = active_leg_for_frame(
        manifest,
        first_frame_id,
    )

    kf = make_filterpy_filter(
        current_leg.start_xy.numpy()
    )

    hidden = model.initial_hidden(
        1,
        device,
        torch.float32,
    )

    accepted_progress = 0.0
    previous_frame = first_frame_id - 1
    previous_leg_index = current_leg.index

    rows = []

    for index in indices:
        frame_id = int(
            cache.frame_ids[
                index
            ].item()
        )

        gt_xy = cache.gt_xy[
            index
        ]

        leg = active_leg_for_frame(
            manifest,
            frame_id,
        )

        leg_changed = (
            leg.index
            != previous_leg_index
        )

        if leg_changed:
            retarget_filterpy_state(
                kf,
                leg,
            )
            accepted_progress = 0.0

        dt = max(
            frame_id - previous_frame,
            1,
        )

        # Standard FilterPy predict step.
        kf.F = filterpy_transition(
            dt
        )
        kf.Q = filterpy_process_covariance(
            dt
        )
        kf.predict()

        predicted_state_np = np.asarray(
            kf.x,
            dtype=np.float64,
        ).reshape(6)

        predicted_state = torch.tensor(
            predicted_state_np,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 6)

        predicted_xy = torch.tensor(
            predicted_state_np[0:2],
            dtype=torch.float32,
        )

        candidate_indices = candidate_indices_forward(
            visual,
            predicted_xy,
            leg,
            accepted_progress,
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

        measurement = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_prob,
            candidate.centers,
            candidate.hardms_xy,
            candidate.hardms_support,
            predicted_state,
            leg.start_xy.to(
                device
            ).reshape(1, 2),
            leg.end_xy.to(
                device
            ).reshape(1, 2),
            torch.tensor(
                [leg_changed],
                device=device,
            ),
            hidden,
        )

        hidden = measurement.hidden

        measurement_xy = (
            measurement.measurement_xy[
                0
            ].cpu().numpy()
        )

        measurement_variance = (
            measurement.measurement_variance[
                0
            ].cpu().numpy()
        )

        # Standard FilterPy update step.
        kf.R = np.diag(
            measurement_variance.astype(
                np.float64
            )
        )

        kf.update(
            measurement_xy.astype(
                np.float64
            )
        )

        final_state = np.asarray(
            kf.x,
            dtype=np.float64,
        ).reshape(6)

        final_xy = final_state[
            0:2
        ]

        along, cross = project_to_leg(
            torch.tensor(
                final_xy,
                dtype=torch.float32,
            ),
            leg,
        )

        accepted_progress = min(
            max(
                accepted_progress,
                along,
                0.0,
            ),
            float(leg.length),
        )

        capture = gt_candidate_captured(
            candidate,
            gt_xy,
        )

        rows.append(
            {
                "frame_id": frame_id,
                "leg_index": leg.index,
                "leg_changed": int(
                    leg_changed
                ),
                "gt_x": float(gt_xy[0]),
                "gt_y": float(gt_xy[1]),
                "prediction_x": float(
                    predicted_state_np[0]
                ),
                "prediction_y": float(
                    predicted_state_np[1]
                ),
                "raw_top1_x": float(
                    candidate.raw_top1_xy[
                        0,
                        0,
                    ].cpu()
                ),
                "raw_top1_y": float(
                    candidate.raw_top1_xy[
                        0,
                        1,
                    ].cpu()
                ),
                "hardms_x": float(
                    candidate.hardms_xy[
                        0,
                        0,
                    ].cpu()
                ),
                "hardms_y": float(
                    candidate.hardms_xy[
                        0,
                        1,
                    ].cpu()
                ),
                "measurement_x": float(
                    measurement_xy[0]
                ),
                "measurement_y": float(
                    measurement_xy[1]
                ),
                "measurement_var_x": float(
                    measurement_variance[0]
                ),
                "measurement_var_y": float(
                    measurement_variance[1]
                ),
                "final_x": float(
                    final_xy[0]
                ),
                "final_y": float(
                    final_xy[1]
                ),
                "vx": float(
                    final_state[2]
                ),
                "vy": float(
                    final_state[3]
                ),
                "ax": float(
                    final_state[4]
                ),
                "ay": float(
                    final_state[5]
                ),
                "accepted_progress_m": float(
                    accepted_progress
                ),
                "cross_track_m": float(
                    cross
                ),
                "candidate_capture": int(
                    capture
                ),
            }
        )

        previous_frame = frame_id
        previous_leg_index = leg.index

    gt = [
        [row["gt_x"], row["gt_y"]]
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
        "RNNMeasurement": metric_block(
            [
                [
                    row["measurement_x"],
                    row["measurement_y"],
                ]
                for row in rows
            ],
            gt,
        ),
        "FilterPyKalmanFinal": metric_block(
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
                    row["candidate_capture"]
                    for row in rows
                ]
            )
            * 100.0
        ),
        "MeanMeasurementVariance": float(
            np.mean(
                [
                    0.5
                    * (
                        row["measurement_var_x"]
                        + row["measurement_var_y"]
                    )
                    for row in rows
                ]
            )
        ),
    }

    if save_csv_path is not None:
        save_csv_path = Path(
            save_csv_path
        )
        save_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with save_csv_path.open(
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
            writer.writerows(rows)

    return summary, rows


def validation_score(summary):
    metric = summary[
        "FilterPyKalmanFinal"
    ]

    return (
        metric["MLE_m"]
        + float(config.VAL_RPE_WEIGHT)
        * metric["RPE_m"]
        + float(config.VAL_JUMP_WEIGHT)
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

    print(
        "Route A temporal leg split:",
        split,
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LR),
        weight_decay=float(
            config.WEIGHT_DECAY
        ),
    )

    best_score = float("inf")
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
            device,
            epoch,
        )

        validation, _ = evaluate_segment_filterpy(
            model,
            visual,
            cache,
            manifest,
            device,
            split["val"][0],
            split["val"][1],
            save_csv_path=None,
        )

        score = validation_score(
            validation
        )

        improved = (
            score < best_score
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
                "uses_mission_waypoints": True,
                "waypoint_source_note": (
                    "Current route_waypoints are GPS-derived "
                    "mission-waypoint proxies."
                ),
                "filterpy_used_for_validation_and_inference": True,
                "state_definition": [
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "ax",
                    "ay",
                ],
            },
            config.TEMPORAL_CHECKPOINT,
        )

        final_metric = validation[
            "FilterPyKalmanFinal"
        ]

        print(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"loss={training['total']:.5f} "
            f"final={training['final']:.4f} "
            f"nll={training['nll']:.4f} "
            f"pred={training['prediction']:.4f} "
            f"vel={training['velocity']:.4f} "
            f"acc={training['acceleration']:.4f} "
            f"capture={training['capture_pct']:.2f}% "
            f"teacher={training['teacher_search_ratio']:.3f} "
            f"val_mle={final_metric['MLE_m']:.3f}m "
            f"val_rpe={final_metric['RPE_m']:.3f}m "
            f"val_jump={final_metric['JumpRate_pct']:.3f}% "
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
            "Temporal training did not produce a checkpoint"
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

    return split


def load_temporal_checkpoint(
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

    if checkpoint.get(
        "architecture"
    ) != ARCHITECTURE_NAME:
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch"
        )

    state = (
        checkpoint.get("best_model")
        or checkpoint["model"]
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    return checkpoint


# ============================================================================
# Main
# ============================================================================

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
            "Reuse the already-trained visual_retrieval_A_only.pt and "
            "restart only the GRU/Kalman temporal training from scratch."
        ),
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

    print("=" * 88)
    print("ROUTE-CONDITIONED SINGLE-FRAME GRU + FILTERPY KALMAN")
    print("=" * 88)
    print("Temporal input per step : ONE current UAV retrieval result")
    print("Persistent RNN state    : GRU h_t")
    print("Physical state          : [x,y,vx,vy,ax,ay]")
    print("Motion prediction       : constant-acceleration polynomial")
    print("Search                  : only forward current-leg SAT corridor")
    print("Mission prior           : known current leg start/end waypoint")
    print("Final inference filter  : FilterPy KalmanFilter")
    print("Visual training         : Route A from scratch")
    print("Temporal training       : Route A from scratch")
    print("Final testing           : Route B + Route C")
    print("=" * 88)

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Training always restarts the NEW temporal model from scratch.
    # Visual retrieval can either be fully retrained (default) or reused after
    # a temporal-stage crash with --reuse-visual.
    if args.mode in (
        "train",
        "train_eval",
    ):
        if config.TEMPORAL_CHECKPOINT.exists():
            config.TEMPORAL_CHECKPOINT.unlink()

        if args.reuse_visual:
            if not config.VISUAL_CHECKPOINT.exists():
                raise FileNotFoundError(
                    "--reuse-visual was requested but the visual checkpoint "
                    f"does not exist: {config.VISUAL_CHECKPOINT}"
                )

            print(
                "reuse visual checkpoint; restart GRU/Kalman temporal "
                "training from scratch:",
                config.VISUAL_CHECKPOINT,
                flush=True,
            )
        else:
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

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Visual checkpoint missing. "
            "Run --mode train_eval for full retraining."
        )

    visual = FrozenVisualLocalizer(
        device
    )

    model = RouteGRUMeasurementModel().to(
        device
    )

    catalog = route_catalog()

    route_a_manifest = load_waypoint_manifest(
        "route_A",
        visual.origin_lat,
        visual.origin_lon,
    )

    if args.mode in (
        "train",
        "train_eval",
    ):
        route_a_cache = build_backbone_cache(
            "route_A",
            catalog["route_A"],
            visual,
            device,
        )

        temporal_split = train_temporal(
            model,
            visual,
            route_a_cache,
            route_a_manifest,
            device,
            int(args.epochs),
        )
    else:
        load_temporal_checkpoint(
            model,
            device,
        )
        temporal_split = split_route_a_legs(
            route_a_manifest
        )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        # If train_model returned the best state, it is already loaded.
        if args.mode == "eval":
            load_temporal_checkpoint(
                model,
                device,
            )

        route_results = {}

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

            start_frame = manifest.legs[
                0
            ].start_frame

            end_frame = manifest.legs[
                -1
            ].end_frame

            csv_path = (
                config.OUTPUT_DIR
                / f"{route_name}_route_rnn_filterpy_frames.csv"
            )

            summary, _ = evaluate_segment_filterpy(
                model,
                visual,
                cache,
                manifest,
                device,
                start_frame,
                end_frame,
                save_csv_path=csv_path,
            )

            route_results[
                route_name
            ] = summary

            metric = summary[
                "FilterPyKalmanFinal"
            ]

            print(
                f"{route_name}: "
                f"MLE={metric['MLE_m']:.3f}m "
                f"P90={metric['P90_m']:.3f}m "
                f"RPE={metric['RPE_m']:.3f}m "
                f"Jump={metric['JumpRate_pct']:.3f}% "
                f"capture={summary['CandidateCaptureRate_pct']:.2f}%",
                flush=True,
            )

        result = {
            "architecture": ARCHITECTURE_NAME,
            "protocol": {
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
                "single_frame_streaming": True,
                "mission_waypoint_prior": True,
                "waypoint_switch_emulation": (
                    "Offline evaluation uses the annotated "
                    "straight-leg frame boundaries as a proxy for "
                    "the mission controller's active waypoint index."
                ),
                "important_waypoint_note": (
                    "The current route_waypoints manifests were "
                    "geometrically derived from GPS telemetry. "
                    "They are not asserted PX4 mission-item coordinates."
                ),
                "filter": (
                    "filterpy.kalman.KalmanFilter"
                ),
                "candidate_search": (
                    "forward-only current mission leg"
                ),
            },
            "state": {
                "gru_hidden_dim": int(
                    config.RNN_HIDDEN_DIM
                ),
                "kalman_state": [
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "ax",
                    "ay",
                ],
            },
            "route_A_temporal_split": temporal_split,
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
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        print(
            f"summary saved: {summary_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
