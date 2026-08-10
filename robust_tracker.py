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
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import RouteBoundedHypothesisLSTM

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "RouteBoundedHypothesisLSTM_v6"


@dataclass
class RouteCache:
    route_name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    uav_clip: torch.Tensor
    image_paths: list

    def __len__(self):
        return int(self.gt_xy.shape[0])


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
        return int(self.start.frame_index)

    @property
    def end_frame(self):
        return int(self.end.frame_index)

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


@dataclass
class FineCandidate:
    centers: torch.Tensor
    z_sat: torch.Tensor
    raw_logits: torch.Tensor
    valid_mask: torch.Tensor


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


# =============================================================================
# Route / data
# =============================================================================

def load_mission_route(route_name, origin_lat, origin_lon):
    path = Path(config.WAYPOINT_FILES[route_name])
    payload = json.loads(path.read_text(encoding="utf-8"))

    raw = sorted(
        payload["waypoints"],
        key=lambda item: int(item["waypoint_order"]),
    )

    waypoints = []

    for item in raw:
        x_m, y_m = meters_from_latlon(
            item["latitude"],
            item["longitude"],
            origin_lat,
            origin_lon,
        )

        waypoints.append(
            MissionWaypoint(
                order=int(item["waypoint_order"]),
                frame_index=int(item.get("frame_index", -1)),
                xy=torch.tensor(
                    [x_m, y_m],
                    dtype=torch.float32,
                ),
            )
        )

    if len(waypoints) < 2:
        raise RuntimeError(f"{route_name}: fewer than two waypoints")

    legs = [
        MissionLeg(
            index=i,
            start=waypoints[i],
            end=waypoints[i + 1],
        )
        for i in range(len(waypoints) - 1)
    ]

    print(
        f"{route_name}: {len(waypoints)} waypoints -> {len(legs)} legs",
        flush=True,
    )

    print(
        "  "
        + " -> ".join(
            f"W{wp.order}[f{wp.frame_index}]"
            for wp in waypoints
        ),
        flush=True,
    )

    return MissionRoute(
        route_name=route_name,
        waypoints=waypoints,
        legs=legs,
    )


@torch.no_grad()
def build_route_cache(route_name, root, visual, device):
    stat = config.VISUAL_CHECKPOINT.stat()

    signature = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

    cache_path = (
        config.OUTPUT_DIR
        / "feature_cache"
        / f"{route_name}_uav_clip.pt"
    )

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")

        if payload.get("signature") == signature:
            print(f"{route_name}: reuse UAV backbone cache", flush=True)

            return RouteCache(
                route_name=route_name,
                frame_ids=payload["frame_ids"],
                gt_xy=payload["gt_xy"],
                uav_clip=payload["uav_clip"],
                image_paths=payload["image_paths"],
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

    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))

        items = [dataset[i] for i in range(start, end)]

        uav = torch.stack(
            [item["uav"] for item in items]
        ).to(device)

        clip = visual.encode_uav_clip(uav)

        clip_rows.append(
            clip.detach().cpu().to(cache_dtype())
        )

        gt_rows.append(
            torch.stack(
                [item["xy"].float() for item in items]
            )
        )

        for item in items:
            frame_rows.append(
                parse_frame_id(item["frame_id"])
            )
            image_paths.append(str(item["image_path"]))

        if (
            start == 0
            or end == len(dataset)
            or (start // batch_size) % 10 == 0
        ):
            print(
                f"{route_name} backbone cache: {end}/{len(dataset)}",
                flush=True,
            )

    result = RouteCache(
        route_name=route_name,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        gt_xy=torch.cat(gt_rows).float(),
        uav_clip=torch.cat(clip_rows),
        image_paths=image_paths,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "signature": signature,
            "frame_ids": result.frame_ids,
            "gt_xy": result.gt_xy,
            "uav_clip": result.uav_clip,
            "image_paths": result.image_paths,
        },
        cache_path,
    )

    return result


@torch.no_grad()
def build_satellite_task_embeddings(visual):
    clip_feat = visual.gallery["clip_feat"]
    xy = visual.gallery["xy"]

    rows = []
    batch_size = 4096

    for start in range(0, int(clip_feat.shape[0]), batch_size):
        end = min(
            start + batch_size,
            int(clip_feat.shape[0]),
        )

        rows.append(
            visual.model.encode_sat_from_clip(
                clip_feat[start:end].float(),
                xy[start:end].float(),
            )
        )

    result = torch.cat(rows, dim=0)

    print(
        "satellite task embeddings:",
        tuple(result.shape),
        flush=True,
    )

    return result


# =============================================================================
# Route geometry
# =============================================================================

def leg_geometry(leg, device=None, dtype=torch.float32):
    start = leg.start_xy
    end = leg.end_xy

    if device is not None:
        start = start.to(device=device, dtype=dtype)
        end = end.to(device=device, dtype=dtype)

    vector = end - start
    length = torch.linalg.norm(vector).clamp_min(1e-6)
    unit = vector / length
    normal = torch.stack([-unit[1], unit[0]])

    return start, end, unit, normal, length


def xy_vector_to_leg_frame(vector_xy, unit, normal):
    parallel = (vector_xy * unit).sum(dim=-1)
    cross = (vector_xy * normal).sum(dim=-1)
    return torch.stack([parallel, cross], dim=-1)


def leg_frame_to_xy(vector_route, unit, normal):
    return (
        vector_route[..., 0:1] * unit
        + vector_route[..., 1:2] * normal
    )


def route_valid_mask(xy, leg):
    start, _, unit, normal, length = leg_geometry(
        leg,
        device=xy.device,
        dtype=xy.dtype,
    )

    if xy.ndim == 2:
        start_view = start.reshape(1, 2)
    elif xy.ndim == 3:
        start_view = start.reshape(1, 1, 2)
    else:
        raise ValueError(f"Unexpected XY shape: {tuple(xy.shape)}")

    relative = xy - start_view

    along = (relative * unit).sum(dim=-1)
    cross = (relative * normal).sum(dim=-1).abs()

    padding = float(config.ROUTE_ALONG_PADDING_M)

    return (
        (along >= -padding)
        & (along <= length + padding)
        & (
            cross
            <= float(config.ROUTE_CORRIDOR_HALF_WIDTH_M)
        )
    )


def route_bank_indices(visual, leg):
    mask = route_valid_mask(
        visual.gallery["xy"],
        leg,
    )

    indices = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten()

    if indices.numel() < 4:
        raise RuntimeError(
            f"Too few SAT patches in W{leg.start.order}->W{leg.end.order}: "
            f"{indices.numel()}"
        )

    return indices


def build_route_banks(visual, route):
    banks = {}

    for leg in route.legs:
        bank = route_bank_indices(visual, leg)
        banks[leg.index] = bank

        print(
            f"{route.route_name} W{leg.start.order}->W{leg.end.order}: "
            f"{bank.numel()} route SAT patches",
            flush=True,
        )

    return banks


def candidate_offsets_route(
    centers,
    reference_xy,
    leg,
):
    _, _, unit, normal, _ = leg_geometry(
        leg,
        device=centers.device,
        dtype=centers.dtype,
    )

    relative = (
        centers
        - reference_xy[:, None, :]
    )

    return xy_vector_to_leg_frame(
        relative,
        unit.reshape(1, 1, 2),
        normal.reshape(1, 1, 2),
    )


def route_context(reference_xy, leg, final_leg):
    start, _, unit, normal, length = leg_geometry(
        leg,
        device=reference_xy.device,
        dtype=reference_xy.dtype,
    )

    relative = reference_xy - start.reshape(1, 2)

    along = (
        relative
        * unit.reshape(1, 2)
    ).sum(dim=1)

    cross = (
        relative
        * normal.reshape(1, 2)
    ).sum(dim=1)

    remaining_ratio = (
        (length - along) / length
    ).clamp(-0.15, 1.25)

    normalized_cross = (
        cross
        / float(config.ROUTE_CROSS_TRACK_SCALE_M)
    ).clamp(-2.0, 2.0)

    normalized_length = (
        torch.log1p(length)
        / math.log1p(
            float(config.ROUTE_LENGTH_LOG_SCALE_M)
        )
    ).reshape(1).expand_as(remaining_ratio)

    final_flag = torch.full_like(
        remaining_ratio,
        1.0 if final_leg else 0.0,
    )

    return torch.stack(
        [
            remaining_ratio,
            normalized_cross,
            normalized_length,
            final_flag,
        ],
        dim=1,
    )


def update_observed_motion(
    previous_state,
    previous_xy,
    current_xy,
    leg,
):
    _, _, unit, normal, _ = leg_geometry(
        leg,
        device=current_xy.device,
        dtype=current_xy.dtype,
    )

    delta = xy_vector_to_leg_frame(
        current_xy - previous_xy,
        unit.reshape(1, 2),
        normal.reshape(1, 2),
    )

    acceleration = (
        delta
        - previous_state[:, 0:2]
    )

    return torch.cat(
        [delta, acceleration],
        dim=1,
    )


def rotate_observed_motion(state, old_leg, new_leg):
    device = state.device
    dtype = state.dtype

    _, _, old_unit, old_normal, _ = leg_geometry(
        old_leg,
        device=device,
        dtype=dtype,
    )

    _, _, new_unit, new_normal, _ = leg_geometry(
        new_leg,
        device=device,
        dtype=dtype,
    )

    delta_xy = leg_frame_to_xy(
        state[:, 0:2],
        old_unit.reshape(1, 2),
        old_normal.reshape(1, 2),
    )

    acceleration_xy = leg_frame_to_xy(
        state[:, 2:4],
        old_unit.reshape(1, 2),
        old_normal.reshape(1, 2),
    )

    new_delta = xy_vector_to_leg_frame(
        delta_xy,
        new_unit.reshape(1, 2),
        new_normal.reshape(1, 2),
    )

    new_acceleration = xy_vector_to_leg_frame(
        acceleration_xy,
        new_unit.reshape(1, 2),
        new_normal.reshape(1, 2),
    )

    return torch.cat(
        [new_delta, new_acceleration],
        dim=1,
    )


# =============================================================================
# Visual candidate construction
# =============================================================================

def global_visual_stats(logits):
    probability = torch.softmax(logits, dim=1)
    count = max(int(logits.shape[1]), 2)

    entropy = -(
        probability
        * probability.clamp_min(1e-8).log()
    ).sum(dim=1) / math.log(float(count))

    top2 = probability.topk(
        k=2,
        dim=1,
    ).values

    margin = top2[:, 0] - top2[:, 1]
    maximum = top2[:, 0]

    std = torch.tanh(
        logits.std(
            dim=1,
            unbiased=False,
        )
        / 10.0
    )

    return torch.stack(
        [entropy, margin, maximum, std],
        dim=1,
    )


@torch.no_grad()
def score_global_route(
    visual,
    z_uav,
    gallery_z_sat,
    bank_indices,
):
    bank_z_sat = gallery_z_sat[
        bank_indices
    ]

    logits = (
        visual.model.logit_scale.exp()
        .clamp(max=100.0)
        * (
            z_uav
            @ bank_z_sat.t()
        )
    )

    top_local = logits.argmax(dim=1)
    top_gallery = bank_indices[top_local]
    top_xy = visual.gallery["xy"][
        top_gallery
    ]

    return logits, top_xy


@torch.no_grad()
def build_bounded_candidate(
    visual,
    uav_clip,
    center_xy,
    leg,
    bank_indices,
):
    candidate = visual.candidate_batch(
        uav_clip,
        center_xy,
        grid_size=config.GRID_SIZE,
    )

    valid = route_valid_mask(
        candidate.centers,
        leg,
    )

    if not bool(
        (valid.sum(dim=1) > 0).all()
    ):
        legal_xy = visual.gallery["xy"][
            bank_indices
        ]

        fallback_rows = []

        for batch_index in range(
            center_xy.shape[0]
        ):
            distance = torch.linalg.norm(
                legal_xy
                - center_xy[
                    batch_index:
                    batch_index + 1
                ],
                dim=1,
            )

            fallback_rows.append(
                legal_xy[
                    distance.argmin()
                ]
            )

        fallback_center = torch.stack(
            fallback_rows,
            dim=0,
        )

        candidate = visual.candidate_batch(
            uav_clip,
            fallback_center,
            grid_size=config.GRID_SIZE,
        )

        valid = route_valid_mask(
            candidate.centers,
            leg,
        )

    if not bool(
        (valid.sum(dim=1) > 0).all()
    ):
        raise RuntimeError(
            "No legal bounded candidate for "
            f"W{leg.start.order}->W{leg.end.order}"
        )

    return FineCandidate(
        centers=candidate.centers,
        z_sat=candidate.z_sat,
        raw_logits=candidate.raw_logits,
        valid_mask=valid,
    )


@torch.no_grad()
def build_waypoint_candidate(
    visual,
    uav_clip,
    endpoint_xy,
):
    """
    Small visual transition neighborhood around the active waypoint.

    This is intentionally not masked to only the old leg because during an
    in-place turn the image can already face the next leg. Spatially it is still
    only one 6x6 lattice around the shared endpoint.
    """
    candidate = visual.candidate_batch(
        uav_clip,
        endpoint_xy,
        grid_size=config.GRID_SIZE,
    )

    valid = torch.ones_like(
        candidate.raw_logits,
        dtype=torch.bool,
    )

    return FineCandidate(
        centers=candidate.centers,
        z_sat=candidate.z_sat,
        raw_logits=candidate.raw_logits,
        valid_mask=valid,
    )


def decode_refined(
    logits,
    centers,
    valid_mask,
):
    masked = torch.where(
        valid_mask,
        logits,
        torch.full_like(
            logits,
            -1e4,
        ),
    )

    probability = torch.softmax(
        masked,
        dim=1,
    )

    xy = (
        probability.unsqueeze(-1)
        * centers
    ).sum(dim=1)

    return xy, probability


def compose_hypotheses(
    previous_xy,
    local_xy,
    recovery_xy,
    waypoint_xy,
    branch_probability,
):
    hypotheses = torch.stack(
        [
            previous_xy,
            local_xy,
            recovery_xy,
            waypoint_xy,
        ],
        dim=1,
    )

    current_xy = (
        branch_probability.unsqueeze(-1)
        * hypotheses
    ).sum(dim=1)

    return current_xy, hypotheses


# =============================================================================
# Route-A supervision
# =============================================================================

def training_leg_for_frame(route, frame_id):
    # Exact waypoint frame belongs to the arriving leg.
    for leg in route.legs:
        if (
            frame_id >= leg.start_frame
            and frame_id <= leg.end_frame
        ):
            return leg

    return route.legs[-1]


def teacher_ratio(epoch_index):
    end_epoch = int(
        config.TEACHER_CENTER_END_EPOCH
    )

    if (
        end_epoch <= 0
        or epoch_index >= end_epoch
    ):
        return 0.0

    return max(
        0.0,
        1.0
        - float(epoch_index)
        / float(end_epoch),
    )


def masked_candidate_ce(
    logits,
    centers,
    valid_mask,
    gt_xy,
):
    distance = torch.linalg.norm(
        centers
        - gt_xy[:, None, :],
        dim=2,
    )

    distance = torch.where(
        valid_mask,
        distance,
        torch.full_like(
            distance,
            1e9,
        ),
    )

    minimum_distance, target = distance.min(
        dim=1
    )

    capture = (
        minimum_distance
        <= float(
            config.CANDIDATE_CAPTURE_RADIUS_M
        )
    )

    if bool(capture.any()):
        loss = F.cross_entropy(
            logits[capture],
            target[capture],
        )
    else:
        loss = logits.sum() * 0.0

    return loss, capture.float().mean()


def branch_distribution_target(
    hypotheses,
    gt_xy,
):
    error = torch.linalg.norm(
        hypotheses.detach()
        - gt_xy[:, None, :],
        dim=2,
    )

    return torch.softmax(
        -error
        / float(
            config.BRANCH_TARGET_TAU_M
        ),
        dim=1,
    )


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model,
    optimizer,
    visual,
    gallery_z_sat,
    cache,
    route,
    banks,
    device,
    epoch_index,
):
    model.train()

    hidden, cell = model.initial_state(
        1,
        device,
        torch.float32,
    )

    observed_motion = torch.zeros(
        1,
        int(config.OBSERVED_MOTION_DIM),
        device=device,
        dtype=torch.float32,
    )

    previous_visual_xy = (
        route.legs[0].start_xy.to(
            device
        ).reshape(1, 2)
    )

    previous_gt = (
        previous_visual_xy.detach()
    )

    previous_z_uav = None
    previous_leg = route.legs[0]

    optimizer.zero_grad(set_to_none=True)

    accumulated_loss = None
    accumulated_steps = 0

    ratio = teacher_ratio(epoch_index)

    logs = []

    for sequence_index in range(len(cache)):
        frame_id = int(
            cache.frame_ids[
                sequence_index
            ].item()
        )

        leg = training_leg_for_frame(
            route,
            frame_id,
        )

        if leg.index != previous_leg.index:
            observed_motion = rotate_observed_motion(
                observed_motion,
                previous_leg,
                leg,
            )
            previous_leg = leg

        final_leg = (
            leg.index
            == len(route.legs) - 1
        )

        gt_xy = cache.gt_xy[
            sequence_index:
            sequence_index + 1
        ].to(device).float()

        local_center = (
            float(ratio)
            * previous_gt
            + (1.0 - float(ratio))
            * previous_visual_xy
        )

        uav_clip = cache.uav_clip[
            sequence_index:
            sequence_index + 1
        ].to(device).float()

        z_uav = visual.model.encode_uav_from_clip(
            uav_clip
        )

        bank_indices = banks[
            leg.index
        ]

        global_logits, global_top_xy = (
            score_global_route(
                visual,
                z_uav,
                gallery_z_sat,
                bank_indices,
            )
        )

        local_candidate = build_bounded_candidate(
            visual,
            uav_clip,
            local_center,
            leg,
            bank_indices,
        )

        recovery_candidate = (
            build_bounded_candidate(
                visual,
                uav_clip,
                global_top_xy,
                leg,
                bank_indices,
            )
        )

        endpoint_center = (
            leg.end_xy.to(
                device
            ).reshape(1, 2)
        )

        waypoint_candidate = (
            build_waypoint_candidate(
                visual,
                uav_clip,
                endpoint_center,
            )
        )

        local_offsets = candidate_offsets_route(
            local_candidate.centers,
            previous_visual_xy,
            leg,
        )

        recovery_offsets = candidate_offsets_route(
            recovery_candidate.centers,
            previous_visual_xy,
            leg,
        )

        waypoint_offsets = candidate_offsets_route(
            waypoint_candidate.centers,
            previous_visual_xy,
            leg,
        )

        context = route_context(
            previous_visual_xy,
            leg,
            final_leg,
        )

        polynomial_delta = (
            observed_motion[:, 0:2]
            + 0.5
            * observed_motion[:, 2:4]
        )

        output = model.forward_step(
            z_uav=z_uav,
            previous_z_uav=previous_z_uav,
            local_z_sat=local_candidate.z_sat,
            local_raw_logits=local_candidate.raw_logits,
            local_valid_mask=local_candidate.valid_mask,
            local_offsets_route=local_offsets,
            recovery_z_sat=recovery_candidate.z_sat,
            recovery_raw_logits=recovery_candidate.raw_logits,
            recovery_valid_mask=recovery_candidate.valid_mask,
            recovery_offsets_route=recovery_offsets,
            waypoint_z_sat=waypoint_candidate.z_sat,
            waypoint_raw_logits=waypoint_candidate.raw_logits,
            waypoint_valid_mask=waypoint_candidate.valid_mask,
            waypoint_offsets_route=waypoint_offsets,
            global_stats=global_visual_stats(
                global_logits
            ),
            route_context=context,
            observed_motion=observed_motion,
            polynomial_delta_route=polynomial_delta,
            hidden=hidden,
            cell=cell,
        )

        local_xy, _ = decode_refined(
            output.local_refined_logits,
            local_candidate.centers,
            local_candidate.valid_mask,
        )

        recovery_xy, _ = decode_refined(
            output.recovery_refined_logits,
            recovery_candidate.centers,
            recovery_candidate.valid_mask,
        )

        waypoint_xy, _ = decode_refined(
            output.waypoint_refined_logits,
            waypoint_candidate.centers,
            waypoint_candidate.valid_mask,
        )

        current_visual_xy, hypotheses = (
            compose_hypotheses(
                previous_visual_xy,
                local_xy,
                recovery_xy,
                waypoint_xy,
                output.branch_probability,
            )
        )

        position_loss = F.smooth_l1_loss(
            current_visual_xy,
            gt_xy,
        )

        branch_target = branch_distribution_target(
            hypotheses,
            gt_xy,
        )

        # Standard class-balancing emphasis for the rare waypoint-transition
        # hypothesis, still fully supervised by Route-A only.
        branch_weight = (
            1.0
            + 2.0
            * branch_target[
                :,
                config.HYPOTHESIS_WAYPOINT
            ]
        )

        branch_loss_per_frame = -(
            branch_target
            * output.branch_probability.clamp_min(
                1e-8
            ).log()
        ).sum(dim=1)

        branch_loss = (
            branch_weight
            * branch_loss_per_frame
        ).mean()

        local_ce, local_capture = (
            masked_candidate_ce(
                output.local_refined_logits,
                local_candidate.centers,
                local_candidate.valid_mask,
                gt_xy,
            )
        )

        recovery_ce, recovery_capture = (
            masked_candidate_ce(
                output.recovery_refined_logits,
                recovery_candidate.centers,
                recovery_candidate.valid_mask,
                gt_xy,
            )
        )

        waypoint_ce, waypoint_capture = (
            masked_candidate_ce(
                output.waypoint_refined_logits,
                waypoint_candidate.centers,
                waypoint_candidate.valid_mask,
                gt_xy,
            )
        )

        target_step = (
            gt_xy
            - previous_gt
        )

        predicted_step = (
            current_visual_xy
            - previous_visual_xy
        )

        step_loss = F.smooth_l1_loss(
            predicted_step,
            target_step,
        )

        loss = (
            float(config.LOSS_POSITION)
            * position_loss
            + float(
                config.LOSS_BRANCH_DISTRIBUTION
            )
            * branch_loss
            + float(
                config.LOSS_LOCAL_CANDIDATE_CE
            )
            * local_ce
            + float(
                config.LOSS_RECOVERY_CANDIDATE_CE
            )
            * recovery_ce
            + float(
                config.LOSS_WAYPOINT_CANDIDATE_CE
            )
            * waypoint_ce
            + float(config.LOSS_STEP)
            * step_loss
        )

        accumulated_loss = (
            loss
            if accumulated_loss is None
            else accumulated_loss + loss
        )

        accumulated_steps += 1

        branch = (
            output.branch_probability[
                0
            ].detach().cpu().numpy()
        )

        logs.append(
            {
                "loss": float(
                    loss.detach().cpu()
                ),
                "position": float(
                    position_loss.detach().cpu()
                ),
                "branch": float(
                    branch_loss.detach().cpu()
                ),
                "hold": float(
                    branch[
                        config.HYPOTHESIS_HOLD
                    ]
                ),
                "local": float(
                    branch[
                        config.HYPOTHESIS_LOCAL
                    ]
                ),
                "recovery": float(
                    branch[
                        config.HYPOTHESIS_RECOVERY
                    ]
                ),
                "waypoint": float(
                    branch[
                        config.HYPOTHESIS_WAYPOINT
                    ]
                ),
                "local_capture": float(
                    local_capture.detach().cpu()
                ),
                "recovery_capture": float(
                    recovery_capture.detach().cpu()
                ),
                "waypoint_capture": float(
                    waypoint_capture.detach().cpu()
                ),
            }
        )

        new_motion = update_observed_motion(
            observed_motion,
            previous_visual_xy,
            current_visual_xy,
            leg,
        )

        hidden = output.hidden
        cell = output.cell
        observed_motion = new_motion

        previous_z_uav = z_uav.detach()
        previous_visual_xy = current_visual_xy
        previous_gt = gt_xy.detach()

        chunk_end = (
            accumulated_steps
            >= int(config.TBPTT_STEPS)
            or sequence_index
            == len(cache) - 1
        )

        if chunk_end:
            normalized = (
                accumulated_loss
                / float(accumulated_steps)
            )

            if not torch.isfinite(normalized):
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

            hidden = hidden.detach()
            cell = cell.detach()
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

    result = {}

    for key in (
        "loss",
        "position",
        "branch",
        "hold",
        "local",
        "recovery",
        "waypoint",
        "local_capture",
        "recovery_capture",
        "waypoint_capture",
    ):
        result[key] = float(
            np.mean(
                [row[key] for row in logs]
            )
        )

    result["teacher_ratio"] = float(ratio)

    return result


def train_temporal_model(
    model,
    visual,
    gallery_z_sat,
    cache,
    route,
    banks,
    device,
    epochs,
):
    print(
        "TEMPORAL v6: all Route-A frames, "
        "HOLD/LOCAL/RECOVERY/WAYPOINT",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(
            config.TEMPORAL_WEIGHT_DECAY
        ),
    )

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch_index in range(int(epochs)):
        metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            visual=visual,
            gallery_z_sat=gallery_z_sat,
            cache=cache,
            route=route,
            banks=banks,
            device=device,
            epoch_index=epoch_index,
        )

        torch.save(
            {
                "architecture": ARCHITECTURE_NAME,
                "model": model.state_dict(),
                "epoch": epoch_index + 1,
                "training_route": "route_A",
                "all_route_a_frames_used": True,
                "test_routes_used_for_training": False,
                "fixed_speed": False,
                "translation_gate": False,
                "free_running_velocity_head": False,
                "constant_velocity_kalman": False,
                "hypotheses": [
                    "HOLD",
                    "LOCAL",
                    "RECOVERY",
                    "WAYPOINT",
                ],
                "waypoint_transition": (
                    "argmax recurrent WAYPOINT hypothesis"
                ),
                "absolute_current_gt_network_input": False,
                "test_waypoint_frame_index_used": False,
            },
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch_index + 1:03d}/{epochs} "
            f"loss={metrics['loss']:.4f} "
            f"pos={metrics['position']:.4f} "
            f"branch={metrics['branch']:.4f} "
            f"teacher={metrics['teacher_ratio']:.2f} "
            f"H/L/R/W="
            f"{metrics['hold']:.2f}/"
            f"{metrics['local']:.2f}/"
            f"{metrics['recovery']:.2f}/"
            f"{metrics['waypoint']:.2f} "
            f"cap L/R/W="
            f"{metrics['local_capture'] * 100.0:.1f}/"
            f"{metrics['recovery_capture'] * 100.0:.1f}/"
            f"{metrics['waypoint_capture'] * 100.0:.1f}%",
            flush=True,
        )

    return {
        "train_frames": int(len(cache)),
        "route_a_waypoints": int(
            len(route.waypoints)
        ),
        "route_a_legs": int(
            len(route.legs)
        ),
    }


def load_temporal_model(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            config.TEMPORAL_CHECKPOINT
        )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location=device,
    )

    if (
        checkpoint.get("architecture")
        != ARCHITECTURE_NAME
    ):
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch: "
            f"{checkpoint.get('architecture')}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    return checkpoint


# =============================================================================
# Position-only Kalman / metrics
# =============================================================================

def make_position_kalman(initial_xy):
    if KalmanFilter is None:
        raise ImportError(
            "FilterPy required: pip install filterpy"
        )

    kf = KalmanFilter(
        dim_x=2,
        dim_z=2,
    )

    kf.x = np.asarray(
        initial_xy,
        dtype=np.float64,
    ).reshape(2)

    kf.F = np.eye(
        2,
        dtype=np.float64,
    )

    kf.H = np.eye(
        2,
        dtype=np.float64,
    )

    kf.P = (
        np.eye(2, dtype=np.float64)
        * float(
            config.KALMAN_INIT_POSITION_VAR
        )
    )

    kf.Q = (
        np.eye(2, dtype=np.float64)
        * float(
            config.KALMAN_Q_POSITION
        )
    )

    return kf


def metric_block(prediction, gt):
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
        pred_step = np.diff(
            prediction,
            axis=0,
        )

        gt_step = np.diff(
            gt,
            axis=0,
        )

        rpe = np.linalg.norm(
            pred_step - gt_step,
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

        jump_rate = float(
            (
                np.linalg.norm(
                    pred_step,
                    axis=1,
                )
                > jump_threshold
            ).mean()
            * 100.0
        )
    else:
        rpe = np.zeros(1)
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
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": jump_rate,
        "JumpThreshold_m": jump_threshold,
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
        "MaxLE_m": float(error.max()),
    }


# =============================================================================
# B/C inference
# =============================================================================

@torch.no_grad()
def run_inference(
    model,
    visual,
    gallery_z_sat,
    cache,
    route,
    banks,
    device,
    csv_path,
):
    model.eval()

    hidden, cell = model.initial_state(
        1,
        device,
        torch.float32,
    )

    observed_motion = torch.zeros(
        1,
        int(config.OBSERVED_MOTION_DIM),
        device=device,
        dtype=torch.float32,
    )

    active_leg_index = 0
    active_leg = route.legs[0]

    visual_xy = (
        active_leg.start_xy.to(
            device
        ).reshape(1, 2)
    )

    previous_z_uav = None

    mission_complete = False
    terminal_frame = None

    kf = make_position_kalman(
        visual_xy[0].cpu().numpy()
    )

    rows = []

    for sequence_index in range(len(cache)):
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
            == len(route.legs) - 1
        )

        previous_xy = visual_xy

        uav_clip = cache.uav_clip[
            sequence_index:
            sequence_index + 1
        ].to(device).float()

        z_uav = visual.model.encode_uav_from_clip(
            uav_clip
        )

        bank_indices = banks[
            active_leg_index
        ]

        global_logits, global_top_xy = (
            score_global_route(
                visual,
                z_uav,
                gallery_z_sat,
                bank_indices,
            )
        )

        local_candidate = build_bounded_candidate(
            visual,
            uav_clip,
            visual_xy,
            active_leg,
            bank_indices,
        )

        recovery_candidate = (
            build_bounded_candidate(
                visual,
                uav_clip,
                global_top_xy,
                active_leg,
                bank_indices,
            )
        )

        endpoint_center = (
            active_leg.end_xy.to(
                device
            ).reshape(1, 2)
        )

        waypoint_candidate = (
            build_waypoint_candidate(
                visual,
                uav_clip,
                endpoint_center,
            )
        )

        local_offsets = candidate_offsets_route(
            local_candidate.centers,
            visual_xy,
            active_leg,
        )

        recovery_offsets = candidate_offsets_route(
            recovery_candidate.centers,
            visual_xy,
            active_leg,
        )

        waypoint_offsets = candidate_offsets_route(
            waypoint_candidate.centers,
            visual_xy,
            active_leg,
        )

        context = route_context(
            visual_xy,
            active_leg,
            final_leg,
        )

        polynomial_delta = (
            observed_motion[:, 0:2]
            + 0.5
            * observed_motion[:, 2:4]
        )

        output = model.forward_step(
            z_uav=z_uav,
            previous_z_uav=previous_z_uav,
            local_z_sat=local_candidate.z_sat,
            local_raw_logits=local_candidate.raw_logits,
            local_valid_mask=local_candidate.valid_mask,
            local_offsets_route=local_offsets,
            recovery_z_sat=recovery_candidate.z_sat,
            recovery_raw_logits=recovery_candidate.raw_logits,
            recovery_valid_mask=recovery_candidate.valid_mask,
            recovery_offsets_route=recovery_offsets,
            waypoint_z_sat=waypoint_candidate.z_sat,
            waypoint_raw_logits=waypoint_candidate.raw_logits,
            waypoint_valid_mask=waypoint_candidate.valid_mask,
            waypoint_offsets_route=waypoint_offsets,
            global_stats=global_visual_stats(
                global_logits
            ),
            route_context=context,
            observed_motion=observed_motion,
            polynomial_delta_route=polynomial_delta,
            hidden=hidden,
            cell=cell,
        )

        local_xy, _ = decode_refined(
            output.local_refined_logits,
            local_candidate.centers,
            local_candidate.valid_mask,
        )

        recovery_xy, _ = decode_refined(
            output.recovery_refined_logits,
            recovery_candidate.centers,
            recovery_candidate.valid_mask,
        )

        waypoint_xy, _ = decode_refined(
            output.waypoint_refined_logits,
            waypoint_candidate.centers,
            waypoint_candidate.valid_mask,
        )

        proposed_xy, hypotheses = (
            compose_hypotheses(
                visual_xy,
                local_xy,
                recovery_xy,
                waypoint_xy,
                output.branch_probability,
            )
        )

        selected_branch = int(
            output.branch_probability[
                0
            ].argmax()
            .cpu()
            .item()
        )

        current_visual_xy = (
            visual_xy
            if mission_complete
            else proposed_xy
        )

        _, _, unit, normal, _ = (
            leg_geometry(
                active_leg,
                device=device,
                dtype=torch.float32,
            )
        )

        polynomial_xy = (
            visual_xy
            + leg_frame_to_xy(
                polynomial_delta,
                unit.reshape(1, 2),
                normal.reshape(1, 2),
            )
        )

        new_motion = update_observed_motion(
            observed_motion,
            visual_xy,
            current_visual_xy,
            active_leg,
        )

        switched = False
        waypoint_selected = (
            not mission_complete
            and selected_branch
            == int(
                config.HYPOTHESIS_WAYPOINT
            )
        )

        if waypoint_selected:
            if final_leg:
                if bool(
                    config.TERMINAL_LOCK_ENABLED
                ):
                    mission_complete = True
                    terminal_frame = int(frame_id)
                    new_motion = torch.zeros_like(
                        new_motion
                    )
            else:
                old_leg = active_leg
                active_leg_index += 1
                new_leg = route.legs[
                    active_leg_index
                ]

                new_motion = rotate_observed_motion(
                    new_motion,
                    old_leg,
                    new_leg,
                )

                switched = True

        kf.predict()

        measurement_variance = (
            output.measurement_variance[
                0
            ].cpu()
            .numpy()
            .astype(np.float64)
        )

        kf.R = np.diag(
            measurement_variance
        )

        kf.update(
            current_visual_xy[
                0
            ].cpu()
            .numpy()
            .astype(np.float64)
        )

        final_xy = np.asarray(
            kf.x,
            dtype=np.float64,
        ).reshape(2)

        gt_xy = cache.gt_xy[
            sequence_index
        ].numpy()

        branch = (
            output.branch_probability[
                0
            ].cpu()
            .numpy()
        )

        local_np = (
            local_xy[0].cpu().numpy()
        )

        recovery_np = (
            recovery_xy[0].cpu().numpy()
        )

        waypoint_np = (
            waypoint_xy[0].cpu().numpy()
        )

        global_np = (
            global_top_xy[0].cpu().numpy()
        )

        visual_np = (
            current_visual_xy[
                0
            ].cpu().numpy()
        )

        polynomial_np = (
            polynomial_xy[
                0
            ].cpu().numpy()
        )

        rows.append(
            {
                "sequence_index": int(
                    sequence_index
                ),
                "frame_id": int(frame_id),
                "image_path": (
                    cache.image_paths[
                        sequence_index
                    ]
                ),
                "active_waypoint_from": int(
                    active_leg.start.order
                ),
                "active_waypoint_to": int(
                    active_leg.end.order
                ),
                "waypoint_branch_selected": int(
                    waypoint_selected
                ),
                "waypoint_switched_after_frame": int(
                    switched
                ),
                "mission_complete": int(
                    mission_complete
                ),
                "gt_x": float(gt_xy[0]),
                "gt_y": float(gt_xy[1]),
                "hold_x": float(
                    previous_xy[
                        0,
                        0
                    ].cpu().item()
                ),
                "hold_y": float(
                    previous_xy[
                        0,
                        1
                    ].cpu().item()
                ),
                "local_x": float(local_np[0]),
                "local_y": float(local_np[1]),
                "recovery_x": float(
                    recovery_np[0]
                ),
                "recovery_y": float(
                    recovery_np[1]
                ),
                "waypoint_x": float(
                    waypoint_np[0]
                ),
                "waypoint_y": float(
                    waypoint_np[1]
                ),
                "global_top1_x": float(
                    global_np[0]
                ),
                "global_top1_y": float(
                    global_np[1]
                ),
                "polynomial_x": float(
                    polynomial_np[0]
                ),
                "polynomial_y": float(
                    polynomial_np[1]
                ),
                "branch_hold": float(
                    branch[
                        config.HYPOTHESIS_HOLD
                    ]
                ),
                "branch_local": float(
                    branch[
                        config.HYPOTHESIS_LOCAL
                    ]
                ),
                "branch_recovery": float(
                    branch[
                        config.HYPOTHESIS_RECOVERY
                    ]
                ),
                "branch_waypoint": float(
                    branch[
                        config.HYPOTHESIS_WAYPOINT
                    ]
                ),
                "selected_branch": int(
                    selected_branch
                ),
                "visual_x": float(
                    visual_np[0]
                ),
                "visual_y": float(
                    visual_np[1]
                ),
                "observed_delta_parallel": float(
                    new_motion[
                        0,
                        0
                    ].cpu().item()
                ),
                "observed_delta_cross": float(
                    new_motion[
                        0,
                        1
                    ].cpu().item()
                ),
                "observed_acc_parallel": float(
                    new_motion[
                        0,
                        2
                    ].cpu().item()
                ),
                "observed_acc_cross": float(
                    new_motion[
                        0,
                        3
                    ].cpu().item()
                ),
                "measurement_var_x": float(
                    measurement_variance[0]
                ),
                "measurement_var_y": float(
                    measurement_variance[1]
                ),
                "final_x": float(final_xy[0]),
                "final_y": float(final_xy[1]),
                "error_visual_m": float(
                    np.linalg.norm(
                        visual_np - gt_xy
                    )
                ),
                "error_final_m": float(
                    np.linalg.norm(
                        final_xy - gt_xy
                    )
                ),
            }
        )

        hidden = output.hidden
        cell = output.cell
        observed_motion = new_motion
        previous_z_uav = z_uav
        visual_xy = current_visual_xy

    gt = np.asarray(
        [
            [row["gt_x"], row["gt_y"]]
            for row in rows
        ],
        dtype=np.float64,
    )

    visual_prediction = np.asarray(
        [
            [row["visual_x"], row["visual_y"]]
            for row in rows
        ],
        dtype=np.float64,
    )

    final_prediction = np.asarray(
        [
            [row["final_x"], row["final_y"]]
            for row in rows
        ],
        dtype=np.float64,
    )

    summary = {
        "RouteBoundedVisual": metric_block(
            visual_prediction,
            gt,
        ),
        "FinalPositionKalman": metric_block(
            final_prediction,
            gt,
        ),
        "WaypointSwitchCount": int(
            sum(
                row[
                    "waypoint_switched_after_frame"
                ]
                for row in rows
            )
        ),
        "ExpectedWaypointSwitchCount": int(
            max(0, len(route.legs) - 1)
        ),
        "TerminalFrame": terminal_frame,
        "MeanHoldProbability": float(
            np.mean(
                [
                    row["branch_hold"]
                    for row in rows
                ]
            )
        ),
        "MeanLocalProbability": float(
            np.mean(
                [
                    row["branch_local"]
                    for row in rows
                ]
            )
        ),
        "MeanRecoveryProbability": float(
            np.mean(
                [
                    row["branch_recovery"]
                    for row in rows
                ]
            )
        ),
        "MeanWaypointProbability": float(
            np.mean(
                [
                    row["branch_waypoint"]
                    for row in rows
                ]
            )
        ),
    }

    csv_path = Path(csv_path)
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
        writer.writerows(rows)

    return summary, rows


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
        "--temporal-epochs",
        type=int,
        default=config.TEMPORAL_EPOCHS,
    )

    parser.add_argument(
        "--reuse-visual",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(config.SEED)

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100, flush=True)
    print(
        "ROUTE-BOUNDED HYPOTHESIS LSTM v6",
        flush=True,
    )
    print("=" * 100, flush=True)
    print(
        "No fixed speed / translation gate / velocity extrapolation.",
        flush=True,
    )
    print(
        "Branches: HOLD / LOCAL / RECOVERY / WAYPOINT.",
        flush=True,
    )
    print(
        "WAYPOINT branch is the learned mission-leg transition.",
        flush=True,
    )
    print(
        "Current-leg visual candidates are bounded by Start->End corridor.",
        flush=True,
    )
    print(
        "Polynomial is observed-motion context only; it cannot move XY.",
        flush=True,
    )
    print("=" * 100, flush=True)

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
                    "--reuse-visual requested but visual checkpoint missing: "
                    f"{config.VISUAL_CHECKPOINT}"
                )
            print(
                "[STAGE 1/4] reuse Route-A visual checkpoint",
                flush=True,
            )
        else:
            print(
                "[STAGE 1/4] train Route-A visual retrieval",
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
        "[STAGE 2/4] frozen visual model + SAT gallery",
        flush=True,
    )

    visual = FrozenVisualLocalizer(
        device
    )

    gallery_z_sat = (
        build_satellite_task_embeddings(
            visual
        )
    )

    model = (
        RouteBoundedHypothesisLSTM()
        .to(device)
    )

    catalog = route_catalog()

    route_a = load_mission_route(
        "route_A",
        visual.origin_lat,
        visual.origin_lon,
    )

    training_description = None

    if args.mode in (
        "train",
        "train_eval",
    ):
        if config.TEMPORAL_CHECKPOINT.exists():
            config.TEMPORAL_CHECKPOINT.unlink()

        route_a_cache = build_route_cache(
            "route_A",
            catalog["route_A"],
            visual,
            device,
        )

        route_a_banks = build_route_banks(
            visual,
            route_a,
        )

        print(
            "[STAGE 3/4] train v6 on all Route-A frames",
            flush=True,
        )

        training_description = (
            train_temporal_model(
                model=model,
                visual=visual,
                gallery_z_sat=gallery_z_sat,
                cache=route_a_cache,
                route=route_a,
                banks=route_a_banks,
                device=device,
                epochs=int(
                    args.temporal_epochs
                ),
            )
        )
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
            "[STAGE 4/4] Route-B / Route-C inference",
            flush=True,
        )

        results = {}
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

            cache = build_route_cache(
                route_name,
                catalog[route_name],
                visual,
                device,
            )

            banks = build_route_banks(
                visual,
                route,
            )

            csv_path = (
                config.OUTPUT_DIR
                / (
                    f"{route_name}_"
                    "route_hypothesis_lstm_frames.csv"
                )
            )

            summary, _ = run_inference(
                model=model,
                visual=visual,
                gallery_z_sat=gallery_z_sat,
                cache=cache,
                route=route,
                banks=banks,
                device=device,
                csv_path=csv_path,
            )

            results[route_name] = summary
            waypoint_counts[route_name] = (
                len(route.waypoints)
            )

            metric = summary[
                "FinalPositionKalman"
            ]

            print(
                f"{route_name}: "
                f"MLE={metric['MLE_m']:.3f}m "
                f"P90={metric['P90_m']:.3f}m "
                f"RPE={metric['RPE_m']:.3f}m "
                f"Jump={metric['JumpRate_pct']:.3f}% "
                f"switch={summary['WaypointSwitchCount']}/"
                f"{summary['ExpectedWaypointSwitchCount']} "
                f"H/L/R/W="
                f"{summary['MeanHoldProbability']:.2f}/"
                f"{summary['MeanLocalProbability']:.2f}/"
                f"{summary['MeanRecoveryProbability']:.2f}/"
                f"{summary['MeanWaypointProbability']:.2f}",
                flush=True,
            )

        payload = {
            "architecture": ARCHITECTURE_NAME,
            "v5_failure_fixed": {
                "translation_gate_removed": True,
                "route_unbounded_drift_removed": True,
                "exact_endpoint_radius_switch_removed": True,
                "local_only_no_recovery_removed": True,
            },
            "training": {
                "route": "route_A",
                "all_route_a_frames_used": True,
                "test_routes_used_for_training": False,
                "absolute_current_gt_network_input": False,
                "gt_role": (
                    "supervision and early causal previous-frame local-center guidance only"
                ),
                "description": training_description,
            },
            "model": {
                "branches": [
                    "HOLD",
                    "LOCAL",
                    "RECOVERY",
                    "WAYPOINT",
                ],
                "waypoint_transition": (
                    "learned WAYPOINT branch argmax"
                ),
                "route_recovery": (
                    "current-image global retrieval only inside active Start->End corridor"
                ),
                "polynomial": (
                    "previous observed image-derived delta + 0.5 acceleration"
                ),
                "polynomial_moves_position": False,
                "fixed_speed": False,
                "translation_gate": False,
            },
            "inference": {
                "test_waypoint_frame_index_used": False,
                "test_gt_used_by_inference": False,
                "route_candidate_space_bounded": True,
                "kalman_state": "[x,y] only",
            },
            "waypoint_counts": waypoint_counts,
            "routes": results,
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
