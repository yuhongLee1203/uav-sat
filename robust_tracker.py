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
    regular_grid_indices,
    train_visual_retrieval_a_only,
)
from visual_model import PureVisualLSTM

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "WaypointInertial_FixedHardMS_FilterPyKalman"


@dataclass
class RouteCache:
    route_name: str
    frame_ids: torch.Tensor
    timestamps_ns: torch.Tensor
    gt_xy: torch.Tensor
    raw_gt_xy: torch.Tensor
    uav_clip: torch.Tensor
    image_paths: list

    def __len__(self):
        return int(self.gt_xy.shape[0])


@dataclass
class MissionWaypoint:
    order: int
    xy: torch.Tensor


@dataclass
class MissionRoute:
    route_name: str
    waypoints: list


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


# =============================================================================
# INFERENCE-ONLY waypoint loader.
#
# Training never calls this function.
# Only waypoint coordinate/order is loaded.
# frame_index and timestamp are intentionally ignored.
# =============================================================================

def load_mission_route_for_inference(
    route_name,
    origin_lat,
    origin_lon,
):
    path = Path(config.WAYPOINT_FILES[route_name])

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    raw_waypoints = sorted(
        payload["waypoints"],
        key=lambda item: int(item["waypoint_order"]),
    )

    waypoints = []

    for item in raw_waypoints:
        x_meter, y_meter = meters_from_latlon(
            item["latitude"],
            item["longitude"],
            origin_lat,
            origin_lon,
        )

        waypoints.append(
            MissionWaypoint(
                order=int(item["waypoint_order"]),
                xy=torch.tensor(
                    [x_meter, y_meter],
                    dtype=torch.float32,
                ),
            )
        )

    if len(waypoints) < 2:
        raise RuntimeError(
            f"{route_name}: fewer than 2 mission waypoints"
        )

    print(
        f"[INFERENCE ONLY] {route_name}: "
        f"loaded {len(waypoints)} waypoint coordinates/order",
        flush=True,
    )

    print(
        "  "
        + " -> ".join(
            f"W{waypoint.order}"
            for waypoint in waypoints
        ),
        flush=True,
    )

    print(
        "  waypoint frame_index/timestamp are NOT used",
        flush=True,
    )

    return MissionRoute(
        route_name=route_name,
        waypoints=waypoints,
    )


# =============================================================================
# Pure visual sequence cache.
# No waypoint is involved.
# =============================================================================

@torch.no_grad()
def build_route_cache(
    route_name,
    root,
    visual,
    device,
):
    checkpoint_stat = config.VISUAL_CHECKPOINT.stat()
    visual_signature = {
        "path": str(config.VISUAL_CHECKPOINT),
        "size": int(checkpoint_stat.st_size),
        "mtime_ns": int(checkpoint_stat.st_mtime_ns),
    }
    cache_path = (
        config.OUTPUT_DIR
        / "feature_cache"
        / f"{route_name}_uav_clip.pt"
    )

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if (
            payload.get("visual_signature")
            == visual_signature
        ):
            print(
                f"{route_name}: reuse cached frozen UAV embeddings",
                flush=True,
            )
            return RouteCache(
                route_name=route_name,
                frame_ids=payload["frame_ids"],
                timestamps_ns=payload["timestamps_ns"],
                gt_xy=payload["gt_xy"],
                raw_gt_xy=payload["raw_gt_xy"],
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
    timestamp_rows = []
    gt_rows = []
    raw_gt_rows = []
    clip_rows = []
    image_paths = []

    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)

    for start in range(0, len(dataset), batch_size):
        end = min(
            start + batch_size,
            len(dataset),
        )

        items = [
            dataset[index]
            for index in range(start, end)
        ]

        uav = torch.stack(
            [
                item["uav"]
                for item in items
            ]
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

        raw_gt_rows.append(
            torch.stack(
                [
                    item["raw_xy"].float()
                    for item in items
                ]
            )
        )

        for item in items:
            frame_rows.append(
                parse_frame_id(
                    item["frame_id"]
                )
            )
            timestamp_rows.append(
                int(item["timestamp_ns"].item())
            )
            image_paths.append(
                str(item["image_path"])
            )

        if (
            start == 0
            or end == len(dataset)
            or (start // batch_size) % 20 == 0
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
        timestamps_ns=torch.tensor(
            timestamp_rows,
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
        image_paths=image_paths,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "visual_signature": visual_signature,
            "frame_ids": result.frame_ids,
            "timestamps_ns": result.timestamps_ns,
            "gt_xy": result.gt_xy,
            "raw_gt_xy": result.raw_gt_xy,
            "uav_clip": result.uav_clip,
            "image_paths": result.image_paths,
        },
        cache_path,
    )
    print(
        f"{route_name}: cached frozen UAV embeddings -> {cache_path}",
        flush=True,
    )
    return result


def split_route_a_by_time(cache):
    count = len(cache)

    train_end = int(
        count
        * float(config.TEMPORAL_TRAIN_FRACTION)
    )

    val_end = train_end + int(
        count
        * float(config.TEMPORAL_VAL_FRACTION)
    )

    train_end = max(
        2,
        min(train_end, count - 2),
    )

    val_end = max(
        train_end + 1,
        min(val_end, count - 1),
    )

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, count),
    }


def derive_nominal_speed_mps(cache):
    """Estimate one deployment-time speed prior from Route-A supervision only."""
    xy = cache.gt_xy.numpy().astype(np.float64)
    timestamps = cache.timestamps_ns.numpy().astype(np.float64)
    displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(timestamps) / 1_000_000_000.0
    speed = displacement / np.clip(dt, 1e-3, None)
    plausible = speed[(speed >= 1.0) & (speed <= float(config.KALMAN_MAX_SPEED_MPS))]
    if plausible.size == 0:
        return float(config.KALMAN_DEFAULT_NOMINAL_SPEED_MPS)
    return float(np.median(plausible))


# =============================================================================
# Candidate construction / output decoding.
#
# Coordinates are used ONLY outside the neural network:
#   - supervised local candidate-image sampling
#   - label construction
#   - converting image probabilities into an output XY
#
# They never enter PureVisualLSTM.forward_step().
# =============================================================================

def nearest_candidate_indices(
    gallery_xy,
    center_xy,
    count,
):
    center_xy = center_xy.to(
        gallery_xy.device
    ).reshape(1, 2)

    distance2 = (
        gallery_xy
        - center_xy
    ).square().sum(dim=1)

    count = min(
        int(count),
        int(gallery_xy.shape[0]),
    )

    return torch.topk(
        distance2,
        k=count,
        largest=False,
    ).indices.reshape(
        1,
        -1,
    )


def deterministic_jitter(
    index,
    jitter_m,
):
    angle = (
        float(index)
        * 2.399963229728653
    )

    radius = (
        float(jitter_m)
        * (
            0.25
            + 0.75
            * (
                (
                    index
                    * 1103515245
                    + 12345
                )
                % 1000
            )
            / 999.0
        )
    )

    return torch.tensor(
        [
            radius * math.cos(angle),
            radius * math.sin(angle),
        ],
        dtype=torch.float32,
    )


@torch.no_grad()
def build_candidate_batch(
    visual,
    uav_clip,
    indices,
):
    device = visual.device

    indices = indices.to(device)

    centers = visual.gallery["xy"][
        indices
    ]

    sat_clip = visual.gallery[
        "clip_feat"
    ][indices]

    z_uav = (
        visual.model.encode_uav_from_clip(
            uav_clip
        )
    )

    z_sat = (
        visual.model.encode_sat_from_clip(
            sat_clip.reshape(
                -1,
                sat_clip.shape[-1],
            ),
            centers.reshape(-1, 2),
        )
        .reshape(
            centers.shape[0],
            centers.shape[1],
            -1,
        )
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

    (
        hardms_xy,
        hardms_support,
    ) = hard_mean_shift(
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


def decode_visual_xy(
    refined_logits,
    centers,
):
    probability = torch.softmax(
        refined_logits,
        dim=1,
    )

    xy = (
        probability.unsqueeze(-1)
        * centers
    ).sum(dim=1)

    return xy, probability


def nearest_gt_label(
    centers,
    gt_xy,
):
    distance = torch.linalg.norm(
        centers
        - gt_xy.reshape(1, 1, 2),
        dim=2,
    )

    return distance.argmin(
        dim=1
    )


# =============================================================================
# PURE-VISUAL temporal training.
#
# NO waypoint loader is called.
# NO coordinate/candidate center enters the LSTM.
# GT XY is supervision only.
# =============================================================================

def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    start_index,
    end_index,
    device,
):
    model.train()

    hidden, cell = model.initial_state(
        1,
        device,
        torch.float32,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_accumulator = None
    chunk_steps = 0

    rows = []
    captures = []

    for index in range(
        int(start_index),
        int(end_index),
    ):
        gt_xy_cpu = cache.gt_xy[index]

        jitter = deterministic_jitter(
            index,
            config.TRAIN_CANDIDATE_JITTER_M,
        )

        # Supervised data sampling only.
        prior_xy = gt_xy_cpu + jitter

        candidate_indices = regular_grid_indices(
            visual.gallery["xy"],
            visual.gallery["pixel"],
            visual.pixel_index,
            prior_xy.reshape(1, 2),
            config.GRID_SIZE,
            config.SAT_STRIDE,
            device,
        )

        uav_clip = cache.uav_clip[
            index:index + 1
        ].to(device).float()

        candidate = build_candidate_batch(
            visual,
            uav_clip,
            candidate_indices,
        )

        # PURE VISUAL INPUTS ONLY.
        output = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_logits,
            candidate.raw_prob,
            hidden,
            cell,
        )

        hidden = output.hidden
        cell = output.cell

        gt_xy = gt_xy_cpu.to(
            device
        ).reshape(1, 2)

        target_index = nearest_gt_label(
            candidate.centers,
            gt_xy,
        )

        measurement_xy, _ = decode_visual_xy(
            output.refined_logits,
            candidate.centers,
        )

        ce_loss = F.cross_entropy(
            output.refined_logits,
            target_index,
            label_smoothing=float(
                config.VISUAL_LABEL_SMOOTHING
            ),
        )

        coord_loss = F.smooth_l1_loss(
            measurement_xy,
            gt_xy,
        )

        nll_loss = F.gaussian_nll_loss(
            measurement_xy,
            gt_xy,
            output.measurement_variance,
            full=False,
            reduction="mean",
        )

        loss = (
            float(config.LOSS_CE)
            * ce_loss
            + float(
                config.LOSS_COORD_SMOOTH_L1
            )
            * coord_loss
            + float(
                config.LOSS_GAUSSIAN_NLL
            )
            * nll_loss
        )

        if loss_accumulator is None:
            loss_accumulator = loss
        else:
            loss_accumulator = (
                loss_accumulator
                + loss
            )

        chunk_steps += 1

        minimum_distance = (
            torch.linalg.norm(
                candidate.centers[0]
                - gt_xy[0].reshape(1, 2),
                dim=1,
            ).min()
        )

        captures.append(
            bool(
                minimum_distance.item()
                <= float(
                    config.CANDIDATE_CAPTURE_RADIUS_M
                )
            )
        )

        rows.append(
            {
                "loss": float(
                    loss.detach().cpu()
                ),
                "ce": float(
                    ce_loss.detach().cpu()
                ),
                "coord": float(
                    coord_loss.detach().cpu()
                ),
                "nll": float(
                    nll_loss.detach().cpu()
                ),
            }
        )

        is_chunk_end = (
            chunk_steps
            >= int(config.TBPTT_STEPS)
            or index
            == int(end_index) - 1
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
                float(
                    config.GRAD_CLIP_NORM
                ),
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            hidden = hidden.detach()
            cell = cell.detach()

            loss_accumulator = None
            chunk_steps = 0

    means = {
        key: float(
            np.mean(
                [
                    row[key]
                    for row in rows
                ]
            )
        )
        for key in rows[0]
    }

    means["capture_pct"] = float(
        np.mean(captures)
        * 100.0
    )

    return means


@torch.no_grad()
def evaluate_visual_local_sequence(
    model,
    visual,
    cache,
    start_index,
    end_index,
    device,
):
    model.eval()

    hidden, cell = model.initial_state(
        1,
        device,
        torch.float32,
    )

    predictions = []
    gt_rows = []

    for index in range(
        int(start_index),
        int(end_index),
    ):
        gt_xy_cpu = cache.gt_xy[index]

        jitter = deterministic_jitter(
            index + 100000,
            config.TRAIN_CANDIDATE_JITTER_M,
        )

        candidate_indices = regular_grid_indices(
            visual.gallery["xy"],
            visual.gallery["pixel"],
            visual.pixel_index,
            (gt_xy_cpu + jitter).reshape(1, 2),
            config.GRID_SIZE,
            config.SAT_STRIDE,
            device,
        )

        uav_clip = cache.uav_clip[
            index:index + 1
        ].to(device).float()

        candidate = build_candidate_batch(
            visual,
            uav_clip,
            candidate_indices,
        )

        output = model.forward_step(
            candidate.z_uav,
            candidate.z_sat,
            candidate.raw_logits,
            candidate.raw_prob,
            hidden,
            cell,
        )

        hidden = output.hidden
        cell = output.cell

        measurement_xy, _ = decode_visual_xy(
            output.refined_logits,
            candidate.centers,
        )

        predictions.append(
            measurement_xy[0].cpu().numpy()
        )

        gt_rows.append(
            gt_xy_cpu.numpy()
        )

    prediction = np.asarray(
        predictions,
        dtype=np.float64,
    )

    gt = np.asarray(
        gt_rows,
        dtype=np.float64,
    )

    error = np.linalg.norm(
        prediction - gt,
        axis=1,
    )

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
    }


def train_temporal_model(
    model,
    visual,
    cache,
    device,
    epochs,
):
    split = split_route_a_by_time(
        cache
    )
    nominal_speed_mps = derive_nominal_speed_mps(cache)

    print(
        "Pure-visual temporal split:",
        split,
        flush=True,
    )
    print(
        f"Route-A nominal speed prior={nominal_speed_mps:.3f} m/s",
        flush=True,
    )

    print(
        "TRAINING INPUT AUDIT:",
        flush=True,
    )

    print(
        "  LSTM INPUT = UAV image embedding + SAT image embeddings "
        "+ visual similarity + previous hidden/cell ONLY",
        flush=True,
    )

    print(
        "  NO waypoint / XY / candidate center / velocity / previous position "
        "/ Kalman state / GPS / timestamp enters LSTM",
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

    best_mle = float("inf")
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
            split["train"][0],
            split["train"][1],
            device,
        )

        validation = (
            evaluate_visual_local_sequence(
                model,
                visual,
                cache,
                split["val"][0],
                split["val"][1],
                device,
            )
        )

        if (
            validation["MLE_m"]
            < best_mle
        ):
            best_mle = (
                validation["MLE_m"]
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
                "architecture": ARCHITECTURE_NAME,
                "model": model.state_dict(),
                "best_model": best_state,
                "epoch": epoch + 1,
                "best_val_mle": best_mle,
                "temporal_train_routes": [
                    "route_A"
                ],
                "training_network_inputs": [
                    "uav_image_embedding",
                    "satellite_image_embeddings",
                    "visual_similarity_logits",
                    "visual_similarity_probabilities",
                    "previous_lstm_hidden",
                    "previous_lstm_cell",
                ],
                "explicitly_not_network_inputs": [
                    "waypoint",
                    "xy_coordinate",
                    "candidate_center",
                    "velocity",
                    "previous_position",
                    "kalman_state",
                    "gps",
                    "timestamp",
                ],
                "waypoint_used_in_training": False,
                "test_gps_used_in_inference": False,
                "nominal_speed_mps": nominal_speed_mps,
            },
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"loss={training['loss']:.5f} "
            f"ce={training['ce']:.4f} "
            f"coord={training['coord']:.4f} "
            f"nll={training['nll']:.4f} "
            f"capture={training['capture_pct']:.2f}% "
            f"val_mle={validation['MLE_m']:.3f}m "
            f"val_p90={validation['P90_m']:.3f}m",
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
            "No temporal best state"
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

    return split, nominal_speed_mps


def load_temporal_model(
    model,
    device,
):
    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location=device,
    )

    if (
        checkpoint.get("architecture")
        != ARCHITECTURE_NAME
    ):
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


# =============================================================================
# Inference-only mission route search manager.
# =============================================================================

def mission_leg_geometry(
    route,
    active_leg_index,
):
    start = route.waypoints[
        active_leg_index
    ].xy

    end = route.waypoints[
        active_leg_index + 1
    ].xy

    vector = end - start

    length = float(
        torch.linalg.norm(
            vector
        ).item()
    )

    unit = (
        vector
        / max(
            length,
            1e-8,
        )
    )

    normal = torch.tensor(
        [
            -float(unit[1]),
            float(unit[0]),
        ],
        dtype=torch.float32,
    )

    return (
        start,
        end,
        unit,
        normal,
        length,
    )


def maybe_advance_waypoint(
    route,
    active_leg_index,
    estimated_xy,
    confirmation_count,
):
    if (
        active_leg_index
        >= len(route.waypoints) - 2
    ):
        return active_leg_index, 0

    (
        _,
        end,
        _,
        _,
        _,
    ) = mission_leg_geometry(
        route,
        active_leg_index,
    )

    estimated_xy = torch.as_tensor(
        estimated_xy,
        dtype=torch.float32,
    )

    distance_to_target = float(
        torch.linalg.norm(
            estimated_xy - end
        ).item()
    )

    reached = (
        distance_to_target
        <= float(
            config.INFER_WAYPOINT_REACHED_RADIUS_M
        )
    )

    confirmation_count = confirmation_count + 1 if reached else 0

    if confirmation_count >= int(config.INFER_WAYPOINT_CONFIRMATION_FRAMES):
        return active_leg_index + 1, 0

    return active_leg_index, confirmation_count


def waypoint_forward_candidate_indices(
    visual,
    route,
    active_leg_index,
    predicted_xy,
):
    """Return an overlapping bank of 6x6 Fixed-HardMS candidate lattices.

    A single 6x6 lattice spans only about 22 m with the present 320/32
    gallery.  Small speed differences can therefore eject the true position
    from the candidate set before a visual update is possible.  The bank keeps
    the original 6x6 local decoder unchanged while providing nine nearby
    recovery proposals.  It uses the Kalman state and the active waypoint-leg
    axes only; it never reads test GPS or ground truth.
    """
    _, _, unit, normal, _ = mission_leg_geometry(
        route,
        active_leg_index,
    )
    predicted = torch.as_tensor(
        predicted_xy,
        dtype=torch.float32,
    ).reshape(1, 2)
    unit = unit.reshape(1, 2)
    normal = normal.reshape(1, 2)
    radius = int(config.RECOVERY_BANK_RADIUS)
    step = float(config.RECOVERY_BANK_CENTER_STEP_M)

    proposal_centers = []
    for along_offset in range(-radius, radius + 1):
        for cross_offset in range(-radius, radius + 1):
            proposal_centers.append(
                predicted
                + step * float(along_offset) * unit
                + step * float(cross_offset) * normal
            )

    proposal_centers = torch.cat(proposal_centers, dim=0)
    return regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        proposal_centers,
        config.GRID_SIZE,
        config.SAT_STRIDE,
        visual.device,
    ), proposal_centers


def select_recovery_proposal(candidate, predicted_xy, route, active_leg_index):
    """Choose one visual 6x6 mode from the recovery bank without GT.

    The learned image score is primary. Mean-shift support rejects an isolated
    response. A small active-leg progress term resolves visually near-tied
    proposals without using ground truth: it prevents the fixed Route-A speed
    prior from repeatedly selecting a slightly-behind lattice when the UAV is
    moving forward.
    """
    peak_logit = candidate.raw_logits.max(dim=1).values
    distance = torch.linalg.norm(
        candidate.hardms_xy
        - torch.as_tensor(
            predicted_xy,
            device=candidate.hardms_xy.device,
            dtype=candidate.hardms_xy.dtype,
        ).reshape(1, 2),
        dim=1,
    )
    _, _, unit, _, _ = mission_leg_geometry(route, active_leg_index)
    relative = candidate.hardms_xy - torch.as_tensor(
        predicted_xy,
        device=candidate.hardms_xy.device,
        dtype=candidate.hardms_xy.dtype,
    ).reshape(1, 2)
    forward_progress = (
        relative * unit.to(candidate.hardms_xy.device).reshape(1, 2)
    ).sum(dim=1).clamp(min=-15.0, max=15.0)

    score = (
        peak_logit
        + 0.5 * candidate.hardms_support.clamp_min(1e-6).log()
        - float(config.RECOVERY_GRID_SELECTION_DISTANCE_WEIGHT) * distance
        + float(config.RECOVERY_GRID_PROGRESS_WEIGHT) * forward_progress
    )
    selected = int(score.argmax().item())
    return selected, score


# =============================================================================
# Inference-only FilterPy Kalman.
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
            float(initial_xy[0]),
            float(initial_xy[1]),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    set_kalman_dt(kf, 1.0 / 3.0)

    kf.H = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
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
    ).astype(np.float64)

    return kf


def set_kalman_dt(kf, dt_seconds):
    """Use source camera time, with velocity represented in m/s."""
    dt = float(np.clip(dt_seconds, 0.05, 1.0))
    kf.F = np.asarray(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    kf.Q = np.diag(
        [
            float(config.KALMAN_Q_POSITION) * dt * dt,
            float(config.KALMAN_Q_POSITION) * dt * dt,
            float(config.KALMAN_Q_VELOCITY) * dt,
            float(config.KALMAN_Q_VELOCITY) * dt,
        ]
    ).astype(np.float64)


def clip_kalman_velocity(kf):
    velocity = np.asarray(kf.x[2:4], dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    maximum = float(config.KALMAN_MAX_SPEED_MPS)
    if speed > maximum:
        kf.x[2:4] = velocity * (maximum / speed)


def gated_visual_measurement(predicted_xy, visual_xy):
    """Limit one ambiguous local mode to a bounded position correction."""
    innovation = np.asarray(visual_xy, dtype=np.float64) - np.asarray(
        predicted_xy,
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(innovation))
    maximum = float(config.KALMAN_MAX_VISUAL_INNOVATION_M)
    if magnitude > maximum:
        innovation *= maximum / magnitude
    return np.asarray(predicted_xy, dtype=np.float64) + innovation


def set_velocity_for_leg(kf, route, active_leg_index, nominal_speed_mps):
    """Initialize/reset inertial velocity from the inference-only mission leg."""
    _, _, unit, _, _ = mission_leg_geometry(route, active_leg_index)
    kf.x[2:4] = (
        unit.numpy().astype(np.float64)
        * min(float(nominal_speed_mps), float(config.KALMAN_MAX_SPEED_MPS))
    )


def constrain_filter_to_active_leg(
    kf,
    route,
    active_leg_index,
    nominal_speed_mps,
):
    """Prevent a prediction from flying beyond an unconfirmed waypoint."""
    start, _, unit, _, length = mission_leg_geometry(route, active_leg_index)
    start_np = start.numpy().astype(np.float64)
    unit_np = unit.numpy().astype(np.float64)
    relative = kf.x[0:2] - start_np
    along = float(np.dot(relative, unit_np))

    if along > length:
        cross = relative - along * unit_np
        kf.x[0:2] = start_np + length * unit_np + cross
        forward_speed = float(np.dot(kf.x[2:4], unit_np))
        if forward_speed > 0.0:
            kf.x[2:4] = kf.x[2:4] - forward_speed * unit_np

    # The local visual response is ambiguous outside its 6x6 coverage, so it
    # must not turn one weak correction into an unbounded speed estimate. The
    # route-A speed prior supplies stable inertial propagation; visual evidence
    # corrects position, while waypoint confirmation changes direction.
    if along < length:
        speed = min(
            float(nominal_speed_mps),
            float(config.KALMAN_MAX_SPEED_MPS),
        )
        kf.x[2:4] = speed * unit_np
    else:
        kf.x[2:4] = 0.0


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
            + float(
                config.JUMP_TOLERANCE_M
            )
        )

        pred_step_length = np.linalg.norm(
            pred_step,
            axis=1,
        )

        jump_rate = float(
            (
                pred_step_length
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
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "ATE_RMSE_m": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": float(jump_rate),
        "JumpThreshold_m": float(
            jump_threshold
        ),
        "LSR@5_pct": float(
            (error <= 5.0).mean() * 100.0
        ),
        "LSR@10_pct": float(
            (error <= 10.0).mean() * 100.0
        ),
        "LSR@15_pct": float(
            (error <= 15.0).mean() * 100.0
        ),
        "LSR@20_pct": float(
            (error <= 20.0).mean() * 100.0
        ),
        "MaxLE_m": float(error.max()),
    }


@torch.no_grad()
def run_waypoint_inference(
    model,
    visual,
    cache,
    route,
    device,
    csv_path,
    nominal_speed_mps,
):
    # The deployed decoder is the image-only Fixed HardMS visual model.  The
    # optional legacy LSTM object is intentionally not consulted here.
    del model, device

    initial_xy = route.waypoints[
        0
    ].xy.numpy()

    kf = make_kalman_filter(
        initial_xy
    )

    active_leg_index = 0
    waypoint_confirmation_count = 0
    previous_timestamp_ns = None
    set_velocity_for_leg(
        kf,
        route,
        active_leg_index,
        nominal_speed_mps,
    )

    rows = []

    for sequence_index in range(
        len(cache)
    ):
        frame_id = int(
            cache.frame_ids[
                sequence_index
            ].item()
        )

        timestamp_ns = int(
            cache.timestamps_ns[
                sequence_index
            ].item()
        )

        if previous_timestamp_ns is None:
            dt_seconds = 1.0 / 3.0
        else:
            dt_seconds = (
                timestamp_ns - previous_timestamp_ns
            ) / 1_000_000_000.0
        previous_timestamp_ns = timestamp_ns

        set_kalman_dt(kf, dt_seconds)

        # Kalman prediction uses previous FILTER OUTPUT only.
        if sequence_index > 0:
            kf.predict()
            clip_kalman_velocity(kf)
            constrain_filter_to_active_leg(
                kf,
                route,
                active_leg_index,
                nominal_speed_mps,
            )

        predicted_xy = np.asarray(
            kf.x[0:2],
            dtype=np.float64,
        )

        # Waypoint is used only by the external SAT-search manager.
        (
            candidate_indices,
            proposal_centers,
        ) = (
            waypoint_forward_candidate_indices(
                visual,
                route,
                active_leg_index,
                predicted_xy,
            )
        )

        uav_clip = cache.uav_clip[
            sequence_index:
            sequence_index + 1
        ].to(visual.device).float().expand(
            candidate_indices.shape[0],
            -1,
        )

        candidate = build_candidate_batch(
            visual,
            uav_clip,
            candidate_indices,
        )

        # The temporal LSTM is deliberately not used as a coordinate decoder.
        # It was trained with a global-centroid loss, whereas the deployed
        # decoder is Fixed HardMS.  Each proposal therefore uses the same
        # image-only 6x6 Fixed-HardMS rule as the visual baseline.
        selected_proposal, proposal_score = select_recovery_proposal(
            candidate,
            predicted_xy,
            route,
            active_leg_index,
        )

        visual_measurement_np = (
            candidate.hardms_xy[selected_proposal]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        visual_support = float(
            candidate.hardms_support[selected_proposal]
            .cpu()
            .item()
        )
        measurement_variance = np.full(
            2,
            max(
                float(config.KALMAN_MIN_MEASUREMENT_VARIANCE),
                16.0 / max(visual_support, 0.20) ** 2,
            ),
            dtype=np.float64,
        )
        gated_measurement_np = gated_visual_measurement(
            predicted_xy,
            visual_measurement_np,
        )

        kf.R = np.diag(
            measurement_variance
        )

        kf.update(
            gated_measurement_np
        )
        clip_kalman_velocity(kf)
        constrain_filter_to_active_leg(
            kf,
            route,
            active_leg_index,
            nominal_speed_mps,
        )

        final_xy = np.asarray(
            kf.x[0:2],
            dtype=np.float64,
        )

        old_leg = (
            active_leg_index
        )

        # No waypoint frame index / GT is used.
        (
            active_leg_index,
            waypoint_confirmation_count,
        ) = (
            maybe_advance_waypoint(
                route,
                active_leg_index,
                final_xy,
                waypoint_confirmation_count,
            )
        )

        if active_leg_index != old_leg:
            set_velocity_for_leg(
                kf,
                route,
                active_leg_index,
                nominal_speed_mps,
            )

        gt_xy = cache.gt_xy[
            sequence_index
        ].numpy()
        raw_gt_xy = cache.raw_gt_xy[
            sequence_index
        ].numpy()

        raw_top1 = (
            candidate.raw_top1_xy[selected_proposal]
            .cpu()
            .numpy()
        )

        hardms = (
            candidate.hardms_xy[selected_proposal]
            .cpu()
            .numpy()
        )

        current_from = int(
            route.waypoints[
                old_leg
            ].order
        )

        current_to = int(
            route.waypoints[
                old_leg + 1
            ].order
        )

        rows.append(
            {
                "sequence_index": int(
                    sequence_index
                ),
                "frame_id": frame_id,
                "timestamp_ns": timestamp_ns,
                "dt_seconds": float(dt_seconds),
                "image_path": (
                    cache.image_paths[
                        sequence_index
                    ]
                ),
                "active_waypoint_from": (
                    current_from
                ),
                "active_waypoint_to": (
                    current_to
                ),
                "waypoint_switched_after_frame": int(
                    active_leg_index
                    != old_leg
                ),
                "gt_x": float(
                    gt_xy[0]
                ),
                "gt_y": float(
                    gt_xy[1]
                ),
                "raw_gps_x": float(raw_gt_xy[0]),
                "raw_gps_y": float(raw_gt_xy[1]),
                "prediction_x": float(
                    predicted_xy[0]
                ),
                "prediction_y": float(
                    predicted_xy[1]
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
                "visual_measurement_x": float(
                    visual_measurement_np[0]
                ),
                "visual_measurement_y": float(
                    visual_measurement_np[1]
                ),
                "gated_measurement_x": float(
                    gated_measurement_np[0]
                ),
                "gated_measurement_y": float(
                    gated_measurement_np[1]
                ),
                "visual_innovation_m": float(
                    np.linalg.norm(
                        visual_measurement_np - predicted_xy
                    )
                ),
                "measurement_var_x": float(
                    measurement_variance[0]
                ),
                "measurement_var_y": float(
                    measurement_variance[1]
                ),
                "hardms_support": float(
                    visual_support
                ),
                "recovery_proposal_index": int(
                    selected_proposal
                ),
                "recovery_proposal_score": float(
                    proposal_score[selected_proposal]
                    .cpu()
                    .item()
                ),
                "recovery_center_x": float(
                    proposal_centers[selected_proposal, 0]
                    .cpu()
                    .item()
                ),
                "recovery_center_y": float(
                    proposal_centers[selected_proposal, 1]
                    .cpu()
                    .item()
                ),
                "final_x": float(
                    final_xy[0]
                ),
                "final_y": float(
                    final_xy[1]
                ),
                "vx": float(kf.x[2]),
                "vy": float(kf.x[3]),
                "waypoint_confirmation_count": int(
                    waypoint_confirmation_count
                ),
                "error_m": float(
                    np.linalg.norm(
                        final_xy - gt_xy
                    )
                ),
            }
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
        "PureVisualLSTMMeasurement": metric_block(
            [
                [
                    row[
                        "visual_measurement_x"
                    ],
                    row[
                        "visual_measurement_y"
                    ],
                ]
                for row in rows
            ],
            gt,
        ),
        "FinalKalman": metric_block(
            [
                [
                    row["final_x"],
                    row["final_y"],
                ]
                for row in rows
            ],
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
        writer.writerows(rows)

    return summary, rows


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

    print(
        "=" * 92,
        flush=True,
    )

    print(
        "VISUAL FIXED-HARDMS + WAYPOINT-INERTIAL FILTERPY KALMAN",
        flush=True,
    )

    print(
        "=" * 92,
        flush=True,
    )

    print(
        "TRAINING NETWORK INPUT: UAV/SAT images and image embeddings ONLY",
        flush=True,
    )

    print(
        "NO waypoint / XY / velocity / GPS / timestamp / Kalman state "
        "enters the visual retrieval network",
        flush=True,
    )

    print(
        "Waypoint is loaded only during B/C inference as a motion prior",
        flush=True,
    )

    print(
        "TEST GT is used ONLY for metrics/visualization",
        flush=True,
    )

    print(
        "=" * 92,
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
                "[STAGE 1/4] reuse visual retrieval checkpoint",
                flush=True,
            )
        else:
            print(
                "[STAGE 1/4] train visual retrieval on Route A",
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

    catalog = route_catalog()

    split = "not_applicable: no learned temporal network"
    nominal_speed_mps = float(config.KALMAN_DEFAULT_NOMINAL_SPEED_MPS)

    if args.mode in (
        "train",
        "train_eval",
    ):
        print(
            "[STAGE 3/4] Fixed-HardMS uses no learned temporal head",
            flush=True,
        )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        print(
            "[STAGE 4/4] NOW load waypoint files for B/C inference only",
            flush=True,
        )

        route_results = {}
        waypoint_counts = {}

        for route_name in (
            "route_B",
            "route_C",
        ):
            route = (
                load_mission_route_for_inference(
                    route_name,
                    visual.origin_lat,
                    visual.origin_lon,
                )
            )

            waypoint_counts[
                route_name
            ] = len(
                route.waypoints
            )

            cache = build_route_cache(
                route_name,
                catalog[route_name],
                visual,
                device,
            )

            csv_path = (
                config.OUTPUT_DIR
                / (
                    f"{route_name}_"
                    "pure_visual_lstm_frames.csv"
                )
            )

            summary, _ = (
                run_waypoint_inference(
                    None,
                    visual,
                    cache,
                    route,
                    device,
                    csv_path,
                    nominal_speed_mps,
                )
            )

            route_results[
                route_name
            ] = summary

            metric = summary[
                "FinalKalman"
            ]

            print(
                f"{route_name}: "
                f"MLE={metric['MLE_m']:.3f}m "
                f"P90={metric['P90_m']:.3f}m "
                f"RPE={metric['RPE_m']:.3f}m "
                f"Jump={metric['JumpRate_pct']:.3f}% "
                f"WaypointSwitches={summary['WaypointSwitchCount']}",
                flush=True,
            )

        payload = {
            "architecture": (
                ARCHITECTURE_NAME
            ),
            "training": {
                "route": "route_A",
                    "network_input_is_image_only": True,
                "waypoint_used": False,
                "coordinate_used_as_network_input": False,
                "gps_used_as_network_input": False,
                "gt_xy_role": (
                    "supervised target / candidate-label construction only"
                ),
                    "temporal_training": split,
            },
            "inference": {
                "routes": [
                    "route_B",
                    "route_C",
                ],
                "waypoint_used": True,
                "waypoint_role": (
                    "external SAT candidate-search direction/order only"
                ),
                "waypoint_frame_index_used": False,
                "waypoint_timestamp_used": False,
                "test_gt_gps_used_by_inference": False,
                "test_gt_role": (
                    "metrics and visualization only"
                ),
                "filter": (
                    "filterpy.kalman.KalmanFilter"
                ),
                "nominal_speed_mps": nominal_speed_mps,
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
