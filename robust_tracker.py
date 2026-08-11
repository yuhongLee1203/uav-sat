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
    train_visual_retrieval_a_only,
)
from visual_model import StableVisualInertialRNN

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "StableVisualInertialRNN_v14"


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
class CandidateSet:
    centers: torch.Tensor
    offsets: torch.Tensor
    z_sat: torch.Tensor
    raw_logits: torch.Tensor


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cache_dtype():
    return (
        torch.float16
        if config.FEATURE_CACHE_DTYPE == "float16"
        else torch.float32
    )


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def load_start_waypoint(route_name, origin_lat, origin_lon):
    path = Path(config.WAYPOINT_FILES[route_name])
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    waypoints = sorted(
        payload["waypoints"],
        key=lambda item: int(item["waypoint_order"]),
    )
    if not waypoints:
        raise RuntimeError("%s has no waypoints" % route_name)

    first = waypoints[0]
    x_m, y_m = meters_from_latlon(
        first["latitude"],
        first["longitude"],
        origin_lat,
        origin_lon,
    )
    return torch.tensor(
        [x_m, y_m],
        dtype=torch.float32,
    )


@torch.no_grad()
def build_route_cache(route_name, root, visual, device):
    stat = config.VISUAL_CHECKPOINT.stat()
    signature = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "architecture": ARCHITECTURE_NAME,
    }

    cache_path = (
        config.OUTPUT_DIR
        / "feature_cache"
        / (route_name + "_uav_clip.pt")
    )

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("signature") == signature:
            print(
                "%s: reuse UAV backbone cache" % route_name,
                flush=True,
            )
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
        items = [dataset[index] for index in range(start, end)]

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
                "%s backbone cache: %d/%d"
                % (route_name, end, len(dataset)),
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
def build_candidate_set(visual, uav_clip, center_xy):
    candidate = visual.candidate_batch(
        uav_clip,
        center_xy,
        grid_size=int(config.GRID_SIZE),
    )

    if int(candidate.centers.shape[1]) != int(config.CANDIDATE_COUNT):
        raise RuntimeError(
            "Expected %d SAT candidates but received %d"
            % (
                int(config.CANDIDATE_COUNT),
                int(candidate.centers.shape[1]),
            )
        )

    offsets = candidate.centers - center_xy[:, None, :]

    return CandidateSet(
        centers=candidate.centers,
        offsets=offsets,
        z_sat=candidate.z_sat,
        raw_logits=candidate.raw_logits,
    )


def candidate_target(candidate, gt_xy):
    distance = torch.linalg.norm(
        candidate.centers - gt_xy[:, None, :],
        dim=2,
    )
    nearest_distance, index = distance.min(dim=1)
    capture = (
        nearest_distance
        <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
    )
    return index, capture, nearest_distance


def teacher_center_ratio(epoch_index):
    epoch_number = int(epoch_index) + 1
    warmup = int(config.TEACHER_CENTER_WARMUP_EPOCHS)
    end = int(config.TEACHER_CENTER_END_EPOCH)

    if epoch_number <= warmup:
        return 1.0
    if epoch_number >= end:
        return 0.0

    fraction = (
        float(epoch_number - warmup)
        / float(max(1, end - warmup))
    )
    return float(max(0.0, 1.0 - fraction))


def clamp_step_xy(previous_xy, proposed_xy, maximum_m):
    delta = proposed_xy - previous_xy
    norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(
        float(maximum_m) / norm.clamp_min(1e-6),
        max=1.0,
    )
    return previous_xy + delta * scale


def straight_through_anchor(logits, centers, training):
    probability = torch.softmax(logits, dim=1)
    hard_index = logits.argmax(dim=1)
    hard = F.one_hot(
        hard_index,
        num_classes=logits.shape[1],
    ).to(probability.dtype)

    if training:
        weight = hard + probability - probability.detach()
    else:
        weight = hard

    anchor_xy = (
        weight.unsqueeze(-1) * centers
    ).sum(dim=1)
    return anchor_xy, probability, hard_index


def gt_motion_target(cache, index, device):
    if index + 1 >= len(cache):
        return torch.zeros(
            1, 2, dtype=torch.float32, device=device
        )

    delta = (
        cache.gt_xy[index + 1]
        - cache.gt_xy[index]
    ).to(device).reshape(1, 2)

    norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(
        float(config.MAX_STEP_M_PER_FRAME)
        / norm.clamp_min(1e-6),
        max=1.0,
    )
    return delta * scale


def stop_positive_weight(cache, train_end):
    if int(train_end) <= 1:
        return 1.0

    delta = (
        cache.gt_xy[1:int(train_end)]
        - cache.gt_xy[: int(train_end) - 1]
    )
    speed = torch.linalg.norm(delta, dim=1)
    stop = speed <= float(config.STOP_STEP_THRESHOLD_M)

    stop_count = int(stop.sum())
    moving_count = int(stop.numel() - stop_count)

    if stop_count <= 0:
        return 1.0

    return float(
        np.clip(
            moving_count / float(stop_count),
            1.0,
            float(config.STOP_POS_WEIGHT_MAX),
        )
    )


def training_losses(
    output,
    measurement_xy,
    gt_xy,
    target_motion,
    previous_pred_motion,
    previous_target_motion,
    candidate,
    target_index,
    capture,
    stop_pos_weight,
):
    position_loss = F.smooth_l1_loss(
        measurement_xy,
        gt_xy,
    )

    motion_loss = F.smooth_l1_loss(
        output.motion_xy,
        target_motion,
    )

    target_stop = (
        torch.linalg.norm(target_motion, dim=1, keepdim=True)
        <= float(config.STOP_STEP_THRESHOLD_M)
    ).float()

    stop_loss = F.binary_cross_entropy_with_logits(
        output.stop_logit,
        target_stop,
        pos_weight=torch.tensor(
            [float(stop_pos_weight)],
            dtype=output.stop_logit.dtype,
            device=output.stop_logit.device,
        ),
    )

    pred_acceleration = (
        output.motion_xy - previous_pred_motion
    )
    target_acceleration = (
        target_motion - previous_target_motion
    )
    acceleration_loss = F.smooth_l1_loss(
        pred_acceleration,
        target_acceleration,
    )

    if bool(capture[0]):
        candidate_loss = F.cross_entropy(
            output.refined_logits,
            target_index.to(output.refined_logits.device),
        )
    else:
        candidate_loss = measurement_xy.sum() * 0.0

    residual_loss = output.residual_xy.abs().mean()

    variance = output.measurement_variance.clamp_min(
        float(config.KALMAN_R_MIN_VAR)
    )
    error = measurement_xy - gt_xy
    nll = 0.5 * (
        error.square() / variance
        + variance.log()
    )
    variance_loss = nll.mean()

    total = (
        float(config.LOSS_POSITION) * position_loss
        + float(config.LOSS_MOTION) * motion_loss
        + float(config.LOSS_STOP) * stop_loss
        + float(config.LOSS_ACCELERATION) * acceleration_loss
        + float(config.LOSS_CANDIDATE_CE) * candidate_loss
        + float(config.LOSS_RESIDUAL) * residual_loss
        + float(config.LOSS_VARIANCE_NLL) * variance_loss
    )

    return {
        "total": total,
        "position": position_loss,
        "motion": motion_loss,
        "stop": stop_loss,
        "acceleration": acceleration_loss,
        "candidate": candidate_loss,
        "residual": residual_loss,
    }


def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    start_xy,
    train_end,
    epoch_index,
    stop_pos_weight,
    device,
):
    model.train()

    teacher_ratio = teacher_center_ratio(epoch_index)

    hidden = None
    previous_uav_state = None
    previous_score_state = None
    previous_motion = torch.zeros(
        1, 2, dtype=torch.float32, device=device
    )
    previous_target_motion = torch.zeros_like(previous_motion)

    predicted_position = start_xy.to(device).reshape(1, 2)

    pending_loss = None
    pending_count = 0

    records = []
    capture_rows = []

    optimizer.zero_grad(set_to_none=True)

    for index in range(int(train_end)):
        if index == 0:
            center_xy = start_xy.to(device).reshape(1, 2)
        else:
            teacher_center = cache.gt_xy[index - 1].to(
                device
            ).reshape(1, 2)

            if float(config.TEACHER_CENTER_JITTER_M) > 0.0:
                teacher_center = teacher_center + (
                    torch.randn_like(teacher_center)
                    * float(config.TEACHER_CENTER_JITTER_M)
                )

            center_xy = (
                float(teacher_ratio) * teacher_center
                + (1.0 - float(teacher_ratio))
                * predicted_position.detach()
            )

        uav_clip = cache.uav_clip[
            index : index + 1
        ].to(device).float()

        candidate = build_candidate_set(
            visual,
            uav_clip,
            center_xy,
        )

        z_uav = visual.model.encode_uav_from_clip(
            uav_clip
        )

        output = model.forward_step(
            z_uav=z_uav,
            z_sat=candidate.z_sat,
            raw_logits=candidate.raw_logits,
            candidate_offsets_xy=candidate.offsets,
            previous_motion_xy=previous_motion,
            hidden=hidden,
            previous_uav_state=previous_uav_state,
            previous_score_state=previous_score_state,
        )

        anchor_xy, _, _ = straight_through_anchor(
            output.refined_logits,
            candidate.centers,
            training=True,
        )

        proposed_measurement = (
            anchor_xy + output.residual_xy
        )

        measurement_xy = clamp_step_xy(
            previous_xy=predicted_position,
            proposed_xy=proposed_measurement,
            maximum_m=float(config.MAX_STEP_M_PER_FRAME),
        )

        gt_xy = cache.gt_xy[
            index : index + 1
        ].to(device)

        target_index, capture, nearest_distance = candidate_target(
            candidate,
            gt_xy,
        )

        target_motion = gt_motion_target(
            cache,
            index,
            device,
        )

        losses = training_losses(
            output=output,
            measurement_xy=measurement_xy,
            gt_xy=gt_xy,
            target_motion=target_motion,
            previous_pred_motion=previous_motion,
            previous_target_motion=previous_target_motion,
            candidate=candidate,
            target_index=target_index,
            capture=capture,
            stop_pos_weight=stop_pos_weight,
        )

        pending_loss = (
            losses["total"]
            if pending_loss is None
            else pending_loss + losses["total"]
        )
        pending_count += 1

        records.append(
            [
                float(losses[key].detach().cpu())
                for key in [
                    "total",
                    "position",
                    "motion",
                    "stop",
                    "acceleration",
                    "candidate",
                    "residual",
                ]
            ]
            + [
                float(
                    torch.linalg.norm(
                        output.motion_xy,
                        dim=1,
                    ).mean().detach().cpu()
                ),
                float(
                    output.stop_probability.mean().detach().cpu()
                ),
                float(
                    nearest_distance.mean().detach().cpu()
                ),
            ]
        )
        capture_rows.append(
            float(capture.float().mean().cpu())
        )

        hidden = output.hidden
        previous_uav_state = output.uav_state
        previous_score_state = output.score_state

        previous_motion = output.motion_xy
        previous_target_motion = target_motion
        predicted_position = measurement_xy

        boundary = (
            pending_count >= int(config.TBPTT_STEPS)
            or index == int(train_end) - 1
        )

        if boundary:
            objective = pending_loss / float(pending_count)

            if not torch.isfinite(objective):
                raise FloatingPointError(
                    "non-finite temporal loss"
                )

            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.GRAD_CLIP_NORM),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            hidden = hidden.detach()
            previous_uav_state = previous_uav_state.detach()
            previous_score_state = previous_score_state.detach()
            previous_motion = previous_motion.detach()
            previous_target_motion = previous_target_motion.detach()
            predicted_position = predicted_position.detach()

            pending_loss = None
            pending_count = 0

    values = np.asarray(records, dtype=np.float64)

    return {
        "teacher": float(teacher_ratio),
        "loss": float(values[:, 0].mean()),
        "position": float(values[:, 1].mean()),
        "motion": float(values[:, 2].mean()),
        "stop": float(values[:, 3].mean()),
        "acceleration": float(values[:, 4].mean()),
        "candidate": float(values[:, 5].mean()),
        "residual": float(values[:, 6].mean()),
        "motion_mag": float(values[:, 7].mean()),
        "stop_probability": float(values[:, 8].mean()),
        "candidate_distance": float(values[:, 9].mean()),
        "capture_pct": float(100.0 * np.mean(capture_rows)),
    }


class PositionKalman2D:
    def __init__(self, initial_xy):
        if KalmanFilter is None:
            raise ImportError(
                "FilterPy is required: pip install filterpy"
            )

        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        self.kf.x = np.asarray(
            initial_xy,
            dtype=np.float64,
        ).reshape(2)
        self.kf.F = np.eye(2, dtype=np.float64)
        self.kf.H = np.eye(2, dtype=np.float64)
        self.kf.P = (
            np.eye(2, dtype=np.float64)
            * float(config.KALMAN_INIT_VAR)
        )
        self.kf.Q = (
            np.eye(2, dtype=np.float64)
            * float(config.KALMAN_Q_VAR)
        )
        self.kf.R = np.eye(2, dtype=np.float64) * 2.0

    def update(self, measurement_xy, variance_xy):
        previous = np.asarray(
            self.kf.x,
            dtype=np.float64,
        ).reshape(2).copy()

        self.kf.predict()

        variance = np.asarray(
            variance_xy,
            dtype=np.float64,
        ).reshape(2)
        variance = np.clip(
            variance,
            float(config.KALMAN_R_MIN_VAR),
            float(config.KALMAN_R_MAX_VAR),
        )
        self.kf.R = np.diag(variance)

        self.kf.update(
            np.asarray(
                measurement_xy,
                dtype=np.float64,
            ).reshape(2)
        )

        updated = np.asarray(
            self.kf.x,
            dtype=np.float64,
        ).reshape(2)

        delta = updated - previous
        norm = float(np.linalg.norm(delta))
        maximum = float(config.MAX_STEP_M_PER_FRAME)

        if norm > maximum:
            updated = (
                previous
                + delta
                * (maximum / max(norm, 1e-9))
            )
            self.kf.x = updated.copy()

        return updated.copy()


@torch.no_grad()
def closed_loop_rollout(
    model,
    visual,
    cache,
    start_xy,
    device,
    collect_rows=False,
    use_kalman=True,
):
    model.eval()

    hidden = None
    previous_uav_state = None
    previous_score_state = None
    previous_motion = torch.zeros(
        1, 2, dtype=torch.float32, device=device
    )

    current_position = start_xy.to(
        device
    ).reshape(1, 2)

    if use_kalman:
        kalman = PositionKalman2D(
            current_position[0].cpu().numpy()
        )
    else:
        kalman = None

    visual_predictions = []
    final_predictions = []
    rows = []

    last_valid_heading_deg = float("nan")

    for index in range(len(cache)):
        # Stable search center:
        # always the previous final visual/Kalman position.
        # RNN motion is a SOFT candidate prior, never an autonomous center jump.
        center_xy = current_position

        uav_clip = cache.uav_clip[
            index : index + 1
        ].to(device).float()

        candidate = build_candidate_set(
            visual,
            uav_clip,
            center_xy,
        )
        z_uav = visual.model.encode_uav_from_clip(
            uav_clip
        )

        output = model.forward_step(
            z_uav=z_uav,
            z_sat=candidate.z_sat,
            raw_logits=candidate.raw_logits,
            candidate_offsets_xy=candidate.offsets,
            previous_motion_xy=previous_motion,
            hidden=hidden,
            previous_uav_state=previous_uav_state,
            previous_score_state=previous_score_state,
        )

        anchor_xy, probability, hard_index = straight_through_anchor(
            output.refined_logits,
            candidate.centers,
            training=False,
        )

        proposed_measurement = (
            anchor_xy + output.residual_xy
        )

        visual_xy = clamp_step_xy(
            previous_xy=current_position,
            proposed_xy=proposed_measurement,
            maximum_m=float(config.MAX_STEP_M_PER_FRAME),
        )

        visual_np = visual_xy[
            0
        ].cpu().numpy()

        if use_kalman:
            final_np = kalman.update(
                visual_np,
                output.measurement_variance[
                    0
                ].cpu().numpy(),
            )
            current_position = torch.tensor(
                final_np,
                dtype=torch.float32,
                device=device,
            ).reshape(1, 2)
        else:
            final_np = visual_np.copy()
            current_position = visual_xy

        motion_np = output.motion_xy[
            0
        ].cpu().numpy()
        motion_mag = float(
            np.linalg.norm(motion_np)
        )

        if motion_mag >= float(config.HEADING_MIN_MOTION_M):
            estimated_heading_deg = math.degrees(
                math.atan2(
                    float(motion_np[1]),
                    float(motion_np[0]),
                )
            )
            last_valid_heading_deg = estimated_heading_deg
            heading_valid = 1
        else:
            estimated_heading_deg = last_valid_heading_deg
            heading_valid = 0

        probability_cpu = probability[
            0
        ].cpu()
        top2 = probability_cpu.topk(
            k=min(2, int(probability_cpu.shape[0]))
        ).values
        probability_margin = float(
            top2[0]
            - (
                top2[1]
                if top2.shape[0] > 1
                else 0.0
            )
        )

        selected_index = int(
            hard_index[0].cpu()
        )
        selected_patch = candidate.centers[
            0, selected_index
        ].cpu().numpy()

        # GT is read only after the complete inference prediction.
        gt_np = cache.gt_xy[
            index
        ].numpy()

        visual_predictions.append(
            visual_np
        )
        final_predictions.append(
            final_np
        )

        row = {
            "sequence_index": int(index),
            "frame_id": int(cache.frame_ids[index]),
            "image_path": cache.image_paths[index],
            "gt_x": float(gt_np[0]),
            "gt_y": float(gt_np[1]),
            "visual_x": float(visual_np[0]),
            "visual_y": float(visual_np[1]),
            "final_x": float(final_np[0]),
            "final_y": float(final_np[1]),
            "selected_patch_x": float(selected_patch[0]),
            "selected_patch_y": float(selected_patch[1]),
            "rnn_motion_dx_m": float(motion_np[0]),
            "rnn_motion_dy_m": float(motion_np[1]),
            "rnn_motion_magnitude_m": float(motion_mag),
            "rnn_stop_probability": float(
                output.stop_probability[0, 0].cpu()
            ),
            "rnn_residual_x_m": float(
                output.residual_xy[0, 0].cpu()
            ),
            "rnn_residual_y_m": float(
                output.residual_xy[0, 1].cpu()
            ),
            "estimated_heading_deg_enu": (
                float(estimated_heading_deg)
                if np.isfinite(estimated_heading_deg)
                else float("nan")
            ),
            "heading_valid": int(heading_valid),
            "candidate_probability_max": float(
                probability_cpu.max()
            ),
            "candidate_probability_margin": float(
                probability_margin
            ),
            "measurement_variance_x": float(
                output.measurement_variance[0, 0].cpu()
            ),
            "measurement_variance_y": float(
                output.measurement_variance[0, 1].cpu()
            ),
            "candidate_count": int(
                candidate.centers.shape[1]
            ),
            "error_visual_m": float(
                np.linalg.norm(visual_np - gt_np)
            ),
            "error_final_m": float(
                np.linalg.norm(final_np - gt_np)
            ),
        }

        if collect_rows:
            rows.append(row)

        hidden = output.hidden
        previous_uav_state = output.uav_state
        previous_score_state = output.score_state
        previous_motion = output.motion_xy

    return (
        np.asarray(visual_predictions, dtype=np.float64),
        np.asarray(final_predictions, dtype=np.float64),
        rows,
    )


@torch.no_grad()
def evaluate_validation(
    model,
    visual,
    cache,
    start_xy,
    val_start,
    device,
):
    _, final_pred, _ = closed_loop_rollout(
        model=model,
        visual=visual,
        cache=cache,
        start_xy=start_xy,
        device=device,
        collect_rows=False,
        use_kalman=True,
    )

    gt = cache.gt_xy.numpy()

    error = np.linalg.norm(
        final_pred[int(val_start) :]
        - gt[int(val_start) :],
        axis=1,
    )

    return {
        "mle": float(np.mean(error)),
        "p90": float(np.quantile(error, 0.90)),
        "lsr15": float(
            100.0 * np.mean(error <= 15.0)
        ),
    }


def train_temporal_model(
    model,
    visual,
    cache,
    start_xy,
    device,
    epochs,
):
    train_end = max(
        8,
        int(
            len(cache)
            * float(config.TEMPORAL_TRAIN_FRACTION)
        ),
    )
    val_start = train_end

    stop_pos_weight = stop_positive_weight(
        cache,
        train_end,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    best_score = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("TEMPORAL v14: Route-A only", flush=True)
    print("  RNN               = nn.RNNCell", flush=True)
    print("  sensor inputs     = UAV/SAT images only", flush=True)
    print("  previous state    = image-derived hidden + motion", flush=True)
    print("  search            = FULL 6x6 = 36 patches", flush=True)
    print(
        "  forward inertia   = learned SOFT candidate prior; never masks rear patches",
        flush=True,
    )
    print(
        "  max motion/output = %.1f m/frame; 0 allowed"
        % float(config.MAX_STEP_M_PER_FRAME),
        flush=True,
    )
    print("  current GT input  = NEVER", flush=True)
    print(
        "  teacher centering = training-only scheduled sampling",
        flush=True,
    )
    print(
        "  validation        = 100%% closed-loop Route-A held-out tail",
        flush=True,
    )
    print(
        "  stop pos_weight   = %.3f"
        % float(stop_pos_weight),
        flush=True,
    )

    for epoch in range(int(epochs)):
        training = train_one_epoch(
            model=model,
            optimizer=optimizer,
            visual=visual,
            cache=cache,
            start_xy=start_xy,
            train_end=train_end,
            epoch_index=epoch,
            stop_pos_weight=stop_pos_weight,
            device=device,
        )

        validation = evaluate_validation(
            model=model,
            visual=visual,
            cache=cache,
            start_xy=start_xy,
            val_start=val_start,
            device=device,
        )

        score = float(validation["mle"])

        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        payload = {
            "architecture": ARCHITECTURE_NAME,
            "model": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "best_model": best_state,
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_mle": best_score,
            "temporal_train_routes": ["route_A"],
            "temporal_validation_routes": ["route_A"],
            "temporal_eval_routes": ["route_B", "route_C"],
            "current_gt_as_model_input": False,
            "previous_gt_as_model_input": False,
            "test_gt_as_model_input": False,
            "test_waypoint_frame_index_used": False,
            "rnn_type": "nn.RNNCell",
            "search_grid": "full_6x6",
            "candidate_count": int(config.CANDIDATE_COUNT),
            "soft_forward_prior": True,
            "max_step_m_per_frame": float(
                config.MAX_STEP_M_PER_FRAME
            ),
            "teacher_center_ratio": float(
                training["teacher"]
            ),
        }

        torch.save(
            payload,
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            "epoch=%03d/%d loss=%.4f pos=%.4f motion=%.4f "
            "stop=%.4f accel=%.4f cand=%.4f residual=%.4f "
            "teacher=%.2f cap=%.1f%% candDist=%.2fm "
            "motionMag=%.3fm stopP=%.3f "
            "valMLE=%.3fm valP90=%.3fm valLSR15=%.2f%% "
            "best=%03d@%.3fm patience=%d"
            % (
                epoch + 1,
                int(epochs),
                training["loss"],
                training["position"],
                training["motion"],
                training["stop"],
                training["acceleration"],
                training["candidate"],
                training["residual"],
                training["teacher"],
                training["capture_pct"],
                training["candidate_distance"],
                training["motion_mag"],
                training["stop_probability"],
                validation["mle"],
                validation["p90"],
                validation["lsr15"],
                best_epoch,
                best_score,
                patience,
            ),
            flush=True,
        )
        print(
            "checkpoint:",
            config.TEMPORAL_CHECKPOINT,
            flush=True,
        )

        # Avoid stopping while the search-center curriculum is still active.
        fully_closed_loop = (
            training["teacher"] <= 1e-6
        )
        if (
            fully_closed_loop
            and epoch + 1
            >= int(config.TEMPORAL_MIN_EPOCHS_BEFORE_STOP)
            and patience
            >= int(config.TEMPORAL_EARLY_STOPPING_PATIENCE)
        ):
            print(
                "temporal early stopping: closed-loop validation stopped improving",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Temporal training did not produce a best checkpoint"
        )

    final_payload = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )
    final_payload["model"] = best_state
    final_payload["best_model"] = best_state
    final_payload["best_epoch"] = best_epoch
    final_payload["best_val_mle"] = best_score

    torch.save(
        final_payload,
        config.TEMPORAL_CHECKPOINT,
    )

    model.load_state_dict(best_state)

    print(
        "best temporal checkpoint: epoch=%d valMLE=%.3fm"
        % (best_epoch, best_score),
        flush=True,
    )


def load_temporal_model(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Temporal checkpoint not found: %s"
            % config.TEMPORAL_CHECKPOINT
        )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )

    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch: %r"
            % checkpoint.get("architecture")
        )

    if checkpoint.get("current_gt_as_model_input") is not False:
        raise RuntimeError(
            "Checkpoint does not prove GT is excluded from RNN inputs"
        )

    state = checkpoint.get("best_model") or checkpoint["model"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(
        "loaded best checkpoint epoch=%s valMLE=%s"
        % (
            checkpoint.get("best_epoch"),
            checkpoint.get("best_val_mle"),
        ),
        flush=True,
    )
    return checkpoint


def metric_block(prediction, gt):
    error = np.linalg.norm(
        prediction - gt,
        axis=1,
    )

    if len(prediction) > 1:
        pred_step = np.linalg.norm(
            np.diff(prediction, axis=0),
            axis=1,
        )
        gt_step = np.linalg.norm(
            np.diff(gt, axis=0),
            axis=1,
        )
        rpe = np.abs(pred_step - gt_step)
        jump = np.mean(
            pred_step
            > (
                gt_step
                + float(config.JUMP_TOLERANCE_M)
            )
        )
    else:
        rpe = np.asarray([0.0])
        jump = 0.0

    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.quantile(error, 0.90)),
        "P95_m": float(np.quantile(error, 0.95)),
        "LSR@5_pct": float(
            100.0 * np.mean(error <= 5.0)
        ),
        "LSR@10_pct": float(
            100.0 * np.mean(error <= 10.0)
        ),
        "LSR@15_pct": float(
            100.0 * np.mean(error <= 15.0)
        ),
        "LSR@20_pct": float(
            100.0 * np.mean(error <= 20.0)
        ),
        "RPE_step_mean_m": float(np.mean(rpe)),
        "JumpRate_pct": float(100.0 * jump),
        "MaxError_m": float(error.max()),
    }


def write_rows(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No inference rows to write"
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_inference(
    model,
    visual,
    cache,
    start_xy,
    device,
    csv_path,
):
    visual_pred, final_pred, rows = closed_loop_rollout(
        model=model,
        visual=visual,
        cache=cache,
        start_xy=start_xy,
        device=device,
        collect_rows=True,
        use_kalman=True,
    )

    write_rows(csv_path, rows)

    gt = cache.gt_xy.numpy()

    summary = {
        "architecture": ARCHITECTURE_NAME,
        "network": "nn.RNNCell",
        "sensor_input": "UAV/SAT images only",
        "previous_state": "image-derived RNN hidden + motion",
        "current_gt_as_model_input": False,
        "previous_gt_as_model_input": False,
        "test_gt_as_model_input": False,
        "test_waypoint_frame_index_used": False,
        "search_grid": "full_6x6",
        "candidate_count": int(config.CANDIDATE_COUNT),
        "soft_forward_prior": True,
        "max_step_m_per_frame": float(
            config.MAX_STEP_M_PER_FRAME
        ),
        "VisualMeasurement": metric_block(
            visual_pred,
            gt,
        ),
        "FinalPositionKalman": metric_block(
            final_pred,
            gt,
        ),
    }

    return summary, rows


def route_catalog():
    return {
        name: Path(root)
        for name, root in zip(
            config.ROUTE_NAMES,
            config.ROUTE_ROOTS,
        )
    }


def ensure_visual_checkpoint(
    device,
    visual_epochs,
    reuse_visual,
):
    if (
        reuse_visual
        and config.VISUAL_CHECKPOINT.exists()
    ):
        print(
            "reuse visual checkpoint:",
            config.VISUAL_CHECKPOINT,
            flush=True,
        )
        return

    print(
        "training Route-A visual retrieval from scratch",
        flush=True,
    )
    train_visual_retrieval_a_only(
        device=device,
        epochs=int(visual_epochs),
        jitter_m=float(config.LOCAL_PRIOR_JITTER_M),
        resume=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval"],
        default="train_eval",
    )
    parser.add_argument(
        "--visual-epochs",
        type=int,
        default=int(config.VISUAL_EPOCHS),
    )
    parser.add_argument(
        "--temporal-epochs",
        type=int,
        default=int(config.TEMPORAL_EPOCHS),
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
    print("STABLE VISUAL-INERTIAL RNN v14", flush=True)
    print("Plain nn.RNNCell; NO LSTM/GRU.", flush=True)
    print(
        "Full 6x6 visual search. Previous RNN motion is a SOFT prior only.",
        flush=True,
    )
    print(
        "Current coordinate is anchored by CURRENT image retrieval; RNN motion cannot drive XY by itself.",
        flush=True,
    )
    print(
        "Maximum position/motion change = %.1f m/frame; zero is allowed."
        % float(config.MAX_STEP_M_PER_FRAME),
        flush=True,
    )
    print(
        "Final filter = position-only [x,y] Kalman; no velocity state.",
        flush=True,
    )
    print(
        "GT is supervision/evaluation only, never RNN inference input.",
        flush=True,
    )
    print("=" * 100, flush=True)

    if args.mode in ("train", "train_eval"):
        ensure_visual_checkpoint(
            device=device,
            visual_epochs=args.visual_epochs,
            reuse_visual=bool(args.reuse_visual),
        )
    elif not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "eval requires visual checkpoint: %s"
            % config.VISUAL_CHECKPOINT
        )

    visual = FrozenVisualLocalizer(device)
    routes = route_catalog()

    if args.mode in ("train", "train_eval"):
        route_name = "route_A"

        cache = build_route_cache(
            route_name,
            routes[route_name],
            visual,
            device,
        )
        start_xy = load_start_waypoint(
            route_name,
            visual.origin_lat,
            visual.origin_lon,
        )

        model = StableVisualInertialRNN().to(device)

        train_temporal_model(
            model=model,
            visual=visual,
            cache=cache,
            start_xy=start_xy,
            device=device,
            epochs=int(args.temporal_epochs),
        )

    if args.mode in ("eval", "train_eval"):
        model = StableVisualInertialRNN().to(device)
        load_temporal_model(model, device)

        results = {}

        for route_name in ["route_B", "route_C"]:
            cache = build_route_cache(
                route_name,
                routes[route_name],
                visual,
                device,
            )
            start_xy = load_start_waypoint(
                route_name,
                visual.origin_lat,
                visual.origin_lon,
            )

            csv_path = (
                config.OUTPUT_DIR
                / (
                    route_name
                    + "_stable_visual_inertial_rnn_frames.csv"
                )
            )

            summary, _ = run_inference(
                model=model,
                visual=visual,
                cache=cache,
                start_xy=start_xy,
                device=device,
                csv_path=csv_path,
            )
            results[route_name] = summary

            metric = summary["FinalPositionKalman"]
            print(
                "%s Final: MLE=%.3fm P90=%.3fm LSR15=%.2f%% Jump=%.2f%%"
                % (
                    route_name,
                    metric["MLE_m"],
                    metric["P90_m"],
                    metric["LSR@15_pct"],
                    metric["JumpRate_pct"],
                ),
                flush=True,
            )

        summary_path = (
            config.OUTPUT_DIR
            / "robust_tracker_summary.json"
        )
        summary_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print("summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
