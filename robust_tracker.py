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
from visual_model import WaypointConditionedGRU


ARCHITECTURE_NAME = "WaypointRouteFrameGRUKalman_v21"


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
class RouteFrame:
    leg_index: int
    start_xy: np.ndarray
    end_xy: np.ndarray
    unit: np.ndarray
    cross: np.ndarray
    length_m: float
    along_m: float
    cross_m: float
    remaining_m: float
    progress: float


class WaypointRoute:
    """Monotonic waypoint route using coordinates only.

    Waypoint frame_index/timestamp is intentionally never used for inference.
    A leg changes only when estimated along-track progress crosses its endpoint.
    """

    def __init__(self, points_xy):
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 2:
            raise ValueError("At least start + one waypoint are required")
        self.points = points

    def _geometry(self, leg_index):
        leg_index = int(np.clip(leg_index, 0, len(self.points) - 2))
        start = self.points[leg_index]
        end = self.points[leg_index + 1]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < float(config.WAYPOINT_MIN_LEG_LENGTH_M):
            unit = np.asarray([1.0, 0.0], dtype=np.float64)
            length = max(length, 1e-6)
        else:
            unit = delta / length
        cross = np.asarray([-unit[1], unit[0]], dtype=np.float64)
        return start, end, unit, cross, length

    def frame(self, position_xy, leg_index):
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        start, end, unit, cross, length = self._geometry(leg_index)
        rel = position - start
        along = float(np.dot(rel, unit))
        cross_m = float(np.dot(rel, cross))
        progress = float(np.clip(along / max(length, 1e-6), 0.0, 1.0))
        remaining = float(max(length - along, 0.0))
        return RouteFrame(
            leg_index=int(leg_index),
            start_xy=start.copy(),
            end_xy=end.copy(),
            unit=unit.copy(),
            cross=cross.copy(),
            length_m=length,
            along_m=along,
            cross_m=cross_m,
            remaining_m=remaining,
            progress=progress,
        )

    def advance(self, position_xy, leg_index):
        leg = int(np.clip(leg_index, 0, len(self.points) - 2))
        while leg < len(self.points) - 2:
            frame = self.frame(position_xy, leg)
            if frame.along_m < frame.length_m:
                break
            leg += 1
        return leg

    def closest_leg(self, position_xy):
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        best_leg = 0
        best_distance = float("inf")
        for leg in range(len(self.points) - 1):
            start, _, unit, _, length = self._geometry(leg)
            along = float(np.clip(np.dot(position - start, unit), 0.0, length))
            nearest = start + along * unit
            distance = float(np.linalg.norm(position - nearest))
            if distance < best_distance:
                best_distance = distance
                best_leg = leg
        return best_leg


class PolynomialKalman2D:
    """External Kalman filter with learned second-order motion prediction.

    State is [x, y, vx, vy]. The GRU supplies current velocity and acceleration.
    Position prediction is explicitly p^- = p + v + 0.5 a.
    """

    def __init__(self, initial_xy):
        self.x = np.asarray(
            [float(initial_xy[0]), float(initial_xy[1]), 0.0, 0.0],
            dtype=np.float64,
        )
        self.P = np.diag(
            [
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
            ]
        ).astype(np.float64)
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
        ).astype(np.float64)

    @staticmethod
    def _bound(vector, maximum):
        vector = np.asarray(vector, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(vector))
        if norm <= maximum or norm <= 1e-9:
            return vector
        return vector * (float(maximum) / norm)

    def position(self):
        return self.x[:2].copy()

    def velocity(self):
        return self.x[2:4].copy()

    def predict(self, velocity_xy, acceleration_xy):
        velocity = self._bound(
            velocity_xy, float(config.MAX_FINAL_SPEED_M_PER_FRAME)
        )
        acceleration = np.asarray(acceleration_xy, dtype=np.float64).reshape(2)
        step = velocity + 0.5 * acceleration
        step = self._bound(step, float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME))
        next_velocity = self._bound(
            velocity + acceleration, float(config.MAX_FINAL_SPEED_M_PER_FRAME)
        )

        previous_position = self.x[:2].copy()
        self.x[:2] = previous_position + step
        self.x[2:4] = next_velocity
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.position()

    def update(self, measurement_xy, variance_xy):
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        variance = np.asarray(variance_xy, dtype=np.float64).reshape(2)
        variance = np.clip(
            variance,
            float(config.KALMAN_R_MIN_VAR),
            float(config.KALMAN_R_MAX_VAR),
        )
        R = np.diag(variance)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ innovation
        I = np.eye(4, dtype=np.float64)
        # Joseph form is numerically safer than P=(I-KH)P.
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return self.position()


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cache_dtype():
    if str(config.FEATURE_CACHE_DTYPE).lower() == "float16":
        return torch.float16
    return torch.float32


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def load_waypoint_xy(route_name, origin_lat, origin_lon):
    path = Path(config.WAYPOINT_FILES[route_name])
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    waypoints = sorted(
        payload["waypoints"], key=lambda item: int(item["waypoint_order"])
    )
    rows = []
    for waypoint in waypoints:
        x_m, y_m = meters_from_latlon(
            waypoint["latitude"],
            waypoint["longitude"],
            origin_lat,
            origin_lon,
        )
        rows.append([float(x_m), float(y_m)])
    if len(rows) < 2:
        raise RuntimeError("%s needs at least two waypoints" % route_name)
    return np.asarray(rows, dtype=np.float64)


@torch.no_grad()
def build_route_cache(route_name, root, visual, device):
    stat = config.VISUAL_CHECKPOINT.stat()
    signature = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "architecture": ARCHITECTURE_NAME,
    }
    cache_path = config.OUTPUT_DIR / "feature_cache" / (route_name + "_uav_clip.pt")
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("signature") == signature:
            print("%s: reuse UAV backbone cache" % route_name, flush=True)
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
        uav = torch.stack([item["uav"] for item in items]).to(device)
        clip = visual.encode_uav_clip(uav)
        clip_rows.append(clip.detach().cpu().to(cache_dtype()))
        gt_rows.append(torch.stack([item["xy"].float() for item in items]))
        for item in items:
            frame_rows.append(parse_frame_id(item["frame_id"]))
            image_paths.append(str(item["image_path"]))
        if start == 0 or end == len(dataset) or (start // batch_size) % 10 == 0:
            print("%s backbone cache: %d/%d" % (route_name, end, len(dataset)), flush=True)

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


def tensor2(value, device):
    return torch.tensor(
        np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device
    ).reshape(1, 2)


def route_tensors(frame, device):
    unit = tensor2(frame.unit, device)
    cross = tensor2(frame.cross, device)
    remaining = torch.tensor(
        [[float(frame.remaining_m)]], dtype=torch.float32, device=device
    )
    cross_track = torch.tensor(
        [[float(frame.cross_m)]], dtype=torch.float32, device=device
    )
    progress = torch.tensor(
        [[float(frame.progress)]], dtype=torch.float32, device=device
    )
    return unit, cross, remaining, cross_track, progress


def project_route_torch(vector_xy, route_unit, cross_unit):
    parallel = (vector_xy * route_unit).sum(dim=1, keepdim=True)
    cross = (vector_xy * cross_unit).sum(dim=1, keepdim=True)
    return torch.cat([parallel, cross], dim=1)


def target_motion(cache, index, route_unit, cross_unit, device):
    current = cache.gt_xy[index].to(device).reshape(1, 2)
    if index <= 0:
        previous_step = cache.gt_xy[min(1, len(cache) - 1)].to(device).reshape(1, 2) - current
    else:
        previous_step = current - cache.gt_xy[index - 1].to(device).reshape(1, 2)

    if index + 1 >= len(cache):
        next_step = previous_step.detach()
    else:
        next_step = cache.gt_xy[index + 1].to(device).reshape(1, 2) - current

    def clip_step(step, maximum):
        norm = torch.linalg.norm(step, dim=1, keepdim=True)
        scale = torch.clamp(float(maximum) / norm.clamp_min(1e-6), max=1.0)
        return step * scale

    previous_step = clip_step(previous_step, config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
    next_step = clip_step(next_step, config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
    velocity_xy = 0.5 * (previous_step + next_step)
    acceleration_xy = next_step - previous_step
    velocity_route = project_route_torch(velocity_xy, route_unit, cross_unit)
    acceleration_route = project_route_torch(
        acceleration_xy, route_unit, cross_unit
    )
    next_step_route = project_route_torch(next_step, route_unit, cross_unit)
    return velocity_route, acceleration_route, next_step_route, next_step


def teacher_ratio_for_epoch(epoch):
    if epoch <= int(config.MOTION_WARMUP_EPOCHS):
        return 1.0
    elapsed = max(0, epoch - int(config.MOTION_WARMUP_EPOCHS))
    fraction = min(1.0, elapsed / max(float(config.TEACHER_DECAY_EPOCHS), 1.0))
    return 1.0 + fraction * (float(config.TEACHER_RATIO_FINAL) - 1.0)


def random_jitter(maximum_m):
    maximum = float(maximum_m)
    if maximum <= 0:
        return np.zeros(2, dtype=np.float64)
    radius = math.sqrt(random.random()) * maximum
    angle = random.random() * 2.0 * math.pi
    return np.asarray([radius * math.cos(angle), radius * math.sin(angle)])


def visual_observation(visual, uav_clip, center_xy, gt_xy=None):
    candidate = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=center_xy,
        grid_size=int(config.GRID_SIZE),
    )
    sat_context = (
        candidate.raw_prob.unsqueeze(-1) * candidate.z_sat
    ).sum(dim=1)
    capture = None
    if gt_xy is not None:
        capture = visual.candidate_contains_gt_anchor(candidate.indices, gt_xy)
    return candidate, sat_context, capture


def forward_temporal(
    model,
    candidate,
    sat_context,
    search_center_xy,
    previous_final_xy,
    route_frame,
    previous_velocity_route,
    previous_acceleration_route,
    hidden,
    device,
):
    unit, cross, remaining, cross_track, progress = route_tensors(
        route_frame, device
    )
    output = model.forward_step(
        z_uav=candidate.z_uav,
        sat_context=sat_context,
        raw_probability=candidate.raw_prob,
        hardms_xy=candidate.hardms_xy,
        raw_top1_xy=candidate.raw_top1_xy,
        hardms_support=candidate.hardms_support,
        search_center_xy=search_center_xy,
        previous_final_xy=previous_final_xy,
        route_unit=unit,
        cross_unit=cross,
        route_remaining_m=remaining,
        route_cross_track_m=cross_track,
        route_progress=progress,
        previous_velocity_route=previous_velocity_route,
        previous_acceleration_route=previous_acceleration_route,
        hidden=hidden,
    )
    return output, unit, cross


def temporal_loss(
    output,
    candidate,
    gt_xy,
    capture,
    target_velocity_route,
    target_acceleration_route,
    target_next_step_route,
    route_unit,
    cross_unit,
):
    captured = bool(capture.reshape(-1)[0].item())
    zero = output.measurement_xy.sum() * 0.0

    if captured:
        measurement_loss = F.smooth_l1_loss(output.measurement_xy, gt_xy)
        residual_xy = output.measurement_xy - gt_xy
        residual_route = project_route_torch(residual_xy, route_unit, cross_unit)
        variance = output.measurement_variance_route.clamp_min(
            float(config.KALMAN_R_MIN_VAR)
        )
        variance_nll = 0.5 * (
            residual_route.square() / variance + variance.log()
        ).mean()
    else:
        measurement_loss = zero
        variance_nll = zero

    next_step_loss = F.smooth_l1_loss(
        output.next_step_route, target_next_step_route
    )
    velocity_loss = F.smooth_l1_loss(
        output.velocity_route, target_velocity_route
    )
    acceleration_loss = F.smooth_l1_loss(
        output.acceleration_route, target_acceleration_route
    )

    raw_error = torch.linalg.norm(candidate.hardms_xy - gt_xy, dim=1, keepdim=True)
    sigma = float(config.CONFIDENCE_TARGET_SIGMA_M)
    confidence_target = torch.exp(-0.5 * raw_error.square() / (sigma * sigma))
    if not captured:
        confidence_target = torch.zeros_like(confidence_target)
    confidence_loss = F.binary_cross_entropy(
        output.confidence, confidence_target.detach().clamp(0.0, 1.0)
    )

    cross_reg = (
        output.velocity_route[:, 1].abs().mean()
        + 0.5 * output.acceleration_route[:, 1].abs().mean()
    )

    total = (
        float(config.LOSS_MEASUREMENT) * measurement_loss
        + float(config.LOSS_NEXT_STEP) * next_step_loss
        + float(config.LOSS_VELOCITY) * velocity_loss
        + float(config.LOSS_ACCELERATION) * acceleration_loss
        + float(config.LOSS_VARIANCE_NLL) * variance_nll
        + float(config.LOSS_CONFIDENCE) * confidence_loss
        + float(config.LOSS_CROSS_MOTION_REG) * cross_reg
    )
    return total, {
        "measurement": float(measurement_loss.detach().cpu()),
        "next": float(next_step_loss.detach().cpu()),
        "velocity": float(velocity_loss.detach().cpu()),
        "acceleration": float(acceleration_loss.detach().cpu()),
        "nll": float(variance_nll.detach().cpu()),
        "confidence": float(confidence_loss.detach().cpu()),
        "capture": 1.0 if captured else 0.0,
    }


# -----------------------------------------------------------------------------
# Training / validation
# -----------------------------------------------------------------------------



def build_monotonic_leg_labels(cache, route):
    labels = []
    leg = 0
    for index in range(len(cache)):
        position = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        leg = route.advance(position, leg)
        labels.append(int(leg))
    return labels


def split_ranges(length):
    guard = int(config.SPLIT_GUARD_FRAMES)
    train_end = int(length * float(config.TRAIN_FRACTION))
    val_end = int(length * (float(config.TRAIN_FRACTION) + float(config.VAL_FRACTION)))
    return {
        "train": (0, max(1, train_end - guard)),
        "val": (
            min(length - 1, train_end + guard),
            max(min(length, val_end - guard), min(length - 1, train_end + guard) + 1),
        ),
    }


@torch.no_grad()
def evaluate_closed_loop_episode(
    model,
    visual,
    cache,
    route,
    start,
    end,
    device,
    leg_labels=None,
):
    model.eval()
    initial_xy = cache.gt_xy[start].cpu().numpy().astype(np.float64)
    kf = PolynomialKalman2D(initial_xy)
    leg_index = (
        int(leg_labels[start])
        if leg_labels is not None
        else route.closest_leg(initial_xy)
    )
    hidden = None
    previous_velocity_route = torch.zeros(1, 2, device=device)
    previous_acceleration_route = torch.zeros(1, 2, device=device)
    previous_velocity_xy = np.zeros(2, dtype=np.float64)
    previous_acceleration_xy = np.zeros(2, dtype=np.float64)
    previous_final = initial_xy.copy()
    errors = []
    captures = []

    for local_index, index in enumerate(range(start, end)):
        if local_index == 0:
            predicted_current = kf.position()
        else:
            predicted_current = kf.predict(previous_velocity_xy, previous_acceleration_xy)

        leg_index = route.advance(predicted_current, leg_index)
        frame = route.frame(predicted_current, leg_index)
        search_center = predicted_current.copy()
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        gt = cache.gt_xy[index : index + 1].to(device).float()
        candidate, sat_context, capture = visual_observation(
            visual,
            uav_clip,
            tensor2(search_center, device),
            gt,
        )
        output, _, _ = forward_temporal(
            model,
            candidate,
            sat_context,
            tensor2(search_center, device),
            tensor2(previous_final, device),
            frame,
            previous_velocity_route,
            previous_acceleration_route,
            hidden,
            device,
        )
        hidden = output.hidden
        measurement = output.measurement_xy[0].detach().cpu().numpy()
        variance = output.measurement_variance_xy[0].detach().cpu().numpy()
        final_xy = kf.update(measurement, variance)

        gt_np = gt[0].detach().cpu().numpy()
        errors.append(float(np.linalg.norm(final_xy - gt_np)))
        captures.append(float(capture.float().item()))

        previous_final = final_xy.copy()
        previous_velocity_route = output.velocity_route.detach()
        previous_acceleration_route = output.acceleration_route.detach()
        previous_velocity_xy = output.velocity_xy[0].detach().cpu().numpy()
        previous_acceleration_xy = output.acceleration_xy[0].detach().cpu().numpy()

    if not errors:
        return {"mle": float("inf"), "p90": float("inf"), "capture_pct": 0.0}
    return {
        "mle": float(np.mean(errors)),
        "p90": float(np.quantile(errors, 0.90)),
        "capture_pct": float(np.mean(captures) * 100.0),
    }


@torch.no_grad()
def evaluate_validation_episodes(
    model, visual, cache, route, val_range, device, leg_labels=None
):
    start, end = val_range
    length = max(1, end - start)
    episode_length = min(int(config.VAL_EPISODE_LENGTH), length)
    count = max(1, int(config.VAL_EPISODE_COUNT))
    max_start = max(start, end - episode_length)
    if count == 1 or max_start <= start:
        starts = [start]
    else:
        starts = np.linspace(start, max_start, count).round().astype(int).tolist()

    metrics = []
    for episode_start in starts:
        episode_end = min(end, episode_start + episode_length)
        metrics.append(
            evaluate_closed_loop_episode(
                model,
                visual,
                cache,
                route,
                int(episode_start),
                int(episode_end),
                device,
                leg_labels=leg_labels,
            )
        )
    return {
        "mle": float(np.mean([item["mle"] for item in metrics])),
        "p90": float(np.mean([item["p90"] for item in metrics])),
        "capture_pct": float(np.mean([item["capture_pct"] for item in metrics])),
    }


def train_temporal_model(
    visual,
    cache,
    route,
    device,
    epochs,
    patience_limit,
    resume=False,
):
    model = WaypointConditionedGRU().to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
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
            raise RuntimeError("Latest temporal checkpoint architecture mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_score = float(payload.get("best_score", float("inf")))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))
        print("resume temporal training from epoch %d" % start_epoch, flush=True)

    split = split_ranges(len(cache))
    train_start, train_end = split["train"]
    val_range = split["val"]
    leg_labels = build_monotonic_leg_labels(cache, route)
    chunk_length = int(config.TBPTT_STEPS)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "temporal split train=[%d,%d) val=[%d,%d)" % (
            train_start,
            train_end,
            val_range[0],
            val_range[1],
        ),
        flush=True,
    )

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        ratio = teacher_ratio_for_epoch(epoch)
        chunk_starts = list(range(train_start, train_end, chunk_length))
        random.shuffle(chunk_starts)
        epoch_losses = []
        epoch_capture = []
        component_rows = []

        for chunk_start in chunk_starts:
            chunk_end = min(train_end, chunk_start + chunk_length)
            if chunk_end <= chunk_start:
                continue

            initial_xy = cache.gt_xy[chunk_start].cpu().numpy().astype(np.float64)
            kf = PolynomialKalman2D(initial_xy)
            leg_index = int(leg_labels[chunk_start])
            hidden = None
            previous_final = initial_xy.copy()
            previous_velocity_route = torch.zeros(1, 2, device=device)
            previous_acceleration_route = torch.zeros(1, 2, device=device)
            previous_velocity_xy = np.zeros(2, dtype=np.float64)
            previous_acceleration_xy = np.zeros(2, dtype=np.float64)
            chunk_loss = None

            for local_index, index in enumerate(range(chunk_start, chunk_end)):
                if local_index == 0:
                    predicted_current = kf.position()
                else:
                    predicted_current = kf.predict(
                        previous_velocity_xy, previous_acceleration_xy
                    )

                leg_index = route.advance(predicted_current, leg_index)
                route_frame = route.frame(predicted_current, leg_index)

                use_teacher = random.random() < ratio
                if use_teacher:
                    gt_center = cache.gt_xy[index].cpu().numpy().astype(np.float64)
                    search_center = gt_center + random_jitter(config.TRAIN_CENTER_JITTER_M)
                else:
                    search_center = predicted_current.copy()

                uav_clip = cache.uav_clip[index : index + 1].to(device).float()
                gt = cache.gt_xy[index : index + 1].to(device).float()
                candidate, sat_context, capture = visual_observation(
                    visual,
                    uav_clip,
                    tensor2(search_center, device),
                    gt,
                )
                output, route_unit, cross_unit = forward_temporal(
                    model,
                    candidate,
                    sat_context,
                    tensor2(search_center, device),
                    tensor2(previous_final, device),
                    route_frame,
                    previous_velocity_route,
                    previous_acceleration_route,
                    hidden,
                    device,
                )
                target_v, target_a, target_next_route, _ = target_motion(
                    cache, index, route_unit, cross_unit, device
                )
                step_loss, components = temporal_loss(
                    output,
                    candidate,
                    gt,
                    capture,
                    target_v,
                    target_a,
                    target_next_route,
                    route_unit,
                    cross_unit,
                )
                if chunk_loss is None:
                    chunk_loss = step_loss
                else:
                    chunk_loss = chunk_loss + step_loss
                component_rows.append(components)
                epoch_capture.append(components["capture"])

                measurement = output.measurement_xy[0].detach().cpu().numpy()
                variance = (
                    output.measurement_variance_xy[0].detach().cpu().numpy()
                )
                final_xy = kf.update(measurement, variance)
                previous_final = final_xy.copy()
                previous_velocity_route = output.velocity_route.detach()
                previous_acceleration_route = output.acceleration_route.detach()
                previous_velocity_xy = output.velocity_xy[0].detach().cpu().numpy()
                previous_acceleration_xy = (
                    output.acceleration_xy[0].detach().cpu().numpy()
                )
                hidden = output.hidden

            if chunk_loss is None:
                continue
            chunk_loss = chunk_loss / float(max(1, chunk_end - chunk_start))
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(chunk_loss):
                raise FloatingPointError(
                    "non-finite temporal loss at epoch %d chunk %d" % (
                        epoch,
                        chunk_start,
                    )
                )
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            epoch_losses.append(float(chunk_loss.detach().cpu()))

        validation = evaluate_validation_episodes(
            model, visual, cache, route, val_range, device, leg_labels=leg_labels
        )
        score = float(validation["mle"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA_M)
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
                    "best_score": best_score,
                    "validation": validation,
                    "train_routes": ["route_A"],
                    "validation_routes": ["route_A"],
                    "eval_routes": ["route_B", "route_C"],
                    "uses_waypoint_coordinates": True,
                    "uses_waypoint_frame_index_at_inference": False,
                    "early_stop_metric": "Route-A held-out closed-loop episode MLE",
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

        average_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_capture = float(np.mean(epoch_capture) * 100.0) if epoch_capture else 0.0
        print(
            "temporal epoch=%03d/%d loss=%.5f teacher=%.3f train_capture=%.2f%% "
            "val_mle=%.3fm val_p90=%.3fm val_capture=%.2f%% best=%.3fm patience=%d/%d"
            % (
                epoch,
                int(epochs),
                average_loss,
                ratio,
                train_capture,
                validation["mle"],
                validation["p90"],
                validation["capture_pct"],
                best_score,
                patience,
                int(patience_limit),
            ),
            flush=True,
        )

        if (
            epoch >= int(config.EARLY_STOP_MIN_EPOCH)
            and patience >= int(patience_limit)
        ):
            print(
                "EARLY STOP: held-out Route-A closed-loop MLE did not improve "
                "by %.3fm for %d epochs."
                % (float(config.EARLY_STOP_MIN_DELTA_M), int(patience_limit)),
                flush=True,
            )
            break

    if best_state is None or not config.TEMPORAL_CHECKPOINT.exists():
        raise RuntimeError("Temporal training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


# -----------------------------------------------------------------------------
# Full closed-loop inference
# -----------------------------------------------------------------------------

def safe_heading_deg(velocity_xy):
    velocity = np.asarray(velocity_xy, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(velocity)) <= 1e-6:
        return float("nan")
    return float(math.degrees(math.atan2(velocity[1], velocity[0])))


def waypoint_alignment(velocity_xy, position_xy, target_xy):
    velocity = np.asarray(velocity_xy, dtype=np.float64).reshape(2)
    direction = np.asarray(target_xy, dtype=np.float64).reshape(2) - np.asarray(
        position_xy, dtype=np.float64
    ).reshape(2)
    denom = float(np.linalg.norm(velocity) * np.linalg.norm(direction))
    if denom <= 1e-9:
        return 0.0
    return float(np.clip(np.dot(velocity, direction) / denom, -1.0, 1.0))


def metric_summary(errors):
    error = np.asarray(errors, dtype=np.float64)
    return {
        "MLE_m": float(np.mean(error)),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.quantile(error, 0.90)),
        "P95_m": float(np.quantile(error, 0.95)),
        "P99_m": float(np.quantile(error, 0.99)),
        "LSR@5_pct": float(np.mean(error <= 5.0) * 100.0),
        "LSR@10_pct": float(np.mean(error <= 10.0) * 100.0),
        "LSR@15_pct": float(np.mean(error <= 15.0) * 100.0),
        "LSR@20_pct": float(np.mean(error <= 20.0) * 100.0),
    }


@torch.no_grad()
def run_route_inference(route_name, visual, model, cache, route, device):
    model.eval()
    initial_xy = route.points[0].copy()  # known start waypoint, not GT.
    kf = PolynomialKalman2D(initial_xy)
    leg_index = 0
    hidden = None
    previous_final = initial_xy.copy()
    previous_velocity_route = torch.zeros(1, 2, device=device)
    previous_acceleration_route = torch.zeros(1, 2, device=device)
    previous_velocity_xy = np.zeros(2, dtype=np.float64)
    previous_acceleration_xy = np.zeros(2, dtype=np.float64)
    rows = []
    errors = []
    captures = []
    final_steps = []

    for index in range(len(cache)):
        if index == 0:
            predicted_current = kf.position()
        else:
            predicted_current = kf.predict(previous_velocity_xy, previous_acceleration_xy)

        leg_index = route.advance(predicted_current, leg_index)
        route_frame = route.frame(predicted_current, leg_index)
        search_center = predicted_current.copy()
        gt = cache.gt_xy[index : index + 1].to(device).float()
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        candidate, sat_context, capture = visual_observation(
            visual,
            uav_clip,
            tensor2(search_center, device),
            gt,
        )
        output, _, _ = forward_temporal(
            model,
            candidate,
            sat_context,
            tensor2(search_center, device),
            tensor2(previous_final, device),
            route_frame,
            previous_velocity_route,
            previous_acceleration_route,
            hidden,
            device,
        )

        measurement = output.measurement_xy[0].cpu().numpy()
        variance = output.measurement_variance_xy[0].cpu().numpy()
        final_xy = kf.update(measurement, variance)
        final_velocity = kf.velocity()
        gt_np = gt[0].cpu().numpy()
        error = float(np.linalg.norm(final_xy - gt_np))
        errors.append(error)
        captures.append(float(capture.float().item()))
        if index == 0:
            final_step = 0.0
        else:
            final_step = float(np.linalg.norm(final_xy - previous_final))
        final_steps.append(final_step)

        target_waypoint = route.points[min(leg_index + 1, len(route.points) - 1)]
        alignment = waypoint_alignment(final_velocity, final_xy, target_waypoint)
        heading = safe_heading_deg(final_velocity)

        rows.append(
            {
                "frame_id": int(cache.frame_ids[index].item()),
                "image_path": cache.image_paths[index],
                "gt_x": float(gt_np[0]),
                "gt_y": float(gt_np[1]),
                "predicted_current_x": float(predicted_current[0]),
                "predicted_current_y": float(predicted_current[1]),
                "search_center_x": float(search_center[0]),
                "search_center_y": float(search_center[1]),
                "raw_top1_x": float(candidate.raw_top1_xy[0, 0].item()),
                "raw_top1_y": float(candidate.raw_top1_xy[0, 1].item()),
                "hardms_x": float(candidate.hardms_xy[0, 0].item()),
                "hardms_y": float(candidate.hardms_xy[0, 1].item()),
                "measurement_x": float(measurement[0]),
                "measurement_y": float(measurement[1]),
                "measurement_var_x": float(variance[0]),
                "measurement_var_y": float(variance[1]),
                "confidence": float(output.confidence[0, 0].item()),
                "v_parallel": float(output.velocity_route[0, 0].item()),
                "v_cross": float(output.velocity_route[0, 1].item()),
                "a_parallel": float(output.acceleration_route[0, 0].item()),
                "a_cross": float(output.acceleration_route[0, 1].item()),
                "poly_next_step_parallel": float(output.next_step_route[0, 0].item()),
                "poly_next_step_cross": float(output.next_step_route[0, 1].item()),
                "model_next_step_m": float(
                    torch.linalg.norm(output.next_step_xy[0]).item()
                ),
                "waypoint_leg": int(leg_index),
                "target_waypoint": int(min(leg_index + 1, len(route.points) - 1)),
                "route_progress": float(route_frame.progress),
                "route_remaining_m": float(route_frame.remaining_m),
                "route_cross_track_m": float(route_frame.cross_m),
                "waypoint_alignment": alignment,
                "movement_heading_deg": heading,
                "candidate_capture": int(bool(capture.reshape(-1)[0].item())),
                "final_x": float(final_xy[0]),
                "final_y": float(final_xy[1]),
                "final_vx": float(final_velocity[0]),
                "final_vy": float(final_velocity[1]),
                "final_speed": float(np.linalg.norm(final_velocity)),
                "final_step_m": final_step,
                "error_final_m": error,
            }
        )

        previous_final = final_xy.copy()
        previous_velocity_route = output.velocity_route.detach()
        previous_acceleration_route = output.acceleration_route.detach()
        previous_velocity_xy = output.velocity_xy[0].cpu().numpy()
        previous_acceleration_xy = output.acceleration_xy[0].cpu().numpy()
        hidden = output.hidden

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / (
        route_name + "_waypoint_routeframe_gru_kalman_frames.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = metric_summary(errors)
    summary["CandidateCapture_pct"] = float(np.mean(captures) * 100.0)
    summary["MeanFinalStep_m"] = float(np.mean(final_steps))
    summary["P95FinalStep_m"] = float(np.quantile(final_steps, 0.95))
    summary["Waypoints"] = int(len(route.points))
    summary["CSV"] = str(csv_path)
    print(
        "%s final MLE=%.3fm P90=%.3fm LSR@15=%.2f%% capture=%.2f%%"
        % (
            route_name,
            summary["MLE_m"],
            summary["P90_m"],
            summary["LSR@15_pct"],
            summary["CandidateCapture_pct"],
        ),
        flush=True,
    )
    return summary


def load_temporal_model(device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Temporal checkpoint missing: %s" % config.TEMPORAL_CHECKPOINT
        )
    payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch: %r" % payload.get("architecture")
        )
    model = WaypointConditionedGRU().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def resolve_device():
    requested = str(config.DEVICE)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; fallback to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def train_pipeline(args, device):
    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume_visual),
        )
    else:
        print("reuse visual checkpoint: %s" % config.VISUAL_CHECKPOINT, flush=True)

    visual = FrozenVisualLocalizer(device)
    cache_a = build_route_cache(
        "route_A", config.ROUTE_ROOTS[0], visual, device
    )
    waypoint_a = load_waypoint_xy(
        "route_A", visual.origin_lat, visual.origin_lon
    )
    route_a = WaypointRoute(waypoint_a)
    _, best_score = train_temporal_model(
        visual=visual,
        cache=cache_a,
        route=route_a,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume_temporal),
    )
    print("best closed-loop validation MLE=%.3fm" % best_score, flush=True)


def eval_pipeline(device):
    visual = FrozenVisualLocalizer(device)
    model = load_temporal_model(device)
    all_summary = {
        "architecture": ARCHITECTURE_NAME,
        "train_routes": ["route_A"],
        "eval_routes": ["route_B", "route_C"],
        "known_at_inference": ["start_coordinate", "waypoint_coordinates"],
        "uses_waypoint_frame_index_at_inference": False,
        "motion_state": ["v_parallel", "v_cross", "a_parallel", "a_cross"],
        "polynomial": "p_next = p_final + v + 0.5*a",
        "final_filter": "external Kalman [x,y,vx,vy]",
    }
    for route_name in ["route_B", "route_C"]:
        route_index = config.ROUTE_NAMES.index(route_name)
        cache = build_route_cache(
            route_name, config.ROUTE_ROOTS[route_index], visual, device
        )
        waypoints = load_waypoint_xy(
            route_name, visual.origin_lat, visual.origin_lon
        )
        route = WaypointRoute(waypoints)
        all_summary[route_name] = run_route_inference(
            route_name, visual, model, cache, route, device
        )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.OUTPUT_DIR / "robust_tracker_summary.json"
    summary_path.write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("summary: %s" % summary_path, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "eval", "train_eval"], default="train_eval"
    )
    parser.add_argument("--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS))
    parser.add_argument(
        "--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument("--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M))
    parser.add_argument("--patience", type=int, default=int(config.EARLY_STOP_PATIENCE))
    parser.add_argument("--reuse-visual", action="store_true")
    parser.add_argument("--resume-visual", action="store_true")
    parser.add_argument("--resume-temporal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = resolve_device()
    print("=" * 100, flush=True)
    print(ARCHITECTURE_NAME, flush=True)
    print("device=%s" % device, flush=True)
    print(
        "Known navigation inputs: start coordinate + waypoint coordinates; "
        "waypoint timestamps/frame_index are NOT used for inference.",
        flush=True,
    )
    print(
        "GRU state: route-frame velocity/acceleration; polynomial search prior; "
        "visual measurement+uncertainty; external Kalman final position.",
        flush=True,
    )
    print("=" * 100, flush=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "train_eval"):
        train_pipeline(args, device)
    if args.mode in ("eval", "train_eval"):
        eval_pipeline(device)


if __name__ == "__main__":
    main()
