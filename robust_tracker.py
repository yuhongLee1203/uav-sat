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
from visual_model import RouteProgressGRU


ARCHITECTURE_NAME = "RouteProgressGRUPolynomialKalman_v25"


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
    s_m: float
    e_m: float
    start_xy: np.ndarray
    end_xy: np.ndarray
    unit: np.ndarray
    cross: np.ndarray
    leg_length_m: float
    leg_progress_m: float
    leg_progress_fraction: float
    remaining_m: float


@dataclass
class VisualObservation:
    candidate: object
    posterior: torch.Tensor
    anchor_xy: torch.Tensor
    anchor_se: torch.Tensor
    response_variance_se: torch.Tensor
    sat_context: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    top1_distance_m: torch.Tensor
    capture: torch.Tensor


class WaypointRoute:
    """Continuous route coordinate built only from ordered waypoint XY."""

    def __init__(self, points_xy):
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 2:
            raise ValueError("At least start + one waypoint are required")
        self.points = points
        units = []
        crosses = []
        lengths = []
        for leg in range(len(points) - 1):
            delta = points[leg + 1] - points[leg]
            length = float(np.linalg.norm(delta))
            if length < float(config.WAYPOINT_MIN_LEG_LENGTH_M):
                unit = np.asarray([1.0, 0.0], dtype=np.float64)
                length = max(length, 1e-6)
            else:
                unit = delta / length
            units.append(unit)
            crosses.append(np.asarray([-unit[1], unit[0]], dtype=np.float64))
            lengths.append(length)
        self.units = np.asarray(units, dtype=np.float64)
        self.crosses = np.asarray(crosses, dtype=np.float64)
        self.leg_lengths = np.asarray(lengths, dtype=np.float64)
        self.cumulative_s = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(self.leg_lengths)]
        )
        self.total_length_m = float(self.cumulative_s[-1])

    def leg_for_s(self, s_m):
        s = float(np.clip(s_m, 0.0, self.total_length_m))
        leg = int(np.searchsorted(self.cumulative_s, s, side="right") - 1)
        return int(np.clip(leg, 0, len(self.points) - 2))

    def frame_from_se(self, s_m, e_m=0.0):
        s = float(np.clip(s_m, 0.0, self.total_length_m))
        leg = self.leg_for_s(s)
        start_s = float(self.cumulative_s[leg])
        along = float(np.clip(s - start_s, 0.0, self.leg_lengths[leg]))
        length = float(self.leg_lengths[leg])
        return RouteFrame(
            leg_index=leg,
            s_m=s,
            e_m=float(e_m),
            start_xy=self.points[leg].copy(),
            end_xy=self.points[leg + 1].copy(),
            unit=self.units[leg].copy(),
            cross=self.crosses[leg].copy(),
            leg_length_m=length,
            leg_progress_m=along,
            leg_progress_fraction=float(along / max(length, 1e-6)),
            remaining_m=float(max(length - along, 0.0)),
        )

    def xy_from_se(self, s_m, e_m):
        frame = self.frame_from_se(s_m, e_m)
        center = frame.start_xy + frame.leg_progress_m * frame.unit
        return center + float(e_m) * frame.cross

    def project_on_leg(self, position_xy, leg):
        leg = int(np.clip(leg, 0, len(self.points) - 2))
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        rel = position - self.points[leg]
        raw_along = float(np.dot(rel, self.units[leg]))
        along = float(np.clip(raw_along, 0.0, self.leg_lengths[leg]))
        center = self.points[leg] + along * self.units[leg]
        e = float(np.dot(position - center, self.crosses[leg]))
        s = float(self.cumulative_s[leg] + along)
        distance = float(np.linalg.norm(position - center))
        return s, e, raw_along, distance

    def project_xy_local(self, position_xy, preferred_leg):
        preferred = int(np.clip(preferred_leg, 0, len(self.points) - 2))
        legs = sorted(
            set(
                int(np.clip(value, 0, len(self.points) - 2))
                for value in [preferred - 1, preferred, preferred + 1]
            )
        )
        best = None
        for leg in legs:
            s, e, raw_along, centerline_distance = self.project_on_leg(position_xy, leg)
            # Pick the geometrically nearest route centerline; use a very small
            # tie-break toward the current leg so intersections do not cause
            # arbitrary segment jumps.
            penalty = 0.05 * abs(leg - preferred)
            score = centerline_distance + penalty
            if best is None or score < best[0]:
                best = (score, s, e, leg, raw_along)
        return float(best[1]), float(best[2]), int(best[3])

    def project_gt_monotonic(self, gt_xy):
        positions = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
        rows = []
        leg = 0
        previous_s = 0.0
        for position in positions:
            while leg < len(self.points) - 2:
                _, _, raw_along, _ = self.project_on_leg(position, leg)
                if raw_along < self.leg_lengths[leg]:
                    break
                leg += 1
            candidates = sorted(
                set(
                    int(np.clip(value, 0, len(self.points) - 2))
                    for value in [leg - 1, leg, leg + 1]
                )
            )
            best = None
            for candidate_leg in candidates:
                s, e, _, centerline_distance = self.project_on_leg(position, candidate_leg)
                if s + 2.0 < previous_s:
                    continue
                if best is None or centerline_distance < best[0]:
                    best = (centerline_distance, s, e, candidate_leg)
            if best is None:
                s, e, _, _ = self.project_on_leg(position, leg)
                best = (0.0, max(previous_s, s), e, leg)
            s = max(previous_s, float(best[1]))
            e = float(best[2])
            leg = max(leg, int(best[3]))
            rows.append([s, e, leg])
            previous_s = s
        return np.asarray(rows, dtype=np.float64)


class RouteKalman:
    """External Kalman in [s, e, vs, ve] after model output."""

    def __init__(self, initial_s=0.0, initial_e=0.0):
        self.x = np.asarray([initial_s, initial_e, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [
                float(config.KALMAN_INIT_PROGRESS_VAR),
                float(config.KALMAN_INIT_CROSS_VAR),
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
                float(config.KALMAN_Q_PROGRESS),
                float(config.KALMAN_Q_CROSS),
                float(config.KALMAN_Q_VELOCITY),
                float(config.KALMAN_Q_VELOCITY),
            ]
        ).astype(np.float64)
        self.last_nis = 0.0
        self.last_r_scale = 1.0
        self._progress_floor = float(initial_s)

    def se(self):
        return self.x[:2].copy()

    def velocity(self):
        return self.x[2:4].copy()

    def predict(self, velocity_se, acceleration_se, total_length_m):
        velocity = np.asarray(velocity_se, dtype=np.float64).reshape(2)
        acceleration = np.asarray(acceleration_se, dtype=np.float64).reshape(2)
        velocity[0] = float(np.clip(
            velocity[0], 0.0, float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        ))
        velocity[1] = float(np.clip(
            velocity[1], -float(config.MAX_CROSS_SPEED_M_PER_FRAME),
            float(config.MAX_CROSS_SPEED_M_PER_FRAME)
        ))
        step = velocity + 0.5 * acceleration
        step[0] = float(np.clip(
            step[0], 0.0, float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        ))
        norm = float(np.linalg.norm(step))
        if norm > float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME):
            step *= float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME) / max(norm, 1e-9)

        self._progress_floor = float(self.x[0])
        self.x[0] = float(np.clip(
            self.x[0] + step[0], self._progress_floor, float(total_length_m)
        ))
        self.x[1] = float(self.x[1] + step[1])
        self.x[2] = float(np.clip(
            velocity[0] + acceleration[0],
            0.0,
            float(config.MAX_FORWARD_SPEED_M_PER_FRAME),
        ))
        self.x[3] = float(np.clip(
            velocity[1] + acceleration[1],
            -float(config.MAX_CROSS_SPEED_M_PER_FRAME),
            float(config.MAX_CROSS_SPEED_M_PER_FRAME),
        ))
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.se()

    def update(self, measurement_se, variance_se, total_length_m):
        z = np.asarray(measurement_se, dtype=np.float64).reshape(2)
        variance = np.asarray(variance_se, dtype=np.float64).reshape(2)
        variance = np.clip(
            variance, float(config.KALMAN_R_MIN_VAR), float(config.KALMAN_R_MAX_VAR)
        )
        R = np.diag(variance)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        nis = float(innovation.T @ S_inv @ innovation)
        threshold = max(float(config.KALMAN_NIS_SOFT_THRESHOLD), 1e-6)
        r_scale = min(
            float(config.KALMAN_NIS_MAX_R_SCALE), max(1.0, nis / threshold)
        )
        if r_scale > 1.0:
            R = R * r_scale
            S = self.H @ self.P @ self.H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
        self.last_nis = nis
        self.last_r_scale = r_scale
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ innovation
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Ordered waypoint navigation is monotonic in route progress.  The
        # current image may correct an over-prediction back to the previous
        # final progress, but never create backward route motion.
        self.x[0] = float(np.clip(
            max(self._progress_floor, self.x[0]), 0.0, float(total_length_m)
        ))
        self.x[2] = float(np.clip(
            self.x[2], 0.0, float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        ))
        self.x[3] = float(np.clip(
            self.x[3], -float(config.MAX_CROSS_SPEED_M_PER_FRAME),
            float(config.MAX_CROSS_SPEED_M_PER_FRAME)
        ))
        return self.se()


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cache_dtype():
    return torch.float16 if str(config.FEATURE_CACHE_DTYPE).lower() == "float16" else torch.float32


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def tensor2(value, device):
    return torch.tensor(
        np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device
    ).reshape(1, 2)


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
            waypoint["latitude"], waypoint["longitude"], origin_lat, origin_lon
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
        Path(root), train=False, origin_lat=visual.origin_lat, origin_lon=visual.origin_lon
    )
    frame_rows, gt_rows, clip_rows, image_paths = [], [], [], []
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


def teacher_ratio_for_epoch(epoch):
    if epoch <= int(config.MOTION_WARMUP_EPOCHS):
        return 1.0
    elapsed = max(0, epoch - int(config.MOTION_WARMUP_EPOCHS))
    fraction = min(1.0, elapsed / max(float(config.TEACHER_DECAY_EPOCHS), 1.0))
    return 1.0 + fraction * (float(config.TEACHER_RATIO_FINAL) - 1.0)


def random_jitter(maximum_m):
    maximum = float(maximum_m)
    if maximum <= 0.0:
        return np.zeros(2, dtype=np.float64)
    radius = math.sqrt(random.random()) * maximum
    angle = random.random() * 2.0 * math.pi
    return np.asarray([radius * math.cos(angle), radius * math.sin(angle)])


# -----------------------------------------------------------------------------
# Route-coordinate targets independent of the model's current/possibly-wrong leg
# -----------------------------------------------------------------------------

def build_gt_route_state(cache, route):
    rows = route.project_gt_monotonic(cache.gt_xy.cpu().numpy())
    se = rows[:, :2]
    legs = rows[:, 2].astype(np.int64)
    ds = np.zeros(len(cache), dtype=np.float64)
    de = np.zeros(len(cache), dtype=np.float64)
    if len(cache) > 1:
        ds[:-1] = np.maximum(0.0, se[1:, 0] - se[:-1, 0])
        de[:-1] = se[1:, 1] - se[:-1, 1]
        ds[-1] = ds[-2]
        de[-1] = de[-2]
    ds = np.clip(ds, 0.0, float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME))
    de = np.clip(
        de,
        -float(config.MAX_CROSS_SPEED_M_PER_FRAME),
        float(config.MAX_CROSS_SPEED_M_PER_FRAME),
    )
    velocity = np.zeros((len(cache), 2), dtype=np.float64)
    acceleration = np.zeros((len(cache), 2), dtype=np.float64)
    step = np.stack([ds, de], axis=1)
    for index in range(len(cache)):
        prev_step = step[index - 1] if index > 0 else step[index]
        next_step = step[index]
        velocity[index] = 0.5 * (prev_step + next_step)
        acceleration[index] = next_step - prev_step
    return {
        "se": se,
        "legs": legs,
        "step": step,
        "velocity": velocity,
        "acceleration": acceleration,
    }


# -----------------------------------------------------------------------------
# Wider local visual posterior around the polynomial prediction
# -----------------------------------------------------------------------------

@torch.no_grad()
def visual_observation(visual, uav_clip, center_xy, route, predicted_se, gt_xy=None):
    candidate = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=center_xy,
        grid_size=int(config.NAV_GRID_SIZE),
    )
    predicted = center_xy.reshape(1, 1, 2)
    distance2 = (candidate.centers - predicted).square().sum(dim=2)
    log_visual = torch.log(
        candidate.raw_prob.clamp_min(float(config.NAV_POSTERIOR_EPS))
    )
    prior = -distance2 / (2.0 * float(config.NAV_MOTION_PRIOR_SIGMA_M) ** 2)
    posterior = torch.softmax(
        log_visual / float(config.NAV_VISUAL_TEMPERATURE)
        + float(config.NAV_MOTION_PRIOR_WEIGHT) * prior,
        dim=1,
    )
    anchor_xy = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)
    sat_context = (posterior.unsqueeze(-1) * candidate.z_sat).sum(dim=1)

    p = posterior.clamp_min(float(config.NAV_POSTERIOR_EPS))
    entropy = -(p * p.log()).sum(dim=1)
    entropy = entropy / max(math.log(max(2, posterior.shape[1])), 1e-6)
    if posterior.shape[1] >= 2:
        top2 = torch.topk(posterior, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.ones_like(entropy)

    predicted_leg = route.leg_for_s(float(predicted_se[0]))
    anchor_np = anchor_xy[0].detach().cpu().numpy()
    anchor_s, anchor_e, _ = route.project_xy_local(anchor_np, predicted_leg)
    anchor_se = torch.tensor(
        [[anchor_s, anchor_e]], dtype=torch.float32, device=anchor_xy.device
    )

    frame = route.frame_from_se(float(predicted_se[0]), float(predicted_se[1]))
    unit = torch.tensor(frame.unit, dtype=torch.float32, device=anchor_xy.device).reshape(1, 1, 2)
    cross = torch.tensor(frame.cross, dtype=torch.float32, device=anchor_xy.device).reshape(1, 1, 2)
    delta = candidate.centers - anchor_xy[:, None, :]
    along_delta = (delta * unit).sum(dim=2)
    cross_delta = (delta * cross).sum(dim=2)
    var_parallel = (posterior * along_delta.square()).sum(dim=1)
    var_cross = (posterior * cross_delta.square()).sum(dim=1)
    response_var = torch.stack([var_parallel, var_cross], dim=1).clamp(
        min=float(config.KALMAN_R_MIN_VAR),
        max=float(config.NAV_MAX_RESPONSE_VARIANCE_M2),
    )

    raw_top1 = candidate.raw_top1_xy
    top1_distance = torch.linalg.norm(raw_top1 - center_xy, dim=1)
    if gt_xy is None:
        capture = torch.ones(1, dtype=torch.bool, device=anchor_xy.device)
    else:
        capture = visual.candidate_contains_gt_anchor(candidate.indices, gt_xy)

    return VisualObservation(
        candidate=candidate,
        posterior=posterior,
        anchor_xy=anchor_xy,
        anchor_se=anchor_se,
        response_variance_se=response_var,
        sat_context=sat_context,
        entropy=entropy,
        margin=margin,
        top1_distance_m=top1_distance,
        capture=capture,
    )


def model_forward(
    model,
    observation,
    previous_z_uav,
    predicted_se,
    previous_measurement_se,
    previous_velocity_se,
    previous_acceleration_se,
    previous_polynomial_step_se,
    route,
    hidden,
    device,
):
    frame = route.frame_from_se(float(predicted_se[0]), float(predicted_se[1]))
    remaining = torch.tensor([[frame.remaining_m]], dtype=torch.float32, device=device)
    predicted_cross = torch.tensor([[float(predicted_se[1])]], dtype=torch.float32, device=device)
    total_fraction = torch.tensor(
        [[float(predicted_se[0]) / max(route.total_length_m, 1e-6)]],
        dtype=torch.float32,
        device=device,
    )
    leg_fraction = torch.tensor(
        [[frame.leg_progress_fraction]], dtype=torch.float32, device=device
    )
    return model.forward_step(
        z_uav=observation.candidate.z_uav,
        previous_z_uav=previous_z_uav,
        sat_context=observation.sat_context,
        posterior_probability=observation.posterior,
        visual_anchor_se=observation.anchor_se,
        response_variance_se=observation.response_variance_se,
        predicted_se=tensor2(predicted_se, device),
        previous_measurement_se=previous_measurement_se,
        previous_velocity_se=previous_velocity_se,
        previous_acceleration_se=previous_acceleration_se,
        polynomial_step_se=previous_polynomial_step_se,
        route_remaining_m=remaining,
        predicted_cross_m=predicted_cross,
        total_progress_fraction=total_fraction,
        leg_progress_fraction=leg_fraction,
        top1_distance_m=observation.top1_distance_m.reshape(-1, 1),
        hardms_support=observation.candidate.hardms_support,
        hidden=hidden,
    )


def temporal_loss(output, observation, target_se, target_velocity, target_acceleration, target_step):
    device = output.measurement_se.device
    captured = bool(observation.capture.reshape(-1)[0].item())
    zero = output.measurement_se.sum() * 0.0
    target_se_t = tensor2(target_se, device)
    target_v_t = tensor2(target_velocity, device)
    target_a_t = tensor2(target_acceleration, device)
    target_step_t = tensor2(target_step, device)

    if captured:
        measurement_loss = F.smooth_l1_loss(output.measurement_se, target_se_t)
        residual = output.measurement_se - target_se_t
        variance = output.measurement_variance_se.clamp_min(
            float(config.KALMAN_R_MIN_VAR)
        )
        variance_nll = 0.5 * (residual.square() / variance + variance.log()).mean()
    else:
        measurement_loss = zero
        variance_nll = zero

    next_loss = F.smooth_l1_loss(output.next_step_se, target_step_t)
    velocity_loss = F.smooth_l1_loss(output.velocity_se, target_v_t)
    acceleration_loss = F.smooth_l1_loss(output.acceleration_se, target_a_t)
    speed_loss = F.smooth_l1_loss(
        output.next_step_se[:, 0], target_step_t[:, 0]
    )
    cross_reg = (
        output.velocity_se[:, 1].abs().mean()
        + 0.5 * output.acceleration_se[:, 1].abs().mean()
    )
    # Explicit progress fidelity.  This is independent of any predicted leg.
    progress_loss = F.smooth_l1_loss(
        output.measurement_se[:, 0], target_se_t[:, 0]
    ) if captured else zero

    total = (
        float(config.LOSS_MEASUREMENT) * measurement_loss
        + float(config.LOSS_NEXT_STEP) * next_loss
        + float(config.LOSS_VELOCITY) * velocity_loss
        + float(config.LOSS_ACCELERATION) * acceleration_loss
        + float(config.LOSS_SPEED) * speed_loss
        + float(config.LOSS_CROSS_MOTION_REG) * cross_reg
        + float(config.LOSS_VARIANCE_NLL) * variance_nll
        + float(config.LOSS_PROGRESS) * progress_loss
    )
    return total, {
        "measurement": float(measurement_loss.detach().cpu()),
        "next": float(next_loss.detach().cpu()),
        "velocity": float(velocity_loss.detach().cpu()),
        "acceleration": float(acceleration_loss.detach().cpu()),
        "speed": float(speed_loss.detach().cpu()),
        "pred_step": float(output.next_step_se[:, 0].mean().detach().cpu()),
        "target_step": float(target_step_t[:, 0].mean().detach().cpu()),
        "pred_velocity": float(output.velocity_se[:, 0].mean().detach().cpu()),
        "target_velocity": float(target_v_t[:, 0].mean().detach().cpu()),
        "capture": 1.0 if captured else 0.0,
    }


# -----------------------------------------------------------------------------
# Sequential closed-loop validation/training
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_closed_loop(model, visual, cache, route, gt_state, metric_range, device):
    model.eval()
    metric_start, metric_end = int(metric_range[0]), int(metric_range[1])
    kf = RouteKalman(0.0, 0.0)
    hidden = None
    previous_z = None
    previous_measurement_se = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)
    errors, speed_errors, progress_errors, captures = [], [], [], []

    for index in range(metric_end):
        if index == 0:
            predicted_se = kf.se()
        else:
            predicted_se = kf.predict(
                previous_velocity[0].cpu().numpy(),
                previous_acceleration[0].cpu().numpy(),
                route.total_length_m,
            )
        search_xy = route.xy_from_se(predicted_se[0], predicted_se[1])
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        gt = cache.gt_xy[index : index + 1].to(device).float()
        obs = visual_observation(
            visual, uav_clip, tensor2(search_xy, device), route, predicted_se, gt
        )
        output = model_forward(
            model, obs, previous_z, predicted_se, previous_measurement_se,
            previous_velocity, previous_acceleration, previous_poly_step,
            route, hidden, device,
        )
        final_se = kf.update(
            output.measurement_se[0].cpu().numpy(),
            output.measurement_variance_se[0].cpu().numpy(),
            route.total_length_m,
        )
        if index >= metric_start:
            final_xy = route.xy_from_se(final_se[0], final_se[1])
            gt_xy = cache.gt_xy[index].cpu().numpy()
            errors.append(float(np.linalg.norm(final_xy - gt_xy)))
            speed_errors.append(
                abs(float(output.velocity_se[0, 0].item()) - float(gt_state["velocity"][index, 0]))
            )
            progress_errors.append(abs(float(final_se[0]) - float(gt_state["se"][index, 0])))
            captures.append(float(obs.capture.float().item()))

        previous_z = obs.candidate.z_uav.detach()
        previous_measurement_se = obs.anchor_se.detach()
        previous_velocity = output.velocity_se.detach()
        previous_acceleration = output.acceleration_se.detach()
        previous_poly_step = output.next_step_se.detach()
        hidden = output.hidden

    if not errors:
        return {
            "mle": float("inf"), "p90": float("inf"), "speed_mae": float("inf"),
            "progress_mae": float("inf"), "capture_pct": 0.0, "score": float("inf")
        }
    mle = float(np.mean(errors))
    speed_mae = float(np.mean(speed_errors))
    progress_mae = float(np.mean(progress_errors))
    score = (
        mle
        + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed_mae
        + float(config.EARLY_SCORE_PROGRESS_WEIGHT) * progress_mae
    )
    return {
        "mle": mle,
        "p90": float(np.quantile(errors, 0.90)),
        "speed_mae": speed_mae,
        "progress_mae": progress_mae,
        "capture_pct": float(np.mean(captures) * 100.0),
        "score": float(score),
    }


def train_temporal_model(visual, cache, route, device, epochs, patience_limit, resume=False):
    model = RouteProgressGRU().to(device)
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
    gt_state = build_gt_route_state(cache, route)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "temporal split train=[%d,%d) val=[%d,%d) route_length=%.1fm"
        % (train_start, train_end, val_range[0], val_range[1], route.total_length_m),
        flush=True,
    )
    print(
        "Route-A GT mean forward step=%.3fm/frame p90=%.3fm/frame"
        % (
            float(np.mean(gt_state["step"][train_start:train_end, 0])),
            float(np.quantile(gt_state["step"][train_start:train_end, 0], 0.90)),
        ),
        flush=True,
    )

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        ratio = teacher_ratio_for_epoch(epoch)
        kf = RouteKalman(0.0, 0.0)
        hidden = None
        previous_z = None
        previous_measurement_se = None
        previous_velocity = torch.zeros(1, 2, device=device)
        previous_acceleration = torch.zeros(1, 2, device=device)
        previous_poly_step = torch.zeros(1, 2, device=device)
        chunk_loss = None
        chunk_count = 0
        losses = []
        component_rows = []

        optimizer.zero_grad(set_to_none=True)
        for index in range(train_start, train_end):
            if index == train_start:
                predicted_se = kf.se()
            else:
                predicted_se = kf.predict(
                    previous_velocity[0].detach().cpu().numpy(),
                    previous_acceleration[0].detach().cpu().numpy(),
                    route.total_length_m,
                )

            predicted_xy = route.xy_from_se(predicted_se[0], predicted_se[1])
            gt_xy_np = cache.gt_xy[index].cpu().numpy().astype(np.float64)
            teacher_xy = gt_xy_np + random_jitter(config.TRAIN_CENTER_JITTER_M)
            search_xy = float(ratio) * teacher_xy + (1.0 - float(ratio)) * predicted_xy

            uav_clip = cache.uav_clip[index : index + 1].to(device).float()
            gt_xy = cache.gt_xy[index : index + 1].to(device).float()
            obs = visual_observation(
                visual, uav_clip, tensor2(search_xy, device), route, predicted_se, gt_xy
            )
            output = model_forward(
                model, obs, previous_z, predicted_se, previous_measurement_se,
                previous_velocity, previous_acceleration, previous_poly_step,
                route, hidden, device,
            )

            step_loss, components = temporal_loss(
                output=output,
                observation=obs,
                target_se=gt_state["se"][index],
                target_velocity=gt_state["velocity"][index],
                target_acceleration=gt_state["acceleration"][index],
                target_step=gt_state["step"][index],
            )
            chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss
            chunk_count += 1
            component_rows.append(components)

            final_se = kf.update(
                output.measurement_se[0].detach().cpu().numpy(),
                output.measurement_variance_se[0].detach().cpu().numpy(),
                route.total_length_m,
            )
            _ = final_se
            previous_z = obs.candidate.z_uav.detach()
            previous_measurement_se = obs.anchor_se.detach()
            previous_velocity = output.velocity_se.detach()
            previous_acceleration = output.acceleration_se.detach()
            previous_poly_step = output.next_step_se.detach()
            hidden = output.hidden

            boundary = (
                chunk_count >= int(config.TBPTT_STEPS)
                or index + 1 >= train_end
            )
            if boundary:
                normalized = chunk_loss / float(max(1, chunk_count))
                if not torch.isfinite(normalized):
                    raise FloatingPointError(
                        "non-finite temporal loss at epoch %d frame %d" % (epoch, index)
                    )
                normalized.backward()
                torch.nn.utils.clip_grad_norm_(parameters, float(config.GRAD_CLIP_NORM))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(normalized.detach().cpu()))
                # Standard sequential TBPTT: detach recurrent state but do not
                # reset Kalman/progress/velocity or shuffle to a GT restart.
                hidden = hidden.detach()
                previous_velocity = previous_velocity.detach()
                previous_acceleration = previous_acceleration.detach()
                previous_poly_step = previous_poly_step.detach()
                previous_measurement_se = previous_measurement_se.detach()
                chunk_loss = None
                chunk_count = 0

        validation = evaluate_closed_loop(
            model, visual, cache, route, gt_state, val_range, device
        )
        score = float(validation["score"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)
        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
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
                    "known_at_inference": ["start_coordinate", "waypoint_coordinates"],
                    "uses_waypoint_frame_index_at_inference": False,
                    "waypoint_state": "continuous filtered route progress s; no learned leg switch classifier",
                    "training": "sequential TBPTT without random GT chunk restarts",
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

        pred_step = float(np.mean([r["pred_step"] for r in component_rows]))
        target_step = float(np.mean([r["target_step"] for r in component_rows]))
        pred_velocity = float(np.mean([r["pred_velocity"] for r in component_rows]))
        target_velocity = float(np.mean([r["target_velocity"] for r in component_rows]))
        capture = float(np.mean([r["capture"] for r in component_rows]) * 100.0)
        print(
            "temporal epoch=%03d/%d loss=%.5f teacher=%.3f capture=%.2f%% "
            "pred_step=%.3f target_step=%.3f pred_v=%.3f target_v=%.3f "
            "val_mle=%.3fm val_p90=%.3fm val_speed_mae=%.3f "
            "val_progress_mae=%.3fm val_capture=%.2f%% score=%.3f best=%.3f patience=%d/%d"
            % (
                epoch, int(epochs), float(np.mean(losses)) if losses else float("nan"),
                ratio, capture, pred_step, target_step, pred_velocity, target_velocity,
                validation["mle"], validation["p90"], validation["speed_mae"],
                validation["progress_mae"], validation["capture_pct"], score,
                best_score, patience, int(patience_limit),
            ),
            flush=True,
        )

        if epoch >= int(config.EARLY_STOP_MIN_EPOCH) and patience >= int(patience_limit):
            print(
                "EARLY STOP: composite closed-loop score did not improve by %.3f for %d epochs."
                % (float(config.EARLY_STOP_MIN_DELTA), int(patience_limit)),
                flush=True,
            )
            break

    if best_state is None or not config.TEMPORAL_CHECKPOINT.exists():
        raise RuntimeError("Temporal training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


# -----------------------------------------------------------------------------
# Inference and output
# -----------------------------------------------------------------------------

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
    gt_state = build_gt_route_state(cache, route)
    kf = RouteKalman(0.0, 0.0)
    hidden = None
    previous_z = None
    previous_measurement_se = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)
    rows, errors, captures, final_steps, speed_errors, progress_errors = [], [], [], [], [], []
    previous_final_xy = route.xy_from_se(0.0, 0.0)

    for index in range(len(cache)):
        if index == 0:
            predicted_se = kf.se()
        else:
            predicted_se = kf.predict(
                previous_velocity[0].cpu().numpy(),
                previous_acceleration[0].cpu().numpy(),
                route.total_length_m,
            )
        predicted_xy = route.xy_from_se(predicted_se[0], predicted_se[1])
        frame_before = route.frame_from_se(predicted_se[0], predicted_se[1])
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        gt_xy_t = cache.gt_xy[index : index + 1].to(device).float()
        obs = visual_observation(
            visual, uav_clip, tensor2(predicted_xy, device), route, predicted_se, gt_xy_t
        )
        output = model_forward(
            model, obs, previous_z, predicted_se, previous_measurement_se,
            previous_velocity, previous_acceleration, previous_poly_step,
            route, hidden, device,
        )
        final_se = kf.update(
            output.measurement_se[0].cpu().numpy(),
            output.measurement_variance_se[0].cpu().numpy(),
            route.total_length_m,
        )
        final_xy = route.xy_from_se(final_se[0], final_se[1])
        frame_after = route.frame_from_se(final_se[0], final_se[1])
        gt_xy = cache.gt_xy[index].cpu().numpy()
        error = float(np.linalg.norm(final_xy - gt_xy))
        errors.append(error)
        captures.append(float(obs.capture.float().item()))
        final_step = 0.0 if index == 0 else float(np.linalg.norm(final_xy - previous_final_xy))
        final_steps.append(final_step)
        target_v = float(gt_state["velocity"][index, 0])
        target_step = float(gt_state["step"][index, 0])
        speed_error = abs(float(output.velocity_se[0, 0].item()) - target_v)
        progress_error = abs(float(final_se[0]) - float(gt_state["se"][index, 0]))
        speed_errors.append(speed_error)
        progress_errors.append(progress_error)

        p = obs.posterior.clamp_min(float(config.NAV_POSTERIOR_EPS))
        entropy = float((-(p * p.log()).sum(dim=1) / max(math.log(max(2, p.shape[1])), 1e-6))[0].item())
        if p.shape[1] >= 2:
            top2 = torch.topk(p, k=2, dim=1).values
            margin = float((top2[:, 0] - top2[:, 1])[0].item())
        else:
            margin = 1.0

        rows.append(
            {
                "frame_id": int(cache.frame_ids[index].item()),
                "image_path": cache.image_paths[index],
                "gt_x": float(gt_xy[0]),
                "gt_y": float(gt_xy[1]),
                "gt_progress_s": float(gt_state["se"][index, 0]),
                "gt_cross_e": float(gt_state["se"][index, 1]),
                "gt_waypoint_leg": int(gt_state["legs"][index]),
                "gt_step_parallel": target_step,
                "gt_velocity_parallel": target_v,
                "predicted_progress_s": float(predicted_se[0]),
                "predicted_cross_e": float(predicted_se[1]),
                "predicted_x": float(predicted_xy[0]),
                "predicted_y": float(predicted_xy[1]),
                "search_grid_size": int(config.NAV_GRID_SIZE),
                "raw_top1_x": float(obs.candidate.raw_top1_xy[0, 0].item()),
                "raw_top1_y": float(obs.candidate.raw_top1_xy[0, 1].item()),
                "hardms_x": float(obs.candidate.hardms_xy[0, 0].item()),
                "hardms_y": float(obs.candidate.hardms_xy[0, 1].item()),
                "visual_anchor_x": float(obs.anchor_xy[0, 0].item()),
                "visual_anchor_y": float(obs.anchor_xy[0, 1].item()),
                "visual_anchor_s": float(obs.anchor_se[0, 0].item()),
                "visual_anchor_e": float(obs.anchor_se[0, 1].item()),
                "visual_entropy": entropy,
                "visual_margin": margin,
                "visual_var_s": float(obs.response_variance_se[0, 0].item()),
                "visual_var_e": float(obs.response_variance_se[0, 1].item()),
                "candidate_capture": int(bool(obs.capture.reshape(-1)[0].item())),
                "measurement_s": float(output.measurement_se[0, 0].item()),
                "measurement_e": float(output.measurement_se[0, 1].item()),
                "measurement_var_s": float(output.measurement_variance_se[0, 0].item()),
                "measurement_var_e": float(output.measurement_variance_se[0, 1].item()),
                "v_parallel": float(output.velocity_se[0, 0].item()),
                "v_cross": float(output.velocity_se[0, 1].item()),
                "a_parallel": float(output.acceleration_se[0, 0].item()),
                "a_cross": float(output.acceleration_se[0, 1].item()),
                "poly_next_step_parallel": float(output.next_step_se[0, 0].item()),
                "poly_next_step_cross": float(output.next_step_se[0, 1].item()),
                "kalman_nis": float(kf.last_nis),
                "kalman_r_scale": float(kf.last_r_scale),
                "final_progress_s": float(final_se[0]),
                "final_cross_e": float(final_se[1]),
                "waypoint_leg_before_update": int(frame_before.leg_index),
                "waypoint_leg": int(frame_after.leg_index),
                "target_waypoint": int(min(frame_after.leg_index + 1, len(route.points) - 1)),
                "route_remaining_m": float(frame_after.remaining_m),
                "route_leg_progress": float(frame_after.leg_progress_fraction),
                "progress_error_m": float(progress_error),
                "speed_error_m_per_frame": float(speed_error),
                "final_x": float(final_xy[0]),
                "final_y": float(final_xy[1]),
                "final_step_m": float(final_step),
                "error_final_m": float(error),
            }
        )

        previous_final_xy = final_xy.copy()
        previous_z = obs.candidate.z_uav.detach()
        previous_measurement_se = obs.anchor_se.detach()
        previous_velocity = output.velocity_se.detach()
        previous_acceleration = output.acceleration_se.detach()
        previous_poly_step = output.next_step_se.detach()
        hidden = output.hidden

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / (
        route_name + "_route_progress_gru_polynomial_kalman_frames.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = metric_summary(errors)
    summary["CandidateCapture_pct"] = float(np.mean(captures) * 100.0)
    summary["MeanFinalStep_m"] = float(np.mean(final_steps))
    summary["MeanSpeedError_m_per_frame"] = float(np.mean(speed_errors))
    summary["MeanProgressError_m"] = float(np.mean(progress_errors))
    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])
    summary["FinalGTWaypointLeg"] = int(rows[-1]["gt_waypoint_leg"])
    summary["Waypoints"] = int(len(route.points))
    summary["CSV"] = str(csv_path)
    print(
        "%s final MLE=%.3fm P90=%.3fm LSR@15=%.2f%% capture=%.2f%% "
        "speed_mae=%.3fm/frame progress_mae=%.3fm final_leg=%d gt_leg=%d"
        % (
            route_name, summary["MLE_m"], summary["P90_m"], summary["LSR@15_pct"],
            summary["CandidateCapture_pct"], summary["MeanSpeedError_m_per_frame"],
            summary["MeanProgressError_m"], summary["FinalPredictedWaypointLeg"],
            summary["FinalGTWaypointLeg"],
        ),
        flush=True,
    )
    return summary


def load_temporal_model(device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError("Temporal checkpoint missing: %s" % config.TEMPORAL_CHECKPOINT)
    payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch: %r" % payload.get("architecture")
        )
    model = RouteProgressGRU().to(device)
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
    cache_a = build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route_a = WaypointRoute(
        load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon)
    )
    _, best_score = train_temporal_model(
        visual=visual,
        cache=cache_a,
        route=route_a,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume_temporal),
    )
    print("best composite closed-loop validation score=%.3f" % best_score, flush=True)


def eval_pipeline(device):
    visual = FrozenVisualLocalizer(device)
    model = load_temporal_model(device)
    all_summary = {
        "architecture": ARCHITECTURE_NAME,
        "train_routes": ["route_A"],
        "eval_routes": ["route_B", "route_C"],
        "known_at_inference": ["start_coordinate", "waypoint_coordinates"],
        "uses_waypoint_frame_index_at_inference": False,
        "route_state": "continuous [s,e,vs,ve] on ordered waypoint polyline",
        "motion_model": "two-frame GRU -> v/a -> second-order polynomial",
        "visual_measurement": "12x12 local posterior around polynomial prediction; no route-global retrieval",
        "waypoint_transition": "deterministic from final filtered route progress after current visual update",
        "training": "sequential TBPTT; no random GT chunk restart",
        "final_filter": "external route-coordinate Kalman",
    }
    for route_name in ["route_B", "route_C"]:
        route_index = config.ROUTE_NAMES.index(route_name)
        cache = build_route_cache(
            route_name, config.ROUTE_ROOTS[route_index], visual, device
        )
        route = WaypointRoute(
            load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
        )
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
    parser.add_argument("--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS))
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
        "Known inference inputs: start coordinate + ordered waypoint coordinates only; "
        "waypoint frame_index/timestamps are NOT used.",
        flush=True,
    )
    print(
        "GRU motion -> second-order polynomial -> 12x12 local visual posterior -> "
        "route-coordinate external Kalman -> final XY.",
        flush=True,
    )
    print(
        "Waypoint index comes from final filtered continuous route progress s; "
        "there is no learned single-frame waypoint switch classifier.",
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
