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
from visual_localizer import FrozenVisualLocalizer, hard_mean_shift, train_visual_retrieval_a_only
from visual_model import WaypointRouteGlobalRecoveryGRU


ARCHITECTURE_NAME = "WaypointRouteGlobalRecoveryGRUKalman_v23"


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


@dataclass
class RouteGallery:
    gallery_indices: torch.Tensor
    centers: torch.Tensor
    z_sat: torch.Tensor
    route_s: torch.Tensor
    leg_indices: torch.Tensor


@dataclass
class RouteGlobalObservation:
    anchor_xy: torch.Tensor
    response_variance_route: torch.Tensor
    sat_context: torch.Tensor
    posterior: torch.Tensor
    route_s: torch.Tensor
    global_entropy: torch.Tensor
    global_margin: torch.Tensor
    mode_mass: torch.Tensor
    waypoint_pass_probability: torch.Tensor
    global_top1_xy: torch.Tensor
    mode_capture: torch.Tensor
    corridor_capture: torch.Tensor
    recovery_distance_m: torch.Tensor
    prior_sigma_m: float


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
        lengths = []
        for leg in range(len(points) - 1):
            lengths.append(float(np.linalg.norm(points[leg + 1] - points[leg])))
        self.leg_lengths = np.asarray(lengths, dtype=np.float64)
        self.cumulative_s = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(self.leg_lengths)]
        )

    def boundary_s(self, waypoint_index):
        waypoint_index = int(np.clip(waypoint_index, 0, len(self.points) - 1))
        return float(self.cumulative_s[waypoint_index])

    def progress_s(self, position_xy, leg_index):
        frame = self.frame(position_xy, leg_index)
        along = float(np.clip(frame.along_m, 0.0, frame.length_m))
        return float(self.cumulative_s[int(leg_index)] + along)

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
        self.last_nis = 0.0
        self.last_r_scale = 1.0

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

    def position_sigma(self):
        variance = max(float(self.P[0, 0]), float(self.P[1, 1]), 0.0)
        return float(math.sqrt(variance))

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
        nis = float(innovation.T @ S_inv @ innovation)
        threshold = max(float(config.KALMAN_NIS_SOFT_THRESHOLD), 1e-6)
        r_scale = min(
            float(config.KALMAN_NIS_MAX_R_SCALE),
            max(1.0, nis / threshold),
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


@torch.no_grad()
def build_route_gallery(visual, route, device):
    """Build cached SAT hypotheses along every waypoint leg.

    A physical SAT anchor may appear on more than one leg when the planned route
    revisits the same area. Keeping those route-state duplicates is intentional:
    the image location is the same, while route progress remains different.
    """
    gallery_xy = visual.gallery["xy"].detach().cpu().numpy().astype(np.float64)
    node_indices = []
    node_s = []
    node_legs = []
    half_width = float(config.ROUTE_CORRIDOR_HALF_WIDTH_M)

    for leg in range(len(route.points) - 1):
        start_xy, _, unit, _, length = route._geometry(leg)
        rel = gallery_xy - start_xy[None, :]
        along = rel @ unit
        clipped = np.clip(along, 0.0, length)
        nearest = start_xy[None, :] + clipped[:, None] * unit[None, :]
        distance = np.linalg.norm(gallery_xy - nearest, axis=1)
        keep = np.nonzero(distance <= half_width)[0]
        if keep.size == 0:
            continue
        node_indices.append(keep.astype(np.int64))
        node_s.append((route.cumulative_s[leg] + clipped[keep]).astype(np.float32))
        node_legs.append(np.full(keep.shape[0], leg, dtype=np.int64))

    if not node_indices:
        raise RuntimeError("No SAT anchors intersect the waypoint route corridor")

    node_indices_np = np.concatenate(node_indices)
    node_s_np = np.concatenate(node_s)
    node_legs_np = np.concatenate(node_legs)
    unique_indices, inverse = np.unique(node_indices_np, return_inverse=True)
    unique_tensor = torch.tensor(unique_indices, dtype=torch.long, device=device)
    z_rows = []
    batch = int(config.ROUTE_GLOBAL_EMBED_BATCH)
    visual.model.eval()
    for offset in range(0, len(unique_indices), batch):
        idx = unique_tensor[offset : offset + batch]
        sat_clip = visual.gallery["clip_feat"][idx].float()
        sat_xy = visual.gallery["xy"][idx].float()
        z_rows.append(visual.model.encode_sat_from_clip(sat_clip, sat_xy))
    z_unique = torch.cat(z_rows, dim=0)
    inverse_tensor = torch.tensor(inverse, dtype=torch.long, device=device)
    node_index_tensor = torch.tensor(node_indices_np, dtype=torch.long, device=device)

    result = RouteGallery(
        gallery_indices=node_index_tensor,
        centers=visual.gallery["xy"][node_index_tensor].float(),
        z_sat=z_unique[inverse_tensor].float(),
        route_s=torch.tensor(node_s_np, dtype=torch.float32, device=device),
        leg_indices=torch.tensor(node_legs_np, dtype=torch.long, device=device),
    )
    print(
        "route-global gallery: nodes=%d unique_sat=%d corridor=+/-%.1fm"
        % (len(node_indices_np), len(unique_indices), half_width),
        flush=True,
    )
    return result


def local_visual_observation(visual, uav_clip, center_xy, gt_xy=None):
    candidate = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=center_xy,
        grid_size=int(config.GRID_SIZE),
    )
    sat_context = (candidate.raw_prob.unsqueeze(-1) * candidate.z_sat).sum(dim=1)
    capture = None
    if gt_xy is not None:
        capture = visual.candidate_contains_gt_anchor(candidate.indices, gt_xy)
    return candidate, sat_context, capture


@torch.no_grad()
def route_global_observation(
    visual,
    route_gallery,
    route,
    z_uav,
    predicted_xy,
    route_frame,
    leg_index,
    kalman_position_sigma,
    gt_xy=None,
):
    device = z_uav.device
    eligible = route_gallery.leg_indices >= int(leg_index)
    if not bool(eligible.any()):
        eligible = torch.ones_like(route_gallery.leg_indices, dtype=torch.bool)

    centers = route_gallery.centers[eligible]
    z_sat = route_gallery.z_sat[eligible]
    route_s = route_gallery.route_s[eligible]

    visual_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None, :] * z_sat[None, :, :]
    ).sum(dim=2)

    predicted = tensor2(predicted_xy, device)
    distance2 = (centers[None, :, :] - predicted[:, None, :]).square().sum(dim=2)
    predicted_s = float(route.progress_s(predicted_xy, leg_index))
    delta_s2 = (route_s[None, :] - predicted_s).square()

    sigma_xy = float(np.clip(
        float(config.ROUTE_PRIOR_SIGMA_XY_MIN_M)
        + float(config.ROUTE_PRIOR_COVARIANCE_SCALE) * float(kalman_position_sigma),
        float(config.ROUTE_PRIOR_SIGMA_XY_MIN_M),
        float(config.ROUTE_PRIOR_SIGMA_XY_MAX_M),
    ))
    sigma_s = max(float(config.ROUTE_PRIOR_PROGRESS_SIGMA_M), sigma_xy)
    local_prior = torch.exp(
        -0.5 * distance2 / max(sigma_xy * sigma_xy, 1e-6)
        -0.5 * delta_s2 / max(sigma_s * sigma_s, 1e-6)
    )
    floor = float(config.ROUTE_PRIOR_UNIFORM_FLOOR)
    prior = floor + (1.0 - floor) * local_prior
    fused_logits = (
        visual_logits / max(float(config.ROUTE_GLOBAL_TEMPERATURE), 1e-3)
        + float(config.ROUTE_PRIOR_LOG_WEIGHT) * prior.clamp_min(1e-12).log()
    )
    posterior = torch.softmax(fused_logits, dim=1)

    count = max(int(posterior.shape[1]), 2)
    entropy = -(
        posterior * posterior.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(float(count))
    top_values, top_indices = posterior.topk(
        k=min(2, int(posterior.shape[1])), dim=1
    )
    if top_values.shape[1] == 1:
        margin = top_values[:, 0]
    else:
        margin = top_values[:, 0] - top_values[:, 1]

    top_index = int(top_indices[0, 0].item())
    top_xy = centers[top_index : top_index + 1]
    top_s = route_s[top_index]
    spatial_distance = torch.linalg.norm(centers - top_xy[0], dim=1)
    route_distance = (route_s - top_s).abs()
    mode_mask = (
        (spatial_distance <= float(config.ROUTE_GLOBAL_MODE_RADIUS_M))
        & (route_distance <= 2.0 * float(config.ROUTE_GLOBAL_MODE_RADIUS_M))
    )
    if not bool(mode_mask.any()):
        mode_mask[top_index] = True

    mode_mass = posterior[:, mode_mask].sum(dim=1).clamp_min(1e-8)
    mode_probability = posterior[:, mode_mask] / mode_mass[:, None]
    mode_centers = centers[mode_mask]
    anchor = (mode_probability.unsqueeze(-1) * mode_centers[None, :, :]).sum(dim=1)
    global_context = (
        posterior.unsqueeze(-1) * z_sat[None, :, :]
    ).sum(dim=1)

    unit = tensor2(route_frame.unit, device)
    cross = tensor2(route_frame.cross, device)
    # Covariance is measured inside the dominant spatial/route mode, then
    # inflated by its posterior mass. This keeps a strong recovery mode useful
    # while making a weak/multimodal route-global response appropriately noisy.
    mode_delta = mode_centers[None, :, :] - anchor[:, None, :]
    mode_parallel = (mode_delta * unit[:, None, :]).sum(dim=2)
    mode_cross = (mode_delta * cross[:, None, :]).sum(dim=2)
    response_variance_route = torch.stack(
        [
            (mode_probability * mode_parallel.square()).sum(dim=1),
            (mode_probability * mode_cross.square()).sum(dim=1),
        ],
        dim=1,
    ) / mode_mass[:, None].square().clamp_min(1e-4)

    if int(leg_index) < len(route.points) - 2:
        boundary_s = float(route.boundary_s(int(leg_index) + 1))
        pass_probability = posterior[:, route_s >= boundary_s].sum(dim=1)
    else:
        pass_probability = torch.zeros(1, dtype=posterior.dtype, device=device)

    mode_capture = torch.zeros(1, dtype=torch.bool, device=device)
    corridor_capture = torch.zeros(1, dtype=torch.bool, device=device)
    if gt_xy is not None:
        gt = gt_xy.reshape(1, 2)
        corridor_capture = (
            torch.linalg.norm(centers[None, :, :] - gt[:, None, :], dim=2).min(dim=1).values
            <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
        )
        mode_capture = (
            torch.linalg.norm(mode_centers[None, :, :] - gt[:, None, :], dim=2).min(dim=1).values
            <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
        )

    recovery_distance = torch.linalg.norm(anchor - predicted, dim=1)
    return RouteGlobalObservation(
        anchor_xy=anchor,
        response_variance_route=response_variance_route,
        sat_context=global_context,
        posterior=posterior,
        route_s=route_s,
        global_entropy=entropy,
        global_margin=margin,
        mode_mass=mode_mass,
        waypoint_pass_probability=pass_probability,
        global_top1_xy=top_xy,
        mode_capture=mode_capture,
        corridor_capture=corridor_capture,
        recovery_distance_m=recovery_distance,
        prior_sigma_m=sigma_xy,
    )


def advance_leg_from_visual(route, leg_index, observation):
    """Advance only from current-frame visual posterior, never motion alone."""
    leg = int(leg_index)
    posterior = observation.posterior[0]
    route_s = observation.route_s
    threshold = float(config.WAYPOINT_PASS_PROBABILITY)
    while leg < len(route.points) - 2:
        boundary = float(route.boundary_s(leg + 1))
        pass_probability = float(posterior[route_s >= boundary].sum().item())
        if pass_probability <= threshold:
            break
        leg += 1
    return leg


def forward_temporal(
    model,
    local_candidate,
    local_sat_context,
    global_observation,
    search_center_xy,
    previous_final_xy,
    route_frame,
    previous_velocity_route,
    previous_acceleration_route,
    previous_z_uav,
    hidden,
    device,
):
    unit, cross, remaining, cross_track, progress = route_tensors(route_frame, device)
    output = model.forward_step(
        z_uav=local_candidate.z_uav,
        previous_z_uav=previous_z_uav,
        local_sat_context=local_sat_context,
        global_sat_context=global_observation.sat_context,
        local_probability=local_candidate.raw_prob,
        global_probability=global_observation.posterior,
        local_hardms_xy=local_candidate.hardms_xy,
        local_top1_xy=local_candidate.raw_top1_xy,
        local_hardms_support=local_candidate.hardms_support,
        global_anchor_xy=global_observation.anchor_xy,
        global_response_variance_route=global_observation.response_variance_route,
        global_entropy=global_observation.global_entropy,
        global_margin=global_observation.global_margin,
        global_mode_mass=global_observation.mode_mass,
        waypoint_pass_probability=global_observation.waypoint_pass_probability,
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

    raw_error = torch.linalg.norm(output.measurement_anchor_xy - gt_xy, dim=1, keepdim=True)
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

    pred_norm = torch.linalg.norm(output.next_step_route, dim=1)
    target_norm = torch.linalg.norm(target_next_step_route, dim=1)
    valid_direction = target_norm > 0.10
    if bool(valid_direction.any()):
        direction_cos = F.cosine_similarity(
            output.next_step_route[valid_direction],
            target_next_step_route[valid_direction],
            dim=1,
        )
        direction_loss = (1.0 - direction_cos).mean()
    else:
        direction_loss = zero
    speed_loss = F.smooth_l1_loss(pred_norm, target_norm)

    total = (
        float(config.LOSS_MEASUREMENT) * measurement_loss
        + float(config.LOSS_NEXT_STEP) * next_step_loss
        + float(config.LOSS_VELOCITY) * velocity_loss
        + float(config.LOSS_ACCELERATION) * acceleration_loss
        + float(config.LOSS_VARIANCE_NLL) * variance_nll
        + float(config.LOSS_CONFIDENCE) * confidence_loss
        + float(config.LOSS_CROSS_MOTION_REG) * cross_reg
        + float(config.LOSS_DIRECTION) * direction_loss
        + float(config.LOSS_SPEED) * speed_loss
    )
    return total, {
        "measurement": float(measurement_loss.detach().cpu()),
        "next": float(next_step_loss.detach().cpu()),
        "velocity": float(velocity_loss.detach().cpu()),
        "acceleration": float(acceleration_loss.detach().cpu()),
        "direction": float(direction_loss.detach().cpu()),
        "speed": float(speed_loss.detach().cpu()),
        "pred_step_m": float(pred_norm.mean().detach().cpu()),
        "target_step_m": float(target_norm.mean().detach().cpu()),
        "pred_v_parallel": float(output.velocity_route[:, 0].mean().detach().cpu()),
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
    route_gallery,
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
    leg_index = int(leg_labels[start]) if leg_labels is not None else route.closest_leg(initial_xy)
    hidden = None
    previous_z_uav = None
    previous_velocity_route = torch.zeros(1, 2, device=device)
    previous_acceleration_route = torch.zeros(1, 2, device=device)
    previous_velocity_xy = np.zeros(2, dtype=np.float64)
    previous_acceleration_xy = np.zeros(2, dtype=np.float64)
    previous_final = initial_xy.copy()
    errors = []
    mode_captures = []
    corridor_captures = []

    for local_index, index in enumerate(range(start, end)):
        if local_index == 0:
            predicted_current = kf.position()
        else:
            predicted_current = kf.predict(previous_velocity_xy, previous_acceleration_xy)

        route_frame = route.frame(predicted_current, leg_index)
        search_center = predicted_current.copy()
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        gt = cache.gt_xy[index : index + 1].to(device).float()
        local_candidate, local_context, _ = local_visual_observation(
            visual, uav_clip, tensor2(search_center, device), gt
        )
        global_obs = route_global_observation(
            visual=visual,
            route_gallery=route_gallery,
            route=route,
            z_uav=local_candidate.z_uav,
            predicted_xy=search_center,
            route_frame=route_frame,
            leg_index=leg_index,
            kalman_position_sigma=kf.position_sigma(),
            gt_xy=gt,
        )
        output, _, _ = forward_temporal(
            model,
            local_candidate,
            local_context,
            global_obs,
            tensor2(search_center, device),
            tensor2(previous_final, device),
            route_frame,
            previous_velocity_route,
            previous_acceleration_route,
            previous_z_uav,
            hidden,
            device,
        )
        measurement = output.measurement_xy[0].detach().cpu().numpy()
        variance = output.measurement_variance_xy[0].detach().cpu().numpy()
        final_xy = kf.update(measurement, variance)

        gt_np = gt[0].detach().cpu().numpy()
        errors.append(float(np.linalg.norm(final_xy - gt_np)))
        mode_captures.append(float(global_obs.mode_capture.float().item()))
        corridor_captures.append(float(global_obs.corridor_capture.float().item()))

        leg_index = advance_leg_from_visual(route, leg_index, global_obs)
        previous_final = final_xy.copy()
        previous_velocity_route = output.velocity_route.detach()
        previous_acceleration_route = output.acceleration_route.detach()
        previous_velocity_xy = output.velocity_xy[0].detach().cpu().numpy()
        previous_acceleration_xy = output.acceleration_xy[0].detach().cpu().numpy()
        previous_z_uav = local_candidate.z_uav.detach()
        hidden = output.hidden

    if not errors:
        return {
            "mle": float("inf"), "p90": float("inf"),
            "capture_pct": 0.0, "corridor_capture_pct": 0.0,
        }
    return {
        "mle": float(np.mean(errors)),
        "p90": float(np.quantile(errors, 0.90)),
        "capture_pct": float(np.mean(mode_captures) * 100.0),
        "corridor_capture_pct": float(np.mean(corridor_captures) * 100.0),
    }

@torch.no_grad()
def evaluate_validation_episodes(
    model, visual, route_gallery, cache, route, val_range, device, leg_labels=None
):
    """Deployment-faithful Route-A validation from the single known start."""
    model.eval()
    metric_start, metric_end = int(val_range[0]), int(val_range[1])
    initial_xy = route.points[0].copy()
    kf = PolynomialKalman2D(initial_xy)
    leg_index = 0
    hidden = None
    previous_z_uav = None
    previous_final = initial_xy.copy()
    previous_velocity_route = torch.zeros(1, 2, device=device)
    previous_acceleration_route = torch.zeros(1, 2, device=device)
    previous_velocity_xy = np.zeros(2, dtype=np.float64)
    previous_acceleration_xy = np.zeros(2, dtype=np.float64)
    errors = []
    mode_captures = []
    corridor_captures = []

    for index in range(metric_end):
        if index == 0:
            predicted_current = kf.position()
        else:
            predicted_current = kf.predict(previous_velocity_xy, previous_acceleration_xy)
        route_frame = route.frame(predicted_current, leg_index)
        search_center = predicted_current.copy()
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        gt = cache.gt_xy[index : index + 1].to(device).float()
        local_candidate, local_context, _ = local_visual_observation(
            visual, uav_clip, tensor2(search_center, device), gt
        )
        global_obs = route_global_observation(
            visual=visual,
            route_gallery=route_gallery,
            route=route,
            z_uav=local_candidate.z_uav,
            predicted_xy=search_center,
            route_frame=route_frame,
            leg_index=leg_index,
            kalman_position_sigma=kf.position_sigma(),
            gt_xy=gt,
        )
        output, _, _ = forward_temporal(
            model,
            local_candidate,
            local_context,
            global_obs,
            tensor2(search_center, device),
            tensor2(previous_final, device),
            route_frame,
            previous_velocity_route,
            previous_acceleration_route,
            previous_z_uav,
            hidden,
            device,
        )
        measurement = output.measurement_xy[0].detach().cpu().numpy()
        variance = output.measurement_variance_xy[0].detach().cpu().numpy()
        final_xy = kf.update(measurement, variance)

        if index >= metric_start:
            gt_np = gt[0].detach().cpu().numpy()
            errors.append(float(np.linalg.norm(final_xy - gt_np)))
            mode_captures.append(float(global_obs.mode_capture.float().item()))
            corridor_captures.append(float(global_obs.corridor_capture.float().item()))

        leg_index = advance_leg_from_visual(route, leg_index, global_obs)
        previous_final = final_xy.copy()
        previous_velocity_route = output.velocity_route.detach()
        previous_acceleration_route = output.acceleration_route.detach()
        previous_velocity_xy = output.velocity_xy[0].detach().cpu().numpy()
        previous_acceleration_xy = output.acceleration_xy[0].detach().cpu().numpy()
        previous_z_uav = local_candidate.z_uav.detach()
        hidden = output.hidden

    if not errors:
        return {
            "mle": float("inf"), "p90": float("inf"),
            "capture_pct": 0.0, "corridor_capture_pct": 0.0,
        }
    return {
        "mle": float(np.mean(errors)),
        "p90": float(np.quantile(errors, 0.90)),
        "capture_pct": float(np.mean(mode_captures) * 100.0),
        "corridor_capture_pct": float(np.mean(corridor_captures) * 100.0),
    }

def train_temporal_model(
    visual,
    route_gallery,
    cache,
    route,
    device,
    epochs,
    patience_limit,
    resume=False,
):
    model = WaypointRouteGlobalRecoveryGRU().to(device)
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
            train_start, train_end, val_range[0], val_range[1]
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
        epoch_corridor_capture = []
        component_rows = []

        for chunk_start in chunk_starts:
            chunk_end = min(train_end, chunk_start + chunk_length)
            if chunk_end <= chunk_start:
                continue

            initial_xy = cache.gt_xy[chunk_start].cpu().numpy().astype(np.float64)
            kf = PolynomialKalman2D(initial_xy)
            leg_index = int(leg_labels[chunk_start])
            hidden = None
            previous_z_uav = None
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

                route_frame = route.frame(predicted_current, leg_index)
                gt_center = cache.gt_xy[index].cpu().numpy().astype(np.float64)
                teacher_center = gt_center + random_jitter(config.TRAIN_CENTER_JITTER_M)
                search_center = (
                    float(ratio) * teacher_center
                    + (1.0 - float(ratio)) * predicted_current
                )

                uav_clip = cache.uav_clip[index : index + 1].to(device).float()
                gt = cache.gt_xy[index : index + 1].to(device).float()
                local_candidate, local_context, _ = local_visual_observation(
                    visual, uav_clip, tensor2(search_center, device), gt
                )
                global_obs = route_global_observation(
                    visual=visual,
                    route_gallery=route_gallery,
                    route=route,
                    z_uav=local_candidate.z_uav,
                    predicted_xy=search_center,
                    route_frame=route_frame,
                    leg_index=leg_index,
                    kalman_position_sigma=kf.position_sigma(),
                    gt_xy=gt,
                )
                output, route_unit, cross_unit = forward_temporal(
                    model,
                    local_candidate,
                    local_context,
                    global_obs,
                    tensor2(search_center, device),
                    tensor2(previous_final, device),
                    route_frame,
                    previous_velocity_route,
                    previous_acceleration_route,
                    previous_z_uav,
                    hidden,
                    device,
                )
                target_v, target_a, target_next_route, _ = target_motion(
                    cache, index, route_unit, cross_unit, device
                )
                step_loss, components = temporal_loss(
                    output,
                    local_candidate,
                    gt,
                    global_obs.mode_capture,
                    target_v,
                    target_a,
                    target_next_route,
                    route_unit,
                    cross_unit,
                )
                chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss
                component_rows.append(components)
                epoch_capture.append(components["capture"])
                epoch_corridor_capture.append(
                    float(global_obs.corridor_capture.float().item())
                )

                measurement = output.measurement_xy[0].detach().cpu().numpy()
                variance = output.measurement_variance_xy[0].detach().cpu().numpy()
                final_xy = kf.update(measurement, variance)
                leg_index = advance_leg_from_visual(route, leg_index, global_obs)
                previous_final = final_xy.copy()
                previous_velocity_route = output.velocity_route.detach()
                previous_acceleration_route = output.acceleration_route.detach()
                previous_velocity_xy = output.velocity_xy[0].detach().cpu().numpy()
                previous_acceleration_xy = output.acceleration_xy[0].detach().cpu().numpy()
                previous_z_uav = local_candidate.z_uav.detach()
                hidden = output.hidden

            if chunk_loss is None:
                continue
            chunk_loss = chunk_loss / float(max(1, chunk_end - chunk_start))
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(chunk_loss):
                raise FloatingPointError(
                    "non-finite temporal loss at epoch %d chunk %d" % (
                        epoch, chunk_start
                    )
                )
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config.GRAD_CLIP_NORM))
            optimizer.step()
            epoch_losses.append(float(chunk_loss.detach().cpu()))

        validation = evaluate_validation_episodes(
            model, visual, route_gallery, cache, route, val_range, device,
            leg_labels=leg_labels
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
                    "waypoint_transition": "current-frame route-global posterior",
                    "early_stop_metric": "Route-A known-start full closed-loop held-out MLE",
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
        train_corridor = float(np.mean(epoch_corridor_capture) * 100.0) if epoch_corridor_capture else 0.0
        pred_step_mean = float(np.mean([row["pred_step_m"] for row in component_rows])) if component_rows else 0.0
        target_step_mean = float(np.mean([row["target_step_m"] for row in component_rows])) if component_rows else 0.0
        v_parallel_mean = float(np.mean([row["pred_v_parallel"] for row in component_rows])) if component_rows else 0.0
        print(
            "temporal epoch=%03d/%d loss=%.5f teacher=%.3f mode_capture=%.2f%% "
            "corridor_capture=%.2f%% pred_step=%.3fm target_step=%.3fm v_parallel=%.3f "
            "val_mle=%.3fm val_p90=%.3fm val_mode_capture=%.2f%% val_corridor=%.2f%% "
            "best=%.3fm patience=%d/%d"
            % (
                epoch, int(epochs), average_loss, ratio, train_capture,
                train_corridor, pred_step_mean, target_step_mean, v_parallel_mean,
                validation["mle"], validation["p90"], validation["capture_pct"],
                validation["corridor_capture_pct"], best_score, patience,
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
def run_route_inference(
    route_name, visual, model, route_gallery, cache, route, device
):
    model.eval()
    initial_xy = route.points[0].copy()
    kf = PolynomialKalman2D(initial_xy)
    leg_index = 0
    hidden = None
    previous_z_uav = None
    previous_final = initial_xy.copy()
    previous_velocity_route = torch.zeros(1, 2, device=device)
    previous_acceleration_route = torch.zeros(1, 2, device=device)
    previous_velocity_xy = np.zeros(2, dtype=np.float64)
    previous_acceleration_xy = np.zeros(2, dtype=np.float64)
    rows = []
    errors = []
    local_captures = []
    mode_captures = []
    corridor_captures = []
    final_steps = []
    transitions = 0

    for index in range(len(cache)):
        if index == 0:
            predicted_current = kf.position()
        else:
            predicted_current = kf.predict(
                previous_velocity_xy, previous_acceleration_xy
            )

        # Important: prediction alone never changes waypoint leg.
        leg_before_visual = int(leg_index)
        route_frame = route.frame(predicted_current, leg_before_visual)
        search_center = predicted_current.copy()
        gt = cache.gt_xy[index : index + 1].to(device).float()
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()

        local_candidate, local_context, local_capture = local_visual_observation(
            visual,
            uav_clip,
            tensor2(search_center, device),
            gt,
        )
        global_obs = route_global_observation(
            visual=visual,
            route_gallery=route_gallery,
            route=route,
            z_uav=local_candidate.z_uav,
            predicted_xy=search_center,
            route_frame=route_frame,
            leg_index=leg_before_visual,
            kalman_position_sigma=kf.position_sigma(),
            gt_xy=gt,
        )
        output, _, _ = forward_temporal(
            model,
            local_candidate,
            local_context,
            global_obs,
            tensor2(search_center, device),
            tensor2(previous_final, device),
            route_frame,
            previous_velocity_route,
            previous_acceleration_route,
            previous_z_uav,
            hidden,
            device,
        )

        measurement = output.measurement_xy[0].cpu().numpy()
        variance = output.measurement_variance_xy[0].cpu().numpy()
        final_xy = kf.update(measurement, variance)
        final_velocity = kf.velocity()

        # Waypoint transition is decided only after the current UAV image has
        # generated a route-global posterior. It is never decided from the
        # polynomial/Kalman prediction alone.
        leg_index = advance_leg_from_visual(route, leg_before_visual, global_obs)
        if leg_index != leg_before_visual:
            transitions += int(leg_index - leg_before_visual)

        gt_np = gt[0].cpu().numpy()
        error = float(np.linalg.norm(final_xy - gt_np))
        errors.append(error)
        local_capture_value = int(bool(local_capture.reshape(-1)[0].item()))
        mode_capture_value = int(bool(global_obs.mode_capture.reshape(-1)[0].item()))
        corridor_capture_value = int(bool(global_obs.corridor_capture.reshape(-1)[0].item()))
        local_captures.append(float(local_capture_value))
        mode_captures.append(float(mode_capture_value))
        corridor_captures.append(float(corridor_capture_value))
        final_step = 0.0 if index == 0 else float(np.linalg.norm(final_xy - previous_final))
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
                "raw_top1_x": float(local_candidate.raw_top1_xy[0, 0].item()),
                "raw_top1_y": float(local_candidate.raw_top1_xy[0, 1].item()),
                "hardms_x": float(local_candidate.hardms_xy[0, 0].item()),
                "hardms_y": float(local_candidate.hardms_xy[0, 1].item()),
                "global_top1_x": float(global_obs.global_top1_xy[0, 0].item()),
                "global_top1_y": float(global_obs.global_top1_xy[0, 1].item()),
                "measurement_anchor_x": float(output.measurement_anchor_xy[0, 0].item()),
                "measurement_anchor_y": float(output.measurement_anchor_xy[0, 1].item()),
                "measurement_x": float(measurement[0]),
                "measurement_y": float(measurement[1]),
                "measurement_var_x": float(variance[0]),
                "measurement_var_y": float(variance[1]),
                "response_var_parallel": float(output.response_variance_route[0, 0].item()),
                "response_var_cross": float(output.response_variance_route[0, 1].item()),
                "confidence": float(output.confidence[0, 0].item()),
                "global_entropy": float(global_obs.global_entropy[0].item()),
                "global_margin": float(global_obs.global_margin[0].item()),
                "global_mode_mass": float(global_obs.mode_mass[0].item()),
                "waypoint_pass_probability": float(global_obs.waypoint_pass_probability[0].item()),
                "route_prior_sigma_m": float(global_obs.prior_sigma_m),
                "recovery_distance_m": float(global_obs.recovery_distance_m[0].item()),
                "kalman_nis": float(kf.last_nis),
                "kalman_r_scale": float(kf.last_r_scale),
                "v_parallel": float(output.velocity_route[0, 0].item()),
                "v_cross": float(output.velocity_route[0, 1].item()),
                "a_parallel": float(output.acceleration_route[0, 0].item()),
                "a_cross": float(output.acceleration_route[0, 1].item()),
                "poly_next_step_parallel": float(output.next_step_route[0, 0].item()),
                "poly_next_step_cross": float(output.next_step_route[0, 1].item()),
                "model_next_step_m": float(torch.linalg.norm(output.next_step_xy[0]).item()),
                "waypoint_leg_before_visual": int(leg_before_visual),
                "waypoint_leg": int(leg_index),
                "target_waypoint": int(min(leg_index + 1, len(route.points) - 1)),
                "route_progress": float(route_frame.progress),
                "route_remaining_m": float(route_frame.remaining_m),
                "route_cross_track_m": float(route_frame.cross_m),
                "waypoint_alignment": alignment,
                "movement_heading_deg": heading,
                "local_candidate_capture": local_capture_value,
                "candidate_capture": mode_capture_value,
                "route_corridor_capture": corridor_capture_value,
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
        previous_z_uav = local_candidate.z_uav.detach()
        hidden = output.hidden

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / (
        route_name + "_waypoint_routeglobal_recovery_gru_kalman_frames.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = metric_summary(errors)
    summary["Local6x6Capture_pct"] = float(np.mean(local_captures) * 100.0)
    summary["GlobalModeCapture_pct"] = float(np.mean(mode_captures) * 100.0)
    summary["RouteCorridorCapture_pct"] = float(np.mean(corridor_captures) * 100.0)
    summary["CandidateCapture_pct"] = summary["GlobalModeCapture_pct"]
    summary["MeanFinalStep_m"] = float(np.mean(final_steps))
    summary["P95FinalStep_m"] = float(np.quantile(final_steps, 0.95))
    summary["VisualWaypointTransitions"] = int(transitions)
    summary["Waypoints"] = int(len(route.points))
    summary["CSV"] = str(csv_path)
    print(
        "%s final MLE=%.3fm P90=%.3fm LSR@15=%.2f%% local6x6=%.2f%% "
        "global_mode=%.2f%% corridor=%.2f%%"
        % (
            route_name,
            summary["MLE_m"],
            summary["P90_m"],
            summary["LSR@15_pct"],
            summary["Local6x6Capture_pct"],
            summary["GlobalModeCapture_pct"],
            summary["RouteCorridorCapture_pct"],
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
    model = WaypointRouteGlobalRecoveryGRU().to(device)
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
    route_gallery_a = build_route_gallery(visual, route_a, device)
    _, best_score = train_temporal_model(
        visual=visual,
        route_gallery=route_gallery_a,
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
        "motion_state": ["signed_v_parallel", "v_cross", "a_parallel", "a_cross"],
        "temporal_visual_input": "previous/current UAV embeddings",
        "visual_measurement": "local 6x6 + route-global waypoint-corridor posterior + learned residual/covariance",
        "polynomial": "p_next = p_final + v + 0.5*a",
        "waypoint_transition": "current-frame route-global posterior only",
        "recovery": "heavy-tailed motion prior never removes route-global visual hypotheses",
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
        route_gallery = build_route_gallery(visual, route, device)
        all_summary[route_name] = run_route_inference(
            route_name, visual, model, route_gallery, cache, route, device
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
        "GRU state: route-frame velocity/acceleration; polynomial SOFT prior; "
        "local 6x6 + route-global visual posterior; external Kalman final position.",
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
