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
from visual_model import ContinuousProgressVisualRNN

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "ContinuousProgressVisualRNN_v11"


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
    xy: torch.Tensor


@dataclass
class ForwardCandidate:
    centers: torch.Tensor
    z_sat: torch.Tensor
    raw_logits: torch.Tensor


class MissionPolynomialPath:
    """
    Continuous ordered route prior. The path is represented as a parametric
    piecewise polynomial/line through the mission waypoints, parameterized by
    cumulative metric progress s.

    The critical design choice is that there is NO discrete leg state to be
    predicted by the RNN. The active waypoint pair is a deterministic function
    of monotonic continuous progress s.
    """

    def __init__(self, waypoints):
        if len(waypoints) < 2:
            raise RuntimeError("Mission path needs at least two waypoints")

        self.waypoints = list(waypoints)
        self.xy = torch.stack([wp.xy.float() for wp in waypoints])

        vector = self.xy[1:] - self.xy[:-1]
        length = torch.linalg.norm(vector, dim=1).clamp_min(1e-6)
        unit = vector / length.unsqueeze(1)

        self.leg_vector = vector
        self.leg_length = length
        self.leg_unit = unit
        self.waypoint_s = torch.cat(
            [
                torch.zeros(1, dtype=torch.float32),
                torch.cumsum(length, dim=0),
            ]
        )
        self.total_length = float(self.waypoint_s[-1].item())

    def leg_index_from_s(self, s_value):
        value = float(s_value)
        if value <= 0.0:
            return 0
        if value >= self.total_length:
            return len(self.leg_length) - 1

        boundaries = self.waypoint_s[1:-1].detach().cpu().numpy()
        return int(np.searchsorted(boundaries, value, side="right"))

    def base_heading_rad(self, s_value):
        leg_index = self.leg_index_from_s(s_value)
        unit = self.leg_unit[leg_index]
        return math.atan2(float(unit[1]), float(unit[0]))

    def base_heading_deg(self, s_value):
        return math.degrees(self.base_heading_rad(s_value))

    def xy_at_s(self, s_value):
        value = float(np.clip(float(s_value), 0.0, self.total_length))
        leg_index = self.leg_index_from_s(value)
        start_s = float(self.waypoint_s[leg_index])
        along = min(
            float(self.leg_length[leg_index]),
            max(0.0, value - start_s),
        )
        return (
            self.xy[leg_index]
            + self.leg_unit[leg_index] * along
        ).clone()

    def xy_at_s_torch(self, s_value):
        s_flat = s_value.reshape(-1)
        device = s_flat.device
        dtype = s_flat.dtype

        waypoint_s = self.waypoint_s.to(device=device, dtype=dtype)
        xy = self.xy.to(device=device, dtype=dtype)
        unit = self.leg_unit.to(device=device, dtype=dtype)
        length = self.leg_length.to(device=device, dtype=dtype)

        clamped = s_flat.clamp(0.0, float(self.total_length))
        boundaries = waypoint_s[1:-1]
        leg_index = torch.bucketize(clamped, boundaries, right=True)
        leg_index = leg_index.clamp(0, int(length.shape[0]) - 1)

        start_s = waypoint_s[leg_index]
        along = (clamped - start_s).clamp_min(0.0)
        along = torch.minimum(along, length[leg_index])

        return xy[leg_index] + unit[leg_index] * along.unsqueeze(1)

    def project_xy(self, xy_value, minimum_s=0.0, maximum_s=None):
        point = torch.as_tensor(xy_value, dtype=torch.float32).reshape(2)
        min_s = max(0.0, float(minimum_s))
        max_s = self.total_length if maximum_s is None else min(
            self.total_length,
            float(maximum_s),
        )
        if max_s < min_s:
            max_s = min_s

        best_s = min_s
        best_distance = float("inf")

        for leg_index in range(len(self.leg_length)):
            leg_start_s = float(self.waypoint_s[leg_index])
            leg_end_s = float(self.waypoint_s[leg_index + 1])

            allowed_start = max(min_s, leg_start_s)
            allowed_end = min(max_s, leg_end_s)
            if allowed_end < allowed_start:
                continue

            start_xy = self.xy[leg_index]
            unit = self.leg_unit[leg_index]

            projection = float(torch.dot(point - start_xy, unit))
            projection = min(
                allowed_end - leg_start_s,
                max(allowed_start - leg_start_s, projection),
            )

            projected_xy = start_xy + unit * projection
            distance = float(torch.linalg.norm(point - projected_xy))

            if distance < best_distance:
                best_distance = distance
                best_s = leg_start_s + projection

        return float(np.clip(best_s, min_s, max_s)), float(best_distance)

    def sequential_gt_progress(self, gt_xy):
        # GT is used only as a training/evaluation LABEL here. It is never fed
        # into the network and never used as an inference search center.
        rows = []
        previous_s = 0.0

        for index in range(int(gt_xy.shape[0])):
            if index == 0:
                minimum_s = 0.0
                maximum_s = min(
                    self.total_length,
                    float(config.GT_LABEL_MAX_FORWARD_M),
                )
            else:
                minimum_s = previous_s
                maximum_s = min(
                    self.total_length,
                    previous_s + float(config.GT_LABEL_MAX_FORWARD_M),
                )

            progress, _ = self.project_xy(
                gt_xy[index],
                minimum_s=minimum_s,
                maximum_s=maximum_s,
            )
            progress = max(previous_s, progress)
            rows.append(progress)
            previous_s = progress

        return torch.tensor(rows, dtype=torch.float32)


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


def wrap_angle_rad(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def heading_deg_from_rad(value):
    result = math.degrees(float(value))
    while result > 180.0:
        result -= 360.0
    while result <= -180.0:
        result += 360.0
    return result


def load_mission_path(route_name, origin_lat, origin_lon):
    path = Path(config.WAYPOINT_FILES[route_name])
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = sorted(payload["waypoints"], key=lambda item: int(item["waypoint_order"]))

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
                xy=torch.tensor([x_m, y_m], dtype=torch.float32),
            )
        )

    result = MissionPolynomialPath(waypoints)
    print(
        "%s: %d waypoints, continuous route length %.1f m"
        % (route_name, len(waypoints), result.total_length),
        flush=True,
    )
    return result


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
            print(
                "%s backbone cache: %d/%d" % (route_name, end, len(dataset)),
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
def build_forward_candidate(
    visual,
    uav_clip,
    center_xy,
    search_heading_rad,
):
    # Build the same 6x6 local lattice used by the existing system.
    full = visual.candidate_batch(
        uav_clip,
        center_xy,
        grid_size=int(config.GRID_SIZE),
    )

    heading = torch.tensor(
        [
            math.cos(float(search_heading_rad)),
            math.sin(float(search_heading_rad)),
        ],
        dtype=full.centers.dtype,
        device=full.centers.device,
    )

    relative = full.centers - center_xy[:, None, :]
    projection = (relative * heading.reshape(1, 1, 2)).sum(dim=2)

    count = int(config.FORWARD_CANDIDATE_COUNT)
    forward_index = projection.topk(
        k=count,
        dim=1,
        largest=True,
    ).indices

    # Guarantee a true zero-motion/current-center visual candidate is included.
    center_distance = torch.linalg.norm(relative, dim=2)
    nearest_index = center_distance.argmin(dim=1)

    for batch_index in range(int(forward_index.shape[0])):
        if not bool((forward_index[batch_index] == nearest_index[batch_index]).any()):
            selected_projection = projection[
                batch_index,
                forward_index[batch_index],
            ]
            replace_at = int(selected_projection.argmin())
            forward_index[batch_index, replace_at] = nearest_index[batch_index]

    gather_xy = forward_index.unsqueeze(-1).expand(-1, -1, 2)
    gather_z = forward_index.unsqueeze(-1).expand(
        -1,
        -1,
        full.z_sat.shape[-1],
    )

    return ForwardCandidate(
        centers=torch.gather(full.centers, 1, gather_xy),
        z_sat=torch.gather(full.z_sat, 1, gather_z),
        raw_logits=torch.gather(full.raw_logits, 1, forward_index),
    )


def candidate_target(candidate, gt_xy):
    distance = torch.linalg.norm(
        candidate.centers - gt_xy[:, None, :],
        dim=2,
    )
    nearest_distance, index = distance.min(dim=1)
    capture = nearest_distance <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
    return index, capture, nearest_distance


def selected_candidate_progress(path, candidate_xy, reference_s):
    return path.project_xy(
        candidate_xy,
        minimum_s=max(
            0.0,
            float(reference_s) - float(config.CANDIDATE_PROJECT_BACK_M),
        ),
        maximum_s=min(
            path.total_length,
            float(reference_s) + float(config.CANDIDATE_PROJECT_FORWARD_M),
        ),
    )


def second_order_inertia_cap(previous_step, previous_previous_step):
    polynomial_step = torch.clamp(
        2.0 * previous_step - previous_previous_step,
        0.0,
        float(config.MAX_STEP_M_PER_FRAME),
    )
    cap = torch.clamp(
        polynomial_step + float(config.INERTIA_ACCEL_MARGIN_M),
        0.0,
        float(config.MAX_STEP_M_PER_FRAME),
    )
    return polynomial_step, cap


def bounded_visual_progress_update(
    current_s,
    candidate_s,
    move_gate,
    previous_step,
    previous_previous_step,
):
    max_step = float(config.MAX_STEP_M_PER_FRAME)

    candidate_s_tensor = torch.as_tensor(
        candidate_s,
        device=current_s.device,
        dtype=current_s.dtype,
    ).reshape_as(current_s)

    visual_step = torch.clamp(
        candidate_s_tensor - current_s,
        0.0,
        max_step,
    )
    rnn_cap = move_gate * max_step

    polynomial_step, inertia_cap = second_order_inertia_cap(
        previous_step,
        previous_previous_step,
    )

    step = torch.minimum(visual_step, rnn_cap)
    step = torch.minimum(step, inertia_cap)

    next_s = torch.clamp(
        current_s + step,
        min=0.0,
    )

    return next_s, step, visual_step, polynomial_step, inertia_cap


def gt_heading_target(path, gt_xy, gt_s, index, device):
    if index <= 0:
        return (
            torch.zeros(1, 1, device=device),
            torch.zeros(1, 1, dtype=torch.bool, device=device),
        )

    displacement = gt_xy[index] - gt_xy[index - 1]
    distance = torch.linalg.norm(displacement)

    if float(distance) < 0.25:
        return (
            torch.zeros(1, 1, device=device),
            torch.zeros(1, 1, dtype=torch.bool, device=device),
        )

    gt_heading = torch.atan2(displacement[1], displacement[0]).to(device)
    base_heading = torch.tensor(
        path.base_heading_rad(float(gt_s[index])),
        dtype=torch.float32,
        device=device,
    )
    residual = wrap_angle_rad(gt_heading - base_heading)

    maximum = math.radians(float(config.RNN_HEADING_RESIDUAL_MAX_DEG))
    residual = residual.clamp(-maximum, maximum)

    return (
        residual.reshape(1, 1),
        torch.ones(1, 1, dtype=torch.bool, device=device),
    )


def temporal_loss(
    pred_s,
    pred_xy,
    pred_step,
    gt_s_value,
    gt_xy_value,
    gt_step,
    candidate_logits,
    candidate_target_index,
    candidate_capture_mask,
    heading_pred,
    heading_target,
    heading_valid,
    variance,
):
    gt_s_tensor = gt_s_value.reshape_as(pred_s).to(pred_s.device)
    gt_xy_tensor = gt_xy_value.reshape(1, 2).to(pred_xy.device)
    gt_step_tensor = gt_step.reshape_as(pred_step).to(pred_step.device)

    progress_loss = F.smooth_l1_loss(pred_s, gt_s_tensor)
    position_loss = F.smooth_l1_loss(pred_xy, gt_xy_tensor)
    step_loss = F.smooth_l1_loss(pred_step, gt_step_tensor)

    if bool(candidate_capture_mask[0]):
        candidate_loss = F.cross_entropy(
            candidate_logits,
            candidate_target_index.to(candidate_logits.device),
        )
    else:
        candidate_loss = pred_s.sum() * 0.0

    if bool(heading_valid[0, 0]):
        heading_error = wrap_angle_rad(heading_pred - heading_target)
        heading_loss = F.smooth_l1_loss(
            heading_error,
            torch.zeros_like(heading_error),
        )
    else:
        heading_loss = pred_s.sum() * 0.0

    ahead = torch.relu(
        pred_s
        - gt_s_tensor
        - float(config.AHEAD_TOLERANCE_M)
    )
    ahead_loss = ahead.square().mean()

    variance = variance.clamp_min(float(config.KALMAN_R_MIN_VAR))
    progress_error = pred_s - gt_s_tensor
    variance_loss = 0.5 * (
        progress_error.square() / variance + variance.log()
    )
    variance_loss = variance_loss.mean()

    total = (
        float(config.LOSS_PROGRESS) * progress_loss
        + float(config.LOSS_POSITION) * position_loss
        + float(config.LOSS_STEP) * step_loss
        + float(config.LOSS_CANDIDATE_CE) * candidate_loss
        + float(config.LOSS_HEADING) * heading_loss
        + float(config.LOSS_AHEAD) * ahead_loss
        + float(config.LOSS_VARIANCE_NLL) * variance_loss
    )

    return {
        "total": total,
        "progress": progress_loss,
        "position": position_loss,
        "step": step_loss,
        "candidate": candidate_loss,
        "heading": heading_loss,
        "ahead": ahead_loss,
    }


def maybe_repeat_training_frame(index):
    if index <= 0:
        return index
    if random.random() < float(config.TRAIN_REPEAT_FRAME_PROB):
        return index - 1
    return index


def train_one_epoch(
    model,
    optimizer,
    visual,
    cache,
    path,
    gt_s,
    train_end,
    device,
):
    model.train()

    hidden = None
    progress_s = torch.zeros(1, 1, dtype=torch.float32, device=device)
    previous_step = torch.zeros_like(progress_s)
    previous_previous_step = torch.zeros_like(progress_s)
    heading_residual = torch.zeros(1, 1, dtype=torch.float32, device=device)

    losses = []
    capture_rows = []
    step_rows = []
    ahead_rows = []

    optimizer.zero_grad(set_to_none=True)
    pending_loss = None
    pending_count = 0

    for index in range(int(train_end)):
        used_index = maybe_repeat_training_frame(index)

        if (
            hidden is not None
            and random.random() < float(config.TRAIN_HIDDEN_RESET_PROB)
        ):
            hidden = torch.zeros_like(hidden)

        center_xy = path.xy_at_s_torch(progress_s).detach()
        base_heading = path.base_heading_rad(float(progress_s.detach().cpu().item()))
        search_heading = base_heading + float(heading_residual.detach().cpu().item())

        uav_clip = cache.uav_clip[
            used_index : used_index + 1
        ].to(device).float()

        candidate = build_forward_candidate(
            visual,
            uav_clip,
            center_xy,
            search_heading,
        )
        z_uav = visual.model.encode_uav_from_clip(uav_clip)

        output = model.forward_step(
            z_uav=z_uav,
            z_sat=candidate.z_sat,
            raw_logits=candidate.raw_logits,
            hidden=hidden,
        )

        selected_index = output.refined_logits.argmax(dim=1)
        selected_xy = candidate.centers[0, selected_index[0]]

        candidate_s, _ = selected_candidate_progress(
            path,
            selected_xy.detach().cpu(),
            float(progress_s.detach().cpu().item()),
        )

        next_s, step, visual_step, polynomial_step, inertia_cap = (
            bounded_visual_progress_update(
                current_s=progress_s,
                candidate_s=candidate_s,
                move_gate=output.move_gate,
                previous_step=previous_step,
                previous_previous_step=previous_previous_step,
            )
        )
        next_s = next_s.clamp(max=float(path.total_length))
        pred_xy = path.xy_at_s_torch(next_s)

        target_index, capture, _ = candidate_target(
            candidate,
            cache.gt_xy[used_index : used_index + 1].to(device),
        )

        gt_s_value = gt_s[used_index : used_index + 1].to(device).reshape(1, 1)

        if used_index <= 0:
            gt_step = torch.zeros(1, 1, device=device)
        else:
            gt_step = (
                gt_s[used_index] - gt_s[used_index - 1]
            ).clamp(
                0.0,
                float(config.MAX_STEP_M_PER_FRAME),
            ).to(device).reshape(1, 1)

        heading_target, heading_valid = gt_heading_target(
            path,
            cache.gt_xy,
            gt_s,
            used_index,
            device,
        )

        batch_loss = temporal_loss(
            pred_s=next_s,
            pred_xy=pred_xy,
            pred_step=step,
            gt_s_value=gt_s_value,
            gt_xy_value=cache.gt_xy[used_index].to(device),
            gt_step=gt_step,
            candidate_logits=output.refined_logits,
            candidate_target_index=target_index,
            candidate_capture_mask=capture,
            heading_pred=output.heading_residual_rad,
            heading_target=heading_target,
            heading_valid=heading_valid,
            variance=output.measurement_variance,
        )

        pending_loss = (
            batch_loss["total"]
            if pending_loss is None
            else pending_loss + batch_loss["total"]
        )
        pending_count += 1

        losses.append(
            [
                float(batch_loss[key].detach().cpu())
                for key in [
                    "total",
                    "progress",
                    "position",
                    "step",
                    "candidate",
                    "heading",
                    "ahead",
                ]
            ]
        )
        capture_rows.append(float(capture.float().mean().cpu()))
        step_rows.append(float(step.detach().cpu().item()))
        ahead_rows.append(
            max(
                0.0,
                float(next_s.detach().cpu().item())
                - float(gt_s_value.detach().cpu().item()),
            )
        )

        hidden = output.hidden
        heading_residual = output.heading_residual_rad
        previous_previous_step = previous_step
        previous_step = step
        progress_s = next_s

        boundary = (
            pending_count >= int(config.TBPTT_STEPS)
            or index == int(train_end) - 1
        )
        if boundary:
            loss_value = pending_loss / float(pending_count)
            if not torch.isfinite(loss_value):
                raise FloatingPointError("non-finite temporal loss")

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.GRAD_CLIP_NORM),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            hidden = hidden.detach()
            heading_residual = heading_residual.detach()
            progress_s = progress_s.detach()
            previous_step = previous_step.detach()
            previous_previous_step = previous_previous_step.detach()
            pending_loss = None
            pending_count = 0

    values = np.asarray(losses, dtype=np.float64)

    return {
        "loss": float(values[:, 0].mean()),
        "progress": float(values[:, 1].mean()),
        "position": float(values[:, 2].mean()),
        "step": float(values[:, 3].mean()),
        "candidate": float(values[:, 4].mean()),
        "heading": float(values[:, 5].mean()),
        "ahead": float(values[:, 6].mean()),
        "capture_pct": float(100.0 * np.mean(capture_rows)),
        "mean_step_m": float(np.mean(step_rows)),
        "mean_ahead_m": float(np.mean(ahead_rows)),
    }


def make_progress_kalman(initial_s):
    if KalmanFilter is None:
        raise ImportError("FilterPy is required: pip install filterpy")

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.asarray([float(initial_s)], dtype=np.float64)
    kf.F = np.eye(1, dtype=np.float64)
    kf.H = np.eye(1, dtype=np.float64)
    kf.P = np.asarray(
        [[float(config.KALMAN_INIT_PROGRESS_VAR)]],
        dtype=np.float64,
    )
    kf.Q = np.asarray(
        [[float(config.KALMAN_Q_PROGRESS)]],
        dtype=np.float64,
    )
    kf.R = np.asarray([[3.0]], dtype=np.float64)
    return kf


@torch.no_grad()
def closed_loop_rollout(
    model,
    visual,
    cache,
    path,
    device,
    collect_rows=False,
    use_kalman=False,
):
    model.eval()

    hidden = None
    progress_s = torch.zeros(1, 1, dtype=torch.float32, device=device)
    previous_step = torch.zeros_like(progress_s)
    previous_previous_step = torch.zeros_like(progress_s)
    heading_residual = torch.zeros(1, 1, dtype=torch.float32, device=device)

    if use_kalman:
        kf = make_progress_kalman(0.0)
        final_s_value = 0.0
    else:
        kf = None
        final_s_value = 0.0

    visual_predictions = []
    final_predictions = []
    rows = []

    for index in range(len(cache)):
        if use_kalman:
            progress_s = torch.tensor(
                [[final_s_value]],
                dtype=torch.float32,
                device=device,
            )
        search_s = float(progress_s.cpu().item())

        center_xy = path.xy_at_s_torch(progress_s)
        base_heading = path.base_heading_rad(search_s)
        search_heading = base_heading + float(heading_residual.cpu().item())

        uav_clip = cache.uav_clip[index : index + 1].to(device).float()

        candidate = build_forward_candidate(
            visual,
            uav_clip,
            center_xy,
            search_heading,
        )
        z_uav = visual.model.encode_uav_from_clip(uav_clip)

        output = model.forward_step(
            z_uav=z_uav,
            z_sat=candidate.z_sat,
            raw_logits=candidate.raw_logits,
            hidden=hidden,
        )

        selected_index = int(output.refined_logits[0].argmax().cpu().item())
        selected_xy = candidate.centers[0, selected_index]

        candidate_s, route_distance = selected_candidate_progress(
            path,
            selected_xy.cpu(),
            search_s,
        )

        visual_s, step, visual_step, polynomial_step, inertia_cap = (
            bounded_visual_progress_update(
                current_s=progress_s,
                candidate_s=candidate_s,
                move_gate=output.move_gate,
                previous_step=previous_step,
                previous_previous_step=previous_previous_step,
            )
        )
        visual_s = visual_s.clamp(max=float(path.total_length))
        visual_xy = path.xy_at_s_torch(visual_s)[0]
        visual_s_value = float(visual_s.cpu().item())

        if use_kalman:
            previous_final_s = float(final_s_value)
            kf.predict()
            variance = float(output.measurement_variance[0, 0].cpu().item())
            kf.R = np.asarray([[variance]], dtype=np.float64)
            kf.update(np.asarray([visual_s_value], dtype=np.float64))

            filtered = float(np.asarray(kf.x).reshape(-1)[0])
            final_s_value = float(
                np.clip(
                    filtered,
                    previous_final_s,
                    min(
                        path.total_length,
                        previous_final_s
                        + float(config.MAX_STEP_M_PER_FRAME),
                    ),
                )
            )
            kf.x = np.asarray([final_s_value], dtype=np.float64)
            final_xy = path.xy_at_s(final_s_value).numpy()
        else:
            final_s_value = visual_s_value
            final_xy = visual_xy.cpu().numpy()

        visual_np = visual_xy.cpu().numpy()
        visual_predictions.append(visual_np)
        final_predictions.append(final_xy)

        # Prediction is complete before GT is read. GT below is evaluation only.
        gt_np = cache.gt_xy[index].numpy()

        base_heading_deg = path.base_heading_deg(final_s_value)
        heading_residual_deg = math.degrees(
            float(output.heading_residual_rad[0, 0].cpu().item())
        )
        estimated_heading_deg = base_heading_deg + heading_residual_deg
        active_leg = path.leg_index_from_s(final_s_value)

        probability = output.candidate_probability[0].cpu()
        top2 = probability.topk(k=min(2, int(probability.shape[0]))).values
        margin = float(
            top2[0]
            - (top2[1] if top2.shape[0] > 1 else 0.0)
        )

        row = {
            "sequence_index": int(index),
            "frame_id": int(cache.frame_ids[index]),
            "image_path": cache.image_paths[index],
            "active_waypoint_from": int(active_leg),
            "active_waypoint_to": int(active_leg + 1),
            "gt_x": float(gt_np[0]),
            "gt_y": float(gt_np[1]),
            "visual_x": float(visual_np[0]),
            "visual_y": float(visual_np[1]),
            "final_x": float(final_xy[0]),
            "final_y": float(final_xy[1]),
            "selected_patch_x": float(selected_xy[0].cpu()),
            "selected_patch_y": float(selected_xy[1].cpu()),
            "progress_s_visual_m": float(visual_s_value),
            "progress_s_final_m": float(final_s_value),
            "selected_patch_progress_s_m": float(candidate_s),
            "predicted_step_m": float(step[0, 0].cpu()),
            "visual_step_cap_m": float(visual_step[0, 0].cpu()),
            "polynomial_step_m": float(polynomial_step[0, 0].cpu()),
            "inertia_cap_m": float(inertia_cap[0, 0].cpu()),
            "move_gate": float(output.move_gate[0, 0].cpu()),
            "candidate_probability_max": float(probability.max()),
            "candidate_probability_margin": float(margin),
            "candidate_route_distance_m": float(route_distance),
            "route_heading_deg": float(base_heading_deg),
            "heading_residual_deg": float(heading_residual_deg),
            "estimated_heading_deg": float(estimated_heading_deg),
            "search_heading_deg": float(heading_deg_from_rad(search_heading)),
            "measurement_variance": float(
                output.measurement_variance[0, 0].cpu()
            ),
            "forward_candidate_count": int(candidate.centers.shape[1]),
            "error_visual_m": float(np.linalg.norm(visual_np - gt_np)),
            "error_final_m": float(np.linalg.norm(final_xy - gt_np)),
        }

        if collect_rows:
            rows.append(row)

        hidden = output.hidden
        heading_residual = output.heading_residual_rad
        previous_previous_step = previous_step
        previous_step = step
        progress_s = visual_s

    return (
        np.asarray(visual_predictions, dtype=np.float64),
        np.asarray(final_predictions, dtype=np.float64),
        rows,
    )


@torch.no_grad()
def evaluate_validation(model, visual, cache, path, val_start, device):
    visual_pred, _, _ = closed_loop_rollout(
        model=model,
        visual=visual,
        cache=cache,
        path=path,
        device=device,
        collect_rows=False,
        use_kalman=False,
    )

    gt = cache.gt_xy.numpy()
    error = np.linalg.norm(
        visual_pred[int(val_start) :] - gt[int(val_start) :],
        axis=1,
    )

    return {
        "mle": float(np.mean(error)),
        "p90": float(np.quantile(error, 0.90)),
        "lsr15": float(100.0 * np.mean(error <= 15.0)),
    }


def train_temporal_model(
    model,
    visual,
    cache,
    path,
    device,
    epochs,
):
    gt_s = path.sequential_gt_progress(cache.gt_xy)
    train_end = max(
        8,
        int(len(cache) * float(config.TEMPORAL_TRAIN_FRACTION)),
    )
    val_start = train_end

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    best_score = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("TEMPORAL v11: Route-A only", flush=True)
    print("  neural unit       = nn.RNNCell (NO LSTM / NO GRU)", flush=True)
    print(
        "  model inputs      = current UAV/SAT image features + previous hidden",
        flush=True,
    )
    print("  GT as model input = NEVER", flush=True)
    print("  XY teacher        = 0.00 from epoch 1", flush=True)
    print(
        "  progress          = continuous scalar s, no discrete leg classifier",
        flush=True,
    )
    print(
        "  max movement      = %.2f m/frame; zero motion is valid"
        % float(config.MAX_STEP_M_PER_FRAME),
        flush=True,
    )
    print(
        "  local search      = forward half of 6x6 = %d patches"
        % int(config.FORWARD_CANDIDATE_COUNT),
        flush=True,
    )

    for epoch in range(int(epochs)):
        training = train_one_epoch(
            model=model,
            optimizer=optimizer,
            visual=visual,
            cache=cache,
            path=path,
            gt_s=gt_s,
            train_end=train_end,
            device=device,
        )

        validation = evaluate_validation(
            model=model,
            visual=visual,
            cache=cache,
            path=path,
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

        torch.save(
            {
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
                "max_step_m_per_frame": float(config.MAX_STEP_M_PER_FRAME),
                "forward_candidate_count": int(config.FORWARD_CANDIDATE_COUNT),
                "teacher_forcing_ratio": 0.0,
            },
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            "epoch=%03d/%d "
            "loss=%.4f progress=%.4f pos=%.4f step=%.4f "
            "cand=%.4f ahead=%.4f cap=%.1f%% "
            "meanStep=%.3fm meanAhead=%.3fm "
            "valMLE=%.3fm valP90=%.3fm valLSR15=%.2f%% "
            "best=%03d@%.3fm patience=%d"
            % (
                epoch + 1,
                int(epochs),
                training["loss"],
                training["progress"],
                training["position"],
                training["step"],
                training["candidate"],
                training["ahead"],
                training["capture_pct"],
                training["mean_step_m"],
                training["mean_ahead_m"],
                validation["mle"],
                validation["p90"],
                validation["lsr15"],
                best_epoch,
                best_score,
                patience,
            ),
            flush=True,
        )
        print("checkpoint:", config.TEMPORAL_CHECKPOINT, flush=True)

        if (
            epoch + 1 >= int(config.TEMPORAL_MIN_EPOCHS_BEFORE_STOP)
            and patience >= int(config.TEMPORAL_EARLY_STOPPING_PATIENCE)
        ):
            print(
                "temporal early stopping: closed-loop Route-A validation stopped improving",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Temporal training did not produce a best checkpoint")

    final_payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    final_payload["model"] = best_state
    final_payload["best_model"] = best_state
    final_payload["best_epoch"] = best_epoch
    final_payload["best_val_mle"] = best_score
    torch.save(final_payload, config.TEMPORAL_CHECKPOINT)

    model.load_state_dict(best_state)
    print(
        "best temporal checkpoint: epoch=%d valMLE=%.3fm"
        % (best_epoch, best_score),
        flush=True,
    )


def load_temporal_model(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Temporal checkpoint not found: %s" % config.TEMPORAL_CHECKPOINT
        )

    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            "Temporal checkpoint architecture mismatch: %r"
            % checkpoint.get("architecture")
        )
    if checkpoint.get("current_gt_as_model_input") is not False:
        raise RuntimeError(
            "Checkpoint does not prove current GT was excluded from inputs"
        )

    state = checkpoint.get("best_model") or checkpoint["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(
        "loaded temporal best checkpoint epoch=%s valMLE=%s"
        % (
            checkpoint.get("best_epoch"),
            checkpoint.get("best_val_mle"),
        ),
        flush=True,
    )


def metric_block(prediction, gt):
    error = np.linalg.norm(prediction - gt, axis=1)

    if len(prediction) > 1:
        pred_step = np.linalg.norm(np.diff(prediction, axis=0), axis=1)
        gt_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
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
        "LSR@5_pct": float(100.0 * np.mean(error <= 5.0)),
        "LSR@10_pct": float(100.0 * np.mean(error <= 10.0)),
        "LSR@15_pct": float(100.0 * np.mean(error <= 15.0)),
        "LSR@20_pct": float(100.0 * np.mean(error <= 20.0)),
        "RPE_step_mean_m": float(np.mean(rpe)),
        "JumpRate_pct": float(100.0 * jump),
        "MaxError_m": float(error.max()),
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No inference rows to write")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_inference(model, visual, cache, path, device, csv_path):
    visual_pred, final_pred, rows = closed_loop_rollout(
        model=model,
        visual=visual,
        cache=cache,
        path=path,
        device=device,
        collect_rows=True,
        use_kalman=True,
    )
    write_rows(csv_path, rows)

    gt = cache.gt_xy.numpy()
    return {
        "architecture": ARCHITECTURE_NAME,
        "network": "nn.RNNCell",
        "model_input": "UAV/SAT image features + previous RNN hidden only",
        "current_gt_as_model_input": False,
        "previous_gt_as_model_input": False,
        "test_gt_as_model_input": False,
        "test_waypoint_frame_index_used": False,
        "max_step_m_per_frame": float(config.MAX_STEP_M_PER_FRAME),
        "forward_candidate_count": int(config.FORWARD_CANDIDATE_COUNT),
        "VisualRNN": metric_block(visual_pred, gt),
        "ProgressKalman": metric_block(final_pred, gt),
    }


def route_catalog():
    return {
        name: Path(root)
        for name, root in zip(config.ROUTE_NAMES, config.ROUTE_ROOTS)
    }


def ensure_visual_checkpoint(device, visual_epochs, reuse_visual):
    if reuse_visual and config.VISUAL_CHECKPOINT.exists():
        print("reuse visual checkpoint:", config.VISUAL_CHECKPOINT, flush=True)
        return

    print("training Route-A visual retrieval from scratch", flush=True)
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
    parser.add_argument("--reuse-visual", action="store_true")
    args = parser.parse_args()

    set_seed(config.SEED)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100, flush=True)
    print("CONTINUOUS-PROGRESS VISUAL RNN v11", flush=True)
    print("Plain nn.RNNCell; NO LSTM/GRU; NO discrete leg state.", flush=True)
    print("Known W0 + waypoints -> continuous route progress s.", flush=True)
    print("Current image searches only forward 3x6 SAT patches.", flush=True)
    print(
        "0 <= movement <= %.2f m/frame."
        % float(config.MAX_STEP_M_PER_FRAME),
        flush=True,
    )
    print(
        "Second-order polynomial is a movement CAP, never an autonomous motion source.",
        flush=True,
    )
    print(
        "RNN outputs move gate, heading residual, uncertainty and next hidden state.",
        flush=True,
    )
    print(
        "Visual progress -> scalar position-only Kalman -> route XY.",
        flush=True,
    )
    print(
        "GT is used only for Route-A supervised labels and post-prediction evaluation.",
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
            "eval requires visual checkpoint: %s" % config.VISUAL_CHECKPOINT
        )

    visual = FrozenVisualLocalizer(device)
    routes = route_catalog()

    if args.mode in ("train", "train_eval"):
        cache = build_route_cache(
            "route_A",
            routes["route_A"],
            visual,
            device,
        )
        path = load_mission_path(
            "route_A",
            visual.origin_lat,
            visual.origin_lon,
        )
        model = ContinuousProgressVisualRNN().to(device)

        train_temporal_model(
            model=model,
            visual=visual,
            cache=cache,
            path=path,
            device=device,
            epochs=int(args.temporal_epochs),
        )

    if args.mode in ("eval", "train_eval"):
        model = ContinuousProgressVisualRNN().to(device)
        load_temporal_model(model, device)

        route_results = {}

        for route_name in ["route_B", "route_C"]:
            cache = build_route_cache(
                route_name,
                routes[route_name],
                visual,
                device,
            )
            path = load_mission_path(
                route_name,
                visual.origin_lat,
                visual.origin_lon,
            )

            csv_path = (
                config.OUTPUT_DIR
                / (route_name + "_continuous_progress_rnn_frames.csv")
            )

            summary = run_inference(
                model=model,
                visual=visual,
                cache=cache,
                path=path,
                device=device,
                csv_path=csv_path,
            )
            route_results[route_name] = summary

            metric = summary["ProgressKalman"]
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

        summary_path = config.OUTPUT_DIR / "robust_tracker_summary.json"
        summary_path.write_text(
            json.dumps(route_results, indent=2),
            encoding="utf-8",
        )
        print("summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
