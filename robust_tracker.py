import argparse
import csv
import json
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
from visual_model import (
    CRFCandidateRefiner,
    CRFInertialRNN,
)

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "CRFInertialRNNKalman_v20"


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
class TemporalState:
    hidden: torch.Tensor
    state: torch.Tensor
    state_np: np.ndarray
    previous_final_xy: np.ndarray
    previous_velocity_xy: np.ndarray
    previous_acceleration_xy: np.ndarray


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


def bound_vector_numpy(vector, maximum):
    vector = np.asarray(
        vector,
        dtype=np.float64,
    ).reshape(2)

    norm = float(
        np.linalg.norm(vector)
    )

    if (
        norm <= float(maximum)
        or norm <= 1e-9
    ):
        return vector

    return vector * (
        float(maximum)
        / norm
    )


def tensor_xy(value, device):
    return torch.tensor(
        np.asarray(
            value,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    ).reshape(1, 2)


def load_waypoint_xy(
    route_name,
    origin_lat,
    origin_lon,
):
    path = Path(
        config.WAYPOINT_FILES[route_name]
    )

    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    waypoints = sorted(
        payload["waypoints"],
        key=lambda item: int(
            item["waypoint_order"]
        ),
    )

    rows = []

    for waypoint in waypoints:
        x_m, y_m = meters_from_latlon(
            waypoint["latitude"],
            waypoint["longitude"],
            origin_lat,
            origin_lon,
        )

        rows.append(
            [
                float(x_m),
                float(y_m),
            ]
        )

    if not rows:
        raise RuntimeError(
            "%s contains no waypoint"
            % route_name
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


@torch.no_grad()
def build_route_cache(
    route_name,
    root,
    visual,
    device,
):
    stat = (
        config.VISUAL_CHECKPOINT.stat()
    )

    signature = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "architecture": ARCHITECTURE_NAME,
    }

    cache_path = (
        config.OUTPUT_DIR
        / "feature_cache"
        / (
            route_name
            + "_uav_clip.pt"
        )
    )

    if cache_path.exists():
        payload = torch.load(
            cache_path,
            map_location="cpu",
        )

        if (
            payload.get("signature")
            == signature
        ):
            print(
                "%s: reuse UAV backbone cache"
                % route_name,
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
            for index
            in range(
                start,
                end,
            )
        ]

        uav = torch.stack(
            [
                item["uav"]
                for item in items
            ]
        ).to(device)

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
                    item["xy"].float()
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

            image_paths.append(
                str(
                    item["image_path"]
                )
            )

        if (
            start == 0
            or end == len(dataset)
            or (
                start // batch_size
            ) % 10 == 0
        ):
            print(
                "%s backbone cache: %d/%d"
                % (
                    route_name,
                    end,
                    len(dataset),
                ),
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
            "signature": signature,
            "frame_ids": result.frame_ids,
            "gt_xy": result.gt_xy,
            "uav_clip": result.uav_clip,
            "image_paths": result.image_paths,
        },
        cache_path,
    )

    return result


class ExternalKalman2D:
    """
    Final localization filter.

    State:
        [x, y, vx, vy]

    It never replaces the visual measurement model. It predicts and then
    updates using the RNN/HardMS visual measurement.
    """

    def __init__(self, initial_xy):
        if KalmanFilter is None:
            raise ImportError(
                "FilterPy is required. Install with: pip install filterpy"
            )

        self.kf = KalmanFilter(
            dim_x=4,
            dim_z=2,
        )

        self.kf.x = np.asarray(
            [
                float(initial_xy[0]),
                float(initial_xy[1]),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        self.kf.F = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.kf.H = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.kf.P = np.diag(
            [
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
            ]
        ).astype(np.float64)

        self.kf.Q = np.diag(
            [
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_VELOCITY),
                float(config.KALMAN_Q_VELOCITY),
            ]
        ).astype(np.float64)

        self.kf.R = (
            np.eye(
                2,
                dtype=np.float64,
            )
            * 2.0
        )

    def state_xy(self):
        return np.asarray(
            self.kf.x[:2],
            dtype=np.float64,
        ).reshape(2)

    def state_velocity(self):
        return np.asarray(
            self.kf.x[2:4],
            dtype=np.float64,
        ).reshape(2)

    def predict(self):
        previous_xy = (
            self.state_xy().copy()
        )

        velocity = bound_vector_numpy(
            self.state_velocity(),
            float(
                config.MAX_FINAL_STEP_M_PER_FRAME
            ),
        )

        self.kf.x[2] = velocity[0]
        self.kf.x[3] = velocity[1]

        self.kf.predict()

        predicted_xy = (
            self.state_xy()
        )

        bounded_step = (
            bound_vector_numpy(
                predicted_xy
                - previous_xy,
                float(
                    config.MAX_FINAL_STEP_M_PER_FRAME
                ),
            )
        )

        bounded_xy = (
            previous_xy
            + bounded_step
        )

        self.kf.x[0] = bounded_xy[0]
        self.kf.x[1] = bounded_xy[1]

        return (
            bounded_xy.copy(),
            self.state_velocity().copy(),
        )

    def update(
        self,
        measurement_xy,
        variance_xy,
    ):
        previous_xy = (
            self.state_xy().copy()
        )

        variance = np.asarray(
            variance_xy,
            dtype=np.float64,
        ).reshape(2)

        variance = np.clip(
            variance,
            float(config.KALMAN_R_MIN_VAR),
            float(config.KALMAN_R_MAX_VAR),
        )

        self.kf.R = (
            np.diag(
                variance
            )
        )

        self.kf.update(
            np.asarray(
                measurement_xy,
                dtype=np.float64,
            ).reshape(2)
        )

        updated_xy = (
            self.state_xy()
        )

        bounded_step = (
            bound_vector_numpy(
                updated_xy
                - previous_xy,
                float(
                    config.MAX_FINAL_STEP_M_PER_FRAME
                ),
            )
        )

        bounded_xy = (
            previous_xy
            + bounded_step
        )

        velocity = (
            bound_vector_numpy(
                self.state_velocity(),
                float(
                    config.MAX_FINAL_STEP_M_PER_FRAME
                ),
            )
        )

        self.kf.x[0] = bounded_xy[0]
        self.kf.x[1] = bounded_xy[1]
        self.kf.x[2] = velocity[0]
        self.kf.x[3] = velocity[1]

        return (
            bounded_xy.copy(),
            velocity.copy(),
        )


def target_next_step(
    cache,
    index,
    device,
):
    if (
        index + 1
        >= len(cache)
    ):
        return torch.zeros(
            1,
            2,
            dtype=torch.float32,
            device=device,
        )

    step = (
        cache.gt_xy[index + 1]
        - cache.gt_xy[index]
    ).to(device).reshape(1, 2)

    norm = torch.linalg.norm(
        step,
        dim=1,
        keepdim=True,
    )

    scale = torch.clamp(
        float(
            config.MAX_POLYNOMIAL_STEP_M_PER_FRAME
        )
        / norm.clamp_min(
            1e-6
        ),
        max=1.0,
    )

    return (
        step * scale
    )


def target_velocity_acceleration(
    cache,
    index,
    device,
):
    """
    Discrete constant-acceleration target:

        previous displacement = d_prev
        next displacement     = d_next

        v_t = 0.5 * (d_prev + d_next)
        a_t = d_next - d_prev

    therefore:
        v_t + 0.5*a_t = d_next
    """

    next_step = target_next_step(
        cache,
        index,
        device,
    )

    if index <= 0:
        previous_step = next_step.detach()
    else:
        previous_step = (
            cache.gt_xy[index]
            - cache.gt_xy[index - 1]
        ).to(device).reshape(1, 2)

        previous_norm = (
            torch.linalg.norm(
                previous_step,
                dim=1,
                keepdim=True,
            )
        )

        previous_scale = torch.clamp(
            float(
                config.MAX_POLYNOMIAL_STEP_M_PER_FRAME
            )
            / previous_norm.clamp_min(
                1e-6
            ),
            max=1.0,
        )

        previous_step = (
            previous_step
            * previous_scale
        )

    velocity = (
        0.5
        * (
            previous_step
            + next_step
        )
    )

    acceleration = (
        next_step
        - previous_step
    )

    # Bound velocity magnitude without losing direction.
    raw_velocity = (
        0.5
        * (
            previous_step
            + next_step
        )
    )

    raw_velocity_norm = (
        torch.linalg.norm(
            raw_velocity,
            dim=1,
            keepdim=True,
        )
    )

    raw_velocity_scale = torch.clamp(
        float(
            config.MAX_MODEL_VELOCITY_M_PER_FRAME
        )
        / raw_velocity_norm.clamp_min(
            1e-6
        ),
        max=1.0,
    )

    bounded_velocity = (
        raw_velocity
        * raw_velocity_scale
    )

    acceleration_norm = (
        torch.linalg.norm(
            acceleration,
            dim=1,
            keepdim=True,
        )
    )

    acceleration_scale = torch.clamp(
        float(
            config.MAX_MODEL_ACCELERATION_M_PER_FRAME2
        )
        / acceleration_norm.clamp_min(
            1e-6
        ),
        max=1.0,
    )

    bounded_acceleration = (
        acceleration
        * acceleration_scale
    )

    return (
        bounded_velocity,
        bounded_acceleration,
        next_step,
    )


def gaussian_candidate_target(
    centers,
    gt_xy,
):
    distance2 = (
        centers
        - gt_xy[:, None, :]
    ).square().sum(dim=-1)

    sigma2 = float(
        config.CANDIDATE_TARGET_SIGMA_M
    ) ** 2

    logits = (
        -0.5
        * distance2
        / sigma2
    )

    return torch.softmax(
        logits,
        dim=1,
    )


def candidate_soft_ce(
    refined_logits,
    target_probability,
):
    log_probability = torch.log_softmax(
        refined_logits,
        dim=1,
    )

    return -(
        target_probability
        * log_probability
    ).sum(dim=1).mean()


def candidate_capture(
    centers,
    gt_xy,
):
    nearest = torch.linalg.norm(
        centers
        - gt_xy[:, None, :],
        dim=2,
    ).min(
        dim=1
    ).values

    capture = (
        nearest
        <= float(
            config.CANDIDATE_CAPTURE_RADIUS_M
        )
    )

    return (
        capture,
        nearest,
    )


def refined_hardms(
    refined_logits,
    centers,
):
    return hard_mean_shift(
        refined_logits,
        centers,
        float(
            config.MEANSHIFT_SCORE_TAU
        ),
        float(
            config.MEANSHIFT_BANDWIDTH_M
        ),
        int(
            config.MEANSHIFT_ITERATIONS
        ),
    )


def decode_state_motion(
    previous_state_np,
):
    if previous_state_np is None:
        return (
            np.zeros(
                2,
                dtype=np.float64,
            ),
            np.zeros(
                2,
                dtype=np.float64,
            ),
        )

    state = np.asarray(
        previous_state_np,
        dtype=np.float64,
    ).reshape(-1)

    if int(
        state.shape[0]
    ) != int(
        config.RNN_STATE_DIM
    ):
        raise RuntimeError(
            "state dimension mismatch"
        )

    velocity = (
        state[0:2]
        * float(
            config.MAX_MODEL_VELOCITY_M_PER_FRAME
        )
    )

    acceleration = (
        state[2:4]
        * float(
            config.MAX_MODEL_ACCELERATION_M_PER_FRAME2
        )
    )

    return (
        bound_vector_numpy(
            velocity,
            float(
                config.MAX_MODEL_VELOCITY_M_PER_FRAME
            ),
        ),
        bound_vector_numpy(
            acceleration,
            float(
                config.MAX_MODEL_ACCELERATION_M_PER_FRAME2
            ),
        ),
    )


def polynomial_step_from_state(
    previous_state_np,
):
    (
        velocity,
        acceleration,
    ) = decode_state_motion(
        previous_state_np
    )

    step = (
        velocity
        + 0.5
        * acceleration
    )

    return (
        bound_vector_numpy(
            step,
            float(
                config.MAX_POLYNOMIAL_STEP_M_PER_FRAME
            ),
        ),
        velocity,
        acceleration,
    )


def make_stage_models(device):
    refiner = (
        CRFCandidateRefiner()
        .to(device)
    )

    temporal = (
        CRFInertialRNN()
        .to(device)
    )

    return (
        refiner,
        temporal,
    )


def build_step(
    refiner,
    temporal,
    visual,
    uav_clip,
    search_center_xy,
    previous_final_xy,
    previous_state,
    previous_state_np,
    hidden,
):
    device = (
        uav_clip.device
    )

    search_center_xy = (
        search_center_xy.reshape(
            1,
            2,
        )
    )

    previous_final_xy_tensor = (
        tensor_xy(
            previous_final_xy,
            device,
        )
    )

    (
        predicted_step_np,
        previous_velocity_np,
        previous_acceleration_np,
    ) = polynomial_step_from_state(
        previous_state_np
    )

    predicted_step_xy = (
        tensor_xy(
            predicted_step_np,
            device,
        )
    )

    previous_velocity_xy = (
        tensor_xy(
            previous_velocity_np,
            device,
        )
    )

    previous_acceleration_xy = (
        tensor_xy(
            previous_acceleration_np,
            device,
        )
    )

    candidate = (
        visual.candidate_batch(
            uav_clip,
            search_center_xy,
            grid_size=int(
                config.GRID_SIZE
            ),
        )
    )

    if previous_state is None:
        previous_state = (
            temporal.initial_state(
                1,
                device,
                candidate.z_uav.dtype,
            )
        )

    refinement = refiner(
        z_uav=candidate.z_uav,
        z_sat=candidate.z_sat,
        raw_logits=candidate.raw_logits,
        raw_prob=candidate.raw_prob,
        centers=candidate.centers,
        search_center_xy=search_center_xy,
        previous_final_xy=previous_final_xy_tensor,
        predicted_step_xy=predicted_step_xy,
        previous_state=previous_state,
    )

    (
        refined_xy,
        refined_support,
    ) = refined_hardms(
        refinement.refined_logits,
        candidate.centers,
    )

    output = temporal.forward_step(
        z_uav=candidate.z_uav,
        sat_context=refinement.sat_context,
        raw_probability=candidate.raw_prob,
        refined_probability=refinement.refined_probability,
        refined_hardms_xy=refined_xy,
        refined_hardms_support=refined_support,
        raw_hardms_xy=candidate.hardms_xy,
        raw_top1_xy=candidate.raw_top1_xy,
        search_center_xy=search_center_xy,
        previous_final_xy=previous_final_xy_tensor,
        predicted_step_xy=predicted_step_xy,
        previous_velocity_xy=previous_velocity_xy,
        previous_acceleration_xy=previous_acceleration_xy,
        previous_state=previous_state,
        hidden=hidden,
    )

    return (
        candidate,
        refinement,
        refined_xy,
        refined_support,
        output,
        predicted_step_np,
    )


def losses_for_step(
    refinement,
    candidate,
    refined_xy,
    output,
    gt_xy,
    velocity_target,
    acceleration_target,
    next_step_target,
):
    target_probability = (
        gaussian_candidate_target(
            candidate.centers,
            gt_xy,
        )
    )

    candidate_loss = (
        candidate_soft_ce(
            refinement.refined_logits,
            target_probability,
        )
    )

    measurement_loss = (
        F.smooth_l1_loss(
            output.measurement_xy,
            gt_xy,
        )
    )

    next_step_loss = (
        F.smooth_l1_loss(
            output.next_step_xy,
            next_step_target,
        )
    )

    velocity_loss = (
        F.smooth_l1_loss(
            output.velocity_xy,
            velocity_target,
        )
    )

    acceleration_loss = (
        F.smooth_l1_loss(
            output.acceleration_xy,
            acceleration_target,
        )
    )

    variance = (
        output.measurement_variance
        .clamp_min(
            float(
                config.KALMAN_R_MIN_VAR
            )
        )
    )

    error = (
        output.measurement_xy
        - gt_xy
    )

    variance_nll = (
        0.5
        * (
            error.square()
            / variance
            + variance.log()
        )
    ).mean()

    correction_reg = (
        output.correction_gate
        * torch.linalg.norm(
            output.correction_xy,
            dim=1,
            keepdim=True,
        )
    ).mean()

    gate_reg = (
        output.correction_gate.mean()
    )

    total = (
        float(
            config.LOSS_CANDIDATE
        )
        * candidate_loss

        + float(
            config.LOSS_MEASUREMENT
        )
        * measurement_loss

        + float(
            config.LOSS_NEXT_STEP
        )
        * next_step_loss

        + float(
            config.LOSS_VELOCITY
        )
        * velocity_loss

        + float(
            config.LOSS_ACCELERATION
        )
        * acceleration_loss

        + float(
            config.LOSS_VARIANCE_NLL
        )
        * variance_nll

        + float(
            config.LOSS_CORRECTION_REG
        )
        * correction_reg

        + float(
            config.LOSS_GATE_REG
        )
        * gate_reg
    )

    return {
        "total": total,
        "candidate": candidate_loss,
        "measurement": measurement_loss,
        "next_step": next_step_loss,
        "velocity": velocity_loss,
        "acceleration": acceleration_loss,
        "variance": variance_nll,
        "correction": correction_reg,
    }


def train_warmup_epoch(
    refiner,
    temporal,
    optimizer,
    visual,
    cache,
    train_end,
    device,
):
    refiner.train()
    temporal.train()

    hidden = None
    previous_state = None
    previous_state_np = None

    previous_final_xy = (
        cache.gt_xy[0]
        .numpy()
        .astype(
            np.float64
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    pending_loss = None
    pending_count = 0

    rows = []

    for index in range(
        max(
            1,
            int(train_end) - 1,
        )
    ):
        gt_xy = (
            cache.gt_xy[
                index:index + 1
            ].to(device)
        )

        center = (
            gt_xy.detach()
            + torch.randn_like(
                gt_xy
            )
            * float(
                config.WARMUP_SEARCH_JITTER_M
            )
        )

        uav_clip = (
            cache.uav_clip[
                index:index + 1
            ].to(device).float()
        )

        (
            candidate,
            refinement,
            refined_xy,
            refined_support,
            output,
            predicted_step_np,
        ) = build_step(
            refiner=refiner,
            temporal=temporal,
            visual=visual,
            uav_clip=uav_clip,
            search_center_xy=center,
            previous_final_xy=previous_final_xy,
            previous_state=previous_state,
            previous_state_np=previous_state_np,
            hidden=hidden,
        )

        (
            velocity_target,
            acceleration_target,
            next_step_target,
        ) = target_velocity_acceleration(
            cache,
            index,
            device,
        )

        losses = losses_for_step(
            refinement=refinement,
            candidate=candidate,
            refined_xy=refined_xy,
            output=output,
            gt_xy=gt_xy,
            velocity_target=velocity_target,
            acceleration_target=acceleration_target,
            next_step_target=next_step_target,
        )

        pending_loss = (
            losses["total"]
            if pending_loss is None
            else pending_loss
            + losses["total"]
        )

        pending_count += 1

        (
            capture,
            nearest,
        ) = candidate_capture(
            candidate.centers,
            gt_xy,
        )

        raw_error = (
            torch.linalg.norm(
                candidate.hardms_xy
                - gt_xy,
                dim=1,
            )
        )

        refined_error = (
            torch.linalg.norm(
                refined_xy
                - gt_xy,
                dim=1,
            )
        )

        measurement_error = (
            torch.linalg.norm(
                output.measurement_xy
                - gt_xy,
                dim=1,
            )
        )

        next_error = (
            torch.linalg.norm(
                output.next_step_xy
                - next_step_target,
                dim=1,
            )
        )

        rows.append(
            [
                float(
                    losses["total"]
                    .detach()
                    .cpu()
                ),
                float(
                    losses["candidate"]
                    .detach()
                    .cpu()
                ),
                float(
                    losses["measurement"]
                    .detach()
                    .cpu()
                ),
                float(
                    losses["next_step"]
                    .detach()
                    .cpu()
                ),
                float(
                    100.0
                    * capture.float()
                    .mean()
                    .cpu()
                ),
                float(
                    nearest.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    raw_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    refined_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    measurement_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    next_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    torch.linalg.norm(
                        output.next_step_xy,
                        dim=1,
                    ).mean()
                    .detach()
                    .cpu()
                ),
                float(
                    output.correction_gate
                    .mean()
                    .detach()
                    .cpu()
                ),
            ]
        )

        # Stage-1 teacher location is ONLY for current local candidate extraction.
        # The recurrent state itself is still propagated from previous model output.
        hidden = (
            output.hidden
        )

        previous_state = (
            output.state
        )

        previous_state_np = (
            output.state[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        previous_final_xy = (
            output.measurement_xy[0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        boundary = (
            pending_count
            >= int(
                config.TBPTT_STEPS
            )
            or index
            == int(train_end) - 2
        )

        if boundary:
            objective = (
                pending_loss
                / float(
                    pending_count
                )
            )

            if not torch.isfinite(
                objective
            ):
                raise FloatingPointError(
                    "non-finite warmup loss"
                )

            objective.backward()

            torch.nn.utils.clip_grad_norm_(
                list(
                    refiner.parameters()
                )
                + list(
                    temporal.parameters()
                ),
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

            previous_state = (
                previous_state.detach()
            )

            pending_loss = None
            pending_count = 0

    values = np.asarray(
        rows,
        dtype=np.float64,
    )

    return {
        "loss": float(
            values[:, 0].mean()
        ),
        "candidate": float(
            values[:, 1].mean()
        ),
        "measurement": float(
            values[:, 2].mean()
        ),
        "next_step": float(
            values[:, 3].mean()
        ),
        "capture_pct": float(
            values[:, 4].mean()
        ),
        "candidate_distance_m": float(
            values[:, 5].mean()
        ),
        "raw_hardms_error_m": float(
            values[:, 6].mean()
        ),
        "refined_hardms_error_m": float(
            values[:, 7].mean()
        ),
        "measurement_error_m": float(
            values[:, 8].mean()
        ),
        "next_error_m": float(
            values[:, 9].mean()
        ),
        "predicted_step_m": float(
            values[:, 10].mean()
        ),
        "gate": float(
            values[:, 11].mean()
        ),
    }


def rollout_training_chunk(
    refiner,
    temporal,
    optimizer,
    visual,
    cache,
    start_index,
    end_index,
    device,
):
    """
    GT is used only for the first position of the chunk.

    frame start+1 ... end:
      previous model state
      -> polynomial next center
      -> current 6x6
      -> candidate CRF refinement
      -> HardMS
      -> RNN
      -> external Kalman
    """

    initial_xy = (
        cache.gt_xy[
            start_index
        ].numpy()
        .astype(
            np.float64
        )
    )

    kalman = (
        ExternalKalman2D(
            initial_xy
        )
    )

    hidden = None
    previous_state = None
    previous_state_np = None
    previous_final_xy = (
        initial_xy.copy()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    pending_loss = None
    pending_count = 0
    rows = []

    for local_offset, index in enumerate(
        range(
            start_index,
            end_index,
        )
    ):
        (
            kf_prediction_np,
            _,
        ) = kalman.predict()

        if local_offset == 0:
            search_center_np = (
                initial_xy.copy()
            )
        else:
            (
                predicted_step_np,
                _,
                _,
            ) = polynomial_step_from_state(
                previous_state_np
            )

            # This is the requested "forward patch search":
            # full 6x6 is moved forward by the RNN polynomial.
            search_center_np = (
                previous_final_xy
                + predicted_step_np
            )

        search_center = (
            tensor_xy(
                search_center_np,
                device,
            )
        )

        uav_clip = (
            cache.uav_clip[
                index:index + 1
            ].to(device).float()
        )

        (
            candidate,
            refinement,
            refined_xy,
            refined_support,
            output,
            predicted_step_np,
        ) = build_step(
            refiner=refiner,
            temporal=temporal,
            visual=visual,
            uav_clip=uav_clip,
            search_center_xy=search_center,
            previous_final_xy=previous_final_xy,
            previous_state=previous_state,
            previous_state_np=previous_state_np,
            hidden=hidden,
        )

        gt_xy = (
            cache.gt_xy[
                index:index + 1
            ].to(device)
        )

        (
            velocity_target,
            acceleration_target,
            next_step_target,
        ) = target_velocity_acceleration(
            cache,
            index,
            device,
        )

        losses = losses_for_step(
            refinement=refinement,
            candidate=candidate,
            refined_xy=refined_xy,
            output=output,
            gt_xy=gt_xy,
            velocity_target=velocity_target,
            acceleration_target=acceleration_target,
            next_step_target=next_step_target,
        )

        pending_loss = (
            losses["total"]
            if pending_loss is None
            else pending_loss
            + losses["total"]
        )

        pending_count += 1

        (
            capture,
            nearest,
        ) = candidate_capture(
            candidate.centers,
            gt_xy,
        )

        prediction_error = (
            torch.linalg.norm(
                search_center
                - gt_xy,
                dim=1,
            )
        )

        raw_error = (
            torch.linalg.norm(
                candidate.hardms_xy
                - gt_xy,
                dim=1,
            )
        )

        refined_error = (
            torch.linalg.norm(
                refined_xy
                - gt_xy,
                dim=1,
            )
        )

        next_error = (
            torch.linalg.norm(
                output.next_step_xy
                - next_step_target,
                dim=1,
            )
        )

        rows.append(
            [
                float(
                    losses["total"]
                    .detach()
                    .cpu()
                ),
                float(
                    losses["candidate"]
                    .detach()
                    .cpu()
                ),
                float(
                    losses["next_step"]
                    .detach()
                    .cpu()
                ),
                float(
                    100.0
                    * capture.float()
                    .mean()
                    .cpu()
                ),
                float(
                    nearest.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    prediction_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    raw_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    refined_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    next_error.mean()
                    .detach()
                    .cpu()
                ),
                float(
                    torch.linalg.norm(
                        output.next_step_xy,
                        dim=1,
                    ).mean()
                    .detach()
                    .cpu()
                ),
                float(
                    output.correction_gate
                    .mean()
                    .detach()
                    .cpu()
                ),
            ]
        )

        (
            final_xy,
            _,
        ) = kalman.update(
            output.measurement_xy[0]
            .detach()
            .cpu()
            .numpy(),
            output.measurement_variance[0]
            .detach()
            .cpu()
            .numpy(),
        )

        hidden = (
            output.hidden
        )
        previous_state = (
            output.state
        )
        previous_state_np = (
            output.state[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        previous_final_xy = (
            final_xy.copy()
        )

        boundary = (
            pending_count
            >= int(
                config.TBPTT_STEPS
            )
            or index
            == end_index - 1
        )

        if boundary:
            objective = (
                pending_loss
                / float(
                    pending_count
                )
            )

            if not torch.isfinite(
                objective
            ):
                raise FloatingPointError(
                    "non-finite autoregressive loss"
                )

            objective.backward()

            torch.nn.utils.clip_grad_norm_(
                list(
                    refiner.parameters()
                )
                + list(
                    temporal.parameters()
                ),
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
            previous_state = (
                previous_state.detach()
            )

            pending_loss = None
            pending_count = 0

    return rows


def train_rollout_epoch(
    refiner,
    temporal,
    optimizer,
    visual,
    cache,
    train_end,
    horizon,
    device,
):
    refiner.train()
    temporal.train()

    all_rows = []

    start = 0

    while start < int(
        train_end
    ):
        end = min(
            start + int(horizon),
            int(train_end),
        )

        rows = rollout_training_chunk(
            refiner=refiner,
            temporal=temporal,
            optimizer=optimizer,
            visual=visual,
            cache=cache,
            start_index=start,
            end_index=end,
            device=device,
        )

        all_rows.extend(rows)
        start = end

    values = np.asarray(
        all_rows,
        dtype=np.float64,
    )

    return {
        "loss": float(
            values[:, 0].mean()
        ),
        "candidate": float(
            values[:, 1].mean()
        ),
        "next_step": float(
            values[:, 2].mean()
        ),
        "capture_pct": float(
            values[:, 3].mean()
        ),
        "candidate_distance_m": float(
            values[:, 4].mean()
        ),
        "prediction_error_m": float(
            values[:, 5].mean()
        ),
        "raw_hardms_error_m": float(
            values[:, 6].mean()
        ),
        "refined_hardms_error_m": float(
            values[:, 7].mean()
        ),
        "next_error_m": float(
            values[:, 8].mean()
        ),
        "predicted_step_m": float(
            values[:, 9].mean()
        ),
        "gate": float(
            values[:, 10].mean()
        ),
    }


@torch.no_grad()
def rollout_range(
    refiner,
    temporal,
    visual,
    cache,
    start_index,
    end_index,
    initial_xy,
    device,
    collect_rows=False,
):
    refiner.eval()
    temporal.eval()

    kalman = (
        ExternalKalman2D(
            initial_xy
        )
    )

    hidden = None
    previous_state = None
    previous_state_np = None
    previous_final_xy = (
        np.asarray(
            initial_xy,
            dtype=np.float64,
        ).copy()
    )

    predicted_rows = []
    raw_hardms_rows = []
    refined_hardms_rows = []
    measurement_rows = []
    final_rows = []
    capture_rows = []
    frame_rows = []

    for local_offset, index in enumerate(
        range(
            int(start_index),
            int(end_index),
        )
    ):
        (
            kf_prediction_np,
            kf_velocity_np,
        ) = kalman.predict()

        if local_offset == 0:
            search_center_np = (
                np.asarray(
                    initial_xy,
                    dtype=np.float64,
                ).copy()
            )
        else:
            (
                predicted_step_np,
                previous_velocity_np,
                previous_acceleration_np,
            ) = polynomial_step_from_state(
                previous_state_np
            )

            search_center_np = (
                previous_final_xy
                + predicted_step_np
            )

        search_center = (
            tensor_xy(
                search_center_np,
                device,
            )
        )

        uav_clip = (
            cache.uav_clip[
                index:index + 1
            ].to(device).float()
        )

        (
            candidate,
            refinement,
            refined_xy,
            refined_support,
            output,
            predicted_step_np,
        ) = build_step(
            refiner=refiner,
            temporal=temporal,
            visual=visual,
            uav_clip=uav_clip,
            search_center_xy=search_center,
            previous_final_xy=previous_final_xy,
            previous_state=previous_state,
            previous_state_np=previous_state_np,
            hidden=hidden,
        )

        (
            final_xy,
            final_velocity_xy,
        ) = kalman.update(
            output.measurement_xy[0]
            .detach()
            .cpu()
            .numpy(),
            output.measurement_variance[0]
            .detach()
            .cpu()
            .numpy(),
        )

        gt_xy = (
            cache.gt_xy[
                index:index + 1
            ].to(device)
        )

        (
            capture,
            nearest,
        ) = candidate_capture(
            candidate.centers,
            gt_xy,
        )

        gt_np = (
            gt_xy[0]
            .detach()
            .cpu()
            .numpy()
        )

        raw_hardms_np = (
            candidate.hardms_xy[0]
            .detach()
            .cpu()
            .numpy()
        )

        refined_hardms_np = (
            refined_xy[0]
            .detach()
            .cpu()
            .numpy()
        )

        measurement_np = (
            output.measurement_xy[0]
            .detach()
            .cpu()
            .numpy()
        )

        predicted_rows.append(
            search_center_np
        )

        raw_hardms_rows.append(
            raw_hardms_np
        )

        refined_hardms_rows.append(
            refined_hardms_np
        )

        measurement_rows.append(
            measurement_np
        )

        final_rows.append(
            final_xy
        )

        capture_rows.append(
            float(
                capture.float()
                .mean()
                .cpu()
            )
        )

        if collect_rows:
            (
                velocity_np,
                acceleration_np,
            ) = decode_state_motion(
                output.state[0]
                .detach()
                .cpu()
                .numpy()
            )

            state_np = (
                output.state[0]
                .detach()
                .cpu()
                .numpy()
            )

            next_step_np = (
                output.next_step_xy[0]
                .detach()
                .cpu()
                .numpy()
            )

            row = {
                "sequence_index": int(index),
                "frame_id": int(
                    cache.frame_ids[index]
                ),
                "image_path": (
                    cache.image_paths[index]
                ),

                "gt_x": float(
                    gt_np[0]
                ),
                "gt_y": float(
                    gt_np[1]
                ),

                "predicted_current_x": float(
                    search_center_np[0]
                ),
                "predicted_current_y": float(
                    search_center_np[1]
                ),

                "candidate_capture": float(
                    capture.float()
                    .mean()
                    .cpu()
                ),
                "candidate_nearest_gt_m": float(
                    nearest.mean()
                    .cpu()
                ),

                "raw_hardms_x": float(
                    raw_hardms_np[0]
                ),
                "raw_hardms_y": float(
                    raw_hardms_np[1]
                ),

                "hardms_x": float(
                    refined_hardms_np[0]
                ),
                "hardms_y": float(
                    refined_hardms_np[1]
                ),

                "hardms_support": float(
                    refined_support[0]
                    .detach()
                    .cpu()
                ),

                "model_next_step_dx": float(
                    next_step_np[0]
                ),
                "model_next_step_dy": float(
                    next_step_np[1]
                ),
                "model_next_step_m": float(
                    np.linalg.norm(
                        next_step_np
                    )
                ),

                "model_velocity_x": float(
                    velocity_np[0]
                ),
                "model_velocity_y": float(
                    velocity_np[1]
                ),

                "model_acceleration_x": float(
                    acceleration_np[0]
                ),
                "model_acceleration_y": float(
                    acceleration_np[1]
                ),

                "correction_gate": float(
                    output.correction_gate[
                        0,
                        0,
                    ]
                    .detach()
                    .cpu()
                ),

                "measurement_x": float(
                    measurement_np[0]
                ),
                "measurement_y": float(
                    measurement_np[1]
                ),

                "measurement_variance_x": float(
                    output.measurement_variance[
                        0,
                        0,
                    ]
                    .detach()
                    .cpu()
                ),
                "measurement_variance_y": float(
                    output.measurement_variance[
                        0,
                        1,
                    ]
                    .detach()
                    .cpu()
                ),

                "kf_predict_x": float(
                    kf_prediction_np[0]
                ),
                "kf_predict_y": float(
                    kf_prediction_np[1]
                ),

                "final_x": float(
                    final_xy[0]
                ),
                "final_y": float(
                    final_xy[1]
                ),
                "final_vx": float(
                    final_velocity_xy[0]
                ),
                "final_vy": float(
                    final_velocity_xy[1]
                ),
                "final_speed": float(
                    np.linalg.norm(
                        final_velocity_xy
                    )
                ),

                "error_predicted_current_m": float(
                    np.linalg.norm(
                        search_center_np
                        - gt_np
                    )
                ),
                "error_raw_hardms_m": float(
                    np.linalg.norm(
                        raw_hardms_np
                        - gt_np
                    )
                ),
                "error_hardms_m": float(
                    np.linalg.norm(
                        refined_hardms_np
                        - gt_np
                    )
                ),
                "error_measurement_m": float(
                    np.linalg.norm(
                        measurement_np
                        - gt_np
                    )
                ),
                "error_final_m": float(
                    np.linalg.norm(
                        final_xy
                        - gt_np
                    )
                ),
                "state_norm": float(
                    np.linalg.norm(
                        state_np
                    )
                ),
            }

            for state_index, state_value in enumerate(
                state_np
            ):
                row[
                    "state_%02d"
                    % state_index
                ] = float(
                    state_value
                )

            frame_rows.append(row)

        hidden = (
            output.hidden
        )
        previous_state = (
            output.state
        )
        previous_state_np = (
            output.state[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        previous_final_xy = (
            final_xy.copy()
        )

    return (
        np.asarray(
            predicted_rows,
            dtype=np.float64,
        ),
        np.asarray(
            raw_hardms_rows,
            dtype=np.float64,
        ),
        np.asarray(
            refined_hardms_rows,
            dtype=np.float64,
        ),
        np.asarray(
            measurement_rows,
            dtype=np.float64,
        ),
        np.asarray(
            final_rows,
            dtype=np.float64,
        ),
        np.asarray(
            capture_rows,
            dtype=np.float64,
        ),
        frame_rows,
    )


def metrics(
    prediction,
    gt,
):
    error = np.linalg.norm(
        prediction
        - gt,
        axis=1,
    )

    if len(
        prediction
    ) > 1:
        pred_step = np.linalg.norm(
            np.diff(
                prediction,
                axis=0,
            ),
            axis=1,
        )

        gt_step = np.linalg.norm(
            np.diff(
                gt,
                axis=0,
            ),
            axis=1,
        )

        rpe = np.abs(
            pred_step
            - gt_step
        )

        jump = np.mean(
            pred_step
            > (
                gt_step
                + float(
                    config.JUMP_TOLERANCE_M
                )
            )
        )
    else:
        rpe = np.asarray(
            [0.0]
        )
        jump = 0.0

    return {
        "MLE_m": float(
            np.mean(
                error
            )
        ),
        "MedLE_m": float(
            np.median(
                error
            )
        ),
        "P90_m": float(
            np.quantile(
                error,
                0.90,
            )
        ),
        "P95_m": float(
            np.quantile(
                error,
                0.95,
            )
        ),
        "LSR@5_pct": float(
            100.0
            * np.mean(
                error <= 5.0
            )
        ),
        "LSR@10_pct": float(
            100.0
            * np.mean(
                error <= 10.0
            )
        ),
        "LSR@15_pct": float(
            100.0
            * np.mean(
                error <= 15.0
            )
        ),
        "LSR@20_pct": float(
            100.0
            * np.mean(
                error <= 20.0
            )
        ),
        "RPE_step_mean_m": float(
            np.mean(
                rpe
            )
        ),
        "JumpRate_pct": float(
            100.0
            * jump
        ),
        "MaxError_m": float(
            np.max(
                error
            )
        ),
    }


def validation_episode_starts(
    cache_length,
    val_start,
):
    episode_length = int(
        config.VAL_EPISODE_LENGTH
    )

    latest_start = max(
        int(val_start),
        int(cache_length)
        - episode_length,
    )

    if latest_start <= int(
        val_start
    ):
        return [
            int(
                val_start
            )
        ]

    count = max(
        1,
        int(
            config.VAL_EPISODE_COUNT
        ),
    )

    values = np.linspace(
        int(val_start),
        int(latest_start),
        num=count,
    )

    return sorted(
        set(
            int(
                round(value)
            )
            for value in values
        )
    )


@torch.no_grad()
def evaluate_episodes(
    refiner,
    temporal,
    visual,
    cache,
    val_start,
    device,
):
    starts = (
        validation_episode_starts(
            len(cache),
            val_start,
        )
    )

    final_errors = []
    prediction_errors = []
    hardms_errors = []
    capture_values = []

    for start in starts:
        end = min(
            start
            + int(
                config.VAL_EPISODE_LENGTH
            ),
            len(cache),
        )

        initial_xy = (
            cache.gt_xy[start]
            .numpy()
            .astype(
                np.float64
            )
        )

        (
            predicted,
            raw_hardms,
            refined_hardms,
            measurement,
            final,
            capture,
            _,
        ) = rollout_range(
            refiner=refiner,
            temporal=temporal,
            visual=visual,
            cache=cache,
            start_index=start,
            end_index=end,
            initial_xy=initial_xy,
            device=device,
            collect_rows=False,
        )

        gt = (
            cache.gt_xy[
                start:end
            ].numpy()
        )

        final_errors.extend(
            np.linalg.norm(
                final - gt,
                axis=1,
            ).tolist()
        )

        prediction_errors.extend(
            np.linalg.norm(
                predicted - gt,
                axis=1,
            ).tolist()
        )

        hardms_errors.extend(
            np.linalg.norm(
                refined_hardms - gt,
                axis=1,
            ).tolist()
        )

        capture_values.extend(
            capture.tolist()
        )

    final_errors = np.asarray(
        final_errors,
        dtype=np.float64,
    )

    return {
        "mle": float(
            np.mean(
                final_errors
            )
        ),
        "p90": float(
            np.quantile(
                final_errors,
                0.90,
            )
        ),
        "lsr15": float(
            100.0
            * np.mean(
                final_errors
                <= 15.0
            )
        ),
        "capture_pct": float(
            100.0
            * np.mean(
                capture_values
            )
        ),
        "prediction_mle": float(
            np.mean(
                prediction_errors
            )
        ),
        "hardms_mle": float(
            np.mean(
                hardms_errors
            )
        ),
        "episodes": int(
            len(starts)
        ),
    }


@torch.no_grad()
def evaluate_full_route_stress(
    refiner,
    temporal,
    visual,
    cache,
    start_xy,
    device,
):
    (
        predicted,
        raw_hardms,
        refined_hardms,
        measurement,
        final,
        capture,
        _,
    ) = rollout_range(
        refiner=refiner,
        temporal=temporal,
        visual=visual,
        cache=cache,
        start_index=0,
        end_index=len(cache),
        initial_xy=start_xy,
        device=device,
        collect_rows=False,
    )

    gt = (
        cache.gt_xy.numpy()
    )

    return {
        "mle": float(
            np.mean(
                np.linalg.norm(
                    final - gt,
                    axis=1,
                )
            )
        ),
        "capture_pct": float(
            100.0
            * np.mean(
                capture
            )
        ),
        "prediction_mle": float(
            np.mean(
                np.linalg.norm(
                    predicted - gt,
                    axis=1,
                )
            )
        ),
        "hardms_mle": float(
            np.mean(
                np.linalg.norm(
                    refined_hardms - gt,
                    axis=1,
                )
            )
        ),
    }


def save_checkpoint(
    path,
    refiner,
    temporal,
    stage,
    best,
    extra,
):
    payload = {
        "architecture": ARCHITECTURE_NAME,
        "stage": stage,

        "candidate_refiner": {
            key: value.detach()
            .cpu()
            for key, value
            in refiner.state_dict()
            .items()
        },

        "temporal": {
            key: value.detach()
            .cpu()
            for key, value
            in temporal.state_dict()
            .items()
        },

        "best": best,

        "train_routes": [
            "route_A"
        ],

        "eval_routes": [
            "route_B",
            "route_C",
        ],

        "current_gt_as_model_input": False,
        "previous_gt_as_model_input": False,
        "test_gt_as_model_input": False,

        "teacher_role": (
            "warmup local SAT crop only"
        ),

        "autoregressive_teacher_center": False,

        "polynomial": (
            "p_next = p_final + v_rnn + 0.5*a_rnn"
        ),

        "forward_search": (
            "full 6x6 centered at polynomial prediction; no hard mask"
        ),

        "final_filter": (
            "external Kalman [x,y,vx,vy]"
        ),
    }

    payload.update(
        extra
    )

    torch.save(
        payload,
        path,
    )


def train_temporal(
    refiner,
    temporal,
    visual,
    cache,
    start_xy,
    device,
    epochs,
):
    train_end = max(
        16,
        int(
            len(cache)
            * float(
                config.TEMPORAL_TRAIN_FRACTION
            )
        ),
    )

    val_start = min(
        len(cache) - 1,
        train_end,
    )

    optimizer = torch.optim.AdamW(
        list(
            refiner.parameters()
        )
        + list(
            temporal.parameters()
        ),
        lr=float(
            config.TEMPORAL_LR
        ),
        weight_decay=float(
            config.TEMPORAL_WEIGHT_DECAY
        ),
    )

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 118, flush=True)
    print(
        "v20 Stage-1: CRF-style candidate calibration + current frame/state -> NEXT GT displacement",
        flush=True,
    )
    print(
        "GT is used only to construct the current local crop and losses. It is not an RNN input.",
        flush=True,
    )
    print("=" * 118, flush=True)

    for warmup_epoch in range(
        int(
            config.WARMUP_EPOCHS
        )
    ):
        training = (
            train_warmup_epoch(
                refiner=refiner,
                temporal=temporal,
                optimizer=optimizer,
                visual=visual,
                cache=cache,
                train_end=train_end,
                device=device,
            )
        )

        print(
            "warmup=%02d/%d loss=%.4f cand=%.4f meas=%.4f next=%.4f "
            "cap=%.1f%% candDist=%.2fm rawMS=%.2fm refinedMS=%.2fm "
            "measErr=%.2fm nextErr=%.2fm predStep=%.2fm gate=%.3f"
            % (
                warmup_epoch + 1,
                int(
                    config.WARMUP_EPOCHS
                ),
                training["loss"],
                training["candidate"],
                training["measurement"],
                training["next_step"],
                training["capture_pct"],
                training["candidate_distance_m"],
                training["raw_hardms_error_m"],
                training["refined_hardms_error_m"],
                training["measurement_error_m"],
                training["next_error_m"],
                training["predicted_step_m"],
                training["gate"],
            ),
            flush=True,
        )

    save_checkpoint(
        config.WARMUP_CHECKPOINT,
        refiner,
        temporal,
        "warmup",
        best=None,
        extra={
            "warmup_epochs": int(
                config.WARMUP_EPOCHS
            ),
        },
    )

    print("=" * 118, flush=True)
    print(
        "v20 Stage-2: adaptive autoregressive rollout",
        flush=True,
    )
    print(
        "Search center = previous FINAL + previous RNN velocity + 0.5*previous RNN acceleration.",
        flush=True,
    )
    print(
        "Full 6x6 is retained. Forward preference is soft inside the CRF candidate score; there is no hard 3x6 mask.",
        flush=True,
    )
    print(
        "Best checkpoint / early stop uses Route-A validation EPISODES, while W0 full-route is only a stress test.",
        flush=True,
    )
    print("=" * 118, flush=True)

    horizon_values = [
        (
            int(train_end)
            if int(value) <= 0
            else min(
                int(value),
                int(train_end),
            )
        )
        for value in config.ROLLOUT_HORIZONS
    ]

    horizon_index = 0
    good_epochs = 0

    best_score = float("inf")
    best_refiner = None
    best_temporal = None
    best_epoch = -1
    best_horizon = -1

    patience = 0

    stage2_epochs = max(
        1,
        int(epochs)
        - int(
            config.WARMUP_EPOCHS
        ),
    )

    for stage2_epoch in range(
        1,
        stage2_epochs + 1,
    ):
        horizon = int(
            horizon_values[
                horizon_index
            ]
        )

        training = (
            train_rollout_epoch(
                refiner=refiner,
                temporal=temporal,
                optimizer=optimizer,
                visual=visual,
                cache=cache,
                train_end=train_end,
                horizon=horizon,
                device=device,
            )
        )

        episode = (
            evaluate_episodes(
                refiner=refiner,
                temporal=temporal,
                visual=visual,
                cache=cache,
                val_start=val_start,
                device=device,
            )
        )

        if (
            stage2_epoch
            % int(
                config.FULL_ROUTE_STRESS_EVERY
            )
            == 0
        ):
            full = (
                evaluate_full_route_stress(
                    refiner=refiner,
                    temporal=temporal,
                    visual=visual,
                    cache=cache,
                    start_xy=start_xy,
                    device=device,
                )
            )
        else:
            full = {
                "mle": float("nan"),
                "capture_pct": float("nan"),
                "prediction_mle": float("nan"),
                "hardms_mle": float("nan"),
            }

        score = float(
            episode["mle"]
        )

        meaningful_improvement = (
            score
            < (
                best_score
                - float(
                    config.EARLY_STOP_MIN_DELTA_M
                )
            )
        )

        if (
            best_refiner is None
            or meaningful_improvement
        ):
            best_score = score
            best_epoch = stage2_epoch
            best_horizon = horizon

            best_refiner = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in refiner.state_dict()
                .items()
            }

            best_temporal = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in temporal.state_dict()
                .items()
            }

            patience = 0
        else:
            if (
                stage2_epoch
                >= int(
                    config.EARLY_STOP_MIN_STAGE2_EPOCH
                )
            ):
                patience += 1

        good_now = (
            training["capture_pct"]
            >= float(
                config.HORIZON_TRAIN_CAPTURE_MIN_PCT
            )
            and training["prediction_error_m"]
            <= float(
                config.HORIZON_TRAIN_PRED_ERROR_MAX_M
            )
            and episode["capture_pct"]
            >= float(
                config.HORIZON_EPISODE_CAPTURE_MIN_PCT
            )
        )

        if good_now:
            good_epochs += 1
        else:
            good_epochs = 0

        advanced = False

        if (
            good_epochs
            >= int(
                config.HORIZON_GOOD_EPOCHS_TO_ADVANCE
            )
            and horizon_index
            < len(
                horizon_values
            ) - 1
        ):
            horizon_index += 1
            good_epochs = 0
            advanced = True

        save_checkpoint(
            config.TEMPORAL_CHECKPOINT,
            refiner,
            temporal,
            "autoregressive",
            best={
                "score_m": (
                    best_score
                ),
                "stage2_epoch": (
                    best_epoch
                ),
                "horizon": (
                    best_horizon
                ),
                "candidate_refiner": (
                    best_refiner
                ),
                "temporal": (
                    best_temporal
                ),
            },
            extra={
                "current_stage2_epoch": (
                    stage2_epoch
                ),
                "current_horizon": (
                    horizon
                ),
                "episode_validation": (
                    episode
                ),
                "full_route_stress": (
                    full
                ),
                "early_stop_patience": (
                    patience
                ),
            },
        )

        print(
            "stage2=%02d/%d H=%d%s loss=%.4f cand=%.4f next=%.4f "
            "trainCap=%.1f%% predErr=%.2fm rawMS=%.2fm refinedMS=%.2fm "
            "nextErr=%.2fm predStep=%.2fm gate=%.3f | "
            "epMLE=%.3fm epP90=%.3fm epCap=%.1f%% epLSR15=%.2f%% "
            "epPred=%.2fm epHardMS=%.2fm | "
            "fullMLE=%.2fm fullCap=%.1f%% | "
            "best=%02d/H%d@%.3fm patience=%d/%d"
            % (
                stage2_epoch,
                stage2_epochs,
                horizon,
                "->ADV" if advanced else "",
                training["loss"],
                training["candidate"],
                training["next_step"],
                training["capture_pct"],
                training["prediction_error_m"],
                training["raw_hardms_error_m"],
                training["refined_hardms_error_m"],
                training["next_error_m"],
                training["predicted_step_m"],
                training["gate"],
                episode["mle"],
                episode["p90"],
                episode["capture_pct"],
                episode["lsr15"],
                episode["prediction_mle"],
                episode["hardms_mle"],
                full["mle"],
                full["capture_pct"],
                best_epoch,
                best_horizon,
                best_score,
                patience,
                int(
                    config.EARLY_STOP_PATIENCE
                ),
            ),
            flush=True,
        )

        if (
            stage2_epoch
            >= int(
                config.EARLY_STOP_MIN_STAGE2_EPOCH
            )
            and patience
            >= int(
                config.EARLY_STOP_PATIENCE
            )
        ):
            print(
                "EARLY STOP: episode valMLE failed to improve by >= %.2fm for %d stage-2 epochs."
                % (
                    float(
                        config.EARLY_STOP_MIN_DELTA_M
                    ),
                    int(
                        config.EARLY_STOP_PATIENCE
                    ),
                ),
                flush=True,
            )
            break

    if (
        best_refiner is None
        or best_temporal is None
    ):
        raise RuntimeError(
            "No best temporal checkpoint"
        )

    refiner.load_state_dict(
        best_refiner
    )
    temporal.load_state_dict(
        best_temporal
    )

    save_checkpoint(
        config.TEMPORAL_CHECKPOINT,
        refiner,
        temporal,
        "best_autoregressive",
        best={
            "score_m": best_score,
            "stage2_epoch": best_epoch,
            "horizon": best_horizon,
            "candidate_refiner": best_refiner,
            "temporal": best_temporal,
        },
        extra={},
    )

    print(
        "best checkpoint: stage2=%d H=%d episodeMLE=%.3fm"
        % (
            best_epoch,
            best_horizon,
            best_score,
        ),
        flush=True,
    )


def load_best(
    refiner,
    temporal,
    device,
):
    if not (
        config.TEMPORAL_CHECKPOINT
        .exists()
    ):
        raise FileNotFoundError(
            config.TEMPORAL_CHECKPOINT
        )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )

    if (
        checkpoint.get(
            "architecture"
        )
        != ARCHITECTURE_NAME
    ):
        raise RuntimeError(
            "checkpoint architecture mismatch"
        )

    best = checkpoint.get(
        "best"
    )

    if (
        isinstance(
            best,
            dict,
        )
        and best.get(
            "candidate_refiner"
        )
        is not None
    ):
        refiner_state = best[
            "candidate_refiner"
        ]
        temporal_state = best[
            "temporal"
        ]
    else:
        refiner_state = checkpoint[
            "candidate_refiner"
        ]
        temporal_state = checkpoint[
            "temporal"
        ]

    refiner.load_state_dict(
        refiner_state
    )
    temporal.load_state_dict(
        temporal_state
    )

    refiner.to(device)
    temporal.to(device)

    refiner.eval()
    temporal.eval()

    print(
        "loaded v20 best checkpoint:",
        best.get(
            "stage2_epoch"
        )
        if isinstance(
            best,
            dict,
        )
        else "?",
        "episodeMLE=",
        best.get(
            "score_m"
        )
        if isinstance(
            best,
            dict,
        )
        else "?",
        flush=True,
    )


def write_rows(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No rows to write"
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_route(
    refiner,
    temporal,
    visual,
    cache,
    initial_xy,
    device,
    csv_path,
):
    (
        predicted,
        raw_hardms,
        refined_hardms,
        measurement,
        final,
        capture,
        rows,
    ) = rollout_range(
        refiner=refiner,
        temporal=temporal,
        visual=visual,
        cache=cache,
        start_index=0,
        end_index=len(cache),
        initial_xy=initial_xy,
        device=device,
        collect_rows=True,
    )

    write_rows(
        csv_path,
        rows,
    )

    gt = (
        cache.gt_xy.numpy()
    )

    return {
        "architecture": (
            ARCHITECTURE_NAME
        ),

        "input_constraint": (
            "current UAV frame + current SAT candidate images/features + previous recurrent state"
        ),

        "polynomial": (
            "p_next = p_final + v_rnn + 0.5*a_rnn"
        ),

        "forward_search": (
            "full 6x6 centered at polynomial prediction; soft forward prior; no hard mask"
        ),

        "candidate_temporal_model": (
            "CRF-inspired emission + inertial transition candidate refinement"
        ),

        "final_filter": (
            "external Kalman"
        ),

        "capture_pct": float(
            100.0
            * np.mean(
                capture
            )
        ),

        "PredictedCenter": (
            metrics(
                predicted,
                gt,
            )
        ),

        "RawHardMS": (
            metrics(
                raw_hardms,
                gt,
            )
        ),

        "RefinedHardMS": (
            metrics(
                refined_hardms,
                gt,
            )
        ),

        "RNNMeasurement": (
            metrics(
                measurement,
                gt,
            )
        ),

        "FinalKalman": (
            metrics(
                final,
                gt,
            )
        ),
    }


def route_catalog():
    return {
        name: Path(root)
        for name, root
        in zip(
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
        epochs=int(
            visual_epochs
        ),
        jitter_m=float(
            config.LOCAL_PRIOR_JITTER_M
        ),
        resume=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "eval",
            "train_eval",
        ],
        default="train_eval",
    )

    parser.add_argument(
        "--visual-epochs",
        type=int,
        default=int(
            config.VISUAL_EPOCHS
        ),
    )

    parser.add_argument(
        "--temporal-epochs",
        type=int,
        default=int(
            config.TEMPORAL_EPOCHS
        ),
    )

    parser.add_argument(
        "--reuse-visual",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    set_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 118, flush=True)
    print(
        "CRF-Inertial RNN + External Kalman v20",
        flush=True,
    )
    print(
        "current frame -> 36 candidate CRF refinement -> HardMS -> RNN measurement/state",
        flush=True,
    )
    print(
        "previous RNN (v,a) -> second-order polynomial -> NEXT full 6x6 search center",
        flush=True,
    )
    print(
        "RNN output measurement -> external Kalman -> FINAL localization",
        flush=True,
    )
    print("=" * 118, flush=True)

    if args.mode in (
        "train",
        "train_eval",
    ):
        ensure_visual_checkpoint(
            device=device,
            visual_epochs=args.visual_epochs,
            reuse_visual=bool(
                args.reuse_visual
            ),
        )

    elif not (
        config.VISUAL_CHECKPOINT
        .exists()
    ):
        raise FileNotFoundError(
            "eval requires visual checkpoint: %s"
            % config.VISUAL_CHECKPOINT
        )

    visual = (
        FrozenVisualLocalizer(
            device
        )
    )

    routes = (
        route_catalog()
    )

    refiner, temporal = (
        make_stage_models(
            device
        )
    )

    if args.mode in (
        "train",
        "train_eval",
    ):
        cache = (
            build_route_cache(
                "route_A",
                routes["route_A"],
                visual,
                device,
            )
        )

        waypoint_xy = (
            load_waypoint_xy(
                "route_A",
                visual.origin_lat,
                visual.origin_lon,
            )
        )

        start_xy = (
            waypoint_xy[0]
        )

        train_temporal(
            refiner=refiner,
            temporal=temporal,
            visual=visual,
            cache=cache,
            start_xy=start_xy,
            device=device,
            epochs=int(
                args.temporal_epochs
            ),
        )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        load_best(
            refiner,
            temporal,
            device,
        )

        results = {}

        for route_name in [
            "route_B",
            "route_C",
        ]:
            cache = (
                build_route_cache(
                    route_name,
                    routes[
                        route_name
                    ],
                    visual,
                    device,
                )
            )

            waypoint_xy = (
                load_waypoint_xy(
                    route_name,
                    visual.origin_lat,
                    visual.origin_lon,
                )
            )

            initial_xy = (
                waypoint_xy[0]
            )

            csv_path = (
                config.OUTPUT_DIR
                / (
                    route_name
                    + "_crf_inertial_rnn_kalman_frames.csv"
                )
            )

            summary = evaluate_route(
                refiner=refiner,
                temporal=temporal,
                visual=visual,
                cache=cache,
                initial_xy=initial_xy,
                device=device,
                csv_path=csv_path,
            )

            results[
                route_name
            ] = summary

            final_metric = (
                summary[
                    "FinalKalman"
                ]
            )

            print(
                "%s Final: MLE=%.3fm P90=%.3fm LSR15=%.2f%% Jump=%.2f%% Capture=%.1f%%"
                % (
                    route_name,
                    final_metric["MLE_m"],
                    final_metric["P90_m"],
                    final_metric["LSR@15_pct"],
                    final_metric["JumpRate_pct"],
                    summary["capture_pct"],
                ),
                flush=True,
            )

        summary_path = (
            config.OUTPUT_DIR
            / "robust_tracker_summary.json"
        )

        summary_path.write_text(
            json.dumps(
                results,
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
