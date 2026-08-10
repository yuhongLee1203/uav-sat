import math
from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# =============================================================================
# Visual retrieval network -- parameter names compatible with visual_localizer.py
# =============================================================================

class FourierCoordEncoder(nn.Module):
    def __init__(self, num_bands=32, max_freq=16.0):
        super().__init__()
        freqs = torch.logspace(
            0.0,
            math.log10(max_freq),
            steps=num_bands,
        )
        self.register_buffer("freqs", freqs)
        in_dim = 2 + 2 * 2 * num_bands
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, config.EMBED_DIM),
        )

    def forward(self, xy):
        xy = xy.float() / 1000.0
        angles = (
            xy[:, :, None]
            * self.freqs[None, None, :]
            * math.pi
        )
        encoded = torch.cat(
            [
                xy,
                angles.sin().flatten(1),
                angles.cos().flatten(1),
            ],
            dim=1,
        )
        return self.mlp(encoded)


class AllMapGeoCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip, _ = open_clip.create_model_from_pretrained(
            config.BACKBONE_NAME
        )
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.use_coord_encoder = bool(
            getattr(config, "USE_COORD_ENCODER", False)
        )

        self.uav_head = nn.Sequential(
            nn.Linear(config.CLIP_DIM, config.CLIP_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.CLIP_DIM, config.EMBED_DIM),
        )

        if self.use_coord_encoder:
            self.coord_encoder = FourierCoordEncoder()
            sat_head_in_dim = config.CLIP_DIM + config.EMBED_DIM
        else:
            self.coord_encoder = None
            sat_head_in_dim = config.CLIP_DIM

        self.sat_head = nn.Sequential(
            nn.Linear(sat_head_in_dim, config.CLIP_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.CLIP_DIM, config.EMBED_DIM),
        )

        self.use_qah_relation = bool(
            getattr(config, "USE_QAH_MS_RELATION", False)
        )
        if self.use_qah_relation:
            relation_dim = int(
                getattr(config, "QAH_RELATION_DIM", 32)
            )
            self.qah_relation_head = nn.Sequential(
                nn.Linear(config.EMBED_DIM * 2 + 1, 256),
                nn.GELU(),
                nn.Linear(256, relation_dim),
            )
        else:
            self.qah_relation_head = None

        self.use_basin_rank_ms = bool(
            getattr(config, "USE_BASIN_RANK_MS", False)
        )
        if self.use_basin_rank_ms:
            self.basin_ranker = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
        else:
            self.basin_ranker = None

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(image.float())

    @torch.no_grad()
    def encode_clip_spatial(self, image, output_size=None):
        output_size = int(
            output_size or config.MOTION_SPATIAL_SIZE
        )
        feature = self.encode_clip_image(
            image
        ).unsqueeze(-1).unsqueeze(-1)
        return F.adaptive_avg_pool2d(
            feature.float(),
            (output_size, output_size),
        )

    def encode_uav_from_clip(self, clip_feat, yaw=None):
        embedding = self.uav_head(clip_feat.float())
        return F.normalize(embedding, dim=1)

    def encode_uav(self, uav, yaw=None):
        return self.encode_uav_from_clip(
            self.encode_clip_image(uav)
        )

    def encode_sat_from_clip(self, sat_clip_feat, xy):
        if self.use_coord_encoder:
            coord_feat = self.coord_encoder(xy.float())
            sat_input = torch.cat(
                [sat_clip_feat.float(), coord_feat],
                dim=1,
            )
        else:
            sat_input = sat_clip_feat.float()

        embedding = self.sat_head(sat_input)
        return F.normalize(embedding, dim=1)

    def encode_relation(self, z_uav, z_sat, logits):
        if self.qah_relation_head is None:
            raise RuntimeError("QAH relation head is disabled")

        pair_product = z_uav.unsqueeze(1) * z_sat
        pair_absolute = torch.abs(
            z_uav.unsqueeze(1) - z_sat
        )
        pair_logit = logits.unsqueeze(-1)

        pair = torch.cat(
            [pair_product, pair_absolute, pair_logit],
            dim=-1,
        )

        relation = self.qah_relation_head(
            pair.reshape(-1, pair.shape[-1])
        )

        return relation.reshape(
            pair.shape[0],
            pair.shape[1],
            -1,
        )


# =============================================================================
# Route-coordinate recurrent measurement model
# =============================================================================


@dataclass
class RouteMeasurementOutput:
    measurement_sd: torch.Tensor
    measurement_variance: torch.Tensor
    visual_expectation_sd: torch.Tensor
    hidden: torch.Tensor


class RouteCoordinateGRU(nn.Module):
    """One current frame per recurrent step.

    The GRU outputs a route-coordinate visual measurement [s,d] and a learned
    diagonal measurement variance. Final filtering is done by Kalman equations.
    """

    def __init__(self):
        super().__init__()
        feature_dim = int(config.RNN_FEATURE_DIM)
        hidden_dim = int(config.RNN_HIDDEN_DIM)

        self.uav_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.sat_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        # entropy, margin, max prob, HardMS support,
        # visual/prediction innovation s,d, HardMS/prediction innovation s,d,
        # normalized progress, remaining, v, vd, dt_seconds, leg-change.
        self.numeric_projection = nn.Sequential(
            nn.Linear(14, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.gru = nn.GRUCell(feature_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(float(config.RNN_DROPOUT))
        self.measurement_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Start from the retrieval expectation, not a random coordinate jump.
        nn.init.zeros_(self.measurement_residual_head[-1].weight)
        nn.init.zeros_(self.measurement_residual_head[-1].bias)
        initial_var = 4.0
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(
            self.variance_head[-1].bias,
            math.log(math.exp(initial_var) - 1.0),
        )

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size),
            int(config.RNN_HIDDEN_DIM),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def normalized_entropy(probability):
        count = max(int(probability.shape[-1]), 2)
        return -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(count))

    def forward_step(
        self,
        z_uav,
        z_sat,
        raw_prob,
        candidate_sd,
        hardms_sd,
        hardms_support,
        predicted_state,
        previous_progress,
        leg_length,
        nominal_speed_mps,
        forward_speed_limit_mps,
        cross_speed_limit_mps,
        dt_seconds,
        leg_change,
        hidden,
    ):
        if hidden is None:
            hidden = self.initial_hidden(
                z_uav.shape[0], z_uav.device, z_uav.dtype
            )

        visual_expectation = (
            raw_prob.unsqueeze(-1) * candidate_sd
        ).sum(dim=1)
        sat_context = (
            raw_prob.unsqueeze(-1) * z_sat
        ).sum(dim=1)

        entropy = self.normalized_entropy(raw_prob)
        top2 = raw_prob.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        max_probability = top2[:, 0]

        predicted_s = predicted_state[:, 0]
        predicted_v = predicted_state[:, 1]
        predicted_d = predicted_state[:, 2]
        predicted_vd = predicted_state[:, 3]

        progress_norm = (
            predicted_s / leg_length.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        remaining_norm = (
            (leg_length - predicted_s) / leg_length.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        v_norm = (
            predicted_v / nominal_speed_mps.clamp_min(1e-6)
        ).clamp(0.0, 2.0)
        vd_norm = (
            predicted_vd / cross_speed_limit_mps.clamp_min(1e-6)
        ).clamp(-2.0, 2.0)

        numeric = torch.stack(
            [
                entropy,
                margin,
                max_probability,
                hardms_support,
                visual_expectation[:, 0] - predicted_s,
                visual_expectation[:, 1] - predicted_d,
                hardms_sd[:, 0] - predicted_s,
                hardms_sd[:, 1] - predicted_d,
                progress_norm,
                remaining_norm,
                v_norm,
                vd_norm,
                dt_seconds,
                leg_change.float(),
            ],
            dim=1,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.sat_projection(sat_context),
                self.numeric_projection(numeric),
            ],
            dim=1,
        )
        hidden = self.gru(recurrent_input, hidden)
        head_hidden = self.dropout(hidden)

        residual = self.measurement_residual_head(head_hidden)
        # Residual magnitude is expressed in metres using physical m/s * dt.
        residual_s = (
            torch.tanh(residual[:, 0])
            * forward_speed_limit_mps
            * dt_seconds
        )
        residual_d = (
            torch.tanh(residual[:, 1])
            * cross_speed_limit_mps
            * dt_seconds
        )

        raw_s = visual_expectation[:, 0] + residual_s
        raw_d = visual_expectation[:, 1] + residual_d

        # Retrieval can look several seconds ahead, but never behind the already
        # accepted mission progress.
        measurement_upper = torch.minimum(
            leg_length,
            previous_progress
            + forward_speed_limit_mps * float(config.SEARCH_LOOKAHEAD_SECONDS),
        )
        measurement_s = torch.maximum(
            previous_progress,
            torch.minimum(raw_s, measurement_upper),
        )
        measurement_d = torch.clamp(
            raw_d,
            min=-float(config.ROUTE_CORRIDOR_HALF_WIDTH_M),
            max=float(config.ROUTE_CORRIDOR_HALF_WIDTH_M),
        )

        variance = (
            F.softplus(self.variance_head(head_hidden))
            + float(config.KALMAN_MIN_VARIANCE)
        ).clamp(max=float(config.KALMAN_MAX_MEASUREMENT_VAR))

        return RouteMeasurementOutput(
            measurement_sd=torch.stack([measurement_s, measurement_d], dim=1),
            measurement_variance=variance,
            visual_expectation_sd=visual_expectation,
            hidden=hidden,
        )


def initial_covariance_torch(batch, device, dtype):
    diagonal = torch.tensor(
        [
            config.KALMAN_INIT_PROGRESS_VAR,
            config.KALMAN_INIT_FORWARD_SPEED_VAR,
            config.KALMAN_INIT_CROSS_TRACK_VAR,
            config.KALMAN_INIT_CROSS_SPEED_VAR,
        ],
        device=device,
        dtype=dtype,
    )
    return torch.diag(diagonal).unsqueeze(0).repeat(int(batch), 1, 1)


def transition_matrix_torch(dt_seconds, device, dtype):
    dt = torch.as_tensor(dt_seconds, device=device, dtype=dtype).reshape(-1)
    matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(
        dt.shape[0], 1, 1
    )
    matrix[:, 0, 1] = dt
    matrix[:, 2, 3] = dt
    return matrix


def process_covariance_torch(dt_seconds, device, dtype):
    dt = torch.as_tensor(dt_seconds, device=device, dtype=dtype).reshape(-1)
    dt = dt.clamp_min(float(config.MIN_DT_SECONDS))
    base = torch.tensor(
        [
            config.KALMAN_Q_PROGRESS,
            config.KALMAN_Q_FORWARD_SPEED,
            config.KALMAN_Q_CROSS_TRACK,
            config.KALMAN_Q_CROSS_SPEED,
        ],
        device=device,
        dtype=dtype,
    )
    return torch.diag_embed(base.unsqueeze(0) * dt.unsqueeze(1))


def kalman_predict_torch(state, covariance, dt_seconds):
    transition = transition_matrix_torch(
        dt_seconds, state.device, state.dtype
    )
    predicted_state = torch.bmm(
        transition, state.unsqueeze(-1)
    ).squeeze(-1)
    predicted_covariance = (
        torch.bmm(
            torch.bmm(transition, covariance),
            transition.transpose(1, 2),
        )
        + process_covariance_torch(dt_seconds, state.device, state.dtype)
    )
    return predicted_state, predicted_covariance


def kalman_update_torch(
    predicted_state,
    predicted_covariance,
    measurement_sd,
    measurement_variance,
):
    batch = predicted_state.shape[0]
    device = predicted_state.device
    dtype = predicted_state.dtype

    observation = torch.zeros(batch, 2, 4, device=device, dtype=dtype)
    observation[:, 0, 0] = 1.0
    observation[:, 1, 2] = 1.0
    measurement_covariance = torch.diag_embed(measurement_variance)
    expected = torch.stack(
        [predicted_state[:, 0], predicted_state[:, 2]], dim=1
    )
    innovation = measurement_sd - expected

    hp = torch.bmm(observation, predicted_covariance)
    innovation_covariance = (
        torch.bmm(hp, observation.transpose(1, 2))
        + measurement_covariance
    )
    ph_t = torch.bmm(
        predicted_covariance, observation.transpose(1, 2)
    )
    gain = torch.linalg.solve(
        innovation_covariance, ph_t.transpose(1, 2)
    ).transpose(1, 2)

    updated_state = predicted_state + torch.bmm(
        gain, innovation.unsqueeze(-1)
    ).squeeze(-1)

    identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(
        batch, 1, 1
    )
    left = identity - torch.bmm(gain, observation)
    updated_covariance = (
        torch.bmm(
            torch.bmm(left, predicted_covariance),
            left.transpose(1, 2),
        )
        + torch.bmm(
            torch.bmm(gain, measurement_covariance),
            gain.transpose(1, 2),
        )
    )
    return updated_state, updated_covariance, gain


def constrain_route_state_torch(
    state,
    previous_progress,
    previous_cross_track,
    leg_length,
    forward_speed_limit_mps,
    cross_speed_limit_mps,
    dt_seconds,
):
    """Project state onto the physically valid mission state set.

    The constraint is structural: within one active leg, progress cannot go
    backward or teleport farther than the Route-A learned m/s envelope allows.
    """
    dt = torch.as_tensor(
        dt_seconds, device=state.device, dtype=state.dtype
    ).reshape(-1)

    max_progress = torch.minimum(
        leg_length,
        previous_progress + forward_speed_limit_mps * dt,
    )
    progress = torch.maximum(
        previous_progress,
        torch.minimum(state[:, 0], max_progress),
    )
    velocity = torch.maximum(
        torch.zeros_like(state[:, 1]),
        torch.minimum(state[:, 1], forward_speed_limit_mps),
    )

    max_cross_delta = cross_speed_limit_mps * dt
    cross_delta = torch.maximum(
        -max_cross_delta,
        torch.minimum(state[:, 2] - previous_cross_track, max_cross_delta),
    )
    cross_track = torch.clamp(
        previous_cross_track + cross_delta,
        min=-float(config.ROUTE_CORRIDOR_HALF_WIDTH_M),
        max=float(config.ROUTE_CORRIDOR_HALF_WIDTH_M),
    )
    cross_velocity = torch.maximum(
        -cross_speed_limit_mps,
        torch.minimum(state[:, 3], cross_speed_limit_mps),
    )

    return torch.stack(
        [progress, velocity, cross_track, cross_velocity], dim=1
    )
