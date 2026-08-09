import math
from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================================
# Existing single-frame retrieval model.
# Parameter names remain compatible with visual_localizer.py.
# ============================================================================

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


# ============================================================================
# Streaming recurrent measurement model.
#
# IMPORTANT:
#   This model consumes ONE frame's retrieval evidence per call.
#   Temporal history lives in hidden_t, not in a stack of UAV images.
# ============================================================================

@dataclass
class MeasurementOutput:
    measurement_xy: torch.Tensor
    measurement_variance: torch.Tensor
    visual_expectation_xy: torch.Tensor
    hidden: torch.Tensor


class RouteGRUMeasurementModel(nn.Module):
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

        # Numeric feature definition (17 dimensions):
        #  0 entropy
        #  1 top1-top2 margin
        #  2 max probability
        #  3 hardms support
        #  4-5 visual expectation - predicted xy
        #  6-7 hardms - predicted xy
        #  8 hardms-to-expectation distance
        #  9-10 active leg unit direction
        # 11 normalized predicted along-leg progress
        # 12 normalized remaining leg distance
        # 13 normalized predicted cross-track displacement
        # 14 predicted along-leg velocity
        # 15 predicted cross-track velocity
        # 16 leg-change flag
        self.numeric_projection = nn.Sequential(
            nn.Linear(17, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.gru = nn.GRUCell(
            feature_dim * 3,
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            float(config.RNN_DROPOUT)
        )

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

        # Start from the probability-weighted retrieval position.
        nn.init.zeros_(
            self.measurement_residual_head[-1].weight
        )
        nn.init.zeros_(
            self.measurement_residual_head[-1].bias
        )

        # Initial diagonal R around 9 m^2 per axis.
        initial_var = 9.0
        inverse_softplus = math.log(
            math.exp(initial_var) - 1.0
        )
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(
            self.variance_head[-1].bias,
            inverse_softplus,
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
        count = max(
            int(probability.shape[-1]),
            2,
        )
        return -(
            probability
            * probability.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(count))

    def forward_step(
        self,
        z_uav,
        z_sat,
        raw_prob,
        centers,
        hardms_xy,
        hardms_support,
        predicted_state,
        leg_start,
        leg_end,
        leg_change,
        hidden,
    ):
        """
        All tensors are current-frame only.

        z_uav:          [B,512]
        z_sat:          [B,36,512]
        raw_prob:       [B,36]
        centers:        [B,36,2]
        predicted_state:[B,6] = [x,y,vx,vy,ax,ay]
        """
        if hidden is None:
            hidden = self.initial_hidden(
                z_uav.shape[0],
                z_uav.device,
                z_uav.dtype,
            )

        visual_expectation = (
            raw_prob.unsqueeze(-1)
            * centers
        ).sum(dim=1)

        sat_context = (
            raw_prob.unsqueeze(-1)
            * z_sat
        ).sum(dim=1)

        entropy = self.normalized_entropy(
            raw_prob
        )

        top2 = raw_prob.topk(
            k=2,
            dim=1,
        ).values

        margin = top2[:, 0] - top2[:, 1]
        max_probability = top2[:, 0]

        predicted_xy = predicted_state[:, 0:2]
        predicted_velocity = predicted_state[:, 2:4]

        leg_vector = leg_end - leg_start
        leg_length = torch.linalg.norm(
            leg_vector,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        leg_unit = leg_vector / leg_length
        leg_normal = torch.stack(
            [-leg_unit[:, 1], leg_unit[:, 0]],
            dim=1,
        )

        relative_predicted = predicted_xy - leg_start

        along = (
            relative_predicted
            * leg_unit
        ).sum(dim=1)

        cross = (
            relative_predicted
            * leg_normal
        ).sum(dim=1)

        along_velocity = (
            predicted_velocity
            * leg_unit
        ).sum(dim=1)

        cross_velocity = (
            predicted_velocity
            * leg_normal
        ).sum(dim=1)

        leg_length_1d = leg_length[:, 0]

        normalized_progress = (
            along / leg_length_1d
        ).clamp(-0.25, 1.25)

        normalized_remaining = (
            (leg_length_1d - along)
            / leg_length_1d
        ).clamp(-0.25, 1.25)

        normalized_cross = (
            cross
            / float(config.ROUTE_CORRIDOR_HALF_WIDTH_M)
        ).clamp(-4.0, 4.0)

        expectation_innovation = (
            visual_expectation - predicted_xy
        )

        hardms_innovation = (
            hardms_xy - predicted_xy
        )

        hardms_to_expectation = torch.linalg.norm(
            hardms_xy - visual_expectation,
            dim=1,
        )

        numeric = torch.cat(
            [
                entropy.unsqueeze(1),
                margin.unsqueeze(1),
                max_probability.unsqueeze(1),
                hardms_support.unsqueeze(1),
                expectation_innovation,
                hardms_innovation,
                hardms_to_expectation.unsqueeze(1),
                leg_unit,
                normalized_progress.unsqueeze(1),
                normalized_remaining.unsqueeze(1),
                normalized_cross.unsqueeze(1),
                along_velocity.unsqueeze(1),
                cross_velocity.unsqueeze(1),
                leg_change.float().reshape(-1, 1),
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

        hidden = self.gru(
            recurrent_input,
            hidden,
        )

        head_hidden = self.dropout(hidden)

        residual = self.measurement_residual_head(
            head_hidden
        )

        measurement_xy = (
            visual_expectation + residual
        )

        measurement_variance = (
            F.softplus(
                self.variance_head(head_hidden)
            )
            + float(config.KALMAN_MIN_VARIANCE)
        ).clamp(
            max=float(
                config.KALMAN_MAX_MEASUREMENT_VAR
            )
        )

        return MeasurementOutput(
            measurement_xy=measurement_xy,
            measurement_variance=measurement_variance,
            visual_expectation_xy=visual_expectation,
            hidden=hidden,
        )


# ============================================================================
# Differentiable Kalman equations used ONLY during Route-A GRU training.
# Deployment/evaluation uses FilterPy's public KalmanFilter implementation.
# The state model is deliberately identical.
# ============================================================================

def transition_matrix_torch(dt, device, dtype):
    dt = torch.as_tensor(
        dt,
        device=device,
        dtype=dtype,
    ).reshape(-1)

    batch = dt.shape[0]

    matrix = torch.eye(
        6,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).repeat(
        batch,
        1,
        1,
    )

    half_dt2 = 0.5 * dt.square()

    matrix[:, 0, 2] = dt
    matrix[:, 1, 3] = dt

    matrix[:, 0, 4] = half_dt2
    matrix[:, 1, 5] = half_dt2

    matrix[:, 2, 4] = dt
    matrix[:, 3, 5] = dt

    return matrix


def process_covariance_torch(dt, device, dtype):
    dt = torch.as_tensor(
        dt,
        device=device,
        dtype=dtype,
    ).reshape(-1).clamp_min(1.0)

    base = torch.tensor(
        [
            config.KALMAN_Q_POSITION,
            config.KALMAN_Q_POSITION,
            config.KALMAN_Q_VELOCITY,
            config.KALMAN_Q_VELOCITY,
            config.KALMAN_Q_ACCELERATION,
            config.KALMAN_Q_ACCELERATION,
        ],
        device=device,
        dtype=dtype,
    )

    diagonal = (
        base.unsqueeze(0)
        * dt.unsqueeze(1)
    )

    return torch.diag_embed(diagonal)


def initial_covariance_torch(batch, device, dtype):
    diagonal = torch.tensor(
        [
            config.KALMAN_INIT_POSITION_VAR,
            config.KALMAN_INIT_POSITION_VAR,
            config.KALMAN_INIT_VELOCITY_VAR,
            config.KALMAN_INIT_VELOCITY_VAR,
            config.KALMAN_INIT_ACCELERATION_VAR,
            config.KALMAN_INIT_ACCELERATION_VAR,
        ],
        device=device,
        dtype=dtype,
    )

    return torch.diag(
        diagonal
    ).unsqueeze(0).repeat(
        int(batch),
        1,
        1,
    )


def kalman_predict_torch(state, covariance, dt):
    transition = transition_matrix_torch(
        dt,
        state.device,
        state.dtype,
    )

    predicted_state = torch.bmm(
        transition,
        state.unsqueeze(-1),
    ).squeeze(-1)

    predicted_covariance = (
        torch.bmm(
            torch.bmm(
                transition,
                covariance,
            ),
            transition.transpose(1, 2),
        )
        + process_covariance_torch(
            dt,
            state.device,
            state.dtype,
        )
    )

    return predicted_state, predicted_covariance


def kalman_update_torch(
    predicted_state,
    predicted_covariance,
    measurement_xy,
    measurement_variance,
):
    batch = predicted_state.shape[0]
    device = predicted_state.device
    dtype = predicted_state.dtype

    observation = torch.zeros(
        batch,
        2,
        6,
        device=device,
        dtype=dtype,
    )
    observation[:, 0, 0] = 1.0
    observation[:, 1, 1] = 1.0

    measurement_covariance = torch.diag_embed(
        measurement_variance.to(dtype)
    )

    innovation = (
        measurement_xy
        - predicted_state[:, 0:2]
    )

    hp = torch.bmm(
        observation,
        predicted_covariance,
    )

    innovation_covariance = (
        torch.bmm(
            hp,
            observation.transpose(1, 2),
        )
        + measurement_covariance
    )

    ph_t = torch.bmm(
        predicted_covariance,
        observation.transpose(1, 2),
    )

    gain = torch.linalg.solve(
        innovation_covariance,
        ph_t.transpose(1, 2),
    ).transpose(1, 2)

    updated_state = (
        predicted_state
        + torch.bmm(
            gain,
            innovation.unsqueeze(-1),
        ).squeeze(-1)
    )

    identity = torch.eye(
        6,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).repeat(
        batch,
        1,
        1,
    )

    kh = torch.bmm(
        gain,
        observation,
    )
    left = identity - kh

    # Joseph form.
    updated_covariance = (
        torch.bmm(
            torch.bmm(
                left,
                predicted_covariance,
            ),
            left.transpose(1, 2),
        )
        + torch.bmm(
            torch.bmm(
                gain,
                measurement_covariance,
            ),
            gain.transpose(1, 2),
        )
    )

    return (
        updated_state,
        updated_covariance,
        gain,
    )
