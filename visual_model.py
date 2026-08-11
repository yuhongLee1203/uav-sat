import math
from dataclasses import dataclass
from typing import Optional, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================================
# Visual retrieval model
# ============================================================================
# This section intentionally keeps the same parameter names used by the existing
# Route-A-only visual checkpoint, so the trained retrieval heads remain usable.
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
            sat_head_in_dim = (
                config.CLIP_DIM
                + config.EMBED_DIM
            )
        else:
            self.coord_encoder = None
            sat_head_in_dim = config.CLIP_DIM

        self.sat_head = nn.Sequential(
            nn.Linear(
                sat_head_in_dim,
                config.CLIP_DIM,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                config.CLIP_DIM,
                config.EMBED_DIM,
            ),
        )

        self.use_qah_relation = bool(
            getattr(
                config,
                "USE_QAH_MS_RELATION",
                False,
            )
        )

        if self.use_qah_relation:
            relation_dim = int(
                getattr(
                    config,
                    "QAH_RELATION_DIM",
                    32,
                )
            )
            self.qah_relation_head = nn.Sequential(
                nn.Linear(
                    config.EMBED_DIM * 2 + 1,
                    256,
                ),
                nn.GELU(),
                nn.Linear(
                    256,
                    relation_dim,
                ),
            )
        else:
            self.qah_relation_head = None

        self.use_basin_rank_ms = bool(
            getattr(
                config,
                "USE_BASIN_RANK_MS",
                False,
            )
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
            torch.ones([])
            * math.log(1.0 / 0.07)
        )

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(
            image.float()
        )

    @torch.no_grad()
    def encode_clip_spatial(
        self,
        image,
        output_size=None,
    ):
        output_size = int(
            output_size
            or config.MOTION_SPATIAL_SIZE
        )

        feature = self.encode_clip_image(
            image
        ).unsqueeze(-1).unsqueeze(-1)

        return F.adaptive_avg_pool2d(
            feature.float(),
            (
                output_size,
                output_size,
            ),
        )

    def encode_uav_from_clip(
        self,
        clip_feat,
        yaw=None,
    ):
        embedding = self.uav_head(
            clip_feat.float()
        )
        return F.normalize(
            embedding,
            dim=1,
        )

    def encode_uav(
        self,
        uav,
        yaw=None,
    ):
        return self.encode_uav_from_clip(
            self.encode_clip_image(uav)
        )

    def encode_sat_from_clip(
        self,
        sat_clip_feat,
        xy,
    ):
        if self.use_coord_encoder:
            coord_feat = self.coord_encoder(
                xy.float()
            )
            sat_input = torch.cat(
                [
                    sat_clip_feat.float(),
                    coord_feat,
                ],
                dim=1,
            )
        else:
            sat_input = sat_clip_feat.float()

        embedding = self.sat_head(
            sat_input
        )

        return F.normalize(
            embedding,
            dim=1,
        )

    def encode_relation(
        self,
        z_uav,
        z_sat,
        logits,
    ):
        if self.qah_relation_head is None:
            raise RuntimeError(
                "QAH relation head is disabled"
            )

        pair_product = (
            z_uav.unsqueeze(1)
            * z_sat
        )

        pair_absolute = torch.abs(
            z_uav.unsqueeze(1)
            - z_sat
        )

        pair_logit = logits.unsqueeze(-1)

        pair = torch.cat(
            [
                pair_product,
                pair_absolute,
                pair_logit,
            ],
            dim=-1,
        )

        relation = self.qah_relation_head(
            pair.reshape(
                -1,
                pair.shape[-1],
            )
        )

        return relation.reshape(
            pair.shape[0],
            pair.shape[1],
            -1,
        )



from dataclasses import dataclass


@dataclass
class CandidateRefinementOutput:
    refined_logits: torch.Tensor
    refined_probability: torch.Tensor
    sat_context: torch.Tensor
    raw_standardized: torch.Tensor
    motion_prior: torch.Tensor
    forward_prior: torch.Tensor


@dataclass
class CRFInertialRNNOutput:
    measurement_xy: torch.Tensor
    correction_xy: torch.Tensor
    correction_gate: torch.Tensor
    velocity_xy: torch.Tensor
    acceleration_xy: torch.Tensor
    next_step_xy: torch.Tensor
    measurement_variance: torch.Tensor
    state: torch.Tensor
    hidden: torch.Tensor


class CRFCandidateRefiner(nn.Module):
    """
    Single-frame recurrent CRF-style candidate calibration.

    Old RTL-CRF useful pieces retained:
      - standardized raw retrieval logit
      - raw probability
      - candidate relative xy / radius
      - learned emission residual

    Teacher-requested temporal transition:
      previous RNN state -> (v,a) -> polynomial next position.
      Each candidate receives a soft transition score according to how well it
      agrees with that inertial prediction. A weak forward-direction term is
      soft only; candidates behind the motion are never removed.
    """

    def __init__(self):
        super().__init__()

        token_dim = int(config.TOKEN_DIM)
        dropout = float(config.TEMPORAL_DROPOUT)

        self.uav_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.sat_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.state_projection = nn.Sequential(
            nn.Linear(int(config.RNN_STATE_DIM), token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.CANDIDATE_NUMERIC_DIM), token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.emission_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, 1),
        )

        nn.init.zeros_(self.emission_head[-1].weight)
        nn.init.zeros_(self.emission_head[-1].bias)

        def inverse_softplus(value):
            return math.log(math.exp(float(value)) - 1.0)

        self.raw_weight_raw = nn.Parameter(
            torch.tensor(inverse_softplus(config.INITIAL_RAW_LOGIT_WEIGHT))
        )
        self.motion_weight_raw = nn.Parameter(
            torch.tensor(inverse_softplus(config.INITIAL_MOTION_PRIOR_WEIGHT))
        )
        self.forward_weight_raw = nn.Parameter(
            torch.tensor(inverse_softplus(config.INITIAL_FORWARD_PRIOR_WEIGHT))
        )

    @staticmethod
    def _safe_cosine(a, b):
        numerator = (a * b).sum(dim=-1)
        denominator = (
            torch.linalg.norm(a, dim=-1)
            * torch.linalg.norm(b, dim=-1)
        ).clamp_min(1e-6)

        cosine = numerator / denominator

        valid = (
            (torch.linalg.norm(a, dim=-1) > 1e-6)
            & (torch.linalg.norm(b, dim=-1) > 1e-6)
        )

        return torch.where(
            valid,
            cosine,
            torch.zeros_like(cosine),
        )

    def forward(
        self,
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        centers,
        search_center_xy,
        previous_final_xy,
        predicted_step_xy,
        previous_state,
    ):
        mean = raw_logits.mean(dim=1, keepdim=True)
        std = raw_logits.std(dim=1, keepdim=True).clamp_min(1e-5)
        standardized = (raw_logits - mean) / std

        relative_search = (
            centers - search_center_xy[:, None, :]
        )
        relative_radius = torch.linalg.norm(
            relative_search,
            dim=-1,
            keepdim=True,
        )

        candidate_step = (
            centers - previous_final_xy[:, None, :]
        )

        step_residual = (
            candidate_step
            - predicted_step_xy[:, None, :]
        )
        step_residual_radius = torch.linalg.norm(
            step_residual,
            dim=-1,
            keepdim=True,
        )

        predicted_step_expanded = predicted_step_xy[:, None, :].expand_as(
            candidate_step
        )

        forward_cosine = self._safe_cosine(
            candidate_step,
            predicted_step_expanded,
        ).unsqueeze(-1)

        numeric = torch.cat(
            [
                standardized.unsqueeze(-1),
                raw_prob.unsqueeze(-1),
                relative_search,
                relative_radius,
                step_residual,
                step_residual_radius,
                forward_cosine,
            ],
            dim=-1,
        )

        if int(numeric.shape[-1]) != int(config.CANDIDATE_NUMERIC_DIM):
            raise RuntimeError(
                "candidate numeric dimension mismatch: got %d expected %d"
                % (
                    int(numeric.shape[-1]),
                    int(config.CANDIDATE_NUMERIC_DIM),
                )
            )

        token = (
            self.uav_projection(z_uav)[:, None, :]
            + self.sat_projection(z_sat)
            + self.numeric_projection(numeric)
            + self.state_projection(previous_state)[:, None, :]
        )

        learned_residual = self.emission_head(token).squeeze(-1)

        sigma2 = float(config.MOTION_PRIOR_SIGMA_M) ** 2
        motion_prior = (
            -0.5
            * step_residual.square().sum(dim=-1)
            / sigma2
        )

        forward_prior = forward_cosine.squeeze(-1)

        refined_logits = (
            F.softplus(self.raw_weight_raw) * standardized
            + learned_residual
            + F.softplus(self.motion_weight_raw) * motion_prior
            + F.softplus(self.forward_weight_raw) * forward_prior
        )

        refined_probability = torch.softmax(
            refined_logits / float(config.MEANSHIFT_SCORE_TAU),
            dim=1,
        )

        sat_context = (
            refined_probability.unsqueeze(-1)
            * z_sat
        ).sum(dim=1)

        return CandidateRefinementOutput(
            refined_logits=refined_logits,
            refined_probability=refined_probability,
            sat_context=sat_context,
            raw_standardized=standardized,
            motion_prior=motion_prior,
            forward_prior=forward_prior,
        )


class CRFInertialRNN(nn.Module):
    """
    One current UAV frame + current SAT candidates + previous state only.

    Current measurement:
      refined candidate logits -> HardMS -> bounded residual correction

    Next-frame dynamics:
      RNN outputs v_t and a_t
      next polynomial step = v_t + 0.5*a_t

    The polynomial position is used OUTSIDE the model as the next 6x6 search
    center. The external Kalman is applied AFTER the measurement output.
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

        self.sat_context_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.previous_state_projection = nn.Sequential(
            nn.Linear(int(config.RNN_STATE_DIM), feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.rnn = nn.RNNCell(
            feature_dim * 4,
            hidden_dim,
            nonlinearity="tanh",
        )

        self.dropout = nn.Dropout(float(config.RNN_DROPOUT))

        def head(output_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        self.correction_head = head(2)
        self.correction_gate_head = head(1)
        self.velocity_head = head(2)
        self.acceleration_head = head(2)
        self.variance_head = head(2)

        self.latent_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                int(config.RNN_LATENT_STATE_DIM),
            ),
            nn.Tanh(),
        )

        for module in (
            self.correction_head,
            self.velocity_head,
            self.acceleration_head,
        ):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

        nn.init.zeros_(self.correction_gate_head[-1].weight)
        nn.init.constant_(
            self.correction_gate_head[-1].bias,
            float(config.CORRECTION_GATE_INITIAL_BIAS),
        )

        initial_var = 2.0
        inv_softplus = math.log(math.exp(initial_var) - 1.0)
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(self.variance_head[-1].bias, inv_softplus)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size),
            int(config.RNN_HIDDEN_DIM),
            device=device,
            dtype=dtype,
        )

    def initial_state(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size),
            int(config.RNN_STATE_DIM),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def bound_head_vector(raw_xy, maximum):
        component = torch.tanh(raw_xy) * float(maximum)
        norm = torch.linalg.norm(component, dim=1, keepdim=True)
        scale = torch.clamp(
            float(maximum) / norm.clamp_min(1e-6),
            max=1.0,
        )
        return component * scale

    @staticmethod
    def clip_vector(vector_xy, maximum):
        norm = torch.linalg.norm(vector_xy, dim=1, keepdim=True)
        scale = torch.clamp(
            float(maximum) / norm.clamp_min(1e-6),
            max=1.0,
        )
        return vector_xy * scale

    @staticmethod
    def entropy(probability):
        count = max(int(probability.shape[1]), 2)
        return -(
            probability
            * probability.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(count))

    @staticmethod
    def top_margin(probability):
        values = probability.topk(
            k=min(2, int(probability.shape[1])),
            dim=1,
        ).values

        if values.shape[1] == 1:
            return values[:, 0]

        return values[:, 0] - values[:, 1]

    def numeric_features(
        self,
        raw_probability,
        refined_probability,
        refined_hardms_xy,
        refined_hardms_support,
        raw_hardms_xy,
        raw_top1_xy,
        search_center_xy,
        previous_final_xy,
        predicted_step_xy,
        previous_velocity_xy,
        previous_acceleration_xy,
    ):
        innovation = (
            refined_hardms_xy - search_center_xy
        )
        innovation_radius = torch.linalg.norm(
            innovation,
            dim=1,
            keepdim=True,
        )

        hardms_step = (
            refined_hardms_xy - previous_final_xy
        )
        hardms_step_radius = torch.linalg.norm(
            hardms_step,
            dim=1,
            keepdim=True,
        )

        hardms_step_residual = (
            hardms_step - predicted_step_xy
        )
        hardms_step_residual_radius = torch.linalg.norm(
            hardms_step_residual,
            dim=1,
            keepdim=True,
        )

        raw_disagreement = (
            refined_hardms_xy - raw_hardms_xy
        )
        raw_disagreement_radius = torch.linalg.norm(
            raw_disagreement,
            dim=1,
            keepdim=True,
        )

        top1_disagreement = (
            refined_hardms_xy - raw_top1_xy
        )
        top1_disagreement_radius = torch.linalg.norm(
            top1_disagreement,
            dim=1,
            keepdim=True,
        )

        features = torch.cat(
            [
                self.entropy(raw_probability).unsqueeze(1),
                self.top_margin(raw_probability).unsqueeze(1),
                self.entropy(refined_probability).unsqueeze(1),
                self.top_margin(refined_probability).unsqueeze(1),
                refined_hardms_support.reshape(-1, 1),

                innovation,
                innovation_radius,

                hardms_step,
                hardms_step_radius,

                hardms_step_residual,
                hardms_step_residual_radius,

                raw_disagreement,
                raw_disagreement_radius,

                top1_disagreement_radius,

                previous_velocity_xy,
                previous_acceleration_xy,
            ],
            dim=1,
        )

        if int(features.shape[1]) != int(config.RNN_NUMERIC_DIM):
            raise RuntimeError(
                "RNN numeric dimension mismatch: got %d expected %d"
                % (
                    int(features.shape[1]),
                    int(config.RNN_NUMERIC_DIM),
                )
            )

        return features

    def forward_step(
        self,
        z_uav,
        sat_context,
        raw_probability,
        refined_probability,
        refined_hardms_xy,
        refined_hardms_support,
        raw_hardms_xy,
        raw_top1_xy,
        search_center_xy,
        previous_final_xy,
        predicted_step_xy,
        previous_velocity_xy,
        previous_acceleration_xy,
        previous_state,
        hidden,
    ):
        if previous_state is None:
            previous_state = self.initial_state(
                z_uav.shape[0],
                z_uav.device,
                z_uav.dtype,
            )

        if hidden is None:
            hidden = self.initial_hidden(
                z_uav.shape[0],
                z_uav.device,
                z_uav.dtype,
            )

        numeric = self.numeric_features(
            raw_probability=raw_probability,
            refined_probability=refined_probability,
            refined_hardms_xy=refined_hardms_xy,
            refined_hardms_support=refined_hardms_support,
            raw_hardms_xy=raw_hardms_xy,
            raw_top1_xy=raw_top1_xy,
            search_center_xy=search_center_xy,
            previous_final_xy=previous_final_xy,
            predicted_step_xy=predicted_step_xy,
            previous_velocity_xy=previous_velocity_xy,
            previous_acceleration_xy=previous_acceleration_xy,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.sat_context_projection(sat_context),
                self.numeric_projection(numeric),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )

        new_hidden = self.rnn(
            recurrent_input,
            hidden,
        )

        head_hidden = self.dropout(new_hidden)

        correction_xy = self.bound_head_vector(
            self.correction_head(head_hidden),
            float(config.MAX_RNN_CORRECTION_M),
        )

        correction_gate = torch.sigmoid(
            self.correction_gate_head(head_hidden)
        )

        measurement_xy = (
            refined_hardms_xy
            + correction_gate * correction_xy
        )

        velocity_xy = self.bound_head_vector(
            self.velocity_head(head_hidden),
            float(config.MAX_MODEL_VELOCITY_M_PER_FRAME),
        )

        acceleration_xy = self.bound_head_vector(
            self.acceleration_head(head_hidden),
            float(config.MAX_MODEL_ACCELERATION_M_PER_FRAME2),
        )

        next_step_xy = (
            velocity_xy + 0.5 * acceleration_xy
        )

        next_step_xy = self.clip_vector(
            next_step_xy,
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME),
        )

        measurement_variance = (
            F.softplus(self.variance_head(head_hidden))
            + float(config.KALMAN_R_MIN_VAR)
        ).clamp(
            max=float(config.KALMAN_R_MAX_VAR)
        )

        latent_state = self.latent_state_head(head_hidden)

        state = torch.cat(
            [
                velocity_xy
                / float(config.MAX_MODEL_VELOCITY_M_PER_FRAME),

                acceleration_xy
                / float(config.MAX_MODEL_ACCELERATION_M_PER_FRAME2),

                correction_xy
                / float(config.MAX_RNN_CORRECTION_M),

                correction_gate,

                measurement_variance
                / float(config.KALMAN_R_MAX_VAR),

                latent_state,
            ],
            dim=1,
        )

        if int(state.shape[1]) != int(config.RNN_STATE_DIM):
            raise RuntimeError(
                "state dimension mismatch: got %d expected %d"
                % (
                    int(state.shape[1]),
                    int(config.RNN_STATE_DIM),
                )
            )

        return CRFInertialRNNOutput(
            measurement_xy=measurement_xy,
            correction_xy=correction_xy,
            correction_gate=correction_gate,
            velocity_xy=velocity_xy,
            acceleration_xy=acceleration_xy,
            next_step_xy=next_step_xy,
            measurement_variance=measurement_variance,
            state=state,
            hidden=new_hidden,
        )
