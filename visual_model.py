import math
from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# =============================================================================
# Existing single-frame visual retrieval model.
# Keep parameter names compatible with visual_localizer.py.
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

        (
            self.clip,
            _,
        ) = open_clip.create_model_from_pretrained(
            config.BACKBONE_NAME
        )

        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.use_coord_encoder = bool(
            getattr(
                config,
                "USE_COORD_ENCODER",
                False,
            )
        )

        self.uav_head = nn.Sequential(
            nn.Linear(
                config.CLIP_DIM,
                config.CLIP_DIM,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                config.CLIP_DIM,
                config.EMBED_DIM,
            ),
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

        feature = (
            self.encode_clip_image(image)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

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


# =============================================================================
# Route-conditioned recurrent visual model.
#
# This is deliberately NOT an absolute-coordinate regressor.
#
# Inputs:
#   current visual evidence
#   local candidate offsets expressed in the CURRENT LEG frame
#   route-relative context derived from start/end waypoints
#   previous model motion state [v_parallel, v_cross, a_parallel, a_cross]
#   LSTM hidden/cell
#
# The network never receives raw latitude/longitude or absolute current XY.
# =============================================================================

@dataclass
class RouteInertialLSTMOutput:
    refined_logits: torch.Tensor
    measurement_variance: torch.Tensor
    next_motion_state: torch.Tensor
    polynomial_delta: torch.Tensor
    inertia_strength: torch.Tensor
    polynomial_sigma: torch.Tensor
    raw_visual_offset: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor


class RouteInertialLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        feature_dim = int(
            config.LSTM_FEATURE_DIM
        )

        hidden_dim = int(
            config.LSTM_HIDDEN_DIM
        )

        self.uav_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.sat_context_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        # entropy, top1-top2 margin, max probability, logit std
        self.visual_stats_projection = nn.Sequential(
            nn.Linear(
                4,
                feature_dim // 2,
            ),
            nn.GELU(),
            nn.LayerNorm(
                feature_dim // 2
            ),
        )

        # route context:
        #   remaining_ratio
        #   normalized_cross_track
        #   normalized_log_leg_length
        self.route_context_projection = nn.Sequential(
            nn.Linear(
                3,
                feature_dim // 2,
            ),
            nn.GELU(),
            nn.LayerNorm(
                feature_dim // 2
            ),
        )

        # previous [v_parallel, v_cross, a_parallel, a_cross]
        self.motion_state_projection = nn.Sequential(
            nn.Linear(
                config.MOTION_STATE_DIM,
                feature_dim // 2,
            ),
            nn.GELU(),
            nn.LayerNorm(
                feature_dim // 2
            ),
        )

        # Raw visual expected displacement inside current candidate lattice.
        self.visual_offset_projection = nn.Sequential(
            nn.Linear(
                2,
                feature_dim // 2,
            ),
            nn.GELU(),
            nn.LayerNorm(
                feature_dim // 2
            ),
        )

        lstm_input_dim = (
            feature_dim
            + feature_dim
            + feature_dim // 2
            + feature_dim // 2
            + feature_dim // 2
            + feature_dim // 2
        )

        self.lstm = nn.LSTMCell(
            lstm_input_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            float(
                config.LSTM_DROPOUT
            )
        )

        # Candidate-wise visual score refinement.
        self.candidate_uav_projection = nn.Linear(
            config.EMBED_DIM,
            96,
        )

        self.candidate_sat_projection = nn.Linear(
            config.EMBED_DIM,
            96,
        )

        self.candidate_hidden_projection = nn.Linear(
            hidden_dim,
            96,
        )

        # Extra candidate features:
        #   standardized raw visual score        1
        #   normalized candidate offset         2
        #   normalized polynomial residual      2
        candidate_feature_dim = (
            96
            + 96
            + 96
            + 1
            + 2
            + 2
        )

        self.candidate_score_head = nn.Sequential(
            nn.Linear(
                candidate_feature_dim,
                160,
            ),
            nn.GELU(),
            nn.Linear(
                160,
                80,
            ),
            nn.GELU(),
            nn.Linear(
                80,
                1,
            ),
        )

        # Recurrent dynamic state for the NEXT frame.
        self.motion_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                4,
            ),
        )

        # Learned trust in the polynomial prior.
        # It can suppress inertia when the current visual evidence disagrees.
        self.inertia_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                2,
            ),
        )

        # Final Kalman measurement variance.
        self.variance_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                2,
            ),
        )

        initial_var = 9.0
        inverse_softplus = math.log(
            math.exp(initial_var)
            - 1.0
        )

        nn.init.zeros_(
            self.variance_head[-1].weight
        )

        nn.init.constant_(
            self.variance_head[-1].bias,
            inverse_softplus,
        )

    def initial_state(
        self,
        batch_size,
        device,
        dtype,
    ):
        hidden = torch.zeros(
            int(batch_size),
            int(config.LSTM_HIDDEN_DIM),
            device=device,
            dtype=dtype,
        )

        cell = torch.zeros_like(
            hidden
        )

        motion = torch.zeros(
            int(batch_size),
            int(config.MOTION_STATE_DIM),
            device=device,
            dtype=dtype,
        )

        return (
            hidden,
            cell,
            motion,
        )

    @staticmethod
    def visual_stats(
        raw_logits,
        raw_prob,
    ):
        count = max(
            int(raw_prob.shape[1]),
            2,
        )

        entropy = -(
            raw_prob
            * raw_prob.clamp_min(
                1e-8
            ).log()
        ).sum(
            dim=1
        ) / math.log(
            float(count)
        )

        top2 = raw_prob.topk(
            k=2,
            dim=1,
        ).values

        margin = (
            top2[:, 0]
            - top2[:, 1]
        )

        max_probability = (
            top2[:, 0]
        )

        logit_std = raw_logits.float().std(
            dim=1,
            unbiased=False,
        )

        logit_std = torch.tanh(
            logit_std / 10.0
        )

        return torch.stack(
            [
                entropy,
                margin,
                max_probability,
                logit_std,
            ],
            dim=1,
        )

    @staticmethod
    def normalize_motion_state(
        previous_motion,
    ):
        velocity = previous_motion[:, 0:2] / float(
            config.MOTION_MAX_V_M_PER_FRAME
        )

        acceleration = previous_motion[:, 2:4] / float(
            config.MOTION_MAX_A_M_PER_FRAME2
        )

        return torch.cat(
            [
                velocity,
                acceleration,
            ],
            dim=1,
        )

    def forward_step(
        self,
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        candidate_offsets_route,
        route_context,
        previous_motion,
        hidden,
        cell,
    ):
        """
        candidate_offsets_route:
            [B, N, 2], meters relative to the PREVIOUS VISUAL state.
            Axis 0 = forward along current start->end leg.
            Axis 1 = cross-track.

        route_context:
            [B, 3], no absolute XY:
              remaining_ratio
              normalized_cross_track
              normalized_log_leg_length

        previous_motion:
            [B, 4]:
              v_parallel, v_cross, a_parallel, a_cross

        The second-order polynomial is:
            delta_poly = v + 0.5*a

        IMPORTANT:
        delta_poly is only a candidate-score prior.
        It never moves the output coordinate on its own.
        """
        batch = z_uav.shape[0]

        if (
            hidden is None
            or cell is None
            or previous_motion is None
        ):
            (
                init_hidden,
                init_cell,
                init_motion,
            ) = self.initial_state(
                batch,
                z_uav.device,
                z_uav.dtype,
            )

            if hidden is None:
                hidden = init_hidden

            if cell is None:
                cell = init_cell

            if previous_motion is None:
                previous_motion = init_motion

        sat_context = (
            raw_prob.unsqueeze(-1)
            * z_sat
        ).sum(
            dim=1
        )

        offset_scale = candidate_offsets_route.abs().amax(
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(
            1.0
        )

        normalized_offsets = (
            candidate_offsets_route
            / offset_scale
        )

        raw_visual_offset = (
            raw_prob.unsqueeze(-1)
            * candidate_offsets_route
        ).sum(
            dim=1
        )

        normalized_visual_offset = (
            raw_visual_offset
            / offset_scale.squeeze(1)
        )

        stats = self.visual_stats(
            raw_logits,
            raw_prob,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(
                    z_uav
                ),
                self.sat_context_projection(
                    sat_context
                ),
                self.visual_stats_projection(
                    stats
                ),
                self.route_context_projection(
                    route_context
                ),
                self.motion_state_projection(
                    self.normalize_motion_state(
                        previous_motion
                    )
                ),
                self.visual_offset_projection(
                    normalized_visual_offset
                ),
            ],
            dim=1,
        )

        (
            hidden,
            cell,
        ) = self.lstm(
            recurrent_input,
            (
                hidden,
                cell,
            ),
        )

        head_hidden = self.dropout(
            hidden
        )

        # --------------------------------------------------------------
        # Explicit second-order polynomial from the PREVIOUS state.
        # --------------------------------------------------------------
        polynomial_delta = (
            previous_motion[:, 0:2]
            + 0.5
            * previous_motion[:, 2:4]
        )

        # --------------------------------------------------------------
        # Learned inertia strength and width.
        # Strength in [0, 1].
        # Sigma in [POLY_SIGMA_MIN_M, POLY_SIGMA_MAX_M].
        # --------------------------------------------------------------
        inertia_raw = self.inertia_head(
            head_hidden
        )

        inertia_strength = torch.sigmoid(
            inertia_raw[:, 0:1]
        )

        sigma_fraction = torch.sigmoid(
            inertia_raw[:, 1:2]
        )

        polynomial_sigma = (
            float(
                config.POLY_SIGMA_MIN_M
            )
            + sigma_fraction
            * (
                float(
                    config.POLY_SIGMA_MAX_M
                )
                - float(
                    config.POLY_SIGMA_MIN_M
                )
            )
        )

        # --------------------------------------------------------------
        # Candidate score refinement.
        # --------------------------------------------------------------
        candidate_count = z_sat.shape[1]

        uav_pair = (
            self.candidate_uav_projection(
                z_uav
            )
            .unsqueeze(1)
            .expand(
                -1,
                candidate_count,
                -1,
            )
        )

        sat_pair = self.candidate_sat_projection(
            z_sat
        )

        hidden_pair = (
            self.candidate_hidden_projection(
                head_hidden
            )
            .unsqueeze(1)
            .expand(
                -1,
                candidate_count,
                -1,
            )
        )

        standardized_raw_logit = (
            raw_logits
            - raw_logits.mean(
                dim=1,
                keepdim=True,
            )
        ) / raw_logits.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(
            1e-5
        )

        polynomial_residual = (
            candidate_offsets_route
            - polynomial_delta.unsqueeze(1)
        )

        normalized_poly_residual = (
            polynomial_residual
            / polynomial_sigma.unsqueeze(1)
        )

        pair_feature = torch.cat(
            [
                uav_pair,
                sat_pair,
                hidden_pair,
                standardized_raw_logit.unsqueeze(
                    -1
                ),
                normalized_offsets,
                normalized_poly_residual,
            ],
            dim=2,
        )

        learned_visual_residual = (
            self.candidate_score_head(
                pair_feature.reshape(
                    -1,
                    pair_feature.shape[-1],
                )
            )
            .reshape(
                batch,
                candidate_count,
            )
        )

        # Polynomial prior only penalizes candidates that disagree with
        # expected inertia. It cannot generate a location without image scores.
        polynomial_prior = -0.5 * (
            normalized_poly_residual.square()
        ).sum(
            dim=2
        )

        refined_logits = (
            standardized_raw_logit
            + learned_visual_residual
            + inertia_strength
            * polynomial_prior
        )

        # --------------------------------------------------------------
        # Next recurrent motion state.
        # --------------------------------------------------------------
        raw_motion = self.motion_head(
            head_hidden
        )

        next_velocity = (
            torch.tanh(
                raw_motion[:, 0:2]
            )
            * float(
                config.MOTION_MAX_V_M_PER_FRAME
            )
        )

        next_acceleration = (
            torch.tanh(
                raw_motion[:, 2:4]
            )
            * float(
                config.MOTION_MAX_A_M_PER_FRAME2
            )
        )

        next_motion_state = torch.cat(
            [
                next_velocity,
                next_acceleration,
            ],
            dim=1,
        )

        measurement_variance = (
            F.softplus(
                self.variance_head(
                    head_hidden
                )
            )
            + float(
                config.KALMAN_R_MIN_VAR
            )
        ).clamp(
            max=float(
                config.KALMAN_R_MAX_VAR
            )
        )

        return RouteInertialLSTMOutput(
            refined_logits=refined_logits,
            measurement_variance=measurement_variance,
            next_motion_state=next_motion_state,
            polynomial_delta=polynomial_delta,
            inertia_strength=inertia_strength,
            polynomial_sigma=polynomial_sigma,
            raw_visual_offset=raw_visual_offset,
            hidden=hidden,
            cell=cell,
        )
