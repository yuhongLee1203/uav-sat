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

        self.register_buffer(
            "freqs",
            freqs,
        )

        in_dim = (
            2
            + 2
            * 2
            * num_bands
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                in_dim,
                512,
            ),
            nn.GELU(),
            nn.Linear(
                512,
                512,
            ),
            nn.GELU(),
            nn.Linear(
                512,
                config.EMBED_DIM,
            ),
        )

    def forward(self, xy):
        xy = (
            xy.float()
            / 1000.0
        )

        angles = (
            xy[
                :,
                :,
                None,
            ]
            * self.freqs[
                None,
                None,
                :,
            ]
            * math.pi
        )

        encoded = torch.cat(
            [
                xy,
                angles.sin().flatten(
                    1
                ),
                angles.cos().flatten(
                    1
                ),
            ],
            dim=1,
        )

        return self.mlp(
            encoded
        )


class AllMapGeoCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        (
            self.clip,
            _,
        ) = (
            open_clip.create_model_from_pretrained(
                config.BACKBONE_NAME
            )
        )

        for parameter in self.clip.parameters():
            parameter.requires_grad_(
                False
            )

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
            nn.Dropout(
                0.1
            ),
            nn.Linear(
                config.CLIP_DIM,
                config.EMBED_DIM,
            ),
        )

        if self.use_coord_encoder:
            self.coord_encoder = (
                FourierCoordEncoder()
            )

            sat_head_in_dim = (
                config.CLIP_DIM
                + config.EMBED_DIM
            )
        else:
            self.coord_encoder = None
            sat_head_in_dim = (
                config.CLIP_DIM
            )

        self.sat_head = nn.Sequential(
            nn.Linear(
                sat_head_in_dim,
                config.CLIP_DIM,
            ),
            nn.GELU(),
            nn.Dropout(
                0.1
            ),
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

            self.qah_relation_head = (
                nn.Sequential(
                    nn.Linear(
                        config.EMBED_DIM
                        * 2
                        + 1,
                        256,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        256,
                        relation_dim,
                    ),
                )
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
            self.basin_ranker = (
                nn.Sequential(
                    nn.Linear(
                        6,
                        32,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        32,
                        16,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        16,
                        1,
                    ),
                )
            )
        else:
            self.basin_ranker = None

        self.logit_scale = nn.Parameter(
            torch.ones([])
            * math.log(
                1.0
                / 0.07
            )
        )

    @torch.no_grad()
    def encode_clip_image(
        self,
        image,
    ):
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
            self.encode_clip_image(
                image
            )
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
        return (
            self.encode_uav_from_clip(
                self.encode_clip_image(
                    uav
                )
            )
        )

    def encode_sat_from_clip(
        self,
        sat_clip_feat,
        xy,
    ):
        if self.use_coord_encoder:
            coord_feat = (
                self.coord_encoder(
                    xy.float()
                )
            )

            sat_input = torch.cat(
                [
                    sat_clip_feat.float(),
                    coord_feat,
                ],
                dim=1,
            )
        else:
            sat_input = (
                sat_clip_feat.float()
            )

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
        if (
            self.qah_relation_head
            is None
        ):
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

        pair_logit = (
            logits.unsqueeze(-1)
        )

        pair = torch.cat(
            [
                pair_product,
                pair_absolute,
                pair_logit,
            ],
            dim=-1,
        )

        relation = (
            self.qah_relation_head(
                pair.reshape(
                    -1,
                    pair.shape[-1],
                )
            )
        )

        return relation.reshape(
            pair.shape[0],
            pair.shape[1],
            -1,
        )


# =============================================================================
# v5: image-motion-gated route LSTM
# =============================================================================

@dataclass
class VisualMotionLSTMOutput:
    refined_logits: torch.Tensor
    phase_logits: torch.Tensor
    phase_probability: torch.Tensor
    translation_gate: torch.Tensor
    stationary_probability: torch.Tensor
    rotation_probability: torch.Tensor
    measurement_variance: torch.Tensor
    inertia_strength: torch.Tensor
    polynomial_sigma: torch.Tensor
    polynomial_delta_route: torch.Tensor
    raw_visual_offset_route: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor


class VisualMotionRouteLSTM(nn.Module):
    """
    No free-running velocity head exists.

    The model sees:
      - current UAV visual embedding
      - previous UAV visual embedding
      - current SAT visual embeddings and visual similarity
      - image-pair motion cue from consecutive UAV frames
      - current candidate relative offsets in the current leg frame
      - route-relative START/END context
      - previous OBSERVED motion state
      - previous hidden/cell

    The previous observed state comes from actual previous image-derived
    localizations, not from a learned constant/fixed speed.
    """

    def __init__(self):
        super().__init__()

        feature_dim = int(
            config.LSTM_FEATURE_DIM
        )

        hidden_dim = int(
            config.LSTM_HIDDEN_DIM
        )

        self.current_uav_projection = (
            nn.Sequential(
                nn.Linear(
                    config.EMBED_DIM,
                    feature_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim
                ),
            )
        )

        # Pair branch explicitly compares consecutive UAV images.
        pair_input_dim = (
            config.EMBED_DIM
            * 4
            + 1
        )

        self.frame_pair_projection = (
            nn.Sequential(
                nn.Linear(
                    pair_input_dim,
                    feature_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim
                ),
            )
        )

        self.sat_context_projection = (
            nn.Sequential(
                nn.Linear(
                    config.EMBED_DIM,
                    feature_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim
                ),
            )
        )

        # entropy, top1-top2 margin, max probability, logit std
        self.visual_stats_projection = (
            nn.Sequential(
                nn.Linear(
                    4,
                    feature_dim // 2,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim // 2
                ),
            )
        )

        self.image_motion_projection = (
            nn.Sequential(
                nn.Linear(
                    config.IMAGE_MOTION_CUE_DIM,
                    feature_dim // 2,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim // 2
                ),
            )
        )

        # remaining ratio, cross track, normalized leg length, final-leg flag
        self.route_context_projection = (
            nn.Sequential(
                nn.Linear(
                    4,
                    feature_dim // 2,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim // 2
                ),
            )
        )

        self.observed_motion_projection = (
            nn.Sequential(
                nn.Linear(
                    config.OBSERVED_MOTION_STATE_DIM,
                    feature_dim // 2,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim // 2
                ),
            )
        )

        self.raw_offset_projection = (
            nn.Sequential(
                nn.Linear(
                    2,
                    feature_dim // 2,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    feature_dim // 2
                ),
            )
        )

        lstm_input_dim = (
            feature_dim
            + feature_dim
            + feature_dim
            + feature_dim // 2
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

        # Explicit 3-state image-motion classifier:
        # stationary / translation / rotation.
        self.phase_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                config.PHASE_COUNT,
            ),
        )

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

        self.candidate_uav_projection = (
            nn.Linear(
                config.EMBED_DIM,
                96,
            )
        )

        self.candidate_sat_projection = (
            nn.Linear(
                config.EMBED_DIM,
                96,
            )
        )

        self.candidate_hidden_projection = (
            nn.Linear(
                hidden_dim,
                96,
            )
        )

        # Candidate feature:
        # uav 96 + sat 96 + hidden 96
        # raw visual score 1
        # normalized offset 2
        # normalized polynomial residual 2
        candidate_feature_dim = (
            96
            + 96
            + 96
            + 1
            + 2
            + 2
        )

        self.candidate_score_head = (
            nn.Sequential(
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
        )

        initial_var = 6.0

        inverse_softplus = math.log(
            math.exp(
                initial_var
            )
            - 1.0
        )

        nn.init.zeros_(
            self.variance_head[
                -1
            ].weight
        )

        nn.init.constant_(
            self.variance_head[
                -1
            ].bias,
            inverse_softplus,
        )

    def initial_state(
        self,
        batch_size,
        device,
        dtype,
    ):
        hidden = torch.zeros(
            int(
                batch_size
            ),
            int(
                config.LSTM_HIDDEN_DIM
            ),
            device=device,
            dtype=dtype,
        )

        cell = torch.zeros_like(
            hidden
        )

        observed_motion = torch.zeros(
            int(
                batch_size
            ),
            int(
                config.OBSERVED_MOTION_STATE_DIM
            ),
            device=device,
            dtype=dtype,
        )

        return (
            hidden,
            cell,
            observed_motion,
        )

    @staticmethod
    def visual_stats(
        raw_logits,
        raw_prob,
    ):
        count = max(
            int(
                raw_prob.shape[1]
            ),
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
            float(
                count
            )
        )

        top2 = raw_prob.topk(
            k=2,
            dim=1,
        ).values

        margin = (
            top2[
                :,
                0
            ]
            - top2[
                :,
                1
            ]
        )

        max_probability = (
            top2[
                :,
                0
            ]
        )

        logit_std = (
            raw_logits.float()
            .std(
                dim=1,
                unbiased=False,
            )
        )

        logit_std = torch.tanh(
            logit_std
            / 10.0
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
    def normalize_observed_motion(
        state,
    ):
        return torch.clamp(
            state
            / float(
                config.OBSERVED_MOTION_NORMALIZE_M
            ),
            min=-3.0,
            max=3.0,
        )

    def forward_step(
        self,
        z_uav,
        previous_z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        candidate_offsets_route,
        route_context,
        image_motion_cue,
        previous_observed_motion,
        hidden,
        cell,
    ):
        batch = z_uav.shape[
            0
        ]

        if hidden is None:
            (
                init_hidden,
                init_cell,
                init_motion,
            ) = self.initial_state(
                batch,
                z_uav.device,
                z_uav.dtype,
            )

            hidden = init_hidden

            if cell is None:
                cell = init_cell

            if previous_observed_motion is None:
                previous_observed_motion = (
                    init_motion
                )

        if cell is None:
            cell = torch.zeros_like(
                hidden
            )

        if previous_observed_motion is None:
            previous_observed_motion = torch.zeros(
                batch,
                int(
                    config.OBSERVED_MOTION_STATE_DIM
                ),
                device=z_uav.device,
                dtype=z_uav.dtype,
            )

        if previous_z_uav is None:
            previous_z_uav = z_uav

        sat_context = (
            raw_prob.unsqueeze(
                -1
            )
            * z_sat
        ).sum(
            dim=1
        )

        stats = self.visual_stats(
            raw_logits,
            raw_prob,
        )

        cosine_previous_current = (
            previous_z_uav
            * z_uav
        ).sum(
            dim=1,
            keepdim=True,
        )

        pair_feature = torch.cat(
            [
                z_uav,
                previous_z_uav,
                torch.abs(
                    z_uav
                    - previous_z_uav
                ),
                z_uav
                * previous_z_uav,
                cosine_previous_current,
            ],
            dim=1,
        )

        offset_scale = (
            candidate_offsets_route.abs()
            .amax(
                dim=(1, 2),
                keepdim=True,
            )
            .clamp_min(
                1.0
            )
        )

        normalized_offsets = (
            candidate_offsets_route
            / offset_scale
        )

        raw_visual_offset_route = (
            raw_prob.unsqueeze(
                -1
            )
            * candidate_offsets_route
        ).sum(
            dim=1
        )

        normalized_raw_visual_offset = (
            raw_visual_offset_route
            / offset_scale.squeeze(
                1
            )
        )

        recurrent_input = torch.cat(
            [
                self.current_uav_projection(
                    z_uav
                ),
                self.frame_pair_projection(
                    pair_feature
                ),
                self.sat_context_projection(
                    sat_context
                ),
                self.visual_stats_projection(
                    stats
                ),
                self.image_motion_projection(
                    image_motion_cue
                ),
                self.route_context_projection(
                    route_context
                ),
                self.observed_motion_projection(
                    self.normalize_observed_motion(
                        previous_observed_motion
                    )
                ),
                self.raw_offset_projection(
                    normalized_raw_visual_offset
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

        phase_logits = self.phase_head(
            head_hidden
        )

        phase_probability = torch.softmax(
            phase_logits,
            dim=1,
        )

        stationary_probability = (
            phase_probability[
                :,
                config.PHASE_STATIONARY:
                config.PHASE_STATIONARY
                + 1
            ]
        )

        translation_gate = (
            phase_probability[
                :,
                config.PHASE_TRANSLATION:
                config.PHASE_TRANSLATION
                + 1
            ]
        )

        rotation_probability = (
            phase_probability[
                :,
                config.PHASE_ROTATION:
                config.PHASE_ROTATION
                + 1
            ]
        )

        # Polynomial comes ONLY from previous OBSERVED image-derived motion.
        previous_velocity = (
            previous_observed_motion[
                :,
                0:2
            ]
        )

        previous_acceleration = (
            previous_observed_motion[
                :,
                2:4
            ]
        )

        polynomial_delta_route = (
            previous_velocity
            + 0.5
            * previous_acceleration
        )

        inertia_raw = self.inertia_head(
            head_hidden
        )

        inertia_strength = torch.sigmoid(
            inertia_raw[
                :,
                0:1
            ]
        )

        sigma_fraction = torch.sigmoid(
            inertia_raw[
                :,
                1:2
            ]
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

        candidate_count = z_sat.shape[
            1
        ]

        uav_pair = (
            self.candidate_uav_projection(
                z_uav
            )
            .unsqueeze(
                1
            )
            .expand(
                -1,
                candidate_count,
                -1,
            )
        )

        sat_pair = (
            self.candidate_sat_projection(
                z_sat
            )
        )

        hidden_pair = (
            self.candidate_hidden_projection(
                head_hidden
            )
            .unsqueeze(
                1
            )
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
            - polynomial_delta_route.unsqueeze(
                1
            )
        )

        normalized_polynomial_residual = (
            polynomial_residual
            / polynomial_sigma.unsqueeze(
                1
            )
        )

        candidate_feature = torch.cat(
            [
                uav_pair,
                sat_pair,
                hidden_pair,
                standardized_raw_logit.unsqueeze(
                    -1
                ),
                normalized_offsets,
                normalized_polynomial_residual,
            ],
            dim=2,
        )

        learned_visual_residual = (
            self.candidate_score_head(
                candidate_feature.reshape(
                    -1,
                    candidate_feature.shape[
                        -1
                    ],
                )
            )
            .reshape(
                batch,
                candidate_count,
            )
        )

        polynomial_prior = -0.5 * (
            normalized_polynomial_residual.square()
        ).sum(
            dim=2
        )

        # Critical anti-runaway rule:
        # polynomial prior disappears when the image-pair classifier says
        # stationary or rotation.
        effective_inertia = (
            translation_gate
            * inertia_strength
        )

        refined_logits = (
            standardized_raw_logit
            + learned_visual_residual
            + effective_inertia
            * polynomial_prior
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

        return VisualMotionLSTMOutput(
            refined_logits=refined_logits,
            phase_logits=phase_logits,
            phase_probability=phase_probability,
            translation_gate=translation_gate,
            stationary_probability=stationary_probability,
            rotation_probability=rotation_probability,
            measurement_variance=measurement_variance,
            inertia_strength=inertia_strength,
            polynomial_sigma=polynomial_sigma,
            polynomial_delta_route=polynomial_delta_route,
            raw_visual_offset_route=raw_visual_offset_route,
            hidden=hidden,
            cell=cell,
        )
