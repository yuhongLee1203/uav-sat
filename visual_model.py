import math
from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# =============================================================================
# Existing single-frame visual model.
# Parameter names remain compatible with visual_localizer.py checkpoints.
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
# v6 Route-bounded recurrent hypothesis model
# =============================================================================

@dataclass
class HypothesisLSTMOutput:
    local_refined_logits: torch.Tensor
    recovery_refined_logits: torch.Tensor
    waypoint_refined_logits: torch.Tensor
    branch_logits: torch.Tensor
    branch_probability: torch.Tensor
    measurement_variance: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor


class RouteBoundedHypothesisLSTM(nn.Module):
    """
    The recurrent network never predicts absolute XY or velocity.

    Four image-conditioned hypotheses:
      HOLD      previous image-derived position
      LOCAL     current image matched near previous position
      RECOVERY  current image retrieved anywhere inside the active route leg
      WAYPOINT  current image matched in a small transition lattice centered
                at the current leg endpoint

    WAYPOINT being the strongest branch is the learned leg-transition event.
    There is no speed timer and no endpoint-distance inference threshold.
    """

    def __init__(self):
        super().__init__()

        feature_dim = int(
            config.LSTM_FEATURE_DIM
        )

        hidden_dim = int(
            config.LSTM_HIDDEN_DIM
        )

        self.current_uav_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        pair_input_dim = (
            config.EMBED_DIM * 4 + 1
        )

        self.frame_pair_projection = nn.Sequential(
            nn.Linear(
                pair_input_dim,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.local_sat_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.recovery_sat_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.waypoint_sat_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        # Numeric:
        # local stats 4
        # recovery stats 4
        # waypoint stats 4
        # route-global stats 4
        # route context 4
        # observed motion 4
        # local raw offset 2
        # recovery raw offset 2
        # waypoint raw offset 2
        # polynomial delta 2
        # total = 32
        numeric_dim = 32

        self.numeric_projection = nn.Sequential(
            nn.Linear(
                numeric_dim,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        lstm_input_dim = (
            feature_dim * 6
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

        # Shared image-candidate refinement head.
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

        self.branch_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                config.HYPOTHESIS_COUNT,
            ),
        )

        # Initial behavior: prefer LOCAL, allow HOLD, strongly suppress
        # RECOVERY/WAYPOINT until current visual evidence justifies them.
        nn.init.zeros_(
            self.branch_head[-1].weight
        )

        with torch.no_grad():
            self.branch_head[-1].bias.copy_(
                torch.tensor(
                    [
                        0.0,   # HOLD
                        1.0,   # LOCAL
                        -1.0,  # RECOVERY
                        -1.5,  # WAYPOINT
                    ],
                    dtype=self.branch_head[-1].bias.dtype,
                )
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

        initial_var = 6.0

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

        return (
            hidden,
            cell,
        )

    @staticmethod
    def _masked_probability(
        logits,
        valid_mask,
    ):
        masked_logits = torch.where(
            valid_mask,
            logits,
            torch.full_like(
                logits,
                -1e4,
            ),
        )

        return torch.softmax(
            masked_logits,
            dim=1,
        )

    @staticmethod
    def visual_stats(
        logits,
        valid_mask,
    ):
        probability = (
            RouteBoundedHypothesisLSTM
            ._masked_probability(
                logits,
                valid_mask,
            )
        )

        valid_count = valid_mask.sum(
            dim=1
        ).clamp_min(
            2
        ).float()

        entropy = -(
            probability
            * probability.clamp_min(
                1e-8
            ).log()
        ).sum(
            dim=1
        ) / valid_count.log()

        top2 = probability.topk(
            k=2,
            dim=1,
        ).values

        margin = (
            top2[:, 0]
            - top2[:, 1]
        )

        maximum = top2[:, 0]

        mask_float = valid_mask.float()

        count = mask_float.sum(
            dim=1
        ).clamp_min(
            1.0
        )

        mean = (
            logits
            * mask_float
        ).sum(
            dim=1
        ) / count

        variance = (
            (
                logits
                - mean.unsqueeze(1)
            ).square()
            * mask_float
        ).sum(
            dim=1
        ) / count

        std = torch.tanh(
            variance.sqrt()
            / 10.0
        )

        return torch.stack(
            [
                entropy,
                margin,
                maximum,
                std,
            ],
            dim=1,
        )

    @staticmethod
    def _normalize_observed_motion(
        observed_motion,
    ):
        return torch.clamp(
            observed_motion
            / float(
                config.OBSERVED_MOTION_SCALE_M
            ),
            min=-3.0,
            max=3.0,
        )

    @staticmethod
    def _weighted_sat_context(
        logits,
        valid_mask,
        z_sat,
    ):
        probability = (
            RouteBoundedHypothesisLSTM
            ._masked_probability(
                logits,
                valid_mask,
            )
        )

        return (
            probability.unsqueeze(-1)
            * z_sat
        ).sum(
            dim=1
        )

    @staticmethod
    def _raw_expected_offset(
        logits,
        valid_mask,
        offsets_route,
    ):
        probability = (
            RouteBoundedHypothesisLSTM
            ._masked_probability(
                logits,
                valid_mask,
            )
        )

        return (
            probability.unsqueeze(-1)
            * offsets_route
        ).sum(
            dim=1
        )

    def _refine_candidate_logits(
        self,
        z_uav,
        z_sat,
        raw_logits,
        valid_mask,
        offsets_route,
        polynomial_delta_route,
        hidden,
    ):
        batch = z_uav.shape[0]
        candidate_count = z_sat.shape[1]

        mask_float = valid_mask.float()

        count = mask_float.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(
            1.0
        )

        mean = (
            raw_logits
            * mask_float
        ).sum(
            dim=1,
            keepdim=True,
        ) / count

        variance = (
            (
                raw_logits
                - mean
            ).square()
            * mask_float
        ).sum(
            dim=1,
            keepdim=True,
        ) / count

        standardized = (
            raw_logits
            - mean
        ) / variance.sqrt().clamp_min(
            1e-5
        )

        standardized = torch.where(
            valid_mask,
            standardized,
            torch.full_like(
                standardized,
                -20.0,
            ),
        )

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

        sat_pair = (
            self.candidate_sat_projection(
                z_sat
            )
        )

        hidden_pair = (
            self.candidate_hidden_projection(
                hidden
            )
            .unsqueeze(1)
            .expand(
                -1,
                candidate_count,
                -1,
            )
        )

        normalized_offset = (
            offsets_route
            / float(
                config.CANDIDATE_OFFSET_SCALE_M
            )
        ).clamp(
            -3.0,
            3.0,
        )

        polynomial_residual = (
            offsets_route
            - polynomial_delta_route.unsqueeze(1)
        )

        normalized_polynomial_residual = (
            polynomial_residual
            / float(
                config.POLYNOMIAL_SCALE_M
            )
        ).clamp(
            -3.0,
            3.0,
        )

        feature = torch.cat(
            [
                uav_pair,
                sat_pair,
                hidden_pair,
                standardized.unsqueeze(-1),
                normalized_offset,
                normalized_polynomial_residual,
            ],
            dim=2,
        )

        residual = (
            self.candidate_score_head(
                feature.reshape(
                    -1,
                    feature.shape[-1],
                )
            )
            .reshape(
                batch,
                candidate_count,
            )
        )

        refined = (
            standardized
            + residual
        )

        refined = torch.where(
            valid_mask,
            refined,
            torch.full_like(
                refined,
                -1e4,
            ),
        )

        return refined

    def forward_step(
        self,
        z_uav,
        previous_z_uav,
        local_z_sat,
        local_raw_logits,
        local_valid_mask,
        local_offsets_route,
        recovery_z_sat,
        recovery_raw_logits,
        recovery_valid_mask,
        recovery_offsets_route,
        waypoint_z_sat,
        waypoint_raw_logits,
        waypoint_valid_mask,
        waypoint_offsets_route,
        global_stats,
        route_context,
        observed_motion,
        polynomial_delta_route,
        hidden,
        cell,
    ):
        batch = z_uav.shape[0]

        if hidden is None or cell is None:
            (
                hidden,
                cell,
            ) = self.initial_state(
                batch,
                z_uav.device,
                z_uav.dtype,
            )

        if previous_z_uav is None:
            previous_z_uav = z_uav

        local_context = (
            self._weighted_sat_context(
                local_raw_logits,
                local_valid_mask,
                local_z_sat,
            )
        )

        recovery_context = (
            self._weighted_sat_context(
                recovery_raw_logits,
                recovery_valid_mask,
                recovery_z_sat,
            )
        )

        waypoint_context = (
            self._weighted_sat_context(
                waypoint_raw_logits,
                waypoint_valid_mask,
                waypoint_z_sat,
            )
        )

        local_stats = self.visual_stats(
            local_raw_logits,
            local_valid_mask,
        )

        recovery_stats = self.visual_stats(
            recovery_raw_logits,
            recovery_valid_mask,
        )

        waypoint_stats = self.visual_stats(
            waypoint_raw_logits,
            waypoint_valid_mask,
        )

        local_raw_offset = (
            self._raw_expected_offset(
                local_raw_logits,
                local_valid_mask,
                local_offsets_route,
            )
        )

        recovery_raw_offset = (
            self._raw_expected_offset(
                recovery_raw_logits,
                recovery_valid_mask,
                recovery_offsets_route,
            )
        )

        waypoint_raw_offset = (
            self._raw_expected_offset(
                waypoint_raw_logits,
                waypoint_valid_mask,
                waypoint_offsets_route,
            )
        )

        pair_cosine = (
            z_uav
            * previous_z_uav
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
                pair_cosine,
            ],
            dim=1,
        )

        numeric = torch.cat(
            [
                local_stats,
                recovery_stats,
                waypoint_stats,
                global_stats,
                route_context,
                self._normalize_observed_motion(
                    observed_motion
                ),
                (
                    local_raw_offset
                    / float(
                        config.CANDIDATE_OFFSET_SCALE_M
                    )
                ).clamp(
                    -3.0,
                    3.0,
                ),
                (
                    recovery_raw_offset
                    / float(
                        config.CANDIDATE_OFFSET_SCALE_M
                    )
                ).clamp(
                    -3.0,
                    3.0,
                ),
                (
                    waypoint_raw_offset
                    / float(
                        config.CANDIDATE_OFFSET_SCALE_M
                    )
                ).clamp(
                    -3.0,
                    3.0,
                ),
                (
                    polynomial_delta_route
                    / float(
                        config.POLYNOMIAL_SCALE_M
                    )
                ).clamp(
                    -3.0,
                    3.0,
                ),
            ],
            dim=1,
        )

        recurrent_input = torch.cat(
            [
                self.current_uav_projection(
                    z_uav
                ),
                self.frame_pair_projection(
                    pair_feature
                ),
                self.local_sat_projection(
                    local_context
                ),
                self.recovery_sat_projection(
                    recovery_context
                ),
                self.waypoint_sat_projection(
                    waypoint_context
                ),
                self.numeric_projection(
                    numeric
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

        local_refined_logits = (
            self._refine_candidate_logits(
                z_uav,
                local_z_sat,
                local_raw_logits,
                local_valid_mask,
                local_offsets_route,
                polynomial_delta_route,
                head_hidden,
            )
        )

        recovery_refined_logits = (
            self._refine_candidate_logits(
                z_uav,
                recovery_z_sat,
                recovery_raw_logits,
                recovery_valid_mask,
                recovery_offsets_route,
                polynomial_delta_route,
                head_hidden,
            )
        )

        waypoint_refined_logits = (
            self._refine_candidate_logits(
                z_uav,
                waypoint_z_sat,
                waypoint_raw_logits,
                waypoint_valid_mask,
                waypoint_offsets_route,
                polynomial_delta_route,
                head_hidden,
            )
        )

        branch_logits = self.branch_head(
            head_hidden
        )

        branch_probability = torch.softmax(
            branch_logits
            / float(
                config.BRANCH_SOFTMAX_TEMPERATURE
            ),
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

        return HypothesisLSTMOutput(
            local_refined_logits=local_refined_logits,
            recovery_refined_logits=recovery_refined_logits,
            waypoint_refined_logits=waypoint_refined_logits,
            branch_logits=branch_logits,
            branch_probability=branch_probability,
            measurement_variance=measurement_variance,
            hidden=hidden,
            cell=cell,
        )
