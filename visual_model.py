import math
from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FourierCoordEncoder(nn.Module):
    def __init__(self, num_bands=32, max_freq=16.0):
        super().__init__()
        freqs = torch.logspace(0.0, math.log10(max_freq), steps=num_bands)
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
        angles = xy[:, :, None] * self.freqs[None, None, :] * math.pi
        encoded = torch.cat(
            [xy, angles.sin().flatten(1), angles.cos().flatten(1)],
            dim=1,
        )
        return self.mlp(encoded)


class AllMapGeoCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, _ = open_clip.create_model_from_pretrained(config.BACKBONE_NAME)
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.use_coord_encoder = bool(getattr(config, "USE_COORD_ENCODER", False))
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

        self.use_qah_relation = bool(getattr(config, "USE_QAH_MS_RELATION", False))
        if self.use_qah_relation:
            relation_dim = int(getattr(config, "QAH_RELATION_DIM", 32))
            self.qah_relation_head = nn.Sequential(
                nn.Linear(config.EMBED_DIM * 2 + 1, 256),
                nn.GELU(),
                nn.Linear(256, relation_dim),
            )
        else:
            self.qah_relation_head = None

        self.use_basin_rank_ms = bool(getattr(config, "USE_BASIN_RANK_MS", False))
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

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(image.float())

    @torch.no_grad()
    def encode_clip_spatial(self, image, output_size=None):
        output_size = int(output_size or config.MOTION_SPATIAL_SIZE)
        feature = self.encode_clip_image(image).unsqueeze(-1).unsqueeze(-1)
        return F.adaptive_avg_pool2d(feature.float(), (output_size, output_size))

    def encode_uav_from_clip(self, clip_feat, yaw=None):
        embedding = self.uav_head(clip_feat.float())
        return F.normalize(embedding, dim=1)

    def encode_uav(self, uav, yaw=None):
        return self.encode_uav_from_clip(self.encode_clip_image(uav))

    def encode_sat_from_clip(self, sat_clip_feat, xy):
        if self.use_coord_encoder:
            coord_feat = self.coord_encoder(xy.float())
            sat_input = torch.cat([sat_clip_feat.float(), coord_feat], dim=1)
        else:
            sat_input = sat_clip_feat.float()
        embedding = self.sat_head(sat_input)
        return F.normalize(embedding, dim=1)

    def encode_relation(self, z_uav, z_sat, logits):
        if self.qah_relation_head is None:
            raise RuntimeError("QAH relation head is disabled")
        pair_product = z_uav.unsqueeze(1) * z_sat
        pair_absolute = torch.abs(z_uav.unsqueeze(1) - z_sat)
        pair_logit = logits.unsqueeze(-1)
        pair = torch.cat([pair_product, pair_absolute, pair_logit], dim=-1)
        relation = self.qah_relation_head(pair.reshape(-1, pair.shape[-1]))
        return relation.reshape(pair.shape[0], pair.shape[1], -1)


@dataclass
class StableVisualInertialRNNOutput:
    refined_logits: torch.Tensor
    candidate_probability: torch.Tensor
    motion_xy: torch.Tensor
    motion_unstopped_xy: torch.Tensor
    stop_logit: torch.Tensor
    stop_probability: torch.Tensor
    residual_xy: torch.Tensor
    measurement_variance: torch.Tensor
    hidden: torch.Tensor
    uav_state: torch.Tensor
    score_state: torch.Tensor


class StableVisualInertialRNN(nn.Module):
    """
    Plain nn.RNNCell.

    Current sensor inputs are visual only. previous_motion_xy is previous model
    state derived from images, not an external sensor.

    The RNN motion does NOT directly set current XY. Current XY is always
    anchored by one of the 36 current-image SAT candidates plus a small bounded
    residual. This prevents the "RNN keeps driving the coordinate forward"
    failures of v12/v13.

    previous_motion_xy is used to softly refine the candidate distribution, so
    temporal inertia/direction can help without deleting rear candidates.
    """

    def __init__(self):
        super().__init__()

        feature_dim = int(config.RNN_FEATURE_DIM)
        hidden_dim = int(config.RNN_HIDDEN_DIM)
        score_dim = int(config.CANDIDATE_COUNT) * 2

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
        self.score_projection = nn.Sequential(
            nn.Linear(score_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.uav_delta_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.score_delta_projection = nn.Sequential(
            nn.Linear(score_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.motion_state_projection = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.rnn = nn.RNNCell(
            feature_dim * 6,
            hidden_dim,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(float(config.RNN_DROPOUT))

        self.candidate_uav_projection = nn.Linear(config.EMBED_DIM, 96)
        self.candidate_sat_projection = nn.Linear(config.EMBED_DIM, 96)
        self.candidate_hidden_projection = nn.Linear(hidden_dim, 96)

        candidate_dim = (
            96
            + 96
            + 96
            + 1
            + 2
            + 2
            + 1
        )
        self.candidate_score_head = nn.Sequential(
            nn.Linear(candidate_dim, 192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.candidate_score_head[-1].weight)
        nn.init.zeros_(self.candidate_score_head[-1].bias)

        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.stop_head[-1].weight)
        nn.init.zeros_(self.stop_head[-1].bias)

        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        initial_var = 2.0
        inv_softplus = math.log(math.exp(initial_var) - 1.0)
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(self.variance_head[-1].bias, inv_softplus)

    def initial_state(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size),
            int(config.RNN_HIDDEN_DIM),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def visual_score_state(logits):
        probability = torch.softmax(logits, dim=1)
        standardized = (
            logits - logits.mean(dim=1, keepdim=True)
        ) / logits.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)
        return torch.cat([probability, standardized], dim=1)

    @staticmethod
    def bound_vector(raw_xy, maximum):
        norm = torch.linalg.norm(raw_xy, dim=1, keepdim=True)
        scale = (
            float(maximum)
            * torch.tanh(norm)
            / norm.clamp_min(1e-6)
        )
        return raw_xy * scale

    def refine_candidates(
        self,
        z_uav,
        z_sat,
        raw_logits,
        candidate_offsets_xy,
        previous_motion_xy,
        hidden,
    ):
        batch, count, _ = z_sat.shape

        q = self.candidate_uav_projection(z_uav).unsqueeze(1).expand(
            -1, count, -1
        )
        k = self.candidate_sat_projection(z_sat)
        h = self.candidate_hidden_projection(hidden).unsqueeze(1).expand(
            -1, count, -1
        )

        raw = torch.tanh(raw_logits.unsqueeze(-1) / 20.0)

        offset = (
            candidate_offsets_xy
            / float(config.CANDIDATE_OFFSET_SCALE_M)
        ).clamp(-2.0, 2.0)

        motion = (
            previous_motion_xy
            / float(config.MAX_STEP_M_PER_FRAME)
        ).clamp(-1.0, 1.0)
        motion_pair = motion.unsqueeze(1).expand(-1, count, -1)

        offset_norm = F.normalize(
            candidate_offsets_xy,
            dim=2,
            eps=1e-6,
        )
        motion_norm = F.normalize(
            previous_motion_xy,
            dim=1,
            eps=1e-6,
        ).unsqueeze(1)
        alignment = (offset_norm * motion_norm).sum(
            dim=2,
            keepdim=True,
        )

        feature = torch.cat(
            [
                q * k,
                torch.abs(q - k),
                h,
                raw,
                offset,
                motion_pair,
                alignment,
            ],
            dim=2,
        )

        residual = self.candidate_score_head(
            feature.reshape(-1, feature.shape[-1])
        ).reshape(batch, count)

        # The trained visual similarity remains the dominant term.
        return (
            raw_logits
            + float(config.CANDIDATE_REFINEMENT_SCALE)
            * torch.tanh(residual)
        )

    def forward_step(
        self,
        z_uav,
        z_sat,
        raw_logits,
        candidate_offsets_xy,
        previous_motion_xy,
        hidden=None,
        previous_uav_state=None,
        previous_score_state=None,
    ):
        if hidden is None:
            hidden = self.initial_state(
                z_uav.shape[0],
                z_uav.device,
                z_uav.dtype,
            )

        refined_logits = self.refine_candidates(
            z_uav=z_uav,
            z_sat=z_sat,
            raw_logits=raw_logits,
            candidate_offsets_xy=candidate_offsets_xy,
            previous_motion_xy=previous_motion_xy,
            hidden=hidden,
        )

        probability = torch.softmax(refined_logits, dim=1)
        sat_context = (
            probability.unsqueeze(-1) * z_sat
        ).sum(dim=1)

        score_state = self.visual_score_state(refined_logits)

        if previous_uav_state is None:
            uav_delta = torch.zeros_like(z_uav)
        else:
            uav_delta = z_uav - previous_uav_state

        if previous_score_state is None:
            score_delta = torch.zeros_like(score_state)
        else:
            score_delta = score_state - previous_score_state

        normalized_previous_motion = (
            previous_motion_xy / float(config.MAX_STEP_M_PER_FRAME)
        ).clamp(-1.0, 1.0)

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.sat_context_projection(sat_context),
                self.score_projection(score_state),
                self.uav_delta_projection(uav_delta),
                self.score_delta_projection(score_delta),
                self.motion_state_projection(normalized_previous_motion),
            ],
            dim=1,
        )

        new_hidden = self.rnn(recurrent_input, hidden)
        head_hidden = self.dropout(new_hidden)

        raw_motion = self.motion_head(head_hidden)
        motion_unstopped = self.bound_vector(
            raw_motion,
            float(config.MAX_STEP_M_PER_FRAME),
        )

        stop_logit = self.stop_head(head_hidden)
        stop_probability = torch.sigmoid(stop_logit)

        motion_xy = motion_unstopped * (1.0 - stop_probability)

        residual_xy = (
            torch.tanh(self.residual_head(head_hidden))
            * float(config.MAX_RESIDUAL_M)
        )

        variance = (
            F.softplus(self.variance_head(head_hidden))
            + float(config.KALMAN_R_MIN_VAR)
        ).clamp(max=float(config.KALMAN_R_MAX_VAR))

        return StableVisualInertialRNNOutput(
            refined_logits=refined_logits,
            candidate_probability=probability,
            motion_xy=motion_xy,
            motion_unstopped_xy=motion_unstopped,
            stop_logit=stop_logit,
            stop_probability=stop_probability,
            residual_xy=residual_xy,
            measurement_variance=variance,
            hidden=new_hidden,
            uav_state=z_uav,
            score_state=score_state,
        )
