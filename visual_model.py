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
    """
    Existing single-frame retrieval model.

    The optional coordinate encoder remains for compatibility with the old
    visual-localizer code, but config.USE_COORD_ENCODER=False in this experiment.
    Therefore the actual visual retrieval network is image-only.
    """

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

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(image.float())

    @torch.no_grad()
    def encode_clip_spatial(self, image, output_size=None):
        output_size = int(output_size or config.MOTION_SPATIAL_SIZE)
        feature = self.encode_clip_image(image).unsqueeze(-1).unsqueeze(-1)
        return F.adaptive_avg_pool2d(
            feature.float(),
            (output_size, output_size),
        )

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

        pair = torch.cat(
            [pair_product, pair_absolute, pair_logit],
            dim=-1,
        )

        relation = self.qah_relation_head(
            pair.reshape(-1, pair.shape[-1])
        )

        return relation.reshape(pair.shape[0], pair.shape[1], -1)


@dataclass
class PureVisualLSTMOutput:
    refined_logits: torch.Tensor
    measurement_variance: torch.Tensor
    next_delta_xy: torch.Tensor
    local_visual_offset: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor


class PureVisualLSTM(nn.Module):
    """
    Temporal model with PURE VISUAL INPUTS ONLY.

    forward_step() accepts:
      z_uav      : current UAV image embedding
      z_sat      : current SAT image embeddings
      raw_logits : image-image similarity scores
      raw_prob   : softmax image-image probabilities
      relative_offsets: translation-invariant 6x6 lattice offsets only
      previous_motion: model-predicted displacement from the prior frame
      hidden/cell: previous recurrent visual state

    It does NOT accept:
      waypoint, absolute XY/GPS, velocity, previous position,
      Kalman state, GPS, timestamp, route direction, frame boundary.
    """

    def __init__(self):
        super().__init__()

        feature_dim = int(config.LSTM_FEATURE_DIM)
        hidden_dim = int(config.LSTM_HIDDEN_DIM)

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

        # Entropy, top1-top2 margin, max probability, logit std.
        # All four are derived strictly from image similarity.
        self.visual_stats_projection = nn.Sequential(
            nn.Linear(4, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
        )

        # Relative 6x6 lattice geometry: this encodes only where a candidate
        # sits *within the current local grid* (not global XY/GPS). It lets the
        # recurrent state interpret a high visual response as left/right or
        # forward/backward evidence for its next-frame displacement.
        self.local_offset_projection = nn.Sequential(
            nn.Linear(2, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
        )

        self.motion_state_projection = nn.Sequential(
            nn.Linear(2, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
        )

        lstm_input_dim = (
            feature_dim
            + feature_dim
            + feature_dim // 2
            + feature_dim // 2
            + feature_dim // 2
        )

        self.lstm = nn.LSTMCell(
            lstm_input_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(float(config.LSTM_DROPOUT))

        # Permutation-equivariant candidate-wise scoring head.
        # Candidate spatial coordinates/order are not inputs.
        self.candidate_uav_projection = nn.Linear(config.EMBED_DIM, 96)
        self.candidate_sat_projection = nn.Linear(config.EMBED_DIM, 96)
        self.candidate_hidden_projection = nn.Linear(hidden_dim, 96)

        self.candidate_score_head = nn.Sequential(
            nn.Linear(96 * 3 + 1, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        initial_var = 9.0
        inverse_softplus = math.log(math.exp(initial_var) - 1.0)

        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(self.variance_head[-1].bias, inverse_softplus)

        # Predict visual displacement to the next image from only RNN state.
        # Zero initialization starts conservatively, without forced motion.
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    def initial_state(self, batch_size, device, dtype):
        hidden = torch.zeros(
            int(batch_size),
            int(config.LSTM_HIDDEN_DIM),
            device=device,
            dtype=dtype,
        )
        cell = torch.zeros_like(hidden)
        return hidden, cell

    @staticmethod
    def visual_stats(raw_logits, raw_prob):
        count = max(int(raw_prob.shape[1]), 2)

        entropy = -(
            raw_prob
            * raw_prob.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(count))

        top2 = raw_prob.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        max_probability = top2[:, 0]

        logit_std = raw_logits.float().std(
            dim=1,
            unbiased=False,
        )
        logit_std = torch.tanh(logit_std / 10.0)

        return torch.stack(
            [entropy, margin, max_probability, logit_std],
            dim=1,
        )

    def forward_step(
        self,
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        relative_offsets,
        previous_motion,
        hidden,
        cell,
    ):
        batch = z_uav.shape[0]

        if hidden is None or cell is None:
            hidden, cell = self.initial_state(
                batch,
                z_uav.device,
                z_uav.dtype,
            )

        sat_context = (
            raw_prob.unsqueeze(-1)
            * z_sat
        ).sum(dim=1)

        if relative_offsets is None:
            relative_offsets = torch.zeros(
                z_sat.shape[0],
                z_sat.shape[1],
                2,
                device=z_sat.device,
                dtype=z_sat.dtype,
            )

        if previous_motion is None:
            previous_motion = torch.zeros(
                z_sat.shape[0],
                2,
                device=z_sat.device,
                dtype=z_sat.dtype,
            )

        # Normalize by the local grid extent, so the geometry is invariant to
        # map origin and expressed purely as a relative directional signal.
        offset_scale = relative_offsets.abs().amax(
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(1e-5)
        normalized_offsets = relative_offsets / offset_scale
        local_visual_offset = (
            raw_prob.unsqueeze(-1) * normalized_offsets
        ).sum(dim=1)

        stats = self.visual_stats(
            raw_logits,
            raw_prob,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.sat_context_projection(sat_context),
                self.visual_stats_projection(stats),
                self.local_offset_projection(local_visual_offset),
                self.motion_state_projection(
                    previous_motion / float(config.RNN_MAX_NEXT_DISPLACEMENT_M)
                ),
            ],
            dim=1,
        )

        hidden, cell = self.lstm(
            recurrent_input,
            (hidden, cell),
        )

        head_hidden = self.dropout(hidden)

        candidate_count = z_sat.shape[1]

        uav_pair = (
            self.candidate_uav_projection(z_uav)
            .unsqueeze(1)
            .expand(-1, candidate_count, -1)
        )

        sat_pair = self.candidate_sat_projection(z_sat)

        hidden_pair = (
            self.candidate_hidden_projection(head_hidden)
            .unsqueeze(1)
            .expand(-1, candidate_count, -1)
        )

        standardized_raw_logit = (
            raw_logits
            - raw_logits.mean(dim=1, keepdim=True)
        ) / raw_logits.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-5)

        pair_feature = torch.cat(
            [
                uav_pair,
                sat_pair,
                hidden_pair,
                standardized_raw_logit.unsqueeze(-1),
            ],
            dim=2,
        )

        learned_residual = (
            self.candidate_score_head(
                pair_feature.reshape(-1, pair_feature.shape[-1])
            )
            .reshape(batch, candidate_count)
        )

        refined_logits = standardized_raw_logit + learned_residual

        measurement_variance = (
            F.softplus(
                self.variance_head(head_hidden)
            )
            + float(config.MIN_MEASUREMENT_VARIANCE)
        ).clamp(
            max=float(config.MAX_MEASUREMENT_VARIANCE)
        )

        next_delta_xy = torch.tanh(
            self.motion_head(
                torch.cat([head_hidden, local_visual_offset], dim=1)
            )
        ) * float(config.RNN_MAX_NEXT_DISPLACEMENT_M)

        return PureVisualLSTMOutput(
            refined_logits=refined_logits,
            measurement_variance=measurement_variance,
            next_delta_xy=next_delta_xy,
            local_visual_offset=local_visual_offset,
            hidden=hidden,
            cell=cell,
        )
