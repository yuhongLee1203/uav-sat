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
class ContinuousProgressRNNOutput:
    refined_logits: torch.Tensor
    candidate_probability: torch.Tensor
    move_gate: torch.Tensor
    heading_residual_rad: torch.Tensor
    measurement_variance: torch.Tensor
    hidden: torch.Tensor


class ContinuousProgressVisualRNN(nn.Module):
    """
    Plain nn.RNNCell.

    Network inputs are image-derived only:
      current UAV embedding
      searched SAT embeddings
      UAV/SAT similarities
      previous recurrent hidden state

    Absolute XY, waypoint index, route progress, frame index and GT/GPS are not
    passed into forward_step().
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
        self.visual_stats_projection = nn.Sequential(
            nn.Linear(5, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        # Requested simple RNN: no LSTM, no GRU.
        self.rnn = nn.RNNCell(
            feature_dim * 3,
            hidden_dim,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(float(config.RNN_DROPOUT))

        self.candidate_uav_projection = nn.Linear(config.EMBED_DIM, 96)
        self.candidate_sat_projection = nn.Linear(config.EMBED_DIM, 96)
        self.candidate_hidden_projection = nn.Linear(hidden_dim, 96)

        self.candidate_score_head = nn.Sequential(
            nn.Linear(96 + 96 + 96 + 1, 160),
            nn.GELU(),
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Linear(80, 1),
        )
        # Epoch 1 starts from raw trained visual retrieval, not random reranking.
        nn.init.zeros_(self.candidate_score_head[-1].weight)
        nn.init.zeros_(self.candidate_score_head[-1].bias)

        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.move_head[-1].weight)
        nn.init.constant_(self.move_head[-1].bias, -0.6)

        self.heading_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.heading_head[-1].weight)
        nn.init.zeros_(self.heading_head[-1].bias)

        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        initial_var = 3.0
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
    def visual_stats(logits):
        probability = torch.softmax(logits, dim=1)
        count = max(2, int(logits.shape[1]))
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(float(count))
        top2 = probability.topk(k=min(2, int(probability.shape[1])), dim=1).values
        maximum = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
        mean = logits.mean(dim=1)
        std = logits.std(dim=1, unbiased=False)
        return torch.stack(
            [
                entropy,
                margin,
                maximum,
                torch.tanh(mean / 20.0),
                torch.tanh(std / 10.0),
            ],
            dim=1,
        )

    def refine_candidates(self, z_uav, z_sat, raw_logits, previous_hidden):
        batch, count, _ = z_sat.shape
        q = self.candidate_uav_projection(z_uav).unsqueeze(1).expand(-1, count, -1)
        k = self.candidate_sat_projection(z_sat)
        h = self.candidate_hidden_projection(previous_hidden).unsqueeze(1).expand(-1, count, -1)
        raw = torch.tanh(raw_logits.unsqueeze(-1) / 20.0)
        pair = torch.cat([q * k, torch.abs(q - k), h, raw], dim=2)
        residual = self.candidate_score_head(
            pair.reshape(-1, pair.shape[-1])
        ).reshape(batch, count)
        return raw_logits + float(config.CANDIDATE_REFINEMENT_SCALE) * torch.tanh(residual)

    def forward_step(self, z_uav, z_sat, raw_logits, hidden):
        if hidden is None:
            hidden = self.initial_state(z_uav.shape[0], z_uav.device, z_uav.dtype)

        refined_logits = self.refine_candidates(
            z_uav,
            z_sat,
            raw_logits,
            hidden,
        )
        probability = torch.softmax(refined_logits, dim=1)
        sat_context = (probability.unsqueeze(-1) * z_sat).sum(dim=1)
        stats = self.visual_stats(refined_logits)

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.sat_context_projection(sat_context),
                self.visual_stats_projection(stats),
            ],
            dim=1,
        )
        new_hidden = self.rnn(recurrent_input, hidden)
        head_hidden = self.dropout(new_hidden)

        move_gate = torch.sigmoid(self.move_head(head_hidden))
        heading_residual_rad = torch.tanh(self.heading_head(head_hidden)) * math.radians(
            float(config.RNN_HEADING_RESIDUAL_MAX_DEG)
        )
        variance = (
            F.softplus(self.variance_head(head_hidden))
            + float(config.KALMAN_R_MIN_VAR)
        ).clamp(max=float(config.KALMAN_R_MAX_VAR))

        return ContinuousProgressRNNOutput(
            refined_logits=refined_logits,
            candidate_probability=probability,
            move_gate=move_gate,
            heading_residual_rad=heading_residual_rad,
            measurement_variance=variance,
            hidden=new_hidden,
        )
