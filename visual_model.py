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
        return F.normalize(self.uav_head(clip_feat.float()), dim=1)

    def encode_uav(self, uav, yaw=None):
        return self.encode_uav_from_clip(self.encode_clip_image(uav))

    def encode_sat_from_clip(self, sat_clip_feat, xy):
        if self.use_coord_encoder:
            coord_feat = self.coord_encoder(xy.float())
            sat_input = torch.cat([sat_clip_feat.float(), coord_feat], dim=1)
        else:
            sat_input = sat_clip_feat.float()
        return F.normalize(self.sat_head(sat_input), dim=1)

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
class ReversibleRouteOutput:
    fusion_logits: torch.Tensor
    fusion_probability: torch.Tensor
    leg_logits: torch.Tensor
    leg_probability: torch.Tensor
    measurement_variance: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor


class ReversibleTopologyRecoveryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = int(config.LSTM_FEATURE_DIM)
        hidden_dim = int(config.LSTM_HIDDEN_DIM)
        self.uav_projection = nn.Sequential(nn.Linear(config.EMBED_DIM, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        pair_dim = config.EMBED_DIM * 4 + 1
        self.pair_projection = nn.Sequential(nn.Linear(pair_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.local_projection = nn.Sequential(nn.Linear(config.EMBED_DIM, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.recovery_projection = nn.Sequential(nn.Linear(config.EMBED_DIM, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        numeric_dim = 4 + 4 + int(config.LEG_EVIDENCE_DIM) + int(config.ROUTE_CONTEXT_DIM)
        self.numeric_projection = nn.Sequential(nn.Linear(numeric_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.lstm = nn.LSTMCell(feature_dim * 5, hidden_dim)
        self.dropout = nn.Dropout(float(config.LSTM_DROPOUT))
        self.fusion_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, int(config.HYPOTHESIS_COUNT)))
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)
        self.leg_head = nn.Sequential(nn.Linear(hidden_dim + int(config.LEG_EVIDENCE_DIM), hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, int(config.LEG_STATE_COUNT)))
        nn.init.zeros_(self.leg_head[-1].weight)
        nn.init.zeros_(self.leg_head[-1].bias)
        with torch.no_grad():
            self.leg_head[-1].bias[int(config.LEG_CURRENT)] = 1.0
        self.variance_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 2))
        initial_var = 5.0
        inv_softplus = math.log(math.exp(initial_var) - 1.0)
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(self.variance_head[-1].bias, inv_softplus)

    def initial_state(self, batch_size, device, dtype):
        hidden = torch.zeros(int(batch_size), int(config.LSTM_HIDDEN_DIM), device=device, dtype=dtype)
        return hidden, torch.zeros_like(hidden)

    @staticmethod
    def masked_probability(logits, valid_mask):
        masked = torch.where(valid_mask, logits, torch.full_like(logits, -1e4))
        return torch.softmax(masked, dim=1)

    @staticmethod
    def visual_stats(logits, valid_mask):
        probability = ReversibleTopologyRecoveryLSTM.masked_probability(logits, valid_mask)
        valid_count = valid_mask.sum(dim=1).clamp_min(2).float()
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1) / valid_count.log()
        top2 = probability.topk(k=min(2, probability.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            margin = top2[:, 0]
            maximum = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]
            maximum = top2[:, 0]
        mask_float = valid_mask.float()
        count = mask_float.sum(dim=1).clamp_min(1.0)
        mean = (logits * mask_float).sum(dim=1) / count
        variance = (((logits - mean.unsqueeze(1)) ** 2) * mask_float).sum(dim=1) / count
        std = torch.tanh(variance.sqrt() / 10.0)
        return torch.stack([entropy, margin, maximum, std], dim=1)

    @staticmethod
    def weighted_context(logits, valid_mask, z_sat):
        p = ReversibleTopologyRecoveryLSTM.masked_probability(logits, valid_mask)
        return (p.unsqueeze(-1) * z_sat).sum(dim=1)

    def forward_step(self, z_uav, previous_z_uav, local_z_sat, local_raw_logits, local_valid_mask,
                     recovery_z_sat, recovery_raw_logits, recovery_valid_mask,
                     leg_evidence, leg_valid_mask, route_context, hidden, cell):
        batch = z_uav.shape[0]
        if hidden is None or cell is None:
            hidden, cell = self.initial_state(batch, z_uav.device, z_uav.dtype)
        if previous_z_uav is None:
            previous_z_uav = z_uav
        local_context = self.weighted_context(local_raw_logits, local_valid_mask, local_z_sat)
        recovery_context = self.weighted_context(recovery_raw_logits, recovery_valid_mask, recovery_z_sat)
        local_stats = self.visual_stats(local_raw_logits, local_valid_mask)
        recovery_stats = self.visual_stats(recovery_raw_logits, recovery_valid_mask)
        pair_cos = (z_uav * previous_z_uav).sum(dim=1, keepdim=True)
        pair = torch.cat([z_uav, previous_z_uav, torch.abs(z_uav - previous_z_uav), z_uav * previous_z_uav, pair_cos], dim=1)
        numeric = torch.cat([local_stats, recovery_stats, leg_evidence, route_context], dim=1)
        recurrent_input = torch.cat([
            self.uav_projection(z_uav), self.pair_projection(pair), self.local_projection(local_context),
            self.recovery_projection(recovery_context), self.numeric_projection(numeric)
        ], dim=1)
        hidden, cell = self.lstm(recurrent_input, (hidden, cell))
        head_hidden = self.dropout(hidden)
        fusion_logits = self.fusion_head(head_hidden)
        fusion_probability = torch.softmax(fusion_logits / float(config.BRANCH_SOFTMAX_TEMPERATURE), dim=1)
        raw_leg_logits = self.leg_head(torch.cat([head_hidden, leg_evidence], dim=1))
        leg_logits = torch.where(leg_valid_mask, raw_leg_logits, torch.full_like(raw_leg_logits, -1e4))
        leg_probability = torch.softmax(leg_logits, dim=1)
        measurement_variance = (F.softplus(self.variance_head(head_hidden)) + float(config.KALMAN_R_MIN_VAR)).clamp(max=float(config.KALMAN_R_MAX_VAR))
        return ReversibleRouteOutput(fusion_logits=fusion_logits, fusion_probability=fusion_probability,
                                     leg_logits=leg_logits, leg_probability=leg_probability,
                                     measurement_variance=measurement_variance, hidden=hidden, cell=cell)
