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
class RecurrentVisualMeasurementOutput:
    refined_logits: torch.Tensor
    candidate_probability: torch.Tensor
    residual_xy: torch.Tensor
    measurement_variance: torch.Tensor
    hidden: torch.Tensor
    uav_state: torch.Tensor
    score_state: torch.Tensor

class RecurrentVisualMeasurementRNN(nn.Module):
    def __init__(self):
        super().__init__()
        f = int(config.RNN_FEATURE_DIM); h = int(config.RNN_HIDDEN_DIM); sdim = int(config.CANDIDATE_COUNT) * 2
        self.uav_projection = nn.Sequential(nn.Linear(config.EMBED_DIM, f), nn.GELU(), nn.LayerNorm(f))
        self.sat_context_projection = nn.Sequential(nn.Linear(config.EMBED_DIM, f), nn.GELU(), nn.LayerNorm(f))
        self.score_projection = nn.Sequential(nn.Linear(sdim, f), nn.GELU(), nn.LayerNorm(f))
        self.uav_delta_projection = nn.Sequential(nn.Linear(config.EMBED_DIM, f), nn.GELU(), nn.LayerNorm(f))
        self.score_delta_projection = nn.Sequential(nn.Linear(sdim, f), nn.GELU(), nn.LayerNorm(f))
        self.visual_stats_projection = nn.Sequential(nn.Linear(5, f), nn.GELU(), nn.LayerNorm(f))
        self.rnn = nn.RNNCell(f * 6, h, nonlinearity="tanh")
        self.dropout = nn.Dropout(float(config.RNN_DROPOUT))
        self.candidate_delta_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, int(config.CANDIDATE_COUNT)))
        nn.init.zeros_(self.candidate_delta_head[-1].weight); nn.init.zeros_(self.candidate_delta_head[-1].bias)
        self.residual_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 2))
        nn.init.zeros_(self.residual_head[-1].weight); nn.init.zeros_(self.residual_head[-1].bias)
        self.variance_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 2))
        initial_var = 3.0; inv_softplus = math.log(math.exp(initial_var) - 1.0)
        nn.init.zeros_(self.variance_head[-1].weight); nn.init.constant_(self.variance_head[-1].bias, inv_softplus)
    def initial_state(self, batch_size, device, dtype):
        return torch.zeros(int(batch_size), int(config.RNN_HIDDEN_DIM), device=device, dtype=dtype)
    @staticmethod
    def score_state(logits):
        p = torch.softmax(logits, dim=1); z = (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4); return torch.cat([p, z], dim=1)
    @staticmethod
    def visual_stats(logits):
        p = torch.softmax(logits, dim=1); entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(logits.shape[1])); top2 = p.topk(k=min(2, int(p.shape[1])), dim=1).values; maximum = top2[:,0]; margin = top2[:,0]-top2[:,1] if top2.shape[1]>1 else top2[:,0]; return torch.stack([entropy, margin, maximum, torch.tanh(logits.mean(dim=1)/20.0), torch.tanh(logits.std(dim=1, unbiased=False)/10.0)], dim=1)
    def forward_step(self, z_uav, z_sat, raw_logits, hidden=None, previous_uav_state=None, previous_score_state=None):
        if hidden is None: hidden = self.initial_state(z_uav.shape[0], z_uav.device, z_uav.dtype)
        raw_p = torch.softmax(raw_logits, dim=1); sat_context = (raw_p.unsqueeze(-1) * z_sat).sum(dim=1); current_score = self.score_state(raw_logits)
        uav_delta = torch.zeros_like(z_uav) if previous_uav_state is None else z_uav - previous_uav_state
        score_delta = torch.zeros_like(current_score) if previous_score_state is None else current_score - previous_score_state
        rnn_input = torch.cat([self.uav_projection(z_uav), self.sat_context_projection(sat_context), self.score_projection(current_score), self.uav_delta_projection(uav_delta), self.score_delta_projection(score_delta), self.visual_stats_projection(self.visual_stats(raw_logits))], dim=1)
        new_hidden = self.rnn(rnn_input, hidden); hh = self.dropout(new_hidden)
        delta = float(config.CANDIDATE_REFINEMENT_SCALE) * torch.tanh(self.candidate_delta_head(hh)); refined = raw_logits + delta; probability = torch.softmax(refined, dim=1)
        residual_xy = torch.tanh(self.residual_head(hh)) * float(config.MAX_SUBPATCH_RESIDUAL_M)
        variance = (F.softplus(self.variance_head(hh)) + float(config.KF_MEASUREMENT_MIN_VAR)).clamp(max=float(config.KF_MEASUREMENT_MAX_VAR))
        return RecurrentVisualMeasurementOutput(refined, probability, residual_xy, variance, new_hidden, z_uav, current_score)
