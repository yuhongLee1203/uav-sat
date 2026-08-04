import math
from dataclasses import dataclass
from typing import List, Optional

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
            [xy, angles.sin().flatten(1), angles.cos().flatten(1)], dim=1
        )
        return self.mlp(encoded)


class AllMapGeoCLIP(nn.Module):
    """The archived visual model.

    Parameter names intentionally match the existing retrieval checkpoint.
    Spatial extraction adds no trainable parameter and therefore does not alter
    checkpoint compatibility.
    """

    def __init__(self):
        super().__init__()
        self.clip, _ = open_clip.create_model_from_pretrained(config.BACKBONE_NAME)
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.use_coord_encoder = bool(getattr(config, "USE_COORD_ENCODER", True))
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
        """Return a frozen NCHW spatial feature map.

        Recent OpenCLIP versions expose ``forward_intermediates``.  A
        ``forward_features`` fallback keeps the code usable with older local
        installations.  The final fallback is the pooled CLIP descriptor as a
        1x1 map, which preserves functionality while clearly exposing that no
        spatial motion cue was available.
        """
        self.clip.eval()
        output_size = int(output_size or config.MOTION_SPATIAL_SIZE)
        feature = None

        try:
            result = self.clip.forward_intermediates(
                image=image.float(),
                image_indices=1,
                intermediates_only=True,
                image_output_fmt="NCHW",
            )
            candidates = result.get("image_intermediates", None)
            if candidates is None:
                candidates = result.get("intermediates", None)
            if candidates is not None and len(candidates) > 0:
                feature = candidates[-1]
        except (AttributeError, RuntimeError, TypeError, NotImplementedError):
            feature = None

        if feature is None:
            visual = getattr(self.clip, "visual", None)
            if visual is not None and hasattr(visual, "forward_features"):
                try:
                    feature = visual.forward_features(image.float())
                except (RuntimeError, TypeError, NotImplementedError):
                    feature = None

        if isinstance(feature, (tuple, list)):
            feature = feature[-1]
        if isinstance(feature, dict):
            for key in ("x", "features", "last_hidden_state"):
                if key in feature:
                    feature = feature[key]
                    break

        if feature is None:
            feature = self.encode_clip_image(image).unsqueeze(-1).unsqueeze(-1)
        elif feature.ndim == 2:
            feature = feature.unsqueeze(-1).unsqueeze(-1)
        elif feature.ndim == 3:
            # NLC token tensor.  Remove a prefix token when the remainder is a
            # square grid, then reshape to NCHW.
            batch, tokens, channels = feature.shape
            spatial_tokens = tokens
            start = 0
            root = int(round(math.sqrt(tokens)))
            if root * root != tokens:
                root = int(round(math.sqrt(tokens - 1)))
                if root * root == tokens - 1:
                    spatial_tokens = tokens - 1
                    start = 1
            if root * root == spatial_tokens:
                feature = (
                    feature[:, start : start + spatial_tokens]
                    .transpose(1, 2)
                    .reshape(batch, channels, root, root)
                )
            else:
                feature = feature.transpose(1, 2).unsqueeze(-1)
        elif feature.ndim != 4:
            raise RuntimeError(
                f"Unsupported spatial feature shape: {tuple(feature.shape)}"
            )

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
            raise RuntimeError(
                "QAH relation head is disabled; set USE_QAH_MS_RELATION=True"
            )
        pair_product = z_uav.unsqueeze(1) * z_sat
        pair_absolute = torch.abs(z_uav.unsqueeze(1) - z_sat)
        pair_logit = logits.unsqueeze(-1)
        pair = torch.cat(
            [pair_product, pair_absolute, pair_logit], dim=-1
        )
        relation = self.qah_relation_head(pair.reshape(-1, pair.shape[-1]))
        return relation.reshape(pair.shape[0], pair.shape[1], -1)



@dataclass
class LatticeOutput:
    emission_logits: torch.Tensor
    path_probability: torch.Tensor
    path_expectation: torch.Tensor
    final_xy: torch.Tensor
    correction_gate: torch.Tensor
    path_entropy: torch.Tensor
    emission_entropy: torch.Tensor
    crf_nll: Optional[torch.Tensor]


class TemporalLatticeCRF(nn.Module):
    """Residual second-order CRF over consecutive retrieval lattices.

    Each frame contributes all candidate nodes rather than one early Top-1.
    A learned second-order transition potential scores velocity continuation
    and acceleration, so the model learns which candidate sequence is visually
    strong and approximately straight.  The final prediction is a learned
    residual correction of Fixed HardMS, which makes the visual baseline an
    explicit fallback instead of allowing catastrophic temporal drift.
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
        self.numeric_projection = nn.Sequential(
            nn.Linear(5, token_dim),
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
        # softplus(0.5413) ~= 1, therefore the untrained model retains the raw
        # retrieval ordering while learning a residual emission calibration.
        self.raw_logit_weight = nn.Parameter(torch.tensor(0.5413249))

        hidden = int(config.TRANSITION_HIDDEN)
        self.first_transition = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.second_transition = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # The initial structured prior is constant velocity.  The MLP learns
        # route-specific deviations; the positive quadratic coefficient remains
        # trainable instead of being a manually selected threshold.
        nn.init.zeros_(self.first_transition[-1].weight)
        nn.init.zeros_(self.first_transition[-1].bias)
        nn.init.zeros_(self.second_transition[-1].weight)
        nn.init.zeros_(self.second_transition[-1].bias)
        self.acceleration_weight_raw = nn.Parameter(torch.tensor(0.0))

        self.correction_gate = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Start close to Fixed HardMS; temporal correction must earn a larger
        # gate through coordinate supervision.
        nn.init.zeros_(self.correction_gate[-1].weight)
        nn.init.constant_(self.correction_gate[-1].bias, -2.0)

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        count = max(int(probability.shape[-1]), 2)
        return -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(count))

    def _emissions(
        self,
        z_uav: torch.Tensor,
        z_sat: torch.Tensor,
        raw_logits: torch.Tensor,
        raw_prob: torch.Tensor,
        centers: torch.Tensor,
    ):
        lattice_center = centers.mean(dim=2, keepdim=True)
        relative = (centers - lattice_center) / float(config.POSITION_SCALE_M)
        radius = torch.linalg.norm(relative, dim=-1, keepdim=True)
        mean = raw_logits.mean(dim=2, keepdim=True)
        std = raw_logits.std(dim=2, keepdim=True).clamp_min(1e-5)
        standardized = (raw_logits - mean) / std
        numeric = torch.cat(
            [
                standardized.unsqueeze(-1),
                raw_prob.unsqueeze(-1),
                relative,
                radius,
            ],
            dim=-1,
        )
        token = (
            self.uav_projection(z_uav)[:, :, None, :]
            + self.sat_projection(z_sat)
            + self.numeric_projection(numeric)
        )
        learned = self.emission_head(token).squeeze(-1)
        emission = learned + F.softplus(self.raw_logit_weight) * standardized
        return emission, token

    def _first_transition_score(
        self,
        centers0: torch.Tensor,
        centers1: torch.Tensor,
        dt: torch.Tensor,
    ):
        velocity = (
            centers1[:, None, :, :] - centers0[:, :, None, :]
        ) / dt[:, None, None, None].clamp_min(1.0)
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        feature = torch.cat(
            [velocity / float(config.POSITION_SCALE_M),
             speed / float(config.POSITION_SCALE_M)],
            dim=-1,
        )
        return self.first_transition(feature).squeeze(-1)

    def _second_transition_score(
        self,
        centers0: torch.Tensor,
        centers1: torch.Tensor,
        centers2: torch.Tensor,
        dt01: torch.Tensor,
        dt12: torch.Tensor,
    ):
        p0 = centers0[:, :, None, None, :]
        p1 = centers1[:, None, :, None, :]
        p2 = centers2[:, None, None, :, :]
        v01 = (p1 - p0) / dt01[:, None, None, None, None].clamp_min(1.0)
        v12 = (p2 - p1) / dt12[:, None, None, None, None].clamp_min(1.0)
        candidate_count = centers0.shape[1]
        v01 = v01.expand(-1, -1, -1, candidate_count, -1)
        v12 = v12.expand(-1, candidate_count, -1, -1, -1)
        acceleration = v12 - v01
        speed01 = torch.linalg.norm(v01, dim=-1, keepdim=True)
        speed12 = torch.linalg.norm(v12, dim=-1, keepdim=True)
        cosine = (v01 * v12).sum(dim=-1, keepdim=True) / (
            speed01 * speed12
        ).clamp_min(1e-5)
        feature = torch.cat(
            [
                v01 / float(config.POSITION_SCALE_M),
                v12 / float(config.POSITION_SCALE_M),
                acceleration / float(config.POSITION_SCALE_M),
                speed01 / float(config.POSITION_SCALE_M),
                speed12 / float(config.POSITION_SCALE_M),
                cosine,
            ],
            dim=-1,
        )
        learned = self.second_transition(feature).squeeze(-1)
        acceleration_energy = acceleration.square().sum(dim=-1) / (
            float(config.POSITION_SCALE_M) ** 2
        )
        return learned - F.softplus(self.acceleration_weight_raw) * acceleration_energy

    @staticmethod
    def _gather_path_score(
        emission: torch.Tensor,
        first_score: torch.Tensor,
        second_scores: List[torch.Tensor],
        target_index: torch.Tensor,
    ):
        batch = torch.arange(emission.shape[0], device=emission.device)
        score = torch.zeros(emission.shape[0], device=emission.device)
        for time in range(emission.shape[1]):
            score = score + emission[batch, time, target_index[:, time]]
        score = score + first_score[
            batch, target_index[:, 0], target_index[:, 1]
        ]
        for time, transition in enumerate(second_scores, start=2):
            score = score + transition[
                batch,
                target_index[:, time - 2],
                target_index[:, time - 1],
                target_index[:, time],
            ]
        return score

    def forward(
        self,
        z_uav: torch.Tensor,
        z_sat: torch.Tensor,
        raw_logits: torch.Tensor,
        raw_prob: torch.Tensor,
        centers: torch.Tensor,
        frame_ids: torch.Tensor,
        hardms_xy: torch.Tensor,
        target_index: Optional[torch.Tensor] = None,
    ) -> LatticeOutput:
        if centers.shape[1] < 3:
            raise ValueError("TemporalLatticeCRF requires at least three frames")

        emission, _ = self._emissions(
            z_uav, z_sat, raw_logits, raw_prob, centers
        )
        dt = (frame_ids[:, 1:] - frame_ids[:, :-1]).float().clamp_min(1.0)
        first_score = self._first_transition_score(
            centers[:, 0], centers[:, 1], dt[:, 0]
        )
        alpha = (
            emission[:, 0, :, None]
            + emission[:, 1, None, :]
            + first_score
        )
        second_scores: List[torch.Tensor] = []
        for time in range(2, centers.shape[1]):
            transition = self._second_transition_score(
                centers[:, time - 2],
                centers[:, time - 1],
                centers[:, time],
                dt[:, time - 2],
                dt[:, time - 1],
            )
            second_scores.append(transition)
            score = (
                alpha[:, :, :, None]
                + transition
                + emission[:, time, None, None, :]
            )
            alpha = torch.logsumexp(score, dim=1)

        log_partition = torch.logsumexp(alpha.flatten(1), dim=1)
        last_log_probability = torch.logsumexp(alpha, dim=1) - log_partition[:, None]
        path_probability = last_log_probability.exp()
        path_expectation = (
            path_probability.unsqueeze(-1) * centers[:, -1]
        ).sum(dim=1)

        emission_probability = torch.softmax(emission[:, -1], dim=1)
        path_entropy = self._entropy(path_probability)
        emission_entropy = self._entropy(emission_probability)
        path_top = path_probability.topk(k=2, dim=1).values
        emission_top = emission_probability.topk(k=2, dim=1).values
        disagreement = torch.linalg.norm(
            path_expectation - hardms_xy[:, -1], dim=1
        ) / float(config.POSITION_SCALE_M)
        gate_feature = torch.stack(
            [
                path_entropy,
                emission_entropy,
                path_top[:, 0] - path_top[:, 1],
                emission_top[:, 0] - emission_top[:, 1],
                disagreement,
            ],
            dim=1,
        )
        correction_gate = torch.sigmoid(self.correction_gate(gate_feature))
        final_xy = hardms_xy[:, -1] + correction_gate * (
            path_expectation - hardms_xy[:, -1]
        )

        crf_nll = None
        if target_index is not None:
            target_score = self._gather_path_score(
                emission, first_score, second_scores, target_index
            )
            crf_nll = (log_partition - target_score).mean()

        return LatticeOutput(
            emission_logits=emission,
            path_probability=path_probability,
            path_expectation=path_expectation,
            final_xy=final_xy,
            correction_gate=correction_gate,
            path_entropy=path_entropy,
            emission_entropy=emission_entropy,
            crf_nll=crf_nll,
        )