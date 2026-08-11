import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config

try:
    import open_clip
except ImportError:
    open_clip = None


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


class FrozenBackboneAdapter(nn.Module):
    """Frozen public visual backbone with one unified encode_image() interface.

    MobileCLIP2-S2 follows the repository's existing behavior exactly.

    TorchVision ImageNet CNNs receive the preprocessing expected by their
    pretrained weights *inside* this adapter:
      RGB [0,1] -> bilinear resize 224x224 -> ImageNet mean/std normalization.

    Only the feature extractor is retained; classification heads are removed.
    """

    def __init__(self, backbone_key):
        super().__init__()
        self.backbone_key = str(backbone_key)
        self.is_mobileclip = self.backbone_key == "mobileclip2_s2"

        if self.is_mobileclip:
            if open_clip is None:
                raise ImportError(
                    "open_clip is required for RTL_BACKBONE=mobileclip2_s2"
                )
            self.model, _ = open_clip.create_model_from_pretrained(
                "hf-hub:timm/MobileCLIP2-S2-OpenCLIP"
            )
            self.feature_dim = 512

        elif self.backbone_key == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            base = models.vgg16(weights=weights)
            # Bearing-UAV uses VGG16 as a feature extractor, not the ImageNet
            # classifier. Removing classifier also avoids keeping ~123M unused
            # fully-connected parameters in GPU memory.
            base.classifier = nn.Identity()
            self.model = base.features
            self.feature_dim = 512

        elif self.backbone_key == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            base = models.resnet18(weights=weights)
            base.fc = nn.Identity()
            self.model = base
            self.feature_dim = 512

        elif self.backbone_key == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            base = models.resnet50(weights=weights)
            base.fc = nn.Identity()
            self.model = base
            self.feature_dim = 2048

        elif self.backbone_key == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            base = models.mobilenet_v3_small(weights=weights)
            base.classifier = nn.Identity()
            self.model = base.features
            self.feature_dim = 576

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_key}")

        if int(self.feature_dim) != int(config.CLIP_DIM):
            raise RuntimeError(
                f"Backbone output dim mismatch: adapter={self.feature_dim}, "
                f"config.CLIP_DIM={config.CLIP_DIM}"
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _torchvision_preprocess(self, image):
        x = image.float()
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self._imagenet_mean) / self._imagenet_std
        return x

    @torch.no_grad()
    def encode_image(self, image):
        self.eval()

        if self.is_mobileclip:
            # Preserve the original repository's exact MobileCLIP input path.
            return self.model.encode_image(image.float())

        x = self._torchvision_preprocess(image)

        if self.backbone_key == "vgg16":
            feature = self.model(x)
            return F.adaptive_avg_pool2d(feature, 1).flatten(1)

        if self.backbone_key in ("resnet18", "resnet50"):
            return self.model(x)

        if self.backbone_key == "mobilenet_v3_small":
            feature = self.model(x)
            return F.adaptive_avg_pool2d(feature, 1).flatten(1)

        raise RuntimeError("Unreachable backbone branch")


class AllMapGeoCLIP(nn.Module):
    """Same retrieval heads as the current repository, swappable frozen backbone."""

    def __init__(self):
        super().__init__()

        # Keep the attribute name `clip` intentionally.  The existing
        # visual_localizer.py excludes every state_dict key beginning with
        # `clip.` from task-specific checkpoints, so all frozen backbones remain
        # public-pretrained-only and checkpoint provenance stays clean.
        self.clip = FrozenBackboneAdapter(config.BACKBONE_KEY)

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

        self.use_basin_rank_ms = bool(
            getattr(config, "USE_BASIN_RANK_MS", False)
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
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(image.float())

    @torch.no_grad()
    def encode_clip_spatial(self, image, output_size=None):
        # The current T2-only pipeline does not consume spatial backbone maps.
        # Keep this compatibility method in case an older helper calls it.
        output_size = int(output_size or config.MOTION_SPATIAL_SIZE)
        pooled = self.encode_clip_image(image)
        feature = pooled.unsqueeze(-1).unsqueeze(-1)
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
            sat_input = torch.cat(
                [sat_clip_feat.float(), coord_feat],
                dim=1,
            )
        else:
            sat_input = sat_clip_feat.float()

        embedding = self.sat_head(sat_input)
        return F.normalize(embedding, dim=1)

    def encode_relation(self, z_uav, z_sat, logits):
        if self.qah_relation_head is None:
            raise RuntimeError(
                "QAH relation head is disabled; "
                "set USE_QAH_MS_RELATION=True"
            )
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
        return relation.reshape(
            pair.shape[0],
            pair.shape[1],
            -1,
        )


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
    """T2-only residual second-order CRF over consecutive retrieval lattices.

    The learned first-order T1 factor is intentionally removed in this
    ablation.  A window is initialized only after three frames are available:
    E1 + E2 + E3 + T2(1,2,3).  The oldest state is then marginalized with
    LogSumExp so that the carried dynamic-programming state remains a pair of
    candidate indices, which is sufficient for a second-order Markov model.

    For longer windows, every additional frame contributes one overlapping T2
    factor.  All spatial offsets, velocities, accelerations, and HardMS
    disagreement use raw meter-based values; no POSITION_SCALE_M is used.
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
        self.second_transition = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # T2 starts from a neutral learned residual while the positive
        # acceleration coefficient supplies the initial second-order preference.
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
        # RAW METERS: no /10 or any other position scaling.
        relative = centers - lattice_center
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
        # RAW METER-BASED FEATURES. Cosine is already dimensionless.
        feature = torch.cat(
            [
                v01,
                v12,
                acceleration,
                speed01,
                speed12,
                cosine,
            ],
            dim=-1,
        )
        learned = self.second_transition(feature).squeeze(-1)
        # RAW acceleration energy; trainable positive coefficient must learn the
        # appropriate strength directly in meter-based units.
        acceleration_energy = acceleration.square().sum(dim=-1)
        return learned - F.softplus(self.acceleration_weight_raw) * acceleration_energy

    @staticmethod
    def _gather_path_score(
        emission: torch.Tensor,
        second_scores: List[torch.Tensor],
        target_index: torch.Tensor,
    ):
        """Score the supervised candidate path using emissions + T2 only."""
        batch = torch.arange(emission.shape[0], device=emission.device)
        score = torch.zeros(emission.shape[0], device=emission.device)

        for time in range(emission.shape[1]):
            score = score + emission[batch, time, target_index[:, time]]

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
        frame_count = int(centers.shape[1])
        if frame_count < 3:
            raise ValueError(
                "T2-only TemporalLatticeCRF requires at least three frames"
            )

        emission, _ = self._emissions(
            z_uav, z_sat, raw_logits, raw_prob, centers
        )
        dt = (frame_ids[:, 1:] - frame_ids[:, :-1]).float().clamp_min(1.0)

        # ------------------------------------------------------------------
        # T2-only initialization with Frames 1, 2, 3.
        #
        # initial_score[j1,j2,j3]
        #   = E1[j1] + E2[j2] + E3[j3] + T2[j1,j2,j3]
        #
        # This is the replacement for the old T1-based two-frame alpha.
        # We immediately marginalize j1 because a second-order model only
        # needs the most recent two states to process the next frame.
        # ------------------------------------------------------------------
        first_t2 = self._second_transition_score(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            dt[:, 0],
            dt[:, 1],
        )
        second_scores: List[torch.Tensor] = [first_t2]

        initial_score = (
            emission[:, 0, :, None, None]
            + emission[:, 1, None, :, None]
            + emission[:, 2, None, None, :]
            + first_t2
        )

        # alpha now represents all histories ending at (j2, j3).
        alpha = torch.logsumexp(initial_score, dim=1)

        # Frames 4...T: add one overlapping T2 factor and current emission,
        # then marginalize the oldest of the three active candidate indices.
        for time in range(3, frame_count):
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

        # alpha is [B, 36, 36] over the final two frames.
        log_partition = torch.logsumexp(alpha.flatten(1), dim=1)
        last_log_probability = (
            torch.logsumexp(alpha, dim=1)
            - log_partition[:, None]
        )
        path_probability = last_log_probability.exp()
        path_expectation = (
            path_probability.unsqueeze(-1) * centers[:, -1]
        ).sum(dim=1)

        emission_probability = torch.softmax(emission[:, -1], dim=1)
        path_entropy = self._entropy(path_probability)
        emission_entropy = self._entropy(emission_probability)
        path_top = path_probability.topk(k=2, dim=1).values
        emission_top = emission_probability.topk(k=2, dim=1).values

        # RAW METERS: no /10 on HardMS/path disagreement.
        disagreement = torch.linalg.norm(
            path_expectation - hardms_xy[:, -1], dim=1
        )
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
                emission, second_scores, target_index
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
