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
class TemporalOutput:
    hidden: torch.Tensor
    delta_local: torch.Tensor
    motion_logvar: torch.Tensor
    motion_center: torch.Tensor
    posterior_logits: torch.Tensor
    posterior_prob: torch.Tensor
    visual_expectation: torch.Tensor
    final_xy: torch.Tensor
    correction_gate: torch.Tensor
    entropy: torch.Tensor
    spatial_std: torch.Tensor


class TemporalMotionRetriever(nn.Module):
    """Temporal Motion-Conditioned Retrieval (TMCR).

    1. Frozen spatial feature correlation estimates frame-to-frame motion.
    2. A causal GRU accumulates the last frames into a temporal state.
    3. The state cross-attends to all SAT candidates and re-ranks the frozen
       retrieval logits.
    4. A learned uncertainty gate combines motion prediction and the decoded
       visual posterior.  There are no hand-written accept/reject thresholds.
    """

    def __init__(self, spatial_channels):
        super().__init__()
        spatial_size = int(config.MOTION_SPATIAL_SIZE)
        correlation_dim = spatial_size**4

        self.spatial_projection = nn.Sequential(
            nn.Conv2d(
                int(spatial_channels),
                int(config.MOTION_CORR_CHANNELS),
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, int(config.MOTION_CORR_CHANNELS)),
            nn.GELU(),
        )
        self.correlation_encoder = nn.Sequential(
            nn.Linear(correlation_dim, 256),
            nn.GELU(),
            nn.Dropout(config.TEMPORAL_DROPOUT),
            nn.Linear(256, config.MOTION_FEATURE_DIM),
        )
        self.global_pair_encoder = nn.Sequential(
            nn.Linear(config.CLIP_DIM * 4, 512),
            nn.GELU(),
            nn.Dropout(config.TEMPORAL_DROPOUT),
            nn.Linear(512, config.GLOBAL_PAIR_DIM),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        gru_input = config.MOTION_FEATURE_DIM + config.GLOBAL_PAIR_DIM + 32
        self.temporal_gru = nn.GRUCell(gru_input, config.TEMPORAL_HIDDEN)

        self.motion_head = nn.Sequential(
            nn.Linear(config.TEMPORAL_HIDDEN, config.TEMPORAL_HIDDEN),
            nn.GELU(),
            nn.Dropout(config.TEMPORAL_DROPOUT),
            nn.Linear(config.TEMPORAL_HIDDEN, 4),
        )
        # Zero residual initialisation makes the untrained architecture exactly
        # a straight-line predictor. Training learns only the image-conditioned
        # forward/lateral residual and its uncertainty.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        with torch.no_grad():
            self.motion_head[-1].bias[2:].fill_(math.log(4.0**2))

        self.uav_query = nn.Sequential(
            nn.Linear(config.TEMPORAL_HIDDEN + config.EMBED_DIM, 512),
            nn.GELU(),
            nn.Linear(512, config.CANDIDATE_TOKEN_DIM),
        )
        self.sat_key = nn.Linear(
            config.EMBED_DIM, config.CANDIDATE_TOKEN_DIM, bias=False
        )
        self.coord_key = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, config.CANDIDATE_TOKEN_DIM),
        )
        self.visual_key = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, config.CANDIDATE_TOKEN_DIM),
        )
        self.visual_logit_weight = nn.Parameter(torch.tensor(0.0))

        self.correction_gate = nn.Sequential(
            nn.Linear(config.TEMPORAL_HIDDEN + 5, 128),
            nn.GELU(),
            nn.Dropout(config.TEMPORAL_DROPOUT),
            nn.Linear(128, 1),
        )

    @staticmethod
    def initial_hidden(batch_size, device, dtype=torch.float32):
        return torch.zeros(
            batch_size, config.TEMPORAL_HIDDEN, device=device, dtype=dtype
        )

    def encode_motion(
        self,
        previous_global,
        current_global,
        previous_spatial,
        current_spatial,
        previous_local_delta,
        previous_speed,
        dt,
        hidden,
    ):
        if bool(config.USE_SPATIAL_CORRELATION):
            previous_map = self.spatial_projection(previous_spatial.float())
            current_map = self.spatial_projection(current_spatial.float())
            previous_map = F.normalize(previous_map.flatten(2), dim=1)
            current_map = F.normalize(current_map.flatten(2), dim=1)
            correlation = torch.einsum("bcn,bcm->bnm", previous_map, current_map)
            correlation_feature = self.correlation_encoder(
                correlation.flatten(1)
            )
        else:
            correlation_feature = torch.zeros(
                previous_global.shape[0],
                config.MOTION_FEATURE_DIM,
                device=previous_global.device,
                dtype=previous_global.dtype,
            )

        pair = torch.cat(
            [
                previous_global,
                current_global,
                current_global - previous_global,
                current_global * previous_global,
            ],
            dim=1,
        )
        global_feature = self.global_pair_encoder(pair.float())

        state_input = torch.cat(
            [
                previous_local_delta / float(config.POSITION_SCALE_M),
                previous_speed.unsqueeze(1) / float(config.POSITION_SCALE_M),
                dt.unsqueeze(1) / float(config.DT_SCALE),
            ],
            dim=1,
        )
        state_feature = self.state_encoder(state_input)
        hidden = self.temporal_gru(
            torch.cat(
                [correlation_feature, global_feature, state_feature], dim=1
            ),
            hidden,
        )

        motion = self.motion_head(hidden)
        straight_line_delta = torch.stack(
            [previous_speed * dt, torch.zeros_like(previous_speed)], dim=1
        )
        delta_local = straight_line_delta + motion[:, :2]
        motion_logvar = motion[:, 2:].clamp(
            min=float(config.MOTION_LOGVAR_MIN),
            max=float(config.MOTION_LOGVAR_MAX),
        )
        return hidden, delta_local, motion_logvar

    def decode_candidates(
        self,
        hidden,
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        candidate_xy,
        motion_center,
        forward_axis,
        lateral_axis,
        motion_logvar,
    ):
        relative = candidate_xy - motion_center[:, None, :]
        forward = (relative * forward_axis[:, None, :]).sum(dim=2)
        lateral = (relative * lateral_axis[:, None, :]).sum(dim=2)
        radius = torch.sqrt(forward.square() + lateral.square() + 1e-8)
        coord = torch.stack([forward, lateral, radius], dim=2) / float(
            config.POSITION_SCALE_M
        )

        raw_mean = raw_logits.mean(dim=1, keepdim=True)
        raw_std = raw_logits.std(dim=1, keepdim=True).clamp_min(1e-5)
        standard_logits = (raw_logits - raw_mean) / raw_std
        visual_statistics = torch.stack([standard_logits, raw_prob], dim=2)

        token = (
            self.sat_key(z_sat)
            + self.coord_key(coord)
            + self.visual_key(visual_statistics)
        )
        query = self.uav_query(torch.cat([hidden, z_uav], dim=1))
        attention = (
            token * query[:, None, :]
        ).sum(dim=2) / math.sqrt(float(config.CANDIDATE_TOKEN_DIM))
        posterior_logits = (
            attention + self.visual_logit_weight.exp() * standard_logits
            if bool(config.USE_TEMPORAL_CANDIDATE_DECODER)
            else standard_logits
        )
        posterior_prob = torch.softmax(
            posterior_logits / float(config.TEMPORAL_TEMPERATURE), dim=1
        )
        visual_expectation = (
            posterior_prob.unsqueeze(2) * candidate_xy
        ).sum(dim=1)

        entropy = -(
            posterior_prob
            * posterior_prob.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(posterior_prob.shape[1]))
        centered = candidate_xy - visual_expectation[:, None, :]
        spatial_variance = (
            posterior_prob.unsqueeze(2) * centered.square()
        ).sum(dim=1)
        spatial_std = torch.sqrt(spatial_variance.sum(dim=1).clamp_min(1e-8))
        top_values = posterior_prob.topk(k=2, dim=1).values
        posterior_margin = top_values[:, 0] - top_values[:, 1]
        motion_std = torch.exp(0.5 * motion_logvar).mean(dim=1)

        gate_input = torch.cat(
            [
                hidden,
                entropy.unsqueeze(1),
                spatial_std.unsqueeze(1) / float(config.POSITION_SCALE_M),
                posterior_margin.unsqueeze(1),
                motion_std.unsqueeze(1) / float(config.POSITION_SCALE_M),
                raw_prob.max(dim=1).values.unsqueeze(1),
            ],
            dim=1,
        )
        if bool(config.USE_LEARNED_UNCERTAINTY_GATE):
            correction_gate = torch.sigmoid(self.correction_gate(gate_input))
        else:
            correction_gate = torch.ones(
                motion_center.shape[0], 1,
                device=motion_center.device, dtype=motion_center.dtype
            )
        final_xy = motion_center + correction_gate * (
            visual_expectation - motion_center
        )

        return (
            posterior_logits,
            posterior_prob,
            visual_expectation,
            final_xy,
            correction_gate,
            entropy,
            spatial_std,
        )

    def forward_step(
        self,
        previous_global,
        current_global,
        previous_spatial,
        current_spatial,
        previous_local_delta,
        previous_speed,
        dt,
        hidden,
        previous_xy,
        forward_axis,
        lateral_axis,
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        candidate_xy,
    ):
        hidden, delta_local, motion_logvar = self.encode_motion(
            previous_global,
            current_global,
            previous_spatial,
            current_spatial,
            previous_local_delta,
            previous_speed,
            dt,
            hidden,
        )
        delta_map = (
            delta_local[:, :1] * forward_axis
            + delta_local[:, 1:2] * lateral_axis
        )
        motion_center = previous_xy + delta_map
        decoded = self.decode_candidates(
            hidden,
            z_uav,
            z_sat,
            raw_logits,
            raw_prob,
            candidate_xy,
            motion_center,
            forward_axis,
            lateral_axis,
            motion_logvar,
        )
        return TemporalOutput(
            hidden=hidden,
            delta_local=delta_local,
            motion_logvar=motion_logvar,
            motion_center=motion_center,
            posterior_logits=decoded[0],
            posterior_prob=decoded[1],
            visual_expectation=decoded[2],
            final_xy=decoded[3],
            correction_gate=decoded[4],
            entropy=decoded[5],
            spatial_std=decoded[6],
        )