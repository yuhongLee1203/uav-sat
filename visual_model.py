import math
from dataclasses import dataclass
from typing import Optional

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
    """Visual retrieval model kept checkpoint-compatible with v20.

    MobileCLIP remains frozen. Only the UAV/SAT projection heads and logit scale
    are task-specific parameters, exactly as expected by visual_localizer.py.
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
            sat_in = config.CLIP_DIM + config.EMBED_DIM
        else:
            self.coord_encoder = None
            sat_in = config.CLIP_DIM

        self.sat_head = nn.Sequential(
            nn.Linear(sat_in, config.CLIP_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.CLIP_DIM, config.EMBED_DIM),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(image.float())

    @torch.no_grad()
    def encode_clip_spatial(self, image, output_size=None):
        output_size = int(output_size or config.MOTION_SPATIAL_SIZE)
        feature = self.encode_clip_image(image).unsqueeze(-1).unsqueeze(-1)
        return F.adaptive_avg_pool2d(
            feature.float(), (output_size, output_size)
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


@dataclass
class WaypointGRUOutput:
    measurement_xy: torch.Tensor
    measurement_anchor_xy: torch.Tensor
    correction_route: torch.Tensor
    measurement_variance_xy: torch.Tensor
    measurement_variance_route: torch.Tensor
    response_variance_route: torch.Tensor
    confidence: torch.Tensor
    velocity_route: torch.Tensor
    acceleration_route: torch.Tensor
    next_step_route: torch.Tensor
    velocity_xy: torch.Tensor
    acceleration_xy: torch.Tensor
    next_step_xy: torch.Tensor
    candidate_probability: torch.Tensor
    hidden: torch.Tensor
    state: torch.Tensor


class WaypointTemporalMotionGRU(nn.Module):
    """Two-frame waypoint-conditioned recurrent state estimator.

    Current and previous UAV embeddings are provided explicitly so the network
    can estimate motion from temporal visual change rather than from a single
    image. The network predicts signed route-frame velocity/acceleration and a
    visual measurement uncertainty. The external Kalman filter remains the
    only module that outputs the final navigation position.
    """

    def __init__(self):
        super().__init__()
        feature_dim = int(config.RNN_FEATURE_DIM)
        hidden_dim = int(config.RNN_HIDDEN_DIM)
        dropout = float(config.RNN_DROPOUT)

        def projection(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
            )

        self.uav_projection = projection(config.EMBED_DIM)
        self.delta_uav_projection = projection(config.EMBED_DIM)
        self.sat_projection = projection(config.EMBED_DIM)
        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        def head(output_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        self.motion_head = head(4)
        self.correction_head = head(2)
        self.variance_head = head(2)
        self.confidence_head = head(1)

        # Signed velocity is important: direction is predicted and supervised,
        # not hard-clamped. Initialize near the measured Route-A cruise speed.
        nn.init.zeros_(self.motion_head[-1].weight)
        target_fraction = 3.0 / float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        target_fraction = min(max(target_fraction, -0.95), 0.95)
        forward_bias = 0.5 * math.log((1.0 + target_fraction) / (1.0 - target_fraction))
        with torch.no_grad():
            self.motion_head[-1].bias.copy_(
                torch.tensor([forward_bias, 0.0, 0.0, 0.0])
            )

        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.variance_head[-1].weight)
        initial_var = 4.0
        inv_softplus = math.log(math.exp(initial_var) - 1.0)
        nn.init.constant_(self.variance_head[-1].bias, inv_softplus)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, -1.0)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size), int(config.RNN_HIDDEN_DIM), device=device, dtype=dtype
        )

    @staticmethod
    def _entropy(probability):
        count = max(int(probability.shape[1]), 2)
        return -(probability * probability.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(count))

    @staticmethod
    def _margin(probability):
        values = probability.topk(k=min(2, int(probability.shape[1])), dim=1).values
        if values.shape[1] == 1:
            return values[:, 0]
        return values[:, 0] - values[:, 1]

    @staticmethod
    def _project_route(vector_xy, route_unit, cross_unit):
        parallel = (vector_xy * route_unit).sum(dim=-1, keepdim=True)
        cross = (vector_xy * cross_unit).sum(dim=-1, keepdim=True)
        return torch.cat([parallel, cross], dim=-1)

    @staticmethod
    def _route_to_xy(vector_route, route_unit, cross_unit):
        return vector_route[:, 0:1] * route_unit + vector_route[:, 1:2] * cross_unit

    @staticmethod
    def _clip_norm(vector, maximum):
        norm = torch.linalg.norm(vector, dim=1, keepdim=True)
        scale = torch.clamp(float(maximum) / norm.clamp_min(1e-6), max=1.0)
        return vector * scale

    def _measurement_distribution(
        self, raw_probability, candidate_centers, search_center_xy, route_unit, cross_unit
    ):
        # Convert the spatial response into a RELATIVE visual innovation.
        # This is deliberately centered on the polynomial search position, not
        # on the quantized gallery grid. Therefore a flat/ambiguous 6x6 response
        # produces ~zero correction instead of pinning the filter to one anchor.
        temperature = max(float(config.TEMPORAL_MEASUREMENT_TAU), 1e-3)
        logits = raw_probability.clamp_min(1e-8).log() / temperature
        probability = torch.softmax(logits, dim=1)
        grid_reference = candidate_centers.mean(dim=1)
        candidate_offset = candidate_centers - grid_reference[:, None, :]
        weighted_offset = (probability.unsqueeze(-1) * candidate_offset).sum(dim=1)
        anchor = search_center_xy + weighted_offset

        centered_offset = candidate_offset - weighted_offset[:, None, :]
        parallel = (centered_offset * route_unit[:, None, :]).sum(dim=-1)
        cross = (centered_offset * cross_unit[:, None, :]).sum(dim=-1)
        response_var = torch.stack(
            [
                (probability * parallel.square()).sum(dim=1),
                (probability * cross.square()).sum(dim=1),
            ],
            dim=1,
        )
        return probability, anchor, response_var

    def _numeric_features(
        self,
        raw_probability,
        hardms_xy,
        raw_top1_xy,
        hardms_support,
        measurement_anchor_xy,
        search_center_xy,
        previous_final_xy,
        route_unit,
        cross_unit,
        route_remaining_m,
        route_cross_track_m,
        route_progress,
        previous_velocity_route,
        previous_acceleration_route,
        z_uav,
        previous_z_uav,
    ):
        hardms_innovation = self._project_route(
            hardms_xy - search_center_xy, route_unit, cross_unit
        ) / float(config.ROUTE_STEP_SCALE_M)
        anchor_innovation = self._project_route(
            measurement_anchor_xy - search_center_xy, route_unit, cross_unit
        ) / float(config.ROUTE_STEP_SCALE_M)
        top1_disagreement = torch.linalg.norm(
            hardms_xy - raw_top1_xy, dim=1, keepdim=True
        ) / float(config.ROUTE_STEP_SCALE_M)
        anchor_hardms_disagreement = torch.linalg.norm(
            measurement_anchor_xy - hardms_xy, dim=1, keepdim=True
        ) / float(config.ROUTE_STEP_SCALE_M)
        temporal_cosine = F.cosine_similarity(
            z_uav, previous_z_uav, dim=1
        ).unsqueeze(1)

        numeric = torch.cat(
            [
                self._entropy(raw_probability).unsqueeze(1),
                self._margin(raw_probability).unsqueeze(1),
                hardms_support.reshape(-1, 1),
                hardms_innovation,
                anchor_innovation,
                top1_disagreement,
                route_remaining_m / float(config.ROUTE_DISTANCE_SCALE_M),
                route_cross_track_m / float(config.ROUTE_CROSS_TRACK_SCALE_M),
                route_progress,
                previous_velocity_route[:, 0:1] / float(config.MAX_FORWARD_SPEED_M_PER_FRAME),
                previous_velocity_route[:, 1:2] / float(config.MAX_CROSS_SPEED_M_PER_FRAME),
                previous_acceleration_route[:, 0:1] / float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
                previous_acceleration_route[:, 1:2] / float(config.MAX_CROSS_ACCEL_M_PER_FRAME2),
                temporal_cosine,
                anchor_hardms_disagreement,
            ],
            dim=1,
        )
        if int(numeric.shape[1]) != int(config.RNN_NUMERIC_DIM):
            raise RuntimeError(
                "RNN numeric dimension mismatch: got %d expected %d"
                % (int(numeric.shape[1]), int(config.RNN_NUMERIC_DIM))
            )
        return numeric

    def forward_step(
        self,
        z_uav,
        previous_z_uav,
        sat_context,
        raw_probability,
        candidate_centers,
        hardms_xy,
        raw_top1_xy,
        hardms_support,
        search_center_xy,
        previous_final_xy,
        route_unit,
        cross_unit,
        route_remaining_m,
        route_cross_track_m,
        route_progress,
        previous_velocity_route,
        previous_acceleration_route,
        hidden=None,
    ):
        if hidden is None:
            hidden = self.initial_hidden(z_uav.shape[0], z_uav.device, z_uav.dtype)
        if previous_z_uav is None:
            previous_z_uav = z_uav

        candidate_probability, measurement_anchor_xy, response_var_route = self._measurement_distribution(
            raw_probability, candidate_centers, search_center_xy, route_unit, cross_unit
        )
        numeric = self._numeric_features(
            raw_probability=raw_probability,
            hardms_xy=hardms_xy,
            raw_top1_xy=raw_top1_xy,
            hardms_support=hardms_support,
            measurement_anchor_xy=measurement_anchor_xy,
            search_center_xy=search_center_xy,
            previous_final_xy=previous_final_xy,
            route_unit=route_unit,
            cross_unit=cross_unit,
            route_remaining_m=route_remaining_m,
            route_cross_track_m=route_cross_track_m,
            route_progress=route_progress,
            previous_velocity_route=previous_velocity_route,
            previous_acceleration_route=previous_acceleration_route,
            z_uav=z_uav,
            previous_z_uav=previous_z_uav,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.delta_uav_projection(z_uav - previous_z_uav),
                self.sat_projection(sat_context),
                self.numeric_projection(numeric),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        h = self.dropout(new_hidden)

        raw_motion = self.motion_head(h)
        velocity_parallel = torch.tanh(raw_motion[:, 0:1]) * float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        velocity_cross = torch.tanh(raw_motion[:, 1:2]) * float(config.MAX_CROSS_SPEED_M_PER_FRAME)
        acceleration_parallel = torch.tanh(raw_motion[:, 2:3]) * float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2)
        acceleration_cross = torch.tanh(raw_motion[:, 3:4]) * float(config.MAX_CROSS_ACCEL_M_PER_FRAME2)
        velocity_route = torch.cat([velocity_parallel, velocity_cross], dim=1)
        acceleration_route = torch.cat([acceleration_parallel, acceleration_cross], dim=1)
        next_step_route = self._clip_norm(
            velocity_route + 0.5 * acceleration_route,
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME),
        )

        raw_correction = torch.tanh(self.correction_head(h))
        correction_route = torch.cat(
            [
                raw_correction[:, 0:1] * float(config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M),
                raw_correction[:, 1:2] * float(config.MAX_MEASUREMENT_CORRECTION_CROSS_M),
            ],
            dim=1,
        )
        correction_xy = self._route_to_xy(correction_route, route_unit, cross_unit)
        measurement_xy = measurement_anchor_xy + correction_xy

        confidence = torch.sigmoid(self.confidence_head(h)).clamp(0.01, 0.995)
        learned_var = F.softplus(self.variance_head(h)) + float(config.KALMAN_R_MIN_VAR)
        # Confidence is squared so ambiguous measurements become genuinely weak
        # Kalman observations rather than repeatedly erasing the motion prior.
        variance_route = (learned_var + response_var_route) / confidence.square().clamp_min(1e-4)
        variance_route = variance_route.clamp(
            min=float(config.KALMAN_R_MIN_VAR),
            max=float(config.KALMAN_R_MAX_VAR),
        )
        var_x = route_unit[:, 0:1].square() * variance_route[:, 0:1] + cross_unit[:, 0:1].square() * variance_route[:, 1:2]
        var_y = route_unit[:, 1:2].square() * variance_route[:, 0:1] + cross_unit[:, 1:2].square() * variance_route[:, 1:2]
        variance_xy = torch.cat([var_x, var_y], dim=1).clamp(
            min=float(config.KALMAN_R_MIN_VAR),
            max=float(config.KALMAN_R_MAX_VAR),
        )

        velocity_xy = self._route_to_xy(velocity_route, route_unit, cross_unit)
        acceleration_xy = self._route_to_xy(acceleration_route, route_unit, cross_unit)
        next_step_xy = self._route_to_xy(next_step_route, route_unit, cross_unit)
        state = torch.cat(
            [velocity_route, acceleration_route, correction_route, confidence], dim=1
        )
        return WaypointGRUOutput(
            measurement_xy=measurement_xy,
            measurement_anchor_xy=measurement_anchor_xy,
            correction_route=correction_route,
            measurement_variance_xy=variance_xy,
            measurement_variance_route=variance_route,
            response_variance_route=response_var_route,
            confidence=confidence,
            velocity_route=velocity_route,
            acceleration_route=acceleration_route,
            next_step_route=next_step_route,
            velocity_xy=velocity_xy,
            acceleration_xy=acceleration_xy,
            next_step_xy=next_step_xy,
            candidate_probability=candidate_probability,
            hidden=new_hidden,
            state=state,
        )


# Backward-compatible alias inside the replacement package only.
WaypointConditionedGRU = WaypointTemporalMotionGRU
