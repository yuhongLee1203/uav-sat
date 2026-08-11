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
    correction_route: torch.Tensor
    measurement_variance_xy: torch.Tensor
    measurement_variance_route: torch.Tensor
    confidence: torch.Tensor
    velocity_route: torch.Tensor
    acceleration_route: torch.Tensor
    next_step_route: torch.Tensor
    velocity_xy: torch.Tensor
    acceleration_xy: torch.Tensor
    next_step_xy: torch.Tensor
    hidden: torch.Tensor
    state: torch.Tensor


class WaypointConditionedGRU(nn.Module):
    """Waypoint-conditioned recurrent visual state estimator.

    The GRU does not directly output the final navigation position. It estimates:
      1) a visual measurement residual around raw HardMS,
      2) visual measurement uncertainty/confidence,
      3) route-frame velocity and acceleration.

    The second-order polynomial p(t+1)=p(t)+v+0.5*a is then used as the next
    local SAT search prior, and an external Kalman filter produces final output.
    """

    def __init__(self):
        super().__init__()
        feature_dim = int(config.RNN_FEATURE_DIM)
        hidden_dim = int(config.RNN_HIDDEN_DIM)
        dropout = float(config.RNN_DROPOUT)

        self.uav_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.sat_projection = nn.Sequential(
            nn.Linear(config.EMBED_DIM, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.gru = nn.GRUCell(feature_dim * 3, hidden_dim)
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

        # Stable motion initialization: start with ~2.5 m/frame forward, zero
        # cross velocity/acceleration, then learn Route-A dynamics.
        nn.init.zeros_(self.motion_head[-1].weight)
        forward_fraction = 2.5 / float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        forward_fraction = min(max(forward_fraction, 1e-4), 1.0 - 1e-4)
        forward_bias = math.log(forward_fraction / (1.0 - forward_fraction))
        with torch.no_grad():
            self.motion_head[-1].bias.copy_(
                torch.tensor([forward_bias, 0.0, 0.0, 0.0])
            )

        # Start conservative: zero residual and moderate visual variance.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.variance_head[-1].weight)
        initial_var = 4.0
        inv_softplus = math.log(math.exp(initial_var) - 1.0)
        nn.init.constant_(self.variance_head[-1].bias, inv_softplus)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, -0.5)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size), int(config.RNN_HIDDEN_DIM), device=device, dtype=dtype
        )

    @staticmethod
    def _entropy(probability):
        count = max(int(probability.shape[1]), 2)
        return -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(count))

    @staticmethod
    def _margin(probability):
        values = probability.topk(
            k=min(2, int(probability.shape[1])), dim=1
        ).values
        if values.shape[1] == 1:
            return values[:, 0]
        return values[:, 0] - values[:, 1]

    @staticmethod
    def _project_route(vector_xy, route_unit, cross_unit):
        parallel = (vector_xy * route_unit).sum(dim=1, keepdim=True)
        cross = (vector_xy * cross_unit).sum(dim=1, keepdim=True)
        return torch.cat([parallel, cross], dim=1)

    @staticmethod
    def _route_to_xy(vector_route, route_unit, cross_unit):
        return (
            vector_route[:, 0:1] * route_unit
            + vector_route[:, 1:2] * cross_unit
        )

    @staticmethod
    def _clip_norm(vector, maximum):
        norm = torch.linalg.norm(vector, dim=1, keepdim=True)
        scale = torch.clamp(float(maximum) / norm.clamp_min(1e-6), max=1.0)
        return vector * scale

    def _numeric_features(
        self,
        raw_probability,
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
    ):
        hardms_innovation = self._project_route(
            hardms_xy - search_center_xy, route_unit, cross_unit
        ) / float(config.ROUTE_STEP_SCALE_M)
        hardms_step = self._project_route(
            hardms_xy - previous_final_xy, route_unit, cross_unit
        ) / float(config.ROUTE_STEP_SCALE_M)
        top1_disagreement = torch.linalg.norm(
            hardms_xy - raw_top1_xy, dim=1, keepdim=True
        ) / float(config.ROUTE_STEP_SCALE_M)

        remaining = route_remaining_m / float(config.ROUTE_DISTANCE_SCALE_M)
        cross_track = route_cross_track_m / float(config.ROUTE_CROSS_TRACK_SCALE_M)
        progress = route_progress

        numeric = torch.cat(
            [
                self._entropy(raw_probability).unsqueeze(1),
                self._margin(raw_probability).unsqueeze(1),
                hardms_support.reshape(-1, 1),
                hardms_innovation,
                hardms_step,
                remaining,
                cross_track,
                progress,
                previous_velocity_route[:, 0:1]
                / float(config.MAX_FORWARD_SPEED_M_PER_FRAME),
                previous_velocity_route[:, 1:2]
                / float(config.MAX_CROSS_SPEED_M_PER_FRAME),
                previous_acceleration_route[:, 0:1]
                / float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
                previous_acceleration_route[:, 1:2]
                / float(config.MAX_CROSS_ACCEL_M_PER_FRAME2),
                top1_disagreement,
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
        sat_context,
        raw_probability,
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
            hidden = self.initial_hidden(
                z_uav.shape[0], z_uav.device, z_uav.dtype
            )

        numeric = self._numeric_features(
            raw_probability=raw_probability,
            hardms_xy=hardms_xy,
            raw_top1_xy=raw_top1_xy,
            hardms_support=hardms_support,
            search_center_xy=search_center_xy,
            previous_final_xy=previous_final_xy,
            route_unit=route_unit,
            cross_unit=cross_unit,
            route_remaining_m=route_remaining_m,
            route_cross_track_m=route_cross_track_m,
            route_progress=route_progress,
            previous_velocity_route=previous_velocity_route,
            previous_acceleration_route=previous_acceleration_route,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.sat_projection(sat_context),
                self.numeric_projection(numeric),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        h = self.dropout(new_hidden)

        raw_motion = self.motion_head(h)
        # Route direction is known from current waypoint -> next waypoint.
        # Positive along-track velocity is therefore a structural navigation
        # prior, while cross-track velocity remains signed.
        velocity_parallel = torch.sigmoid(raw_motion[:, 0:1]) * float(
            config.MAX_FORWARD_SPEED_M_PER_FRAME
        )
        velocity_cross = torch.tanh(raw_motion[:, 1:2]) * float(
            config.MAX_CROSS_SPEED_M_PER_FRAME
        )
        acceleration_parallel = torch.tanh(raw_motion[:, 2:3]) * float(
            config.MAX_FORWARD_ACCEL_M_PER_FRAME2
        )
        acceleration_cross = torch.tanh(raw_motion[:, 3:4]) * float(
            config.MAX_CROSS_ACCEL_M_PER_FRAME2
        )
        velocity_route = torch.cat([velocity_parallel, velocity_cross], dim=1)
        acceleration_route = torch.cat(
            [acceleration_parallel, acceleration_cross], dim=1
        )
        next_step_route = velocity_route + 0.5 * acceleration_route
        next_step_route = self._clip_norm(
            next_step_route, float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )

        raw_correction = torch.tanh(self.correction_head(h))
        correction_route = torch.cat(
            [
                raw_correction[:, 0:1]
                * float(config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M),
                raw_correction[:, 1:2]
                * float(config.MAX_MEASUREMENT_CORRECTION_CROSS_M),
            ],
            dim=1,
        )
        correction_xy = self._route_to_xy(
            correction_route, route_unit, cross_unit
        )
        measurement_xy = hardms_xy + correction_xy

        confidence = torch.sigmoid(self.confidence_head(h)).clamp(0.05, 0.995)
        base_variance_route = F.softplus(self.variance_head(h)) + float(
            config.KALMAN_R_MIN_VAR
        )
        # A low-confidence visual observation automatically receives a larger R.
        variance_route = (base_variance_route / confidence).clamp(
            min=float(config.KALMAN_R_MIN_VAR),
            max=float(config.KALMAN_R_MAX_VAR),
        )
        # Rotate route-diagonal variance into XY and retain only the diagonal
        # because the external 2D Kalman update uses independent XY measurement R.
        var_x = (
            route_unit[:, 0:1].square() * variance_route[:, 0:1]
            + cross_unit[:, 0:1].square() * variance_route[:, 1:2]
        )
        var_y = (
            route_unit[:, 1:2].square() * variance_route[:, 0:1]
            + cross_unit[:, 1:2].square() * variance_route[:, 1:2]
        )
        variance_xy = torch.cat([var_x, var_y], dim=1).clamp(
            min=float(config.KALMAN_R_MIN_VAR),
            max=float(config.KALMAN_R_MAX_VAR),
        )

        velocity_xy = self._route_to_xy(
            velocity_route, route_unit, cross_unit
        )
        acceleration_xy = self._route_to_xy(
            acceleration_route, route_unit, cross_unit
        )
        next_step_xy = self._route_to_xy(
            next_step_route, route_unit, cross_unit
        )

        state = torch.cat(
            [
                velocity_route,
                acceleration_route,
                confidence,
                variance_route,
            ],
            dim=1,
        )

        return WaypointGRUOutput(
            measurement_xy=measurement_xy,
            correction_route=correction_route,
            measurement_variance_xy=variance_xy,
            measurement_variance_route=variance_route,
            confidence=confidence,
            velocity_route=velocity_route,
            acceleration_route=acceleration_route,
            next_step_route=next_step_route,
            velocity_xy=velocity_xy,
            acceleration_xy=acceleration_xy,
            next_step_xy=next_step_xy,
            hidden=new_hidden,
            state=state,
        )
