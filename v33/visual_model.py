import math
from dataclasses import dataclass
from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


class TorchvisionImageEncoder(nn.Module):
    """Frozen ImageNet backbone exposing the encode_image API used by v33."""

    def __init__(self, key):
        super().__init__()
        if key == "resnet18":
            network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            network.fc = nn.Identity()
        elif key == "resnet50":
            network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            network.fc = nn.Identity()
        elif key == "mobilenet_v3_small":
            network = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            network.classifier = nn.Identity()
        elif key == "vgg16":
            network = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            network.classifier = nn.Sequential(*list(network.classifier.children())[:-1])
        else:
            raise ValueError("unsupported torchvision backbone: %s" % key)
        self.network = network
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def encode_image(self, image):
        image = F.interpolate(
            image.float(), size=(224, 224), mode="bilinear", align_corners=False
        )
        return self.network((image - self.mean) / self.std)


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
    """Checkpoint-compatible visual retrieval model."""

    def __init__(self):
        super().__init__()
        if str(config.BACKBONE_NAME).startswith("torchvision:"):
            self.clip = TorchvisionImageEncoder(config.BACKBONE_KEY)
        else:
            self.clip, _ = open_clip.create_model_from_pretrained(config.BACKBONE_NAME)
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.use_coord_encoder = bool(getattr(config, "USE_COORD_ENCODER", False))
        head_dim = int(getattr(config, "BACKBONE_HEAD_DIM", config.CLIP_DIM))
        self.uav_head = nn.Sequential(
            nn.Linear(config.CLIP_DIM, head_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_dim, config.EMBED_DIM),
        )

        if self.use_coord_encoder:
            self.coord_encoder = FourierCoordEncoder()
            sat_in = config.CLIP_DIM + config.EMBED_DIM
        else:
            self.coord_encoder = None
            sat_in = config.CLIP_DIM

        self.sat_head = nn.Sequential(
            nn.Linear(sat_in, head_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_dim, config.EMBED_DIM),
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


@dataclass
class RouteProgressGRUOutput:
    measurement_se: torch.Tensor
    correction_se: torch.Tensor
    measurement_variance_se: torch.Tensor
    velocity_se: torch.Tensor
    acceleration_se: torch.Tensor
    next_step_se: torch.Tensor
    heading_residual_rad: torch.Tensor
    turn_rate_rad: torch.Tensor
    hidden: torch.Tensor
    state: torch.Tensor


class ThreeFrameRouteStateGRU(nn.Module):
    """Three-frame recurrent route-state model with learned acquisition scoring.

    Short-term motion evidence comes from three UAV embeddings. Long-term state
    is carried by the GRU hidden state. A separate recurrent acquisition scorer
    ranks multiple *local* SAT windows before the main motion/measurement update.
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
        self.clip_mean_projection = projection(config.EMBED_DIM)
        self.delta_recent_projection = projection(config.EMBED_DIM)
        self.delta_accel_projection = projection(config.EMBED_DIM)
        self.sat_projection = projection(config.EMBED_DIM)
        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        def head(out_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        self.motion_head = head(4)
        self.heading_head = head(2)
        self.correction_head = head(2)
        self.variance_head = head(2)

        # Local posterior / recurrent acquisition scorer. It sees current 3-frame motion,
        # the previous recurrent state, each local SAT context and local score
        # quality. It does NOT search the whole route as one flat global gallery.
        self.acq_hidden_projection = projection(hidden_dim)
        self.acq_numeric_projection = nn.Sequential(
            nn.Linear(int(config.ACQ_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.acq_scorer = nn.Sequential(
            nn.Linear(feature_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Avoid a large arbitrary initial speed. A small positive bias makes the
        # state numerically well behaved while acquisition provides the initial
        # displacement evidence.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        init_speed = 0.75
        self.motion_head[-1].bias.data[0] = math.log(math.exp(init_speed) - 1.0)

        nn.init.zeros_(self.heading_head[-1].weight)
        nn.init.zeros_(self.heading_head[-1].bias)

        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.variance_head[-1].weight)
        initial_extra_var = 1.0
        nn.init.constant_(
            self.variance_head[-1].bias,
            math.log(math.exp(initial_extra_var) - 1.0),
        )

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size), int(config.RNN_HIDDEN_DIM), device=device, dtype=dtype
        )

    @staticmethod
    def _safe_margin(probability):
        if probability.shape[1] < 2:
            return torch.ones(probability.shape[0], 1, device=probability.device)
        top2 = torch.topk(probability, k=2, dim=1).values
        return top2[:, 0:1] - top2[:, 1:2]

    def _three_frame_features(self, z_uav, previous_z_uav, previous2_z_uav):
        if previous_z_uav is None:
            previous_z_uav = z_uav
        if previous2_z_uav is None:
            previous2_z_uav = previous_z_uav
        delta_old = previous_z_uav - previous2_z_uav
        delta_recent = z_uav - previous_z_uav
        delta_accel = delta_recent - delta_old
        clip_mean = (previous2_z_uav + previous_z_uav + z_uav) / 3.0
        return previous_z_uav, previous2_z_uav, delta_recent, delta_accel, clip_mean

    def score_hypotheses(
        self,
        z_uav,
        previous_z_uav,
        previous2_z_uav,
        sat_context,
        entropy,
        margin,
        response_variance_se,
        hypothesis_anchor_se,
        predicted_se,
        top1_distance_m,
        hardms_support,
        hidden: Optional[torch.Tensor] = None,
    ):
        """Return one learned score per local acquisition hypothesis."""
        if z_uav.shape[0] != 1:
            raise ValueError("score_hypotheses expects one UAV frame")
        if hidden is None:
            hidden = self.initial_hidden(1, z_uav.device, z_uav.dtype)

        _, _, delta_recent, delta_accel, _ = self._three_frame_features(
            z_uav, previous_z_uav, previous2_z_uav
        )
        count = int(sat_context.shape[0])
        uav_h = self.uav_projection(z_uav).expand(count, -1)
        recent_h = self.delta_recent_projection(delta_recent).expand(count, -1)
        accel_h = self.delta_accel_projection(delta_accel).expand(count, -1)
        sat_h = self.sat_projection(sat_context)
        hidden_h = self.acq_hidden_projection(hidden).expand(count, -1)

        offset = hypothesis_anchor_se - predicted_se.expand(count, -1)
        numeric = torch.cat(
            [
                entropy.reshape(-1, 1),
                margin.reshape(-1, 1),
                torch.log1p(response_variance_se.clamp_min(0.0)) / 7.0,
                offset[:, 0:1] / float(config.ROUTE_PROGRESS_SCALE_M),
                offset[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),
                top1_distance_m.reshape(-1, 1) / 30.0,
                hardms_support.reshape(-1, 1),
            ],
            dim=1,
        )
        if int(numeric.shape[1]) != int(config.ACQ_NUMERIC_DIM):
            raise RuntimeError(
                "Acquisition numeric dimension mismatch: got %d expected %d"
                % (int(numeric.shape[1]), int(config.ACQ_NUMERIC_DIM))
            )
        numeric_h = self.acq_numeric_projection(numeric)
        features = torch.cat(
            [uav_h, recent_h, accel_h, sat_h, hidden_h, numeric_h], dim=1
        )
        return self.acq_scorer(features).reshape(-1)

    def forward_step(
        self,
        z_uav,
        previous_z_uav,
        previous2_z_uav,
        sat_context,
        posterior_probability,
        visual_anchor_se,
        response_variance_se,
        predicted_se,
        previous_measurement_se,
        previous_velocity_se,
        previous_acceleration_se,
        previous_heading_state,
        polynomial_step_se,
        route_remaining_m,
        predicted_cross_m,
        total_progress_fraction,
        leg_progress_fraction,
        top1_distance_m,
        hardms_support,
        hidden: Optional[torch.Tensor] = None,
    ):
        if hidden is None:
            hidden = self.initial_hidden(z_uav.shape[0], z_uav.device, z_uav.dtype)
        previous_z_uav, previous2_z_uav, delta_recent, delta_accel, clip_mean = (
            self._three_frame_features(z_uav, previous_z_uav, previous2_z_uav)
        )
        if previous_measurement_se is None:
            previous_measurement_se = visual_anchor_se.detach()

        p = posterior_probability.clamp_min(float(config.ACQ_POSTERIOR_EPS))
        entropy = -(p * p.log()).sum(dim=1, keepdim=True)
        entropy = entropy / max(math.log(max(2, p.shape[1])), 1e-6)
        margin = self._safe_margin(p)

        innovation_se = visual_anchor_se - predicted_se
        visual_step_se = visual_anchor_se - previous_measurement_se

        numeric = torch.cat(
            [
                entropy,
                margin,
                torch.log1p(response_variance_se.clamp_min(0.0)) / 7.0,
                torch.cat(
                    [
                        innovation_se[:, 0:1] / float(config.ROUTE_STEP_SCALE_M),
                        innovation_se[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        visual_step_se[:, 0:1] / float(config.ROUTE_STEP_SCALE_M),
                        visual_step_se[:, 1:2] / float(config.ROUTE_CROSS_TRACK_SCALE_M),
                    ],
                    dim=1,
                ),
                previous_velocity_se / float(config.ROUTE_STEP_SCALE_M),
                previous_acceleration_se / float(config.ROUTE_STEP_SCALE_M),
                previous_heading_state[:, 0:1] / math.radians(float(config.MAX_HEADING_RESIDUAL_DEG)),
                previous_heading_state[:, 1:2] / math.radians(float(config.MAX_TURN_RATE_DEG_PER_FRAME)),
                route_remaining_m / float(config.ROUTE_REMAINING_SCALE_M),
                predicted_cross_m / float(config.ROUTE_CROSS_TRACK_SCALE_M),
                total_progress_fraction,
                leg_progress_fraction,
                polynomial_step_se / float(config.ROUTE_STEP_SCALE_M),
                top1_distance_m / 30.0,
                hardms_support.reshape(-1, 1),
            ],
            dim=1,
        )
        if int(numeric.shape[1]) != int(config.RNN_NUMERIC_DIM):
            raise RuntimeError(
                "RNN numeric dimension mismatch: got %d expected %d"
                % (int(numeric.shape[1]), int(config.RNN_NUMERIC_DIM))
            )

        recurrent_input = torch.cat(
            [
                self.uav_projection(z_uav),
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.numeric_projection(numeric),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        h = self.dropout(new_hidden)

        raw_motion = self.motion_head(h)
        v_parallel = F.softplus(raw_motion[:, 0:1]).clamp(
            max=float(config.MAX_FORWARD_SPEED_M_PER_FRAME)
        )
        v_cross = torch.tanh(raw_motion[:, 1:2]) * float(
            config.MAX_CROSS_SPEED_M_PER_FRAME
        )
        a_parallel = torch.tanh(raw_motion[:, 2:3]) * float(
            config.MAX_FORWARD_ACCEL_M_PER_FRAME2
        )
        a_cross = torch.tanh(raw_motion[:, 3:4]) * float(
            config.MAX_CROSS_ACCEL_M_PER_FRAME2
        )
        velocity = torch.cat([v_parallel, v_cross], dim=1)
        acceleration = torch.cat([a_parallel, a_cross], dim=1)

        # v31 uses a strictly causal direction state. The raw heading/turn
        # predictions are rate-limited against the previous recurrent heading
        # state *before* they are allowed to rotate the polynomial. Turn-rate is
        # a state describing rotation already observed in the recent frames; it
        # is NOT added as a look-ahead term, so the estimator cannot pre-turn.
        raw_heading = self.heading_head(h)
        raw_heading_residual = torch.tanh(raw_heading[:, 0:1]) * math.radians(
            float(config.MAX_HEADING_RESIDUAL_DEG)
        )
        raw_turn_rate = torch.tanh(raw_heading[:, 1:2]) * math.radians(
            float(config.MAX_TURN_RATE_DEG_PER_FRAME)
        )
        prev_heading = previous_heading_state[:, 0:1]
        prev_turn = previous_heading_state[:, 1:2]
        heading_delta = torch.atan2(
            torch.sin(raw_heading_residual - prev_heading),
            torch.cos(raw_heading_residual - prev_heading),
        ).clamp(
            min=-math.radians(float(config.MAX_HEADING_DELTA_DEG_PER_FRAME)),
            max= math.radians(float(config.MAX_HEADING_DELTA_DEG_PER_FRAME)),
        )
        turn_delta = (raw_turn_rate - prev_turn).clamp(
            min=-math.radians(float(config.MAX_TURN_RATE_DELTA_DEG_PER_FRAME2)),
            max= math.radians(float(config.MAX_TURN_RATE_DELTA_DEG_PER_FRAME2)),
        )
        heading_residual = prev_heading + float(config.HEADING_STATE_EMA_ALPHA) * heading_delta
        turn_rate = prev_turn + float(config.TURN_RATE_EMA_ALPHA) * turn_delta

        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
        effective_heading = heading_residual
        cos_h = torch.cos(effective_heading)
        sin_h = torch.sin(effective_heading)
        next_parallel = base_forward * cos_h - base_cross * sin_h
        next_cross = base_forward * sin_h + base_cross * cos_h
        next_parallel = next_parallel.clamp(min=0.0)
        next_step = torch.cat([next_parallel, next_cross], dim=1)
        norm = torch.linalg.norm(next_step, dim=1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME) / norm, max=1.0
        )
        next_step = next_step * scale

        raw_correction = torch.tanh(self.correction_head(h))
        correction = torch.cat(
            [
                raw_correction[:, 0:1]
                * float(config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M),
                raw_correction[:, 1:2]
                * float(config.MAX_MEASUREMENT_CORRECTION_CROSS_M),
            ],
            dim=1,
        )
        measurement = visual_anchor_se + correction

        extra_variance = F.softplus(self.variance_head(h))
        measurement_variance = (
            response_variance_se.clamp_min(float(config.KALMAN_R_MIN_VAR))
            + extra_variance
        ).clamp(
            min=float(config.KALMAN_R_MIN_VAR),
            max=float(config.KALMAN_R_MAX_VAR),
        )

        state = torch.cat(
            [velocity, acceleration, heading_residual, turn_rate, correction, measurement_variance], dim=1
        )
        return RouteProgressGRUOutput(
            measurement_se=measurement,
            correction_se=correction,
            measurement_variance_se=measurement_variance,
            velocity_se=velocity,
            acceleration_se=acceleration,
            next_step_se=next_step,
            heading_residual_rad=heading_residual,
            turn_rate_rad=turn_rate,
            hidden=new_hidden,
            state=state,
        )


# Compatibility aliases for the existing pipeline.
RouteProgressGRU = ThreeFrameRouteStateGRU
WaypointLocalPrimaryRecoveryGRU = ThreeFrameRouteStateGRU
WaypointRouteGlobalRecoveryGRU = ThreeFrameRouteStateGRU
WaypointTemporalMotionGRU = ThreeFrameRouteStateGRU
WaypointConditionedGRU = ThreeFrameRouteStateGRU
