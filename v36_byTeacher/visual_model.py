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
    """Frozen ImageNet backbone exposing the encode_image API."""

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
    """Frozen image backbone with trainable UAV/SAT projection heads."""

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


def _wrap_angle_tensor(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rotate_local_to_global(local_xy, heading_rad):
    c = torch.cos(heading_rad)
    s = torch.sin(heading_rad)
    x = local_xy[:, 0:1]
    y = local_xy[:, 1:2]
    return torch.cat([c * x - s * y, s * x + c * y], dim=1)


def _signed_log1p(value):
    return torch.sign(value) * torch.log1p(torch.abs(value))


@dataclass
class MotionGRUOutput:
    velocity_local: torch.Tensor
    acceleration_local: torch.Tensor
    velocity_xy: torch.Tensor
    acceleration_xy: torch.Tensor
    next_step_xy: torch.Tensor
    heading_rad: torch.Tensor
    turn_rate_rad: torch.Tensor
    hidden: torch.Tensor
    state: torch.Tensor

    @property
    def velocity_se(self):
        return self.velocity_xy

    @property
    def acceleration_se(self):
        return self.acceleration_xy

    @property
    def next_step_se(self):
        return self.next_step_xy

    @property
    def heading_residual_rad(self):
        return self.heading_rad


class TwoFrameRouteStateGRU(nn.Module):
    """Causal motion predictor for the rewritten v36_byTeacher.

    The recurrent inputs deliberately follow the current byTeacher temporal
    design as closely as possible while removing Mean-Shift uncertainty:

      * current/previous UAV embedding -> temporal mean and first difference;
      * current MS1 coordinate relative to previous MS1 coordinate;
      * previous predicted velocity;
      * previous predicted heading + turn-rate;
      * previous GRU hidden state.

    There is no response-variance input, no current reference/GT coordinate, and
    no current Kalman prior in the GRU input.  The output heads predict v/a and
    heading/turn.  No hand-coded speed, acceleration, heading, turn-rate, EMA,
    or per-frame step limit is applied.
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

        self.clip_mean_projection = projection(config.EMBED_DIM)
        self.delta_recent_projection = projection(config.EMBED_DIM)
        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.gru = nn.GRUCell(feature_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        def head(out_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        self.motion_head = head(4)  # v_forward, v_cross, a_forward, a_cross
        self.heading_head = head(2)  # heading_delta, turn_rate

        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        nn.init.zeros_(self.heading_head[-1].weight)
        nn.init.zeros_(self.heading_head[-1].bias)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size), int(config.RNN_HIDDEN_DIM), device=device, dtype=dtype
        )

    def _two_frame_features(self, z_uav, previous_z_uav):
        frame_count = int(getattr(config, "EXPERIMENT_FRAME_COUNT", 2))
        if frame_count == 1:
            return torch.zeros_like(z_uav), z_uav
        if frame_count != 2:
            raise RuntimeError("v36_byTeacher supports only 1-frame or 2-frame input")
        if previous_z_uav is None:
            previous_z_uav = z_uav
        clip_mean = (previous_z_uav + z_uav) / 2.0
        delta_recent = z_uav - previous_z_uav
        return delta_recent, clip_mean

    def forward_step(
        self,
        z_uav,
        previous_z_uav,
        ms1_xy,
        previous_ms1_xy,
        previous_velocity_xy,
        previous_heading_state,
        hidden: Optional[torch.Tensor] = None,
    ):
        if hidden is None:
            hidden = self.initial_hidden(z_uav.shape[0], z_uav.device, z_uav.dtype)

        delta_recent, clip_mean = self._two_frame_features(z_uav, previous_z_uav)
        if previous_ms1_xy is None:
            previous_ms1_xy = ms1_xy.detach()

        # Same position-domain temporal cue as the current byTeacher idea:
        # current visual localization relative to the previous visual localization.
        visual_step_xy = ms1_xy - previous_ms1_xy

        numeric = torch.cat(
            [
                _signed_log1p(visual_step_xy),
                _signed_log1p(previous_velocity_xy),
                previous_heading_state,
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
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.numeric_projection(numeric),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        h = self.dropout(new_hidden)

        raw_motion = self.motion_head(h)
        velocity_local = raw_motion[:, 0:2]
        acceleration_local = raw_motion[:, 2:4]

        raw_heading = self.heading_head(h)
        previous_heading = previous_heading_state[:, 0:1]
        heading = _wrap_angle_tensor(previous_heading + raw_heading[:, 0:1])
        turn_rate = raw_heading[:, 1:2]

        velocity_xy = _rotate_local_to_global(velocity_local, heading)
        acceleration_xy = _rotate_local_to_global(acceleration_local, heading)

        # dt = 1 frame. No clipping, min/max speed, or turn limit is applied.
        next_step_local = velocity_local + 0.5 * acceleration_local
        next_step_xy = _rotate_local_to_global(next_step_local, heading)

        state = torch.cat(
            [
                velocity_local,
                acceleration_local,
                velocity_xy,
                acceleration_xy,
                next_step_xy,
                heading,
                turn_rate,
            ],
            dim=1,
        )
        return MotionGRUOutput(
            velocity_local=velocity_local,
            acceleration_local=acceleration_local,
            velocity_xy=velocity_xy,
            acceleration_xy=acceleration_xy,
            next_step_xy=next_step_xy,
            heading_rad=heading,
            turn_rate_rad=turn_rate,
            hidden=new_hidden,
            state=state,
        )


RouteProgressGRUOutput = MotionGRUOutput
ThreeFrameRouteStateGRU = TwoFrameRouteStateGRU
RouteProgressGRU = TwoFrameRouteStateGRU
WaypointLocalPrimaryRecoveryGRU = TwoFrameRouteStateGRU
WaypointRouteGlobalRecoveryGRU = TwoFrameRouteStateGRU
WaypointTemporalMotionGRU = TwoFrameRouteStateGRU
WaypointConditionedGRU = TwoFrameRouteStateGRU
