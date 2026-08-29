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
    """Checkpoint-compatible UAV/SAT retrieval model."""

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
class MotionGRUOutput:
    speed_m_per_frame: torch.Tensor
    acceleration_m_per_frame2: torch.Tensor
    heading_rad: torch.Tensor
    heading_delta_rad: torch.Tensor
    delta_xy: torch.Tensor
    hidden: torch.Tensor


class AutonomousMotionGRU(nn.Module):
    """Predict v, a and heading from visual/MS temporal evidence.

    Inputs contain the current MS#1 position and temporal state, but no MS
    uncertainty and no per-frame reference/GT coordinate.  The predicted motion
    polynomial is:
        d_t = v_t + 0.5 * a_t
        Delta_t = d_t [cos(theta_t), sin(theta_t)]
    There is no hand-coded maximum speed, acceleration, turn-rate, EMA, or
    heading-rate clamp.
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

        self.visual_mean_projection = projection(config.EMBED_DIM)
        self.visual_delta_projection = projection(config.EMBED_DIM)
        self.numeric_projection = nn.Sequential(
            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.gru = nn.GRUCell(feature_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.heading_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Neutral initialization: near-zero speed/acceleration and heading
        # increment. No dataset speed is hard-coded here.
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.constant_(self.motion_head[-1].bias, -2.0)
        self.motion_head[-1].bias.data[1] = 0.0
        nn.init.zeros_(self.heading_head[-1].weight)
        nn.init.zeros_(self.heading_head[-1].bias)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size), int(config.RNN_HIDDEN_DIM), device=device, dtype=dtype
        )

    def forward_step(
        self,
        z_uav,
        previous_z_uav,
        ms1_xy,
        prior_xy,
        previous_ms1_xy,
        previous_delta_xy,
        previous_speed,
        previous_acceleration,
        previous_heading_rad,
        hidden: Optional[torch.Tensor] = None,
    ):
        if hidden is None:
            hidden = self.initial_hidden(z_uav.shape[0], z_uav.device, z_uav.dtype)
        if previous_z_uav is None:
            previous_z_uav = z_uav
        if previous_ms1_xy is None:
            previous_ms1_xy = ms1_xy.detach()

        visual_mean = 0.5 * (z_uav + previous_z_uav)
        visual_delta = z_uav - previous_z_uav

        numeric = torch.cat(
            [
                (ms1_xy - prior_xy) / float(config.POSITION_INPUT_SCALE_M),
                (ms1_xy - previous_ms1_xy) / float(config.STEP_INPUT_SCALE_M),
                previous_delta_xy / float(config.STEP_INPUT_SCALE_M),
                previous_speed,
                previous_acceleration,
                torch.sin(previous_heading_rad),
                torch.cos(previous_heading_rad),
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
                self.visual_mean_projection(visual_mean),
                self.visual_delta_projection(visual_delta),
                self.numeric_projection(numeric),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        h = self.dropout(new_hidden)

        raw_motion = self.motion_head(h)
        speed = F.softplus(raw_motion[:, 0:1])
        acceleration = raw_motion[:, 1:2]

        heading_delta = self.heading_head(h)
        heading = previous_heading_rad + heading_delta
        heading = torch.atan2(torch.sin(heading), torch.cos(heading))

        travel = speed + 0.5 * acceleration
        delta_xy = torch.cat(
            [travel * torch.cos(heading), travel * torch.sin(heading)], dim=1
        )

        return MotionGRUOutput(
            speed_m_per_frame=speed,
            acceleration_m_per_frame2=acceleration,
            heading_rad=heading,
            heading_delta_rad=heading_delta,
            delta_xy=delta_xy,
            hidden=new_hidden,
        )


# Existing entry points import this name.
ThreeFrameRouteStateGRU = AutonomousMotionGRU
RouteProgressGRU = AutonomousMotionGRU
TwoFrameRouteStateGRU = AutonomousMotionGRU
