"""Frozen retrieval front-end and differentiable candidate provider."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import config
from visual_model import AllMapGeoCLIP


@dataclass
class CandidateBatch:
    indices: torch.Tensor
    centers: torch.Tensor
    z_uav: torch.Tensor
    z_sat: torch.Tensor
    raw_logits: torch.Tensor
    raw_prob: torch.Tensor
    raw_top1_xy: torch.Tensor
    hardms_xy: torch.Tensor
    hardms_support: torch.Tensor


def build_pixel_index(pixels: torch.Tensor):
    return {
        (int(round(x)), int(round(y))): index
        for index, (x, y) in enumerate(pixels.tolist())
    }


def regular_grid_indices(
    gallery_xy,
    gallery_pixel,
    pixel_index,
    prior_xy,
    grid_size,
    stride,
    device,
):
    """Return one fixed square lattice around the nearest gallery anchor."""
    grid_size = int(grid_size)
    if grid_size % 2 == 0:
        raise ValueError(
            "TMCR uses an odd GRID_SIZE so the motion prediction is the exact "
            "centre anchor."
        )
    radius = grid_size // 2
    offsets = range(-radius, radius + 1)
    gallery_xy_cpu = gallery_xy.cpu()
    gallery_pixel_cpu = gallery_pixel.cpu()
    rows = []

    for x_meter, y_meter in prior_xy.detach().cpu().tolist():
        distance_squared = (
            (gallery_xy_cpu[:, 0] - x_meter).square()
            + (gallery_xy_cpu[:, 1] - y_meter).square()
        )
        center_index = int(distance_squared.argmin())
        center_x, center_y = (
            int(round(value))
            for value in gallery_pixel_cpu[center_index].tolist()
        )

        row = []
        complete = True
        for offset_y in offsets:
            for offset_x in offsets:
                index = pixel_index.get(
                    (
                        center_x + offset_x * int(stride),
                        center_y + offset_y * int(stride),
                    )
                )
                if index is None:
                    complete = False
                    break
                row.append(index)
            if not complete:
                break

        # Near map borders, use the nearest unique anchors.  This is a geometry
        # fallback only; it does not inspect GT or retrieval scores.
        if not complete:
            row = torch.topk(
                distance_squared,
                k=grid_size * grid_size,
                largest=False,
            ).indices.tolist()
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long, device=device)


def hard_mean_shift(logits, centers, tau, bandwidth_m, iterations):
    """Archived Fixed HardMS baseline."""
    epsilon = 1e-8
    probability = torch.softmax(logits / float(tau), dim=1)
    modes = centers.clone()
    for _ in range(int(iterations)):
        distance_squared = torch.cdist(modes, centers).square()
        kernel = torch.exp(
            -distance_squared / (2.0 * float(bandwidth_m) ** 2)
        )
        weights = kernel * probability[:, None, :]
        modes = weights @ centers / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(epsilon)

    distance_squared = torch.cdist(modes, centers).square()
    support = (
        torch.exp(
            -distance_squared / (2.0 * float(bandwidth_m) ** 2)
        )
        * probability[:, None, :]
    ).sum(dim=2)
    mode_index = support.argmax(dim=1)
    chosen_mode = modes[
        torch.arange(modes.shape[0], device=modes.device), mode_index
    ]
    anchor_index = torch.cdist(
        chosen_mode[:, None], centers
    ).squeeze(1).argmin(dim=1)
    hard_xy = centers[
        torch.arange(centers.shape[0], device=centers.device), anchor_index
    ]
    return hard_xy, support.max(dim=1).values


class FrozenVisualLocalizer:
    """Loads the archived retrieval model and cached satellite gallery."""

    def __init__(self, device):
        checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
        self.origin_lat = float(checkpoint["origin_lat"])
        self.origin_lon = float(checkpoint["origin_lon"])
        self.model = AllMapGeoCLIP().to(device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.device = device
        self.gallery = {
            key: value.to(device) for key, value in checkpoint["gallery"].items()
        }
        self.pixel_index = build_pixel_index(self.gallery["pixel"])

    @torch.no_grad()
    def encode_uav_clip(self, uav):
        return self.model.encode_clip_image(
            uav.to(self.device, non_blocking=True)
        )

    @torch.no_grad()
    def encode_uav_spatial(self, uav):
        return self.model.encode_clip_spatial(
            uav.to(self.device, non_blocking=True),
            output_size=config.MOTION_SPATIAL_SIZE,
        )

    def candidate_batch(self, uav_clip, center_xy, grid_size=None):
        grid_size = int(grid_size or config.GRID_SIZE)
        indices = regular_grid_indices(
            self.gallery["xy"],
            self.gallery["pixel"],
            self.pixel_index,
            center_xy,
            grid_size,
            config.SAT_STRIDE,
            self.device,
        )
        centers = self.gallery["xy"][indices]
        satellite_clip = self.gallery["clip_feat"][indices]

        z_uav = self.model.encode_uav_from_clip(uav_clip)
        z_sat = self.model.encode_sat_from_clip(
            satellite_clip.reshape(-1, satellite_clip.shape[-1]),
            centers.reshape(-1, 2),
        ).reshape(centers.shape[0], centers.shape[1], -1)
        raw_logits = self.model.logit_scale.exp().clamp(max=100.0) * (
            z_uav[:, None] * z_sat
        ).sum(dim=2)
        raw_prob = torch.softmax(
            raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
        )
        raw_index = raw_logits.argmax(dim=1)
        raw_top1_xy = centers[
            torch.arange(centers.shape[0], device=self.device), raw_index
        ]
        hardms_xy, hardms_support = hard_mean_shift(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
        )
        return CandidateBatch(
            indices=indices,
            centers=centers,
            z_uav=z_uav,
            z_sat=z_sat,
            raw_logits=raw_logits,
            raw_prob=raw_prob,
            raw_top1_xy=raw_top1_xy,
            hardms_xy=hardms_xy,
            hardms_support=hardms_support,
        )

    @torch.no_grad()
    def candidate_contains_gt_anchor(self, indices, gt_xy):
        gallery_xy = self.gallery["xy"]
        distance_squared = (
            gallery_xy[None, :, :] - gt_xy[:, None, :]
        ).square().sum(dim=2)
        nearest_index = distance_squared.argmin(dim=1)
        return (indices == nearest_index[:, None]).any(dim=1)