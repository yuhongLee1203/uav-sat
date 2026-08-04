"""Frozen 6x6 visual measurement model used by temporal tracking."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import config
from visual_model import AllMapGeoCLIP


@dataclass
class VisualMeasurement:
    visual_xy: torch.Tensor
    fused_xy: torch.Tensor
    raw_top1_xy: torch.Tensor
    raw_logits: torch.Tensor
    fused_logits: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    top1_score: torch.Tensor
    basin_support: torch.Tensor
    innovation: torch.Tensor
    centers: torch.Tensor
    raw_top_index: torch.Tensor
    grid_size: int


def build_pixel_index(pixels: torch.Tensor):
    return {(int(round(x)), int(round(y))): i for i, (x, y) in enumerate(pixels.tolist())}


def regular_grid_indices(gallery_xy, gallery_pixel, pixel_index, prior_xy, grid_size, stride, device):
    """Return a fixed lattice window centred on the anchor nearest `prior_xy`."""
    n = int(grid_size)
    offsets = list(range(-(n // 2), -(n // 2) + n))
    xy_cpu, pix_cpu = gallery_xy.cpu(), gallery_pixel.cpu()
    rows = []
    for x_m, y_m in prior_xy.detach().cpu().tolist():
        d2 = (xy_cpu[:, 0] - x_m).square() + (xy_cpu[:, 1] - y_m).square()
        center = int(d2.argmin())
        cx, cy = (int(round(v)) for v in pix_cpu[center].tolist())
        row, complete = [], True
        for oy in offsets:
            for ox in offsets:
                idx = pixel_index.get((cx + ox * stride, cy + oy * stride))
                if idx is None:
                    complete = False
                    break
                row.append(idx)
            if not complete:
                break
        if not complete:
            row = torch.topk(d2, k=n * n, largest=False).indices.tolist()
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def hard_mean_shift(logits, centers, tau, bandwidth_m, iterations):
    """Run fixed-density HardMS and snap the strongest mode to an anchor."""
    eps = 1e-8
    prob = torch.softmax(logits / float(tau), dim=1)
    modes = centers.clone()
    for _ in range(int(iterations)):
        dist2 = torch.cdist(modes, centers).square()
        kernel = torch.exp(-dist2 / (2.0 * float(bandwidth_m) ** 2))
        weights = kernel * prob[:, None, :]
        modes = weights @ centers / weights.sum(dim=2, keepdim=True).clamp_min(eps)
    dist2 = torch.cdist(modes, centers).square()
    support = (torch.exp(-dist2 / (2.0 * float(bandwidth_m) ** 2)) * prob[:, None, :]).sum(dim=2)
    mode_id = support.argmax(dim=1)
    chosen_mode = modes[torch.arange(modes.shape[0], device=modes.device), mode_id]
    anchor_id = torch.cdist(chosen_mode[:, None], centers).squeeze(1).argmin(dim=1)
    hard_xy = centers[torch.arange(centers.shape[0], device=centers.device), anchor_id]
    return hard_xy, support.max(dim=1).values


class FrozenVisualLocalizer:
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
        self.gallery = {k: v.to(device) for k, v in checkpoint["gallery"].items()}
        self.pixel_index = build_pixel_index(self.gallery["pixel"])

    @torch.no_grad()
    def encode_uav_clip(self, uav):
        return self.model.encode_clip_image(uav.to(self.device, non_blocking=True))

    @torch.no_grad()
    def measure(self, uav_clip, predicted_xy, grid_size=None, motion_sigma=None):
        grid_size = int(grid_size or config.GRID_SIZE)
        motion_sigma = float(motion_sigma or config.MOTION_SIGMA_M)
        index = regular_grid_indices(
            self.gallery["xy"], self.gallery["pixel"], self.pixel_index,
            predicted_xy, grid_size, config.SAT_STRIDE, self.device,
        )
        centers = self.gallery["xy"][index]
        sat_clip = self.gallery["clip_feat"][index]
        z_uav = self.model.encode_uav_from_clip(uav_clip)
        z_sat = self.model.encode_sat_from_clip(
            sat_clip.reshape(-1, sat_clip.shape[-1]), centers.reshape(-1, 2)
        ).reshape(centers.shape[0], centers.shape[1], -1)
        raw_logits = self.model.logit_scale.exp().clamp(max=100.0) * (z_uav[:, None] * z_sat).sum(dim=-1)
        raw_prob = torch.softmax(raw_logits / config.MEANSHIFT_SCORE_TAU, dim=1)
        entropy = -(raw_prob * raw_prob.clamp_min(1e-8).log()).sum(dim=1) / torch.log(torch.tensor(float(raw_prob.shape[1]), device=self.device))
        values = raw_logits.topk(k=2, dim=1).values
        margin = values[:, 0] - values[:, 1]
        raw_top_index = raw_logits.argmax(dim=1)
        raw_top = centers[torch.arange(centers.shape[0], device=self.device), raw_top_index]
        visual_xy, basin_support = hard_mean_shift(
            raw_logits, centers, config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M, config.MEANSHIFT_ITERATIONS,
        )

        # Bayesian motion prior: visual local likelihood times a Gaussian prior.
        prior_log = -(centers - predicted_xy[:, None]).square().sum(dim=2) / (2.0 * motion_sigma * motion_sigma)
        fused_logits = torch.log_softmax(raw_logits / config.MEANSHIFT_SCORE_TAU, dim=1) + prior_log
        fused_xy, _ = hard_mean_shift(
            fused_logits, centers, 1.0, config.MEANSHIFT_BANDWIDTH_M, config.MEANSHIFT_ITERATIONS,
        )
        return VisualMeasurement(
            visual_xy=visual_xy,
            fused_xy=fused_xy,
            raw_top1_xy=raw_top,
            raw_logits=raw_logits,
            fused_logits=fused_logits,
            entropy=entropy,
            margin=margin,
            top1_score=values[:, 0],
            basin_support=basin_support,
            # The gate receives the motion-fused measurement, rather than an
            # unconstrained raw peak that can sit at the opposite grid edge.
            innovation=fused_xy - predicted_xy,
            centers=centers,
            raw_top_index=raw_top_index,
            grid_size=grid_size,
        )

    @staticmethod
    def at_grid_edge(measurement):
        n = int(measurement.grid_size)
        row = measurement.raw_top_index // n
        col = measurement.raw_top_index % n
        return (row == 0) | (row == n - 1) | (col == 0) | (col == n - 1)
