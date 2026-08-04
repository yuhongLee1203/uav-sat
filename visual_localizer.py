"""Frozen local HardMS measurement with a straight-line anisotropic prior."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

import config
from visual_model import AllMapGeoCLIP


@dataclass
class VisualMeasurement:
    # Raw visual-only HardMS output.
    raw_visual_xy: torch.Tensor
    # Motion-prior fused HardMS output. The tracker must use this field.
    fused_xy: torch.Tensor
    raw_top1_xy: torch.Tensor
    raw_logits: torch.Tensor
    fused_logits: torch.Tensor
    entropy: torch.Tensor
    peak_prob: torch.Tensor
    margin_prob: torch.Tensor
    basin_support: torch.Tensor
    centers: torch.Tensor
    candidate_indices: torch.Tensor
    raw_top_index: torch.Tensor
    fused_anchor_index: torch.Tensor
    grid_size: int


def build_pixel_index(pixels: torch.Tensor):
    return {(int(round(x)), int(round(y))): i for i, (x, y) in enumerate(pixels.tolist())}


def _axis_offset_options(n: int):
    """Return symmetric alternatives for odd/even lattice sizes.

    The old implementation always used [-3, -2, -1, 0, 1, 2] for n=6,
    producing a permanent one-stride directional bias. For an even grid there
    are two valid half-cell-centred alternatives; choose the one whose mean is
    closest to the continuous prior.
    """
    half = n // 2
    if n % 2 == 1:
        return [list(range(-half, half + 1))]
    return [list(range(-half, half)), list(range(-half + 1, half + 1))]


def regular_grid_indices(
    gallery_xy,
    gallery_pixel,
    pixel_index,
    prior_xy,
    grid_size,
    stride,
    device,
):
    """Return a lattice window whose geometric centre is closest to prior_xy."""
    n = int(grid_size)
    xy_cpu = gallery_xy.detach().cpu()
    pix_cpu = gallery_pixel.detach().cpu()
    rows = []

    x_options = _axis_offset_options(n)
    y_options = _axis_offset_options(n)

    for x_m, y_m in prior_xy.detach().cpu().tolist():
        d2 = (xy_cpu[:, 0] - x_m).square() + (xy_cpu[:, 1] - y_m).square()
        nearest = int(d2.argmin())
        cx, cy = (int(round(v)) for v in pix_cpu[nearest].tolist())

        best_row = None
        best_error = float("inf")
        for x_offsets, y_offsets in product(x_options, y_options):
            row = []
            complete = True
            for oy in y_offsets:
                for ox in x_offsets:
                    idx = pixel_index.get((cx + ox * stride, cy + oy * stride))
                    if idx is None:
                        complete = False
                        break
                    row.append(idx)
                if not complete:
                    break
            if not complete:
                continue

            mean_xy = xy_cpu[row].mean(dim=0)
            centre_error = float((mean_xy[0] - x_m) ** 2 + (mean_xy[1] - y_m) ** 2)
            if centre_error < best_error:
                best_error = centre_error
                best_row = row

        # At map boundaries, retain the nearest n*n anchors as a safe fallback.
        if best_row is None:
            best_row = torch.topk(d2, k=n * n, largest=False).indices.tolist()
        rows.append(best_row)

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
    support = (
        torch.exp(-dist2 / (2.0 * float(bandwidth_m) ** 2))
        * prob[:, None, :]
    ).sum(dim=2)
    mode_id = support.argmax(dim=1)
    batch = torch.arange(modes.shape[0], device=modes.device)
    chosen_mode = modes[batch, mode_id]
    anchor_id = torch.cdist(chosen_mode[:, None], centers).squeeze(1).argmin(dim=1)
    hard_xy = centers[batch, anchor_id]
    return hard_xy, support.max(dim=1).values, anchor_id


def anisotropic_prior_log(
    centers: torch.Tensor,
    predicted_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    sigma_along: float,
    sigma_cross: float,
):
    residual = centers - predicted_xy[:, None]
    speed = velocity_xy.norm(dim=1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0], device=centers.device, dtype=centers.dtype)
    direction = torch.where(
        speed > 1e-6,
        velocity_xy / speed.clamp_min(1e-6),
        fallback.expand_as(velocity_xy),
    )
    normal = torch.stack([-direction[:, 1], direction[:, 0]], dim=1)
    along = (residual * direction[:, None]).sum(dim=2)
    cross = (residual * normal[:, None]).sum(dim=2)
    return -0.5 * (
        along.square() / float(sigma_along) ** 2
        + cross.square() / float(sigma_cross) ** 2
    )


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
    def measure(
        self,
        uav_clip,
        predicted_xy,
        velocity_xy,
        grid_size=None,
        sigma_along=None,
        sigma_cross=None,
    ):
        grid_size = int(grid_size or config.GRID_SIZE)
        sigma_along = float(sigma_along or config.MOTION_SIGMA_ALONG_M)
        sigma_cross = float(sigma_cross or config.MOTION_SIGMA_CROSS_M)

        index = regular_grid_indices(
            self.gallery["xy"],
            self.gallery["pixel"],
            self.pixel_index,
            predicted_xy,
            grid_size,
            config.SAT_STRIDE,
            self.device,
        )
        centers = self.gallery["xy"][index]
        sat_clip = self.gallery["clip_feat"][index]

        z_uav = self.model.encode_uav_from_clip(uav_clip)
        z_sat = self.model.encode_sat_from_clip(
            sat_clip.reshape(-1, sat_clip.shape[-1]),
            centers.reshape(-1, 2),
        ).reshape(centers.shape[0], centers.shape[1], -1)

        raw_logits = self.model.logit_scale.exp().clamp(max=100.0) * (
            z_uav[:, None] * z_sat
        ).sum(dim=-1)

        raw_visual_xy, basin_support, _ = hard_mean_shift(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
        )

        prior_log = anisotropic_prior_log(
            centers,
            predicted_xy,
            velocity_xy,
            sigma_along,
            sigma_cross,
        )
        # log_softmax keeps the visual term numerically stable. The prior is in
        # log-probability units and directly suppresses lateral field-row hops.
        fused_logits = (
            torch.log_softmax(raw_logits / config.MEANSHIFT_SCORE_TAU, dim=1)
            + prior_log
        )
        fused_xy, _, fused_anchor_index = hard_mean_shift(
            fused_logits,
            centers,
            1.0,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
        )

        fused_prob = torch.softmax(fused_logits, dim=1)
        entropy = -(
            fused_prob * fused_prob.clamp_min(1e-8).log()
        ).sum(dim=1) / torch.log(
            torch.tensor(float(fused_prob.shape[1]), device=self.device)
        )
        top_values = fused_prob.topk(k=2, dim=1).values
        peak_prob = top_values[:, 0]
        margin_prob = top_values[:, 0] - top_values[:, 1]

        raw_top_index = raw_logits.argmax(dim=1)
        batch = torch.arange(centers.shape[0], device=self.device)
        raw_top = centers[batch, raw_top_index]

        return VisualMeasurement(
            raw_visual_xy=raw_visual_xy,
            fused_xy=fused_xy,
            raw_top1_xy=raw_top,
            raw_logits=raw_logits,
            fused_logits=fused_logits,
            entropy=entropy,
            peak_prob=peak_prob,
            margin_prob=margin_prob,
            basin_support=basin_support,
            centers=centers,
            candidate_indices=index,
            raw_top_index=raw_top_index,
            fused_anchor_index=fused_anchor_index,
            grid_size=grid_size,
        )

    @staticmethod
    def at_grid_edge(measurement):
        n = int(measurement.grid_size)
        row = measurement.fused_anchor_index // n
        col = measurement.fused_anchor_index % n
        return (row == 0) | (row == n - 1) | (col == 0) | (col == n - 1)

    @torch.no_grad()
    def candidate_contains_gt(self, measurement, gt_xy):
        """Evaluation-only CCR: is the gallery anchor nearest GT in candidates?"""
        gallery_xy = self.gallery["xy"]
        d2 = (gallery_xy[None] - gt_xy[:, None]).square().sum(dim=2)
        nearest_global = d2.argmin(dim=1)
        return (measurement.candidate_indices == nearest_global[:, None]).any(dim=1)