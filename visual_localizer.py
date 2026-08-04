"""Frozen retrieval localizer with multi-mode Fixed Hard Mean-Shift.

The localizer does not make the final temporal decision. It returns the four
strongest spatial modes and grid-size-invariant concentration statistics so the
tracker can choose a smooth, motion-consistent path without trusting one noisy
Top-1 anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch

import config
from visual_model import AllMapGeoCLIP


@dataclass
class VisualMeasurement:
    raw_visual_xy: torch.Tensor
    fused_xy: torch.Tensor
    raw_top1_xy: torch.Tensor
    raw_logits: torch.Tensor
    fused_logits: torch.Tensor
    raw_prob: torch.Tensor
    fused_prob: torch.Tensor

    # [B, M, ...] multi-mode outputs.
    mode_xy: torch.Tensor
    mode_support: torch.Tensor
    mode_local_mass: torch.Tensor
    mode_spatial_std: torch.Tensor
    mode_peak_prob: torch.Tensor
    mode_anchor_index: torch.Tensor

    entropy: torch.Tensor
    candidate_indices: torch.Tensor
    centers: torch.Tensor
    search_centers: torch.Tensor
    grid_size: int


def build_pixel_index(pixels: torch.Tensor):
    return {
        (int(round(x)), int(round(y))): i
        for i, (x, y) in enumerate(pixels.tolist())
    }


def _axis_offset_options(n: int):
    """Return symmetric alternatives for odd/even lattice sizes."""
    half = n // 2
    if n % 2 == 1:
        return [list(range(-half, half + 1))]
    return [list(range(-half, half)), list(range(-half + 1, half + 1))]


def regular_grid_indices_single(
    gallery_xy: torch.Tensor,
    gallery_pixel: torch.Tensor,
    pixel_index: dict,
    prior_xy: torch.Tensor,
    grid_size: int,
    stride: int,
):
    """Build one n x n gallery lattice closest to a continuous prior."""
    n = int(grid_size)
    xy_cpu = gallery_xy.detach().cpu()
    pix_cpu = gallery_pixel.detach().cpu()
    prior_cpu = prior_xy.detach().cpu()

    d2 = (xy_cpu[:, 0] - prior_cpu[0]).square() + (
        xy_cpu[:, 1] - prior_cpu[1]
    ).square()
    nearest = int(d2.argmin())
    cx, cy = (int(round(v)) for v in pix_cpu[nearest].tolist())

    best_row = None
    best_error = float("inf")
    for x_offsets, y_offsets in product(
        _axis_offset_options(n), _axis_offset_options(n)
    ):
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
        centre_error = float(((mean_xy - prior_cpu) ** 2).sum())
        if centre_error < best_error:
            best_error = centre_error
            best_row = row

    if best_row is None:
        k = min(n * n, int(gallery_xy.shape[0]))
        best_row = torch.topk(d2, k=k, largest=False).indices.tolist()
    return best_row


def union_grid_indices(
    gallery_xy: torch.Tensor,
    gallery_pixel: torch.Tensor,
    pixel_index: dict,
    search_centers: torch.Tensor,
    grid_size: int,
    stride: int,
    device: torch.device,
):
    """Union several local windows into one recovery corridor.

    This tracker evaluates one UAV frame at a time, so search_centers has shape
    [S, 2]. Duplicate anchors are removed while preserving deterministic order.
    """
    if search_centers.ndim != 2 or search_centers.shape[1] != 2:
        raise ValueError("search_centers must have shape [S, 2]")

    ordered = []
    seen = set()
    for centre in search_centers:
        row = regular_grid_indices_single(
            gallery_xy,
            gallery_pixel,
            pixel_index,
            centre,
            grid_size,
            stride,
        )
        for idx in row:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)

    if not ordered:
        raise RuntimeError("No satellite candidates were generated")
    return torch.tensor([ordered], dtype=torch.long, device=device)


def anisotropic_prior_log(
    centers: torch.Tensor,
    predicted_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    sigma_along: float,
    sigma_cross: float,
):
    residual = centers - predicted_xy[:, None]
    speed = velocity_xy.norm(dim=1, keepdim=True)
    fallback = torch.tensor(
        [1.0, 0.0], device=centers.device, dtype=centers.dtype
    )
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


def hard_mean_shift_top_modes(
    logits: torch.Tensor,
    centers: torch.Tensor,
    tau: float,
    bandwidth_m: float,
    iterations: int,
    top_modes: int,
    nms_radius_m: float,
    local_radius_m: float,
):
    """Run Fixed HardMS and retain several distinct spatial modes."""
    eps = 1e-8
    batch_size, candidate_count = logits.shape
    prob = torch.softmax(logits / float(tau), dim=1)
    seed_modes = centers.clone()

    for _ in range(int(iterations)):
        dist2 = torch.cdist(seed_modes, centers).square()
        kernel = torch.exp(-dist2 / (2.0 * float(bandwidth_m) ** 2))
        weights = kernel * prob[:, None, :]
        seed_modes = weights @ centers / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(eps)

    dist2 = torch.cdist(seed_modes, centers).square()
    support = (
        torch.exp(-dist2 / (2.0 * float(bandwidth_m) ** 2))
        * prob[:, None, :]
    ).sum(dim=2)

    output_xy = []
    output_support = []
    output_mass = []
    output_std = []
    output_peak = []
    output_anchor = []

    for b in range(batch_size):
        chosen = []
        chosen_seed_support = []
        order = torch.argsort(support[b], descending=True)

        for seed_id_tensor in order:
            seed_id = int(seed_id_tensor.item())
            mode = seed_modes[b, seed_id]
            anchor_id = int(
                torch.cdist(mode[None, None], centers[b:b + 1])
                .reshape(-1)
                .argmin()
                .item()
            )
            anchor_xy = centers[b, anchor_id]
            if any(
                float(torch.norm(anchor_xy - old_xy).item())
                < float(nms_radius_m)
                for old_xy, _, _ in chosen
            ):
                continue
            chosen.append((anchor_xy, anchor_id, seed_id))
            chosen_seed_support.append(support[b, seed_id])
            if len(chosen) >= int(top_modes):
                break

        # Fallback: fill missing modes using high-probability distinct anchors.
        if len(chosen) < int(top_modes):
            for anchor_tensor in torch.argsort(prob[b], descending=True):
                anchor_id = int(anchor_tensor.item())
                anchor_xy = centers[b, anchor_id]
                if any(
                    float(torch.norm(anchor_xy - old_xy).item())
                    < float(nms_radius_m)
                    for old_xy, _, _ in chosen
                ):
                    continue
                chosen.append((anchor_xy, anchor_id, anchor_id))
                chosen_seed_support.append(prob[b, anchor_id])
                if len(chosen) >= int(top_modes):
                    break

        while len(chosen) < int(top_modes):
            chosen.append(chosen[-1])
            chosen_seed_support.append(chosen_seed_support[-1])

        positions = []
        masses = []
        stds = []
        peaks = []
        anchors = []
        for anchor_xy, anchor_id, _ in chosen[: int(top_modes)]:
            d2_local = ((centers[b] - anchor_xy) ** 2).sum(dim=1)
            local_mask = d2_local <= float(local_radius_m) ** 2
            local_weight = prob[b] * local_mask.to(prob.dtype)
            mass = local_weight.sum().clamp_min(eps)
            variance = (local_weight * d2_local).sum() / mass

            positions.append(anchor_xy)
            masses.append(mass)
            stds.append(torch.sqrt(variance.clamp_min(0.0)))
            peaks.append(prob[b, anchor_id])
            anchors.append(anchor_id)

        seed_support_tensor = torch.stack(
            chosen_seed_support[: int(top_modes)]
        )
        seed_support_tensor = seed_support_tensor / seed_support_tensor.max().clamp_min(
            eps
        )

        output_xy.append(torch.stack(positions))
        output_support.append(seed_support_tensor)
        output_mass.append(torch.stack(masses))
        output_std.append(torch.stack(stds))
        output_peak.append(torch.stack(peaks))
        output_anchor.append(
            torch.tensor(anchors, device=centers.device, dtype=torch.long)
        )

    return (
        torch.stack(output_xy),
        torch.stack(output_support),
        torch.stack(output_mass),
        torch.stack(output_std),
        torch.stack(output_peak),
        torch.stack(output_anchor),
        prob,
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
        uav_clip: torch.Tensor,
        predicted_xy: torch.Tensor,
        velocity_xy: torch.Tensor,
        search_centers: torch.Tensor,
        grid_size: int,
        sigma_along: float,
        sigma_cross: float,
        prior_weight: float,
    ):
        if uav_clip.shape[0] != 1:
            raise ValueError("This tracker measures one UAV frame at a time")

        search_centers = search_centers.to(self.device)
        index = union_grid_indices(
            self.gallery["xy"],
            self.gallery["pixel"],
            self.pixel_index,
            search_centers,
            int(grid_size),
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
        raw_prob = torch.softmax(
            raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
        )

        (
            raw_mode_xy,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = hard_mean_shift_top_modes(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.TOP_MODES,
            config.MODE_NMS_RADIUS_M,
            config.MODE_LOCAL_RADIUS_M,
        )

        prior_log = anisotropic_prior_log(
            centers,
            predicted_xy,
            velocity_xy,
            sigma_along,
            sigma_cross,
        )
        fused_logits = (
            torch.log_softmax(
                raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
            )
            + float(prior_weight) * prior_log
        )

        (
            mode_xy,
            mode_support,
            mode_local_mass,
            mode_spatial_std,
            mode_peak_prob,
            mode_anchor_index,
            fused_prob,
        ) = hard_mean_shift_top_modes(
            fused_logits,
            centers,
            1.0,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.TOP_MODES,
            config.MODE_NMS_RADIUS_M,
            config.MODE_LOCAL_RADIUS_M,
        )

        entropy = -(
            fused_prob * fused_prob.clamp_min(1e-8).log()
        ).sum(dim=1) / torch.log(
            torch.tensor(float(fused_prob.shape[1]), device=self.device)
        )
        raw_top_index = raw_logits.argmax(dim=1)
        batch = torch.arange(centers.shape[0], device=self.device)
        raw_top1_xy = centers[batch, raw_top_index]

        return VisualMeasurement(
            raw_visual_xy=raw_mode_xy[:, 0],
            fused_xy=mode_xy[:, 0],
            raw_top1_xy=raw_top1_xy,
            raw_logits=raw_logits,
            fused_logits=fused_logits,
            raw_prob=raw_prob,
            fused_prob=fused_prob,
            mode_xy=mode_xy,
            mode_support=mode_support,
            mode_local_mass=mode_local_mass,
            mode_spatial_std=mode_spatial_std,
            mode_peak_prob=mode_peak_prob,
            mode_anchor_index=mode_anchor_index,
            entropy=entropy,
            candidate_indices=index,
            centers=centers,
            search_centers=search_centers,
            grid_size=int(grid_size),
        )

    @staticmethod
    def mode_at_search_boundary(measurement: VisualMeasurement, mode_id: int):
        mode_xy = measurement.mode_xy[0, int(mode_id)]
        delta = torch.abs(measurement.search_centers - mode_xy[None])
        chebyshev = delta.max(dim=1).values
        nearest = chebyshev.min()
        half_extent = (
            max(int(measurement.grid_size) - 1, 1)
            * float(config.CANDIDATE_SPACING_M)
            / 2.0
        )
        return bool(
            nearest.item() >= float(config.BOUNDARY_RATIO) * half_extent
        )

    @torch.no_grad()
    def candidate_contains_gt(self, measurement, gt_xy):
        """Evaluation-only CCR; never used by tracking decisions."""
        gallery_xy = self.gallery["xy"]
        d2 = (gallery_xy[None] - gt_xy[:, None]).square().sum(dim=2)
        nearest_global = d2.argmin(dim=1)
        return (
            measurement.candidate_indices == nearest_global[:, None]
        ).any(dim=1)