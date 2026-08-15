
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import RouteDataset, SatPatchGallery
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
    softms_xy: torch.Tensor
    softms_support: torch.Tensor


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
    grid_size = int(grid_size)
    start = -(grid_size // 2)
    offsets = range(start, start + grid_size)
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

        if not complete:
            row = torch.topk(
                distance_squared,
                k=grid_size * grid_size,
                largest=False,
            ).indices.tolist()
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long, device=device)


def soft_mean_shift(logits, centers, tau, bandwidth_m, iterations, beta):
    """Density-weight all shifted modes; no Top-K and no candidate snapping."""
    epsilon = 1e-8
    probability = torch.softmax(logits / max(float(tau), 1e-6), dim=1)
    centers = centers.float()
    modes = centers.clone()
    bandwidth = max(float(bandwidth_m), 1e-6)

    for _ in range(int(iterations)):
        distance_squared = torch.cdist(modes, centers).square()
        kernel = torch.exp(
            -distance_squared / (2.0 * bandwidth ** 2)
        )
        weights = kernel * probability[:, None, :]
        modes = weights @ centers / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(epsilon)

    final_distance_squared = torch.cdist(modes, centers).square()
    density = (
        torch.exp(-final_distance_squared / (2.0 * bandwidth ** 2))
        * probability[:, None, :]
    ).sum(dim=2)
    mode_weights = torch.softmax(float(beta) * density, dim=1)
    soft_xy = (mode_weights.unsqueeze(-1) * modes).sum(dim=1)
    soft_support = (mode_weights * density).sum(dim=1)
    compact = (
        mode_weights * (modes - soft_xy.unsqueeze(1)).square().sum(dim=2)
    ).sum(dim=1).mean()
    return soft_xy, soft_support, modes, density, mode_weights, compact


def _validate_visual_provenance(checkpoint):
    train_routes = checkpoint.get("visual_train_routes")
    validation_routes = checkpoint.get("visual_validation_routes")
    previous_loaded = checkpoint.get("previous_task_checkpoint_loaded")

    if train_routes != ["route_A"]:
        raise RuntimeError(
            "Visual checkpoint is not strict Route-A-only training: "
            f"visual_train_routes={train_routes}"
        )
    if validation_routes != ["route_A"]:
        raise RuntimeError(
            "Visual checkpoint validation was not Route A only: "
            f"visual_validation_routes={validation_routes}"
        )
    if previous_loaded is not False:
        raise RuntimeError(
            "Visual checkpoint provenance does not prove that no previous "
            "task-specific checkpoint was loaded."
        )
    if checkpoint.get("model_format") != "task_specific_only":
        raise RuntimeError("Unsupported visual checkpoint format")


class FrozenVisualLocalizer:
    def __init__(self, device):
        if not config.VISUAL_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"A-only visual checkpoint not found: {config.VISUAL_CHECKPOINT}. "
                "Run robust_tracker.py with --mode train_eval first."
            )

        checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
        _validate_visual_provenance(checkpoint)

        self.origin_lat = float(checkpoint["origin_lat"])
        self.origin_lon = float(checkpoint["origin_lon"])
        self.model = AllMapGeoCLIP().to(device)

        incompatible = self.model.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        non_clip_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("clip.")
        ]
        if unexpected or non_clip_missing:
            raise RuntimeError(
                "Visual checkpoint state mismatch: "
                f"unexpected={unexpected}, non_clip_missing={non_clip_missing}"
            )

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.device = device
        self.gallery = {
            key: value.to(device)
            for key, value in checkpoint["gallery"].items()
        }
        self.pixel_index = build_pixel_index(self.gallery["pixel"])

    @torch.no_grad()
    def encode_uav_clip(self, uav):
        return self.model.encode_clip_image(
            uav.to(self.device, non_blocking=True)
        )

    @torch.no_grad()
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
        softms_xy, softms_support, _, _, _, _ = soft_mean_shift(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )

        return CandidateBatch(
            indices=indices,
            centers=centers,
            z_uav=z_uav,
            z_sat=z_sat,
            raw_logits=raw_logits,
            raw_prob=raw_prob,
            raw_top1_xy=raw_top1_xy,
            softms_xy=softms_xy,
            softms_support=softms_support,
        )

    @torch.no_grad()
    def candidate_contains_gt_anchor(self, indices, gt_xy):
        candidate_xy = self.gallery["xy"][indices]
        nearest_distance = torch.linalg.norm(
            candidate_xy - gt_xy[:, None, :], dim=2
        ).min(dim=1).values
        return nearest_distance <= float(config.CANDIDATE_CAPTURE_RADIUS_M)


@dataclass
class VisualTrainCache:
    uav_clip: torch.Tensor
    gt_xy: torch.Tensor
    frame_ids: torch.Tensor
    candidate_indices: torch.Tensor
    target_indices: torch.Tensor
    capture: torch.Tensor

    def __len__(self):
        return int(self.gt_xy.shape[0])


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def _deterministic_jitter(length, route_index, maximum_m):
    if maximum_m <= 0:
        return torch.zeros(length, 2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.SEED) + 1009 * int(route_index))
    radius = (
        torch.sqrt(torch.rand(length, 1, generator=generator))
        * float(maximum_m)
    )
    angle = (
        torch.rand(length, 1, generator=generator)
        * (2.0 * math.pi)
    )
    return torch.cat(
        [radius * angle.cos(), radius * angle.sin()],
        dim=1,
    )


def _split_ranges(length):
    guard = int(config.SPLIT_GUARD_FRAMES)
    train_end = int(length * float(config.TRAIN_FRACTION))
    val_end = int(
        length
        * (
            float(config.TRAIN_FRACTION)
            + float(config.VAL_FRACTION)
        )
    )
    return {
        "train": (0, max(0, train_end - guard)),
        "val": (
            min(length, train_end + guard),
            max(min(length, train_end + guard), val_end - guard),
        ),
    }


def _nearest_target(centers, gt_xy):
    return (
        centers - gt_xy[:, None, :]
    ).square().sum(dim=-1).argmin(dim=-1)


def _task_specific_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("clip.")
    }


def _load_task_specific_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_clip_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("clip.")
    ]
    if unexpected or non_clip_missing:
        raise RuntimeError(
            "Visual state mismatch: "
            f"unexpected={unexpected}, non_clip_missing={non_clip_missing}"
        )


@torch.no_grad()
def _build_satellite_backbone_gallery(
    model,
    origin_lat,
    origin_lon,
    device,
):
    dataset = SatPatchGallery(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    clip_rows = []
    xy_rows = []
    pixel_rows = []
    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)
    model.eval()

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        satellite = torch.stack([item["sat"] for item in items]).to(device)
        clip_rows.append(
            model.encode_clip_image(satellite).detach().cpu().to(torch.float16)
        )
        xy_rows.append(torch.stack([item["xy"] for item in items]).float())
        pixel_rows.append(torch.stack([item["pixel"] for item in items]).float())
        if start == 0 or end == len(dataset) or (start // batch_size) % 20 == 0:
            print(f"satellite backbone cache: {end}/{len(dataset)}", flush=True)

    gallery = {
        "clip_feat": torch.cat(clip_rows),
        "xy": torch.cat(xy_rows),
        "pixel": torch.cat(pixel_rows),
    }
    print(f"satellite gallery size={len(dataset)}", flush=True)
    return gallery


@torch.no_grad()
def _build_route_a_visual_cache(
    model,
    gallery,
    device,
    jitter_m,
):
    dataset = RouteDataset(Path(config.ROUTE_ROOTS[0]), train=False)
    uav_rows = []
    gt_rows = []
    frame_rows = []
    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)
    model.eval()

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        uav_rows.append(
            model.encode_clip_image(uav).detach().cpu().to(torch.float16)
        )
        gt_rows.append(torch.stack([item["xy"] for item in items]).float())
        frame_rows.extend(_parse_frame_id(item["frame_id"]) for item in items)
        if start == 0 or end == len(dataset) or (start // batch_size) % 20 == 0:
            print(f"Route A backbone cache: {end}/{len(dataset)}", flush=True)

    gt_xy = torch.cat(gt_rows)
    prior_xy = gt_xy + _deterministic_jitter(
        len(dataset),
        route_index=0,
        maximum_m=float(jitter_m),
    )
    candidate_indices = regular_grid_indices(
        gallery["xy"],
        gallery["pixel"],
        build_pixel_index(gallery["pixel"]),
        prior_xy,
        config.GRID_SIZE,
        config.SAT_STRIDE,
        device=torch.device("cpu"),
    )
    candidate_centers = gallery["xy"][candidate_indices]
    target_indices = _nearest_target(candidate_centers, gt_xy)
    nearest_distance = torch.linalg.norm(
        candidate_centers - gt_xy[:, None, :],
        dim=2,
    ).min(dim=1).values
    capture = nearest_distance <= float(config.CANDIDATE_CAPTURE_RADIUS_M)

    cache = VisualTrainCache(
        uav_clip=torch.cat(uav_rows),
        gt_xy=gt_xy,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        candidate_indices=candidate_indices,
        target_indices=target_indices,
        capture=capture,
    )
    print(
        "Route A visual candidate capture="
        f"{capture.float().mean().item() * 100:.2f}%",
        flush=True,
    )
    return cache, float(dataset.origin_lat), float(dataset.origin_lon)


def _visual_forward_batch(model, cache, gallery, batch_indices, device):
    batch_indices = batch_indices.long()
    candidate_indices = cache.candidate_indices[batch_indices]
    centers = gallery["xy"][candidate_indices].to(device)
    sat_clip = gallery["clip_feat"][candidate_indices].to(device).float()
    uav_clip = cache.uav_clip[batch_indices].to(device).float()
    target = cache.target_indices[batch_indices].to(device)
    gt_xy = cache.gt_xy[batch_indices].to(device)

    z_uav = model.encode_uav_from_clip(uav_clip)
    z_sat = model.encode_sat_from_clip(
        sat_clip.reshape(-1, sat_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    logits = model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    softms_xy, _, _, _, _, _ = soft_mean_shift(
        logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )

    ce = F.cross_entropy(
        logits,
        target,
        label_smoothing=float(config.VISUAL_LABEL_SMOOTHING),
    )
    coordinate = F.smooth_l1_loss(softms_xy, gt_xy)
    loss = ce + float(config.VISUAL_COORD_LOSS_WEIGHT) * coordinate
    return loss, logits, centers, gt_xy


@torch.no_grad()
def _evaluate_visual_model(model, cache, gallery, indices, device):
    model.eval()
    top1_rows = []
    softms_rows = []
    gt_rows = []
    batch_size = int(config.VISUAL_BATCH_SIZE)

    for offset in range(0, len(indices), batch_size):
        batch = torch.tensor(indices[offset : offset + batch_size])
        _, logits, centers, gt_xy = _visual_forward_batch(
            model, cache, gallery, batch, device
        )
        top1_index = logits.argmax(dim=1)
        top1 = centers[
            torch.arange(centers.shape[0], device=device),
            top1_index,
        ]
        softms, _, _, _, _, _ = soft_mean_shift(
            logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )
        top1_rows.append(top1.cpu())
        softms_rows.append(softms.cpu())
        gt_rows.append(gt_xy.cpu())

    top1 = torch.cat(top1_rows)
    softms = torch.cat(softms_rows)
    gt = torch.cat(gt_rows)
    top1_error = torch.linalg.norm(top1 - gt, dim=1)
    softms_error = torch.linalg.norm(softms - gt, dim=1)
    return {
        "Top1_MLE_m": float(top1_error.mean().item()),
        "SoftMS_MLE_m": float(softms_error.mean().item()),
        "SoftMS_P90_m": float(torch.quantile(softms_error, 0.90).item()),
        "LSR@10_pct": float((softms_error <= 10.0).float().mean().item() * 100),
    }


def train_visual_retrieval_a_only(device, epochs, jitter_m, resume=False):
    """Train random task heads only on Route A.

    AllMapGeoCLIP loads the public MobileCLIP pretrained backbone. The backbone
    remains frozen. No prior UAV/SAT task checkpoint is read.
    """
    _set_seed(int(config.SEED))
    model = AllMapGeoCLIP().to(device)

    route_a = RouteDataset(Path(config.ROUTE_ROOTS[0]), train=False)
    origin_lat = float(route_a.origin_lat)
    origin_lon = float(route_a.origin_lon)
    gallery = _build_satellite_backbone_gallery(
        model, origin_lat, origin_lon, device
    )
    cache, origin_lat, origin_lon = _build_route_a_visual_cache(
        model, gallery, device, jitter_m
    )

    split = _split_ranges(len(cache))
    train_start, train_end = split["train"]
    val_start, val_end = split["val"]
    train_indices = [
        index
        for index in range(train_start, train_end)
        if bool(cache.capture[index])
    ]
    val_indices = list(range(val_start, val_end))
    if not train_indices or not val_indices:
        raise RuntimeError("Route A visual train/validation split is empty")

    train_capture = cache.capture[train_start:train_end].float().mean().item()
    if train_capture < float(config.MIN_TRAIN_CAPTURE_RATE):
        raise RuntimeError(
            "Route A visual candidate capture is below "
            f"{100.0 * float(config.MIN_TRAIN_CAPTURE_RATE):.1f}%"
        )

    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.VISUAL_LR),
        weight_decay=float(config.VISUAL_WEIGHT_DECAY),
    )
    start_epoch = 0
    best_score = float("inf")
    best_state = None
    patience = 0

    if resume and config.VISUAL_CHECKPOINT.exists():
        checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
        _validate_visual_provenance(checkpoint)
        _load_task_specific_state(model, checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", float("inf")))
        best_state = checkpoint.get("best_model")
        print(f"resume visual training from epoch {start_epoch}", flush=True)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.VISUAL_BATCH_SIZE)

    for epoch in range(start_epoch, int(epochs)):
        model.train()
        model.clip.eval()
        shuffled = list(train_indices)
        random.shuffle(shuffled)
        losses = []

        for offset in range(0, len(shuffled), batch_size):
            batch = torch.tensor(shuffled[offset : offset + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = _visual_forward_batch(
                model, cache, gallery, batch, device
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite visual loss at epoch {epoch + 1}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,
                float(config.GRAD_CLIP_NORM),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = _evaluate_visual_model(
            model, cache, gallery, val_indices, device
        )
        score = float(validation["SoftMS_MLE_m"])
        if score < best_score:
            best_score = score
            best_state = _task_specific_state_dict(model)
            patience = 0
        else:
            patience += 1

        torch.save(
            {
                "model": _task_specific_state_dict(model),
                "best_model": best_state,
                "model_format": "task_specific_only",
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "gallery": gallery,
                "epoch": epoch + 1,
                "best_score": best_score,
                "visual_train_routes": ["route_A"],
                "visual_validation_routes": ["route_A"],
                "visual_eval_routes": ["route_B", "route_C"],
                "previous_task_checkpoint_loaded": False,
                "backbone_source": config.BACKBONE_NAME,
                "task_specific_initialization": "random",
                "jitter_m": float(jitter_m),
            },
            config.VISUAL_CHECKPOINT,
        )
        print(
            f"visual epoch={epoch + 1:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"val_softms_mle={validation['SoftMS_MLE_m']:.3f}m "
            f"val_p90={validation['SoftMS_P90_m']:.3f}m "
            f"val_lsr10={validation['LSR@10_pct']:.2f}%",
            flush=True,
        )

        if patience >= int(config.VISUAL_EARLY_STOPPING_PATIENCE):
            print(
                "visual early stopping: Route A validation stopped improving",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Visual training did not produce a best checkpoint")

    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    checkpoint["model"] = best_state
    checkpoint["best_model"] = best_state
    torch.save(checkpoint, config.VISUAL_CHECKPOINT)
    print(
        f"best Route A visual validation SoftMS MLE={best_score:.3f}m",
        flush=True,
    )
