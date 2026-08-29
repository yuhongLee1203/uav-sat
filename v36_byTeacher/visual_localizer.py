from __future__ import annotations

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
    softms_mode_count: torch.Tensor
    softms_variance_xy: torch.Tensor | None = None


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
    """Return a fixed local grid around a supplied *predicted* center.

    The function has no access to frame labels/GT.  It simply snaps the supplied
    center to the permanent satellite lattice and retrieves the requested grid.
    """
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
    """Differentiable Mean-Shift with basin consolidation."""
    epsilon = 1e-8
    probability = torch.softmax(logits / max(float(tau), 1e-6), dim=1)
    centers = centers.float()
    modes = centers.clone()
    bandwidth = max(float(bandwidth_m), 1e-6)

    for _ in range(int(iterations)):
        distance_squared = torch.cdist(modes, centers).square()
        kernel = torch.exp(-distance_squared / (2.0 * bandwidth**2))
        weights = kernel * probability[:, None, :]
        modes = weights @ centers / weights.sum(dim=2, keepdim=True).clamp_min(epsilon)

    final_distance_squared = torch.cdist(modes, centers).square()
    density = (
        torch.exp(-final_distance_squared / (2.0 * bandwidth**2))
        * probability[:, None, :]
    ).sum(dim=2)
    seed_weights = torch.softmax(float(beta) * density, dim=1)

    merge_radius = float(getattr(config, "MEANSHIFT_MODE_MERGE_RADIUS_M", 2.0))
    merged_modes = []
    merged_density = []
    merged_weights = []
    merged_soft_xy = []
    merged_soft_support = []
    seed_count = int(modes.shape[1])

    for batch_index in range(int(modes.shape[0])):
        distance = torch.cdist(
            modes[batch_index].detach(), modes[batch_index].detach()
        )
        remaining = set(range(seed_count))
        groups = []
        while remaining:
            seed = min(remaining)
            frontier = [seed]
            group = set()
            while frontier:
                current = frontier.pop()
                if current in group:
                    continue
                group.add(current)
                neighbours = torch.nonzero(
                    distance[current] <= merge_radius, as_tuple=False
                ).flatten().tolist()
                frontier.extend(
                    n for n in neighbours if n in remaining and n not in group
                )
            remaining.difference_update(group)
            groups.append(sorted(group))

        row_modes = []
        row_density = []
        row_weights = []
        for group in groups:
            indices = torch.as_tensor(group, device=modes.device, dtype=torch.long)
            group_weights = seed_weights[batch_index, indices]
            total_weight = group_weights.sum().clamp_min(epsilon)
            row_weights.append(total_weight)
            row_modes.append(
                (group_weights[:, None] * modes[batch_index, indices]).sum(dim=0)
                / total_weight
            )
            row_density.append(
                (group_weights * density[batch_index, indices]).sum() / total_weight
            )

        active_modes = torch.stack(row_modes)
        active_density = torch.stack(row_density)
        active_weights = torch.stack(row_weights)
        merged_soft_xy.append((active_weights[:, None] * active_modes).sum(dim=0))
        merged_soft_support.append((active_weights * active_density).sum())

        padding = seed_count - len(groups)
        if padding:
            row_modes.extend([row_modes[0].new_zeros(2) for _ in range(padding)])
            row_density.extend([row_density[0].new_zeros(()) for _ in range(padding)])
            row_weights.extend([row_weights[0].new_zeros(()) for _ in range(padding)])
        merged_modes.append(torch.stack(row_modes))
        merged_density.append(torch.stack(row_density))
        merged_weights.append(torch.stack(row_weights))

    modes = torch.stack(merged_modes)
    density = torch.stack(merged_density)
    mode_weights = torch.stack(merged_weights)
    soft_xy = torch.stack(merged_soft_xy)
    soft_support = torch.stack(merged_soft_support)
    compact = (
        mode_weights * (modes - soft_xy.unsqueeze(1)).square().sum(dim=2)
    ).sum(dim=1).mean()
    return soft_xy, soft_support, modes, density, mode_weights, compact


def response_variance_xy(raw_prob, centers, softms_xy):
    """Visual-only XY variance used by Kalman; no learned/GT quantity enters."""
    diff = centers.float() - softms_xy[:, None, :]
    return (raw_prob[:, :, None] * diff.square()).sum(dim=1)


def _validate_visual_provenance(checkpoint):
    if checkpoint.get("visual_train_routes") != [config.TRAIN_ROUTE_NAME]:
        raise RuntimeError("visual checkpoint is not Route-A-only training")
    if checkpoint.get("visual_validation_routes") != [config.VALIDATION_ROUTE_NAME]:
        raise RuntimeError("visual checkpoint was not selected on Route B")
    if checkpoint.get("visual_test_routes") != [config.TEST_ROUTE_NAME]:
        raise RuntimeError("visual checkpoint metadata does not reserve Route C for test")
    if checkpoint.get("current_reference_used_as_search_prior") is not False:
        raise RuntimeError("visual checkpoint does not prove reference-free candidate search")
    if checkpoint.get("previous_task_checkpoint_loaded") is not False:
        raise RuntimeError("visual checkpoint loaded a previous task-specific model")
    if checkpoint.get("model_format") != "task_specific_only":
        raise RuntimeError("unsupported visual checkpoint format")


class FrozenVisualLocalizer:
    def __init__(self, device):
        if not config.VISUAL_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"visual checkpoint not found: {config.VISUAL_CHECKPOINT}. "
                "Run robust_tracker.py --mode train first."
            )
        checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
        _validate_visual_provenance(checkpoint)

        self.origin_lat = float(checkpoint["origin_lat"])
        self.origin_lon = float(checkpoint["origin_lon"])
        self.model = AllMapGeoCLIP().to(device)
        incompatible = self.model.load_state_dict(checkpoint["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        non_clip_missing = [
            key for key in incompatible.missing_keys if not key.startswith("clip.")
        ]
        if unexpected or non_clip_missing:
            raise RuntimeError(
                "visual checkpoint state mismatch: "
                f"unexpected={unexpected}, non_clip_missing={non_clip_missing}"
            )

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.gallery = {key: value.to(device) for key, value in checkpoint["gallery"].items()}
        self.pixel_index = build_pixel_index(self.gallery["pixel"])

    @torch.no_grad()
    def encode_uav_clip(self, uav):
        return self.model.encode_clip_image(uav.to(self.device, non_blocking=True))

    @torch.no_grad()
    def candidate_batch(self, uav_clip, center_xy, grid_size=None):
        """Full local-grid visual localization around a supplied predicted center."""
        grid_size = int(grid_size or config.MS2_GRID_SIZE)
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
        raw_prob = torch.softmax(raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1)
        raw_index = raw_logits.argmax(dim=1)
        raw_top1_xy = centers[
            torch.arange(centers.shape[0], device=self.device), raw_index
        ]
        softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )
        variance_xy = response_variance_xy(raw_prob, centers, softms_xy)
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
            softms_mode_count=(mode_weights > 0).sum(dim=1),
            softms_variance_xy=variance_xy,
        )

    @torch.no_grad()
    def candidate_contains_reference_anchor(self, indices, reference_xy):
        """Evaluation-only capture metric; never used to choose candidates."""
        candidate_xy = self.gallery["xy"][indices]
        nearest_distance = torch.linalg.norm(
            candidate_xy - reference_xy[:, None, :], dim=2
        ).min(dim=1).values
        return nearest_distance <= float(config.CANDIDATE_CAPTURE_RADIUS_M)

    # Backward-compatible name.  It remains evaluation-only.
    candidate_contains_gt_anchor = candidate_contains_reference_anchor


@dataclass
class VisualRouteCache:
    route_name: str
    uav_clip: torch.Tensor
    reference_xy: torch.Tensor
    frame_ids: torch.Tensor
    target_indices: torch.Tensor

    def __len__(self):
        return int(self.reference_xy.shape[0])


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
        key for key in incompatible.missing_keys if not key.startswith("clip.")
    ]
    if unexpected or non_clip_missing:
        raise RuntimeError(
            "visual state mismatch: "
            f"unexpected={unexpected}, non_clip_missing={non_clip_missing}"
        )


@torch.no_grad()
def _build_satellite_backbone_gallery(model, origin_lat, origin_lon, device):
    dataset = SatPatchGallery(origin_lat=origin_lat, origin_lon=origin_lon)
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


def _nearest_gallery_indices(gallery_xy, reference_xy, chunk_size=512):
    rows = []
    gallery_xy = gallery_xy.float().cpu()
    reference_xy = reference_xy.float().cpu()
    for start in range(0, len(reference_xy), int(chunk_size)):
        end = min(start + int(chunk_size), len(reference_xy))
        distance = torch.cdist(reference_xy[start:end], gallery_xy)
        rows.append(distance.argmin(dim=1))
    return torch.cat(rows).long()


@torch.no_grad()
def _build_visual_route_cache(
    model,
    route_name,
    route_root,
    origin_lat,
    origin_lon,
    gallery,
    device,
):
    dataset = RouteDataset(
        Path(route_root),
        train=False,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    uav_rows = []
    reference_rows = []
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
        reference_rows.append(torch.stack([item["xy"] for item in items]).float())
        frame_rows.extend(_parse_frame_id(item["frame_id"]) for item in items)
        if start == 0 or end == len(dataset) or (start // batch_size) % 20 == 0:
            print(f"{route_name} backbone cache: {end}/{len(dataset)}", flush=True)

    reference_xy = torch.cat(reference_rows)
    target_indices = _nearest_gallery_indices(gallery["xy"], reference_xy)
    return VisualRouteCache(
        route_name=route_name,
        uav_clip=torch.cat(uav_rows),
        reference_xy=reference_xy,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        target_indices=target_indices,
    )


def _visual_forward_batch(
    model,
    cache,
    gallery_clip,
    gallery_xy,
    batch_indices,
    device,
):
    batch_indices = batch_indices.long()
    uav_clip = cache.uav_clip[batch_indices].to(device).float()
    reference_xy = cache.reference_xy[batch_indices].to(device).float()
    target = cache.target_indices[batch_indices].to(device)

    # Full-gallery training: current reference positions are labels only.  They
    # never decide which satellite candidates the network sees.
    z_uav = model.encode_uav_from_clip(uav_clip)
    z_sat = model.encode_sat_from_clip(gallery_clip, gallery_xy)
    logits = model.logit_scale.exp().clamp(max=100.0) * (z_uav @ z_sat.t())

    ce = F.cross_entropy(
        logits,
        target,
        label_smoothing=float(config.VISUAL_LABEL_SMOOTHING),
    )
    probability = torch.softmax(logits, dim=1)
    weighted_xy = probability @ gallery_xy
    coordinate = F.smooth_l1_loss(weighted_xy, reference_xy)
    loss = ce + float(config.VISUAL_COORD_LOSS_WEIGHT) * coordinate
    return loss, logits, weighted_xy, reference_xy


@torch.no_grad()
def _evaluate_visual_model(model, cache, gallery_clip, gallery_xy, device):
    model.eval()
    z_sat = model.encode_sat_from_clip(gallery_clip, gallery_xy)
    top1_rows = []
    weighted_rows = []
    reference_rows = []
    batch_size = int(config.VISUAL_BATCH_SIZE)
    for offset in range(0, len(cache), batch_size):
        batch = torch.arange(offset, min(offset + batch_size, len(cache)))
        uav_clip = cache.uav_clip[batch].to(device).float()
        z_uav = model.encode_uav_from_clip(uav_clip)
        logits = model.logit_scale.exp().clamp(max=100.0) * (z_uav @ z_sat.t())
        probability = torch.softmax(logits, dim=1)
        top1_index = logits.argmax(dim=1)
        top1_rows.append(gallery_xy[top1_index].cpu())
        weighted_rows.append((probability @ gallery_xy).cpu())
        reference_rows.append(cache.reference_xy[batch].cpu())

    top1 = torch.cat(top1_rows)
    weighted = torch.cat(weighted_rows)
    reference = torch.cat(reference_rows)
    top1_error = torch.linalg.norm(top1 - reference, dim=1)
    weighted_error = torch.linalg.norm(weighted - reference, dim=1)
    return {
        "Top1_MLE_m": float(top1_error.mean().item()),
        "Top1_P90_m": float(torch.quantile(top1_error, 0.90).item()),
        "Weighted_MLE_m": float(weighted_error.mean().item()),
        "LSR@15_pct": float((top1_error <= 15.0).float().mean().item() * 100.0),
    }


def train_visual_retrieval_a_only(device, epochs, jitter_m=0.0, resume=False):
    """Train visual heads on Route A, select on Route B, reserve Route C for test.

    ``jitter_m`` is accepted only for backward CLI compatibility and is ignored.
    No GT/reference-centered local window is constructed during training.
    """
    del jitter_m
    _set_seed(int(config.SEED))
    model = AllMapGeoCLIP().to(device)

    route_a = RouteDataset(Path(config.ROUTE_ROOTS[0]), train=False)
    origin_lat = float(route_a.origin_lat)
    origin_lon = float(route_a.origin_lon)
    gallery = _build_satellite_backbone_gallery(model, origin_lat, origin_lon, device)

    train_cache = _build_visual_route_cache(
        model,
        config.TRAIN_ROUTE_NAME,
        config.ROUTE_ROOTS[config.ROUTE_NAMES.index(config.TRAIN_ROUTE_NAME)],
        origin_lat,
        origin_lon,
        gallery,
        device,
    )
    val_cache = _build_visual_route_cache(
        model,
        config.VALIDATION_ROUTE_NAME,
        config.ROUTE_ROOTS[config.ROUTE_NAMES.index(config.VALIDATION_ROUTE_NAME)],
        origin_lat,
        origin_lon,
        gallery,
        device,
    )

    gallery_clip = gallery["clip_feat"].to(device).float()
    gallery_xy = gallery["xy"].to(device).float()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", float("inf")))
        best_state = checkpoint.get("best_model")
        patience = int(checkpoint.get("patience", 0))
        print(f"resume visual training from epoch {start_epoch}", flush=True)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.VISUAL_BATCH_SIZE)

    for epoch in range(start_epoch, int(epochs)):
        model.train()
        model.clip.eval()
        shuffled = list(range(len(train_cache)))
        random.shuffle(shuffled)
        losses = []

        for offset in range(0, len(shuffled), batch_size):
            batch = torch.tensor(shuffled[offset : offset + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = _visual_forward_batch(
                model,
                train_cache,
                gallery_clip,
                gallery_xy,
                batch,
                device,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite visual loss at epoch {epoch + 1}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config.GRAD_CLIP_NORM))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = _evaluate_visual_model(
            model, val_cache, gallery_clip, gallery_xy, device
        )
        score = float(validation["Top1_MLE_m"])
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
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_score": best_score,
                "patience": patience,
                "visual_train_routes": [config.TRAIN_ROUTE_NAME],
                "visual_validation_routes": [config.VALIDATION_ROUTE_NAME],
                "visual_test_routes": [config.TEST_ROUTE_NAME],
                "previous_task_checkpoint_loaded": False,
                "current_reference_used_as_search_prior": False,
                "training_candidate_policy": "complete satellite gallery",
                "reference_role": "target index + loss/validation metric only",
                "backbone_source": config.BACKBONE_NAME,
                "task_specific_initialization": "random",
            },
            config.VISUAL_CHECKPOINT,
        )
        print(
            f"visual epoch={epoch + 1:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"B_top1_mle={validation['Top1_MLE_m']:.3f}m "
            f"B_p90={validation['Top1_P90_m']:.3f}m "
            f"B_lsr15={validation['LSR@15_pct']:.2f}%",
            flush=True,
        )

        if patience >= int(config.VISUAL_EARLY_STOPPING_PATIENCE):
            print("visual early stopping on Route B", flush=True)
            break

    if best_state is None:
        raise RuntimeError("visual training did not produce a checkpoint")

    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    checkpoint["model"] = best_state
    checkpoint["best_model"] = best_state
    torch.save(checkpoint, config.VISUAL_CHECKPOINT)
    print(f"best Route B visual Top1 MLE={best_score:.3f}m", flush=True)


# Clean name for new callers.
train_visual_retrieval = train_visual_retrieval_a_only
