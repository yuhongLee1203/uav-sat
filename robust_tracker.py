
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import RouteDataset
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import TemporalLatticeCRF

ARCHITECTURE_NAME = "ResidualSecondOrderTemporalLatticeCRF"


@dataclass
class RouteCache:
    name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    z_uav: torch.Tensor
    z_sat: torch.Tensor
    centers: torch.Tensor
    raw_logits: torch.Tensor
    raw_prob: torch.Tensor
    raw_top1: torch.Tensor
    hardms: torch.Tensor
    capture: torch.Tensor

    def __len__(self):
        return int(self.gt_xy.shape[0])


@dataclass
class SplitRange:
    start: int
    end: int

    @property
    def length(self):
        return max(0, self.end - self.start)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_frame_id(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def cache_dtype():
    return torch.float16 if config.FEATURE_CACHE_DTYPE == "float16" else torch.float32


def deterministic_jitter(length: int, route_index: int, maximum_m: float):
    if maximum_m <= 0:
        return torch.zeros(length, 2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.SEED) + 1009 * int(route_index))
    radius = torch.sqrt(torch.rand(length, 1, generator=generator)) * float(maximum_m)
    angle = torch.rand(length, 1, generator=generator) * (2.0 * math.pi)
    return torch.cat([radius * angle.cos(), radius * angle.sin()], dim=1)


@torch.no_grad()
def build_route_cache(
    root: Path,
    name: str,
    route_index: int,
    visual: FrozenVisualLocalizer,
    device: torch.device,
    jitter_m: float,
) -> RouteCache:
    dataset = RouteDataset(
        root,
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )
    jitter = deterministic_jitter(len(dataset), route_index, jitter_m)

    frame_id_rows: List[int] = []
    gt_rows: List[torch.Tensor] = []
    z_uav_rows: List[torch.Tensor] = []
    z_sat_rows: List[torch.Tensor] = []
    center_rows: List[torch.Tensor] = []
    logit_rows: List[torch.Tensor] = []
    probability_rows: List[torch.Tensor] = []
    top1_rows: List[torch.Tensor] = []
    hardms_rows: List[torch.Tensor] = []
    capture_rows: List[torch.Tensor] = []

    batch_size = int(config.EVAL_BATCH_SIZE)
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        gt_batch = torch.stack([item["xy"].float() for item in items])
        prior_batch = gt_batch + jitter[start:end]
        clip = visual.encode_uav_clip(uav)
        candidate = visual.candidate_batch(
            clip,
            prior_batch.to(device),
            grid_size=config.GRID_SIZE,
        )
        capture = visual.candidate_contains_gt_anchor(
            candidate.indices, gt_batch.to(device)
        )

        frame_id_rows.extend(parse_frame_id(item["frame_id"]) for item in items)
        gt_rows.extend(value.cpu() for value in gt_batch)
        z_uav_rows.append(candidate.z_uav.cpu().to(cache_dtype()))
        z_sat_rows.append(candidate.z_sat.cpu().to(cache_dtype()))
        center_rows.append(candidate.centers.cpu().float())
        logit_rows.append(candidate.raw_logits.cpu().float())
        probability_rows.append(candidate.raw_prob.cpu().float())
        top1_rows.append(candidate.raw_top1_xy.cpu().float())
        hardms_rows.append(candidate.hardms_xy.cpu().float())
        capture_rows.append(capture.cpu())

    cache = RouteCache(
        name=name,
        frame_ids=torch.tensor(frame_id_rows, dtype=torch.long),
        gt_xy=torch.stack(gt_rows).float(),
        z_uav=torch.cat(z_uav_rows),
        z_sat=torch.cat(z_sat_rows),
        centers=torch.cat(center_rows),
        raw_logits=torch.cat(logit_rows),
        raw_prob=torch.cat(probability_rows),
        raw_top1=torch.cat(top1_rows),
        hardms=torch.cat(hardms_rows),
        capture=torch.cat(capture_rows),
    )
    print(
        f"cached {name}: frames={len(cache)} "
        f"capture={cache.capture.float().mean().item() * 100:.2f}%",
        flush=True,
    )
    return cache


def contiguous_splits(length: int) -> Dict[str, SplitRange]:
    guard = int(config.SPLIT_GUARD_FRAMES)
    train_end = int(length * float(config.TRAIN_FRACTION))
    val_end = int(length * (float(config.TRAIN_FRACTION) + float(config.VAL_FRACTION)))
    train = SplitRange(0, max(0, train_end - guard))
    val_start = min(length, train_end + guard)
    val = SplitRange(val_start, max(val_start, val_end - guard))
    test_start = min(length, val_end + guard)
    return {
        "train": train,
        "val": val,
        "test": SplitRange(test_start, length),
        "all": SplitRange(0, length),
    }


def make_windows(caches: Sequence[RouteCache], split_name: str):
    window = int(config.TEMPORAL_WINDOW)
    stride = int(config.WINDOW_STRIDE)
    rows: List[Tuple[int, int]] = []
    for route_index, cache in enumerate(caches):
        segment = contiguous_splits(len(cache))[split_name]
        for start in range(segment.start, segment.end - window + 1, stride):
            if bool(cache.capture[start : start + window].all()):
                rows.append((route_index, start))
    return rows


def nearest_candidate_target(centers: torch.Tensor, gt_xy: torch.Tensor):
    return (centers - gt_xy[:, :, None, :]).square().sum(dim=-1).argmin(dim=-1)


def gather_batch(caches, windows, device):
    window = int(config.TEMPORAL_WINDOW)
    z_uav = torch.stack([
        caches[r].z_uav[s : s + window] for r, s in windows
    ]).to(device).float()
    z_sat = torch.stack([
        caches[r].z_sat[s : s + window] for r, s in windows
    ]).to(device).float()
    centers = torch.stack([
        caches[r].centers[s : s + window] for r, s in windows
    ]).to(device)
    raw_logits = torch.stack([
        caches[r].raw_logits[s : s + window] for r, s in windows
    ]).to(device)
    raw_prob = torch.stack([
        caches[r].raw_prob[s : s + window] for r, s in windows
    ]).to(device)
    hardms = torch.stack([
        caches[r].hardms[s : s + window] for r, s in windows
    ]).to(device)
    frame_ids = torch.stack([
        caches[r].frame_ids[s : s + window] for r, s in windows
    ]).to(device)
    gt = torch.stack([
        caches[r].gt_xy[s : s + window] for r, s in windows
    ]).to(device)
    target = nearest_candidate_target(centers, gt)
    return {
        "z_uav": z_uav,
        "z_sat": z_sat,
        "centers": centers,
        "raw_logits": raw_logits,
        "raw_prob": raw_prob,
        "hardms": hardms,
        "frame_ids": frame_ids,
        "gt": gt,
        "target": target,
    }


def model_loss(model, batch):
    output = model(
        batch["z_uav"],
        batch["z_sat"],
        batch["raw_logits"],
        batch["raw_prob"],
        batch["centers"],
        batch["frame_ids"],
        batch["hardms"],
        batch["target"],
    )
    gt_last = batch["gt"][:, -1]
    final_coord = F.smooth_l1_loss(output.final_xy, gt_last)
    path_coord = F.smooth_l1_loss(output.path_expectation, gt_last)
    loss = (
        float(config.LOSS_CRF) * output.crf_nll
        + float(config.LOSS_FINAL_COORD) * final_coord
        + float(config.LOSS_PATH_COORD) * path_coord
    )
    return loss, output


def metric_block(prediction, gt):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    error = np.linalg.norm(prediction - gt, axis=1)
    if len(prediction) > 1:
        predicted_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)
        rpe = np.linalg.norm(predicted_step - gt_step, axis=1)
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = float(np.percentile(gt_step_length, 99)) + float(
            config.JUMP_TOLERANCE_M
        )
        jump_rate = float(
            (np.linalg.norm(predicted_step, axis=1) > jump_threshold).mean() * 100
        )
        stationary = gt_step_length < 1e-3
        stationary_drift = np.linalg.norm(predicted_step, axis=1)[stationary]
        stationary_p90 = (
            float(np.percentile(stationary_drift, 90))
            if len(stationary_drift) else 0.0
        )
        path_ratio = float(
            np.linalg.norm(predicted_step, axis=1).sum()
            / max(gt_step_length.sum(), 1e-8)
        )
    else:
        rpe = np.zeros(1)
        jump_rate = 0.0
        stationary_p90 = 0.0
        path_ratio = 0.0
        jump_threshold = 0.0
    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "ATE_RMSE_m": float(np.sqrt(np.mean(error ** 2))),
        "LSR@5_pct": float((error <= 5).mean() * 100),
        "LSR@10_pct": float((error <= 10).mean() * 100),
        "LSR@15_pct": float((error <= 15).mean() * 100),
        "LSR@20_pct": float((error <= 20).mean() * 100),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": jump_rate,
        "JumpThreshold_m": jump_threshold,
        "StationaryDriftP90_m": stationary_p90,
        "PathLengthRatio": path_ratio,
        "MaxLE_m": float(error.max()),
    }


@torch.no_grad()
def predict_split(model, caches, split_name, device, save_rows=False):
    model.eval()
    route_outputs = []
    window = int(config.TEMPORAL_WINDOW)
    for route_index, cache in enumerate(caches):
        segment = contiguous_splits(len(cache))[split_name]
        rows = []
        for index in range(segment.start + window - 1, segment.end):
            start = index - window + 1
            # Evaluation must include every frame.  Capture is reported as a
            # diagnostic, not used to hide difficult windows.  Training still
            # uses only fully representable windows in make_windows().
            batch = gather_batch(caches, [(route_index, start)], device)
            output = model(
                batch["z_uav"], batch["z_sat"], batch["raw_logits"],
                batch["raw_prob"], batch["centers"], batch["frame_ids"],
                batch["hardms"], target_index=None,
            )
            rows.append({
                "frame_id": int(cache.frame_ids[index]),
                "gt": cache.gt_xy[index].tolist(),
                "raw_top1": cache.raw_top1[index].tolist(),
                "hardms": cache.hardms[index].tolist(),
                "path": output.path_expectation[0].cpu().tolist(),
                "final": output.final_xy[0].cpu().tolist(),
                "gate": float(output.correction_gate[0].item()),
                "path_entropy": float(output.path_entropy[0].item()),
                "emission_entropy": float(output.emission_entropy[0].item()),
                "capture": bool(cache.capture[index]),
            })
        if not rows:
            continue
        gt = [row["gt"] for row in rows]
        summary = {
            "route": cache.name,
            "split": split_name,
            "RawTop1": metric_block([row["raw_top1"] for row in rows], gt),
            "FixedHardMS": metric_block([row["hardms"] for row in rows], gt),
            "TemporalPathExpectation": metric_block([row["path"] for row in rows], gt),
            "RTL_CRF": metric_block([row["final"] for row in rows], gt),
            "CandidateCaptureRate_pct": float(
                cache.capture[segment.start:segment.end].float().mean().item() * 100
            ),
            "MeanCorrectionGate": float(np.mean([row["gate"] for row in rows])),
            "MeanPathEntropy": float(np.mean([row["path_entropy"] for row in rows])),
        }
        route_outputs.append((cache.name, summary, rows))

        if save_rows:
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = config.OUTPUT_DIR / f"{cache.name}_robust_frames.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "frame_id", "gt_x", "gt_y", "raw_top1_x", "raw_top1_y",
                    "hardms_x", "hardms_y", "path_x", "path_y",
                    "temporal_x", "temporal_y", "correction_gate",
                    "path_entropy", "emission_entropy", "candidate_capture",
                ])
                for row in rows:
                    writer.writerow([
                        row["frame_id"], *row["gt"], *row["raw_top1"],
                        *row["hardms"], *row["path"], *row["final"],
                        row["gate"], row["path_entropy"],
                        row["emission_entropy"], int(row["capture"]),
                    ])
    return route_outputs


def validation_objective(summary):
    metric = summary["RTL_CRF"]
    return (
        metric["MLE_m"]
        + float(config.VAL_RPE_WEIGHT) * metric["RPE_m"]
        + float(config.VAL_JUMP_WEIGHT) * metric["JumpRate_pct"]
        + float(config.VAL_STATIONARY_WEIGHT) * metric["StationaryDriftP90_m"]
    )


def train_model(model, caches, device, epochs, resume=False):
    windows = make_windows(caches, "train")
    if not windows:
        raise RuntimeError("No valid training windows; check candidate capture")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.LR), weight_decay=float(config.WEIGHT_DECAY)
    )
    start_epoch = 0
    best_score = float("inf")
    best_state = None
    patience = 0

    if resume and config.TEMPORAL_CHECKPOINT.exists():
        checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
        if checkpoint.get("architecture") == "ResidualSecondOrderTemporalLatticeCRF":
            model.load_state_dict(checkpoint["model"], strict=True)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_score = float(checkpoint.get("best_score", float("inf")))
            best_state = checkpoint.get("best_model")
            print(f"resume from epoch {start_epoch}", flush=True)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.TRAIN_BATCH_SIZE)
    for epoch in range(start_epoch, int(epochs)):
        model.train()
        random.shuffle(windows)
        losses = []
        for offset in range(0, len(windows), batch_size):
            selected = windows[offset : offset + batch_size]
            batch = gather_batch(caches, selected, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.GRAD_CLIP_NORM))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = predict_split(model, caches, "val", device, save_rows=False)
        if not validation:
            raise RuntimeError("No validation predictions")
        scores = [validation_objective(summary) for _, summary, _ in validation]
        score = float(np.mean(scores))
        val_mle = float(np.mean([summary["RTL_CRF"]["MLE_m"] for _, summary, _ in validation]))
        val_p90 = float(np.mean([summary["RTL_CRF"]["P90_m"] for _, summary, _ in validation]))
        val_jump = float(np.mean([summary["RTL_CRF"]["JumpRate_pct"] for _, summary, _ in validation]))

        improved = score < best_score
        if improved:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        torch.save({
            "model": model.state_dict(),
            "best_model": best_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_score": best_score,
            "architecture": "ResidualSecondOrderTemporalLatticeCRF",
            "jitter_m": float(config.LOCAL_PRIOR_JITTER_M),
        }, config.TEMPORAL_CHECKPOINT)
        print(
            f"epoch={epoch + 1:03d}/{epochs} loss={np.mean(losses):.5f} "
            f"val_mle={val_mle:.3f}m val_p90={val_p90:.3f}m "
            f"val_jump={val_jump:.2f}% score={score:.3f}",
            flush=True,
        )
        if patience >= int(config.EARLY_STOPPING_PATIENCE):
            print("early stopping: validation objective stopped improving", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
        checkpoint["model"] = best_state
        torch.save(checkpoint, config.TEMPORAL_CHECKPOINT)
    print(f"best validation objective={best_score:.3f}", flush=True)


def load_checkpoint(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {config.TEMPORAL_CHECKPOINT}; run train_eval"
        )
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
    if checkpoint.get("architecture") != "ResidualSecondOrderTemporalLatticeCRF":
        raise RuntimeError(
            "existing checkpoint belongs to the failed TMCR architecture; retrain this version"
        )
    state = checkpoint.get("best_model") or checkpoint["model"]
    model.load_state_dict(state, strict=True)
    return checkpoint


def route_catalog():
    return {
        name: (index, Path(root))
        for index, (name, root) in enumerate(
            zip(config.ROUTE_NAMES, config.ROUTE_ROOTS)
        )
    }


def route_record(name: str):
    catalog = route_catalog()
    if name not in catalog:
        raise ValueError(
            f"unknown route {name}; valid routes={list(catalog)}"
        )
    index, root = catalog[name]
    return index, root, name


def remove_stale_outputs():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        config.VISUAL_CHECKPOINT,
        config.TEMPORAL_CHECKPOINT,
        config.OUTPUT_DIR / "robust_tracker_summary.json",
        config.OUTPUT_DIR / "route_B_robust_frames.csv",
        config.OUTPUT_DIR / "route_C_robust_frames.csv",
    ):
        if path.exists():
            path.unlink()


def write_temporal_provenance():
    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )
    checkpoint.update(
        {
            "architecture": ARCHITECTURE_NAME,
            "temporal_train_routes": ["route_A"],
            "temporal_validation_routes": ["route_A"],
            "temporal_eval_routes": ["route_B", "route_C"],
            "visual_checkpoint": str(config.VISUAL_CHECKPOINT),
            "visual_train_routes": ["route_A"],
            "previous_task_checkpoint_loaded": False,
        }
    )
    torch.save(checkpoint, config.TEMPORAL_CHECKPOINT)


def validate_temporal_provenance(checkpoint):
    if checkpoint.get("temporal_train_routes") != ["route_A"]:
        raise RuntimeError(
            "Temporal checkpoint is not Route-A-only: "
            f"{checkpoint.get('temporal_train_routes')}"
        )
    if checkpoint.get("visual_train_routes") != ["route_A"]:
        raise RuntimeError(
            "Temporal checkpoint was not built from an A-only visual model"
        )
    if checkpoint.get("previous_task_checkpoint_loaded") is not False:
        raise RuntimeError(
            "Temporal checkpoint provenance does not prove a fresh task model"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Strict experiment: train retrieval heads and RTL-CRF only on "
            "Route A, then evaluate unseen Route B and Route C."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_eval"),
        default="train_eval",
    )
    parser.add_argument(
        "--visual-epochs",
        type=int,
        default=config.VISUAL_EPOCHS,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="RTL-CRF epochs",
    )
    parser.add_argument(
        "--eval-split",
        choices=("test", "all"),
        default="all",
        help=(
            "all evaluates the complete unseen B/C routes; test evaluates "
            "only their last 15 percent"
        ),
    )
    parser.add_argument(
        "--jitter-m",
        type=float,
        default=config.LOCAL_PRIOR_JITTER_M,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume only the strict A-only checkpoints in "
            "outputs/strict_train_A_test_BC"
        ),
    )
    args = parser.parse_args()

    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)
    set_seed(int(config.SEED))
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    print("=" * 76, flush=True)
    print("STRICT TRAIN-A / TEST-B+C EXPERIMENT", flush=True)
    print("visual train routes:      ['route_A']", flush=True)
    print("visual validation routes: ['route_A']", flush=True)
    print("temporal train routes:    ['route_A']", flush=True)
    print("temporal validation:      ['route_A']", flush=True)
    print("evaluation routes:        ['route_B', 'route_C']", flush=True)
    print(f"evaluation split:         {args.eval_split}", flush=True)
    print("previous task checkpoint: DISALLOWED", flush=True)
    print(f"public backbone:          {config.BACKBONE_NAME}", flush=True)
    print(f"output directory:         {config.OUTPUT_DIR}", flush=True)
    print("=" * 76, flush=True)

    if args.mode in ("train", "train_eval") and not args.resume:
        # This guarantees that neither the old visual best.pt nor an older
        # temporal checkpoint can influence the new experiment.
        remove_stale_outputs()

    if args.mode in ("train", "train_eval"):
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume),
        )
    elif not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing strict A-only visual checkpoint: {config.VISUAL_CHECKPOINT}"
        )

    # FrozenVisualLocalizer rejects any checkpoint whose provenance is not
    # exactly Route-A-only with no prior task checkpoint.
    visual = FrozenVisualLocalizer(device)

    model = TemporalLatticeCRF().to(device)

    if args.mode in ("train", "train_eval"):
        a_index, a_root, a_name = route_record("route_A")
        train_cache = build_route_cache(
            a_root,
            a_name,
            a_index,
            visual,
            device,
            args.jitter_m,
        )
        train_capture = float(
            train_cache.capture.float().mean().item()
        )
        if train_capture < float(config.MIN_TRAIN_CAPTURE_RATE):
            raise RuntimeError(
                "Route A candidate capture is below "
                f"{100.0 * float(config.MIN_TRAIN_CAPTURE_RATE):.1f}%"
            )

        # Only Route A is passed to train_model. Its validation call also sees
        # only this same Route A cache.
        train_model(
            model,
            [train_cache],
            device,
            int(args.epochs),
            resume=bool(args.resume),
        )
        write_temporal_provenance()

    if args.mode in ("eval", "train_eval"):
        if args.mode == "eval":
            checkpoint = load_checkpoint(model, device)
            validate_temporal_provenance(checkpoint)

        # B/C caches are deliberately constructed only after all optimization
        # has finished. They cannot affect visual or temporal checkpoint choice.
        eval_caches = []
        for route_name in ("route_B", "route_C"):
            route_index, root, name = route_record(route_name)
            cache = build_route_cache(
                root,
                name,
                route_index,
                visual,
                device,
                args.jitter_m,
            )
            eval_caches.append(cache)

        outputs = predict_split(
            model,
            eval_caches,
            args.eval_split,
            device,
            save_rows=True,
        )

        visual_checkpoint = torch.load(
            config.VISUAL_CHECKPOINT,
            map_location="cpu",
        )
        temporal_checkpoint = torch.load(
            config.TEMPORAL_CHECKPOINT,
            map_location="cpu",
        )
        validate_temporal_provenance(temporal_checkpoint)

        summary = {
            "method": ARCHITECTURE_NAME,
            "protocol": (
                "Public frozen MobileCLIP backbone; randomly initialized "
                "retrieval heads trained/validated only on Route A; randomly "
                "initialized RTL-CRF trained/validated only on Route A; "
                "Route B/C used only after training for evaluation; "
                "GT+jitter controlled local prior"
            ),
            "visual_train_routes": visual_checkpoint["visual_train_routes"],
            "visual_validation_routes": visual_checkpoint[
                "visual_validation_routes"
            ],
            "temporal_train_routes": temporal_checkpoint[
                "temporal_train_routes"
            ],
            "temporal_validation_routes": temporal_checkpoint[
                "temporal_validation_routes"
            ],
            "eval_routes": ["route_B", "route_C"],
            "eval_split": args.eval_split,
            "previous_task_checkpoint_loaded": False,
            "backbone_source": config.BACKBONE_NAME,
            "jitter_m": float(args.jitter_m),
            "visual_checkpoint": str(config.VISUAL_CHECKPOINT),
            "temporal_checkpoint": str(config.TEMPORAL_CHECKPOINT),
            "routes": {
                name: route_summary
                for name, route_summary, _ in outputs
            },
        }

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = config.OUTPUT_DIR / "robust_tracker_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
