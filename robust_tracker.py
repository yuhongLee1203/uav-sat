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
from visual_localizer import FrozenVisualLocalizer
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
    return (
        torch.float16
        if config.FEATURE_CACHE_DTYPE == "float16"
        else torch.float32
    )


def deterministic_jitter(length: int, route_index: int, maximum_m: float):
    if maximum_m <= 0:
        return torch.zeros(length, 2)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(config.SEED) + 1009 * int(route_index)
    )

    radius = (
        torch.sqrt(torch.rand(length, 1, generator=generator))
        * float(maximum_m)
    )
    angle = (
        torch.rand(length, 1, generator=generator)
        * (2.0 * math.pi)
    )

    return torch.cat(
        [
            radius * angle.cos(),
            radius * angle.sin(),
        ],
        dim=1,
    )


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

    jitter = deterministic_jitter(
        len(dataset),
        route_index,
        jitter_m,
    )

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
        items = [
            dataset[index]
            for index in range(start, end)
        ]

        uav = torch.stack(
            [item["uav"] for item in items]
        ).to(device)

        gt_batch = torch.stack(
            [item["xy"].float() for item in items]
        )

        prior_batch = gt_batch + jitter[start:end]

        clip = visual.encode_uav_clip(uav)

        candidate = visual.candidate_batch(
            clip,
            prior_batch.to(device),
            grid_size=config.GRID_SIZE,
        )

        capture = visual.candidate_contains_gt_anchor(
            candidate.indices,
            gt_batch.to(device),
        )

        frame_id_rows.extend(
            parse_frame_id(item["frame_id"])
            for item in items
        )
        gt_rows.extend(
            value.cpu()
            for value in gt_batch
        )

        z_uav_rows.append(
            candidate.z_uav.cpu().to(cache_dtype())
        )
        z_sat_rows.append(
            candidate.z_sat.cpu().to(cache_dtype())
        )
        center_rows.append(
            candidate.centers.cpu().float()
        )
        logit_rows.append(
            candidate.raw_logits.cpu().float()
        )
        probability_rows.append(
            candidate.raw_prob.cpu().float()
        )
        top1_rows.append(
            candidate.raw_top1_xy.cpu().float()
        )
        hardms_rows.append(
            candidate.hardms_xy.cpu().float()
        )
        capture_rows.append(capture.cpu())

    cache = RouteCache(
        name=name,
        frame_ids=torch.tensor(
            frame_id_rows,
            dtype=torch.long,
        ),
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
        f"capture="
        f"{cache.capture.float().mean().item() * 100:.2f}%",
        flush=True,
    )

    return cache


def contiguous_splits(length: int) -> Dict[str, SplitRange]:
    guard = int(config.SPLIT_GUARD_FRAMES)

    train_end = int(
        length * float(config.TRAIN_FRACTION)
    )
    val_end = int(
        length
        * (
            float(config.TRAIN_FRACTION)
            + float(config.VAL_FRACTION)
        )
    )

    train = SplitRange(
        0,
        max(0, train_end - guard),
    )

    val_start = min(
        length,
        train_end + guard,
    )
    val = SplitRange(
        val_start,
        max(val_start, val_end - guard),
    )

    test_start = min(
        length,
        val_end + guard,
    )

    return {
        "train": train,
        "val": val,
        "test": SplitRange(test_start, length),
        "all": SplitRange(0, length),
    }


def make_windows(
    caches: Sequence[RouteCache],
    split_name: str,
):
    window = int(config.TEMPORAL_WINDOW)
    stride = int(config.WINDOW_STRIDE)

    rows: List[Tuple[int, int]] = []

    for route_index, cache in enumerate(caches):
        segment = contiguous_splits(
            len(cache)
        )[split_name]

        for start in range(
            segment.start,
            segment.end - window + 1,
            stride,
        ):
            if bool(
                cache.capture[
                    start : start + window
                ].all()
            ):
                rows.append((route_index, start))

    return rows


def nearest_candidate_target(
    centers: torch.Tensor,
    gt_xy: torch.Tensor,
):
    return (
        centers - gt_xy[:, :, None, :]
    ).square().sum(dim=-1).argmin(dim=-1)


def gather_batch(caches, windows, device):
    window = int(config.TEMPORAL_WINDOW)

    z_uav = torch.stack(
        [
            caches[route_index].z_uav[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device).float()

    z_sat = torch.stack(
        [
            caches[route_index].z_sat[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device).float()

    centers = torch.stack(
        [
            caches[route_index].centers[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device)

    raw_logits = torch.stack(
        [
            caches[route_index].raw_logits[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device)

    raw_prob = torch.stack(
        [
            caches[route_index].raw_prob[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device)

    hardms = torch.stack(
        [
            caches[route_index].hardms[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device)

    frame_ids = torch.stack(
        [
            caches[route_index].frame_ids[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device)

    gt = torch.stack(
        [
            caches[route_index].gt_xy[
                start : start + window
            ]
            for route_index, start in windows
        ]
    ).to(device)

    target = nearest_candidate_target(
        centers,
        gt,
    )

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

    final_coord = F.smooth_l1_loss(
        output.final_xy,
        gt_last,
    )
    path_coord = F.smooth_l1_loss(
        output.path_expectation,
        gt_last,
    )

    loss = (
        float(config.LOSS_CRF) * output.crf_nll
        + float(config.LOSS_FINAL_COORD)
        * final_coord
        + float(config.LOSS_PATH_COORD)
        * path_coord
    )

    return loss, output


def metric_block(prediction, gt):
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    error = np.linalg.norm(
        prediction - gt,
        axis=1,
    )

    if len(prediction) > 1:
        predicted_step = np.diff(
            prediction,
            axis=0,
        )
        gt_step = np.diff(
            gt,
            axis=0,
        )

        rpe = np.linalg.norm(
            predicted_step - gt_step,
            axis=1,
        )
        gt_step_length = np.linalg.norm(
            gt_step,
            axis=1,
        )

        jump_threshold = float(
            np.percentile(
                gt_step_length,
                99,
            )
        ) + float(config.JUMP_TOLERANCE_M)

        jump_rate = float(
            (
                np.linalg.norm(
                    predicted_step,
                    axis=1,
                )
                > jump_threshold
            ).mean()
            * 100
        )

        stationary = gt_step_length < 1e-3
        stationary_drift = np.linalg.norm(
            predicted_step,
            axis=1,
        )[stationary]

        stationary_p90 = (
            float(
                np.percentile(
                    stationary_drift,
                    90,
                )
            )
            if len(stationary_drift)
            else 0.0
        )

        path_ratio = float(
            np.linalg.norm(
                predicted_step,
                axis=1,
            ).sum()
            / max(
                gt_step_length.sum(),
                1e-8,
            )
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
        "ATE_RMSE_m": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "LSR@5_pct": float(
            (error <= 5).mean() * 100
        ),
        "LSR@10_pct": float(
            (error <= 10).mean() * 100
        ),
        "LSR@15_pct": float(
            (error <= 15).mean() * 100
        ),
        "LSR@20_pct": float(
            (error <= 20).mean() * 100
        ),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": jump_rate,
        "JumpThreshold_m": jump_threshold,
        "StationaryDriftP90_m": stationary_p90,
        "PathLengthRatio": path_ratio,
        "MaxLE_m": float(error.max()),
    }


@torch.no_grad()
def predict_split(
    model,
    caches,
    split_name,
    device,
    save_rows=False,
):
    model.eval()

    route_outputs = []
    window = int(config.TEMPORAL_WINDOW)

    for route_index, cache in enumerate(caches):
        segment = contiguous_splits(
            len(cache)
        )[split_name]

        rows = []

        for index in range(
            segment.start + window - 1,
            segment.end,
        ):
            start = index - window + 1

            # Evaluation includes every frame.
            # Capture is only reported as a diagnostic.
            batch = gather_batch(
                caches,
                [(route_index, start)],
                device,
            )

            output = model(
                batch["z_uav"],
                batch["z_sat"],
                batch["raw_logits"],
                batch["raw_prob"],
                batch["centers"],
                batch["frame_ids"],
                batch["hardms"],
                target_index=None,
            )

            rows.append(
                {
                    "frame_id": int(
                        cache.frame_ids[index]
                    ),
                    "gt": cache.gt_xy[
                        index
                    ].tolist(),
                    "raw_top1": cache.raw_top1[
                        index
                    ].tolist(),
                    "hardms": cache.hardms[
                        index
                    ].tolist(),
                    "path": output.path_expectation[
                        0
                    ].cpu().tolist(),
                    "final": output.final_xy[
                        0
                    ].cpu().tolist(),
                    "gate": float(
                        output.correction_gate[
                            0
                        ].item()
                    ),
                    "path_entropy": float(
                        output.path_entropy[
                            0
                        ].item()
                    ),
                    "emission_entropy": float(
                        output.emission_entropy[
                            0
                        ].item()
                    ),
                    "capture": bool(
                        cache.capture[index]
                    ),
                }
            )

        if not rows:
            continue

        gt = [
            row["gt"]
            for row in rows
        ]

        summary = {
            "route": cache.name,
            "split": split_name,
            "RawTop1": metric_block(
                [
                    row["raw_top1"]
                    for row in rows
                ],
                gt,
            ),
            "FixedHardMS": metric_block(
                [
                    row["hardms"]
                    for row in rows
                ],
                gt,
            ),
            "TemporalPathExpectation": metric_block(
                [
                    row["path"]
                    for row in rows
                ],
                gt,
            ),
            "RTL_CRF": metric_block(
                [
                    row["final"]
                    for row in rows
                ],
                gt,
            ),
            "CandidateCaptureRate_pct": float(
                cache.capture[
                    segment.start : segment.end
                ].float().mean().item()
                * 100
            ),
            "MeanCorrectionGate": float(
                np.mean(
                    [
                        row["gate"]
                        for row in rows
                    ]
                )
            ),
            "MeanPathEntropy": float(
                np.mean(
                    [
                        row["path_entropy"]
                        for row in rows
                    ]
                )
            ),
        }

        route_outputs.append(
            (
                cache.name,
                summary,
                rows,
            )
        )

        if save_rows:
            config.OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            csv_path = (
                config.OUTPUT_DIR
                / f"{cache.name}_robust_frames.csv"
            )

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)

                writer.writerow(
                    [
                        "frame_id",
                        "gt_x",
                        "gt_y",
                        "raw_top1_x",
                        "raw_top1_y",
                        "hardms_x",
                        "hardms_y",
                        "path_x",
                        "path_y",
                        "temporal_x",
                        "temporal_y",
                        "correction_gate",
                        "path_entropy",
                        "emission_entropy",
                        "candidate_capture",
                    ]
                )

                for row in rows:
                    writer.writerow(
                        [
                            row["frame_id"],
                            *row["gt"],
                            *row["raw_top1"],
                            *row["hardms"],
                            *row["path"],
                            *row["final"],
                            row["gate"],
                            row["path_entropy"],
                            row["emission_entropy"],
                            int(row["capture"]),
                        ]
                    )

    return route_outputs


def validation_objective(summary):
    metric = summary["RTL_CRF"]

    return (
        metric["MLE_m"]
        + float(config.VAL_RPE_WEIGHT)
        * metric["RPE_m"]
        + float(config.VAL_JUMP_WEIGHT)
        * metric["JumpRate_pct"]
        + float(config.VAL_STATIONARY_WEIGHT)
        * metric["StationaryDriftP90_m"]
    )


def train_model(
    model,
    caches,
    device,
    epochs,
    resume=False,
):
    windows = make_windows(
        caches,
        "train",
    )

    if not windows:
        raise RuntimeError(
            "No valid training windows; "
            "check candidate capture"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LR),
        weight_decay=float(
            config.WEIGHT_DECAY
        ),
    )

    start_epoch = 0
    best_score = float("inf")
    best_state = None
    patience = 0

    if (
        resume
        and config.TEMPORAL_CHECKPOINT.exists()
    ):
        checkpoint = torch.load(
            config.TEMPORAL_CHECKPOINT,
            map_location=device,
        )

        if (
            checkpoint.get("architecture")
            == ARCHITECTURE_NAME
        ):
            model.load_state_dict(
                checkpoint["model"],
                strict=True,
            )

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )

            start_epoch = int(
                checkpoint.get("epoch", 0)
            )
            best_score = float(
                checkpoint.get(
                    "best_score",
                    float("inf"),
                )
            )
            best_state = checkpoint.get(
                "best_model"
            )

            print(
                f"resume from epoch {start_epoch}",
                flush=True,
            )

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_size = int(
        config.TRAIN_BATCH_SIZE
    )

    for epoch in range(
        start_epoch,
        int(epochs),
    ):
        model.train()
        random.shuffle(windows)

        losses = []

        for offset in range(
            0,
            len(windows),
            batch_size,
        ):
            selected = windows[
                offset : offset + batch_size
            ]

            batch = gather_batch(
                caches,
                selected,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, _ = model_loss(
                model,
                batch,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite loss at "
                    f"epoch {epoch + 1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    config.GRAD_CLIP_NORM
                ),
            )

            optimizer.step()
            losses.append(
                float(loss.detach().cpu())
            )

        # Only the supplied training caches are used
        # for validation. In this protocol that is Route A.
        validation = predict_split(
            model,
            caches,
            "val",
            device,
            save_rows=False,
        )

        if not validation:
            raise RuntimeError(
                "No validation predictions"
            )

        scores = [
            validation_objective(summary)
            for _, summary, _ in validation
        ]

        score = float(
            np.mean(scores)
        )

        val_mle = float(
            np.mean(
                [
                    summary["RTL_CRF"]["MLE_m"]
                    for _, summary, _ in validation
                ]
            )
        )

        val_p90 = float(
            np.mean(
                [
                    summary["RTL_CRF"]["P90_m"]
                    for _, summary, _ in validation
                ]
            )
        )

        val_jump = float(
            np.mean(
                [
                    summary["RTL_CRF"][
                        "JumpRate_pct"
                    ]
                    for _, summary, _ in validation
                ]
            )
        )

        improved = score < best_score

        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        torch.save(
            {
                "model": model.state_dict(),
                "best_model": best_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_score": best_score,
                "architecture": ARCHITECTURE_NAME,
                "jitter_m": float(
                    config.LOCAL_PRIOR_JITTER_M
                ),
            },
            config.TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"val_mle={val_mle:.3f}m "
            f"val_p90={val_p90:.3f}m "
            f"val_jump={val_jump:.2f}% "
            f"score={score:.3f}",
            flush=True,
        )

        if (
            patience
            >= int(
                config.EARLY_STOPPING_PATIENCE
            )
        ):
            print(
                "early stopping: validation "
                "objective stopped improving",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state,
            strict=True,
        )

        checkpoint = torch.load(
            config.TEMPORAL_CHECKPOINT,
            map_location="cpu",
        )
        checkpoint["model"] = best_state

        torch.save(
            checkpoint,
            config.TEMPORAL_CHECKPOINT,
        )

    print(
        f"best validation objective="
        f"{best_score:.3f}",
        flush=True,
    )


def load_checkpoint(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "checkpoint not found: "
            f"{config.TEMPORAL_CHECKPOINT}; "
            "run train_eval first"
        )

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location=device,
    )

    if (
        checkpoint.get("architecture")
        != ARCHITECTURE_NAME
    ):
        raise RuntimeError(
            "existing checkpoint belongs to "
            "another architecture; retrain "
            "this RTL-CRF version"
        )

    state = (
        checkpoint.get("best_model")
        or checkpoint["model"]
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    return checkpoint


def route_catalog() -> Dict[
    str,
    Tuple[int, Path],
]:
    if (
        len(config.ROUTE_NAMES)
        != len(config.ROUTE_ROOTS)
    ):
        raise RuntimeError(
            "config.ROUTE_NAMES and "
            "config.ROUTE_ROOTS have "
            "different lengths"
        )

    return {
        name: (
            index,
            Path(root),
        )
        for index, (name, root)
        in enumerate(
            zip(
                config.ROUTE_NAMES,
                config.ROUTE_ROOTS,
            )
        )
    }


def selected_routes(
    names: Optional[Iterable[str]],
):
    selected_names = list(
        config.ROUTE_NAMES
        if not names
        else names
    )

    catalog = route_catalog()

    missing = sorted(
        set(selected_names)
        - set(catalog)
    )

    if missing:
        raise ValueError(
            f"unknown routes: {missing}; "
            f"valid routes are "
            f"{list(config.ROUTE_NAMES)}"
        )

    result = []
    seen = set()

    for name in selected_names:
        if name in seen:
            continue

        route_index, root = catalog[name]
        result.append(
            (
                route_index,
                root,
                name,
            )
        )
        seen.add(name)

    return result


def configure_experiment_paths(
    experiment_name: str,
):
    if (
        not experiment_name
        or any(
            token in experiment_name
            for token in (
                "/",
                "\\",
                "..",
            )
        )
    ):
        raise ValueError(
            "--experiment-name must be "
            "a simple folder name"
        )

    output_dir = (
        config.PROJECT_ROOT
        / "outputs"
        / "temporal_prior_hardms_cross_route"
        / experiment_name
    )

    config.OUTPUT_DIR = output_dir
    config.CHECKPOINT_DIR = (
        output_dir / "checkpoints"
    )
    config.TEMPORAL_CHECKPOINT = (
        config.CHECKPOINT_DIR
        / "temporal_motion_retriever.pt"
    )


def check_resume_metadata(
    train_names: Sequence[str],
    jitter_m: float,
):
    if not config.TEMPORAL_CHECKPOINT.exists():
        return

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )

    saved_train_routes = checkpoint.get(
        "train_routes"
    )

    if (
        saved_train_routes is not None
        and list(saved_train_routes)
        != list(train_names)
    ):
        raise RuntimeError(
            "resume checkpoint was trained on "
            f"{saved_train_routes}, not "
            f"{list(train_names)}"
        )

    saved_jitter = checkpoint.get("jitter_m")

    if (
        saved_jitter is not None
        and abs(
            float(saved_jitter)
            - float(jitter_m)
        )
        > 1e-9
    ):
        raise RuntimeError(
            "resume checkpoint used "
            f"jitter={saved_jitter}, but "
            f"current command uses "
            f"jitter={jitter_m}"
        )


def write_checkpoint_metadata(
    train_names: Sequence[str],
    eval_names: Sequence[str],
    experiment_name: str,
):
    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location="cpu",
    )

    checkpoint["train_routes"] = list(
        train_names
    )
    checkpoint["validation_routes"] = list(
        train_names
    )
    checkpoint["eval_routes"] = list(
        eval_names
    )
    checkpoint["experiment_name"] = (
        experiment_name
    )

    torch.save(
        checkpoint,
        config.TEMPORAL_CHECKPOINT,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train RTL-CRF on Route A and "
            "evaluate the unseen Route B/C."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "train",
            "eval",
            "train_eval",
        ),
        default="train_eval",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
    )

    parser.add_argument(
        "--eval-split",
        choices=(
            "test",
            "all",
        ),
        default="test",
        help=(
            "test = last 15%% of each unseen "
            "route; all = complete unseen route"
        ),
    )

    parser.add_argument(
        "--train-routes",
        nargs="+",
        default=["route_A"],
        help=(
            "routes used by train_model() and "
            "validation; default: route_A"
        ),
    )

    parser.add_argument(
        "--eval-routes",
        nargs="+",
        default=[
            "route_B",
            "route_C",
        ],
        help=(
            "routes used only by predict_split(); "
            "default: route_B route_C"
        ),
    )

    parser.add_argument(
        "--experiment-name",
        default="train_A_test_BC",
    )

    parser.add_argument(
        "--jitter-m",
        type=float,
        default=config.LOCAL_PRIOR_JITTER_M,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--render-video",
        action="store_true",
        help=(
            "render inference MP4 files "
            "after evaluation"
        ),
    )

    parser.add_argument(
        "--video-fps",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=900,
    )

    args = parser.parse_args()

    train_records = selected_routes(
        args.train_routes
    )
    eval_records = selected_routes(
        args.eval_routes
    )

    train_names = [
        name
        for _, _, name in train_records
    ]
    eval_names = [
        name
        for _, _, name in eval_records
    ]

    overlap = sorted(
        set(train_names)
        & set(eval_names)
    )

    if overlap:
        raise ValueError(
            "training and evaluation routes "
            "must be disjoint; overlap="
            f"{overlap}"
        )

    # This is an additional safety check for
    # the exact requested protocol.
    if train_names != ["route_A"]:
        print(
            "warning: requested protocol is "
            "Train A -> Test B/C, but current "
            f"--train-routes are {train_names}",
            flush=True,
        )

    configure_experiment_paths(
        args.experiment_name
    )

    config.LOCAL_PRIOR_JITTER_M = float(
        args.jitter_m
    )

    if args.resume:
        check_resume_metadata(
            train_names,
            args.jitter_m,
        )

    set_seed(int(config.SEED))

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 76, flush=True)
    print(
        "RTL-CRF unseen-route experiment",
        flush=True,
    )
    print(
        f"train routes:      {train_names}",
        flush=True,
    )
    print(
        f"validation routes: {train_names}",
        flush=True,
    )
    print(
        f"evaluation routes: {eval_names}",
        flush=True,
    )
    print(
        f"evaluation split:  "
        f"{args.eval_split}",
        flush=True,
    )
    print(
        f"jitter:            "
        f"{args.jitter_m:.3f} m",
        flush=True,
    )
    print(
        f"output directory:  "
        f"{config.OUTPUT_DIR}",
        flush=True,
    )
    print(
        f"checkpoint:        "
        f"{config.TEMPORAL_CHECKPOINT}",
        flush=True,
    )
    print("=" * 76, flush=True)

    visual = FrozenVisualLocalizer(
        device
    )

    required_records: Dict[
        str,
        Tuple[int, Path, str],
    ] = {}

    if args.mode in (
        "train",
        "train_eval",
    ):
        for record in train_records:
            required_records[
                record[2]
            ] = record

    if args.mode in (
        "eval",
        "train_eval",
    ):
        for record in eval_records:
            required_records[
                record[2]
            ] = record

    cache_map: Dict[
        str,
        RouteCache,
    ] = {}

    # The original global route index is
    # preserved:
    # A=0, B=1, C=2.
    # Therefore deterministic jitter stays
    # identical to the original A+B+C run.
    for (
        route_index,
        root,
        name,
    ) in required_records.values():
        cache_map[name] = build_route_cache(
            root=root,
            name=name,
            route_index=route_index,
            visual=visual,
            device=device,
            jitter_m=args.jitter_m,
        )

    train_caches = [
        cache_map[name]
        for name in train_names
        if name in cache_map
    ]

    eval_caches = [
        cache_map[name]
        for name in eval_names
        if name in cache_map
    ]

    if args.mode in (
        "train",
        "train_eval",
    ):
        if not train_caches:
            raise RuntimeError(
                "no training caches were built"
            )

        training_capture_rates = {
            cache.name: float(
                cache.capture.float()
                .mean().item()
            )
            for cache in train_caches
        }

        minimum_capture = min(
            training_capture_rates.values()
        )

        if minimum_capture < float(
            config.MIN_TRAIN_CAPTURE_RATE
        ):
            raise RuntimeError(
                "training candidate lattice "
                "represents less than "
                f"{100.0 * float(config.MIN_TRAIN_CAPTURE_RATE):.1f}% "
                "of GT locations; run "
                "--jitter-m 0 to verify geometry"
            )

        if minimum_capture < 0.98:
            print(
                "warning: Route A training "
                "capture is below 98%; "
                "training uses only fully "
                "captured windows",
                flush=True,
            )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        if not eval_caches:
            raise RuntimeError(
                "no evaluation caches were built"
            )

        for cache in eval_caches:
            capture_rate = float(
                cache.capture.float()
                .mean().item()
            )

            if capture_rate < 0.98:
                print(
                    f"warning: {cache.name} "
                    "evaluation capture="
                    f"{capture_rate * 100:.2f}%; "
                    "evaluation still includes "
                    "every frame",
                    flush=True,
                )

    model = TemporalLatticeCRF().to(
        device
    )

    if args.mode in (
        "train",
        "train_eval",
    ):
        # Only Route A caches are passed here.
        train_model(
            model=model,
            caches=train_caches,
            device=device,
            epochs=args.epochs,
            resume=args.resume,
        )

        write_checkpoint_metadata(
            train_names=train_names,
            eval_names=eval_names,
            experiment_name=(
                args.experiment_name
            ),
        )

    if args.mode in (
        "eval",
        "train_eval",
    ):
        if args.mode == "eval":
            checkpoint = load_checkpoint(
                model,
                device,
            )

            saved_train_routes = (
                checkpoint.get(
                    "train_routes"
                )
            )

            if (
                saved_train_routes
                is not None
                and list(saved_train_routes)
                != train_names
            ):
                raise RuntimeError(
                    "checkpoint was trained "
                    f"on {saved_train_routes}, "
                    f"not {train_names}"
                )

        # Only Route B/C caches are passed here.
        outputs = predict_split(
            model=model,
            caches=eval_caches,
            split_name=args.eval_split,
            device=device,
            save_rows=True,
        )

        summary = {
            "method": ARCHITECTURE_NAME,
            "protocol": (
                "Train and validate only on "
                "Route A; evaluate unseen "
                "Route B and Route C; "
                "GT+jitter local prior; "
                "no model-output candidate "
                "propagation"
            ),
            "experiment_name": (
                args.experiment_name
            ),
            "train_routes": train_names,
            "validation_routes": train_names,
            "eval_routes": eval_names,
            "eval_split": args.eval_split,
            "jitter_m": float(
                args.jitter_m
            ),
            "checkpoint": str(
                config.TEMPORAL_CHECKPOINT
            ),
            "routes": {
                name: route_summary
                for (
                    name,
                    route_summary,
                    _,
                ) in outputs
            },
        }

        config.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = (
            config.OUTPUT_DIR
            / "robust_tracker_summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        print(
            f"summary saved to: "
            f"{summary_path}",
            flush=True,
        )

        if args.render_video:
            from render_results_video import (
                render_routes,
            )

            video_paths = render_routes(
                [
                    (root, name)
                    for _, root, name
                    in eval_records
                ],
                fps=float(
                    args.video_fps
                ),
                video_width=int(
                    args.video_width
                ),
                video_height=int(
                    args.video_height
                ),
            )

            for video_path in video_paths:
                print(
                    f"inference video: "
                    f"{video_path}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
