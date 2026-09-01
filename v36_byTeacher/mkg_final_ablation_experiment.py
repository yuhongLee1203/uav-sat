"""Controlled MKG final-method ablation experiment.

Protocol
--------
* Route A trains temporal G when the selected pipeline contains G.
* Routes B/C are evaluation only.
* Every evaluated frame uses its frame-aligned reference XY as the local SAT
  search center. Optional deterministic jitter is used only for robustness
  sensitivity and is never fed to G as a feature.
* The visual backbone/checkpoint stays frozen and identical across runs.
* M is a local visual decoder over an N x N candidate lattice. N, decoder,
  MeanShift bandwidth, and score temperature are independently controllable.
* Pipelines supported for component ablation: M, MK, MG, MKG.
* G input-branch ablations zero exactly one branch while keeping architecture
  dimensions and all other settings unchanged.
* Each run writes to an isolated suite/run directory; no existing experiment
  checkpoint or output is overwritten.

The script also measures per-frame latency/FPS and peak CUDA memory on B/C.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker as rt
from six_architecture_autoref_experiment import StandardXYKalman
from six_architecture_model import PositionRefinementGRU, PositionGRUOutput
from visual_localizer import FrozenVisualLocalizer

PIPELINES = ("M", "MK", "MG", "MKG")
DECODERS = ("softms", "weighted", "top1")
GRU_ABLATIONS = (
    "full",
    "no_xy",
    "no_variance",
    "no_temporal_mean",
    "no_first_difference",
)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def xy_tensor(xy, device):
    return torch.as_tensor(xy, dtype=torch.float32, device=device).reshape(1, 2)


class AblatedPositionRefinementGRU(PositionRefinementGRU):
    """Same six-architecture GRU with one input branch optionally zeroed."""

    def __init__(self, ablation="full", **kwargs):
        super().__init__(**kwargs)
        if ablation not in GRU_ABLATIONS:
            raise ValueError("unknown GRU ablation: %s" % ablation)
        self.ablation = ablation

    def forward_step(
        self,
        stage_xy: torch.Tensor,
        variance_xy: torch.Tensor,
        z_uav: torch.Tensor,
        previous_z_uav: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
    ) -> PositionGRUOutput:
        if previous_z_uav is None:
            previous_z_uav = z_uav
        if hidden is None:
            hidden = self.initial_hidden(
                z_uav.shape[0], z_uav.device, z_uav.dtype
            )

        temporal_mean = 0.5 * (z_uav + previous_z_uav)
        first_difference = z_uav - previous_z_uav
        xy_norm = stage_xy.float() / max(self.position_scale_m, 1e-6)
        var_norm = torch.log1p(
            variance_xy.float().clamp_min(0.0)
            / max(self.variance_scale_m2, 1e-6)
        )

        xy_feature = self.xy_projector(xy_norm)
        var_feature = self.var_projector(var_norm)
        mean_feature = self.mean_projector(temporal_mean)
        diff_feature = self.diff_projector(first_difference)

        if self.ablation == "no_xy":
            xy_feature = torch.zeros_like(xy_feature)
        elif self.ablation == "no_variance":
            var_feature = torch.zeros_like(var_feature)
        elif self.ablation == "no_temporal_mean":
            mean_feature = torch.zeros_like(mean_feature)
        elif self.ablation == "no_first_difference":
            diff_feature = torch.zeros_like(diff_feature)

        recurrent_input = torch.cat(
            [xy_feature, var_feature, mean_feature, diff_feature], dim=1
        )
        new_hidden = self.gru(recurrent_input, hidden)
        correction_xy = self.position_head(self.dropout(new_hidden))
        corrected_xy = stage_xy.float() + correction_xy
        return PositionGRUOutput(corrected_xy, correction_xy, new_hidden)


def build_model(device, gru_ablation):
    return AblatedPositionRefinementGRU(
        ablation=gru_ablation,
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)


@torch.no_grad()
def visual_decode(visual, uav_clip, center_xy, grid_size, decoder):
    batch = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=xy_tensor(center_xy, visual.device),
        grid_size=int(grid_size),
    )
    expected = int(grid_size) * int(grid_size)
    if int(batch.centers.shape[1]) != expected:
        raise RuntimeError(
            "candidate count mismatch: expected %d got %d"
            % (expected, int(batch.centers.shape[1]))
        )

    probability = torch.softmax(
        batch.raw_logits / max(float(config.MEANSHIFT_SCORE_TAU), 1e-6), dim=1
    )
    if decoder == "softms":
        xy = batch.softms_xy
        support = batch.softms_support
    elif decoder == "weighted":
        xy = (probability[:, :, None] * batch.centers).sum(dim=1)
        support = probability.max(dim=1).values
    elif decoder == "top1":
        xy = batch.raw_top1_xy
        support = probability.max(dim=1).values
    else:
        raise ValueError("unknown decoder: %s" % decoder)

    diff = batch.centers - xy[:, None, :]
    variance = (
        probability[:, :, None] * diff.square()
    ).sum(dim=1).clamp_min(1e-3)
    return {
        "xy": xy,
        "variance": variance,
        "support": support,
        "z_uav": batch.z_uav,
        "candidate_count": expected,
    }


def make_state(initial_xy):
    return {
        "kalman": StandardXYKalman(initial_xy),
        "hidden": None,
        "previous_z": None,
    }


def apply_k(state, xy, variance, device):
    filtered = state["kalman"].step(
        xy[0].detach().cpu().numpy(),
        variance[0].detach().cpu().numpy(),
    )
    return xy_tensor(filtered, device)


def apply_g(model, xy, variance, z_uav, state):
    out = model.forward_step(
        stage_xy=xy,
        variance_xy=variance,
        z_uav=z_uav,
        previous_z_uav=state["previous_z"],
        hidden=state["hidden"],
    )
    state["hidden"] = out.hidden
    return out.corrected_xy, out


def forward_frame(
    pipeline,
    model,
    visual,
    uav_clip,
    center_xy,
    state,
    device,
    grid_size,
    decoder,
):
    visual_out = visual_decode(
        visual, uav_clip, center_xy, grid_size, decoder
    )
    stage_xy = visual_out["xy"]
    variance = visual_out["variance"]
    z_uav = visual_out["z_uav"]
    trace = {
        "visual_xy": stage_xy,
        "visual_support": visual_out["support"],
    }
    gru_out = None

    for symbol in pipeline[1:]:
        if symbol == "K":
            stage_xy = apply_k(state, stage_xy, variance, device)
            trace["kalman_xy"] = stage_xy
        elif symbol == "G":
            if model is None:
                raise RuntimeError("pipeline contains G but model is None")
            stage_xy, gru_out = apply_g(
                model, stage_xy, variance, z_uav, state
            )
            trace["gru_xy"] = stage_xy
        else:
            raise ValueError("unsupported pipeline symbol: %s" % symbol)

    state["previous_z"] = z_uav.detach()
    return stage_xy, variance, gru_out, trace


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(
        route_name, config.ROUTE_ROOTS[idx], visual, device
    )


def deterministic_jitter(index, magnitude_m, seed):
    magnitude = float(magnitude_m)
    if magnitude <= 0:
        return np.zeros(2, dtype=np.float64)
    phase = (
        0.61803398875 * float(index)
        + 0.01745329252 * float(int(seed) % 360)
    )
    return magnitude * np.array([math.cos(phase), math.sin(phase)], dtype=np.float64)


def run_root(args):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "mkg_final_ablation_suite"
        / args.suite_tag
        / args.run_id
    )


def checkpoint_path(args):
    return run_root(args) / "temporal_checkpoint.pt"


def train_model(args, visual, route_a, device):
    if "G" not in args.pipeline:
        return None, None

    model = build_model(device, args.gru_ablation)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    outdir = run_root(args)
    outdir.mkdir(parents=True, exist_ok=True)

    initial_xy = route_a.gt_xy[0].detach().cpu().numpy().astype(np.float64)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        state = make_state(initial_xy)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []
        train_errors = []

        for index in range(len(route_a)):
            uav_clip = route_a.uav_clip[index:index + 1].to(device).float()
            reference_xy = (
                route_a.gt_xy[index].detach().cpu().numpy().astype(np.float64)
            )
            final_xy, _, gru_out, _ = forward_frame(
                args.pipeline,
                model,
                visual,
                uav_clip,
                reference_xy,
                state,
                device,
                args.grid_size,
                args.decoder,
            )
            if gru_out is None:
                raise RuntimeError("training pipeline did not execute G")

            target = route_a.gt_xy[index:index + 1].to(device).float()
            loss = F.smooth_l1_loss(final_xy, target)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            with torch.no_grad():
                train_errors.append(
                    float(torch.linalg.norm(final_xy - target, dim=1).mean().cpu())
                )

            is_last = index == len(route_a) - 1
            if chunk_count >= int(args.tbptt) or is_last:
                normalized = chunk_loss / float(chunk_count)
                normalized.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.GRAD_CLIP_NORM)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_losses.append(float(normalized.detach().cpu()))
                if state["hidden"] is not None:
                    state["hidden"] = state["hidden"].detach()
                if state["previous_z"] is not None:
                    state["previous_z"] = state["previous_z"].detach()
                chunk_loss = None
                chunk_count = 0

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        train_mle = float(np.mean(train_errors)) if train_errors else float("inf")

        improved = mean_loss < (best_loss - float(args.early_stop_delta))
        if improved:
            best_loss = mean_loss
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "model": best_state,
                    "epoch": int(epoch),
                    "train_loss": float(best_loss),
                    "run_id": args.run_id,
                    "suite_tag": args.suite_tag,
                    "pipeline": args.pipeline,
                    "grid_size": int(args.grid_size),
                    "decoder": args.decoder,
                    "gru_ablation": args.gru_ablation,
                    "meanshift_bandwidth_m": float(args.bandwidth),
                    "meanshift_score_tau": float(args.tau),
                    "seed": int(args.seed),
                    "train_route": "route_A",
                    "frame_aligned_reference_center": True,
                },
                checkpoint_path(args),
            )
        else:
            bad_epochs += 1

        print(
            "[MKG-ablation:%s] epoch=%03d/%d pipeline=%s grid=%dx%d decoder=%s "
            "gru=%s train_loss=%.6f train_mle=%.3fm best=%.6f bad=%d/%d"
            % (
                args.run_id,
                epoch,
                args.epochs,
                args.pipeline,
                args.grid_size,
                args.grid_size,
                args.decoder,
                args.gru_ablation,
                mean_loss,
                train_mle,
                best_loss,
                bad_epochs,
                args.early_stop_patience,
            ),
            flush=True,
        )

        if (
            epoch >= int(args.early_stop_min_epoch)
            and bad_epochs >= int(args.early_stop_patience)
        ):
            print(
                "[MKG-ablation:%s] early-stop at epoch %d" % (args.run_id, epoch),
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("no temporal checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    return model, float(best_loss)


@torch.no_grad()
def evaluate_one(args, visual, model, route_name, cache, device, jitter_m):
    initial_xy = cache.gt_xy[0].detach().cpu().numpy().astype(np.float64)
    state = make_state(initial_xy)
    if model is not None:
        model.eval()

    final_errors = []
    visual_errors = []
    rows = []
    elapsed = []
    warmup = min(int(args.latency_warmup), max(0, len(cache) // 5))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index:index + 1].to(device).float()
        reference_xy = cache.gt_xy[index].detach().cpu().numpy().astype(np.float64)
        search_center = reference_xy + deterministic_jitter(index, jitter_m, args.seed)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        final_t, _, _, trace = forward_frame(
            args.pipeline,
            model,
            visual,
            uav_clip,
            search_center,
            state,
            device,
            args.grid_size,
            args.decoder,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if index >= warmup:
            elapsed.append(dt)

        final_xy = final_t[0].detach().cpu().numpy().astype(np.float64)
        visual_xy = trace["visual_xy"][0].detach().cpu().numpy().astype(np.float64)
        final_error = float(np.linalg.norm(final_xy - reference_xy))
        visual_error = float(np.linalg.norm(visual_xy - reference_xy))
        final_errors.append(final_error)
        visual_errors.append(visual_error)

        row = {
            "frame_index": int(index),
            "frame_id": int(cache.frame_ids[index]),
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "search_center_x": float(search_center[0]),
            "search_center_y": float(search_center[1]),
            "jitter_m": float(jitter_m),
            "visual_x": float(visual_xy[0]),
            "visual_y": float(visual_xy[1]),
            "visual_error_m": visual_error,
            "final_x": float(final_xy[0]),
            "final_y": float(final_xy[1]),
            "final_error_m": final_error,
        }
        for name in ("kalman_xy", "gru_xy"):
            if name in trace:
                v = trace[name][0].detach().cpu().numpy().astype(np.float64)
                row[name + "_x"] = float(v[0])
                row[name + "_y"] = float(v[1])
        rows.append(row)

        if state["hidden"] is not None:
            state["hidden"] = state["hidden"].detach()
        if state["previous_z"] is not None:
            state["previous_z"] = state["previous_z"].detach()

    mean_latency_ms = 1000.0 * float(np.mean(elapsed)) if elapsed else float("nan")
    fps = (1000.0 / mean_latency_ms) if mean_latency_ms > 0 else float("nan")
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device.type == "cuda"
        else float("nan")
    )

    outdir = run_root(args) / ("jitter_%gm" % float(jitter_m))
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / ("%s_frames.csv" % route_name)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = rt.metric_summary(final_errors)
    summary.update(
        {
            "Route": route_name,
            "VisualMLE_m": float(np.mean(visual_errors)),
            "Latency_ms_per_frame": mean_latency_ms,
            "FPS": fps,
            "PeakCUDAAllocated_MB": peak_memory_mb,
            "Jitter_m": float(jitter_m),
            "CandidateCount": int(args.grid_size) ** 2,
            "CSV": str(csv_path),
        }
    )
    return summary


def parse_jitters(value):
    values = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values or [0.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite-tag", required=True)
    parser.add_argument("--pipeline", choices=PIPELINES, default="MKG")
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--decoder", choices=DECODERS, default="softms")
    parser.add_argument("--gru-ablation", choices=GRU_ABLATIONS, default="full")
    parser.add_argument("--bandwidth", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=float(getattr(config, "TEMPORAL_LR", 3e-4)))
    parser.add_argument("--tbptt", type=int, default=int(getattr(config, "TBPTT_STEPS", 32)))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-jitters", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-epoch", type=int, default=20)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    parser.add_argument("--latency-warmup", type=int, default=30)
    args = parser.parse_args()

    if args.grid_size < 2:
        raise SystemExit("--grid-size must be >=2")

    device = rt.resolve_device(args.device)
    set_seed(args.seed)
    config.MEANSHIFT_BANDWIDTH_M = float(args.bandwidth)
    config.MEANSHIFT_SCORE_TAU = float(args.tau)

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)

    visual = FrozenVisualLocalizer(device)
    route_a = build_cache("route_A", visual, device)
    model, best_train_loss = train_model(args, visual, route_a, device)

    results = {
        "run_id": args.run_id,
        "suite_tag": args.suite_tag,
        "protocol": "controlled frame-aligned reference-centered MKG ablation",
        "training_from_scratch": bool("G" in args.pipeline),
        "visual_backbone_frozen": True,
        "train_route": "route_A",
        "test_routes": ["route_B", "route_C"],
        "pipeline": args.pipeline,
        "grid_size": int(args.grid_size),
        "candidate_count": int(args.grid_size) ** 2,
        "decoder": args.decoder,
        "gru_ablation": args.gru_ablation,
        "meanshift_bandwidth_m": float(args.bandwidth),
        "meanshift_score_tau": float(args.tau),
        "seed": int(args.seed),
        "best_train_loss": best_train_loss,
        "frame_aligned_reference_center": True,
        "forward_search": False,
        "eval_jitters_m": parse_jitters(args.eval_jitters),
        "results": {},
    }

    for jitter_m in parse_jitters(args.eval_jitters):
        jitter_key = "%g" % float(jitter_m)
        results["results"][jitter_key] = {}
        for route_name in ("route_B", "route_C"):
            cache = build_cache(route_name, visual, device)
            summary = evaluate_one(
                args, visual, model, route_name, cache, device, jitter_m
            )
            results["results"][jitter_key][route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    outdir = run_root(args)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
