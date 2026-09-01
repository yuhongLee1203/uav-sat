"""Thesis-grade MKG ablation suite.

Protocol
--------
* Train Route A only; evaluate Routes B/C.
* Final method is M -> K -> G.
* The controlled reference position is used only as the local search center for
  this ablation suite, matching six_architecture_gtref_experiment.py.
* Every experiment gets its own checkpoint/output tag; existing six-order,
  delayed, autonomous, and route-tube results are never overwritten.

Ablations
---------
1. Component contribution: M, M+K, M+G, M+K+G.
2. Candidate window: 3x3, 5x5, 6x6, 7x7, 9x9.
3. Decoder: Soft MeanShift, weighted centroid, Top-1.
4. GRU inputs/state: no XY, no variance, no temporal mean,
   no first difference, no recurrent hidden state.
5. MeanShift bandwidth: 4 m, 8 m, 12 m.
6. Coarse-center robustness: 0/5/10/15/20/25/30 m deterministic offsets.
7. Seed stability: 42/123/2026 for the full 6x6 MKG model.
8. Efficiency: latency/FPS, candidate count, candidate radius and peak GPU memory.
9. Existing six-order results are collected without retraining them.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker as rt
import six_architecture_autoref_experiment as core
from six_architecture_model import PositionGRUOutput, PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer

ABLS = (
    "full",
    "no_xy",
    "no_variance",
    "no_temporal_mean",
    "no_first_difference",
    "no_hidden",
)
DECS = ("softms", "weighted", "top1")
COMPS = ("M", "MK", "MG", "MKG")
METRICS = (
    "MLE_m",
    "MedLE_m",
    "P90_m",
    "P95_m",
    "P99_m",
    "CVaR90_m",
    "LSR@5_pct",
    "LSR@10_pct",
    "LSR@15_pct",
    "LSR@20_pct",
)
WINDOWS = (3, 5, 6, 7, 9)
ROBUSTNESS_M = (0, 5, 10, 15, 20, 25, 30)
STABILITY_SEEDS = (42, 123, 2026)


class MaskedGRU(PositionRefinementGRU):
    """Exact PositionRefinementGRU with one input/state family zeroed."""

    def __init__(self, ablation):
        super().__init__(
            int(config.RNN_FEATURE_DIM),
            int(config.RNN_HIDDEN_DIM),
            float(config.RNN_DROPOUT),
        )
        self.ablation = str(ablation)

    def forward_step(self, stage_xy, variance_xy, z_uav, previous_z_uav, hidden):
        if previous_z_uav is None:
            previous_z_uav = z_uav
        if hidden is None or self.ablation == "no_hidden":
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

        if self.ablation == "no_xy":
            xy_norm = torch.zeros_like(xy_norm)
        if self.ablation == "no_variance":
            var_norm = torch.zeros_like(var_norm)
        if self.ablation == "no_temporal_mean":
            temporal_mean = torch.zeros_like(temporal_mean)
        if self.ablation == "no_first_difference":
            first_difference = torch.zeros_like(first_difference)

        recurrent_input = torch.cat(
            [
                self.xy_projector(xy_norm),
                self.var_projector(var_norm),
                self.mean_projector(temporal_mean),
                self.diff_projector(first_difference),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        correction_xy = self.position_head(self.dropout(new_hidden))
        corrected_xy = stage_xy.float() + correction_xy
        return PositionGRUOutput(corrected_xy, correction_xy, new_hidden)


def root():
    return Path(config.BACKBONE_OUTPUT_DIR) / "mkg_final_thesis_ablation"


def ckpt(tag):
    return Path(config.CHECKPOINT_DIR) / (
        f"mkg_final_ablation_{tag}_{config.BACKBONE_KEY}.pt"
    )


def rdir(tag):
    return root() / "runs" / tag


def build_cache(name, visual, device):
    idx = config.ROUTE_NAMES.index(name)
    return rt.build_route_cache(
        name, config.ROUTE_ROOTS[idx], visual, device
    )


def make_state(name, visual):
    _, xy, _ = rt.planned_route_start(
        name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": core.StandardXYKalman(xy),
        "hidden": None,
        "previous_z": None,
    }


def measure(visual, uav, center, grid, decoder):
    center_t = torch.as_tensor(
        center, dtype=torch.float32, device=visual.device
    ).reshape(1, 2)
    batch = visual.candidate_batch(
        uav_clip=uav,
        center_xy=center_t,
        grid_size=int(grid),
    )
    probability = torch.softmax(
        batch.raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )

    if decoder == "softms":
        xy = batch.softms_xy
    elif decoder == "weighted":
        xy = (probability[:, :, None] * batch.centers).sum(dim=1)
    elif decoder == "top1":
        xy = batch.raw_top1_xy
    else:
        raise ValueError(decoder)

    diff = batch.centers - xy[:, None, :]
    variance = (
        probability[:, :, None] * diff.square()
    ).sum(dim=1).clamp_min(1e-3)
    return xy, variance, batch.z_uav, batch.centers


def forward_frame(component, model, visual, uav, center, state, device, grid, decoder):
    xy, variance, z_uav, centers = measure(
        visual, uav, center, grid, decoder
    )
    base_xy = xy
    gru_out = None

    for symbol in component[1:]:
        if symbol == "K":
            filtered = state["kalman"].step(
                xy[0].detach().cpu().numpy(),
                variance[0].detach().cpu().numpy(),
            )
            xy = torch.as_tensor(
                filtered, dtype=torch.float32, device=device
            ).reshape(1, 2)
        elif symbol == "G":
            hidden = None if model.ablation == "no_hidden" else state["hidden"]
            gru_out = model.forward_step(
                xy, variance, z_uav, state["previous_z"], hidden
            )
            xy = gru_out.corrected_xy
            state["hidden"] = (
                None if model.ablation == "no_hidden" else gru_out.hidden
            )
        else:
            raise ValueError("unsupported component symbol: %s" % symbol)

    state["previous_z"] = z_uav.detach()
    return xy, base_xy, variance, centers, gru_out


def model_for(ablation, device):
    return MaskedGRU(ablation).to(device)


def train(tag, args, visual, device):
    if "G" not in args.component:
        return None, None

    model = model_for(args.gru_ablation, device)
    route_a = build_cache("route_A", visual, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        state = make_state("route_A", visual)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []

        for index in range(len(route_a)):
            uav = route_a.uav_clip[index:index + 1].to(device).float()
            reference = (
                route_a.gt_xy[index].detach().cpu().numpy().astype(np.float64)
            )
            _, _, _, _, gru_out = forward_frame(
                args.component,
                model,
                visual,
                uav,
                reference,
                state,
                device,
                args.grid_size,
                args.decoder,
            )
            if gru_out is None:
                raise RuntimeError("training component did not execute G")

            target = route_a.gt_xy[index:index + 1].to(device).float()
            current_loss = F.smooth_l1_loss(gru_out.corrected_xy, target)
            chunk_loss = (
                current_loss
                if chunk_loss is None
                else chunk_loss + current_loss
            )
            chunk_count += 1

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
                chunk_loss = None
                chunk_count = 0
                if state["hidden"] is not None:
                    state["hidden"] = state["hidden"].detach()
                if state["previous_z"] is not None:
                    state["previous_z"] = state["previous_z"].detach()

        mean_loss = (
            float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            ckpt(tag).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": best_state,
                    "tag": tag,
                    "component": args.component,
                    "grid_size": int(args.grid_size),
                    "decoder": args.decoder,
                    "gru_ablation": args.gru_ablation,
                    "ms_bandwidth_m": float(config.MEANSHIFT_BANDWIDTH_M),
                    "seed": int(args.seed),
                    "epoch": int(epoch),
                    "loss": float(best_loss),
                    "train_route": "route_A",
                    "reference_mode": "frame-aligned reference XY local-search center",
                },
                ckpt(tag),
            )

        print(
            f"[{tag}] epoch={epoch:03d}/{args.epochs} "
            f"loss={mean_loss:.6f} best={best_loss:.6f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no checkpoint produced for %s" % tag)
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def load_model(args, device):
    if "G" not in args.component:
        return None
    path = ckpt(args.load_tag or args.tag)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    model = model_for(args.gru_ablation, device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def deterministic_offset(route_name, index, radius_m, seed):
    if float(radius_m) <= 0.0:
        return np.zeros(2, dtype=np.float64)
    route_index = config.ROUTE_NAMES.index(route_name)
    rng = np.random.default_rng(
        int(seed) + 100003 * route_index + 9176 * int(index)
    )
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return float(radius_m) * np.array(
        [math.cos(angle), math.sin(angle)], dtype=np.float64
    )


@torch.no_grad()
def evaluate(route_name, args, visual, model, device):
    route = build_cache(route_name, visual, device)
    state = make_state(route_name, visual)
    errors = []
    base_errors = []
    captures = []
    max_radii = []
    elapsed_ms = []
    warmup_frames = min(20, max(0, len(route) // 10))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for index in range(len(route)):
        uav = route.uav_clip[index:index + 1].to(device).float()
        reference = route.gt_xy[index].cpu().numpy().astype(np.float64)
        center = reference + deterministic_offset(
            route_name, index, args.prior_error_m, args.seed
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output, base_xy, _, centers, _ = forward_frame(
            args.component,
            model,
            visual,
            uav,
            center,
            state,
            device,
            args.grid_size,
            args.decoder,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        duration_ms = (time.perf_counter() - started) * 1000.0
        if index >= warmup_frames:
            elapsed_ms.append(duration_ms)

        output_np = output[0].cpu().numpy()
        base_np = base_xy[0].cpu().numpy()
        centers_np = centers[0].cpu().numpy()
        errors.append(float(np.linalg.norm(output_np - reference)))
        base_errors.append(float(np.linalg.norm(base_np - reference)))
        nearest = np.linalg.norm(
            centers_np - reference[None, :], axis=1
        )
        captures.append(
            float(nearest.min() <= float(config.CANDIDATE_CAPTURE_RADIUS_M))
        )
        max_radii.append(
            float(
                np.linalg.norm(
                    centers_np - center[None, :], axis=1
                ).max()
            )
        )

    latency = float(np.mean(elapsed_ms)) if elapsed_ms else float("nan")
    summary = rt.metric_summary(errors)
    summary.update(
        {
            "BaseVisualMLE_m": float(np.mean(base_errors)),
            "CandidateCapture_pct": 100.0 * float(np.mean(captures)),
            "CandidateCount": int(args.grid_size) ** 2,
            "MeanCandidateMaxRadius_m": float(np.mean(max_radii)),
            "Latency_ms_per_frame": latency,
            "FPS": 1000.0 / latency if latency > 0.0 else 0.0,
            "TimingWarmupFrames": int(warmup_frames),
            "PeakAllocatedGPU_MB": (
                float(torch.cuda.max_memory_allocated(device) / 1048576.0)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    return summary


def run(args):
    if args.ms_bandwidth is not None:
        config.MEANSHIFT_BANDWIDTH_M = float(args.ms_bandwidth)

    rt.set_seed(int(args.seed))
    device = rt.resolve_device(args.device)
    visual = FrozenVisualLocalizer(device)

    if args.mode == "train-eval":
        model, train_loss = train(args.tag, args, visual, device)
    else:
        model, train_loss = load_model(args, device), None

    obj = {
        "tag": args.tag,
        "group": args.group,
        "component": args.component,
        "grid": int(args.grid_size),
        "candidate_count": int(args.grid_size) ** 2,
        "decoder": args.decoder,
        "gru": args.gru_ablation,
        "bandwidth": float(config.MEANSHIFT_BANDWIDTH_M),
        "prior_error": float(args.prior_error_m),
        "seed": int(args.seed),
        "train_loss": train_loss,
        "train_route": "route_A",
        "test_routes": ["route_B", "route_C"],
        "results": {},
    }
    for route_name in ("route_B", "route_C"):
        obj["results"][route_name] = evaluate(
            route_name, args, visual, model, device
        )
        print(
            "[%s][%s] %s"
            % (
                args.tag,
                route_name,
                json.dumps(obj["results"][route_name]),
            ),
            flush=True,
        )

    out = rdir(args.tag)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect():
    rows = []
    runs = root() / "runs"
    for path in sorted(runs.glob("*/summary.json")):
        obj = json.load(path.open(encoding="utf-8"))
        results = obj["results"]
        row = {
            key: obj[key]
            for key in (
                "tag",
                "group",
                "component",
                "grid",
                "candidate_count",
                "decoder",
                "gru",
                "bandwidth",
                "prior_error",
                "seed",
            )
        }
        for key in METRICS + (
            "BaseVisualMLE_m",
            "CandidateCapture_pct",
            "MeanCandidateMaxRadius_m",
            "Latency_ms_per_frame",
            "FPS",
            "PeakAllocatedGPU_MB",
        ):
            row["B_" + key] = float(results["route_B"][key])
            row["C_" + key] = float(results["route_C"][key])
            row["Avg_" + key] = float(
                np.mean(
                    [
                        results["route_B"][key],
                        results["route_C"][key],
                    ]
                )
            )
        rows.append(row)

    tables = root() / "tables"
    _write_csv(tables / "all_runs_bc_average.csv", rows)

    groups = (
        ("component", "component_ablation.csv"),
        ("window", "window_size_ablation.csv"),
        ("decoder", "decoder_ablation.csv"),
        ("gru", "gru_ablation.csv"),
        ("bandwidth", "bandwidth_ablation.csv"),
        ("robustness", "coarse_prior_robustness.csv"),
        ("seed", "seed_stability.csv"),
    )
    for group_name, filename in groups:
        selected = [
            row
            for row in rows
            if row["group"] in ("baseline", group_name)
        ]
        _write_csv(tables / filename, selected)
    _write_csv(tables / "efficiency_all.csv", rows)

    # Reuse already-completed six-order controlled results instead of wasting GPU.
    old_root = (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "six_architecture_gt_center_center6x6"
    )
    order_rows = []
    for architecture in ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG"):
        summary_path = old_root / architecture.lower() / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.load(summary_path.open(encoding="utf-8"))["results"]
        row = {"Architecture": architecture}
        for metric in METRICS:
            row["B_" + metric] = float(payload["route_B"][metric])
            row["C_" + metric] = float(payload["route_C"][metric])
            row["Avg_" + metric] = float(
                np.mean(
                    [
                        payload["route_B"][metric],
                        payload["route_C"][metric],
                    ]
                )
            )
        order_rows.append(row)
    _write_csv(tables / "architecture_order_existing.csv", order_rows)

    # Compact mean/std table for the three independent full-model seeds.
    seed_rows = [
        row
        for row in rows
        if row["tag"] == "baseline_mkg_g6_softms_bw8"
        or row["group"] == "seed"
    ]
    stability = []
    if seed_rows:
        result = {"Method": "MKG-6x6-SoftMS"}
        for metric in METRICS:
            values = np.asarray(
                [row["Avg_" + metric] for row in seed_rows],
                dtype=np.float64,
            )
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_std"] = float(values.std(ddof=0))
        result["Seeds"] = ";".join(str(row["seed"]) for row in seed_rows)
        stability.append(result)
    _write_csv(tables / "seed_stability_mean_std.csv", stability)

    manifest = {
        "final_method": "MKG",
        "protocol": "Route-A train; Route-B/C controlled reference-centered eval",
        "new_runs": len(rows),
        "windows": list(WINDOWS),
        "candidate_counts": [value * value for value in WINDOWS],
        "decoders": list(DECS),
        "gru_ablations": list(ABLS),
        "bandwidth_m": [4, 8, 12],
        "prior_errors_m": list(ROBUSTNESS_M),
        "stability_seeds": list(STABILITY_SEEDS),
        "existing_six_order_rerun": False,
        "tables": sorted(path.name for path in tables.glob("*.csv")),
    }
    with (root() / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train-eval", "eval", "collect"),
        default="train-eval",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--group", default="baseline")
    parser.add_argument("--component", choices=COMPS, default="MKG")
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--decoder", choices=DECS, default="softms")
    parser.add_argument("--gru-ablation", choices=ABLS, default="full")
    parser.add_argument("--prior-error-m", type=float, default=0.0)
    parser.add_argument("--ms-bandwidth", type=float, default=None)
    parser.add_argument("--load-tag", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument("--lr", type=float, default=float(config.TEMPORAL_LR))
    parser.add_argument("--tbptt", type=int, default=int(config.TBPTT_STEPS))
    parser.add_argument("--seed", type=int, default=int(config.SEED))
    args = parser.parse_args()

    if args.mode == "collect":
        collect()
        return
    if int(args.grid_size) < 2:
        raise SystemExit("--grid-size must be >= 2")
    if not args.tag:
        raise SystemExit("--tag is required for train/eval")
    run(args)


if __name__ == "__main__":
    main()
