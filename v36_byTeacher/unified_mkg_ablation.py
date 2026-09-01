"""Unified MKG ablation under the same fixed-8m formal protocol.

Every ordinary ablation changes exactly one requested factor while preserving:
  fixed 8m prior, Route-A training, Route-B/C testing, same frozen Stage-1.
Robustness is the only mode that changes prior error. A robustness level whose
candidate capture falls below the formal threshold is skipped, not reported as
a localization failure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker_base as rb
import unified_protocol as up
import unified_main_architectures as uma
from six_architecture_model import PositionRefinementGRU, PositionGRUOutput
from visual_localizer import FrozenVisualLocalizer

PIPELINES = ("M", "MK", "MG", "MKG")
DECODERS = ("softms", "weighted")
GRU_ABLATIONS = (
    "full",
    "no_xy",
    "no_variance",
    "no_temporal_mean",
    "no_first_difference",
)
ROOT = Path(config.BACKBONE_OUTPUT_DIR) / "unified_fixed8m_v1" / "ablations"


class AblatedPositionRefinementGRU(PositionRefinementGRU):
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


def new_model(device, ablation):
    return AblatedPositionRefinementGRU(
        ablation=ablation,
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)


def run_dir(args):
    return ROOT / args.run_id


def checkpoint_path(args):
    return run_dir(args) / "temporal_checkpoint.pt"


def _ref_np(cache, index):
    return cache.gt_xy[index].detach().cpu().numpy().astype(np.float64).reshape(2)


def _initial_state(cache, route_name, jitter_m):
    ref0 = _ref_np(cache, 0)
    center0 = up.search_center(
        ref0, 0, route_name, magnitude_m=float(jitter_m), seed=up.MAIN_SEED
    )
    return up.make_state(center0)


def train_model(args, visual, route_a, device):
    if "G" not in args.pipeline:
        return None

    rb.set_seed(int(args.model_seed))
    model = new_model(device, args.gru_ablation)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        state = _initial_state(route_a, "route_A", up.MAIN_JITTER_M)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        losses = []
        train_errors = []

        for index in range(len(route_a)):
            ref = _ref_np(route_a, index)
            center = up.search_center(ref, index, "route_A")
            uav = route_a.uav_clip[index:index + 1].to(device).float()
            final_xy, _, gru_out, _ = uma.same_frame_forward(
                args.pipeline,
                model,
                visual,
                uav,
                center,
                state,
                device,
                grid_size=args.grid_size,
                decoder=args.decoder,
                bandwidth_m=args.bandwidth,
                tau=args.tau,
            )
            if gru_out is None:
                raise RuntimeError("training pipeline did not execute G")
            target = route_a.gt_xy[index:index + 1].to(device).float()
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1
            with torch.no_grad():
                train_errors.append(
                    float(torch.linalg.norm(final_xy - target, dim=1).mean().cpu())
                )

            if chunk_count >= int(args.tbptt) or index == len(route_a) - 1:
                normalized = chunk_loss / float(chunk_count)
                normalized.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.GRAD_CLIP_NORM)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(normalized.detach().cpu()))
                if state["hidden"] is not None:
                    state["hidden"] = state["hidden"].detach()
                if state["previous_z"] is not None:
                    state["previous_z"] = state["previous_z"].detach()
                chunk_loss = None
                chunk_count = 0

        mean_loss = float(np.mean(losses))
        final_train_mle = float(np.mean(train_errors))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            checkpoint_path(args).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": best_state,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "run_id": args.run_id,
                    "pipeline": args.pipeline,
                    "grid_size": args.grid_size,
                    "decoder": args.decoder,
                    "gru_ablation": args.gru_ablation,
                    "bandwidth_m": args.bandwidth,
                    "tau": args.tau,
                    "model_seed": args.model_seed,
                    "protocol": up.protocol_metadata(),
                },
                checkpoint_path(args),
            )
        print(
            "[ablation:%s] epoch=%03d/%d loss=%.6f train_final_mle=%.3fm best=%.6f"
            % (args.run_id, epoch, args.epochs, mean_loss, final_train_mle, best_loss),
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    return model


def load_main_mkg(device):
    path = uma.checkpoint_path("MKG")
    if not path.exists():
        raise FileNotFoundError(
            "main MKG checkpoint is required before robustness: %s" % path
        )
    payload = torch.load(path, map_location="cpu")
    model = new_model(device, "full")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    args,
    visual,
    model,
    route_name,
    cache,
    device,
    jitter_m,
    write_frames=True,
):
    capture = up.capture_report(
        visual,
        cache,
        route_name,
        jitter_m=float(jitter_m),
        grid_size=int(args.grid_size),
        seed=up.MAIN_SEED,
    )
    state = _initial_state(cache, route_name, jitter_m)
    final_errors = []
    visual_errors = []
    rows = []

    for index in range(len(cache)):
        ref = _ref_np(cache, index)
        center = up.search_center(
            ref,
            index,
            route_name,
            magnitude_m=float(jitter_m),
            seed=up.MAIN_SEED,
        )
        uav = cache.uav_clip[index:index + 1].to(device).float()
        final_t, _, _, trace = uma.same_frame_forward(
            args.pipeline,
            model,
            visual,
            uav,
            center,
            state,
            device,
            grid_size=args.grid_size,
            decoder=args.decoder,
            bandwidth_m=args.bandwidth,
            tau=args.tau,
        )
        final_xy = final_t[0].cpu().numpy().astype(np.float64)
        visual_xy = trace["base_ms_xy"][0].cpu().numpy().astype(np.float64)
        final_error = float(np.linalg.norm(final_xy - ref))
        visual_error = float(np.linalg.norm(visual_xy - ref))
        final_errors.append(final_error)
        visual_errors.append(visual_error)
        if write_frames:
            rows.append(
                {
                    "frame_id": int(cache.frame_ids[index]),
                    "reference_x": float(ref[0]),
                    "reference_y": float(ref[1]),
                    "search_center_error_m": float(np.linalg.norm(center - ref)),
                    "visual_error_m": visual_error,
                    "final_error_m": final_error,
                }
            )
        if state["hidden"] is not None:
            state["hidden"] = state["hidden"].detach()
        if state["previous_z"] is not None:
            state["previous_z"] = state["previous_z"].detach()

    summary = rb.metric_summary(final_errors)
    summary.update(
        {
            "Route": route_name,
            "VisualMLE_m": float(np.mean(visual_errors)),
            "Jitter_m": float(jitter_m),
            "CandidateCaptureRate_pct": capture["CandidateCaptureRate_pct"],
            "CandidateNearestP95_m": capture["NearestCandidateP95_m"],
        }
    )
    if write_frames:
        out = run_dir(args) / ("jitter_%gm" % float(jitter_m))
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / ("%s_frames.csv" % route_name)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["CSV"] = str(csv_path)
    return summary


def normal_run(args, visual, device):
    route_a = up.build_cache("route_A", visual, device)
    train_capture = up.capture_report(
        visual, route_a, "route_A",
        jitter_m=up.MAIN_JITTER_M,
        grid_size=args.grid_size,
    )
    if not args.allow_low_capture:
        up.assert_capture(train_capture)

    model = train_model(args, visual, route_a, device)
    if "G" not in args.pipeline:
        model = None

    results = {}
    for route_name in ("route_B", "route_C"):
        cache = up.build_cache(route_name, visual, device)
        report = up.capture_report(
            visual, cache, route_name,
            jitter_m=up.MAIN_JITTER_M,
            grid_size=args.grid_size,
        )
        if not args.allow_low_capture:
            up.assert_capture(report)
        summary = evaluate(
            args, visual, model, route_name, cache, device,
            jitter_m=up.MAIN_JITTER_M,
        )
        results[route_name] = summary
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    payload = {
        "run_id": args.run_id,
        "protocol": up.protocol_metadata(),
        "changed_factor": {
            "pipeline": args.pipeline,
            "grid_size": args.grid_size,
            "decoder": args.decoder,
            "gru_ablation": args.gru_ablation,
            "bandwidth_m": args.bandwidth,
            "tau": args.tau,
            "model_seed": args.model_seed,
        },
        "formal_jitter_m": float(up.MAIN_JITTER_M),
        "train_capture": train_capture,
        "training_from_scratch": bool("G" in args.pipeline),
        "results": results,
    }
    up.write_json(run_dir(args) / "summary.json", payload)


def robustness_run(args, visual, device):
    args.pipeline = "MKG"
    args.grid_size = up.MAIN_GRID_SIZE
    args.decoder = up.MAIN_DECODER
    args.gru_ablation = "full"
    args.bandwidth = up.MAIN_BANDWIDTH_M
    args.tau = up.MAIN_TAU
    model = load_main_mkg(device)

    levels = [float(x) for x in args.robustness_levels.split(",") if x.strip()]
    output = {}
    for level in levels:
        route_map = {}
        valid = True
        capture_map = {}
        for route_name in ("route_B", "route_C"):
            cache = up.build_cache(route_name, visual, device)
            cap = up.capture_report(
                visual, cache, route_name,
                jitter_m=level,
                grid_size=up.MAIN_GRID_SIZE,
            )
            capture_map[route_name] = cap
            if (
                float(cap["CandidateCaptureRate_pct"]) / 100.0
                < up.MAIN_CAPTURE_MIN_RATE
            ):
                valid = False

        if not valid:
            output[str(level)] = {
                "Skipped": True,
                "Reason": "candidate capture below formal threshold",
                "Capture": capture_map,
            }
            print(
                "[robustness] jitter=%.1fm SKIP because capture < %.1f%%"
                % (level, 100.0 * up.MAIN_CAPTURE_MIN_RATE),
                flush=True,
            )
            continue

        for route_name in ("route_B", "route_C"):
            cache = up.build_cache(route_name, visual, device)
            route_map[route_name] = evaluate(
                args, visual, model, route_name, cache, device,
                jitter_m=level, write_frames=False,
            )
        output[str(level)] = {
            "Skipped": False,
            "Capture": capture_map,
            "Results": route_map,
        }

    payload = {
        "run_id": args.run_id,
        "protocol": up.protocol_metadata(),
        "purpose": "prior-error robustness; no zero-m oracle",
        "levels": output,
    }
    up.write_json(run_dir(args) / "summary.json", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("normal", "robustness"), default="normal")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pipeline", choices=PIPELINES, default="MKG")
    parser.add_argument("--grid-size", type=int, default=up.MAIN_GRID_SIZE)
    parser.add_argument("--decoder", choices=DECODERS, default=up.MAIN_DECODER)
    parser.add_argument("--gru-ablation", choices=GRU_ABLATIONS, default="full")
    parser.add_argument("--bandwidth", type=float, default=up.MAIN_BANDWIDTH_M)
    parser.add_argument("--tau", type=float, default=up.MAIN_TAU)
    parser.add_argument("--model-seed", type=int, default=up.MAIN_SEED)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=float(config.TEMPORAL_LR))
    parser.add_argument("--tbptt", type=int, default=int(config.TBPTT_STEPS))
    parser.add_argument("--allow-low-capture", action="store_true")
    parser.add_argument(
        "--robustness-levels",
        default="4,8,12,16,20",
        help="0m is intentionally excluded from formal robustness",
    )
    args = parser.parse_args()

    device = rb.resolve_device(args.device)
    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)
    visual = FrozenVisualLocalizer(device)

    if args.mode == "robustness":
        robustness_run(args, visual, device)
    else:
        normal_run(args, visual, device)


if __name__ == "__main__":
    main()
