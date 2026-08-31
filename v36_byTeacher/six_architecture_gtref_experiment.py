"""Controlled six-architecture experiment with frame-aligned reference center.

This is the corrected controlled baseline corresponding to the original
six-architecture GT/reference-assisted protocol, except that every MeanShift
now uses the full centered 6x6 = 36 satellite patches as requested.

Controlled search rule per frame:
  current-frame reference XY -> directly center the local 6x6 SAT window
  -> all 36 patches scored -> MeanShift -> visual position + variance.

There is NO waypoint-bank remapping and NO non-backtracking selector in this
controlled version. Therefore it can be compared fairly with the first
six-architecture controlled experiment. The autonomous experiment remains in
six_architecture_autoref_experiment.py and does not use the current-frame
reference coordinate for search.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker as rt
import six_architecture_autoref_experiment as core
from six_architecture_model import PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

ARCH_CHOICES = core.ARCH_CHOICES


def _make_state(route_name, visual):
    _, start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": core.StandardXYKalman(start_xy),
        "hidden": None,
        "previous_z": None,
    }


def forward_frame(arch, model, visual, uav_clip, reference_xy, state, device):
    """Use current-frame reference XY directly as the centered 6x6 MS center."""
    search_center_xy = np.asarray(reference_xy, dtype=np.float64).reshape(2).copy()

    base = core.centered_visual_meanshift(visual, uav_clip, search_center_xy)
    stage_xy = base["xy"]
    variance = base["variance"]
    z_uav = base["z_uav"]
    trace = {
        "search_center_xy": search_center_xy,
        "base_ms_xy": stage_xy,
        "base_ms_support": base["support"],
        "base_ms_candidate_count": base["candidate_count"],
    }
    gru_out = None

    symbols = arch[1:] if arch[0] == "M" else arch
    for symbol in symbols:
        if symbol == "M":
            correction = core.centered_visual_meanshift(visual, uav_clip, stage_xy)
            stage_xy = correction["xy"]
            variance = correction["variance"]
            trace["center_ms_xy"] = stage_xy
            trace["center_ms_support"] = correction["support"]
            trace["center_ms_candidate_count"] = correction["candidate_count"]
        elif symbol == "G":
            stage_xy, gru_out = core._apply_g(
                model, stage_xy, variance, z_uav, state
            )
            trace["gru_xy"] = stage_xy
        elif symbol == "K":
            stage_xy = core._apply_k(state, stage_xy, variance, device)
            trace["kalman_xy"] = stage_xy
        else:
            raise ValueError("unknown architecture symbol: %s" % symbol)

    state["previous_z"] = z_uav.detach()
    return stage_xy, variance, gru_out, trace


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "six_gtcenter_center6x6_%s_%s.pt" % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "six_architecture_gt_center_center6x6"
        / arch.lower()
    )


def train_architecture(arch, visual, route_a, device, epochs, lr, tbptt):
    model = PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        state = _make_state("route_A", visual)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []
        gru_input_errors = []

        for index in range(len(route_a)):
            uav_clip = route_a.uav_clip[index:index + 1].to(device).float()
            reference_xy = (
                route_a.gt_xy[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            _, _, gru_out, _ = forward_frame(
                arch, model, visual, uav_clip, reference_xy, state, device
            )
            if gru_out is None:
                raise RuntimeError("architecture %s did not execute GRU" % arch)

            target_xy = route_a.gt_xy[index:index + 1].to(device).float()
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_xy)

            with torch.no_grad():
                gru_input_xy = gru_out.corrected_xy - gru_out.correction_xy
                gru_input_errors.append(
                    float(torch.linalg.norm(gru_input_xy - target_xy, dim=1).mean().cpu())
                )

            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            if chunk_count >= int(tbptt) or index == len(route_a) - 1:
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
        mean_input_error = (
            float(np.mean(gru_input_errors)) if gru_input_errors else float("inf")
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            _checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": arch,
                    "model": best_state,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "reference_mode": "current-frame reference XY is direct local-search center",
                    "frame_aligned_reference_prior": True,
                    "reference_bank_used_for_search": False,
                    "meanshift_candidate_count": 36,
                    "forward_search": False,
                    "visual_position": "full centered 6x6 MeanShift",
                    "train_routes": ["route_A"],
                },
                _checkpoint_path(arch),
            )

        print(
            "[%s-gtcenter-center6x6] epoch=%03d/%d "
            "train_position_loss=%.6f gru_input_mle=%.3fm best=%.6f"
            % (arch, epoch, epochs, mean_loss, mean_input_error, best_loss),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no checkpoint produced")
    model.load_state_dict(best_state)
    return model


def load_architecture(arch, device):
    path = _checkpoint_path(arch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("meanshift_candidate_count") != 36:
        raise RuntimeError("checkpoint is not from the full centered 6x6 experiment")
    if payload.get("frame_aligned_reference_prior") is not True:
        raise RuntimeError("checkpoint is not the corrected GT-centered control")
    if payload.get("reference_bank_used_for_search") is not False:
        raise RuntimeError("checkpoint used the obsolete GT->reference-bank remapping")

    model = PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_architecture(arch, visual, model, route_name, cache, device):
    model.eval()
    state = _make_state(route_name, visual)
    rows = []
    final_errors = []
    base_ms_errors = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index:index + 1].to(device).float()
        reference_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        final_xy_t, variance, _, trace = forward_frame(
            arch, model, visual, uav_clip, reference_xy, state, device
        )
        final_xy = final_xy_t[0].detach().cpu().numpy().astype(np.float64)
        base_ms_xy = (
            trace["base_ms_xy"][0].detach().cpu().numpy().astype(np.float64)
        )

        final_error = float(np.linalg.norm(final_xy - reference_xy))
        base_error = float(np.linalg.norm(base_ms_xy - reference_xy))
        final_errors.append(final_error)
        base_ms_errors.append(base_error)

        row = {
            "frame_id": int(cache.frame_ids[index]),
            "image_path": cache.image_paths[index],
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "search_center_x": float(reference_xy[0]),
            "search_center_y": float(reference_xy[1]),
            "search_center_error_m": 0.0,
            "base_ms_x": float(base_ms_xy[0]),
            "base_ms_y": float(base_ms_xy[1]),
            "base_ms_error_m": base_error,
            "base_ms_candidate_count": int(trace["base_ms_candidate_count"]),
            "variance_x": float(variance[0, 0]),
            "variance_y": float(variance[0, 1]),
            "final_x": float(final_xy[0]),
            "final_y": float(final_xy[1]),
            "error_final_m": final_error,
        }
        for name in ("gru_xy", "kalman_xy", "center_ms_xy"):
            if name in trace:
                row[name + "_x"] = float(trace[name][0, 0])
                row[name + "_y"] = float(trace[name][0, 1])
        if "center_ms_candidate_count" in trace:
            row["center_ms_candidate_count"] = int(trace["center_ms_candidate_count"])
        rows.append(row)

        if state["hidden"] is not None:
            state["hidden"] = state["hidden"].detach()

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_gtcenter_center6x6_frames.csv" % (route_name, arch.lower())
    )
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
            "Architecture": arch,
            "Route": route_name,
            "SearchCenterMLE_m": 0.0,
            "BaseVisualMS_MLE_m": float(np.mean(base_ms_errors)),
            "CSV": str(csv_path),
            "ReferenceUsage": "current-frame reference XY directly centers local SAT search",
            "FrameAlignedReferencePrior": True,
            "ReferenceBankUsedForSearch": False,
            "VisualPosition": "always full centered 6x6 MeanShift; variance centered on MS observation",
            "MeanShiftCandidateCount": 36,
            "ForwardSearch": False,
        }
    )
    return summary


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare-visual", "train", "eval", "train-eval"),
        default="train-eval",
    )
    parser.add_argument("--arch", choices=ARCH_CHOICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epochs", type=int, default=int(getattr(config, "TEMPORAL_EPOCHS", 60))
    )
    parser.add_argument(
        "--lr", type=float, default=float(getattr(config, "TEMPORAL_LR", 3e-4))
    )
    parser.add_argument(
        "--tbptt", type=int, default=int(getattr(config, "TBPTT_STEPS", 32))
    )
    parser.add_argument(
        "--visual-epochs", type=int, default=int(getattr(config, "VISUAL_EPOCHS", 20))
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(getattr(config, "LOCAL_PRIOR_JITTER_M", 12.0))
    )
    args = parser.parse_args()

    device = rt.resolve_device(args.device)
    rt.set_seed(int(config.SEED))

    if args.mode == "prepare-visual":
        train_visual_retrieval_a_only(
            device=device,
            epochs=args.visual_epochs,
            jitter_m=args.jitter_m,
            resume=False,
        )
        return
    if args.arch is None:
        raise SystemExit("--arch is required")
    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)

    visual = FrozenVisualLocalizer(device)
    model = None

    if args.mode in ("train", "train-eval"):
        route_a = build_cache("route_A", visual, device)
        model = train_architecture(
            args.arch,
            visual,
            route_a,
            device,
            args.epochs,
            args.lr,
            args.tbptt,
        )

    if args.mode in ("eval", "train-eval"):
        if model is None:
            model = load_architecture(args.arch, device)

        results = {
            "architecture": args.arch,
            "train_route": "route_A",
            "test_routes": ["route_B", "route_C"],
            "reference_mode": "frame-aligned reference XY direct search center",
            "frame_aligned_reference_prior": True,
            "reference_bank_used_for_search": False,
            "meanshift_search": "full centered 6x6 for every MS stage",
            "meanshift_candidate_count": 36,
            "forward_search": False,
            "results": {},
        }

        for route_name in ("route_B", "route_C"):
            cache = build_cache(route_name, visual, device)
            summary = evaluate_architecture(
                args.arch, visual, model, route_name, cache, device
            )
            results["results"][route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)

        outdir = _output_dir(args.arch)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
