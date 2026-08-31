"""One-frame delayed visual-finalization experiment for pairwise UAV frames.

Purpose
-------
Test the user's delayed-correction idea independently from the existing
same-frame six-permutation ablation.

Sliding-time semantics
----------------------
For pair [I(t-1), I(t)]:
  1) MeanShift localizes frame t-1.
  2) KG or GK produces a provisional position x'_t for frame t.

For the next pair [I(t), I(t+1)]:
  3) MeanShift on I(t), centered at the previous x'_t, produces the FINAL
     position x_t for frame t.
  4) KG or GK then produces x'_(t+1).

Therefore every reported final position (except initialization) is delayed by
one frame and is produced by visual MeanShift, while KG/GK only proposes the
next search center.

Two variants
------------
KG:
    MS_t -> K(current visual update) -> G(pair t,t+1 one-step prediction)
    -> provisional x'_(t+1)

GK:
    MS_t -> G(pair t,t+1 one-step prediction) -> K(pseudo-measurement update)
    -> provisional x'_(t+1)

The GRU reuses PositionRefinementGRU, but here its corrected_xy is supervised
against the NEXT frame position. Thus its residual head is interpreted as a
one-step displacement from the current stage coordinate.

Training protocol
-----------------
Route A uses the current frame reference coordinate ONLY to center that frame's
6x6 visual MS during training. The target for G is the next Route-A reference
position. This isolates learning of the pairwise temporal predictor from a
search-center failure.

Evaluation protocol
-------------------
Routes B/C do NOT use per-frame reference coordinates to open the search.
Only the known route start initializes frame 0. Thereafter:
    previous provisional x'_t -> centered 6x6 MS on I(t) -> final x_t.
Reference coordinates are read only after inference for metrics.

Every MeanShift uses the full centered 6x6 = 36 satellite patches.
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

ARCH_CHOICES = ("KG", "GK")


def _xy_tensor(xy, device):
    return torch.as_tensor(xy, dtype=torch.float32, device=device).reshape(1, 2)


def _make_state(route_name, visual):
    _, start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": core.StandardXYKalman(start_xy),
        "hidden": None,
        "start_xy": np.asarray(start_xy, dtype=np.float64).reshape(2).copy(),
    }


@torch.no_grad()
def _encode_uav(visual, uav_clip):
    return visual.model.encode_uav_from_clip(uav_clip)


def _pair_g(model, stage_xy, variance, z_prev, z_next, state):
    """Predict the next-frame XY from current XY and UAV pair [t, t+1]."""
    out = model.forward_step(
        stage_xy=stage_xy,
        variance_xy=variance,
        z_uav=z_next,
        previous_z_uav=z_prev,
        hidden=state["hidden"],
    )
    state["hidden"] = out.hidden
    return out.corrected_xy, out


def _apply_k(state, xy, variance, device):
    filtered = state["kalman"].step(
        xy[0].detach().cpu().numpy(),
        variance[0].detach().cpu().numpy(),
    )
    return _xy_tensor(filtered, device)


def pair_step(
    arch,
    model,
    visual,
    uav_clip_current,
    uav_clip_next,
    search_center_xy,
    state,
    device,
):
    """Finalize current frame by MS, then predict the next frame provisionally."""
    ms = core.centered_visual_meanshift(
        visual, uav_clip_current, search_center_xy
    )
    final_current_xy = ms["xy"]
    variance = ms["variance"]
    z_current = ms["z_uav"]
    with torch.no_grad():
        z_next = _encode_uav(visual, uav_clip_next)

    trace = {
        "search_center_xy": np.asarray(search_center_xy, dtype=np.float64).reshape(2).copy(),
        "final_ms_xy": final_current_xy,
        "variance": variance,
        "ms_support": ms["support"],
        "candidate_count": ms["candidate_count"],
    }

    if arch == "KG":
        # K consumes the finalized current visual position. G then uses the
        # pair [I(t), I(t+1)] to extrapolate one frame ahead.
        kalman_current_xy = _apply_k(
            state, final_current_xy, variance, device
        )
        provisional_next_xy, gru_out = _pair_g(
            model,
            kalman_current_xy,
            variance,
            z_current,
            z_next,
            state,
        )
        trace["kalman_current_xy"] = kalman_current_xy
        trace["gru_next_xy"] = provisional_next_xy

    elif arch == "GK":
        # G first proposes the next-frame position. K then fuses this learned
        # proposal with its own CV prior. The visual variance from frame t is
        # reused as the measurement uncertainty for this controlled ablation.
        gru_next_xy, gru_out = _pair_g(
            model,
            final_current_xy,
            variance,
            z_current,
            z_next,
            state,
        )
        provisional_next_xy = _apply_k(
            state, gru_next_xy, variance, device
        )
        trace["gru_next_xy"] = gru_next_xy
        trace["kalman_next_xy"] = provisional_next_xy

    else:
        raise ValueError("unknown delayed architecture: %s" % arch)

    trace["provisional_next_xy"] = provisional_next_xy
    return final_current_xy, provisional_next_xy, gru_out, trace


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "delayed_pair_%s_center6x6_%s.pt" % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "delayed_pair_one_frame_center6x6"
        / arch.lower()
    )


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(
        route_name, config.ROUTE_ROOTS[idx], visual, device
    )


def train_architecture(arch, visual, route_a, device, epochs, lr, tbptt):
    """Train pairwise one-step G on Route A with reference-centered current MS."""
    if len(route_a) < 2:
        raise RuntimeError("Route A requires at least two frames")

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
        gru_next_errors = []
        chain_next_errors = []

        for index in range(len(route_a) - 1):
            uav_current = route_a.uav_clip[index:index + 1].to(device).float()
            uav_next = route_a.uav_clip[index + 1:index + 2].to(device).float()

            # Training-only controlled center: current frame reference directly
            # opens the full centered 6x6 visual window.
            reference_current = (
                route_a.gt_xy[index].detach().cpu().numpy().astype(np.float64)
            )

            _, provisional_next, gru_out, trace = pair_step(
                arch,
                model,
                visual,
                uav_current,
                uav_next,
                reference_current,
                state,
                device,
            )

            target_next = route_a.gt_xy[index + 1:index + 2].to(device).float()

            # K is a numerical filter and is not differentiable. Train G's
            # one-step proposal directly; for KG this is also the chain output.
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_next)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            with torch.no_grad():
                gru_next_errors.append(
                    float(
                        torch.linalg.norm(
                            gru_out.corrected_xy - target_next, dim=1
                        ).mean().cpu()
                    )
                )
                chain_next_errors.append(
                    float(
                        torch.linalg.norm(
                            provisional_next - target_next, dim=1
                        ).mean().cpu()
                    )
                )

            is_last = index == len(route_a) - 2
            if chunk_count >= int(tbptt) or is_last:
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
                chunk_loss = None
                chunk_count = 0

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        gru_next_mle = (
            float(np.mean(gru_next_errors)) if gru_next_errors else float("inf")
        )
        chain_next_mle = (
            float(np.mean(chain_next_errors)) if chain_next_errors else float("inf")
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
                    "protocol": "one-frame delayed finalization",
                    "pair_input": "[I(t), I(t+1)]",
                    "gru_target": "next-frame XY",
                    "train_search_center": "current-frame reference XY",
                    "eval_search_center": "previous provisional next-frame XY",
                    "final_output": "next-pair MeanShift of current frame",
                    "meanshift_candidate_count": 36,
                    "delay_frames": 1,
                    "train_routes": ["route_A"],
                },
                _checkpoint_path(arch),
            )

        print(
            "[%s-delayed1] epoch=%03d/%d train_next_loss=%.6f "
            "gru_next_mle=%.3fm chain_next_mle=%.3fm best=%.6f"
            % (
                arch,
                epoch,
                epochs,
                mean_loss,
                gru_next_mle,
                chain_next_mle,
                best_loss,
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no delayed-pair checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    return model


def load_architecture(arch, device):
    path = _checkpoint_path(arch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("protocol") != "one-frame delayed finalization":
        raise RuntimeError("checkpoint protocol mismatch")
    if payload.get("meanshift_candidate_count") != 36:
        raise RuntimeError("checkpoint is not full centered 6x6")

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
    """Closed-loop B/C evaluation with one-frame delayed MS finalization."""
    if len(cache) < 3:
        raise RuntimeError("delayed evaluation requires at least three frames")

    model.eval()
    state = _make_state(route_name, visual)
    search_center = state["start_xy"].copy()

    rows = []
    final_errors = []
    search_errors = []
    provisional_next_errors = []
    correction_gains = []
    correction_success = []

    # Pair [i, i+1] finalizes frame i, then predicts provisional frame i+1.
    # Frame 0 is initialization and is excluded from the delayed-final metric.
    # The last frame has no following pair, so it has provisional but no final.
    for index in range(len(cache) - 1):
        uav_current = cache.uav_clip[index:index + 1].to(device).float()
        uav_next = cache.uav_clip[index + 1:index + 2].to(device).float()

        reference_current = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        reference_next = cache.gt_xy[index + 1].cpu().numpy().astype(np.float64)

        final_current_t, provisional_next_t, _, trace = pair_step(
            arch,
            model,
            visual,
            uav_current,
            uav_next,
            search_center,
            state,
            device,
        )

        final_current = (
            final_current_t[0].detach().cpu().numpy().astype(np.float64)
        )
        provisional_next = (
            provisional_next_t[0].detach().cpu().numpy().astype(np.float64)
        )
        search_center_used = np.asarray(
            trace["search_center_xy"], dtype=np.float64
        ).reshape(2)

        final_error = float(np.linalg.norm(final_current - reference_current))
        search_error = float(np.linalg.norm(search_center_used - reference_current))
        provisional_next_error = float(
            np.linalg.norm(provisional_next - reference_next)
        )
        provisional_next_errors.append(provisional_next_error)

        is_initial = index == 0
        if not is_initial:
            final_errors.append(final_error)
            search_errors.append(search_error)
            correction_gains.append(search_error - final_error)
            correction_success.append(float(final_error < search_error))

        row = {
            "pair_current_index": int(index),
            "pair_next_index": int(index + 1),
            "current_frame_id": int(cache.frame_ids[index]),
            "next_frame_id": int(cache.frame_ids[index + 1]),
            "current_image_path": cache.image_paths[index],
            "next_image_path": cache.image_paths[index + 1],
            "is_initial_frame": bool(is_initial),
            "reference_current_x": float(reference_current[0]),
            "reference_current_y": float(reference_current[1]),
            "search_center_x": float(search_center_used[0]),
            "search_center_y": float(search_center_used[1]),
            "search_center_error_m": search_error,
            "final_ms_x": float(final_current[0]),
            "final_ms_y": float(final_current[1]),
            "final_error_m": final_error,
            "variance_x": float(trace["variance"][0, 0]),
            "variance_y": float(trace["variance"][0, 1]),
            "ms_candidate_count": int(trace["candidate_count"]),
            "reference_next_x": float(reference_next[0]),
            "reference_next_y": float(reference_next[1]),
            "provisional_next_x": float(provisional_next[0]),
            "provisional_next_y": float(provisional_next[1]),
            "provisional_next_error_m": provisional_next_error,
        }

        for name in (
            "kalman_current_xy",
            "gru_next_xy",
            "kalman_next_xy",
        ):
            if name in trace:
                value = trace[name][0].detach().cpu().numpy().astype(np.float64)
                row[name + "_x"] = float(value[0])
                row[name + "_y"] = float(value[1])

        if not is_initial:
            row["ms_correction_gain_m"] = search_error - final_error
            row["ms_improved_search_center"] = bool(final_error < search_error)

        rows.append(row)

        # This is the key delayed feedback: x'_(t+1) becomes the center used by
        # the NEXT pair's MeanShift to finalize frame t+1.
        search_center = provisional_next.copy()

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_delayed1_center6x6_frames.csv" % (route_name, arch.lower())
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
            "Protocol": "one-frame delayed MS finalization",
            "LatencyFrames": 1,
            "FinalizedFrameCount": int(len(final_errors)),
            "ProvisionalNextMLE_m": float(np.mean(provisional_next_errors)),
            "SearchCenterMLE_m": float(np.mean(search_errors)),
            "MeanMSCorrectionGain_m": float(np.mean(correction_gains)),
            "MSCorrectionSuccess_pct": 100.0 * float(np.mean(correction_success)),
            "InitialSearchCenter": "known planned-route start only",
            "TrainReferenceCentered": True,
            "FrameAlignedReferencePriorEval": False,
            "FinalOutput": "MeanShift on I(t) when pair [I(t), I(t+1)] arrives",
            "NextSearchCenter": "previous pair KG/GK provisional x'_t",
            "MeanShiftCandidateCount": 36,
            "ForwardSearch": False,
            "CSV": str(csv_path),
        }
    )
    return summary


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
        "--epochs",
        type=int,
        default=int(getattr(config, "TEMPORAL_EPOCHS", 60)),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(getattr(config, "TEMPORAL_LR", 3e-4)),
    )
    parser.add_argument(
        "--tbptt",
        type=int,
        default=int(getattr(config, "TBPTT_STEPS", 32)),
    )
    parser.add_argument(
        "--visual-epochs",
        type=int,
        default=int(getattr(config, "VISUAL_EPOCHS", 20)),
    )
    parser.add_argument(
        "--jitter-m",
        type=float,
        default=float(getattr(config, "LOCAL_PRIOR_JITTER_M", 12.0)),
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
            "protocol": "one-frame delayed MS finalization",
            "latency_frames": 1,
            "train_search_center": "current-frame reference XY",
            "eval_initialization": "known planned-route start only",
            "eval_search_center": "previous provisional next-frame XY",
            "final_output": "next-pair current-frame MeanShift",
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
