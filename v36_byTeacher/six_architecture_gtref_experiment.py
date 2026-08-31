"""One-frame-delayed GT/reference-centered six-architecture control.

Teacher-required alignment:
  frame t arrives -> direct-reference centered full 6x6 M_t -> pending only
  frame t+1 arrives -> direct-reference centered full 6x6 M_{t+1}
  only then finalize target frame t using the pair [z_t, z_{t+1}].

Examples:
  MKG: M_t -> K_t -> G([z_t,z_{t+1}]) -> Final(t)
  MGK: M_t -> G([z_t,z_{t+1}]) -> K_t -> Final(t)

The last frame is not evaluated because it has no t+1 look-ahead.
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

ARCH_CHOICES = core.ARCH_CHOICES
TEMPORAL_ALIGNMENT = core.TEMPORAL_ALIGNMENT
LOOKAHEAD_FRAMES = core.LOOKAHEAD_FRAMES


def _make_state(route_name, visual):
    _, start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": core.StandardXYKalman(start_xy),
        "hidden": None,
    }


def _prepare_gt_visual(visual, uav_clip, reference_xy):
    """Controlled only: current frame reference XY directly centers M."""
    center_xy = np.asarray(reference_xy, dtype=np.float64).reshape(2).copy()
    base = core.centered_visual_meanshift(visual, uav_clip, center_xy)
    return {
        "uav_clip": uav_clip,
        "base": base,
        "selection_query_xy": center_xy.copy(),
        "selected_ref_xy": center_xy.copy(),
        "selected_ref_index": -1,
    }


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "six_gtcenter_delay1_center6x6_%s_%s.pt"
        % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "six_architecture_gt_center_delay1_center6x6"
        / arch.lower()
    )


def train_architecture(arch, visual, route_a, device, epochs, lr, tbptt):
    if len(route_a) < 2:
        raise RuntimeError("one-frame delayed training requires at least 2 frames")

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

        first_uav = route_a.uav_clip[0:1].to(device).float()
        first_reference = (
            route_a.gt_xy[0].detach().cpu().numpy().astype(np.float64)
        )
        pending = _prepare_gt_visual(visual, first_uav, first_reference)

        for lookahead_index in range(1, len(route_a)):
            lookahead_uav = (
                route_a.uav_clip[lookahead_index:lookahead_index + 1]
                .to(device)
                .float()
            )
            lookahead_reference = (
                route_a.gt_xy[lookahead_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            lookahead = _prepare_gt_visual(
                visual, lookahead_uav, lookahead_reference
            )

            target_index = lookahead_index - 1
            _, _, gru_out, _ = core.finalize_pending(
                arch, model, visual, pending, lookahead, state, device
            )
            if gru_out is None:
                raise RuntimeError("architecture %s did not execute GRU" % arch)

            target_xy = (
                route_a.gt_xy[target_index:target_index + 1]
                .to(device)
                .float()
            )
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_xy)

            with torch.no_grad():
                gru_input_xy = (
                    gru_out.corrected_xy - gru_out.correction_xy
                )
                gru_input_errors.append(
                    float(
                        torch.linalg.norm(
                            gru_input_xy - target_xy, dim=1
                        ).mean().cpu()
                    )
                )

            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            is_last = lookahead_index == len(route_a) - 1
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

            pending = lookahead

        mean_loss = (
            float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        )
        mean_input_error = (
            float(np.mean(gru_input_errors))
            if gru_input_errors
            else float("inf")
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            _checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": arch,
                    "model": best_state,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "reference_mode": (
                        "frame-aligned reference XY directly centers each "
                        "full 6x6 visual search"
                    ),
                    "frame_aligned_reference_prior": True,
                    "reference_bank_used_for_search": False,
                    "temporal_alignment": TEMPORAL_ALIGNMENT,
                    "lookahead_frames": LOOKAHEAD_FRAMES,
                    "target_finalized_on_next_frame": True,
                    "gru_visual_pair": "[z_t, z_t_plus_1]",
                    "meanshift_candidate_count": 36,
                    "forward_search": False,
                    "train_routes": ["route_A"],
                },
                _checkpoint_path(arch),
            )

        print(
            "[%s-gtcenter-delay1-center6x6] epoch=%03d/%d "
            "train_position_loss=%.6f gru_input_mle=%.3fm best=%.6f"
            % (
                arch,
                epoch,
                epochs,
                mean_loss,
                mean_input_error,
                best_loss,
            ),
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
        raise RuntimeError("checkpoint is not full centered 6x6")
    if payload.get("frame_aligned_reference_prior") is not True:
        raise RuntimeError("checkpoint is not GT/reference-centered")
    if payload.get("reference_bank_used_for_search") is not False:
        raise RuntimeError("checkpoint used an obsolete route-bank remapping")
    if payload.get("temporal_alignment") != TEMPORAL_ALIGNMENT:
        raise RuntimeError("checkpoint is not the one-frame delayed experiment")

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
    if len(cache) < 2:
        raise RuntimeError("one-frame delayed evaluation requires at least 2 frames")

    model.eval()
    state = _make_state(route_name, visual)
    rows = []
    final_errors = []
    base_ms_errors = []

    first_uav = cache.uav_clip[0:1].to(device).float()
    first_reference = (
        cache.gt_xy[0].cpu().numpy().astype(np.float64)
    )
    pending = _prepare_gt_visual(visual, first_uav, first_reference)

    for lookahead_index in range(1, len(cache)):
        lookahead_uav = (
            cache.uav_clip[lookahead_index:lookahead_index + 1]
            .to(device)
            .float()
        )
        lookahead_reference = (
            cache.gt_xy[lookahead_index]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        lookahead = _prepare_gt_visual(
            visual, lookahead_uav, lookahead_reference
        )

        target_index = lookahead_index - 1
        final_xy_t, variance, _, trace = core.finalize_pending(
            arch, model, visual, pending, lookahead, state, device
        )

        target_reference = (
            cache.gt_xy[target_index].cpu().numpy().astype(np.float64)
        )
        final_xy = (
            final_xy_t[0].detach().cpu().numpy().astype(np.float64)
        )
        base_ms_xy = (
            trace["base_ms_xy"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        final_error = float(np.linalg.norm(final_xy - target_reference))
        base_error = float(np.linalg.norm(base_ms_xy - target_reference))
        final_errors.append(final_error)
        base_ms_errors.append(base_error)

        row = {
            "target_frame_id": int(cache.frame_ids[target_index]),
            "lookahead_frame_id": int(cache.frame_ids[lookahead_index]),
            "target_image_path": cache.image_paths[target_index],
            "lookahead_image_path": cache.image_paths[lookahead_index],
            "reference_x": float(target_reference[0]),
            "reference_y": float(target_reference[1]),
            "search_center_x": float(target_reference[0]),
            "search_center_y": float(target_reference[1]),
            "search_center_error_m": 0.0,
            "base_ms_x": float(base_ms_xy[0]),
            "base_ms_y": float(base_ms_xy[1]),
            "base_ms_error_m": base_error,
            "base_ms_candidate_count": int(
                trace["base_ms_candidate_count"]
            ),
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
        rows.append(row)

        if state["hidden"] is not None:
            state["hidden"] = state["hidden"].detach()
        pending = lookahead

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_gtcenter_delay1_center6x6_frames.csv"
        % (route_name, arch.lower())
    )
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = rt.metric_summary(final_errors)
    summary.update(
        {
            "Architecture": arch,
            "Route": route_name,
            "SearchCenterMLE_m": 0.0,
            "BaseVisualMS_MLE_m": float(np.mean(base_ms_errors)),
            "EvaluatedTargetFrames": len(rows),
            "DroppedLastFrameWithoutLookahead": True,
            "TemporalAlignment": TEMPORAL_ALIGNMENT,
            "LookaheadFrames": LOOKAHEAD_FRAMES,
            "GRUVisualPair": "[z_t, z_t_plus_1]",
            "CSV": str(csv_path),
            "ReferenceUsage": (
                "frame-aligned reference XY directly centers each local SAT search"
            ),
            "FrameAlignedReferencePrior": True,
            "ReferenceBankUsedForSearch": False,
            "MeanShiftCandidateCount": 36,
            "ForwardSearch": False,
        }
    )
    return summary


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(
        route_name, config.ROUTE_ROOTS[idx], visual, device
    )


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
            "reference_mode": (
                "frame-aligned reference XY direct search center"
            ),
            "frame_aligned_reference_prior": True,
            "reference_bank_used_for_search": False,
            "temporal_alignment": TEMPORAL_ALIGNMENT,
            "lookahead_frames": LOOKAHEAD_FRAMES,
            "target_finalized_on_next_frame": True,
            "gru_visual_pair": "[z_t, z_t_plus_1]",
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
        with (outdir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
