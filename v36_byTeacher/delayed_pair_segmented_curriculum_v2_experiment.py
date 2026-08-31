"""Adaptive segmented closed-loop curriculum for delayed MS -> KG/GK tracking.

This v2 fixes two problems in the first segmented experiment:
1) starting with 16-frame rollouts was already outside the reliable 6x6 local-search basin;
2) checkpoint selection compared losses across different rollout lengths, which biased the
   final model toward easy short-rollout epochs.

Training (Route A)
------------------
Each temporal segment receives supervision ONLY at its start:
  * initial position = segment-start reference XY;
  * initial velocity = reference[start+1] - reference[start].
After that, the segment is fully closed-loop:
  previous provisional x'_t -> centered 6x6 MS on I(t) -> KG/GK -> x'_(t+1).
Frame-aligned references inside the segment are used only as training targets/diagnostics.

Curriculum
----------
Start from very short rollouts (default 4 frames).  Segment length is NOT increased simply
because the epoch number increased.  It advances only after closed-loop search-center metrics
(excluding the teacher-initialized first pair of every segment) are stable for several epochs.
Overlapping segments provide many independent recovery/init points across Route A.

Checkpoint selection
--------------------
A checkpoint from a longer successfully reached rollout always outranks one from a shorter
rollout.  Loss is compared only within the same rollout length.  This prevents epoch-1/2 short
segments from remaining the permanent 'best' checkpoint.

Evaluation (Routes B/C)
-----------------------
One full-route closed-loop rollout, no segment reset and no frame-aligned reference search
center.  Only the known planned-route start is used for position initialization.  Initial
velocity uses the planned-route tangent and the median Route-A training speed prior.

Every MeanShift is full centered 6x6 = 36 candidates; no forward 3x6 selection.
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
import delayed_pair_ms_kg_gk_experiment as base
import delayed_pair_segmented_closed_loop_experiment as old
from six_architecture_model import PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

ARCH_CHOICES = ("KG", "GK")


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "delayed_pair_segmented_curriculum_v2_%s_center6x6_%s.pt"
        % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "delayed_pair_segmented_curriculum_v2_center6x6"
        / arch.lower()
    )


def _save_checkpoint(
    arch,
    model,
    epoch,
    loss,
    segment_frames,
    train_speed_prior,
    args,
):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": arch,
            "model": state,
            "epoch": int(epoch),
            "train_loss": float(loss),
            "protocol": "adaptive segmented closed-loop curriculum v2",
            "segment_frames_reached": int(segment_frames),
            "segment_frames_start": int(args.segment_frames_start),
            "segment_frames_end": int(args.segment_frames_end),
            "segment_step": int(args.segment_step),
            "advance_search_mle_m": float(args.advance_search_mle),
            "advance_within20_pct": float(args.advance_within20),
            "advance_patience": int(args.advance_patience),
            "segment_start_position": "Route-A reference XY once per training segment",
            "segment_initial_velocity": "Route-A reference delta once per training segment",
            "inside_segment_search_center": "previous provisional next-frame XY",
            "gru_target": "next-frame Route-A reference XY",
            "eval_segment_resets": False,
            "eval_frame_aligned_reference_prior": False,
            "eval_initial_speed_source": "median Route-A training speed + planned-route tangent",
            "train_speed_prior_mpf": float(train_speed_prior),
            "meanshift_candidate_count": 36,
            "delay_frames": 1,
            "train_routes": ["route_A"],
        },
        _checkpoint_path(arch),
    )
    return state


def train_architecture(arch, visual, route_a, device, args):
    if len(route_a) < 4:
        raise RuntimeError("Route A requires at least four frames")

    model = PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    train_speed_prior = old._training_speed_prior(route_a)
    current_seg_frames = int(args.segment_frames_start)
    stable_epochs = 0

    # Lexicographic checkpoint criterion: rollout length first, loss second.
    best_stage = -1
    best_stage_loss = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        seg_frames = int(current_seg_frames)

        # 50% overlap gives many independent segment starts while keeping each
        # segment internally closed-loop after its first pair.
        stride = max(1, seg_frames // 2)
        starts = list(range(0, len(route_a) - 1, stride))
        tail_start = max(0, len(route_a) - seg_frames)
        if tail_start not in starts and tail_start < len(route_a) - 1:
            starts.append(tail_start)
        starts = sorted(set(starts))

        epoch_losses = []
        provisional_errors = []
        final_ms_errors = []

        # Gate metrics explicitly exclude the first pair in each segment because
        # that pair uses the segment-start supervised initialization.
        gate_search_errors = []
        gate_within20 = []

        for seg_start in starts:
            seg_end = min(len(route_a), seg_start + seg_frames)
            if seg_end - seg_start < 2:
                continue

            initial_xy = old._xy_np(route_a.gt_xy[seg_start])
            initial_velocity = old._segment_initial_velocity(route_a, seg_start)
            state = old._make_state(initial_xy, initial_velocity)
            search_center = initial_xy.copy()

            optimizer.zero_grad(set_to_none=True)
            segment_loss = None
            pair_count = 0

            for index in range(seg_start, seg_end - 1):
                uav_current = route_a.uav_clip[index:index + 1].to(device).float()
                uav_next = route_a.uav_clip[index + 1:index + 2].to(device).float()

                final_current_t, provisional_next_t, gru_out, _ = base.pair_step(
                    arch,
                    model,
                    visual,
                    uav_current,
                    uav_next,
                    search_center,
                    state,
                    device,
                )

                target_current = route_a.gt_xy[index:index + 1].to(device).float()
                target_next = route_a.gt_xy[index + 1:index + 2].to(device).float()

                # G learns the next-frame proposal.  Search-center feedback is
                # deliberately detached because MS/candidate selection is not
                # differentiable; nevertheless the next step receives the actual
                # model-generated closed-loop center.
                loss = F.smooth_l1_loss(gru_out.corrected_xy, target_next)
                segment_loss = loss if segment_loss is None else segment_loss + loss
                pair_count += 1

                with torch.no_grad():
                    provisional_error = float(
                        torch.linalg.norm(provisional_next_t - target_next, dim=1)
                        .mean().cpu()
                    )
                    final_error = float(
                        torch.linalg.norm(final_current_t - target_current, dim=1)
                        .mean().cpu()
                    )
                    provisional_errors.append(provisional_error)
                    final_ms_errors.append(final_error)

                    # Only true feedback steps participate in curriculum gating.
                    if index > seg_start:
                        target_current_np = old._xy_np(target_current[0])
                        search_error = float(
                            np.linalg.norm(
                                np.asarray(search_center, dtype=np.float64).reshape(2)
                                - target_current_np
                            )
                        )
                        gate_search_errors.append(search_error)
                        gate_within20.append(float(search_error <= 20.0))

                search_center = old._xy_np(provisional_next_t[0])

            if pair_count == 0:
                continue

            normalized = segment_loss / float(pair_count)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            epoch_losses.append(float(normalized.detach().cpu()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        provisional_mle = (
            float(np.mean(provisional_errors)) if provisional_errors else float("inf")
        )
        final_ms_mle = (
            float(np.mean(final_ms_errors)) if final_ms_errors else float("inf")
        )
        closed_search_mle = (
            float(np.mean(gate_search_errors)) if gate_search_errors else float("inf")
        )
        closed_within20 = (
            100.0 * float(np.mean(gate_within20)) if gate_within20 else 0.0
        )

        # Longer achieved rollout always outranks shorter rollout.  Within the
        # same rollout length, choose the smaller training loss.
        if seg_frames > best_stage or (
            seg_frames == best_stage and mean_loss < best_stage_loss
        ):
            best_stage = seg_frames
            best_stage_loss = mean_loss
            best_state = _save_checkpoint(
                arch,
                model,
                epoch,
                mean_loss,
                seg_frames,
                train_speed_prior,
                args,
            )

        gate_ok = (
            np.isfinite(closed_search_mle)
            and closed_search_mle <= float(args.advance_search_mle)
            and closed_within20 >= float(args.advance_within20)
        )
        stable_epochs = stable_epochs + 1 if gate_ok else 0

        advance = False
        if (
            stable_epochs >= int(args.advance_patience)
            and current_seg_frames < int(args.segment_frames_end)
        ):
            current_seg_frames = min(
                int(args.segment_frames_end),
                current_seg_frames + int(args.segment_step),
            )
            stable_epochs = 0
            advance = True

        print(
            "[%s-segcv2] epoch=%03d/%d segment_frames=%d "
            "train_next_loss=%.6f provisional_mle=%.3fm final_ms_mle=%.3fm "
            "closed_search_mle=%.3fm closed_within20=%.2f%% stable=%d/%d "
            "best_stage=%d%s"
            % (
                arch,
                epoch,
                args.epochs,
                seg_frames,
                mean_loss,
                provisional_mle,
                final_ms_mle,
                closed_search_mle,
                closed_within20,
                stable_epochs,
                args.advance_patience,
                best_stage,
                " -> advance_to_%d" % current_seg_frames if advance else "",
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no adaptive segmented checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    return model, float(train_speed_prior), int(best_stage)


def load_architecture(arch, device):
    path = _checkpoint_path(arch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("protocol") != "adaptive segmented closed-loop curriculum v2":
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
    return (
        model,
        float(payload.get("train_speed_prior_mpf", 0.0)),
        int(payload.get("segment_frames_reached", 0)),
    )


@torch.no_grad()
def evaluate_architecture(arch, visual, model, route_name, cache, device, train_speed_prior):
    if len(cache) < 3:
        raise RuntimeError("evaluation requires at least three frames")

    _, planned_start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    initial_velocity = old._planned_initial_velocity(
        route_name, visual, train_speed_prior
    )
    state = old._make_state(planned_start_xy, initial_velocity)
    search_center = np.asarray(planned_start_xy, dtype=np.float64).reshape(2).copy()

    model.eval()
    rows = []
    final_errors = []
    provisional_errors = []
    search_errors = []
    correction_gains = []
    correction_success = []

    for index in range(len(cache) - 1):
        uav_current = cache.uav_clip[index:index + 1].to(device).float()
        uav_next = cache.uav_clip[index + 1:index + 2].to(device).float()

        # No B/C frame-aligned reference is read before inference.
        final_current_t, provisional_next_t, _, trace = base.pair_step(
            arch,
            model,
            visual,
            uav_current,
            uav_next,
            search_center,
            state,
            device,
        )
        final_current = old._xy_np(final_current_t[0])
        provisional_next = old._xy_np(provisional_next_t[0])
        search_center_used = np.asarray(
            trace["search_center_xy"], dtype=np.float64
        ).reshape(2)

        # Metrics only, after inference.
        reference_current = old._xy_np(cache.gt_xy[index])
        reference_next = old._xy_np(cache.gt_xy[index + 1])
        final_error = float(np.linalg.norm(final_current - reference_current))
        provisional_error = float(np.linalg.norm(provisional_next - reference_next))
        search_error = float(np.linalg.norm(search_center_used - reference_current))

        provisional_errors.append(provisional_error)
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
            "is_initial_frame": bool(is_initial),
            "search_center_x": float(search_center_used[0]),
            "search_center_y": float(search_center_used[1]),
            "final_ms_x": float(final_current[0]),
            "final_ms_y": float(final_current[1]),
            "provisional_next_x": float(provisional_next[0]),
            "provisional_next_y": float(provisional_next[1]),
            "reference_current_x": float(reference_current[0]),
            "reference_current_y": float(reference_current[1]),
            "reference_next_x": float(reference_next[0]),
            "reference_next_y": float(reference_next[1]),
            "search_center_error_m": search_error,
            "final_error_m": final_error,
            "provisional_next_error_m": provisional_error,
            "ms_candidate_count": int(trace["candidate_count"]),
        }
        if not is_initial:
            row["ms_correction_gain_m"] = search_error - final_error
            row["ms_improved_search_center"] = bool(final_error < search_error)
        for name in ("kalman_current_xy", "gru_next_xy", "kalman_next_xy"):
            if name in trace:
                value = old._xy_np(trace[name][0])
                row[name + "_x"] = float(value[0])
                row[name + "_y"] = float(value[1])
        rows.append(row)

        # Entire B/C route stays closed-loop; never reset from reference labels.
        search_center = provisional_next.copy()

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_segcv2_full_closedloop_frames.csv" % (route_name, arch.lower())
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
            "Protocol": "adaptive segmented closed-loop train v2; full-route closed-loop eval",
            "LatencyFrames": 1,
            "FinalizedFrameCount": int(len(final_errors)),
            "ProvisionalNextMLE_m": float(np.mean(provisional_errors)),
            "SearchCenterMLE_m": float(np.mean(search_errors)),
            "MeanMSCorrectionGain_m": float(np.mean(correction_gains)),
            "MSCorrectionSuccess_pct": 100.0 * float(np.mean(correction_success)),
            "EvalFrameAlignedReferencePrior": False,
            "EvalSegmentResets": False,
            "EvalInitialPosition": "known planned-route start",
            "EvalInitialVelocity": "planned-route tangent * Route-A median training speed",
            "TrainSpeedPrior_m_per_frame": float(train_speed_prior),
            "MeanShiftCandidateCount": 36,
            "ForwardSearch": False,
            "CSV": str(csv_path),
        }
    )
    return summary


def build_cache(route_name, visual, device):
    return old.build_cache(route_name, visual, device)


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
    parser.add_argument("--segment-frames-start", type=int, default=4)
    parser.add_argument("--segment-frames-end", type=int, default=48)
    parser.add_argument("--segment-step", type=int, default=2)
    parser.add_argument("--advance-search-mle", type=float, default=12.0)
    parser.add_argument("--advance-within20", type=float, default=85.0)
    parser.add_argument("--advance-patience", type=int, default=2)
    parser.add_argument(
        "--visual-epochs", type=int, default=int(getattr(config, "VISUAL_EPOCHS", 20))
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(getattr(config, "LOCAL_PRIOR_JITTER_M", 12.0))
    )
    args = parser.parse_args()

    if args.segment_frames_start < 3 or args.segment_frames_end < 3:
        raise SystemExit("segment frame counts must be >= 3")
    if args.segment_frames_end < args.segment_frames_start:
        raise SystemExit("--segment-frames-end must be >= --segment-frames-start")
    if args.segment_step < 1 or args.advance_patience < 1:
        raise SystemExit("segment step and advance patience must be >= 1")

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
    train_speed_prior = None
    best_stage = None

    if args.mode in ("train", "train-eval"):
        route_a = build_cache("route_A", visual, device)
        model, train_speed_prior, best_stage = train_architecture(
            args.arch, visual, route_a, device, args
        )

    if args.mode in ("eval", "train-eval"):
        if model is None:
            model, train_speed_prior, best_stage = load_architecture(args.arch, device)

        results = {
            "architecture": args.arch,
            "train_route": "route_A",
            "test_routes": ["route_B", "route_C"],
            "protocol": "adaptive segmented closed-loop curriculum v2",
            "best_segment_frames_reached": int(best_stage),
            "training_segment_initialization": "segment-start reference XY + reference velocity only",
            "training_inside_segment_search": "previous provisional next-frame XY",
            "evaluation_segment_resets": False,
            "evaluation_frame_aligned_reference_prior": False,
            "evaluation_initial_velocity": "planned-route tangent * Route-A median training speed",
            "train_speed_prior_m_per_frame": float(train_speed_prior),
            "meanshift_candidate_count": 36,
            "forward_search": False,
            "results": {},
        }

        for route_name in ("route_B", "route_C"):
            cache = build_cache(route_name, visual, device)
            summary = evaluate_architecture(
                args.arch,
                visual,
                model,
                route_name,
                cache,
                device,
                train_speed_prior,
            )
            results["results"][route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)

        outdir = _output_dir(args.arch)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
