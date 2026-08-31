"""Segmented closed-loop training for one-frame delayed MS finalization.

Goal
----
Bridge the gap between reference-centered training and fully closed-loop testing.

Training on Route A
-------------------
Route A is split into short temporal segments.  For each segment ONLY the
segment start is initialized from supervision:
  * initial XY = first reference position of the segment;
  * initial velocity = reference[start+1] - reference[start].

Inside the segment there is NO per-frame reference search center.  Instead:
  previous provisional x'_t -> centered 6x6 MS on I(t) -> final x_t
  -> KG/GK with [I(t), I(t+1)] -> provisional x'_(t+1)
  -> next frame search center.

Reference positions inside the segment are used only as training targets and
for diagnostics.  Segment length follows a curriculum from short to longer
closed-loop rollouts so early errors do not poison the entire route from the
first epoch.

Evaluation on Routes B/C
------------------------
No per-frame reference is used for inference and there are NO segment resets.
The whole route is one closed-loop rollout.  Initialization uses:
  * known planned-route start;
  * an initial velocity obtained from the planned-route tangent multiplied by
    the median Route-A training speed prior.
Thus B/C frame-aligned references are read only after inference for metrics.

Variants
--------
KG: MS(t) -> K -> G([I(t),I(t+1)]) -> provisional x'_(t+1)
GK: MS(t) -> G([I(t),I(t+1)]) -> K -> provisional x'_(t+1)

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
import delayed_pair_ms_kg_gk_experiment as base
from six_architecture_model import PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

ARCH_CHOICES = ("KG", "GK")


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


def _xy_np(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64).reshape(2)


def _segment_initial_velocity(cache, start_index):
    if start_index + 1 >= len(cache):
        return np.zeros(2, dtype=np.float64)
    a = _xy_np(cache.gt_xy[start_index])
    b = _xy_np(cache.gt_xy[start_index + 1])
    return b - a


def _training_speed_prior(route_a):
    xy = route_a.gt_xy.detach().cpu().numpy().astype(np.float64)
    if len(xy) < 2:
        return 0.0
    speed = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    speed = speed[np.isfinite(speed)]
    speed = speed[speed > 1e-6]
    if len(speed) == 0:
        return 0.0
    return float(np.median(speed))


def _planned_initial_velocity(route_name, visual, speed_mpf):
    waypoint_xy = np.asarray(
        rt.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon),
        dtype=np.float64,
    ).reshape(-1, 2)
    direction = None
    for a, b in zip(waypoint_xy[:-1], waypoint_xy[1:]):
        delta = b - a
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            direction = delta / norm
            break
    if direction is None:
        direction = np.zeros(2, dtype=np.float64)
    return direction * float(speed_mpf)


def _make_state(initial_xy, initial_velocity):
    initial_xy = np.asarray(initial_xy, dtype=np.float64).reshape(2)
    initial_velocity = np.asarray(initial_velocity, dtype=np.float64).reshape(2)
    kalman = base.core.StandardXYKalman(initial_xy)
    kalman.x[2:] = initial_velocity
    return {
        "kalman": kalman,
        "hidden": None,
        "start_xy": initial_xy.copy(),
        "initial_velocity": initial_velocity.copy(),
    }


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "delayed_pair_segmented_closedloop_%s_center6x6_%s.pt"
        % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "delayed_pair_segmented_closed_loop_center6x6"
        / arch.lower()
    )


def _segment_length(epoch, epochs, start_frames, end_frames):
    if int(epochs) <= 1:
        return int(end_frames)
    alpha = float(epoch - 1) / float(max(1, epochs - 1))
    value = int(round(float(start_frames) + alpha * (float(end_frames) - float(start_frames))))
    return max(3, value)


def train_architecture(
    arch,
    visual,
    route_a,
    device,
    epochs,
    lr,
    segment_frames_start,
    segment_frames_end,
):
    if len(route_a) < 3:
        raise RuntimeError("Route A requires at least three frames")

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

    train_speed_prior = _training_speed_prior(route_a)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        seg_frames = _segment_length(
            epoch, epochs, segment_frames_start, segment_frames_end
        )
        stride = max(1, seg_frames - 1)
        starts = list(range(0, len(route_a) - 1, stride))

        epoch_losses = []
        provisional_errors = []
        search_errors = []
        final_ms_errors = []
        segment_survival = []

        for seg_start in starts:
            seg_end = min(len(route_a), seg_start + seg_frames)
            if seg_end - seg_start < 2:
                continue

            initial_xy = _xy_np(route_a.gt_xy[seg_start])
            initial_velocity = _segment_initial_velocity(route_a, seg_start)
            state = _make_state(initial_xy, initial_velocity)
            search_center = initial_xy.copy()

            optimizer.zero_grad(set_to_none=True)
            segment_loss = None
            pair_count = 0
            survived_pairs = 0

            for index in range(seg_start, seg_end - 1):
                uav_current = route_a.uav_clip[index:index + 1].to(device).float()
                uav_next = route_a.uav_clip[index + 1:index + 2].to(device).float()

                final_current_t, provisional_next_t, gru_out, trace = base.pair_step(
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

                # Supervision trains the next-frame proposal.  The search center
                # itself remains fully closed-loop after the segment start.
                loss = F.smooth_l1_loss(gru_out.corrected_xy, target_next)
                segment_loss = loss if segment_loss is None else segment_loss + loss
                pair_count += 1

                with torch.no_grad():
                    provisional_error = float(
                        torch.linalg.norm(
                            provisional_next_t - target_next, dim=1
                        ).mean().cpu()
                    )
                    final_error = float(
                        torch.linalg.norm(
                            final_current_t - target_current, dim=1
                        ).mean().cpu()
                    )
                    target_current_np = _xy_np(target_current[0])
                    search_error = float(
                        np.linalg.norm(
                            np.asarray(search_center, dtype=np.float64).reshape(2)
                            - target_current_np
                        )
                    )
                    provisional_errors.append(provisional_error)
                    final_ms_errors.append(final_error)
                    search_errors.append(search_error)
                    if search_error <= 20.0:
                        survived_pairs += 1

                # Critical closed-loop feedback: there is no per-frame teacher
                # center inside the segment.
                search_center = _xy_np(provisional_next_t[0])

            if pair_count == 0:
                continue

            normalized = segment_loss / float(pair_count)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            epoch_losses.append(float(normalized.detach().cpu()))
            segment_survival.append(100.0 * survived_pairs / float(pair_count))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        provisional_mle = (
            float(np.mean(provisional_errors)) if provisional_errors else float("inf")
        )
        search_mle = float(np.mean(search_errors)) if search_errors else float("inf")
        final_ms_mle = (
            float(np.mean(final_ms_errors)) if final_ms_errors else float("inf")
        )
        survival_pct = (
            float(np.mean(segment_survival)) if segment_survival else 0.0
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
                    "protocol": "segmented closed-loop one-frame delayed finalization",
                    "segment_frames_start": int(segment_frames_start),
                    "segment_frames_end": int(segment_frames_end),
                    "segment_start_position": "Route-A reference at segment start only",
                    "segment_initial_velocity": "Route-A reference delta at segment start only",
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

        print(
            "[%s-segclosed] epoch=%03d/%d segment_frames=%d "
            "train_next_loss=%.6f provisional_mle=%.3fm search_mle=%.3fm "
            "final_ms_mle=%.3fm within20=%.2f%% best=%.6f"
            % (
                arch,
                epoch,
                epochs,
                seg_frames,
                mean_loss,
                provisional_mle,
                search_mle,
                final_ms_mle,
                survival_pct,
                best_loss,
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no segmented closed-loop checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    return model, float(train_speed_prior)


def load_architecture(arch, device):
    path = _checkpoint_path(arch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("protocol") != "segmented closed-loop one-frame delayed finalization":
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
    return model, float(payload.get("train_speed_prior_mpf", 0.0))


@torch.no_grad()
def evaluate_architecture(
    arch,
    visual,
    model,
    route_name,
    cache,
    device,
    train_speed_prior,
):
    if len(cache) < 3:
        raise RuntimeError("evaluation requires at least three frames")

    _, planned_start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    initial_velocity = _planned_initial_velocity(
        route_name, visual, train_speed_prior
    )
    state = _make_state(planned_start_xy, initial_velocity)
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

        # Inference is completed before any B/C frame-aligned reference is read.
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
        final_current = _xy_np(final_current_t[0])
        provisional_next = _xy_np(provisional_next_t[0])
        search_center_used = np.asarray(
            trace["search_center_xy"], dtype=np.float64
        ).reshape(2)

        # Metrics only, after inference.
        reference_current = _xy_np(cache.gt_xy[index])
        reference_next = _xy_np(cache.gt_xy[index + 1])

        final_error = float(np.linalg.norm(final_current - reference_current))
        provisional_error = float(
            np.linalg.norm(provisional_next - reference_next)
        )
        search_error = float(
            np.linalg.norm(search_center_used - reference_current)
        )

        is_initial = index == 0
        provisional_errors.append(provisional_error)
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
                value = _xy_np(trace[name][0])
                row[name + "_x"] = float(value[0])
                row[name + "_y"] = float(value[1])
        rows.append(row)

        # Full-route closed loop: never reset on B/C.
        search_center = provisional_next.copy()

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_segmented_train_full_closedloop_eval_frames.csv"
        % (route_name, arch.lower())
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
            "Protocol": "segmented closed-loop train; full-route closed-loop eval",
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
    parser.add_argument("--segment-frames-start", type=int, default=16)
    parser.add_argument("--segment-frames-end", type=int, default=96)
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

    if args.segment_frames_start < 3 or args.segment_frames_end < 3:
        raise SystemExit("segment frame counts must be >= 3")
    if args.segment_frames_end < args.segment_frames_start:
        raise SystemExit("--segment-frames-end must be >= --segment-frames-start")

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

    if args.mode in ("train", "train-eval"):
        route_a = build_cache("route_A", visual, device)
        model, train_speed_prior = train_architecture(
            args.arch,
            visual,
            route_a,
            device,
            args.epochs,
            args.lr,
            args.segment_frames_start,
            args.segment_frames_end,
        )

    if args.mode in ("eval", "train-eval"):
        if model is None:
            model, train_speed_prior = load_architecture(args.arch, device)

        results = {
            "architecture": args.arch,
            "train_route": "route_A",
            "test_routes": ["route_B", "route_C"],
            "protocol": "segmented closed-loop train; full-route closed-loop eval",
            "segment_frames_start": int(args.segment_frames_start),
            "segment_frames_end": int(args.segment_frames_end),
            "training_segment_initialization": "segment-start reference XY + segment-start reference velocity only",
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
