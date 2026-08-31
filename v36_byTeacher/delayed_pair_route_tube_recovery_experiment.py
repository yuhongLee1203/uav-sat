"""Route-tube constrained delayed KG/GK closed-loop experiment.

This experiment keeps the one-frame delayed formulation but adds a planned-route
safety/recovery prior that never reads the current B/C frame reference during
inference.

Key ideas
---------
1. The known planned waypoint polyline is densified into an ordered route bank.
2. A causal route tracker prevents progress from jumping arbitrarily far forward.
3. A soft corridor loss teaches G to keep its raw next-frame proposal near the
   planned route (default soft margin 12 m).
4. A hard corridor guard prevents the NEXT visual search center from leaving the
   route tube (default hard width 20 m). If raw x' is outside the tube, the search
   center is snapped back to the nearest causally allowed route point.
5. Training injects lateral route-centered perturbations so the model repeatedly
   experiences recoverable off-center visual searches.
6. Route A is still segmented/curriculum trained. Only each segment start receives
   reference XY + reference velocity. Inside the segment, search is closed-loop.
7. Routes B/C are evaluated as one full closed-loop rollout with no segment reset
   and no frame-aligned reference prior.

Every MeanShift remains full centered 6x6 = 36 candidates.
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


class CausalRouteTube:
    """Ordered planned-route corridor with a bounded per-frame progress step."""

    def __init__(self, route_name, visual, spacing_m, hard_width_m, max_progress_m):
        waypoint_xy = rt.load_waypoint_xy(
            route_name, visual.origin_lat, visual.origin_lon
        )
        self.xy = base.core._densify_polyline(waypoint_xy, float(spacing_m))
        self.spacing_m = float(spacing_m)
        self.hard_width_m = float(hard_width_m)
        self.max_forward_steps = max(
            1, int(np.ceil(float(max_progress_m) / max(self.spacing_m, 1e-6)))
        )
        self.last_index = 0

    def reset_to_xy(self, xy):
        p = np.asarray(xy, dtype=np.float64).reshape(2)
        self.last_index = int(np.linalg.norm(self.xy - p[None, :], axis=1).argmin())

    def tangent(self, index=None):
        i = self.last_index if index is None else int(index)
        if len(self.xy) < 2:
            return np.array([1.0, 0.0], dtype=np.float64)
        a = max(0, min(i, len(self.xy) - 2))
        d = self.xy[a + 1] - self.xy[a]
        n = float(np.linalg.norm(d))
        if n < 1e-9 and a > 0:
            d = self.xy[a] - self.xy[a - 1]
            n = float(np.linalg.norm(d))
        if n < 1e-9:
            return np.array([1.0, 0.0], dtype=np.float64)
        return d / n

    def constrain(self, raw_xy):
        p = np.asarray(raw_xy, dtype=np.float64).reshape(2)
        start = int(self.last_index)
        end = min(len(self.xy), start + self.max_forward_steps + 1)
        local = self.xy[start:end]
        if len(local) == 0:
            local = self.xy[-1:]
            start = len(self.xy) - 1
        local_i = int(np.linalg.norm(local - p[None, :], axis=1).argmin())
        index = start + local_i
        route_xy = self.xy[index].copy()
        route_distance = float(np.linalg.norm(p - route_xy))
        triggered = bool(route_distance > self.hard_width_m)

        # Inside the tube we preserve the learned proposal. Outside the tube we
        # recover to the route centerline so the next 6x6 window is opened in a
        # geometrically plausible region rather than at the tube boundary.
        protected = route_xy.copy() if triggered else p.copy()
        progress_delta_m = float((index - self.last_index) * self.spacing_m)
        self.last_index = max(self.last_index, index)
        return {
            "raw_xy": p,
            "protected_xy": protected,
            "route_xy": route_xy,
            "route_distance_m": route_distance,
            "triggered": triggered,
            "route_index": int(self.last_index),
            "progress_delta_m": progress_delta_m,
            "tangent": self.tangent(self.last_index),
        }


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "delayed_pair_route_tube_recovery_%s_center6x6_%s.pt"
        % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "delayed_pair_route_tube_recovery_center6x6"
        / arch.lower()
    )


def build_cache(route_name, visual, device):
    return old.build_cache(route_name, visual, device)


def _route_bank_tensor(route_name, visual, device, spacing_m):
    waypoint_xy = rt.load_waypoint_xy(
        route_name, visual.origin_lat, visual.origin_lon
    )
    dense = base.core._densify_polyline(waypoint_xy, float(spacing_m))
    return torch.as_tensor(dense, dtype=torch.float32, device=device)


def _soft_corridor_loss(pred_xy, route_bank_t, soft_margin_m, hard_width_m):
    distance = torch.linalg.norm(
        pred_xy[:, None, :] - route_bank_t[None, :, :], dim=2
    ).amin(dim=1)
    violation = F.relu(distance - float(soft_margin_m))
    scale = max(float(hard_width_m), 1.0)
    return (violation / scale).square().mean(), distance.mean()


def _augment_center(center_xy, tube, rng, probability, max_offset_m):
    center = np.asarray(center_xy, dtype=np.float64).reshape(2).copy()
    if rng.random() >= float(probability) or float(max_offset_m) <= 0.0:
        return center, 0.0
    tangent = tube.tangent()
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    magnitude = rng.uniform(0.35 * float(max_offset_m), float(max_offset_m))
    return center + sign * magnitude * normal, float(sign * magnitude)


def _repair_temporal_state_after_recovery(arch, state, guard, max_speed_mpf, reset_hidden):
    if not guard["triggered"]:
        return
    if reset_hidden:
        state["hidden"] = None

    # In GK, Kalman has already advanced to t+1, so its position can be safely
    # reconciled with the protected t+1 route center. In KG, Kalman is still at
    # frame t after its current-frame update; changing its position to t+1 would
    # introduce a one-frame phase error, so only its velocity is repaired.
    kf = state["kalman"]
    if arch == "GK":
        kf.x[:2] = guard["protected_xy"]

    speed = float(np.linalg.norm(kf.x[2:]))
    speed = min(speed, float(max_speed_mpf))
    if not np.isfinite(speed):
        speed = 0.0
    kf.x[2:] = guard["tangent"] * speed


def _save_checkpoint(arch, model, epoch, loss, segment_frames, train_speed_prior, args):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": arch,
            "model": state,
            "epoch": int(epoch),
            "train_loss": float(loss),
            "protocol": "route-tube recovery curriculum",
            "segment_frames_reached": int(segment_frames),
            "soft_corridor_m": float(args.soft_corridor_m),
            "hard_corridor_m": float(args.hard_corridor_m),
            "corridor_loss_weight": float(args.corridor_loss_weight),
            "route_spacing_m": float(args.route_spacing_m),
            "max_progress_m": float(args.max_progress_m),
            "recovery_aug_probability": float(args.recovery_aug_probability),
            "recovery_aug_max_m": float(args.recovery_aug_max_m),
            "segment_start_position": "Route-A reference XY once per segment",
            "segment_initial_velocity": "Route-A reference delta once per segment",
            "inside_segment_search_center": "route-tube protected previous provisional XY",
            "eval_frame_aligned_reference_prior": False,
            "eval_segment_resets": False,
            "train_speed_prior_mpf": float(train_speed_prior),
            "meanshift_candidate_count": 36,
            "train_routes": ["route_A"],
        },
        _checkpoint_path(arch),
    )
    return state


def train_architecture(arch, visual, route_a, device, args):
    model = PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(config.TEMPORAL_WEIGHT_DECAY)
    )
    route_bank_t = _route_bank_tensor("route_A", visual, device, args.route_spacing_m)
    train_speed_prior = old._training_speed_prior(route_a)

    current_seg_frames = int(args.segment_frames_start)
    stable_epochs = 0
    best_stage = -1
    best_stage_loss = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        seg_frames = int(current_seg_frames)
        stride = max(1, seg_frames // 2)
        starts = list(range(0, len(route_a) - 1, stride))
        tail_start = max(0, len(route_a) - seg_frames)
        if tail_start < len(route_a) - 1:
            starts.append(tail_start)
        starts = sorted(set(starts))

        losses = []
        raw_next_errors = []
        protected_next_errors = []
        final_ms_errors = []
        closed_search_errors = []
        closed_within20 = []
        raw_route_distances = []
        hard_recoveries = []
        aug_offsets = []

        for seg_start in starts:
            seg_end = min(len(route_a), seg_start + seg_frames)
            if seg_end - seg_start < 2:
                continue

            initial_xy = old._xy_np(route_a.gt_xy[seg_start])
            initial_velocity = old._segment_initial_velocity(route_a, seg_start)
            state = old._make_state(initial_xy, initial_velocity)
            tube = CausalRouteTube(
                "route_A", visual, args.route_spacing_m,
                args.hard_corridor_m, args.max_progress_m,
            )
            tube.reset_to_xy(initial_xy)
            search_center = initial_xy.copy()
            rng = np.random.default_rng(int(config.SEED) + epoch * 100003 + seg_start)

            optimizer.zero_grad(set_to_none=True)
            segment_loss = None
            pair_count = 0

            for index in range(seg_start, seg_end - 1):
                uav_current = route_a.uav_clip[index:index + 1].to(device).float()
                uav_next = route_a.uav_clip[index + 1:index + 2].to(device).float()

                used_center, aug_offset = _augment_center(
                    search_center, tube, rng,
                    args.recovery_aug_probability,
                    min(args.recovery_aug_max_m, 0.75 * args.hard_corridor_m),
                )
                aug_offsets.append(abs(float(aug_offset)))

                final_current_t, raw_next_t, gru_out, _ = base.pair_step(
                    arch, model, visual, uav_current, uav_next,
                    used_center, state, device,
                )
                target_current = route_a.gt_xy[index:index + 1].to(device).float()
                target_next = route_a.gt_xy[index + 1:index + 2].to(device).float()

                position_loss = F.smooth_l1_loss(gru_out.corrected_xy, target_next)
                corridor_loss, learned_route_distance = _soft_corridor_loss(
                    gru_out.corrected_xy, route_bank_t,
                    args.soft_corridor_m, args.hard_corridor_m,
                )
                loss = position_loss + float(args.corridor_loss_weight) * corridor_loss
                segment_loss = loss if segment_loss is None else segment_loss + loss
                pair_count += 1

                raw_next = old._xy_np(raw_next_t[0])
                guard = tube.constrain(raw_next)
                protected_next = guard["protected_xy"]
                _repair_temporal_state_after_recovery(
                    arch, state, guard, args.max_speed_mpf,
                    bool(args.reset_hidden_on_recovery),
                )

                with torch.no_grad():
                    target_current_np = old._xy_np(target_current[0])
                    target_next_np = old._xy_np(target_next[0])
                    raw_next_errors.append(float(np.linalg.norm(raw_next - target_next_np)))
                    protected_next_errors.append(float(np.linalg.norm(protected_next - target_next_np)))
                    final_ms_errors.append(float(np.linalg.norm(old._xy_np(final_current_t[0]) - target_current_np)))
                    raw_route_distances.append(float(guard["route_distance_m"]))
                    hard_recoveries.append(float(guard["triggered"]))
                    if index > seg_start:
                        e = float(np.linalg.norm(np.asarray(used_center) - target_current_np))
                        closed_search_errors.append(e)
                        closed_within20.append(float(e <= 20.0))

                search_center = protected_next.copy()

            if pair_count == 0:
                continue
            normalized = segment_loss / float(pair_count)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.GRAD_CLIP_NORM))
            optimizer.step()
            losses.append(float(normalized.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("inf")
        raw_mle = float(np.mean(raw_next_errors)) if raw_next_errors else float("inf")
        protected_mle = float(np.mean(protected_next_errors)) if protected_next_errors else float("inf")
        final_mle = float(np.mean(final_ms_errors)) if final_ms_errors else float("inf")
        search_mle = float(np.mean(closed_search_errors)) if closed_search_errors else float("inf")
        within20 = 100.0 * float(np.mean(closed_within20)) if closed_within20 else 0.0
        raw_route_mle = float(np.mean(raw_route_distances)) if raw_route_distances else float("inf")
        recovery_pct = 100.0 * float(np.mean(hard_recoveries)) if hard_recoveries else 0.0

        if seg_frames > best_stage or (seg_frames == best_stage and mean_loss < best_stage_loss):
            best_stage = seg_frames
            best_stage_loss = mean_loss
            best_state = _save_checkpoint(
                arch, model, epoch, mean_loss, seg_frames, train_speed_prior, args
            )

        gate_ok = (
            np.isfinite(search_mle)
            and search_mle <= float(args.advance_search_mle)
            and within20 >= float(args.advance_within20)
        )
        stable_epochs = stable_epochs + 1 if gate_ok else 0
        advance = False
        if stable_epochs >= int(args.advance_patience) and current_seg_frames < int(args.segment_frames_end):
            current_seg_frames = min(
                int(args.segment_frames_end), current_seg_frames + int(args.segment_step)
            )
            stable_epochs = 0
            advance = True

        print(
            "[%s-routetube] epoch=%03d/%d seg=%d loss=%.6f raw_next=%.3fm "
            "protected_next=%.3fm search=%.3fm final_ms=%.3fm route_dist=%.3fm "
            "recover=%.2f%% within20=%.2f%% stable=%d/%d best_stage=%d%s"
            % (
                arch, epoch, args.epochs, seg_frames, mean_loss, raw_mle,
                protected_mle, search_mle, final_mle, raw_route_mle,
                recovery_pct, within20, stable_epochs, args.advance_patience,
                best_stage,
                " -> advance_to_%d" % current_seg_frames if advance else "",
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no route-tube checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    return model, float(train_speed_prior), int(best_stage)


def load_architecture(arch, device):
    path = _checkpoint_path(arch)
    payload = torch.load(path, map_location="cpu")
    if payload.get("protocol") != "route-tube recovery curriculum":
        raise RuntimeError("checkpoint protocol mismatch")
    model = PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, float(payload.get("train_speed_prior_mpf", 0.0)), int(payload.get("segment_frames_reached", 0))


@torch.no_grad()
def evaluate_architecture(arch, visual, model, route_name, cache, device, train_speed_prior, args):
    _, planned_start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    initial_velocity = old._planned_initial_velocity(route_name, visual, train_speed_prior)
    state = old._make_state(planned_start_xy, initial_velocity)
    tube = CausalRouteTube(
        route_name, visual, args.route_spacing_m,
        args.hard_corridor_m, args.max_progress_m,
    )
    tube.reset_to_xy(planned_start_xy)
    search_center = np.asarray(planned_start_xy, dtype=np.float64).reshape(2).copy()

    rows = []
    final_errors = []
    raw_next_errors = []
    protected_next_errors = []
    search_errors = []
    raw_route_distances = []
    recovery_flags = []

    for index in range(len(cache) - 1):
        uav_current = cache.uav_clip[index:index + 1].to(device).float()
        uav_next = cache.uav_clip[index + 1:index + 2].to(device).float()

        # Inference is completed before frame-aligned B/C references are read.
        final_current_t, raw_next_t, _, trace = base.pair_step(
            arch, model, visual, uav_current, uav_next,
            search_center, state, device,
        )
        final_current = old._xy_np(final_current_t[0])
        raw_next = old._xy_np(raw_next_t[0])
        guard = tube.constrain(raw_next)
        protected_next = guard["protected_xy"]
        _repair_temporal_state_after_recovery(
            arch, state, guard, args.max_speed_mpf,
            bool(args.reset_hidden_on_recovery),
        )

        # Metrics only after inference/guarding.
        reference_current = old._xy_np(cache.gt_xy[index])
        reference_next = old._xy_np(cache.gt_xy[index + 1])
        final_error = float(np.linalg.norm(final_current - reference_current))
        raw_error = float(np.linalg.norm(raw_next - reference_next))
        protected_error = float(np.linalg.norm(protected_next - reference_next))
        search_error = float(np.linalg.norm(search_center - reference_current))

        is_initial = index == 0
        raw_next_errors.append(raw_error)
        protected_next_errors.append(protected_error)
        raw_route_distances.append(float(guard["route_distance_m"]))
        recovery_flags.append(float(guard["triggered"]))
        if not is_initial:
            final_errors.append(final_error)
            search_errors.append(search_error)

        row = {
            "pair_current_index": int(index),
            "pair_next_index": int(index + 1),
            "current_frame_id": int(cache.frame_ids[index]),
            "next_frame_id": int(cache.frame_ids[index + 1]),
            "is_initial_frame": bool(is_initial),
            "search_center_x": float(search_center[0]),
            "search_center_y": float(search_center[1]),
            "final_ms_x": float(final_current[0]),
            "final_ms_y": float(final_current[1]),
            "raw_next_x": float(raw_next[0]),
            "raw_next_y": float(raw_next[1]),
            "protected_next_x": float(protected_next[0]),
            "protected_next_y": float(protected_next[1]),
            "route_center_x": float(guard["route_xy"][0]),
            "route_center_y": float(guard["route_xy"][1]),
            "route_index": int(guard["route_index"]),
            "route_distance_raw_m": float(guard["route_distance_m"]),
            "route_recovery_triggered": bool(guard["triggered"]),
            "route_progress_delta_m": float(guard["progress_delta_m"]),
            "reference_current_x": float(reference_current[0]),
            "reference_current_y": float(reference_current[1]),
            "reference_next_x": float(reference_next[0]),
            "reference_next_y": float(reference_next[1]),
            "search_center_error_m": search_error,
            "final_error_m": final_error,
            "raw_next_error_m": raw_error,
            "protected_next_error_m": protected_error,
            "ms_candidate_count": int(trace["candidate_count"]),
        }
        rows.append(row)
        search_center = protected_next.copy()

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / ("%s_%s_route_tube_frames.csv" % (route_name, arch.lower()))
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
            "Protocol": "route-tube recovery; full-route closed-loop eval",
            "RawProvisionalNextMLE_m": float(np.mean(raw_next_errors)),
            "ProtectedProvisionalNextMLE_m": float(np.mean(protected_next_errors)),
            "SearchCenterMLE_m": float(np.mean(search_errors)),
            "RawRouteDistanceMLE_m": float(np.mean(raw_route_distances)),
            "HardRecovery_pct": 100.0 * float(np.mean(recovery_flags)),
            "SoftCorridor_m": float(args.soft_corridor_m),
            "HardCorridor_m": float(args.hard_corridor_m),
            "MaxProgressPerFrame_m": float(args.max_progress_m),
            "EvalFrameAlignedReferencePrior": False,
            "EvalSegmentResets": False,
            "MeanShiftCandidateCount": 36,
            "ForwardSearch": False,
            "CSV": str(csv_path),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare-visual", "train", "eval", "train-eval"), default="train-eval")
    parser.add_argument("--arch", choices=ARCH_CHOICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=float(getattr(config, "TEMPORAL_LR", 3e-4)))
    parser.add_argument("--segment-frames-start", type=int, default=4)
    parser.add_argument("--segment-frames-end", type=int, default=64)
    parser.add_argument("--segment-step", type=int, default=2)
    parser.add_argument("--advance-search-mle", type=float, default=12.0)
    parser.add_argument("--advance-within20", type=float, default=85.0)
    parser.add_argument("--advance-patience", type=int, default=2)
    parser.add_argument("--route-spacing-m", type=float, default=2.0)
    parser.add_argument("--soft-corridor-m", type=float, default=12.0)
    parser.add_argument("--hard-corridor-m", type=float, default=20.0)
    parser.add_argument("--corridor-loss-weight", type=float, default=2.0)
    parser.add_argument("--max-progress-m", type=float, default=10.0)
    parser.add_argument("--max-speed-mpf", type=float, default=8.0)
    parser.add_argument("--recovery-aug-probability", type=float, default=0.30)
    parser.add_argument("--recovery-aug-max-m", type=float, default=10.0)
    parser.add_argument("--reset-hidden-on-recovery", type=int, choices=(0, 1), default=1)
    parser.add_argument("--visual-epochs", type=int, default=int(getattr(config, "VISUAL_EPOCHS", 20)))
    parser.add_argument("--jitter-m", type=float, default=float(getattr(config, "LOCAL_PRIOR_JITTER_M", 12.0)))
    args = parser.parse_args()

    if args.hard_corridor_m <= args.soft_corridor_m:
        raise SystemExit("--hard-corridor-m must be greater than --soft-corridor-m")
    if args.segment_frames_start < 3 or args.segment_frames_end < args.segment_frames_start:
        raise SystemExit("invalid segment-frame range")

    device = rt.resolve_device(args.device)
    rt.set_seed(int(config.SEED))

    if args.mode == "prepare-visual":
        train_visual_retrieval_a_only(
            device=device, epochs=args.visual_epochs,
            jitter_m=args.jitter_m, resume=False,
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
            "protocol": "route-tube recovery curriculum",
            "best_segment_frames_reached": int(best_stage),
            "soft_corridor_m": float(args.soft_corridor_m),
            "hard_corridor_m": float(args.hard_corridor_m),
            "max_progress_per_frame_m": float(args.max_progress_m),
            "evaluation_frame_aligned_reference_prior": False,
            "evaluation_segment_resets": False,
            "meanshift_candidate_count": 36,
            "results": {},
        }
        for route_name in ("route_B", "route_C"):
            cache = build_cache(route_name, visual, device)
            summary = evaluate_architecture(
                args.arch, visual, model, route_name, cache,
                device, train_speed_prior, args,
            )
            results["results"][route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        outdir = _output_dir(args.arch)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
