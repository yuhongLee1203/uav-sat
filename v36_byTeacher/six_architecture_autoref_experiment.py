"""One-frame-delayed autonomous six-architecture experiment.

Teacher-required temporal alignment
-----------------------------------
A target frame t is NOT finalized when frame t arrives.

1. Frame t arrives -> autonomous reference selection -> full centered 6x6 M_t.
   M_t is stored as a pending visual observation.
2. Frame t+1 arrives -> autonomous reference selection from the previous
   visual localization (no frame-aligned label) -> full centered 6x6 M_{t+1}.
3. Only after M_{t+1} exists is frame t finalized.  The GRU temporal pair is
   [z_t, z_{t+1}], while the coordinate being refined is the target-frame
   coordinate from M_t.
4. The final block order for target t follows the requested architecture.
   Examples:
       MKG: M_t -> K_t -> G([z_t,z_{t+1}]) -> Final(t)
       MGK: M_t -> G([z_t,z_{t+1}]) -> K_t -> Final(t)
5. Frame t+1 then becomes the next pending target.
6. The final dataset frame is not scored because no t+1 look-ahead exists.

Autonomous reference rule
-------------------------
No current/future frame reference coordinate is used to open the SAT window.
The first query is the planned-route start.  Thereafter the previous pending
base MeanShift position is used to select the nearest non-backtracking point
from the predefined ordered waypoint-polyline reference bank.  This avoids a
circular dependency on Final(t), which by definition is unavailable until
M_{t+1} has already been computed.

Every MeanShift uses the full centered 6x6 = 36 satellite patches.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker as rt
from six_architecture_model import PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

ARCH_CHOICES = ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG")
TEMPORAL_ALIGNMENT = "target_t_finalized_after_base_M_t_plus_1"
LOOKAHEAD_FRAMES = 1


def _xy_tensor(xy, device):
    return torch.as_tensor(xy, dtype=torch.float32, device=device).reshape(1, 2)


def _densify_polyline(points, spacing_m):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("reference polyline requires at least two points")
    spacing = max(float(spacing_m), 1e-3)
    rows = [points[0].copy()]
    for a, b in zip(points[:-1], points[1:]):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            continue
        steps = max(1, int(math.ceil(length / spacing)))
        for j in range(1, steps + 1):
            alpha = float(j) / float(steps)
            p = a + alpha * delta
            if np.linalg.norm(p - rows[-1]) > 1e-6:
                rows.append(p)
    bank = np.asarray(rows, dtype=np.float64)
    if len(bank) < 2:
        raise RuntimeError("failed to build reference bank")
    return bank


class RouteReferenceBank:
    """Ordered non-frame-aligned bank built only from planned route waypoints."""

    def __init__(self, route_name, visual, spacing_m):
        waypoint_xy = rt.load_waypoint_xy(
            route_name, visual.origin_lat, visual.origin_lon
        )
        self.xy = _densify_polyline(waypoint_xy, spacing_m)
        self.last_index = 0

    def reset(self):
        self.last_index = 0

    def select(self, query_xy):
        query = np.asarray(query_xy, dtype=np.float64).reshape(2)
        tail = self.xy[self.last_index :]
        local_index = int(np.linalg.norm(tail - query[None, :], axis=1).argmin())
        index = self.last_index + local_index
        self.last_index = max(self.last_index, index)
        return self.xy[self.last_index].copy(), int(self.last_index)


class StandardXYKalman:
    """Standard CV Kalman used only when the architecture contains K."""

    def __init__(self, initial_xy):
        p = np.asarray(initial_xy, dtype=np.float64).reshape(2)
        self.x = np.array([p[0], p[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([
            float(config.KALMAN_INIT_POSITION_VAR),
            float(config.KALMAN_INIT_POSITION_VAR),
            float(config.KALMAN_INIT_VELOCITY_VAR),
            float(config.KALMAN_INIT_VELOCITY_VAR),
        ])
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        self.Q = np.diag([
            float(config.KALMAN_Q_POSITION),
            float(config.KALMAN_Q_POSITION),
            float(config.KALMAN_Q_VELOCITY),
            float(config.KALMAN_Q_VELOCITY),
        ])

    def step(self, measurement_xy, variance_xy):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        var = np.asarray(variance_xy, dtype=np.float64).reshape(2)
        base_r = float(config.KALMAN_R_POSITION)
        R = np.diag(np.maximum(var, base_r))
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        gain = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + gain @ innovation
        eye = np.eye(4, dtype=np.float64)
        ikh = eye - gain @ self.H
        self.P = ikh @ self.P @ ikh.T + gain @ R @ gain.T
        return self.x[:2].copy()


@torch.no_grad()
def centered_visual_meanshift(visual, uav_clip, center_xy):
    """Full centered 6x6 search; all 36 candidates participate in MeanShift."""
    batch = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=_xy_tensor(center_xy, visual.device),
        grid_size=6,
    )
    if int(batch.centers.shape[1]) != 36:
        raise RuntimeError(
            "MeanShift must use the full centered 6x6 candidate set (36 patches)"
        )
    prob = torch.softmax(
        batch.raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    diff = batch.centers - batch.softms_xy[:, None, :]
    variance = (prob[:, :, None] * diff.square()).sum(dim=1).clamp_min(1e-3)
    return {
        "xy": batch.softms_xy,
        "variance": variance,
        "support": batch.softms_support,
        "z_uav": batch.z_uav,
        "candidate_count": int(batch.centers.shape[1]),
    }


def _make_state(route_name, visual, spacing_m):
    _, start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": StandardXYKalman(start_xy),
        "reference_bank": RouteReferenceBank(route_name, visual, spacing_m),
        "hidden": None,
        "route_start_xy": np.asarray(start_xy, dtype=np.float64).reshape(2),
    }


def _apply_g_pair(model, xy, variance, target_z, lookahead_z, state):
    """Refine target t using the visual pair [z_t, z_{t+1}]."""
    out = model.forward_step(
        stage_xy=xy,
        variance_xy=variance,
        z_uav=lookahead_z,
        previous_z_uav=target_z,
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


def _prepare_autonomous_visual(visual, uav_clip, state, query_xy):
    """Select a route-bank reference without any frame-aligned label, then run M."""
    selected_ref_xy, selected_ref_index = state["reference_bank"].select(query_xy)
    base = centered_visual_meanshift(visual, uav_clip, selected_ref_xy)
    return {
        "uav_clip": uav_clip,
        "base": base,
        "selection_query_xy": np.asarray(query_xy, dtype=np.float64).reshape(2).copy(),
        "selected_ref_xy": np.asarray(selected_ref_xy, dtype=np.float64).reshape(2).copy(),
        "selected_ref_index": int(selected_ref_index),
    }


def _next_autonomous_query(pending):
    """Causal query for frame t+1 from model-derived M_t only."""
    return (
        pending["base"]["xy"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def finalize_pending(arch, model, visual, pending, lookahead, state, device):
    """Finalize target frame t only after base M for frame t+1 exists."""
    stage_xy = pending["base"]["xy"]
    variance = pending["base"]["variance"]
    target_z = pending["base"]["z_uav"]
    lookahead_z = lookahead["base"]["z_uav"]

    trace = {
        "selection_query_xy": pending["selection_query_xy"],
        "selected_ref_xy": pending["selected_ref_xy"],
        "selected_ref_index": pending["selected_ref_index"],
        "base_ms_xy": stage_xy,
        "base_ms_support": pending["base"]["support"],
        "base_ms_candidate_count": pending["base"]["candidate_count"],
        "lookahead_base_ms_xy": lookahead["base"]["xy"],
        "lookahead_selected_ref_xy": lookahead["selected_ref_xy"],
        "lookahead_selected_ref_index": lookahead["selected_ref_index"],
    }
    gru_out = None

    symbols = arch[1:] if arch[0] == "M" else arch
    for symbol in symbols:
        if symbol == "M":
            correction = centered_visual_meanshift(
                visual, pending["uav_clip"], stage_xy
            )
            stage_xy = correction["xy"]
            variance = correction["variance"]
            trace["center_ms_xy"] = stage_xy
            trace["center_ms_support"] = correction["support"]
            trace["center_ms_candidate_count"] = correction["candidate_count"]
        elif symbol == "G":
            stage_xy, gru_out = _apply_g_pair(
                model,
                stage_xy,
                variance,
                target_z,
                lookahead_z,
                state,
            )
            trace["gru_xy"] = stage_xy
        elif symbol == "K":
            stage_xy = _apply_k(state, stage_xy, variance, device)
            trace["kalman_xy"] = stage_xy
        else:
            raise ValueError("unknown architecture symbol: %s" % symbol)

    return stage_xy, variance, gru_out, trace


def _checkpoint_path(arch):
    return (
        Path(config.CHECKPOINT_DIR)
        / (
            "six_autoref_delay1_center6x6_%s_%s.pt"
            % (arch.lower(), config.BACKBONE_KEY)
        )
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "six_architecture_autonomous_reference_delay1_center6x6"
        / arch.lower()
    )


def train_architecture(arch, visual, route_a, device, epochs, lr, tbptt, spacing_m):
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
        state = _make_state("route_A", visual, spacing_m)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []

        first_uav = route_a.uav_clip[0:1].to(device).float()
        pending = _prepare_autonomous_visual(
            visual, first_uav, state, state["route_start_xy"]
        )

        for lookahead_index in range(1, len(route_a)):
            lookahead_uav = (
                route_a.uav_clip[lookahead_index:lookahead_index + 1]
                .to(device)
                .float()
            )
            lookahead = _prepare_autonomous_visual(
                visual,
                lookahead_uav,
                state,
                _next_autonomous_query(pending),
            )

            target_index = lookahead_index - 1
            _, _, gru_out, _ = finalize_pending(
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

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
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
                        "autonomous ordered route bank; next query from previous "
                        "base visual MeanShift"
                    ),
                    "reference_bank_spacing_m": float(spacing_m),
                    "frame_aligned_reference_prior": False,
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
            "[%s-autoref-delay1-center6x6] epoch=%03d/%d "
            "train_position_loss=%.6f best=%.6f"
            % (arch, epoch, epochs, mean_loss, best_loss),
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
    if payload.get("temporal_alignment") != TEMPORAL_ALIGNMENT:
        raise RuntimeError("checkpoint is not the one-frame delayed experiment")
    if payload.get("frame_aligned_reference_prior") is not False:
        raise RuntimeError("checkpoint is not autonomous")

    model = PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_architecture(arch, visual, model, route_name, cache, device, spacing_m):
    if len(cache) < 2:
        raise RuntimeError("one-frame delayed evaluation requires at least 2 frames")

    model.eval()
    state = _make_state(route_name, visual, spacing_m)
    rows = []
    final_errors = []
    search_errors = []
    base_ms_errors = []

    first_uav = cache.uav_clip[0:1].to(device).float()
    pending = _prepare_autonomous_visual(
        visual, first_uav, state, state["route_start_xy"]
    )

    for lookahead_index in range(1, len(cache)):
        lookahead_uav = (
            cache.uav_clip[lookahead_index:lookahead_index + 1]
            .to(device)
            .float()
        )
        lookahead = _prepare_autonomous_visual(
            visual,
            lookahead_uav,
            state,
            _next_autonomous_query(pending),
        )

        target_index = lookahead_index - 1
        final_xy_t, variance, _, trace = finalize_pending(
            arch, model, visual, pending, lookahead, state, device
        )

        reference_xy = (
            cache.gt_xy[target_index].cpu().numpy().astype(np.float64)
        )
        final_xy = (
            final_xy_t[0].detach().cpu().numpy().astype(np.float64)
        )
        base_ms_xy = (
            trace["base_ms_xy"][0].detach().cpu().numpy().astype(np.float64)
        )
        selected_ref_xy = np.asarray(
            trace["selected_ref_xy"], dtype=np.float64
        )
        selection_query = np.asarray(
            trace["selection_query_xy"], dtype=np.float64
        )

        final_error = float(np.linalg.norm(final_xy - reference_xy))
        search_error = float(np.linalg.norm(selected_ref_xy - reference_xy))
        base_error = float(np.linalg.norm(base_ms_xy - reference_xy))
        final_errors.append(final_error)
        search_errors.append(search_error)
        base_ms_errors.append(base_error)

        row = {
            "target_frame_id": int(cache.frame_ids[target_index]),
            "lookahead_frame_id": int(cache.frame_ids[lookahead_index]),
            "target_image_path": cache.image_paths[target_index],
            "lookahead_image_path": cache.image_paths[lookahead_index],
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "selection_query_x": float(selection_query[0]),
            "selection_query_y": float(selection_query[1]),
            "selected_ref_index": int(trace["selected_ref_index"]),
            "selected_ref_x": float(selected_ref_xy[0]),
            "selected_ref_y": float(selected_ref_xy[1]),
            "selected_ref_error_m": search_error,
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
        rows.append(row)

        if state["hidden"] is not None:
            state["hidden"] = state["hidden"].detach()
        pending = lookahead

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_autoref_delay1_center6x6_frames.csv"
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
            "SearchReferenceMLE_m": float(np.mean(search_errors)),
            "BaseVisualMS_MLE_m": float(np.mean(base_ms_errors)),
            "EvaluatedTargetFrames": len(rows),
            "DroppedLastFrameWithoutLookahead": True,
            "TemporalAlignment": TEMPORAL_ALIGNMENT,
            "LookaheadFrames": LOOKAHEAD_FRAMES,
            "GRUVisualPair": "[z_t, z_t_plus_1]",
            "CSV": str(csv_path),
            "ReferenceUsage": (
                "no frame-aligned lookup; first query is route start, then "
                "previous base visual MS selects the ordered route-bank reference"
            ),
            "FrameAlignedReferencePrior": False,
            "ReferenceBankSpacing_m": float(spacing_m),
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
    parser.add_argument("--reference-spacing-m", type=float, default=5.0)
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
            args.reference_spacing_m,
        )

    if args.mode in ("eval", "train-eval"):
        if model is None:
            model = load_architecture(args.arch, device)

        results = {
            "architecture": args.arch,
            "train_route": "route_A",
            "test_routes": ["route_B", "route_C"],
            "reference_mode": (
                "autonomous ordered route bank; previous base MS selects next "
                "reference"
            ),
            "frame_aligned_reference_prior": False,
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
                args.arch,
                visual,
                model,
                route_name,
                cache,
                device,
                args.reference_spacing_m,
            )
            results["results"][route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)

        outdir = _output_dir(args.arch)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
