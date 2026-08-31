"""Autonomous reference-bank six-architecture experiment.

This version removes frame-aligned reference leakage.

Runtime rules
-------------
1. The current frame index is NEVER used to choose the SAT search center.
2. A dense ordered reference bank is built only from the predefined route
   waypoint polyline. No per-frame route label is used by the selector.
3. The Kalman prior predicts the current search query from its previous
   posterior. The selector chooses the nearest non-backtracking reference point
   in the static route bank to that predicted query.
4. Every architecture first performs a visual MeanShift (base MS1):
      selected reference -> centered 6x6 lattice -> nearest forward 3x6 -> MS
   This always produces visual_position and visual_variance.
5. If architecture symbol M is first, that symbol is exactly the base MS1.
   If M appears later, M means an additional correction MeanShift:
      incoming stage XY -> full centered 6x6 -> MS2 corrected XY + variance.
6. GRU is current-frame residual refinement only. Kalman is standard CV.
7. Route-A labels are used only after inference for GRU supervision. B/C labels
   are used only for metrics.

Architectures after the common visual MS definition:
  MKG: base MS -> K -> G
  MGK: base MS -> G -> K
  GMK: base MS -> G -> centered MS -> K
  GKM: base MS -> G -> K -> centered MS (final MS correction)
  KGM: base MS -> K -> G -> centered MS (final MS correction)
  KMG: base MS -> K -> centered MS -> G
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
    """Ordered, non-frame-aligned reference bank from the waypoint polyline."""

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
        ref_xy = self.xy[self.last_index].copy()

        if self.last_index + 1 < len(self.xy):
            tangent = self.xy[self.last_index + 1] - self.xy[self.last_index]
        else:
            tangent = self.xy[self.last_index] - self.xy[self.last_index - 1]
        if float(np.linalg.norm(tangent)) < 1e-9:
            heading = 0.0
        else:
            heading = float(math.atan2(tangent[1], tangent[0]))
        return ref_xy, int(self.last_index), heading


class StandardXYKalman:
    """Standard CV Kalman: prior comes only from its own previous posterior."""

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

    def predicted_position(self):
        return (self.F @ self.x)[:2].copy()

    def step(self, measurement_xy, variance_xy):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        var = np.asarray(variance_xy, dtype=np.float64).reshape(2)
        base_r = float(config.KALMAN_R_POSITION)
        R = np.diag(np.maximum(var, base_r))
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ innovation
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return self.x[:2].copy()


def _nearest_forward_3x6(full_centers, center_xy, heading_rad):
    """Center-adjacent forward 3x6: 0,+1,+2 or 0,-1,-2 along heading axis."""
    relative = full_centers - center_xy[:, None, :]
    headings = torch.as_tensor(
        heading_rad, dtype=relative.dtype, device=relative.device
    ).reshape(-1)
    if headings.numel() == 1 and relative.shape[0] > 1:
        headings = headings.expand(relative.shape[0])
    if headings.numel() != relative.shape[0]:
        raise ValueError("heading count must match batch size")

    c = torch.cos(headings)
    s = torch.sin(headings)
    use_x = c.abs() >= s.abs()
    sign_x = torch.where(c >= 0, torch.ones_like(c), -torch.ones_like(c))
    sign_y = torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))
    primary = torch.where(
        use_x[:, None],
        relative[:, :, 0] * sign_x[:, None],
        relative[:, :, 1] * sign_y[:, None],
    )
    secondary = torch.where(
        use_x[:, None], relative[:, :, 1], relative[:, :, 0]
    )
    forward_mask = primary >= -1e-4
    keep = int(getattr(config, "MS1_CANDIDATE_COUNT", 18))
    if not bool(torch.all(forward_mask.sum(dim=1) >= keep)):
        raise RuntimeError("6x6 grid does not contain enough forward candidates")

    huge = torch.full_like(primary, 1e9)
    cost = torch.where(forward_mask, primary.abs(), huge)
    local = torch.topk(cost, k=keep, dim=1, largest=False, sorted=False).indices
    p = torch.gather(primary, 1, local)
    q = torch.gather(secondary, 1, local)
    order = torch.argsort(p * 1000.0 + q, dim=1)
    return torch.gather(local, 1, order)


@torch.no_grad()
def base_visual_meanshift(visual, uav_clip, center_xy, heading_rad):
    """Base visual localization: 6x6 geometry -> nearest forward 3x6 -> MS."""
    center_xy = _xy_tensor(center_xy, visual.device)
    full_indices = rt.regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        center_xy,
        6,
        config.SAT_STRIDE,
        visual.device,
    )
    full_centers = visual.gallery["xy"][full_indices]
    local = _nearest_forward_3x6(
        full_centers,
        center_xy,
        torch.tensor([heading_rad], dtype=torch.float32, device=visual.device),
    )
    indices = torch.gather(full_indices, 1, local)
    centers = visual.gallery["xy"][indices]
    sat_clip = visual.gallery["clip_feat"][indices]

    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        sat_clip.reshape(-1, sat_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    prob = torch.softmax(logits / float(config.MEANSHIFT_SCORE_TAU), dim=1)
    ms_xy, support, _, _, mode_weights, _ = rt.soft_mean_shift(
        logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    diff = centers - ms_xy[:, None, :]
    variance = (prob[:, :, None] * diff.square()).sum(dim=1).clamp_min(1e-3)
    return {
        "xy": ms_xy,
        "variance": variance,
        "support": support,
        "z_uav": z_uav,
        "centers": centers,
        "logits": logits,
        "mode_count": (mode_weights > 0).sum(dim=1),
    }


@torch.no_grad()
def centered_correction_meanshift(visual, uav_clip, center_xy):
    """Correction MS: full centered 6x6 around incoming stage XY."""
    batch = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=center_xy,
        grid_size=6,
    )
    prob = torch.softmax(
        batch.raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    diff = batch.centers - batch.softms_xy[:, None, :]
    variance = (prob[:, :, None] * diff.square()).sum(dim=1).clamp_min(1e-3)
    return batch.softms_xy, variance, batch.softms_support


def _make_state(route_name, visual, spacing_m):
    _, start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": StandardXYKalman(start_xy),
        "reference_bank": RouteReferenceBank(route_name, visual, spacing_m),
        "hidden": None,
        "previous_z": None,
    }


def _apply_g(model, xy, variance, z_uav, state):
    out = model.forward_step(
        stage_xy=xy,
        variance_xy=variance,
        z_uav=z_uav,
        previous_z_uav=state["previous_z"],
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


def forward_frame(arch, model, visual, uav_clip, state, device):
    """No frame label is accepted by this function; search is autonomous."""
    predicted_query = state["kalman"].predicted_position()
    selected_ref_xy, selected_ref_index, heading = state["reference_bank"].select(
        predicted_query
    )
    base = base_visual_meanshift(
        visual, uav_clip, selected_ref_xy, heading
    )
    stage_xy = base["xy"]
    variance = base["variance"]
    z_uav = base["z_uav"]
    trace = {
        "predicted_query_xy": predicted_query,
        "selected_ref_xy": selected_ref_xy,
        "selected_ref_index": selected_ref_index,
        "base_ms_xy": stage_xy,
        "base_ms_support": base["support"],
    }
    gru_out = None

    symbols = arch[1:] if arch[0] == "M" else arch
    for symbol in symbols:
        if symbol == "M":
            stage_xy, variance, support = centered_correction_meanshift(
                visual, uav_clip, stage_xy
            )
            trace["center_ms_xy"] = stage_xy
            trace["center_ms_support"] = support
        elif symbol == "G":
            stage_xy, gru_out = _apply_g(
                model, stage_xy, variance, z_uav, state
            )
            trace["gru_xy"] = stage_xy
        elif symbol == "K":
            stage_xy = _apply_k(state, stage_xy, variance, device)
            trace["kalman_xy"] = stage_xy
        else:
            raise ValueError("unknown architecture symbol: %s" % symbol)

    state["previous_z"] = z_uav.detach()
    return stage_xy, variance, gru_out, trace


def _checkpoint_path(arch):
    return (
        Path(config.CHECKPOINT_DIR)
        / ("six_autoref_%s_%s.pt" % (arch.lower(), config.BACKBONE_KEY))
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "six_architecture_autonomous_reference"
        / arch.lower()
    )


def train_architecture(arch, visual, route_a, device, epochs, lr, tbptt, spacing_m):
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

        for index in range(len(route_a)):
            uav_clip = route_a.uav_clip[index:index+1].to(device).float()
            _, _, gru_out, _ = forward_frame(
                arch, model, visual, uav_clip, state, device
            )
            if gru_out is None:
                raise RuntimeError("architecture %s did not execute GRU" % arch)
            target_xy = route_a.gt_xy[index:index+1].to(device).float()
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_xy)
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
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            _checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "architecture": arch,
                "model": best_state,
                "epoch": epoch,
                "train_loss": best_loss,
                "reference_mode": "autonomous waypoint-polyline reference bank",
                "reference_bank_spacing_m": float(spacing_m),
                "frame_aligned_reference_prior": False,
                "visual_position": "always MeanShift",
                "train_routes": ["route_A"],
            }, _checkpoint_path(arch))
        print(
            "[%s-autoref] epoch=%03d/%d train_position_loss=%.6f best=%.6f"
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
    model.eval()
    state = _make_state(route_name, visual, spacing_m)
    rows = []
    final_errors = []
    search_errors = []
    base_ms_errors = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index:index+1].to(device).float()
        final_xy_t, variance, _, trace = forward_frame(
            arch, model, visual, uav_clip, state, device
        )
        reference_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        final_xy = final_xy_t[0].detach().cpu().numpy().astype(np.float64)
        base_ms_xy = trace["base_ms_xy"][0].detach().cpu().numpy().astype(np.float64)
        selected_ref_xy = np.asarray(trace["selected_ref_xy"], dtype=np.float64)
        predicted_query = np.asarray(trace["predicted_query_xy"], dtype=np.float64)

        final_error = float(np.linalg.norm(final_xy - reference_xy))
        search_error = float(np.linalg.norm(selected_ref_xy - reference_xy))
        base_error = float(np.linalg.norm(base_ms_xy - reference_xy))
        final_errors.append(final_error)
        search_errors.append(search_error)
        base_ms_errors.append(base_error)

        row = {
            "frame_id": int(cache.frame_ids[index]),
            "image_path": cache.image_paths[index],
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "predicted_query_x": float(predicted_query[0]),
            "predicted_query_y": float(predicted_query[1]),
            "selected_ref_index": int(trace["selected_ref_index"]),
            "selected_ref_x": float(selected_ref_xy[0]),
            "selected_ref_y": float(selected_ref_xy[1]),
            "selected_ref_error_m": search_error,
            "base_ms_x": float(base_ms_xy[0]),
            "base_ms_y": float(base_ms_xy[1]),
            "base_ms_error_m": base_error,
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

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / ("%s_%s_autoref_frames.csv" % (route_name, arch.lower()))
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
    summary.update({
        "Architecture": arch,
        "Route": route_name,
        "SearchReferenceMLE_m": float(np.mean(search_errors)),
        "BaseVisualMS_MLE_m": float(np.mean(base_ms_errors)),
        "CSV": str(csv_path),
        "ReferenceUsage": (
            "waypoint-polyline static bank selected from causal Kalman prediction; "
            "no current-frame reference lookup"
        ),
        "FrameAlignedReferencePrior": False,
        "ReferenceBankSpacing_m": float(spacing_m),
        "VisualPosition": "always MeanShift; variance centered on MS observation",
        "BaseMS": "selected reference center -> 6x6 geometry -> nearest forward 3x6 -> MeanShift",
        "CorrectionMS": "when M is not first: full centered 6x6 around incoming stage -> MeanShift",
    })
    return summary


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare-visual", "train", "eval", "train-eval"), default="train-eval")
    parser.add_argument("--arch", choices=ARCH_CHOICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=int(getattr(config, "TEMPORAL_EPOCHS", 60)))
    parser.add_argument("--lr", type=float, default=float(getattr(config, "TEMPORAL_LR", 3e-4)))
    parser.add_argument("--tbptt", type=int, default=int(getattr(config, "TBPTT_STEPS", 32)))
    parser.add_argument("--visual-epochs", type=int, default=int(getattr(config, "VISUAL_EPOCHS", 20)))
    parser.add_argument("--jitter-m", type=float, default=float(getattr(config, "LOCAL_PRIOR_JITTER_M", 12.0)))
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
            args.arch, visual, route_a, device,
            args.epochs, args.lr, args.tbptt, args.reference_spacing_m,
        )
    if args.mode in ("eval", "train-eval"):
        if model is None:
            model = load_architecture(args.arch, device)
        results = {
            "architecture": args.arch,
            "train_route": "route_A",
            "test_routes": ["route_B", "route_C"],
            "reference_mode": "autonomous waypoint-polyline reference bank",
            "frame_aligned_reference_prior": False,
            "results": {},
        }
        for route_name in ("route_B", "route_C"):
            cache = build_cache(route_name, visual, device)
            summary = evaluate_architecture(
                args.arch, visual, model, route_name, cache, device,
                args.reference_spacing_m,
            )
            results["results"][route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        outdir = _output_dir(args.arch)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
