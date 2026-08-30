"""Train/evaluate all six permutations of MeanShift (M), GRU (G), Kalman (K).

The six variants are MKG, MGK, GMK, GKM, KGM, and KMG.

This experiment deliberately removes the old feedback path
    previous final position + GRU motion delta -> next-frame search center.

Roles used here:
  M: current-frame visual mode refinement with MeanShift.
  G: current-frame recurrent position residual refinement.
  K: standard constant-velocity Kalman filtering whose prior is its own
     previous posterior.

Protocol:
  * Route A trains the GRU.
  * Routes B and C are evaluation routes.
  * A predefined route reference point only opens the local candidate window;
    it is not concatenated into the GRU input and is not a Kalman measurement.
  * Initial support is the strict forward half of a 6x6 lattice (3x6 = 18).
  * When M is not first, the initial visual coordinate is a posterior-weighted
    centroid over the same forward 18 candidates, not MeanShift.
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
from six_architecture_model import PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only


ARCH_CHOICES = ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG")


def _xy_tensor(xy, device):
    return torch.as_tensor(xy, dtype=torch.float32, device=device).reshape(1, 2)


def _heading_from_route(route_name, visual):
    _, _, heading = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return float(heading)


def _strict_forward_half(full_centers, heading_rad):
    """Return indices of the forward 18 cells from a regular 6x6 lattice."""
    batch = int(full_centers.shape[0])
    headings = torch.as_tensor(
        heading_rad, dtype=full_centers.dtype, device=full_centers.device
    ).reshape(-1)
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match batch size")

    geometric_center = full_centers.mean(dim=1, keepdim=True)
    relative = full_centers - geometric_center
    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    use_x = cos_h.abs() >= sin_h.abs()

    sign = torch.where(
        use_x,
        torch.where(cos_h >= 0, torch.ones_like(cos_h), -torch.ones_like(cos_h)),
        torch.where(sin_h >= 0, torch.ones_like(sin_h), -torch.ones_like(sin_h)),
    )
    longitudinal = torch.where(
        use_x[:, None], relative[:, :, 0], relative[:, :, 1]
    ) * sign[:, None]
    return torch.topk(
        longitudinal, k=18, dim=1, largest=True, sorted=False
    ).indices


@torch.no_grad()
def forward_visual_batch(visual, uav_clip, center_xy, heading_rad):
    """Build strict forward 3x6 visual evidence and decode raw/MS positions."""
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
    local = _strict_forward_half(full_centers, heading_rad)
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
    prob = torch.softmax(
        logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    weighted_xy = (prob[:, :, None] * centers).sum(dim=1)
    diff = centers - weighted_xy[:, None, :]
    variance_xy = (
        prob[:, :, None] * diff.square()
    ).sum(dim=1).clamp_min(1e-3)

    ms_xy, ms_support, _, _, mode_weights, _ = rt.soft_mean_shift(
        logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    return {
        "indices": indices,
        "centers": centers,
        "z_uav": z_uav,
        "logits": logits,
        "prob": prob,
        "raw_xy": weighted_xy,
        "variance_xy": variance_xy,
        "meanshift_xy": ms_xy,
        "meanshift_support": ms_support,
        "mode_count": (mode_weights > 0).sum(dim=1),
    }


@torch.no_grad()
def centered_meanshift(visual, uav_clip, center_xy):
    """Run a centered full 6x6 visual MeanShift around the incoming XY."""
    batch = visual.candidate_batch(
        uav_clip=uav_clip, center_xy=center_xy, grid_size=6
    )
    prob = torch.softmax(
        batch.raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    diff = batch.centers - batch.softms_xy[:, None, :]
    variance_xy = (
        prob[:, :, None] * diff.square()
    ).sum(dim=1).clamp_min(1e-3)
    return batch.softms_xy, variance_xy, batch.softms_support


class StandardXYKalman:
    """Constant-velocity KF whose prior is its own previous posterior."""

    def __init__(self, initial_xy):
        p = np.asarray(initial_xy, dtype=np.float64).reshape(2)
        self.x = np.array([p[0], p[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_POSITION_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
                float(config.KALMAN_INIT_VELOCITY_VAR),
            ]
        )
        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.Q = np.diag(
            [
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_POSITION),
                float(config.KALMAN_Q_VELOCITY),
                float(config.KALMAN_Q_VELOCITY),
            ]
        )

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


def _checkpoint_path(arch):
    return (
        Path(config.CHECKPOINT_DIR)
        / f"six_arch_{arch.lower()}_{config.BACKBONE_KEY}.pt"
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "six_architecture_ablation"
        / arch.lower()
    )


def _make_state(route_name, visual):
    _, start_xy, _ = rt.planned_route_start(
        route_name, visual.origin_lat, visual.origin_lon
    )
    return {
        "kalman": StandardXYKalman(start_xy),
        "hidden": None,
        "previous_z": None,
    }


def _apply_g(model, xy, variance_xy, z_uav, state):
    output = model.forward_step(
        stage_xy=xy,
        variance_xy=variance_xy,
        z_uav=z_uav,
        previous_z_uav=state["previous_z"],
        hidden=state["hidden"],
    )
    state["hidden"] = output.hidden
    return output.corrected_xy, output


def _apply_k(state, xy, variance_xy, device):
    filtered_xy = state["kalman"].step(
        xy[0].detach().cpu().numpy(),
        variance_xy[0].detach().cpu().numpy(),
    )
    return _xy_tensor(filtered_xy, device)


def forward_frame(
    arch,
    model,
    visual,
    uav_clip,
    reference_prior_xy,
    heading_rad,
    state,
    device,
):
    """Execute exactly one M/G/K permutation for a single frame."""
    initial = forward_visual_batch(
        visual,
        uav_clip,
        _xy_tensor(reference_prior_xy, device),
        torch.tensor([heading_rad], dtype=torch.float32, device=device),
    )
    z_uav = initial["z_uav"]
    variance = initial["variance_xy"]

    if arch[0] == "M":
        stage_xy = initial["meanshift_xy"]
        start_index = 1
    else:
        stage_xy = initial["raw_xy"]
        start_index = 0

    trace = {
        "raw_xy": initial["raw_xy"],
        "forward_ms_xy": initial["meanshift_xy"],
    }
    gru_out = None

    for symbol in arch[start_index:]:
        if symbol == "M":
            stage_xy, variance, support = centered_meanshift(
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
            raise ValueError(f"unknown architecture symbol: {symbol}")

    state["previous_z"] = z_uav.detach()
    return stage_xy, variance, gru_out, trace


def train_architecture(arch, visual, route_a, device, epochs, lr, tbptt):
    """Train only the G block on Route A for the selected ordering."""
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
    heading = _heading_from_route("route_A", visual)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        state = _make_state("route_A", visual)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []

        for index in range(len(route_a)):
            uav_clip = route_a.uav_clip[index : index + 1].to(device).float()
            reference_xy = (
                route_a.gt_xy[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            _, _, gru_out, _ = forward_frame(
                arch,
                model,
                visual,
                uav_clip,
                reference_xy,
                heading,
                state,
                device,
            )
            if gru_out is None:
                raise RuntimeError(f"architecture {arch} did not execute GRU")

            target_xy = route_a.gt_xy[index : index + 1].to(device).float()
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_xy)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            is_last = index == len(route_a) - 1
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
                if state["previous_z"] is not None:
                    state["previous_z"] = state["previous_z"].detach()
                chunk_loss = None
                chunk_count = 0

        mean_loss = (
            float(np.mean(epoch_losses)) if epoch_losses else float("inf")
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
                    "gru_semantics": (
                        "current-frame position refinement; "
                        "no final-position motion feedback"
                    ),
                    "train_routes": ["route_A"],
                },
                _checkpoint_path(arch),
            )

        print(
            f"[{arch}] epoch={epoch:03d}/{epochs} "
            f"train_position_loss={mean_loss:.6f} best={best_loss:.6f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no checkpoint produced")
    model.load_state_dict(best_state)
    return model


def load_architecture(arch, device):
    ckpt = _checkpoint_path(arch)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location="cpu")
    if payload.get("architecture") != arch:
        raise RuntimeError("checkpoint architecture mismatch")

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
    heading = _heading_from_route(route_name, visual)
    rows = []
    errors = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        reference_xy = (
            cache.gt_xy[index].detach().cpu().numpy().astype(np.float64)
        )
        final_xy_t, variance, _, trace = forward_frame(
            arch,
            model,
            visual,
            uav_clip,
            reference_xy,
            heading,
            state,
            device,
        )
        final_xy = final_xy_t[0].detach().cpu().numpy().astype(np.float64)
        error = float(np.linalg.norm(final_xy - reference_xy))
        errors.append(error)

        row = {
            "frame_id": int(cache.frame_ids[index]),
            "image_path": cache.image_paths[index],
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "raw_visual_x": float(trace["raw_xy"][0, 0]),
            "raw_visual_y": float(trace["raw_xy"][0, 1]),
            "forward_ms_x": float(trace["forward_ms_xy"][0, 0]),
            "forward_ms_y": float(trace["forward_ms_xy"][0, 1]),
            "variance_x": float(variance[0, 0]),
            "variance_y": float(variance[0, 1]),
            "final_x": float(final_xy[0]),
            "final_y": float(final_xy[1]),
            "error_final_m": error,
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
    csv_path = outdir / f"{route_name}_{arch.lower()}_frames.csv"

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = rt.metric_summary(errors)
    summary.update(
        {
            "Architecture": arch,
            "Route": route_name,
            "CSV": str(csv_path),
            "ReferenceUsage": (
                "opens initial local support and provides "
                "offline supervision/metrics only"
            ),
            "GRU": (
                "current-frame position refinement; no previous-final input; "
                "no polynomial motion feedback"
            ),
            "Kalman": (
                "standard CV KF; prior from its own previous posterior"
            ),
        }
    )
    return summary


def build_cache(route_name, visual, device):
    index = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(
        route_name, config.ROUTE_ROOTS[index], visual, device
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

    if args.mode == "prepare-visual":
        train_visual_retrieval_a_only(
            device=device,
            epochs=args.visual_epochs,
            jitter_m=args.jitter_m,
            resume=False,
        )
        return

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Visual checkpoint missing: {config.VISUAL_CHECKPOINT}. "
            "Run --mode prepare-visual once first."
        )
    if args.arch is None:
        raise SystemExit("--arch is required for train/eval")

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
        with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
