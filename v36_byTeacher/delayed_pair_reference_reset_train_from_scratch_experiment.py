"""From-scratch reference-reset one-step delayed KG/GK comparison.

This experiment is intentionally different from the previous controlled eval
that centered the FINAL MeanShift directly at the frame-aligned reference.
That previous protocol necessarily made KG/GK final MS outputs identical.

Here each independent sample is:
  reference(t) -> MS on I(t) -> KG/GK predicts provisional x'_(t+1)
  -> MS on I(t+1) centered at x'_(t+1) -> FINAL x_(t+1).

The next sample resets at reference(t+1). Therefore there is no long-horizon
closed-loop drift, while the final next-frame MS is still causally dependent on
KG/GK. This cleanly measures one-step delayed correction quality.

Route A trains KG/GK from random initialization. Routes B/C evaluate the same
reference-reset one-step protocol. Every MeanShift uses full centered 6x6=36.
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
from visual_localizer import FrozenVisualLocalizer

ARCH_CHOICES = ("KG", "GK")


def _xy_np(t):
    return t.detach().cpu().numpy().astype(np.float64).reshape(2)


def build_cache(route_name, visual, device):
    return base.build_cache(route_name, visual, device)


def _pair_state(reference_current, reference_previous):
    current = np.asarray(reference_current, dtype=np.float64).reshape(2)
    previous = np.asarray(reference_previous, dtype=np.float64).reshape(2)
    kf = base.core.StandardXYKalman(current)
    kf.x[2:] = current - previous
    return {"kalman": kf, "hidden": None, "start_xy": current.copy()}


def _checkpoint_path(arch):
    return Path(config.CHECKPOINT_DIR) / (
        "delayed_pair_reference_reset_scratch_%s_center6x6_%s.pt"
        % (arch.lower(), config.BACKBONE_KEY)
    )


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "delayed_pair_reference_reset_scratch_center6x6"
        / arch.lower()
    )


def _new_model(device):
    return PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)


def train_architecture(arch, visual, route_a, device, epochs, lr):
    if len(route_a) < 3:
        raise RuntimeError("Route A requires at least three frames")

    model = _new_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        provisional_errors = []
        final_errors = []
        correction_gains = []

        # Every pair is independent: current frame is reference-reset, while
        # final next-frame MS MUST use KG/GK provisional x' as its center.
        for index in range(1, len(route_a) - 1):
            current_ref = _xy_np(route_a.gt_xy[index])
            previous_ref = _xy_np(route_a.gt_xy[index - 1])
            state = _pair_state(current_ref, previous_ref)

            uav_current = route_a.uav_clip[index:index + 1].to(device).float()
            uav_next = route_a.uav_clip[index + 1:index + 2].to(device).float()
            target_next = route_a.gt_xy[index + 1:index + 2].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            _, provisional_next_t, gru_out, _ = base.pair_step(
                arch,
                model,
                visual,
                uav_current,
                uav_next,
                current_ref,
                state,
                device,
            )

            # Differentiable supervision is on G's next-frame proposal.
            # The following visual MS is non-differentiable and is used as the
            # architecture-dependent final localization metric.
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            with torch.no_grad():
                provisional_next = _xy_np(provisional_next_t[0])
                target_next_np = _xy_np(target_next[0])
                final_ms = base.core.centered_visual_meanshift(
                    visual, uav_next, provisional_next
                )
                final_next = _xy_np(final_ms["xy"][0])
                provisional_error = float(
                    np.linalg.norm(provisional_next - target_next_np)
                )
                final_error = float(np.linalg.norm(final_next - target_next_np))
                provisional_errors.append(provisional_error)
                final_errors.append(final_error)
                correction_gains.append(provisional_error - final_error)

        mean_loss = float(np.mean(losses)) if losses else float("inf")
        provisional_mle = (
            float(np.mean(provisional_errors)) if provisional_errors else float("inf")
        )
        final_mle = float(np.mean(final_errors)) if final_errors else float("inf")
        gain = float(np.mean(correction_gains)) if correction_gains else 0.0

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
                    "epoch": int(epoch),
                    "train_loss": float(best_loss),
                    "protocol": "reference-reset one-step delayed scratch",
                    "current_ms_center": "current-frame reference XY",
                    "next_ms_center": "KG/GK provisional next-frame XY",
                    "final_output": "next-frame MeanShift centered at provisional prediction",
                    "meanshift_candidate_count": 36,
                    "train_routes": ["route_A"],
                },
                _checkpoint_path(arch),
            )

        print(
            "[%s-delay-reset-scratch] epoch=%03d/%d loss=%.6f "
            "provisional_mle=%.3fm final_ms_mle=%.3fm ms_gain=%.3fm best=%.6f"
            % (
                arch,
                epoch,
                epochs,
                mean_loss,
                provisional_mle,
                final_mle,
                gain,
                best_loss,
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no scratch delayed checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    return model


def load_architecture(arch, device):
    path = _checkpoint_path(arch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("protocol") != "reference-reset one-step delayed scratch":
        raise RuntimeError("checkpoint protocol mismatch")
    model = _new_model(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_architecture(arch, visual, model, route_name, cache, device):
    model.eval()
    rows = []
    provisional_errors = []
    final_errors = []
    correction_gains = []
    correction_success = []

    for index in range(1, len(cache) - 1):
        current_ref = _xy_np(cache.gt_xy[index])
        previous_ref = _xy_np(cache.gt_xy[index - 1])
        next_ref = _xy_np(cache.gt_xy[index + 1])
        state = _pair_state(current_ref, previous_ref)

        uav_current = cache.uav_clip[index:index + 1].to(device).float()
        uav_next = cache.uav_clip[index + 1:index + 2].to(device).float()

        current_ms_t, provisional_next_t, _, trace = base.pair_step(
            arch,
            model,
            visual,
            uav_current,
            uav_next,
            current_ref,
            state,
            device,
        )
        provisional_next = _xy_np(provisional_next_t[0])

        # Critical comparison step: FINAL next-frame visual localization is
        # centered at the architecture's provisional prediction, NOT reference.
        final_ms = base.core.centered_visual_meanshift(
            visual, uav_next, provisional_next
        )
        final_next = _xy_np(final_ms["xy"][0])

        provisional_error = float(np.linalg.norm(provisional_next - next_ref))
        final_error = float(np.linalg.norm(final_next - next_ref))
        gain = provisional_error - final_error

        provisional_errors.append(provisional_error)
        final_errors.append(final_error)
        correction_gains.append(gain)
        correction_success.append(float(final_error < provisional_error))

        current_ms = _xy_np(current_ms_t[0])
        rows.append(
            {
                "current_index": int(index),
                "next_index": int(index + 1),
                "current_frame_id": int(cache.frame_ids[index]),
                "next_frame_id": int(cache.frame_ids[index + 1]),
                "current_reference_x": float(current_ref[0]),
                "current_reference_y": float(current_ref[1]),
                "current_ms_x": float(current_ms[0]),
                "current_ms_y": float(current_ms[1]),
                "provisional_next_x": float(provisional_next[0]),
                "provisional_next_y": float(provisional_next[1]),
                "next_reference_x": float(next_ref[0]),
                "next_reference_y": float(next_ref[1]),
                "provisional_next_error_m": provisional_error,
                "final_next_ms_x": float(final_next[0]),
                "final_next_ms_y": float(final_next[1]),
                "final_next_error_m": final_error,
                "ms_correction_gain_m": gain,
                "ms_improved_provisional": bool(final_error < provisional_error),
                "current_ms_candidate_count": int(trace["candidate_count"]),
                "final_ms_candidate_count": int(final_ms["candidate_count"]),
            }
        )

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_reference_reset_scratch_frames.csv" % (route_name, arch.lower())
    )
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = rt.metric_summary(final_errors)
    summary.update(
        {
            "Architecture": arch,
            "Route": route_name,
            "Protocol": "reference-reset one-step delayed scratch",
            "LatencyFrames": 1,
            "ProvisionalNextMLE_m": float(np.mean(provisional_errors)),
            "FinalMS_MLE_m": float(np.mean(final_errors)),
            "MeanMSCorrectionGain_m": float(np.mean(correction_gains)),
            "MSCorrectionSuccess_pct": 100.0 * float(np.mean(correction_success)),
            "CurrentSearchCenter": "frame-aligned reference XY",
            "FinalNextSearchCenter": "KG/GK provisional next-frame XY",
            "FinalOutput": "MeanShift on next frame centered at provisional prediction",
            "MeanShiftCandidateCount": 36,
            "ForwardSearch": False,
            "CSV": str(csv_path),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval", "train-eval"), default="train-eval")
    parser.add_argument("--arch", choices=ARCH_CHOICES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epochs", type=int, default=int(getattr(config, "TEMPORAL_EPOCHS", 60))
    )
    parser.add_argument(
        "--lr", type=float, default=float(getattr(config, "TEMPORAL_LR", 3e-4))
    )
    args = parser.parse_args()

    device = rt.resolve_device(args.device)
    rt.set_seed(int(config.SEED))
    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)

    visual = FrozenVisualLocalizer(device)
    model = None
    if args.mode in ("train", "train-eval"):
        route_a = build_cache("route_A", visual, device)
        model = train_architecture(
            args.arch, visual, route_a, device, args.epochs, args.lr
        )
    if args.mode in ("eval", "train-eval"):
        if model is None:
            model = load_architecture(args.arch, device)
        results = {
            "architecture": args.arch,
            "train_route": "route_A",
            "test_routes": ["route_B", "route_C"],
            "protocol": "reference-reset one-step delayed scratch",
            "training_from_scratch": True,
            "current_ms_center": "frame-aligned reference XY",
            "final_next_ms_center": "KG/GK provisional next-frame XY",
            "meanshift_candidate_count": 36,
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
