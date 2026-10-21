"""Unified eight-architecture experiment under one fixed 8 m protocol.

Architectures:
  MKG, MGK, GMK, GKM, KGM, KMG
  delayKG, delayGK

All formal runs share the exact protocol in unified_protocol.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import robust_tracker_base as rb
import unified_protocol as up
from six_architecture_model import PositionRefinementGRU
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

SAME_ARCH = ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG")
DELAY_ARCH = ("delayKG", "delayGK")
ARCH_CHOICES = SAME_ARCH + DELAY_ARCH
OUTPUT_ROOT = Path(config.BACKBONE_OUTPUT_DIR) / "unified_fixed8m_v1"
MAIN_ROOT = OUTPUT_ROOT / "main_architectures"


def new_model(device):
    return PositionRefinementGRU(
        feature_dim=int(getattr(config, "RNN_FEATURE_DIM", 128)),
        hidden_dim=int(getattr(config, "RNN_HIDDEN_DIM", 256)),
        dropout=float(getattr(config, "RNN_DROPOUT", 0.0)),
    ).to(device)


def checkpoint_path(arch):
    return MAIN_ROOT / arch.lower() / "temporal_checkpoint.pt"


def output_dir(arch):
    return MAIN_ROOT / arch.lower()


def _reference_np(cache, index):
    return cache.gt_xy[index].detach().cpu().numpy().astype(np.float64).reshape(2)


def _initial_same_state(cache, route_name):
    ref0 = _reference_np(cache, 0)
    center0 = up.search_center(ref0, 0, route_name)
    return up.make_state(center0)


def _capture_from_visual_result(result, reference_xy):
    ref = up.xy_tensor(reference_xy, result["centers"].device)
    nearest = torch.linalg.norm(
        result["centers"] - ref[:, None, :], dim=2
    ).min(dim=1).values
    captured = nearest <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
    return float(nearest[0].item()), bool(captured[0].item())


def same_frame_forward(
    arch, model, visual, uav_clip, coarse_center_xy, state, device,
    grid_size=up.MAIN_GRID_SIZE, decoder=up.MAIN_DECODER,
    bandwidth_m=up.MAIN_BANDWIDTH_M, tau=up.MAIN_TAU,
):
    base = up.decode_visual(
        visual,
        uav_clip,
        coarse_center_xy,
        grid_size=grid_size,
        decoder=decoder,
        bandwidth_m=bandwidth_m,
        tau=tau,
    )
    stage_xy = base["xy"]
    variance = base["variance"]
    z_uav = base["z_uav"]
    trace = {
        "base": base,
        "base_ms_xy": stage_xy,
        "base_ms_support": base["support"],
    }
    gru_out = None

    symbols = arch[1:] if arch.startswith("M") else arch
    for symbol in symbols:
        if symbol == "M":
            correction = up.decode_visual(
                visual,
                uav_clip,
                stage_xy[0].detach().cpu().numpy(),
                grid_size=grid_size,
                decoder=decoder,
                bandwidth_m=bandwidth_m,
                tau=tau,
            )
            stage_xy = correction["xy"]
            variance = correction["variance"]
            trace["second_m"] = correction
            trace["center_ms_xy"] = stage_xy
        elif symbol == "K":
            stage_xy = up.apply_k(state, stage_xy, variance, device)
            trace["kalman_xy"] = stage_xy
        elif symbol == "G":
            stage_xy, gru_out = up.apply_g(
                model, stage_xy, variance, z_uav, state
            )
            trace["gru_xy"] = stage_xy
        else:
            raise ValueError("unknown architecture symbol: %s" % symbol)

    state["previous_z"] = z_uav.detach()
    return stage_xy, variance, gru_out, trace


def _pair_g(model, stage_xy, variance, z_current, z_next, state):
    out = model.forward_step(
        stage_xy=stage_xy,
        variance_xy=variance,
        z_uav=z_next,
        previous_z_uav=z_current,
        hidden=state["hidden"],
    )
    state["hidden"] = out.hidden
    return out.corrected_xy, out


def delayed_pair_step(
    arch, model, visual, uav_current, uav_next, coarse_current_xy, state, device,
    grid_size=up.MAIN_GRID_SIZE, decoder=up.MAIN_DECODER,
    bandwidth_m=up.MAIN_BANDWIDTH_M, tau=up.MAIN_TAU,
):
    current_m = up.decode_visual(
        visual,
        uav_current,
        coarse_current_xy,
        grid_size=grid_size,
        decoder=decoder,
        bandwidth_m=bandwidth_m,
        tau=tau,
    )
    current_xy = current_m["xy"]
    variance = current_m["variance"]
    z_current = current_m["z_uav"]
    with torch.no_grad():
        z_next = visual.model.encode_uav_from_clip(uav_next)

    trace = {"current_m": current_m, "current_ms_xy": current_xy}
    if arch == "delayKG":
        k_xy = up.apply_k(state, current_xy, variance, device)
        provisional, gru_out = _pair_g(
            model, k_xy, variance, z_current, z_next, state
        )
        trace["kalman_current_xy"] = k_xy
        trace["gru_next_xy"] = provisional
    elif arch == "delayGK":
        g_xy, gru_out = _pair_g(
            model, current_xy, variance, z_current, z_next, state
        )
        provisional = up.apply_k(state, g_xy, variance, device)
        trace["gru_next_xy"] = g_xy
        trace["kalman_next_xy"] = provisional
    else:
        raise ValueError("not delayed architecture: %s" % arch)
    trace["provisional_next_xy"] = provisional
    return current_xy, provisional, gru_out, trace


def preflight_visual(visual, device, formal_assert=True):
    reports = {}
    for route_name in ("route_A", "route_B", "route_C"):
        cache = up.build_cache(route_name, visual, device)
        report = up.capture_report(
            visual,
            cache,
            route_name,
            jitter_m=up.MAIN_JITTER_M,
            grid_size=up.MAIN_GRID_SIZE,
            seed=up.MAIN_SEED,
        )
        if formal_assert:
            up.assert_capture(report)
        reports[route_name] = report
        print("[capture] " + json.dumps(report, ensure_ascii=False), flush=True)
    payload = {
        "protocol": up.protocol_metadata(),
        "capture": reports,
    }
    up.write_json(OUTPUT_ROOT / "formal_capture_preflight.json", payload)
    return reports


def train_same(arch, visual, route_a, device, epochs, lr, tbptt):
    rb.set_seed(up.MAIN_SEED)
    model = new_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        state = _initial_same_state(route_a, "route_A")
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = None
        chunk_count = 0
        epoch_losses = []
        gru_errors = []

        for index in range(len(route_a)):
            ref = _reference_np(route_a, index)
            center = up.search_center(ref, index, "route_A")
            uav = route_a.uav_clip[index:index + 1].to(device).float()
            _, _, gru_out, _ = same_frame_forward(
                arch, model, visual, uav, center, state, device
            )
            if gru_out is None:
                raise RuntimeError("%s did not execute G" % arch)
            target = route_a.gt_xy[index:index + 1].to(device).float()
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1
            with torch.no_grad():
                gru_errors.append(
                    float(
                        torch.linalg.norm(
                            gru_out.corrected_xy - target, dim=1
                        ).mean().cpu()
                    )
                )

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

        mean_loss = float(np.mean(epoch_losses))
        mean_gru = float(np.mean(gru_errors))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": arch,
                    "model": best_state,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "protocol": up.protocol_metadata(),
                    "training_from_scratch": True,
                },
                checkpoint_path(arch),
            )
        print(
            "[%s unified8] epoch=%03d/%d loss=%.6f gru_mle=%.3fm best=%.6f"
            % (arch, epoch, epochs, mean_loss, mean_gru, best_loss),
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    return model


def _delayed_pair_state(route_a, route_name, current_index):
    current_ref = _reference_np(route_a, current_index)
    previous_ref = _reference_np(route_a, current_index - 1)
    current_center = up.search_center(current_ref, current_index, route_name)
    previous_center = up.search_center(
        previous_ref, current_index - 1, route_name
    )
    velocity = current_center - previous_center
    return up.make_state(current_center, velocity), current_center


def train_delayed(arch, visual, route_a, device, epochs, lr):
    rb.set_seed(up.MAIN_SEED)
    model = new_model(device)
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
        for index in range(1, len(route_a) - 1):
            state, current_center = _delayed_pair_state(
                route_a, "route_A", index
            )
            uav_current = route_a.uav_clip[index:index + 1].to(device).float()
            uav_next = route_a.uav_clip[index + 1:index + 2].to(device).float()
            target_next = route_a.gt_xy[index + 1:index + 2].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            _, provisional, gru_out, _ = delayed_pair_step(
                arch,
                model,
                visual,
                uav_current,
                uav_next,
                current_center,
                state,
                device,
            )
            loss = F.smooth_l1_loss(gru_out.corrected_xy, target_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.GRAD_CLIP_NORM)
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                provisional_errors.append(
                    float(
                        torch.linalg.norm(
                            provisional - target_next, dim=1
                        ).mean().cpu()
                    )
                )

        mean_loss = float(np.mean(losses))
        provisional_mle = float(np.mean(provisional_errors))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            checkpoint_path(arch).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": arch,
                    "model": best_state,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "protocol": up.protocol_metadata(),
                    "training_from_scratch": True,
                    "delay_frames": 1,
                    "pair_reset": "coarse current/previous centers, both fixed-8m perturbed",
                    "next_ms_center": "architecture provisional next XY",
                },
                checkpoint_path(arch),
            )
        print(
            "[%s unified8] epoch=%03d/%d loss=%.6f provisional_mle=%.3fm best=%.6f"
            % (arch, epoch, epochs, mean_loss, provisional_mle, best_loss),
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    return model


def load_model(arch, device):
    path = checkpoint_path(arch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    protocol = payload.get("protocol", {})
    if float(protocol.get("main_jitter_m", -1.0)) != float(up.MAIN_JITTER_M):
        raise RuntimeError("checkpoint is not unified fixed-8m protocol")
    model = new_model(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_same(arch, visual, model, route_name, cache, device):
    report = up.capture_report(visual, cache, route_name)
    up.assert_capture(report)
    state = _initial_same_state(cache, route_name)
    final_errors = []
    base_errors = []
    rows = []
    second_capture = []

    for index in range(len(cache)):
        ref = _reference_np(cache, index)
        center = up.search_center(ref, index, route_name)
        uav = cache.uav_clip[index:index + 1].to(device).float()
        final_t, variance, _, trace = same_frame_forward(
            arch, model, visual, uav, center, state, device
        )
        final_xy = final_t[0].cpu().numpy().astype(np.float64)
        base_xy = trace["base_ms_xy"][0].cpu().numpy().astype(np.float64)
        final_error = float(np.linalg.norm(final_xy - ref))
        base_error = float(np.linalg.norm(base_xy - ref))
        final_errors.append(final_error)
        base_errors.append(base_error)

        second_nearest = None
        second_ok = None
        if "second_m" in trace:
            second_nearest, second_ok = _capture_from_visual_result(
                trace["second_m"], ref
            )
            second_capture.append(float(second_ok))

        row = {
            "frame_id": int(cache.frame_ids[index]),
            "reference_x": float(ref[0]),
            "reference_y": float(ref[1]),
            "search_center_x": float(center[0]),
            "search_center_y": float(center[1]),
            "search_center_error_m": float(np.linalg.norm(center - ref)),
            "base_ms_x": float(base_xy[0]),
            "base_ms_y": float(base_xy[1]),
            "base_ms_error_m": base_error,
            "final_x": float(final_xy[0]),
            "final_y": float(final_xy[1]),
            "error_final_m": final_error,
            "variance_x": float(variance[0, 0]),
            "variance_y": float(variance[0, 1]),
        }
        if second_nearest is not None:
            row["second_ms_nearest_candidate_m"] = float(second_nearest)
            row["second_ms_capture"] = bool(second_ok)
        rows.append(row)

        if state["hidden"] is not None:
            state["hidden"] = state["hidden"].detach()
        if state["previous_z"] is not None:
            state["previous_z"] = state["previous_z"].detach()

    out = output_dir(arch)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / ("%s_frames.csv" % route_name)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = rb.metric_summary(final_errors)
    summary.update(
        {
            "Architecture": arch,
            "Route": route_name,
            "BaseVisualMS_MLE_m": float(np.mean(base_errors)),
            "SearchCenterMLE_m": float(up.MAIN_JITTER_M),
            "FormalCandidateCaptureRate_pct": report["CandidateCaptureRate_pct"],
            "SecondMSCaptureRate_pct": (
                100.0 * float(np.mean(second_capture))
                if second_capture else None
            ),
            "CSV": str(csv_path),
        }
    )
    return summary


@torch.no_grad()
def evaluate_delayed(arch, visual, model, route_name, cache, device):
    report = up.capture_report(visual, cache, route_name)
    up.assert_capture(report)
    provisional_errors = []
    final_errors = []
    current_ms_errors = []
    final_capture = []
    rows = []

    for index in range(1, len(cache) - 1):
        state, current_center = _delayed_pair_state(cache, route_name, index)
        current_ref = _reference_np(cache, index)
        next_ref = _reference_np(cache, index + 1)
        uav_current = cache.uav_clip[index:index + 1].to(device).float()
        uav_next = cache.uav_clip[index + 1:index + 2].to(device).float()

        current_t, provisional_t, _, _ = delayed_pair_step(
            arch,
            model,
            visual,
            uav_current,
            uav_next,
            current_center,
            state,
            device,
        )
        current_xy = current_t[0].cpu().numpy().astype(np.float64)
        provisional = provisional_t[0].cpu().numpy().astype(np.float64)

        final_m = up.decode_visual(
            visual,
            uav_next,
            provisional,
            grid_size=up.MAIN_GRID_SIZE,
            decoder=up.MAIN_DECODER,
            bandwidth_m=up.MAIN_BANDWIDTH_M,
            tau=up.MAIN_TAU,
        )
        final_xy = final_m["xy"][0].cpu().numpy().astype(np.float64)
        nearest, captured = _capture_from_visual_result(final_m, next_ref)

        current_error = float(np.linalg.norm(current_xy - current_ref))
        provisional_error = float(np.linalg.norm(provisional - next_ref))
        final_error = float(np.linalg.norm(final_xy - next_ref))
        current_ms_errors.append(current_error)
        provisional_errors.append(provisional_error)
        final_errors.append(final_error)
        final_capture.append(float(captured))

        rows.append(
            {
                "current_frame_id": int(cache.frame_ids[index]),
                "next_frame_id": int(cache.frame_ids[index + 1]),
                "current_reference_x": float(current_ref[0]),
                "current_reference_y": float(current_ref[1]),
                "current_search_center_x": float(current_center[0]),
                "current_search_center_y": float(current_center[1]),
                "current_search_center_error_m": float(
                    np.linalg.norm(current_center - current_ref)
                ),
                "current_ms_error_m": current_error,
                "provisional_next_x": float(provisional[0]),
                "provisional_next_y": float(provisional[1]),
                "provisional_next_error_m": provisional_error,
                "final_next_x": float(final_xy[0]),
                "final_next_y": float(final_xy[1]),
                "final_next_error_m": final_error,
                "final_next_nearest_candidate_m": float(nearest),
                "final_next_candidate_capture": bool(captured),
                "ms_correction_gain_m": provisional_error - final_error,
            }
        )

    out = output_dir(arch)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / ("%s_frames.csv" % route_name)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = rb.metric_summary(final_errors)
    summary.update(
        {
            "Architecture": arch,
            "Route": route_name,
            "LatencyFrames": 1,
            "CurrentMS_MLE_m": float(np.mean(current_ms_errors)),
            "ProvisionalNextMLE_m": float(np.mean(provisional_errors)),
            "FinalMS_MLE_m": float(np.mean(final_errors)),
            "MeanMSCorrectionGain_m": float(
                np.mean(np.asarray(provisional_errors) - np.asarray(final_errors))
            ),
            "FinalNextCandidateCaptureRate_pct": 100.0 * float(np.mean(final_capture)),
            "FormalCurrentCandidateCaptureRate_pct": report[
                "CandidateCaptureRate_pct"
            ],
            "CurrentSearchCenterMLE_m": float(up.MAIN_JITTER_M),
            "CSV": str(csv_path),
        }
    )
    return summary


def save_summary(arch, results):
    payload = {
        "architecture": arch,
        "protocol": up.protocol_metadata(),
        "training_from_scratch": True,
        "results": results,
    }
    up.write_json(output_dir(arch) / "summary.json", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare-visual", "preflight", "train", "eval", "train-eval"),
        default="train-eval",
    )
    parser.add_argument("--arch", choices=ARCH_CHOICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--visual-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=float(config.TEMPORAL_LR))
    parser.add_argument("--tbptt", type=int, default=int(config.TBPTT_STEPS))
    args = parser.parse_args()

    device = rb.resolve_device(args.device)
    rb.set_seed(up.MAIN_SEED)

    if args.mode == "prepare-visual":
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(up.MAIN_JITTER_M),
            resume=False,
        )
        return

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "%s -- run --mode prepare-visual first" % config.VISUAL_CHECKPOINT
        )

    visual = FrozenVisualLocalizer(device)
    if args.mode == "preflight":
        preflight_visual(visual, device, formal_assert=True)
        return

    if args.arch is None:
        raise SystemExit("--arch is required for train/eval")

    route_a = None
    if args.mode in ("train", "train-eval"):
        route_a = up.build_cache("route_A", visual, device)
        up.assert_capture(up.capture_report(visual, route_a, "route_A"))

    model = None
    if args.mode in ("train", "train-eval"):
        if args.arch in SAME_ARCH:
            model = train_same(
                args.arch, visual, route_a, device,
                args.epochs, args.lr, args.tbptt
            )
        else:
            model = train_delayed(
                args.arch, visual, route_a, device,
                args.epochs, args.lr
            )

    if args.mode in ("eval", "train-eval"):
        if model is None:
            model = load_model(args.arch, device)
        result_map = {}
        for route_name in ("route_B", "route_C"):
            cache = up.build_cache(route_name, visual, device)
            if args.arch in SAME_ARCH:
                summary = evaluate_same(
                    args.arch, visual, model, route_name, cache, device
                )
            else:
                summary = evaluate_delayed(
                    args.arch, visual, model, route_name, cache, device
                )
            result_map[route_name] = summary
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        save_summary(args.arch, result_map)


if __name__ == "__main__":
    main()
