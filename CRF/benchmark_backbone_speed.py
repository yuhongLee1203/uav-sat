#!/usr/bin/env python3
"""
True streaming end-to-end latency benchmark for the UAV-SAT localization model.

TIMED INTERVAL (steady state):
    prepared UAV tensor
      -> CPU->GPU transfer
      -> visual backbone
      -> UAV projection head
      -> 6x6 satellite candidate selection from cached gallery
      -> SAT projection head
      -> cosine retrieval logits / probabilities
      -> Fixed Hard Mean-Shift
      -> append current frame to temporal buffer
      -> T2-only RTL-CRF
      -> Correction Gate / final XY
      -> XY -> GPS latitude/longitude
    = one final GPS output

NOT INCLUDED:
    - image file disk I/O
    - PIL / torchvision image preprocessing
    - model/checkpoint loading
    - one-time satellite backbone-gallery creation

The satellite gallery is already stored in the trained visual checkpoint and is
loaded once at initialization. This matches the intended online deployment:
static satellite features are cached; each new UAV frame is processed online.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure true prepared-UAV-tensor -> final GPS latency and "
            "localization accuracy for one trained backbone."
        )
    )
    parser.add_argument(
        "--backbone",
        choices=(
            "mobileclip2_s2",
            "vgg16",
            "resnet18",
            "mobilenet_v3_small",
            "resnet50",
        ),
        required=True,
    )
    parser.add_argument(
        "--route",
        choices=("route_B", "route_C"),
        required=True,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        choices=(3, 4, 5),
    )
    parser.add_argument(
        "--jitter-m",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help=(
            "Number of valid GPS outputs excluded from steady-state latency "
            "statistics. Accuracy still uses the complete route."
        ),
    )
    parser.add_argument(
        "--max-timed-outputs",
        type=int,
        default=0,
        help=(
            "0 = time every output after warm-up. Positive value = only keep "
            "this many steady-state latency samples. Accuracy is always full route."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def configure_experiment(args):
    # config_backbone reads these at import time.
    os.environ["RTL_BACKBONE"] = args.backbone
    os.environ["RTL_TEMPORAL_WINDOW"] = str(args.window)

    import config_backbone as config

    # The MobileCLIP2-S2 baseline was already trained in the original strict
    # T2-only experiment rather than under backbone_ablation_mobileclip2_s2.
    # Reuse that exact A-only checkpoint if the ablation checkpoint is absent.
    if args.backbone == "mobileclip2_s2" and not config.VISUAL_CHECKPOINT.exists():
        strict_output = (
            config.PROJECT_ROOT
            / "outputs"
            / f"strict_train_A_test_BC_t2only_w{args.window}"
        )
        strict_ckpt = strict_output / "checkpoints"
        visual_ckpt = strict_ckpt / "visual_retrieval_A_only.pt"
        temporal_ckpt = strict_ckpt / "rtl_crf_A_only.pt"

        if visual_ckpt.exists() and temporal_ckpt.exists():
            config.OUTPUT_DIR = strict_output
            config.CHECKPOINT_DIR = strict_ckpt
            config.VISUAL_CHECKPOINT = visual_ckpt
            config.TEMPORAL_CHECKPOINT = temporal_ckpt

    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)

    # Force all repository modules imported below to see this ablation config.
    sys.modules["config"] = config

    # For MobileCLIP, use the exact production visual model rather than the
    # backbone-ablation wrapper.  The wrapper is required for torchvision
    # backbones, but even a thin wrapper changes the measured Python/model
    # path and makes its latency incomparable with the archived T2-only
    # production benchmark.
    if args.backbone == "mobileclip2_s2":
        import visual_model as visual_model
    else:
        import visual_model_backbone as visual_model

    # visual_localizer imports "visual_model" by this literal module name.
    sys.modules["visual_model"] = visual_model

    return config, visual_model


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def deterministic_jitter(length, route_index, maximum_m, seed):
    if maximum_m <= 0:
        return torch.zeros(length, 2, dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1009 * int(route_index))

    radius = (
        torch.sqrt(torch.rand(length, 1, generator=generator))
        * float(maximum_m)
    )
    angle = (
        torch.rand(length, 1, generator=generator)
        * (2.0 * math.pi)
    )
    return torch.cat(
        [radius * angle.cos(), radius * angle.sin()],
        dim=1,
    ).float()


def xy_to_latlon(x, y, origin_lat, origin_lon):
    """Exact inverse of data.meters_from_latlon()'s local tangent formula."""
    origin_lat_rad = math.radians(float(origin_lat))
    lat = float(origin_lat) + math.degrees(float(y) / EARTH_RADIUS_M)
    lon = float(origin_lon) + math.degrees(
        float(x) / (
            EARTH_RADIUS_M
            * max(abs(math.cos(origin_lat_rad)), 1e-12)
        )
    )
    return float(lat), float(lon)


def metric_block(prediction, gt, jump_tolerance_m):
    """
    Same metric definitions as robust_tracker.py:
      MLE, MedLE, P90/P95, ATE RMSE, LSR, RPE, JumpRate, etc.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    error = np.linalg.norm(prediction - gt, axis=1)

    if len(prediction) > 1:
        predicted_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)

        rpe = np.linalg.norm(
            predicted_step - gt_step,
            axis=1,
        )

        gt_step_length = np.linalg.norm(
            gt_step,
            axis=1,
        )

        jump_threshold = (
            float(np.percentile(gt_step_length, 99))
            + float(jump_tolerance_m)
        )

        predicted_step_length = np.linalg.norm(
            predicted_step,
            axis=1,
        )

        jump_rate = float(
            (predicted_step_length > jump_threshold).mean()
            * 100.0
        )

        stationary = gt_step_length < 1e-3
        stationary_drift = predicted_step_length[stationary]

        stationary_p90 = (
            float(np.percentile(stationary_drift, 90))
            if len(stationary_drift)
            else 0.0
        )

        path_ratio = float(
            predicted_step_length.sum()
            / max(gt_step_length.sum(), 1e-8)
        )
    else:
        rpe = np.zeros(1, dtype=np.float64)
        jump_rate = 0.0
        jump_threshold = 0.0
        stationary_p90 = 0.0
        path_ratio = 0.0

    return {
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "ATE_RMSE_m": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "LSR@5_pct": float(
            (error <= 5.0).mean() * 100.0
        ),
        "LSR@10_pct": float(
            (error <= 10.0).mean() * 100.0
        ),
        "LSR@15_pct": float(
            (error <= 15.0).mean() * 100.0
        ),
        "LSR@20_pct": float(
            (error <= 20.0).mean() * 100.0
        ),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": float(jump_rate),
        "JumpThreshold_m": float(jump_threshold),
        "StationaryDriftP90_m": float(stationary_p90),
        "PathLengthRatio": float(path_ratio),
        "MaxLE_m": float(error.max()),
    }


def latency_stats(values_ms):
    values = np.asarray(values_ms, dtype=np.float64)

    if values.size == 0:
        return {
            "samples": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "gps_outputs_per_second": None,
        }

    mean_ms = float(values.mean())

    return {
        "samples": int(values.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        # Latency-based streaming FPS: one new frame -> one new GPS output.
        "gps_outputs_per_second": float(
            1000.0 / max(mean_ms, 1e-12)
        ),
    }


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_temporal_model(config, visual_model, device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Missing temporal checkpoint: "
            f"{config.TEMPORAL_CHECKPOINT}"
        )

    model = visual_model.TemporalLatticeCRF().to(device)

    checkpoint = torch.load(
        config.TEMPORAL_CHECKPOINT,
        map_location=device,
    )

    state = (
        checkpoint.get("best_model")
        or checkpoint["model"]
    )

    model.load_state_dict(
        state,
        strict=True,
    )
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, checkpoint


def temporal_input_from_buffer(buffer, device):
    z_uav = torch.stack(
        [entry["z_uav"][0] for entry in buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    z_sat = torch.stack(
        [entry["z_sat"][0] for entry in buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    centers = torch.stack(
        [entry["centers"][0] for entry in buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    raw_logits = torch.stack(
        [entry["raw_logits"][0] for entry in buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    raw_prob = torch.stack(
        [entry["raw_prob"][0] for entry in buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    hardms = torch.stack(
        [entry["hardms"][0] for entry in buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    frame_ids = torch.tensor(
        [[entry["frame_id"] for entry in buffer]],
        dtype=torch.long,
        device=device,
    )

    return (
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        centers,
        frame_ids,
        hardms,
    )


@torch.inference_mode()
def run_route(
    args,
    config,
    visual_model,
    device,
):
    from data import RouteDataset
    from visual_localizer import FrozenVisualLocalizer

    route_map = {
        "route_B": (1, Path(config.ROUTE_ROOTS[1])),
        "route_C": (2, Path(config.ROUTE_ROOTS[2])),
    }

    route_index, route_root = route_map[args.route]

    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Missing visual checkpoint: "
            f"{config.VISUAL_CHECKPOINT}"
        )

    print(
        f"loading visual checkpoint: {config.VISUAL_CHECKPOINT}",
        flush=True,
    )
    visual = FrozenVisualLocalizer(device)

    print(
        f"loading temporal checkpoint: {config.TEMPORAL_CHECKPOINT}",
        flush=True,
    )
    temporal, temporal_checkpoint = load_temporal_model(
        config,
        visual_model,
        device,
    )

    dataset = RouteDataset(
        route_root,
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )

    jitter = deterministic_jitter(
        len(dataset),
        route_index=route_index,
        maximum_m=float(args.jitter_m),
        seed=int(config.SEED),
    )

    window = int(args.window)
    buffer = deque(maxlen=window)

    pred_xy_rows = []
    gt_xy_rows = []
    pred_latlon_rows = []
    gt_latlon_rows = []
    frame_rows = []
    capture_rows = []
    gate_rows = []
    all_output_latency_ms = []
    steady_latency_ms = []
    first_frame_compute_ms = []
    csv_rows = []

    valid_output_count = 0
    timed_output_count = 0

    print(
        f"route={args.route} frames={len(dataset)} "
        f"window={window} backbone={args.backbone}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Streaming evaluation.
    #
    # dataset[index] (disk I/O + image transform) happens BEFORE the timer.
    # Timing begins with the prepared CPU UAV tensor entering the model.
    # ------------------------------------------------------------------
    for index in range(len(dataset)):
        item = dataset[index]

        # Prepared model input; disk I/O and preprocessing are not timed.
        uav_cpu = item["uav"].unsqueeze(0)

        gt_xy_cpu = item["xy"].float()
        prior_xy_cpu = (
            gt_xy_cpu
            + jitter[index]
        ).unsqueeze(0)

        frame_id = int(item["frame_id"])

        # Synchronize so previous asynchronous CUDA work cannot leak into
        # this frame's measured interval.
        cuda_sync(device)
        start_time = time.perf_counter()

        # 1) Current UAV tensor -> visual backbone.
        uav_clip = visual.encode_uav_clip(
            uav_cpu
        )

        # 2) Candidate selection + SAT head + retrieval + HardMS.
        # Match the current evaluation protocol by moving the GT+jitter prior
        # to the device before candidate_batch.
        candidate = visual.candidate_batch(
            uav_clip,
            prior_xy_cpu.to(
                device,
                non_blocking=True,
            ),
            grid_size=int(config.GRID_SIZE),
        )

        buffer.append(
            {
                "frame_id": frame_id,
                "z_uav": candidate.z_uav,
                "z_sat": candidate.z_sat,
                "centers": candidate.centers,
                "raw_logits": candidate.raw_logits,
                "raw_prob": candidate.raw_prob,
                "hardms": candidate.hardms_xy,
            }
        )

        final_xy = None
        final_latlon = None
        correction_gate = None

        # 3) Once the temporal window is full:
        #    T2-only RTL-CRF -> correction gate -> final XY -> GPS.
        if len(buffer) == window:
            (
                z_uav,
                z_sat,
                raw_logits,
                raw_prob,
                centers,
                frame_ids,
                hardms,
            ) = temporal_input_from_buffer(
                buffer,
                device,
            )

            output = temporal(
                z_uav,
                z_sat,
                raw_logits,
                raw_prob,
                centers,
                frame_ids,
                hardms,
                target_index=None,
            )

            final_xy_tensor = output.final_xy[0]

            # GPU result -> CPU scalar values is part of producing an actual
            # usable GPS result.
            x_meter = float(
                final_xy_tensor[0].item()
            )
            y_meter = float(
                final_xy_tensor[1].item()
            )

            final_latlon = xy_to_latlon(
                x_meter,
                y_meter,
                visual.origin_lat,
                visual.origin_lon,
            )
            final_xy = (x_meter, y_meter)

            correction_gate = float(
                output.correction_gate[0].item()
            )

        cuda_sync(device)
        end_time = time.perf_counter()

        frame_compute_ms = (
            (end_time - start_time) * 1000.0
        )

        # The first GPS requires WINDOW prepared UAV frames.
        if index < window:
            first_frame_compute_ms.append(
                frame_compute_ms
            )

        if final_xy is None:
            continue

        valid_output_count += 1

        # Current-frame model input -> current GPS output latency.
        all_output_latency_ms.append(
            frame_compute_ms
        )

        # Warm-up is excluded only from latency statistics, never accuracy.
        after_warmup = (
            valid_output_count
            > int(args.warmup)
        )

        under_limit = (
            int(args.max_timed_outputs) <= 0
            or timed_output_count
            < int(args.max_timed_outputs)
        )

        if after_warmup and under_limit:
            steady_latency_ms.append(
                frame_compute_ms
            )
            timed_output_count += 1

        gt_xy = (
            float(gt_xy_cpu[0].item()),
            float(gt_xy_cpu[1].item()),
        )

        gt_latlon = (
            float(item["latlon"][0].item()),
            float(item["latlon"][1].item()),
        )

        capture = bool(
            visual.candidate_contains_gt_anchor(
                candidate.indices,
                gt_xy_cpu.unsqueeze(0).to(device),
            )[0].item()
        )

        error_m = math.hypot(
            final_xy[0] - gt_xy[0],
            final_xy[1] - gt_xy[1],
        )

        pred_xy_rows.append(final_xy)
        gt_xy_rows.append(gt_xy)
        pred_latlon_rows.append(final_latlon)
        gt_latlon_rows.append(gt_latlon)
        frame_rows.append(frame_id)
        capture_rows.append(capture)
        gate_rows.append(correction_gate)

        csv_rows.append(
            {
                "frame_id": frame_id,
                "gt_x_m": gt_xy[0],
                "gt_y_m": gt_xy[1],
                "pred_x_m": final_xy[0],
                "pred_y_m": final_xy[1],
                "gt_lat": gt_latlon[0],
                "gt_lon": gt_latlon[1],
                "pred_lat": final_latlon[0],
                "pred_lon": final_latlon[1],
                "error_m": error_m,
                "latency_ms": frame_compute_ms,
                "correction_gate": correction_gate,
                "candidate_capture": int(capture),
            }
        )

        if (
            valid_output_count == 1
            or valid_output_count % 250 == 0
        ):
            print(
                f"{args.route}: output "
                f"{valid_output_count}/"
                f"{max(len(dataset) - window + 1, 0)} "
                f"frame={frame_id} "
                f"latency={frame_compute_ms:.3f} ms "
                f"error={error_m:.3f} m",
                flush=True,
            )

    if not pred_xy_rows:
        raise RuntimeError(
            "No valid temporal GPS outputs were produced."
        )

    accuracy = metric_block(
        pred_xy_rows,
        gt_xy_rows,
        jump_tolerance_m=float(
            config.JUMP_TOLERANCE_M
        ),
    )

    # Sum of the model-compute times of the first WINDOW prepared UAV tensors:
    # input frame 1 ... input frame WINDOW -> first final GPS.
    first_gps_ms = float(
        sum(first_frame_compute_ms[:window])
    )

    output_latency = latency_stats(
        all_output_latency_ms
    )
    steady = latency_stats(
        steady_latency_ms
    )

    result = {
        "backbone": args.backbone,
        "backbone_name": config.BACKBONE_NAME,
        "route": args.route,
        "temporal_window": window,
        "jitter_m": float(args.jitter_m),
        "device": str(device),
        "checkpoints": {
            "visual": str(
                config.VISUAL_CHECKPOINT
            ),
            "temporal": str(
                config.TEMPORAL_CHECKPOINT
            ),
        },
        "timing_definition": {
            "steady_state": (
                "prepared UAV tensor -> CPU/GPU transfer -> visual backbone "
                "-> retrieval heads -> 6x6 candidate selection -> cosine "
                "retrieval -> Fixed HardMS -> T2-only RTL-CRF -> correction "
                "gate -> final XY -> GPS latitude/longitude"
            ),
            "first_gps_output": (
                f"sum of model-compute latency for the first {window} "
                "prepared UAV tensors required to produce the first GPS output"
            ),
            "included": [
                "CPU-to-GPU tensor transfer performed by the model path",
                "visual backbone",
                "UAV projection head",
                "6x6 candidate selection",
                "SAT projection head using cached satellite backbone features",
                "cosine retrieval logits and probabilities",
                "Fixed Hard Mean-Shift",
                "T2-only RTL-CRF",
                "Correction Gate",
                "final XY to GPS latitude/longitude conversion",
            ],
            "excluded": [
                "image disk I/O",
                "PIL/torchvision preprocessing",
                "checkpoint/model loading",
                "one-time satellite backbone-gallery construction",
            ],
        },
        "accuracy": accuracy,
        "candidate_capture_rate_pct": float(
            np.mean(capture_rows) * 100.0
        ),
        "mean_correction_gate": float(
            np.mean(gate_rows)
        ),
        "latency": {
            "first_gps_output_ms": first_gps_ms,
            "all_valid_outputs": output_latency,
            "steady_state_after_warmup": steady,
            "warmup_outputs_excluded_from_latency": int(
                args.warmup
            ),
        },
        "counts": {
            "route_frames": int(len(dataset)),
            "gps_outputs": int(
                len(pred_xy_rows)
            ),
            "timed_steady_state_outputs": int(
                len(steady_latency_ms)
            ),
        },
        "temporal_checkpoint_architecture": temporal_checkpoint.get(
            "architecture",
            "unknown",
        ),
    }

    return result, csv_rows


def write_outputs(args, result, csv_rows):
    if args.output is None:
        out_dir = Path(
            "outputs/full_pipeline_latency"
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path = (
            out_dir
            / (
                f"{args.backbone}_"
                f"w{args.window}_"
                f"{args.route}.json"
            )
        )
    else:
        output_path = args.output
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_path.with_suffix(
        ".frames.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = list(
            csv_rows[0].keys()
        )
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    return output_path, csv_path


def print_summary(result):
    accuracy = result["accuracy"]
    steady = result["latency"][
        "steady_state_after_warmup"
    ]

    print()
    print("=" * 78)
    print("TRUE END-TO-END UAV TENSOR -> FINAL GPS RESULT")
    print("=" * 78)
    print(
        f"Backbone       : {result['backbone']}"
    )
    print(
        f"Route          : {result['route']}"
    )
    print(
        f"Temporal window: {result['temporal_window']}"
    )
    print()
    print(
        f"MLE            : {accuracy['MLE_m']:.4f} m"
    )
    print(
        f"P90            : {accuracy['P90_m']:.4f} m"
    )
    print(
        f"RPE            : {accuracy['RPE_m']:.4f} m"
    )
    print(
        f"JumpRate       : {accuracy['JumpRate_pct']:.4f} %"
    )
    print()
    print(
        "First GPS      : "
        f"{result['latency']['first_gps_output_ms']:.3f} ms "
        f"({result['temporal_window']} input frames)"
    )

    if steady["mean_ms"] is not None:
        print(
            f"E2E mean       : {steady['mean_ms']:.3f} ms / GPS"
        )
        print(
            f"E2E median     : {steady['median_ms']:.3f} ms / GPS"
        )
        print(
            f"E2E P95        : {steady['p95_ms']:.3f} ms / GPS"
        )
        print(
            "Complete FPS   : "
            f"{steady['gps_outputs_per_second']:.2f} GPS outputs/s"
        )
    else:
        print(
            "No steady-state samples remain after warm-up."
        )

    print("=" * 78)


def main():
    args = parse_args()

    config, visual_model = configure_experiment(
        args
    )

    set_seed(
        int(config.SEED)
    )

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    result, csv_rows = run_route(
        args,
        config,
        visual_model,
        device,
    )

    output_path, csv_path = write_outputs(
        args,
        result,
        csv_rows,
    )

    print_summary(result)

    print(
        f"JSON saved: {output_path}",
        flush=True,
    )
    print(
        f"Frame CSV saved: {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
