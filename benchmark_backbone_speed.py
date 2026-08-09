#!/usr/bin/env python3
"""
Full end-to-end UAV localization latency benchmark.

Main timed steady-state path:
    prepared UAV image tensor
        -> GPU transfer
        -> visual backbone
        -> UAV retrieval head
        -> 36 local SAT candidate selection
        -> SAT retrieval head from precomputed SAT backbone gallery
        -> cosine retrieval scores / probability
        -> Fixed Hard Mean-Shift
        -> 5-frame T2-only RTL-CRF
        -> correction gate / final local (x, y)
        -> GPS latitude / longitude

Excluded from the main "model_input_to_gps" latency:
    - checkpoint/model loading
    - satellite gallery loading / precomputation
    - disk image read / PIL decode
    - RouteDataset CPU transform
    - camera waiting time
    - controlled GT+jitter prior construction
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


EARTH_RADIUS_M = 6378137.0

SUPPORTED_BACKBONES = (
    "mobileclip2_s2",
    "vgg16",
    "resnet18",
    "mobilenet_v3_small",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure complete localization latency from one prepared UAV "
            "image tensor entering the model to one final GPS coordinate."
        )
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        required=True,
    )
    parser.add_argument(
        "--route",
        choices=("route_A", "route_B", "route_C"),
        default="route_B",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        choices=(3, 4, 5),
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--first-output-repeats", type=int, default=20)
    parser.add_argument("--jitter-m", type=float, default=12.0)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def configure_modules(backbone, window):
    os.environ["RTL_TEMPORAL_WINDOW"] = str(int(window))

    if backbone == "mobileclip2_s2":
        config = importlib.import_module("config")
        visual_model = importlib.import_module("visual_model")
        return config, visual_model

    os.environ["RTL_BACKBONE"] = backbone

    config = importlib.import_module("config_backbone")
    sys.modules["config"] = config

    visual_model = importlib.import_module("visual_model_backbone")
    sys.modules["visual_model"] = visual_model

    return config, visual_model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_summary_ms(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}

    mean_ms = float(array.mean())

    return {
        "count": int(array.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(array)),
        "std_ms": float(array.std()),
        "min_ms": float(array.min()),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "gps_outputs_per_second": float(
            1000.0 / max(mean_ms, 1e-12)
        ),
    }


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
        [
            radius * angle.cos(),
            radius * angle.sin(),
        ],
        dim=1,
    ).float()


def local_xy_to_latlon(
    x_meter,
    y_meter,
    origin_lat,
    origin_lon,
):
    origin_lat_rad = math.radians(float(origin_lat))

    latitude = float(origin_lat) + math.degrees(
        float(y_meter) / EARTH_RADIUS_M
    )

    longitude = float(origin_lon) + math.degrees(
        float(x_meter)
        / (
            EARTH_RADIUS_M
            * max(abs(math.cos(origin_lat_rad)), 1e-12)
        )
    )

    return latitude, longitude


def route_info(config, route_name):
    catalog = {
        name: (index, Path(root))
        for index, (name, root) in enumerate(
            zip(config.ROUTE_NAMES, config.ROUTE_ROOTS)
        )
    }

    if route_name not in catalog:
        raise ValueError(
            "Unknown route {!r}; available={}".format(
                route_name,
                sorted(catalog),
            )
        )

    return catalog[route_name]


def load_temporal_model(config, temporal_class, device):
    checkpoint_path = Path(config.TEMPORAL_CHECKPOINT)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Temporal checkpoint does not exist:\n  {}\n"
            "Run the corresponding trained experiment first.".format(
                checkpoint_path
            )
        )

    model = temporal_class().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state = checkpoint.get("best_model")
    if state is None:
        state = checkpoint.get("model")

    if state is None:
        raise RuntimeError(
            "{} contains neither 'best_model' nor 'model'".format(
                checkpoint_path
            )
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def candidate_record(candidate, frame_id):
    device = candidate.z_uav.device

    return {
        "z_uav": candidate.z_uav[0],
        "z_sat": candidate.z_sat[0],
        "raw_logits": candidate.raw_logits[0],
        "raw_prob": candidate.raw_prob[0],
        "centers": candidate.centers[0],
        "hardms": candidate.hardms_xy[0],
        "frame_id": torch.tensor(
            int(frame_id),
            dtype=torch.long,
            device=device,
        ),
    }


def make_temporal_batch(history):
    rows = list(history)

    return {
        "z_uav": torch.stack(
            [row["z_uav"] for row in rows],
            dim=0,
        ).unsqueeze(0),
        "z_sat": torch.stack(
            [row["z_sat"] for row in rows],
            dim=0,
        ).unsqueeze(0),
        "raw_logits": torch.stack(
            [row["raw_logits"] for row in rows],
            dim=0,
        ).unsqueeze(0),
        "raw_prob": torch.stack(
            [row["raw_prob"] for row in rows],
            dim=0,
        ).unsqueeze(0),
        "centers": torch.stack(
            [row["centers"] for row in rows],
            dim=0,
        ).unsqueeze(0),
        "frame_ids": torch.stack(
            [row["frame_id"] for row in rows],
            dim=0,
        ).unsqueeze(0),
        "hardms": torch.stack(
            [row["hardms"] for row in rows],
            dim=0,
        ).unsqueeze(0),
    }


@torch.inference_mode()
def temporal_forward(model, batch):
    return model(
        batch["z_uav"],
        batch["z_sat"],
        batch["raw_logits"],
        batch["raw_prob"],
        batch["centers"],
        batch["frame_ids"],
        batch["hardms"],
        target_index=None,
    )


def prepare_input(dataset, jitter, index):
    """
    Disk/PIL/CPU preprocessing is done BEFORE the timer starts.

    The returned UAV tensor is exactly the object considered to be the
    "model input" for the main benchmark.
    """
    item = dataset[index]

    prior_xy = (
        item["xy"].float()
        + jitter[index]
    ).unsqueeze(0)

    frame_id = int(str(item["frame_id"]))

    return {
        "uav": item["uav"].unsqueeze(0),
        "prior_xy": prior_xy,
        "frame_id": frame_id,
    }


@torch.inference_mode()
def visual_frame_forward(
    visual,
    prepared,
    grid_size,
):
    uav_backbone_feature = visual.encode_uav_clip(
        prepared["uav"]
    )

    candidate = visual.candidate_batch(
        uav_backbone_feature,
        prepared["prior_xy"].to(
            visual.device,
            non_blocking=True,
        ),
        grid_size=grid_size,
    )

    return candidate


@torch.inference_mode()
def process_prepared_frame(
    visual,
    temporal_model,
    history,
    prepared,
    window,
    grid_size,
):
    candidate = visual_frame_forward(
        visual,
        prepared,
        grid_size,
    )

    history.append(
        candidate_record(
            candidate,
            prepared["frame_id"],
        )
    )

    if len(history) < window:
        return None

    temporal_batch = make_temporal_batch(history)

    output = temporal_forward(
        temporal_model,
        temporal_batch,
    )

    final_xy = (
        output.final_xy[0]
        .detach()
        .cpu()
        .double()
    )

    latitude, longitude = local_xy_to_latlon(
        float(final_xy[0].item()),
        float(final_xy[1].item()),
        visual.origin_lat,
        visual.origin_lon,
    )

    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "x_meter": float(final_xy[0].item()),
        "y_meter": float(final_xy[1].item()),
    }


def benchmark_first_gps(
    visual,
    temporal_model,
    dataset,
    jitter,
    device,
    window,
    grid_size,
    start_index,
    repeats,
):
    """
    Measure:
        window prepared UAV tensors -> FIRST final GPS.

    Camera waiting and disk/PIL preprocessing are excluded.
    """
    latencies = []
    example_gps = None

    maximum_start = len(dataset) - window

    if maximum_start < 0:
        raise RuntimeError(
            "Route is shorter than temporal window"
        )

    for repeat in range(repeats):
        base = start_index + repeat * window

        if base > maximum_start:
            base = repeat % max(maximum_start + 1, 1)

        prepared_window = [
            prepare_input(
                dataset,
                jitter,
                index,
            )
            for index in range(
                base,
                base + window,
            )
        ]

        history = deque(maxlen=window)

        sync(device)
        start = time.perf_counter()

        gps = None

        for prepared in prepared_window:
            gps = process_prepared_frame(
                visual,
                temporal_model,
                history,
                prepared,
                window,
                grid_size,
            )

        sync(device)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if gps is None:
            raise RuntimeError(
                "No GPS output was produced from a complete window"
            )

        latencies.append(elapsed_ms)
        example_gps = gps

    return latency_summary_ms(latencies), example_gps


def benchmark_steady_state(
    visual,
    temporal_model,
    dataset,
    jitter,
    device,
    window,
    grid_size,
    warmup_outputs,
    samples,
):
    """
    Sliding-window benchmark:
        ONE new prepared UAV tensor -> ONE final GPS.

    Previous window-1 temporal states are already available.
    """
    history = deque(maxlen=window)

    total_needed = (
        (window - 1)
        + warmup_outputs
        + samples
    )

    if total_needed > len(dataset):
        samples = (
            len(dataset)
            - (window - 1)
            - warmup_outputs
        )

    if samples <= 0:
        raise RuntimeError(
            "Not enough route frames for requested warmup/samples"
        )

    timed_latencies = []
    visual_latencies = []
    temporal_gps_latencies = []
    output_index = 0
    last_gps = None

    frame_count = (
        (window - 1)
        + warmup_outputs
        + samples
    )

    for index in range(frame_count):
        # Model input preparation is outside the timed region.
        prepared = prepare_input(
            dataset,
            jitter,
            index,
        )

        # Initial history-fill frames do not yet produce a GPS output.
        if len(history) < window - 1:
            candidate = visual_frame_forward(
                visual,
                prepared,
                grid_size,
            )

            history.append(
                candidate_record(
                    candidate,
                    prepared["frame_id"],
                )
            )
            continue

        is_timed = output_index >= warmup_outputs

        if is_timed:
            sync(device)
            total_start = time.perf_counter()
            visual_start = total_start

        candidate = visual_frame_forward(
            visual,
            prepared,
            grid_size,
        )

        history.append(
            candidate_record(
                candidate,
                prepared["frame_id"],
            )
        )

        if is_timed:
            sync(device)

            visual_ms = (
                time.perf_counter()
                - visual_start
            ) * 1000.0

            temporal_start = time.perf_counter()

        temporal_batch = make_temporal_batch(history)

        output = temporal_forward(
            temporal_model,
            temporal_batch,
        )

        final_xy = (
            output.final_xy[0]
            .detach()
            .cpu()
            .double()
        )

        latitude, longitude = local_xy_to_latlon(
            float(final_xy[0].item()),
            float(final_xy[1].item()),
            visual.origin_lat,
            visual.origin_lon,
        )

        last_gps = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "x_meter": float(final_xy[0].item()),
            "y_meter": float(final_xy[1].item()),
        }

        if is_timed:
            sync(device)

            temporal_gps_ms = (
                time.perf_counter()
                - temporal_start
            ) * 1000.0

            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000.0

            timed_latencies.append(total_ms)
            visual_latencies.append(visual_ms)
            temporal_gps_latencies.append(
                temporal_gps_ms
            )

            if len(timed_latencies) >= samples:
                break

        output_index += 1

    return {
        "model_input_to_gps": latency_summary_ms(
            timed_latencies
        ),
        "visual_stage": latency_summary_ms(
            visual_latencies
        ),
        "temporal_plus_gps": latency_summary_ms(
            temporal_gps_latencies
        ),
        "example_final_gps": last_gps,
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this latency benchmark"
        )

    config, visual_model = configure_modules(
        args.backbone,
        args.window,
    )

    # Import these only AFTER config/model routing above.
    data_module = importlib.import_module("data")
    visual_localizer_module = importlib.import_module(
        "visual_localizer"
    )

    RouteDataset = data_module.RouteDataset
    FrozenVisualLocalizer = (
        visual_localizer_module.FrozenVisualLocalizer
    )
    TemporalLatticeCRF = (
        visual_model.TemporalLatticeCRF
    )

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    route_index, route_root = route_info(
        config,
        args.route,
    )

    print("=" * 96)
    print("FULL UAV MODEL INPUT -> FINAL GPS LATENCY BENCHMARK")
    print("=" * 96)
    print("backbone            : {}".format(args.backbone))
    print(
        "GPU                 : {}".format(
            torch.cuda.get_device_name(device)
        )
    )
    print("temporal window     : {}".format(args.window))
    print("route               : {}".format(args.route))
    print(
        "visual checkpoint   : {}".format(
            config.VISUAL_CHECKPOINT
        )
    )
    print(
        "temporal checkpoint : {}".format(
            config.TEMPORAL_CHECKPOINT
        )
    )
    print(
        "candidate count     : {}".format(
            int(config.GRID_SIZE) ** 2
        )
    )
    print("batch size          : 1")
    print("-" * 96)
    print(
        "TIMED: prepared UAV tensor -> backbone -> retrieval -> HardMS -> "
        "T2-only RTL-CRF -> final x/y -> GPS lat/lon"
    )
    print(
        "NOT TIMED: model loading, satellite gallery loading/precompute, "
        "disk/PIL read, dataset CPU transform, camera waiting"
    )
    print("=" * 96)

    # Loading is intentionally outside the timed region.
    visual = FrozenVisualLocalizer(device)

    temporal_model = load_temporal_model(
        config,
        TemporalLatticeCRF,
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
        route_index,
        args.jitter_m,
        int(config.SEED),
    )

    # GPU/kernel warm-up using real route inputs.
    warm_history = deque(maxlen=args.window)

    warm_frames = min(
        len(dataset),
        args.window - 1 + max(args.warmup, 10),
    )

    for index in range(warm_frames):
        prepared = prepare_input(
            dataset,
            jitter,
            index,
        )

        process_prepared_frame(
            visual,
            temporal_model,
            warm_history,
            prepared,
            args.window,
            int(config.GRID_SIZE),
        )

    sync(device)

    # Measure compute required for the very first GPS from a complete window.
    first_start = min(
        warm_frames,
        max(
            0,
            len(dataset)
            - args.window
            - (
                args.first_output_repeats
                * args.window
            )
            - 1,
        ),
    )

    first_gps, first_example = benchmark_first_gps(
        visual,
        temporal_model,
        dataset,
        jitter,
        device,
        args.window,
        int(config.GRID_SIZE),
        first_start,
        args.first_output_repeats,
    )

    # Measure normal sliding-window operation.
    steady = benchmark_steady_state(
        visual,
        temporal_model,
        dataset,
        jitter,
        device,
        args.window,
        int(config.GRID_SIZE),
        args.warmup,
        args.samples,
    )

    result = {
        "benchmark_definition": (
            "prepared UAV tensor entering full localization model "
            "to final GPS latitude/longitude"
        ),
        "backbone": args.backbone,
        "backbone_name": str(
            getattr(
                config,
                "BACKBONE_NAME",
                args.backbone,
            )
        ),
        "architecture": (
            "T2-only ResidualSecondOrderTemporalLatticeCRF"
        ),
        "temporal_window": int(args.window),
        "route": args.route,
        "device": torch.cuda.get_device_name(device),
        "candidate_count": int(config.GRID_SIZE) ** 2,
        "batch_size": 1,
        "jitter_m": float(args.jitter_m),
        "checkpoints": {
            "visual": str(config.VISUAL_CHECKPOINT),
            "temporal": str(
                config.TEMPORAL_CHECKPOINT
            ),
        },
        "first_gps_output": {
            "definition": (
                "{} prepared UAV frames -> first final GPS"
            ).format(args.window),
            "camera_waiting_time_included": False,
            "latency": first_gps,
            "example_final_gps": first_example,
        },
        "steady_state": {
            "definition": (
                "one new prepared UAV frame -> one new final GPS; "
                "previous temporal history already exists"
            ),
            **steady,
        },
        "included_operations": [
            "CPU UAV tensor -> GPU transfer",
            "visual backbone",
            "UAV retrieval head",
            "local 36-candidate selection",
            "SAT retrieval head from precomputed backbone gallery",
            "cosine retrieval scores",
            "retrieval probability",
            "Fixed Hard Mean-Shift",
            "temporal-window tensor assembly",
            "T2-only RTL-CRF",
            "correction gate",
            "final local x/y",
            "local x/y -> GPS latitude/longitude",
        ],
        "excluded_operations": [
            "checkpoint/model loading",
            "satellite gallery loading",
            "satellite backbone gallery precomputation",
            "disk image read",
            "PIL decode",
            "RouteDataset CPU image transform",
            "camera waiting time",
            "controlled prior construction",
        ],
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("outputs")
            / "full_pipeline_latency"
            / (
                "{}_w{}_{}.json".format(
                    args.backbone,
                    args.window,
                    args.route,
                )
            )
        )

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

    steady_main = steady["model_input_to_gps"]

    print()
    print("=" * 96)
    print("RESULT: COMPLETE MODEL INPUT -> FINAL GPS")
    print("=" * 96)

    print(
        "FIRST GPS from {} prepared UAV frames".format(
            args.window
        )
    )
    print(
        "  mean   : {:.3f} ms".format(
            first_gps["mean_ms"]
        )
    )
    print(
        "  median : {:.3f} ms".format(
            first_gps["median_ms"]
        )
    )
    print(
        "  P95    : {:.3f} ms".format(
            first_gps["p95_ms"]
        )
    )

    print()
    print(
        "STEADY STATE: one new UAV image -> one final GPS"
    )
    print(
        "  mean latency   : {:.3f} ms".format(
            steady_main["mean_ms"]
        )
    )
    print(
        "  median latency : {:.3f} ms".format(
            steady_main["median_ms"]
        )
    )
    print(
        "  P90 latency    : {:.3f} ms".format(
            steady_main["p90_ms"]
        )
    )
    print(
        "  P95 latency    : {:.3f} ms".format(
            steady_main["p95_ms"]
        )
    )
    print(
        "  P99 latency    : {:.3f} ms".format(
            steady_main["p99_ms"]
        )
    )
    print(
        "  GPS output FPS : {:.2f} outputs/s".format(
            steady_main[
                "gps_outputs_per_second"
            ]
        )
    )

    print()
    print("Diagnostic breakdown only:")
    print(
        "  visual stage mean   : {:.3f} ms".format(
            steady["visual_stage"]["mean_ms"]
        )
    )
    print(
        "  temporal + GPS mean : {:.3f} ms".format(
            steady["temporal_plus_gps"][
                "mean_ms"
            ]
        )
    )

    if steady["example_final_gps"] is not None:
        gps = steady["example_final_gps"]

        print()
        print(
            "Example final GPS: "
            "lat={:.9f}, lon={:.9f}".format(
                gps["latitude"],
                gps["longitude"],
            )
        )

    print()
    print("Saved JSON: {}".format(output_path))
    print("=" * 96)


if __name__ == "__main__":
    main()
