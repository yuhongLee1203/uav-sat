#!/usr/bin/env python3
"""Benchmark current T2-only RTL-CRF latency for one final GPS output."""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import config
from data import RouteDataset
from visual_localizer import FrozenVisualLocalizer
from visual_model import TemporalLatticeCRF

EARTH_RADIUS_M = 6378137.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure T2-only RTL-CRF inference time per final GPS coordinate."
    )
    p.add_argument("--route", choices=tuple(config.ROUTE_NAMES), default="route_B")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument(
        "--jitter-m",
        type=float,
        default=float(getattr(config, "LOCAL_PRIOR_JITTER_M", 12.0)),
    )
    p.add_argument("--temporal-only-repeats", type=int, default=200)
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summary_ms(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "count": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "std_ms": float(a.std()),
        "min_ms": float(a.min()),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "mean_fps": float(1000.0 / max(a.mean(), 1e-12)),
    }


def deterministic_jitter(length, route_index, maximum_m):
    if maximum_m <= 0:
        return torch.zeros(length, 2, dtype=torch.float32)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(config.SEED) + 1009 * int(route_index))
    radius = torch.sqrt(torch.rand(length, 1, generator=g)) * float(maximum_m)
    angle = torch.rand(length, 1, generator=g) * (2.0 * math.pi)
    return torch.cat([radius * angle.cos(), radius * angle.sin()], dim=1).float()


def local_xy_to_latlon(x_meter, y_meter, origin_lat, origin_lon):
    """Inverse of data.meters_from_latlon(), using the same local approximation."""
    origin_lat_rad = math.radians(float(origin_lat))
    lat = float(origin_lat) + math.degrees(float(y_meter) / EARTH_RADIUS_M)
    lon = float(origin_lon) + math.degrees(
        float(x_meter)
        / (EARTH_RADIUS_M * max(abs(math.cos(origin_lat_rad)), 1e-12))
    )
    return lat, lon


def route_info(name):
    catalog = {
        route_name: (idx, Path(root))
        for idx, (route_name, root) in enumerate(zip(config.ROUTE_NAMES, config.ROUTE_ROOTS))
    }
    if name not in catalog:
        raise ValueError("unknown route: {}".format(name))
    return catalog[name]


def load_temporal_model(device):
    path = Path(config.TEMPORAL_CHECKPOINT)
    if not path.exists():
        raise FileNotFoundError(
            "T2-only temporal checkpoint not found: {}".format(path)
        )
    model = TemporalLatticeCRF().to(device)
    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("architecture")
    if arch is not None and arch != "ResidualSecondOrderTemporalLatticeCRF":
        raise RuntimeError("unexpected architecture: {}".format(arch))
    state = ckpt.get("best_model")
    if state is None:
        state = ckpt.get("model")
    if state is None:
        raise RuntimeError("checkpoint has neither best_model nor model")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
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
        "frame_id": torch.tensor(int(frame_id), dtype=torch.long, device=device),
    }


def make_temporal_batch(history):
    rows = list(history)
    return {
        "z_uav": torch.stack([r["z_uav"] for r in rows], 0).unsqueeze(0),
        "z_sat": torch.stack([r["z_sat"] for r in rows], 0).unsqueeze(0),
        "raw_logits": torch.stack([r["raw_logits"] for r in rows], 0).unsqueeze(0),
        "raw_prob": torch.stack([r["raw_prob"] for r in rows], 0).unsqueeze(0),
        "centers": torch.stack([r["centers"] for r in rows], 0).unsqueeze(0),
        "frame_ids": torch.stack([r["frame_id"] for r in rows], 0).unsqueeze(0),
        "hardms": torch.stack([r["hardms"] for r in rows], 0).unsqueeze(0),
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


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU latency benchmark")

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    window = int(config.TEMPORAL_WINDOW)
    if window < 3:
        raise RuntimeError("T2-only RTL-CRF requires at least 3 frames")

    route_index, route_root = route_info(args.route)

    print("=" * 88)
    print("T2-ONLY RTL-CRF INFERENCE LATENCY BENCHMARK")
    print("=" * 88)
    print("GPU                    : {}".format(torch.cuda.get_device_name(device)))
    print("temporal window        : {} frames".format(window))
    print("route                  : {}".format(args.route))
    print("visual checkpoint      : {}".format(config.VISUAL_CHECKPOINT))
    print("temporal checkpoint    : {}".format(config.TEMPORAL_CHECKPOINT))
    print("candidate count        : {}".format(int(config.GRID_SIZE) ** 2))
    print("batch size             : 1")
    print("warm-up outputs        : {}".format(args.warmup))
    print("timed GPS outputs      : {}".format(args.samples))
    print("- SAT backbone gallery is already cached, matching current inference code")
    print("- checkpoint/model loading is excluded")
    print("- GPS conversion from final (x,y) is included")
    print("- camera waiting time for collecting the temporal window is excluded")
    print("=" * 88)

    # Loading is intentionally outside the timed region.
    visual = FrozenVisualLocalizer(device)
    model = load_temporal_model(device)
    dataset = RouteDataset(
        route_root,
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )
    jitter = deterministic_jitter(len(dataset), route_index, args.jitter_m)

    max_outputs = len(dataset) - window + 1
    if max_outputs <= args.warmup:
        raise RuntimeError("route is too short for requested warm-up")
    if args.warmup + args.samples > max_outputs:
        args.samples = max_outputs - args.warmup
        print("adjusted timed samples to {}".format(args.samples))

    required_frames = (window - 1) + args.warmup + args.samples
    history = deque(maxlen=window)

    steady_model_to_gps = []
    preprocess_only = []
    full_from_disk = []
    last_latlon = None
    last_temporal_batch = None

    first_visual_compute = 0.0
    first_preprocess = 0.0
    first_temporal_gps = None
    output_count = 0

    for index in range(required_frames):
        # Report disk/PIL/transform separately; this is not normally counted as model inference.
        t_pre = time.perf_counter()
        item = dataset[index]
        pre_ms = (time.perf_counter() - t_pre) * 1000.0

        prior = (item["xy"].float() + jitter[index]).unsqueeze(0)
        frame_id = int(str(item["frame_id"]))

        # Current-frame inference: preprocessed UAV -> candidate scores/HardMS.
        sync(device)
        t_visual = time.perf_counter()
        uav_clip = visual.encode_uav_clip(item["uav"].unsqueeze(0))
        candidate = visual.candidate_batch(
            uav_clip,
            prior.to(device),
            grid_size=config.GRID_SIZE,
        )
        sync(device)
        visual_ms = (time.perf_counter() - t_visual) * 1000.0

        history.append(candidate_record(candidate, frame_id))

        if index < window:
            first_visual_compute += visual_ms
            first_preprocess += pre_ms

        if len(history) < window:
            continue

        batch = make_temporal_batch(history)
        last_temporal_batch = batch

        sync(device)
        t_temporal = time.perf_counter()
        output = temporal_forward(model, batch)
        sync(device)

        xy = output.final_xy[0].detach().cpu().double()
        lat, lon = local_xy_to_latlon(
            float(xy[0].item()),
            float(xy[1].item()),
            visual.origin_lat,
            visual.origin_lon,
        )
        temporal_gps_ms = (time.perf_counter() - t_temporal) * 1000.0
        last_latlon = (lat, lon)

        if output_count == 0:
            first_temporal_gps = temporal_gps_ms

        if output_count >= args.warmup and len(steady_model_to_gps) < args.samples:
            model_ms = visual_ms + temporal_gps_ms
            steady_model_to_gps.append(model_ms)
            preprocess_only.append(pre_ms)
            full_from_disk.append(pre_ms + model_ms)

        output_count += 1
        if len(steady_model_to_gps) >= args.samples:
            break

    if first_temporal_gps is None:
        raise RuntimeError("no complete temporal window was produced")

    first_compute_ms = first_visual_compute + first_temporal_gps
    first_disk_ms = first_preprocess + first_compute_ms

    # Temporal-model-only profile on one real window.
    temporal_only = []
    if last_temporal_batch is not None and args.temporal_only_repeats > 0:
        for _ in range(20):
            temporal_forward(model, last_temporal_batch)
        sync(device)
        for _ in range(args.temporal_only_repeats):
            sync(device)
            t0 = time.perf_counter()
            temporal_forward(model, last_temporal_batch)
            sync(device)
            temporal_only.append((time.perf_counter() - t0) * 1000.0)

    steady = summary_ms(steady_model_to_gps)
    pre = summary_ms(preprocess_only)
    full = summary_ms(full_from_disk)
    temporal = summary_ms(temporal_only)

    result = {
        "architecture": "T2-only ResidualSecondOrderTemporalLatticeCRF",
        "temporal_window": window,
        "route": args.route,
        "device": torch.cuda.get_device_name(device),
        "candidate_count": int(config.GRID_SIZE) ** 2,
        "batch_size": 1,
        "jitter_m": float(args.jitter_m),
        "first_gps_output_compute": {
            "frames_required": window,
            "model_compute_ms": float(first_compute_ms),
            "including_image_read_preprocess_ms": float(first_disk_ms),
            "camera_waiting_time_included": False,
        },
        "steady_state_one_new_gps": {
            "preprocessed_uav_to_gps": steady,
            "image_read_preprocess_only": pre,
            "disk_image_to_gps": full,
        },
        "t2_only_rtl_crf_forward_only": temporal,
        "example_final_gps": {
            "latitude": float(last_latlon[0]),
            "longitude": float(last_latlon[1]),
        } if last_latlon is not None else None,
        "notes": {
            "sat_gallery_precomputed": True,
            "model_loading_included": False,
            "local_prior_generation_included": False,
            "gps_conversion_included": True,
        },
    }

    print()
    print("=" * 88)
    print("RESULT")
    print("=" * 88)
    print(
        "First GPS compute (need {} frames, no camera waiting): {:.3f} ms".format(
            window, first_compute_ms
        )
    )
    print("First GPS incl. image read/preprocess              : {:.3f} ms".format(first_disk_ms))
    print()
    print("STEADY STATE: one new UAV frame -> one final GPS")
    print("  mean   : {:.3f} ms  ({:.2f} GPS outputs/s)".format(steady["mean_ms"], steady["mean_fps"]))
    print("  median : {:.3f} ms".format(steady["median_ms"]))
    print("  P90    : {:.3f} ms".format(steady["p90_ms"]))
    print("  P95    : {:.3f} ms".format(steady["p95_ms"]))
    print("  P99    : {:.3f} ms".format(steady["p99_ms"]))
    print("  min/max: {:.3f} / {:.3f} ms".format(steady["min_ms"], steady["max_ms"]))

    if temporal:
        print()
        print("T2-only RTL-CRF forward ONLY (visual stage excluded)")
        print("  mean   : {:.3f} ms".format(temporal["mean_ms"]))
        print("  median : {:.3f} ms".format(temporal["median_ms"]))
        print("  P95    : {:.3f} ms".format(temporal["p95_ms"]))

    if pre:
        print()
        print("Image read + preprocess ONLY")
        print("  mean   : {:.3f} ms".format(pre["mean_ms"]))

    if last_latlon is not None:
        print()
        print("Example final GPS: lat={:.9f}, lon={:.9f}".format(last_latlon[0], last_latlon[1]))

    output_path = Path(args.output) if args.output else Path(config.OUTPUT_DIR) / "inference_time_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved JSON: {}".format(output_path))
    print("=" * 88)


if __name__ == "__main__":
    main()
