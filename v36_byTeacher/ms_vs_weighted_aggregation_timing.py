"""Measure only the final XY aggregation cost for Weighted Centroid vs SoftMS.

This benchmark intentionally EXCLUDES:
- UAV/SAT feature extraction
- candidate retrieval
- similarity scoring / softmax construction
- MeanShift iterations / convergence
- Kalman / GRU
- full-frame pipeline latency or FPS

The comparison starts only after all required points and weights already exist:
1) Weighted Centroid: N original patch centers + N posterior weights -> final XY.
2) SoftMS final aggregation: K already-converged/merged modes + K mode weights -> final XY.

Thus a 6x6 experiment compares aggregation over N=36 patch points against
aggregation over the data-dependent K active MeanShift modes (often only a few).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, soft_mean_shift


def _xy_tensor(xy, device):
    return torch.as_tensor(xy, dtype=torch.float32, device=device).reshape(1, 2)


def _build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


def _sample_indices(length, requested):
    if length <= 0:
        return []
    if requested <= 0 or requested >= length:
        return list(range(length))
    return np.linspace(0, length - 1, num=requested, dtype=np.int64).tolist()


def _cuda_benchmark(items, fn, repeats, warmup, device):
    for _ in range(int(warmup)):
        for item in items:
            fn(item)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(repeats)):
        for item in items:
            fn(item)
    end.record()
    torch.cuda.synchronize(device)
    total_ms = float(start.elapsed_time(end))
    calls = int(repeats) * len(items)
    return total_ms / max(calls, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--frames-per-route", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--suite-tag", required=True)
    args = parser.parse_args()

    device = rt.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("aggregation timing benchmark requires CUDA events")

    rt.set_seed(int(config.SEED))
    visual = FrozenVisualLocalizer(device)

    weighted_items = []
    ms_items = []
    frame_rows = []
    n_expected = int(args.grid_size) ** 2

    for route_name in ("route_B", "route_C"):
        cache = _build_cache(route_name, visual, device)
        indices = _sample_indices(len(cache), int(args.frames_per_route))
        for index in indices:
            uav_clip = cache.uav_clip[index:index + 1].to(device).float()
            center_xy = cache.gt_xy[index].detach().cpu().numpy().astype(np.float64)
            batch = visual.candidate_batch(
                uav_clip=uav_clip,
                center_xy=_xy_tensor(center_xy, device),
                grid_size=int(args.grid_size),
            )
            if int(batch.centers.shape[1]) != n_expected:
                raise RuntimeError("candidate count mismatch")

            # All scoring/softmax is finished before timing.
            patch_weights = torch.softmax(
                batch.raw_logits / max(float(config.MEANSHIFT_SCORE_TAU), 1e-6),
                dim=1,
            ).detach()
            patch_centers = batch.centers.detach()

            # MeanShift convergence + basin merge is also finished before timing.
            _, _, modes, _, mode_weights, _ = soft_mean_shift(
                batch.raw_logits,
                batch.centers,
                config.MEANSHIFT_SCORE_TAU,
                config.MEANSHIFT_BANDWIDTH_M,
                config.MEANSHIFT_ITERATIONS,
                config.MEANSHIFT_MODE_BETA,
            )
            active = mode_weights[0] > 0
            active_modes = modes[0, active].detach()
            active_weights = mode_weights[0, active].detach()
            active_weights = active_weights / active_weights.sum().clamp_min(1e-12)

            weighted_items.append((patch_weights, patch_centers))
            ms_items.append((active_weights, active_modes))
            frame_rows.append(
                {
                    "route": route_name,
                    "frame_index": int(index),
                    "patch_count_N": int(n_expected),
                    "active_ms_modes_K": int(active_modes.shape[0]),
                }
            )

    def weighted_aggregate(item):
        weights, centers = item
        return (weights[:, :, None] * centers).sum(dim=1)

    def ms_aggregate(item):
        weights, modes = item
        return (weights[:, None] * modes).sum(dim=0)

    weighted_ms = _cuda_benchmark(
        weighted_items, weighted_aggregate, args.repeats, args.warmup, device
    )
    softms_ms = _cuda_benchmark(
        ms_items, ms_aggregate, args.repeats, args.warmup, device
    )

    k_values = np.asarray([r["active_ms_modes_K"] for r in frame_rows], dtype=np.float64)
    result = {
        "protocol": "final-coordinate aggregation only",
        "excluded_from_timing": [
            "feature extraction",
            "candidate retrieval",
            "similarity scoring",
            "softmax construction",
            "MeanShift iterations/convergence",
            "Kalman",
            "GRU",
            "full-pipeline latency/FPS",
        ],
        "grid_size": int(args.grid_size),
        "patch_count_N": int(n_expected),
        "sampled_frames": int(len(frame_rows)),
        "routes": ["route_B", "route_C"],
        "mean_active_ms_modes_K": float(k_values.mean()),
        "median_active_ms_modes_K": float(np.median(k_values)),
        "min_active_ms_modes_K": int(k_values.min()),
        "max_active_ms_modes_K": int(k_values.max()),
        "weighted_centroid_aggregation_ms": float(weighted_ms),
        "softms_final_mode_aggregation_ms": float(softms_ms),
        "aggregation_speed_ratio_weighted_over_softms": float(weighted_ms / max(softms_ms, 1e-12)),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "timing_device": str(device),
    }

    root = (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "mkg_final_ablation_suite"
        / args.suite_tag
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "decoder_aggregation_timing.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "decoder_aggregation_mode_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
