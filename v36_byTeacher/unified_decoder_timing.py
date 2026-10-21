"""Weighted-vs-SoftMS final coordinate aggregation timing.

Timing starts only after all visual scores are available and, for SoftMS, after
MeanShift has already converged and basins have already been consolidated.

Measured:
  Weighted: 36 existing patch coordinates + existing weights -> final XY
  SoftMS:   K existing converged mode coordinates + existing mode weights -> final XY

Not measured:
  backbone, candidate retrieval, similarity, softmax, MeanShift iterations,
  basin consolidation, Kalman, GRU, or full-pipeline FPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker_base as rb
import unified_protocol as up
from visual_localizer import FrozenVisualLocalizer, soft_mean_shift

ROOT = Path(config.BACKBONE_OUTPUT_DIR) / "unified_fixed8m_v1" / "decoder_timing"


@torch.no_grad()
def collect_samples(visual, device, max_frames):
    weighted_inputs = []
    mode_inputs = []
    mode_rows = []

    per_route = max(1, int(max_frames) // 2)
    for route_name in ("route_B", "route_C"):
        cache = up.build_cache(route_name, visual, device)
        capture = up.capture_report(visual, cache, route_name)
        up.assert_capture(capture)
        count = min(per_route, len(cache))
        indices = np.linspace(0, len(cache) - 1, num=count, dtype=int)

        for index in indices.tolist():
            ref = cache.gt_xy[index].cpu().numpy().astype(np.float64)
            center = up.search_center(ref, index, route_name)
            uav = cache.uav_clip[index:index + 1].to(device).float()
            raw = up.raw_candidates(
                visual, uav, center, grid_size=up.MAIN_GRID_SIZE
            )
            probability = torch.softmax(
                raw["logits"] / float(up.MAIN_TAU), dim=1
            )
            _, _, modes, _, mode_weights, _ = soft_mean_shift(
                raw["logits"],
                raw["centers"],
                float(up.MAIN_TAU),
                float(up.MAIN_BANDWIDTH_M),
                int(config.MEANSHIFT_ITERATIONS),
                float(config.MEANSHIFT_MODE_BETA),
            )
            active = mode_weights[0] > 0
            active_modes = modes[0, active].contiguous()
            active_weights = mode_weights[0, active].contiguous()

            weighted_inputs.append(
                (
                    raw["centers"][0].contiguous(),
                    probability[0].contiguous(),
                )
            )
            mode_inputs.append((active_modes, active_weights))
            mode_rows.append(
                {
                    "route": route_name,
                    "frame_index": int(index),
                    "patch_count_N": int(raw["candidate_count"]),
                    "active_ms_modes_K": int(active.sum().item()),
                }
            )

    return weighted_inputs, mode_inputs, mode_rows


def cuda_time_per_frame_ms(fn, repeats, frames, device, warmup):
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(repeats)):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    total_ms = float(start.elapsed_time(end))
    return total_ms / float(int(repeats) * int(frames))


def cpu_time_per_frame_ms(fn, repeats, frames, warmup):
    for _ in range(int(warmup)):
        fn()
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / float(int(repeats) * int(frames))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    device = rb.resolve_device(args.device)
    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)
    visual = FrozenVisualLocalizer(device)

    weighted_inputs, mode_inputs, mode_rows = collect_samples(
        visual, device, args.frames
    )
    frames = len(weighted_inputs)
    if frames == 0:
        raise RuntimeError("no timing samples")

    def weighted_once():
        outputs = []
        for centers, weights in weighted_inputs:
            outputs.append((weights[:, None] * centers).sum(dim=0))
        return outputs

    def softms_once():
        outputs = []
        for modes, weights in mode_inputs:
            outputs.append((weights[:, None] * modes).sum(dim=0))
        return outputs

    if device.type == "cuda":
        weighted_ms = cuda_time_per_frame_ms(
            weighted_once, args.repeats, frames, device, args.warmup
        )
        softms_ms = cuda_time_per_frame_ms(
            softms_once, args.repeats, frames, device, args.warmup
        )
    else:
        weighted_ms = cpu_time_per_frame_ms(
            weighted_once, args.repeats, frames, args.warmup
        )
        softms_ms = cpu_time_per_frame_ms(
            softms_once, args.repeats, frames, args.warmup
        )

    mode_counts = np.asarray(
        [row["active_ms_modes_K"] for row in mode_rows], dtype=np.float64
    )
    payload = {
        "protocol": "final-coordinate aggregation only",
        "formal_search_protocol": up.protocol_metadata(),
        "excluded_from_timing": [
            "feature extraction",
            "candidate retrieval",
            "similarity scoring",
            "softmax construction",
            "MeanShift iterations/convergence",
            "basin consolidation",
            "Kalman",
            "GRU",
            "full-pipeline latency/FPS",
        ],
        "grid_size": int(up.MAIN_GRID_SIZE),
        "patch_count_N": int(up.MAIN_GRID_SIZE) ** 2,
        "sampled_frames": int(frames),
        "routes": ["route_B", "route_C"],
        "mean_active_ms_modes_K": float(mode_counts.mean()),
        "median_active_ms_modes_K": float(np.median(mode_counts)),
        "min_active_ms_modes_K": int(mode_counts.min()),
        "max_active_ms_modes_K": int(mode_counts.max()),
        "weighted_centroid_aggregation_ms": float(weighted_ms),
        "softms_final_mode_aggregation_ms": float(softms_ms),
        "aggregation_speed_ratio_weighted_over_softms": (
            float(weighted_ms / softms_ms) if softms_ms > 0 else None
        ),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "timing_device": str(device),
        "fps_reported": False,
    }

    ROOT.mkdir(parents=True, exist_ok=True)
    up.write_json(ROOT / "decoder_aggregation_timing.json", payload)
    with (ROOT / "mode_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mode_rows)

    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
