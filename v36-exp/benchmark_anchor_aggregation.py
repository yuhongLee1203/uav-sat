#!/usr/bin/env python3
"""Benchmark only final coordinate aggregation; Mean-Shift itself is excluded."""

import json
import math
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v36-exp/outputs/aggregation_benchmark.json"


def elapsed_ms(fn, warmup=500, repeats=10000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the aggregation timing table")
    torch.manual_seed(2033)
    device = torch.device("cuda")
    # Tracker inference is causal and processes one frame at a time.  Using a
    # huge synthetic batch and dividing its runtime would report throughput,
    # not the requested per-frame latency.
    batch = 1
    patches = 18
    # This benchmark starts after matching / Mean-Shift. Weighted Centroid has
    # all 18 patch coordinates. SoftMS has a data-dependent number of
    # consolidated modes, so time every possible K=1..18. The collector uses
    # the measured mean mode count from the actual route CSV.
    centers = torch.randn(batch, patches, 2, device=device)
    weights = torch.softmax(torch.randn(batch, patches, device=device), dim=1)
    weighted_batch_ms = elapsed_ms(lambda: (weights.unsqueeze(-1) * centers).sum(dim=1))
    softms_ms_by_mode = {}
    for mode_count in range(1, patches + 1):
        modes = torch.randn(batch, mode_count, 2, device=device)
        mode_weights = torch.softmax(torch.randn(batch, mode_count, device=device), dim=1)
        batch_ms = elapsed_ms(
            lambda m=modes, w=mode_weights: (w.unsqueeze(-1) * m).sum(dim=1)
        )
        softms_ms_by_mode[str(mode_count)] = batch_ms / batch
    result = {
        "scope": "final coordinate aggregation only; image/backbone/matching/Mean-Shift/GRU/Kalman excluded",
        "device": torch.cuda.get_device_name(0),
        "batch": batch,
        "weighted_centroid_patch_count": patches,
        "weighted_centroid_ms_per_frame": weighted_batch_ms / batch,
        "softms_ms_per_frame_by_mode_count": softms_ms_by_mode,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
