"""Low-CPU GPU runner for unified experiments.

It applies two runtime fixes before importing the target experiment:
1) GPU-native nearest-grid lookup (avoids full gallery copies to CPU).
2) Stage-1 visual training jitter is changed from random radius [0,R] to the
   same deterministic fixed-radius R used by the unified formal protocol.
"""

import math
import os
import runpy
import sys

import numpy as np
import torch

CPU_THREADS = max(1, int(os.environ.get("UAVSAT_CPU_THREADS", "2")))
MAIN_SEED = int(os.environ.get("UAVSAT_MAIN_SEED", "2026"))

torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import visual_localizer as vl


def regular_grid_indices_gpu(
    gallery_xy,
    gallery_pixel,
    pixel_index,
    prior_xy,
    grid_size,
    stride,
    device,
):
    grid_size = int(grid_size)
    stride = int(stride)
    start = -(grid_size // 2)
    offsets = range(start, start + grid_size)
    rows = []

    prior_xy = prior_xy.to(gallery_xy.device, dtype=gallery_xy.dtype)
    for prior in prior_xy:
        distance_squared = (
            (gallery_xy[:, 0] - prior[0]).square()
            + (gallery_xy[:, 1] - prior[1]).square()
        )
        center_index = int(distance_squared.argmin().item())
        center_pixel = gallery_pixel[center_index]
        center_x = int(round(float(center_pixel[0].item())))
        center_y = int(round(float(center_pixel[1].item())))

        row = []
        complete = True
        for offset_y in offsets:
            for offset_x in offsets:
                index = pixel_index.get(
                    (center_x + offset_x * stride, center_y + offset_y * stride)
                )
                if index is None:
                    complete = False
                    break
                row.append(int(index))
            if not complete:
                break

        if complete:
            row_tensor = torch.tensor(
                row, dtype=torch.long, device=gallery_xy.device
            )
        else:
            row_tensor = torch.topk(
                distance_squared,
                k=grid_size * grid_size,
                largest=False,
            ).indices.to(dtype=torch.long)
        rows.append(row_tensor)

    return torch.stack(rows, dim=0).to(device=device, dtype=torch.long)


def deterministic_fixed_radius_jitter(length, route_index, maximum_m):
    """Stage-1 jitter with exactly R metres, not random radius <=R."""
    length = int(length)
    magnitude = float(maximum_m)
    if magnitude <= 0.0:
        return torch.zeros(length, 2)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rows = []
    for index in range(length):
        phase = (
            float(index) * golden_angle
            + float(int(route_index)) * 0.8115781021773633
            + float(MAIN_SEED % 100000) * 0.00017320508075688773
        ) % (2.0 * math.pi)
        rows.append(
            [magnitude * math.cos(phase), magnitude * math.sin(phase)]
        )
    return torch.tensor(np.asarray(rows), dtype=torch.float32)


vl.regular_grid_indices = regular_grid_indices_gpu
vl._deterministic_jitter = deterministic_fixed_radius_jitter
vl.config.SEED = MAIN_SEED


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python3 unified_gpu_runner.py <module.py> [arguments...]"
        )
    target = sys.argv.pop(1)
    if target.endswith(".py"):
        target = target[:-3]
    target = target.replace("/", ".")
    print(
        "[unified-runner] GPU grid lookup; CPU threads=%d; Stage-1 jitter=fixed-radius"
        % CPU_THREADS,
        flush=True,
    )
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
