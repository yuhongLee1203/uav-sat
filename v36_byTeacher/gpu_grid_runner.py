"""Low-CPU entry point for the six-architecture experiments.

This wrapper preserves the experiment math but replaces the legacy local-grid
lookup that copied the complete satellite gallery to CPU for every MeanShift.
Nearest-center distance is computed on the gallery device (GPU), and only the
selected scalar center index / final 36 indices are synchronized.
"""

import os
import runpy
import sys

import torch


CPU_THREADS = max(1, int(os.environ.get("UAVSAT_CPU_THREADS", "2")))
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
    """Exact grid lookup semantics without full-gallery GPU->CPU copies."""
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


vl.regular_grid_indices = regular_grid_indices_gpu


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python3 gpu_grid_runner.py <module.py> [module arguments...]"
        )
    target = sys.argv.pop(1)
    if target.endswith(".py"):
        target = target[:-3]
    target = target.replace("/", ".")
    print(
        "[cpu-fix] GPU satellite grid lookup enabled; CPU threads/process=%d"
        % CPU_THREADS,
        flush=True,
    )
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
