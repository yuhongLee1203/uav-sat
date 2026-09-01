"""GPU/low-CPU runner for the MKG ablation suite.

Besides the existing GPU grid lookup optimization, this wrapper makes decoder
latency fair: Top-1 and weighted-centroid runs do not execute SoftMeanShift
internally just to construct CandidateBatch metadata. SoftMS runs keep the
original candidate_batch implementation unchanged.
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


def requested_decoder():
    if "--decoder" not in sys.argv:
        return "softms"
    i = sys.argv.index("--decoder")
    if i + 1 >= len(sys.argv):
        return "softms"
    return sys.argv[i + 1].strip().lower()


_ORIGINAL_CANDIDATE_BATCH = vl.FrozenVisualLocalizer.candidate_batch


@torch.no_grad()
def candidate_batch_without_softms(self, uav_clip, center_xy, grid_size=None):
    grid_size = int(grid_size or vl.config.GRID_SIZE)
    indices = vl.regular_grid_indices(
        self.gallery["xy"],
        self.gallery["pixel"],
        self.pixel_index,
        center_xy,
        grid_size,
        vl.config.SAT_STRIDE,
        self.device,
    )
    centers = self.gallery["xy"][indices]
    satellite_clip = self.gallery["clip_feat"][indices]

    z_uav = self.model.encode_uav_from_clip(uav_clip)
    z_sat = self.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)

    raw_logits = self.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    raw_prob = torch.softmax(
        raw_logits / float(vl.config.MEANSHIFT_SCORE_TAU), dim=1
    )
    raw_index = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=self.device), raw_index
    ]
    dummy_support = raw_prob.max(dim=1).values
    dummy_modes = torch.ones(
        raw_logits.shape[0], dtype=torch.long, device=self.device
    )

    return vl.CandidateBatch(
        indices=indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=raw_top1_xy,
        softms_support=dummy_support,
        softms_mode_count=dummy_modes,
    )


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python3 mkg_ablation_gpu_runner.py <module.py> [arguments...]"
        )
    target = sys.argv.pop(1)
    decoder = requested_decoder()
    if decoder in ("top1", "weighted"):
        vl.FrozenVisualLocalizer.candidate_batch = candidate_batch_without_softms
        skip_note = "SoftMS skipped internally"
    else:
        vl.FrozenVisualLocalizer.candidate_batch = _ORIGINAL_CANDIDATE_BATCH
        skip_note = "SoftMS active"

    if target.endswith(".py"):
        target = target[:-3]
    target = target.replace("/", ".")
    print(
        "[mkg-runner] GPU grid lookup; CPU threads=%d; decoder=%s; %s"
        % (CPU_THREADS, decoder, skip_note),
        flush=True,
    )
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
