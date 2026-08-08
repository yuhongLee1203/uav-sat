#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Must be set before config_backbone is imported.
    os.environ["RTL_BACKBONE"] = args.backbone
    os.environ.setdefault("RTL_TEMPORAL_WINDOW", "5")

    import config_backbone as config
    sys.modules["config"] = config

    from visual_model_backbone import AllMapGeoCLIP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AllMapGeoCLIP().to(device).eval()

    backbone_params = sum(
        p.numel() for p in model.clip.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def measure(batch, height, width, warmup, iters):
        x = torch.rand(
            batch, 3, height, width,
            device=device,
            dtype=torch.float32,
        )

        for _ in range(warmup):
            model.encode_clip_image(x)
        sync()

        start = time.perf_counter()
        for _ in range(iters):
            model.encode_clip_image(x)
        sync()
        elapsed = time.perf_counter() - start

        images = batch * iters
        return {
            "batch": int(batch),
            "input_hw": [int(height), int(width)],
            "iterations": int(iters),
            "total_seconds": float(elapsed),
            "ms_per_batch": float(1000.0 * elapsed / iters),
            "ms_per_image": float(
                1000.0 * elapsed / max(images, 1)
            ),
            "images_per_second": float(
                images / max(elapsed, 1e-12)
            ),
        }

    # Online UAV inference: one UAV image per frame.
    uav = measure(
        batch=1,
        height=256,
        width=256,
        warmup=int(args.warmup),
        iters=int(args.iters),
    )

    # Offline/cache build characteristic: many satellite patches at once.
    cache_batch = measure(
        batch=int(args.batch_size),
        height=320,
        width=320,
        warmup=max(5, int(args.warmup) // 3),
        iters=max(20, int(args.iters) // 5),
    )

    result = {
        "backbone_key": config.BACKBONE_KEY,
        "backbone_name": config.BACKBONE_NAME,
        "device": str(device),
        "backbone_parameters": int(backbone_params),
        "task_trainable_parameters": int(trainable_params),
        "uav_online_batch1": uav,
        "satellite_cache_batch": cache_batch,
        "note": (
            "Satellite backbone features are cached in the localization "
            "pipeline, so online frame-to-frame backbone cost is represented "
            "primarily by uav_online_batch1."
        ),
    }

    out_dir = Path("outputs/backbone_speed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.backbone}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
