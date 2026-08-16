#!/usr/bin/env python3
"""Route-A adaptation of official paper backbone families under one protocol.

The original repositories use mutually incompatible datasets and metrics. This
adapter keeps each paper's official backbone family while enforcing the same
cached Route-A training pairs and Route-B/C forward-3x6 local gallery used by
V36, so metre error and FPS are directly computable on this project.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "v36-exp"
COMMON_CACHE = EXP / "cache/paper_common_256.pt"
SPECS = {
    "DenseUAV": {
        "model": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "size": 224,
        "source": "others_paper/DenseUAV",
        "adapter": "official DenseUAV ViT-S backbone family + shared retrieval head",
    },
    "Sample4Geo": {
        "model": "convnext_base.fb_in22k_ft_in1k_384",
        "size": 384,
        "source": "others_paper/Sample4Geo",
        "adapter": "official Sample4Geo ConvNeXt-B shared-weight dual encoder",
    },
    "Game4Loc": {
        "model": "convnext_base.fb_in22k_ft_in1k_384",
        "size": 384,
        "source": "others_paper/GTA-UAV",
        "adapter": "official Game4Loc ConvNeXt-B descriptor backbone; local Top-1 retrieval",
    },
    "InfoGeo": {
        "model": "vit_base_patch14_dinov2.lvd142m",
        "size": 252,
        "source": "others_paper/InfoGeo",
        "adapter": "official InfoGeo DINOv2-B backbone family; common global descriptor adapter",
    },
    "Bearing-UAV": {
        "model": "vgg16.tv_in1k",
        "size": 224,
        "source": "others_paper/Bearing-UAV",
        "adapter": "official Bearing-UAV VGG-16 backbone family; common local-gallery adapter",
    },
}


def preprocess(images, size, device):
    x = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def encode(model, images, size, device):
    features = model(preprocess(images, size, device))
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if features.ndim > 2:
        features = features.flatten(2).mean(-1)
    return F.normalize(features.float(), dim=1)


def quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(SPECS), required=True)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V36_PAPER_EPOCHS", "12")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("V36_PAPER_BATCH", "8")))
    args = parser.parse_args()
    if not COMMON_CACHE.exists():
        raise FileNotFoundError(f"Run prepare_paper_cache.py first: {COMMON_CACHE}")
    torch.manual_seed(2033)
    np.random.seed(2033)
    device = torch.device("cuda")
    spec = SPECS[args.method]
    output = EXP / "outputs/papers" / args.method
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "route_A_adapted.pt"
    summary_path = output / "summary.json"
    if summary_path.exists() and os.environ.get("V36_EXP_FORCE", "0") != "1":
        print(f"SKIP completed {args.method}: {summary_path}")
        return

    cache = torch.load(COMMON_CACHE, map_location="cpu")
    try:
        model = timm.create_model(spec["model"], pretrained=True, num_classes=0, img_size=spec["size"])
    except TypeError:
        model = timm.create_model(spec["model"], pretrained=True, num_classes=0)
    model = model.to(device)

    if checkpoint_path.exists() and os.environ.get("V36_EXP_FORCE", "0") != "1":
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
        print(f"reuse trained paper adapter: {checkpoint_path}")
    else:
        train = cache["routes"]["route_A"]
        query = train["query_images"]
        positive = cache["sat_images"][train["positive_sat_rows"]]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()
        model.train()
        for epoch in range(args.epochs):
            order = torch.randperm(len(query))
            losses = []
            for start in range(0, len(order), args.batch_size):
                ids = order[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    q = encode(model, query[ids], spec["size"], device)
                    s = encode(model, positive[ids], spec["size"], device)
                    logits = 14.285714 * q @ s.T
                    target = torch.arange(len(ids), device=device)
                    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            print(f"{args.method} epoch {epoch + 1:02d}/{args.epochs} loss={np.mean(losses):.5f}", flush=True)
        torch.save({"model": model.state_dict(), "spec": spec, "train_route": "route_A"}, checkpoint_path)

    model.eval()
    sat_feature_path = output / "sat_features.pt"
    if sat_feature_path.exists() and os.environ.get("V36_EXP_FORCE", "0") != "1":
        sat_features = torch.load(sat_feature_path, map_location="cpu")
    else:
        rows = []
        with torch.no_grad():
            for start in range(0, len(cache["sat_images"]), args.batch_size):
                rows.append(encode(model, cache["sat_images"][start:start + args.batch_size], spec["size"], device).cpu().half())
        sat_features = torch.cat(rows)
        torch.save(sat_features, sat_feature_path)
    sat_features_gpu = sat_features.float().to(device)

    summary = {
        "method": args.method,
        "official_source": str((EXP / spec["source"]).resolve()),
        "model": spec["model"],
        "adapter": spec["adapter"],
        "fairness_protocol": cache["protocol"],
        "train_routes": ["route_A"],
        "eval_routes": ["route_B", "route_C"],
    }
    all_errors = []
    all_latencies = []
    with torch.no_grad():
        for route_name in ("route_B", "route_C"):
            route = cache["routes"][route_name]
            errors, latencies = [], []
            for i in range(len(route["query_images"])):
                if i == 30:
                    latencies.clear()
                torch.cuda.synchronize()
                started = time.perf_counter()
                q = encode(model, route["query_images"][i:i + 1], spec["size"], device)
                candidate_rows = route["candidate_sat_rows"][i].to(device)
                candidate_features = sat_features_gpu[candidate_rows]
                selected_local = int((q @ candidate_features.T).argmax().item())
                selected_row = int(candidate_rows[selected_local].item())
                pred_xy = cache["sat_xy"][selected_row].numpy()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if i >= 30:
                    latencies.append(elapsed_ms)
                gt_xy = route["gt_xy"][i].numpy()
                errors.append(float(np.linalg.norm(pred_xy - gt_xy)))
            values = np.asarray(errors)
            route_summary = {
                "frames": int(len(values)),
                "MLE_m": float(values.mean()),
                "Median_m": float(np.median(values)),
                "P90_m": quantile(values, 0.90),
                "P95_m": quantile(values, 0.95),
                "LSR@5_pct": float((values <= 5).mean() * 100),
                "LSR@10_pct": float((values <= 10).mean() * 100),
                "LSR@15_pct": float((values <= 15).mean() * 100),
                "latency_ms": float(np.mean(latencies)),
                "FPS": float(1000.0 / np.mean(latencies)),
            }
            summary[route_name] = route_summary
            all_errors.extend(errors)
            all_latencies.extend(latencies)
            print(args.method, route_name, route_summary, flush=True)
    values = np.asarray(all_errors)
    summary["combined"] = {
        "frames": int(len(values)),
        "MLE_m": float(values.mean()),
        "Median_m": float(np.median(values)),
        "P90_m": quantile(values, 0.90),
        "P95_m": quantile(values, 0.95),
        "LSR@5_pct": float((values <= 5).mean() * 100),
        "LSR@10_pct": float((values <= 10).mean() * 100),
        "LSR@15_pct": float((values <= 15).mean() * 100),
        "latency_ms": float(np.mean(all_latencies)),
        "FPS": float(1000.0 / np.mean(all_latencies)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
