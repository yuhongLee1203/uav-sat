#!/usr/bin/env python3
"""Route-A training for released Bearing-UAV PARCASGM-v5a.

Bearing-UAV expects four satellite context patches and a UAV image.  This
adapter constructs those four patches as the fixed 2x2 partition of the full
satellite map, so every frame receives identical global context and no query
GT, local candidate window, jitter, trajectory state, or temporal result is
provided at inference.  Position targets are normalized full-map coordinates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
SOURCE = EXP / "others_paper" / "Bearing-UAV"
BASE = ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
CHECKPOINT_SOURCE = EXP / "outputs" / "native-unreported" / "Bearing-UAV"
OUT = Path(os.environ.get("UAVSAT_NATIVE_OUTPUT_DIR", str(CHECKPOINT_SOURCE))).resolve()
sys.path.insert(0, str(SOURCE))

from cvphr.models.posaglreg.models import PARCASGM_v5a  # noqa: E402


def root_config():
    spec = importlib.util.spec_from_file_location("uavsat_experiment_config", EXP / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_map_context():
    with Image.open(root_config().SAT_IMAGE) as image:
        satellite = image.convert("RGB")
        width, height = satellite.size
        mid_x, mid_y = width // 2, height // 2
        crops = [
            satellite.crop((0, 0, mid_x, mid_y)),
            satellite.crop((0, mid_y, mid_x, height)),
            satellite.crop((mid_x, 0, width, mid_y)),
            satellite.crop((mid_x, mid_y, width, height)),
        ]
    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return torch.stack([transform(crop) for crop in crops]), width, height


def route_payloads():
    checkpoint = torch.load(BASE / "checkpoints" / "visual_retrieval_A_only.pt", map_location="cpu")
    gallery = checkpoint["gallery"]
    gallery_xy = gallery["xy"].float().cpu().numpy()
    gallery_pixel = gallery["pixel"].float().cpu().numpy()
    tree = cKDTree(gallery_xy)
    routes = {}
    for route in ("route_A", "route_B", "route_C"):
        payload = torch.load(BASE / "feature_cache" / f"{route}_uav_clip.pt", map_location="cpu")
        gt_xy = payload["gt_xy"].float().cpu().numpy()
        nearest = tree.query(gt_xy, k=1)[1]
        pixels = gallery_pixel[nearest]
        delta = np.zeros_like(gt_xy)
        delta[1:] = gt_xy[1:] - gt_xy[:-1]
        if len(delta) > 1:
            delta[0] = delta[1]
        headings = np.arctan2(delta[:, 1], delta[:, 0])
        routes[route] = {
            "paths": [Path(path) for path in payload["image_paths"]],
            "gt_xy": gt_xy,
            "pixel": pixels,
            "heading": headings,
        }
    return routes, gallery_pixel, gallery_xy


class BearingRoute(Dataset):
    def __init__(self, payload, context, width, height, train):
        self.paths = payload["paths"]
        self.pixel = payload["pixel"].astype(np.float32)
        self.heading = payload["heading"].astype(np.float32)
        self.context = context
        self.width = float(width - 1)
        self.height = float(height - 1)
        ops = [transforms.Resize((256, 256), antialias=True)]
        if train:
            ops += [transforms.ColorJitter(0.12, 0.12, 0.12, 0.04)]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            uav = self.transform(image.convert("RGB"))
        xy = self.pixel[idx]
        position = torch.tensor([
            2.0 * xy[0] / self.width - 1.0,
            2.0 * xy[1] / self.height - 1.0,
        ], dtype=torch.float32)
        heading = self.heading[idx]
        direction = torch.tensor([math.cos(float(heading)), math.sin(float(heading))], dtype=torch.float32)
        return torch.cat((self.context, uav.unsqueeze(0)), dim=0), position, direction


def error_summary(errors):
    values = np.asarray(errors, dtype=np.float64)
    return {
        "frames": int(values.size),
        "MLE_m": float(values.mean()),
        "Median_m": float(np.median(values)),
        "P90_m": float(np.quantile(values, 0.90)),
        "P95_m": float(np.quantile(values, 0.95)),
        "LSR@5_pct": float((values <= 5).mean() * 100),
        "LSR@10_pct": float((values <= 10).mean() * 100),
        "LSR@15_pct": float((values <= 15).mean() * 100),
    }


@torch.no_grad()
def evaluate(model, route_name, payload, context, width, height, px_to_xy, device, batch, trajectory_dir=None):
    dataset = BearingRoute(payload, context, width, height, train=False)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    predicted = []
    started = time.perf_counter()
    for patches, _, _ in loader:
        position, _ = model(patches.to(device, non_blocking=True))
        predicted.append(position.cpu().float().numpy())
    seconds = time.perf_counter() - started
    normalized = np.concatenate(predicted)
    pixels = np.column_stack(((normalized[:, 0] + 1.0) * 0.5 * (width - 1), (normalized[:, 1] + 1.0) * 0.5 * (height - 1)))
    matrix = np.column_stack((pixels, np.ones(len(pixels))))
    xy = matrix @ px_to_xy
    errors = np.linalg.norm(xy - payload["gt_xy"], axis=1)
    if trajectory_dir is not None:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_dir / f"{route_name}_predictions.npz",
            frame_id=np.arange(len(errors), dtype=np.int32),
            gt_xy=payload["gt_xy"].astype(np.float32),
            pred_xy=xy.astype(np.float32),
            error_m=errors.astype(np.float32),
        )
    np.save(OUT / f"{route_name}_errors_m.npy", errors)
    result = error_summary(errors)
    result["latency_ms_end_to_end"] = float(seconds / len(dataset) * 1000)
    result["FPS_end_to_end"] = float(len(dataset) / seconds)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    context, width, height = build_map_context()
    routes, gallery_pixel, gallery_xy = route_payloads()
    # Affine map-pixel -> world metre conversion obtained from the fixed gallery.
    px_to_xy, *_ = np.linalg.lstsq(
        np.column_stack((gallery_pixel, np.ones(len(gallery_pixel)))), gallery_xy, rcond=None
    )
    model = PARCASGM_v5a().to(device)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_SOURCE / "route_A_bearinguav_parcasgm_v5a.pt" if args.evaluate_only else OUT / "route_A_bearinguav_parcasgm_v5a.pt"
    if args.evaluate_only:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    else:
        train_set = BearingRoute(routes["route_A"], context, width, height, train=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for patches, position, direction in train_loader:
                optimizer.zero_grad(set_to_none=True)
                patches = patches.to(device, non_blocking=True)
                position = position.to(device, non_blocking=True)
                direction = direction.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    pos_pred, dir_pred = model(patches)
                    loss = 0.8 * F.smooth_l1_loss(pos_pred, position) + 0.2 * F.smooth_l1_loss(dir_pred, direction)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            print(f"Bearing-UAV Route-A epoch {epoch}/{args.epochs}: loss={np.mean(losses):.6f}", flush=True)
        torch.save({"model": model.state_dict(), "epochs": args.epochs, "seed": args.seed}, checkpoint)
    model.eval()
    summary = {
        "method": "Bearing-UAV",
        "implementation": "released PARCASGM-v5a model and position+heading SmoothL1 objective",
        "protocol": "Route A train; four fixed full-map 2x2 satellite context patches; Route B/C test; no local prior/jitter/temporal state",
        "checkpoint": str(checkpoint),
        "routes": {
            "route_B": evaluate(model, "route_B", routes["route_B"], context, width, height, px_to_xy, device, args.batch_size, args.trajectory_dir),
            "route_C": evaluate(model, "route_C", routes["route_C"], context, width, height, px_to_xy, device, args.batch_size, args.trajectory_dir),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
