#!/usr/bin/env python3
"""Route-A training and fixed-gallery evaluation for released InfoGeo slots.

This runner uses the released InfoGeo object-centric slot-attention / cross-view
fusion modules.  The public release omitted its local DINOv2 checkout and
training helpers, so the backbone is loaded through timm's public DINOv2-B
weights.  It never uses a local prior, jitter, trajectory model, or temporal
state.  Route A is the sole training route; Routes B/C share one fixed gallery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
INFOGEO = EXP / "others_paper" / "InfoGeo"
DATA = EXP / "native-paper-data" / "U1652"
MANIFEST = EXP / "native-paper-data" / "manifest.json"
CHECKPOINT_SOURCE = EXP / "outputs" / "native-unreported" / "InfoGeo"
OUT = Path(os.environ.get("UAVSAT_NATIVE_OUTPUT_DIR", str(CHECKPOINT_SOURCE))).resolve()
sys.path.insert(0, str(INFOGEO))

from cvgl_base.model_slot_dias import (  # noqa: E402
    SlotMixFusion,
    SlotMixVPR,
    SlotCrossMoE,
)
from slot_attention.dias_slot_attention import NormalShared  # noqa: E402
from slot_attention.dias_slot_wrapper_cv import SlotAttentionWithAllAttent  # noqa: E402


class PairedRouteA(Dataset):
    def __init__(self, root: Path, transform):
        self.transform = transform
        self.items = []
        drone = root / "train" / "drone"
        satellite = root / "train" / "satellite"
        for class_dir in sorted(drone.iterdir()):
            if not class_dir.is_dir():
                continue
            sat = satellite / class_dir.name / f"{class_dir.name}.jpg"
            if not sat.exists():
                continue
            for image in sorted(class_dir.iterdir()):
                if image.is_file() or image.is_symlink():
                    self.items.append((image, sat))
        if not self.items:
            raise RuntimeError(f"No Route-A pairs in {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        query, sat = self.items[idx]
        with Image.open(query) as im:
            query_tensor = self.transform(im.convert("RGB"))
        with Image.open(sat) as im:
            sat_tensor = self.transform(im.convert("RGB"))
        return query_tensor, sat_tensor


def image_transform(train: bool):
    ops = [transforms.Resize((448, 448), antialias=True)]
    if train:
        ops += [
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(ops)


class InfoGeoReleasedSlots(nn.Module):
    """Released InfoGeo slot stack plus a portable public DINOv2-B backbone."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            global_pool="",
            img_size=448,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        vfm_dim, emb_dim, num_slots = 768, 1024, 16
        self.slot_mixvpr = SlotMixVPR(
            in_channels=vfm_dim, in_h=32, in_w=32, mix_depth=2, out_channels=emb_dim
        )
        self.initializ = NormalShared(num=num_slots, dim=emb_dim)
        self.slot_attention = SlotAttentionWithAllAttent(
            num_iter=3, embed_dim=emb_dim, ffn_dim=emb_dim * 4, kv_dim=vfm_dim, trunc_bp=None
        )
        self.mix_fusion = SlotMixFusion(
            in_channels=emb_dim, slot_channels=vfm_dim, slot_proj_dim=vfm_dim,
            num_slots=num_slots, alpha=0.8,
        )
        self.slot_moe = SlotCrossMoE(dim=emb_dim, nheads=4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def _initial_slots(self, batch: int):
        init = self.initializ
        if self.training:
            return init(batch)
        return init.mean.expand(batch, init.num, -1)

    def encode(self, image):
        tokens = self.backbone(image)
        tokens = tokens[:, 1:, :]
        feats = tokens.transpose(1, 2).reshape(image.shape[0], 768, 32, 32)
        _, attention_steps = self.slot_attention(
            feats.flatten(2).transpose(1, 2), self._initial_slots(image.shape[0])
        )
        attention = attention_steps[-1]
        attention, _ = self.slot_moe(attention, attention)
        mixed = self.mix_fusion(feats, attention)
        descriptor = self.slot_mixvpr(mixed)
        return F.normalize(descriptor, dim=1)

    def forward(self, query, sat):
        return self.encode(query), self.encode(sat)


def metric(errors):
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "frames": int(arr.size),
        "MLE_m": float(arr.mean()),
        "Median_m": float(np.median(arr)),
        "P90_m": float(np.quantile(arr, 0.90)),
        "P95_m": float(np.quantile(arr, 0.95)),
        "LSR@5_pct": float((arr <= 5).mean() * 100),
        "LSR@10_pct": float((arr <= 10).mean() * 100),
        "LSR@15_pct": float((arr <= 15).mean() * 100),
    }


@torch.no_grad()
def encode_paths(model, paths, transform, device, batch_size):
    output = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        output.append(model.encode(torch.stack(images).to(device)).cpu())
    return torch.cat(output)


def evaluate(model, manifest, device, batch_size, trajectory_dir=None):
    transform = image_transform(False)
    gallery_by_class = manifest["gallery"]
    classes = sorted(gallery_by_class, key=lambda value: int(value))
    gallery_paths = [DATA / "test" / "gallery_satellite" / f"{int(c):06d}" / f"{int(c):06d}.jpg" for c in classes]
    gallery_xy = np.asarray([gallery_by_class[c]["xy"] for c in classes], dtype=np.float64)
    gallery_feat = encode_paths(model, gallery_paths, transform, device, batch_size).to(device)
    routes = {}
    for route in ("route_B", "route_C"):
        records = [record for record in manifest["queries"] if record["route"] == route]
        paths = [DATA / "test" / "query_drone" / f"{int(record['class_id']):06d}" / record["file"] for record in records]
        query_feat = encode_paths(model, paths, transform, device, batch_size).to(device)
        started = time.perf_counter()
        pred = (query_feat @ gallery_feat.T).argmax(dim=1).cpu().numpy()
        elapsed = (time.perf_counter() - started) / max(len(paths), 1)
        gt = np.asarray([record["gt_xy"] for record in records], dtype=np.float64)
        pred_xy = gallery_xy[pred]
        errors = np.linalg.norm(pred_xy - gt, axis=1)
        if trajectory_dir is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trajectory_dir / f"{route}_predictions.npz",
                frame_id=np.asarray([record["frame_id"] for record in records], dtype=np.int32),
                gt_xy=gt.astype(np.float32),
                pred_xy=pred_xy.astype(np.float32),
                error_m=errors.astype(np.float32),
            )
        np.save(OUT / f"{route}_errors_m.npy", errors)
        result = metric(errors)
        result["latency_ms_fixed_gallery_similarity_only"] = float(elapsed * 1000)
        result["FPS_fixed_gallery_similarity_only"] = float(1.0 / elapsed) if elapsed else float("inf")
        routes[route] = result
    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    args = parser.parse_args()
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing native dataset manifest: {MANIFEST}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    model = InfoGeoReleasedSlots().to(device)
    checkpoint = CHECKPOINT_SOURCE / "route_A_infogeo_released_slots.pt" if args.evaluate_only else OUT / "route_A_infogeo_released_slots.pt"
    if args.evaluate_only:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    else:
        dataset = PairedRouteA(DATA, image_transform(True))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=6.5e-4, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for query, sat in loader:
                query, sat = query.to(device, non_blocking=True), sat.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    q, s = model(query, sat)
                    logits = model.logit_scale.exp() * q @ s.T
                    labels = torch.arange(q.shape[0], device=device)
                    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            print(f"InfoGeo Route-A epoch {epoch}/{args.epochs}: loss={np.mean(losses):.6f}", flush=True)
        torch.save({"model": model.state_dict(), "epochs": args.epochs, "seed": args.seed}, checkpoint)
    model.eval()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    routes = evaluate(model, manifest, device, args.eval_batch_size, args.trajectory_dir)
    summary = {
        "method": "InfoGeo",
        "implementation": "released InfoGeo slot-attention/cross-view fusion modules with portable public DINOv2-B replacement for omitted upstream local backbone paths",
        "protocol": "Route A train; fixed global satellite gallery; Route B/C test; no local prior/jitter/temporal state",
        "checkpoint": str(checkpoint),
        "routes": routes,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
