#!/usr/bin/env python3
"""Train/evaluate public cross-view baselines on the same Bearing route cache.

Supported public methods
------------------------
* university1652 : layumi/University1652-Baseline official ResNet50 two-view net
* sues200        : Reza-Zhu/SUES-200-Benchmark official ResNet50 two-view net
* denseuav       : Dmmm1997/DenseUAV official ViT-S/SingleBranch objective
* gtauav         : Yux1angJi/GTA-UAV official Game4Loc ViT/InfoNCE model

This is a ROUTE-ADAPTED REPRODUCTION, not the methods' original published data
protocol.  The official model implementation and training objective are used,
but all methods are retrained only on our retained train_01 pairs and evaluated
on exactly the same test_01/test_02 UAV frames and same planned-route satellite
gallery.  The gallery contains no per-frame test-GT centres.

CPU/I/O policy
--------------
All source JPEG/RSI decoding is performed once by bearing_route_baseline_cache.py.
This runner reads uncompressed .npy arrays with mmap_mode='r', uses no DataLoader
workers, and performs resize/normalization/most augmentation on the GPU.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

METHODS = ("university1652", "sues200", "denseuav", "gtauav")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRED = (238, 45, 45, 255)
GT = (176, 78, 245, 255)
REF = (215, 215, 215, 180)
HALO = (255, 255, 255, 238)


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for route-adapted baseline training")
    return torch.device("cuda:0")


def _batched_indices(n: int, batch: int, *, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, n, batch):
        yield idx[s:s + batch]


def _raw_batch(arr: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    # Fancy indexing creates one contiguous batch copy from mmap without JPEG decode.
    x = torch.from_numpy(np.asarray(arr[idx], dtype=np.uint8)).to(device, non_blocking=True)
    return x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


def _prep(
    x: torch.Tensor,
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    train: bool,
    satellite: bool,
) -> torch.Tensor:
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    if train:
        # Lightweight GPU-side counterparts of the public repositories' common
        # cross-view augmentations.  They never alter labels/coordinates.
        if torch.rand((), device=x.device).item() < 0.5:
            x = torch.flip(x, dims=[3])
        if satellite:
            k = int(torch.randint(0, 4, (), device=x.device).item())
            if k:
                x = torch.rot90(x, k, dims=(2, 3))
        # Mild brightness/contrast jitter, kept deliberately conservative.
        if torch.rand((), device=x.device).item() < 0.25:
            b = float(torch.empty((), device=x.device).uniform_(0.9, 1.1).item())
            c = float(torch.empty((), device=x.device).uniform_(0.9, 1.1).item())
            mu = x.mean(dim=(2, 3), keepdim=True)
            x = ((x - mu) * c + mu) * b
            x = x.clamp_(0.0, 1.0)
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


class Adapter:
    def __init__(self, method: str, repo: Path, nclasses: int, device: torch.device, batch: int):
        self.method = method
        self.repo = repo
        self.nclasses = nclasses
        self.device = device
        self.batch = batch
        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scheduler = None
        self.scheduler_per_step = False
        self.criterion = None
        self.input_size = 256
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.epochs = 1
        self.amp = True
        self.description = ""
        self._build()

    def _front(self, path: Path):
        p = str(path.resolve())
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    def _build(self):
        if self.method == "university1652":
            self._front(self.repo)
            um = importlib.import_module("model")
            self.model = um.two_view_net(
                self.nclasses, droprate=0.75, stride=1, pool="avg",
                share_weight=True, VGG16=False,
            ).to(self.device)
            ignored = set(map(id, self.model.classifier.parameters()))
            base = [p for p in self.model.parameters() if id(p) not in ignored]
            self.optimizer = torch.optim.SGD(
                [
                    {"params": base, "lr": 0.001},
                    {"params": self.model.classifier.parameters(), "lr": 0.01},
                ], weight_decay=5e-4, momentum=0.9, nesterov=True,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.1)
            self.criterion = nn.CrossEntropyLoss()
            self.input_size = 256
            self.epochs = 120
            self.description = "official University-1652 ResNet50 shared two-view CE recipe, route-adapted"

        elif self.method == "sues200":
            self._front(self.repo)
            sm = importlib.import_module("model_")
            self.model = sm.ResNet(self.nclasses, 0.5, share_weight=False, pretrained=True).to(self.device)
            ignored = set(map(id, self.model.classifier.parameters()))
            base = [p for p in self.model.parameters() if id(p) not in ignored]
            self.optimizer = torch.optim.SGD(
                [
                    {"params": base, "lr": 1e-4},
                    {"params": self.model.classifier.parameters(), "lr": 1e-3},
                ], weight_decay=1e-4, momentum=0.9, nesterov=True,
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.1)
            self.criterion = nn.CrossEntropyLoss()
            self.input_size = 384
            self.epochs = 60
            self.description = "official SUES-200 ResNet50 two-branch CE recipe, route-adapted"

        elif self.method == "denseuav":
            self._front(self.repo)
            # Official taskflow implementation, using the baseline settings from
            # train_test_local.sh (ViTS-224 + SingleBranch + CE + WSTL + KL).
            taskflow = importlib.import_module("models.taskflow")
            lossmod = importlib.import_module("losses.loss")
            optmod = importlib.import_module("optimizers.make_optimizer")
            opt = SimpleNamespace(
                nclasses=self.nclasses, droprate=0.5, backbone="ViTS-224",
                head="SingleBranch", head_pool="avg", num_bottleneck=512,
                load_from="no", h=224, w=224, block=1,
                cls_loss="CELoss", feature_loss="WeightedSoftTripletLoss",
                kl_loss="KLLoss", sample_num=1, batchsize=self.batch,
                lr=0.01,
            )
            self.dense_opt = opt
            self.model = taskflow.make_model(opt).to(self.device)
            self.criterion = lossmod.Loss(opt).to(self.device)
            self.optimizer, self.scheduler = optmod.make_optimizer(self.model, opt)
            self.input_size = 224
            self.epochs = 120
            self.description = "official DenseUAV ViT-S/SingleBranch CE+WeightedSoftTriplet+KL recipe, route-adapted"

        elif self.method == "gtauav":
            self._front(self.repo / "Game4Loc")
            gm = importlib.import_module("game4loc.models.model")
            gl = importlib.import_module("game4loc.loss")
            self.model = gm.DesModel(
                model_name="vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
                pretrained=True, img_size=256, share_weights=True,
            ).to(self.device)
            cfg = self.model.get_config()
            self.mean = tuple(float(v) for v in cfg["mean"])
            self.std = tuple(float(v) for v in cfg["std"])
            self.criterion = gl.InfoNCE(nn.CrossEntropyLoss(), device=str(self.device))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
            self.input_size = 256
            self.epochs = 20
            self.scheduler_per_step = True
            self.description = "official GTA-UAV/Game4Loc ViT shared-weight standard InfoNCE recipe, route-adapted"
        else:
            raise ValueError(self.method)

    def config(self) -> dict:
        return {
            "method": self.method,
            "description": self.description,
            "epochs": self.epochs,
            "batch": self.batch,
            "input_size": self.input_size,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    def prepare_scheduler(self, steps_per_epoch: int):
        if self.method != "gtauav":
            return
        total = max(1, steps_per_epoch * self.epochs)
        warm = max(1, int(round(0.1 * steps_per_epoch)))
        def lr_lambda(step: int):
            if step < warm:
                return float(step + 1) / warm
            progress = (step - warm) / max(total - warm, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, sat: torch.Tensor, uav: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.method in ("university1652", "sues200"):
            out_s, out_u = self.model(sat, uav)
            return self.criterion(out_s, labels) + self.criterion(out_u, labels)
        if self.method == "denseuav":
            out_u, out_s = self.model(uav, sat)
            loss, _, _, _ = self.criterion(out_u, out_s, labels, labels)
            return loss
        if self.method == "gtauav":
            fu, fs = self.model(img1=uav, img2=sat)
            return self.criterion(fu, fs, self.model.logit_scale.exp())
        raise ValueError(self.method)

    def embed(self, x: torch.Tensor, view: str) -> torch.Tensor:
        if self.method == "university1652":
            branch = self.model.model_1 if view == "sat" else self.model.model_2
            z = branch(x)
            z = self.model.classifier.add_block(z)
        elif self.method == "sues200":
            branch = self.model.model_1 if view == "sat" else self.model.model_2
            z = branch(x)
            z = self.model.classifier.add_block(z)
        elif self.method == "denseuav":
            if view == "uav":
                out, _ = self.model(x, None)
            else:
                _, out = self.model(None, x)
            z = out[1]
        elif self.method == "gtauav":
            z = self.model(img1=x) if view == "uav" else self.model(img2=x)
        else:
            raise ValueError(self.method)
        if z.ndim > 2:
            z = z.flatten(1)
        return F.normalize(z.float(), dim=-1)


def _train(adapter: Adapter, cache: Path, checkpoint: Path, fingerprint: str, force: bool, seed: int):
    if checkpoint.exists() and not force:
        ckpt = torch.load(checkpoint, map_location="cpu")
        if ckpt.get("fingerprint") == fingerprint:
            adapter.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"[BASELINE] checkpoint cache hit: {checkpoint}", flush=True)
            return

    uav = np.load(cache / "train_uav.npy", mmap_mode="r")
    sat = np.load(cache / "train_sat.npy", mmap_mode="r")
    if len(uav) != len(sat) or len(uav) != adapter.nclasses:
        raise RuntimeError("train cache mismatch")

    steps = int(math.ceil(len(uav) / adapter.batch))
    adapter.prepare_scheduler(steps)
    scaler = torch.cuda.amp.GradScaler(enabled=adapter.amp)
    rng = np.random.default_rng(seed)
    adapter.model.train()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, adapter.epochs + 1):
        total = 0.0
        seen = 0
        for idx in _batched_indices(len(uav), adapter.batch, shuffle=True, rng=rng):
            xu = _prep(_raw_batch(uav, idx, adapter.device), adapter.input_size, adapter.mean, adapter.std,
                       train=True, satellite=False)
            xs = _prep(_raw_batch(sat, idx, adapter.device), adapter.input_size, adapter.mean, adapter.std,
                       train=True, satellite=True)
            y = torch.as_tensor(idx, device=adapter.device, dtype=torch.long)
            adapter.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=adapter.amp):
                loss = adapter.train_step(xs, xu, y)
            scaler.scale(loss).backward()
            scaler.unscale_(adapter.optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 100.0)
            scaler.step(adapter.optimizer)
            scaler.update()
            if adapter.scheduler_per_step and adapter.scheduler is not None:
                adapter.scheduler.step()
            total += float(loss.detach().item()) * len(idx)
            seen += len(idx)
        if not adapter.scheduler_per_step and adapter.scheduler is not None:
            adapter.scheduler.step()
        avg = total / max(seen, 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == adapter.epochs:
            lr = adapter.optimizer.param_groups[0]["lr"]
            print(f"[{adapter.method}] epoch={epoch:03d}/{adapter.epochs} loss={avg:.6f} lr={lr:.3e}", flush=True)

    if best_state is not None:
        adapter.model.load_state_dict(best_state, strict=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fingerprint": fingerprint,
        "model_state_dict": adapter.model.state_dict(),
        "config": adapter.config(),
        "best_train_loss": best_loss,
    }, checkpoint)
    print(f"[BASELINE] saved {checkpoint}", flush=True)


def _features(adapter: Adapter, arr: np.ndarray, view: str, batch: int) -> torch.Tensor:
    adapter.model.eval()
    out = []
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        for idx in _batched_indices(len(arr), batch, shuffle=False, rng=rng):
            x = _prep(_raw_batch(arr, idx, adapter.device), adapter.input_size, adapter.mean, adapter.std,
                      train=False, satellite=(view == "sat"))
            with torch.cuda.amp.autocast(enabled=adapter.amp):
                z = adapter.embed(x, view)
            out.append(z.detach())
    return torch.cat(out, dim=0)


def _metrics(pred: np.ndarray, gt: np.ndarray, top1: np.ndarray, gallery: np.ndarray) -> dict:
    err = np.linalg.norm(pred - gt, axis=1)
    # Route-gallery retrieval recall: exact top-1 match to the nearest candidate to
    # GT.  This is NOT the papers' published identity Recall@1 and is named so.
    d2 = ((gt[:, None, :] - gallery[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    return {
        "frames": int(len(err)),
        "RouteGalleryRecall@1_pct": float(100.0 * np.mean(top1 == nearest)),
        "MLE_m": float(err.mean()),
        "MedLE_m": float(np.median(err)),
        "P90_m": float(np.percentile(err, 90)),
        "P95_m": float(np.percentile(err, 95)),
        "P99_m": float(np.percentile(err, 99)),
        "LSR@5_pct": float(100.0 * np.mean(err <= 5.0)),
        "LSR@10_pct": float(100.0 * np.mean(err <= 10.0)),
        "LSR@15_pct": float(100.0 * np.mean(err <= 15.0)),
        "LSR@20_pct": float(100.0 * np.mean(err <= 20.0)),
        "distance_errors_m": err.tolist(),
    }


def _waypoints(prepared: Path, route: str) -> List[Tuple[float, float]]:
    p = _json(prepared / "routes" / route / "waypoints.json")
    return [(float(x["pixel_x"]), float(x["pixel_y"])) for x in sorted(p["waypoints"], key=lambda x: int(x["waypoint_order"]))]


def _font(size: int, bold=False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _dashed(draw, pts, fill, width, dash=34, gap=20):
    for a, b in zip(pts[:-1], pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L <= 0: continue
        ux, uy = dx/L, dy/L
        s = 0.0
        while s < L:
            e = min(L, s + dash)
            draw.line((a[0]+ux*s, a[1]+uy*s, a[0]+ux*e, a[1]+uy*e), fill=fill, width=width)
            s += dash + gap


def _figure(prepared: Path, route: str, pred_m: np.ndarray, gt_m: np.ndarray, metrics: dict, out: Path, method_name: str):
    sm = _json(prepared / "bearing_satellite.json")
    mpp = float(sm["mpp"])
    src = Image.open(sm["satellite_image"]).convert("RGB")
    base = ImageEnhance.Brightness(src).enhance(0.84).convert("RGBA")
    pred = [(float(x/mpp), float(y/mpp)) for x, y in pred_m]
    gt = [(float(x/mpp), float(y/mpp)) for x, y in gt_m]
    ref = _waypoints(prepared, route)

    allp = pred + gt + ref
    xs, ys = [p[0] for p in allp], [p[1] for p in allp]
    margin = max(220, int(0.06 * max(max(xs)-min(xs), max(ys)-min(ys), 1)))
    box = (max(0,int(min(xs))-margin), max(0,int(min(ys))-margin), min(base.width,int(max(xs))+margin), min(base.height,int(max(ys))+margin))

    draw = ImageDraw.Draw(base, "RGBA")
    # planned route: context only
    _dashed(draw, ref, (255,255,255,140), 6, 18, 16)
    _dashed(draw, ref, REF, 3, 18, 16)
    # true per-frame GT: strong purple dashed + white halo
    _dashed(draw, gt, HALO, 20, 36, 18)
    _dashed(draw, gt, GT, 13, 36, 18)
    # prediction: strong red solid + white halo
    draw.line(pred, fill=HALO, width=23, joint="curve")
    draw.line(pred, fill=PRED, width=16, joint="curve")
    r = 13
    for p, c in ((gt[0], GT),(gt[-1],GT),(pred[0],PRED),(pred[-1],PRED)):
        draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r), fill=c, outline=HALO, width=4)

    crop = base.crop(box)
    overlay = Image.new("RGBA", crop.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay, "RGBA")
    tf, bf = _font(27, True), _font(21, False)
    lines = [
        f"{method_name} | {route}",
        f"MLE {metrics['MLE_m']:.2f} m   P90 {metrics['P90_m']:.2f} m   LSR@15 {metrics['LSR@15_pct']:.1f}%",
        "Purple dashed: per-frame GT    Red solid: prediction",
        "Gray dotted: planned waypoint route",
    ]
    pad, lh = 22, 34
    bw, bh = min(crop.width-2*pad, 850), pad*2 + lh*len(lines)
    od.rounded_rectangle((pad,pad,pad+bw,pad+bh), radius=14, fill=(0,0,0,185), outline=(255,255,255,180), width=2)
    for i, text in enumerate(lines):
        od.text((pad+18,pad+15+i*lh), text, font=tf if i==0 else bf, fill=(255,255,255,255))
    crop.alpha_composite(overlay)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(out, quality=96, subsampling=0)


def _evaluate(adapter: Adapter, cache: Path, prepared: Path, outdir: Path) -> dict:
    result = {}
    for route in ("test_01", "test_02"):
        q = np.load(cache / f"{route}_uav.npy", mmap_mode="r")
        gt = np.load(cache / f"{route}_gt_m.npy")
        g = np.load(cache / f"{route}_gallery_sat.npy", mmap_mode="r")
        gxy = np.load(cache / f"{route}_gallery_xy_m.npy")
        qf = _features(adapter, q, "uav", max(adapter.batch, 16))
        gf = _features(adapter, g, "sat", max(adapter.batch, 32))
        sim = qf @ gf.T
        top1 = torch.argmax(sim, dim=1).detach().cpu().numpy()
        pred = gxy[top1]
        met = _metrics(pred, gt, top1, gxy)
        result[route] = met

        csv_path = outdir / f"{route}_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_id","gt_x_m","gt_y_m","pred_x_m","pred_y_m","gallery_index","error_m"])
            for i, (gg, pp, gi, e) in enumerate(zip(gt, pred, top1, met["distance_errors_m"])):
                w.writerow([i,float(gg[0]),float(gg[1]),float(pp[0]),float(pp[1]),int(gi),float(e)])
        _figure(prepared, route, pred, gt, met, outdir / f"{route}_final_result.jpg", adapter.method)
        print(f"[{adapter.method}] {route}: MLE={met['MLE_m']:.3f}m MedLE={met['MedLE_m']:.3f}m LSR15={met['LSR@15_pct']:.2f}%", flush=True)

        del qf, gf, sim
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--repo-commit", default="unknown")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.backends.cudnn.benchmark = True
    _seed(args.seed)

    cache = Path(args.cache_dir).resolve()
    prepared = Path(args.prepared_root).resolve()
    outdir = Path(args.output_root).resolve() / args.method / args.city
    outdir.mkdir(parents=True, exist_ok=True)
    cache_meta = _json(cache / "cache_meta.json")
    nclasses = int(cache_meta["train_frames"])
    default_batch = {"university1652": 16, "sues200": 16, "denseuav": 16, "gtauav": 32}[args.method]
    batch = args.batch_size if args.batch_size > 0 else min(default_batch, nclasses)

    device = _device()
    adapter = Adapter(args.method, Path(args.repo_dir).resolve(), nclasses, device, batch)
    fp_payload = {
        "cache_fingerprint": cache_meta["fingerprint"],
        "repo_commit": args.repo_commit,
        "adapter": adapter.config(),
        "seed": args.seed,
        "runner_version": "route_baseline_v1",
    }
    fingerprint = _sha(fp_payload)
    checkpoint = outdir / "checkpoint.pt"
    _train(adapter, cache, checkpoint, fingerprint, args.force, args.seed)
    adapter.model.to(device).eval()

    routes = _evaluate(adapter, cache, prepared, outdir)
    allerr = np.asarray(routes["test_01"]["distance_errors_m"] + routes["test_02"]["distance_errors_m"], dtype=np.float64)
    pooled = {
        "frames": int(len(allerr)),
        "MLE_m": float(allerr.mean()),
        "MedLE_m": float(np.median(allerr)),
        "P90_m": float(np.percentile(allerr, 90)),
        "LSR@5_pct": float(100*np.mean(allerr<=5)),
        "LSR@10_pct": float(100*np.mean(allerr<=10)),
        "LSR@15_pct": float(100*np.mean(allerr<=15)),
        "LSR@20_pct": float(100*np.mean(allerr<=20)),
    }
    payload = {
        "method": args.method,
        "city": args.city,
        "kind": "route-adapted reproduction using public official model implementation/objective",
        "repo": str(Path(args.repo_dir).resolve()),
        "repo_commit": args.repo_commit,
        "training_scope": "only retained train_01 UAV/satellite pairs for this city",
        "test_scope": "exact same retained test_01/test_02 UAV frames as ours",
        "gallery_scope": cache_meta["gallery_policy"],
        "published_protocol_equivalent": False,
        "protocol_note": "common route adaptation for fair same-data comparison; do not replace the methods' published benchmark rows with these values",
        "config": adapter.config(),
        "cache_fingerprint": cache_meta["fingerprint"],
        "routes": routes,
        "aggregate_two_routes": pooled,
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[BASELINE-DONE] {args.method}/{args.city}: MLE={pooled['MLE_m']:.3f}m LSR15={pooled['LSR@15_pct']:.2f}%", flush=True)

    del adapter
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
