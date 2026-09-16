#!/usr/bin/env python3
"""Native matching-to-tile reproductions on the Bearing route frames.

Supported public implementations:
  * University-1652: layumi/University1652-Baseline
  * SUES-200:        Reza-Zhu/SUES-200-Benchmark
  * DenseUAV:        Dmmm1997/DenseUAV
  * GTA-UAV:         Yux1angJi/GTA-UAV / Game4Loc

Protocol
--------
These baselines are NOT given our waypoint route, temporal history, local prior,
previous position, or a route-restricted gallery.  They train on the selected
Route-A UAV observations using their normal cross-view objective, paired with
the fixed Bearing RST tile containing each observation.  At test time every UAV
frame independently searches ALL 256 fixed RST tiles of the city RSI.  The
predicted position is the center of the retrieved top-1 tile, exactly preserving
the matching-to-tile quantization behavior that Bearing-UAV compares against.

A method may drift, jump, or fail to approach the route endpoint; this script
never corrects it with our route knowledge.  Waypoints are used only as a faint
visual overlay in the output figure and never enter inference.

CPU/I/O policy: source JPEGs are decoded once by bearing_route_baseline_cache.py;
this runner mmap's uint8 arrays, uses no DataLoader workers, and performs resize
and lightweight augmentation on GPU.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

METHODS = ("university1652", "sues200", "denseuav", "gtauav")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RUNNER_VERSION = "native_m2t_full_city_v2"

PRED = (238, 45, 45, 255)
GT = (176, 78, 245, 255)
REF = (210, 210, 210, 180)
HALO = (255, 255, 255, 240)


def _seed(seed: int) -> None:
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
        raise RuntimeError("CUDA is required for baseline training")
    return torch.device("cuda:0")


def _normal_batches(n: int, batch: int, rng: np.random.Generator):
    idx = np.arange(n)
    rng.shuffle(idx)
    chunks = [idx[s:s + batch] for s in range(0, n, batch)]
    if chunks and len(chunks[-1]) == 1 and len(chunks) > 1:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    for c in chunks:
        if len(c) >= 2:
            yield c


def _unique_class_batches(labels: np.ndarray, batch: int, rng: np.random.Generator):
    """One randomly chosen UAV per RST class per batch (avoids InfoNCE false negatives)."""
    classes = np.unique(labels)
    rng.shuffle(classes)
    chosen = []
    for c in classes:
        candidates = np.flatnonzero(labels == c)
        chosen.append(int(rng.choice(candidates)))
    idx = np.asarray(chosen, dtype=np.int64)
    chunks = [idx[s:s + batch] for s in range(0, len(idx), batch)]
    if chunks and len(chunks[-1]) == 1 and len(chunks) > 1:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    for c in chunks:
        if len(c) >= 2:
            yield c


def _sequential_batches(n: int, batch: int):
    for s in range(0, n, batch):
        yield np.arange(s, min(s + batch, n), dtype=np.int64)


def _raw_batch(arr: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(arr[idx], dtype=np.uint8)).to(device, non_blocking=True)
    return x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


def _prep(x: torch.Tensor, size: int, mean: Sequence[float], std: Sequence[float], *, train: bool, satellite: bool) -> torch.Tensor:
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    if train:
        if torch.rand((), device=x.device).item() < 0.5:
            x = torch.flip(x, dims=[3])
        if satellite:
            # Rotation augmentation is native to these tile-matching pipelines.
            k = int(torch.randint(0, 4, (), device=x.device).item())
            if k:
                x = torch.rot90(x, k, dims=(2, 3))
        if torch.rand((), device=x.device).item() < 0.20:
            gain = float(torch.empty((), device=x.device).uniform_(0.90, 1.10).item())
            x = (x * gain).clamp_(0.0, 1.0)
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

    def _front(self, path: Path) -> None:
        p = str(path.resolve())
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    def _build(self) -> None:
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
                [{"params": base, "lr": 0.001}, {"params": self.model.classifier.parameters(), "lr": 0.01}],
                weight_decay=5e-4, momentum=0.9, nesterov=True,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.1)
            self.criterion = nn.CrossEntropyLoss()
            self.input_size = 256
            self.epochs = 120
            self.description = "University-1652 official ResNet50 shared two-view CE recipe"

        elif self.method == "sues200":
            self._front(self.repo)
            sm = importlib.import_module("model_")
            # Current official SUES-200 settings.yaml selects the ViT baseline.
            self.model = sm.ViT(self.nclasses, 0.5, share_weight=False, pretrained=True).to(self.device)
            ignored = set(map(id, self.model.classifier.parameters()))
            base = [p for p in self.model.parameters() if id(p) not in ignored]
            self.optimizer = torch.optim.SGD(
                [{"params": base, "lr": 1e-4}, {"params": self.model.classifier.parameters(), "lr": 1e-3}],
                weight_decay=1e-4, momentum=0.9, nesterov=True,
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.1)
            self.criterion = nn.CrossEntropyLoss()
            self.input_size = 384
            self.epochs = 60
            self.description = "SUES-200 official ViT two-branch CE recipe"

        elif self.method == "denseuav":
            self._front(self.repo)
            taskflow = importlib.import_module("models.taskflow")
            lossmod = importlib.import_module("losses.loss")
            optmod = importlib.import_module("optimizers.make_optimizer")
            opt = SimpleNamespace(
                nclasses=self.nclasses, droprate=0.5, backbone="ViTS-224",
                head="SingleBranch", head_pool="avg", num_bottleneck=512,
                load_from="no", h=224, w=224, block=1,
                cls_loss="CELoss", feature_loss="WeightedSoftTripletLoss",
                kl_loss="KLLoss", sample_num=1, batchsize=self.batch, lr=0.01,
            )
            self.dense_opt = opt
            self.model = taskflow.make_model(opt).to(self.device)
            self.criterion = lossmod.Loss(opt).to(self.device)
            self.optimizer, self.scheduler = optmod.make_optimizer(self.model, opt)
            self.input_size = 224
            self.epochs = 120
            self.description = "DenseUAV official ViT-S/SingleBranch CE+WeightedSoftTriplet+KL recipe"

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
            self.description = "GTA-UAV/Game4Loc official shared ViT standard InfoNCE recipe"
        else:
            raise ValueError(self.method)

    def config(self) -> dict:
        return {
            "method": self.method,
            "description": self.description,
            "epochs": self.epochs,
            "batch": self.batch,
            "input_size": self.input_size,
            "nclasses": self.nclasses,
        }

    def prepare_scheduler(self, steps_per_epoch: int) -> None:
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
            z = self.model.classifier.add_block(branch(x))
        elif self.method == "sues200":
            branch = self.model.model_1 if view == "sat" else self.model.model_2
            z = self.model.classifier.add_block(branch(x))
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


def _train(adapter: Adapter, cache: Path, checkpoint: Path, fingerprint: str, force: bool, seed: int) -> None:
    if checkpoint.exists() and not force:
        ckpt = torch.load(checkpoint, map_location="cpu")
        if ckpt.get("fingerprint") == fingerprint:
            adapter.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"[M2T] checkpoint cache hit: {checkpoint}", flush=True)
            return

    uav = np.load(cache / "train_uav.npy", mmap_mode="r")
    gallery = np.load(cache / "tile_gallery_sat.npy", mmap_mode="r")
    tile_idx = np.load(cache / "train_tile_index.npy")
    labels = np.load(cache / "train_class_index.npy")
    if not (len(uav) == len(tile_idx) == len(labels)):
        raise RuntimeError("training cache length mismatch")
    if len(np.unique(labels)) != adapter.nclasses:
        raise RuntimeError("training class count mismatch")

    rng = np.random.default_rng(seed)
    if adapter.method == "gtauav":
        steps = max(1, int(math.ceil(adapter.nclasses / adapter.batch)))
    else:
        steps = max(1, int(math.ceil(len(uav) / adapter.batch)))
    adapter.prepare_scheduler(steps)
    scaler = torch.cuda.amp.GradScaler(enabled=adapter.amp)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, adapter.epochs + 1):
        adapter.model.train()
        batches = (
            _unique_class_batches(labels, adapter.batch, rng)
            if adapter.method == "gtauav"
            else _normal_batches(len(uav), adapter.batch, rng)
        )
        total, seen = 0.0, 0
        for idx in batches:
            sat_idx = tile_idx[idx]
            xu = _prep(_raw_batch(uav, idx, adapter.device), adapter.input_size, adapter.mean, adapter.std, train=True, satellite=False)
            xs = _prep(_raw_batch(gallery, sat_idx, adapter.device), adapter.input_size, adapter.mean, adapter.std, train=True, satellite=True)
            y = torch.as_tensor(labels[idx], device=adapter.device, dtype=torch.long)
            adapter.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=adapter.amp):
                loss = adapter.train_step(xs, xu, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{adapter.method}: non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(adapter.optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 100.0)
            scaler.step(adapter.optimizer)
            scaler.update()
            if adapter.scheduler_per_step and adapter.scheduler is not None:
                adapter.scheduler.step()
            total += float(loss.detach().item()) * len(idx)
            seen += len(idx)
        if seen == 0:
            raise RuntimeError(f"{adapter.method}: no usable training batch")
        if not adapter.scheduler_per_step and adapter.scheduler is not None:
            adapter.scheduler.step()
        avg = total / seen
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == adapter.epochs:
            print(f"[{adapter.method}] epoch={epoch:03d}/{adapter.epochs} loss={avg:.6f}", flush=True)

    if best_state is not None:
        adapter.model.load_state_dict(best_state, strict=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fingerprint": fingerprint,
        "model_state_dict": adapter.model.state_dict(),
        "config": adapter.config(),
        "best_train_loss": best_loss,
    }, checkpoint)
    print(f"[M2T] saved {checkpoint}", flush=True)


def _features(adapter: Adapter, arr: np.ndarray, view: str, batch: int) -> torch.Tensor:
    adapter.model.eval()
    out = []
    with torch.inference_mode():
        for idx in _sequential_batches(len(arr), batch):
            x = _prep(_raw_batch(arr, idx, adapter.device), adapter.input_size, adapter.mean, adapter.std, train=False, satellite=(view == "sat"))
            with torch.cuda.amp.autocast(enabled=adapter.amp):
                z = adapter.embed(x, view)
            out.append(z.detach())
    return torch.cat(out, dim=0)


def _metrics(pred: np.ndarray, gt: np.ndarray, top1: np.ndarray, true_tile: np.ndarray) -> dict:
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "frames": int(len(err)),
        "Recall@1_pct": float(100.0 * np.mean(top1 == true_tile)),
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
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        ux, uy = dx / length, dy / length
        s = 0.0
        while s < length:
            e = min(length, s + dash)
            draw.line((a[0] + ux*s, a[1] + uy*s, a[0] + ux*e, a[1] + uy*e), fill=fill, width=width)
            s += dash + gap


def _figure(prepared: Path, route: str, pred_m: np.ndarray, gt_m: np.ndarray, metrics: dict, out: Path, method_name: str) -> None:
    sm = _json(prepared / "bearing_satellite.json")
    mpp = float(sm["mpp"])
    src = Image.open(sm["satellite_image"]).convert("RGB")
    base = ImageEnhance.Brightness(src).enhance(0.82).convert("RGBA")
    pred = [(float(x/mpp), float(y/mpp)) for x, y in pred_m]
    gt = [(float(x/mpp), float(y/mpp)) for x, y in gt_m]
    ref = _waypoints(prepared, route)

    allp = pred + gt + ref
    xs, ys = [p[0] for p in allp], [p[1] for p in allp]
    margin = max(220, int(0.05 * max(max(xs)-min(xs), max(ys)-min(ys), 1)))
    box = (
        max(0, int(min(xs))-margin), max(0, int(min(ys))-margin),
        min(base.width, int(max(xs))+margin), min(base.height, int(max(ys))+margin),
    )

    draw = ImageDraw.Draw(base, "RGBA")
    # Planned route is visual context ONLY, never model input.
    _dashed(draw, ref, (255,255,255,120), 7, 18, 16)
    _dashed(draw, ref, REF, 4, 18, 16)
    # Actual selected-frame GT and raw baseline prediction.
    _dashed(draw, gt, HALO, 22, 38, 18)
    _dashed(draw, gt, GT, 14, 38, 18)
    draw.line(pred, fill=HALO, width=25, joint="curve")
    draw.line(pred, fill=PRED, width=17, joint="curve")
    r = 14
    for p, c in ((gt[0],GT), (gt[-1],GT), (pred[0],PRED), (pred[-1],PRED)):
        draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r), fill=c, outline=HALO, width=4)

    crop = base.crop(box)
    overlay = Image.new("RGBA", crop.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay, "RGBA")
    tf, bf = _font(27, True), _font(21, False)
    lines = [
        f"{method_name} | {route}",
        f"R@1 {metrics['Recall@1_pct']:.1f}%   MLE {metrics['MLE_m']:.2f} m   LSR@15 {metrics['LSR@15_pct']:.1f}%",
        "Purple dashed = per-frame GT    Red solid = raw prediction",
        "Gray dotted = route display only (NOT an input to this baseline)",
    ]
    pad, lh = 22, 34
    bw = min(max(500, crop.width-2*pad), 900)
    bh = pad*2 + lh*len(lines)
    od.rounded_rectangle((pad,pad,pad+bw,pad+bh), radius=14, fill=(0,0,0,190), outline=(255,255,255,190), width=2)
    for i, text in enumerate(lines):
        od.text((pad+18,pad+15+i*lh), text, font=tf if i==0 else bf, fill=(255,255,255,255))
    crop.alpha_composite(overlay)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(out, quality=96, subsampling=0)


def _evaluate(adapter: Adapter, cache: Path, prepared: Path, outdir: Path) -> dict:
    gallery = np.load(cache / "tile_gallery_sat.npy", mmap_mode="r")
    gallery_xy = np.load(cache / "tile_gallery_xy_m.npy")
    gf = _features(adapter, gallery, "sat", max(adapter.batch, 32))
    result = {}
    for route in ("test_01", "test_02"):
        q = np.load(cache / f"{route}_uav.npy", mmap_mode="r")
        gt = np.load(cache / f"{route}_gt_m.npy")
        true_tile = np.load(cache / f"{route}_tile_index.npy")
        qf = _features(adapter, q, "uav", max(adapter.batch, 16))
        sim = qf @ gf.T
        top1 = torch.argmax(sim, dim=1).detach().cpu().numpy()
        pred = gallery_xy[top1]
        met = _metrics(pred, gt, top1, true_tile)
        result[route] = met

        with (outdir / f"{route}_frames.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_id","gt_x_m","gt_y_m","pred_x_m","pred_y_m","gt_tile","pred_tile","error_m"])
            for i, (gg, pp, gt_t, pr_t, e) in enumerate(zip(gt, pred, true_tile, top1, met["distance_errors_m"])):
                w.writerow([i,float(gg[0]),float(gg[1]),float(pp[0]),float(pp[1]),int(gt_t),int(pr_t),float(e)])
        _figure(prepared, route, pred, gt, met, outdir / f"{route}_final_result.jpg", adapter.method)
        print(
            f"[{adapter.method}] {route}: R1={met['Recall@1_pct']:.2f}% "
            f"MLE={met['MLE_m']:.3f}m MedLE={met['MedLE_m']:.3f}m LSR15={met['LSR@15_pct']:.2f}%",
            flush=True,
        )
        del qf, sim
        torch.cuda.empty_cache()
    del gf
    return result


def main() -> None:
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
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.benchmark = True
    _seed(args.seed)

    cache = Path(args.cache_dir).resolve()
    prepared = Path(args.prepared_root).resolve()
    outdir = Path(args.output_root).resolve() / args.method / args.city
    outdir.mkdir(parents=True, exist_ok=True)
    cache_meta = _json(cache / "cache_meta.json")
    nclasses = int(cache_meta["train_class_count"])
    if nclasses < 2:
        raise RuntimeError(f"{args.city}: fewer than 2 training RST classes")
    default_batch = {"university1652": 16, "sues200": 12, "denseuav": 16, "gtauav": 32}[args.method]
    batch = args.batch_size if args.batch_size > 0 else min(default_batch, int(cache_meta["train_frames"]))
    batch = max(2, batch)

    adapter = Adapter(args.method, Path(args.repo_dir).resolve(), nclasses, _device(), batch)
    fingerprint = _sha({
        "cache_fingerprint": cache_meta["fingerprint"],
        "repo_commit": args.repo_commit,
        "adapter": adapter.config(),
        "seed": args.seed,
        "runner_version": RUNNER_VERSION,
    })
    checkpoint = outdir / "checkpoint.pt"
    _train(adapter, cache, checkpoint, fingerprint, args.force, args.seed)
    adapter.model.eval()

    routes = _evaluate(adapter, cache, prepared, outdir)
    allerr = np.asarray(routes["test_01"]["distance_errors_m"] + routes["test_02"]["distance_errors_m"], dtype=np.float64)
    total_frames = routes["test_01"]["frames"] + routes["test_02"]["frames"]
    recall_hits = (
        routes["test_01"]["Recall@1_pct"] * routes["test_01"]["frames"] +
        routes["test_02"]["Recall@1_pct"] * routes["test_02"]["frames"]
    ) / max(total_frames, 1)
    pooled = {
        "frames": int(len(allerr)),
        "Recall@1_pct": float(recall_hits),
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
        "kind": "native matching-to-tile route-adapted reproduction using public official implementation/objective",
        "repo_commit": args.repo_commit,
        "training_scope": cache_meta["training_policy"],
        "test_scope": "same selected test_01/test_02 UAV frames as our experiment",
        "gallery_scope": cache_meta["gallery_policy"],
        "uses_waypoint_or_route_prior": False,
        "uses_temporal_history": False,
        "endpoint_completion_required": False,
        "published_protocol_equivalent": False,
        "protocol_note": (
            "The public method is retrained on selected Route-A observations but localizes each test frame independently "
            "against all 256 city RST tiles. A bad/looping/drifting trajectory is retained as-is."
        ),
        "config": adapter.config(),
        "cache_fingerprint": cache_meta["fingerprint"],
        "routes": routes,
        "aggregate_two_routes": pooled,
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[M2T-DONE] {args.method}/{args.city}: R1={pooled['Recall@1_pct']:.2f}% "
        f"MLE={pooled['MLE_m']:.3f}m LSR15={pooled['LSR@15_pct']:.2f}%",
        flush=True,
    )
    del adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
