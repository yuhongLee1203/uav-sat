#!/usr/bin/env python3
"""Retrain the public Bearing-UAV architecture on our selected Route-A only.

This is deliberately separate from beariguav_official_route_eval.py:

* route-adapted: official PARCASGM_v5a architecture/objective, trained only on
  this city's selected train_01 rows, then evaluated on selected test_01/02;
* official-pretrained: authors' released full-benchmark checkpoint, evaluated on
  the same selected test rows.

For low CPU/I/O load, the five input images per selected sample are decoded once
into mmap-able uint8 .npy arrays.  All later epochs normalize and apply mild
"gentle" augmentation on the GPU with zero DataLoader workers.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bearing_prepare as bearing
from bearing_route_baseline_runner import _figure

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PATCH = 256.0


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_path(raw: object, dataset_root: Path) -> Path:
    text = str(raw).replace("\\", "/")
    p = Path(text)
    if p.exists():
        return p.resolve()
    marker = "Bearing_UAV_90K/"
    if marker in text:
        q = dataset_root / text.split(marker, 1)[1]
        if q.exists():
            return q.resolve()
    for anchor in (
        "citya", "cityb", "cityc", "cityd", "city_rsi",
        "c4m_254k_96bc_b15_s100_v3d", "c4m_254k_96bc_b15_s100",
        "c1_254k_96bc_b15_s1_v3d", "c1_254k_96bc_b15_s1",
        "c1_254k_37bc_b15_s1_v3d",
    ):
        token = f"/{anchor}/"
        if token in text:
            q = dataset_root / anchor / text.split(token, 1)[1]
            if q.exists():
                return q.resolve()
    raise FileNotFoundError(f"cannot resolve Bearing path: {raw}")


def _u8(path: Path, size=256) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _fingerprint(prepared: Path, city: str, repo_commit: str) -> str:
    h = hashlib.sha256()
    for route in ("train_01", "test_01", "test_02"):
        p = prepared / "routes" / route / "manifest.csv"
        h.update(p.read_bytes())
    h.update(city.encode())
    h.update(repo_commit.encode())
    h.update(b"bearing_route_adapted_v1")
    return h.hexdigest()


def _build_native_cache(
    dataset_root: Path,
    prepared: Path,
    city: str,
    cache_dir: Path,
    repo_commit: str,
    force: bool,
) -> Tuple[str, Dict[str, Path]]:
    fp = _fingerprint(prepared, city, repo_commit)
    meta_path = cache_dir / "meta.json"
    names = {}
    for route in ("train_01", "test_01", "test_02"):
        for kind in ("patches", "coords", "agl", "blocks", "gt_m"):
            names[f"{route}_{kind}"] = cache_dir / f"{route}_{kind}.npy"
    if not force and meta_path.exists():
        old = _json(meta_path)
        if old.get("fingerprint") == fp and all(p.exists() for p in names.values()):
            print(f"[BEARING-ROUTE-CACHE] hit {city}: {cache_dir}", flush=True)
            return fp, names

    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(bearing._find_metadata(dataset_root))
    city_rows = bearing._city_rows(metadata, city)
    cols = ["p1_path", "p2_path", "p3_path", "p4_path", "target_path"]

    route_counts = {}
    for route in ("train_01", "test_01", "test_02"):
        manifest = _rows(prepared / "routes" / route / "manifest.csv")
        n = len(manifest)
        if n == 0:
            raise RuntimeError(f"{city}/{route}: empty manifest")
        route_counts[route] = n
        patches = np.lib.format.open_memmap(
            names[f"{route}_patches"], mode="w+", dtype=np.uint8,
            shape=(n, 5, 256, 256, 3),
        )
        coords = np.empty((n, 2), dtype=np.float32)
        agl = np.empty((n, 2), dtype=np.float32)
        blocks = np.empty((n, 2), dtype=np.float32)
        gt_m = np.empty((n, 2), dtype=np.float32)

        for i, frame in enumerate(manifest):
            idx = int(frame["source_index"])
            if idx < 0 or idx >= len(city_rows):
                raise RuntimeError(f"{city}/{route}: bad source_index={idx}")
            row = city_rows.iloc[idx]
            for j, col in enumerate(cols):
                patches[i, j] = _u8(_resolve_path(row[col], dataset_root))
            coords[i] = [float(row["x_norm"]), float(row["y_norm"])]
            agl[i] = [float(row["x_cosa"]), float(row["y_sina"])]
            blocks[i] = [float(row["block_x"]), float(row["block_y"])]
            gt_m[i] = [float(frame["x_m"]), float(frame["y_m"])]
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"[BEARING-ROUTE-CACHE] {city}/{route}: {i+1}/{n}", flush=True)
        patches.flush(); del patches
        np.save(names[f"{route}_coords"], coords)
        np.save(names[f"{route}_agl"], agl)
        np.save(names[f"{route}_blocks"], blocks)
        np.save(names[f"{route}_gt_m"], gt_m)

    meta_path.write_text(json.dumps({
        "fingerprint": fp,
        "city": city,
        "repo_commit": repo_commit,
        "routes": route_counts,
        "policy": "selected train/test source rows only; image decode once then mmap",
    }, indent=2), encoding="utf-8")
    return fp, names


def _batch(raw: np.ndarray, idx: np.ndarray, device: torch.device, train: bool) -> torch.Tensor:
    # B,5,H,W,C -> B,5,C,H,W
    x = torch.from_numpy(np.asarray(raw[idx], dtype=np.uint8)).to(device, non_blocking=True)
    x = x.permute(0, 1, 4, 2, 3).contiguous().float().div_(255.0)
    if train:
        b, v = x.shape[:2]
        flat = x.view(b * v, 3, x.shape[-2], x.shape[-1])
        # GPU counterpart of the official gentle augmentation.  Parameters and
        # probability are intentionally small; labels and geometry are untouched.
        mask = torch.rand((b*v, 1, 1, 1), device=device) < 0.10
        brightness = torch.empty((b*v,1,1,1), device=device).uniform_(0.85,1.15)
        contrast = torch.empty((b*v,1,1,1), device=device).uniform_(0.85,1.15)
        mean = flat.mean(dim=(2,3), keepdim=True)
        jitter = ((flat - mean) * contrast + mean) * brightness
        flat = torch.where(mask, jitter.clamp(0,1), flat)
        nmask = torch.rand((b*v,1,1,1), device=device) < 0.10
        noise = torch.randn_like(flat) * 0.015
        flat = torch.where(nmask, (flat + noise).clamp(0,1), flat)
        x = flat.view(b, v, 3, x.shape[-2], x.shape[-1])
    mean = torch.tensor(MEAN, device=device).view(1,1,3,1,1)
    std = torch.tensor(STD, device=device).view(1,1,3,1,1)
    return (x - mean) / std


def _indices(n: int, batch: int, rng: np.random.Generator, shuffle=True):
    idx = np.arange(n)
    if shuffle: rng.shuffle(idx)
    for s in range(0, n, batch):
        yield idx[s:s+batch]


def _load_official_model(official_root: Path, device: torch.device):
    p = str(official_root.resolve())
    if p in sys.path: sys.path.remove(p)
    sys.path.insert(0, p)
    from cvphr.models.posaglreg import models as bm
    model = bm.PARCASGM_v5a(**dict(bm.model_kwargs_par_ca_sgm_v5a)).to(device)
    return model, bm.model_kwargs_par_ca_sgm_v5a


def _train(
    model: nn.Module,
    names: Dict[str, Path],
    checkpoint: Path,
    fingerprint: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    force: bool,
    seed: int,
):
    if checkpoint.exists() and not force:
        ck = torch.load(checkpoint, map_location="cpu")
        if ck.get("fingerprint") == fingerprint:
            model.load_state_dict(ck["model_state_dict"], strict=True)
            print(f"[BEARING-ROUTE] checkpoint cache hit: {checkpoint}", flush=True)
            return

    raw = np.load(names["train_01_patches"], mmap_mode="r")
    coords = np.load(names["train_01_coords"])
    agl = np.load(names["train_01_agl"])
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps_epoch = max(1, math.ceil(len(raw) / batch_size))
    total_steps = max(1, steps_epoch * epochs)
    warm = max(1, int(0.1 * total_steps))
    def lam(step):
        if step < warm: return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return max(1e-2, 0.5 * (1 + math.cos(math.pi * min(max(p,0),1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best = None

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; seen = 0
        for idx in _indices(len(raw), batch_size, rng, True):
            x = _batch(raw, idx, device, True)
            y_pos = torch.from_numpy(coords[idx]).to(device)
            y_dir = torch.from_numpy(agl[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                ppos, pdir = model(x)
                lp = criterion(ppos, y_pos)
                ld = criterion(pdir, y_dir)
                loss = 0.8 * lp + 0.2 * ld
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total += float(loss.detach()) * len(idx); seen += len(idx)
        avg = total / max(seen,1)
        if avg < best_loss:
            best_loss = avg
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[Bearing-UAV-route] epoch={epoch:03d}/{epochs} loss={avg:.6f} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
    if best is not None:
        model.load_state_dict(best, strict=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint":fingerprint,"model_state_dict":model.state_dict(),"best_train_loss":best_loss}, checkpoint)


def _evaluate_route(model, names, route: str, device, batch_size: int, mpp: float):
    raw = np.load(names[f"{route}_patches"], mmap_mode="r")
    coords = np.load(names[f"{route}_coords"])
    agl = np.load(names[f"{route}_agl"])
    blocks = np.load(names[f"{route}_blocks"])
    gt_m = np.load(names[f"{route}_gt_m"])
    rng = np.random.default_rng(0)
    pos_all=[]; dir_all=[]
    model.eval()
    with torch.inference_mode():
        for idx in _indices(len(raw), batch_size, rng, False):
            x = _batch(raw, idx, device, False)
            with torch.cuda.amp.autocast(enabled=True):
                p, d = model(x)
            pos_all.append(p.float().cpu().numpy()); dir_all.append(d.float().cpu().numpy())
    pos = np.concatenate(pos_all).astype(np.float64)
    direc = np.concatenate(dir_all).astype(np.float64)

    # Bearing-UAV official coordinate contract: normalized offset within one
    # 256-px block, with the four-RST centre at block*256+256.
    gt_from_meta_px = blocks * PATCH + PATCH + coords * PATCH
    gt_from_meta_m = gt_from_meta_px * mpp
    coord_err = np.linalg.norm(gt_from_meta_m - gt_m, axis=1)
    if float(coord_err.max()) > 1e-2:
        raise RuntimeError(f"{route}: metadata/global GT contract mismatch max={coord_err.max():.6f}m")
    pred_px = blocks * PATCH + PATCH + pos * PATCH
    pred_m = pred_px * mpp
    err = np.linalg.norm(pred_m - gt_m, axis=1)

    recall = np.all(np.sign(pos) == np.sign(coords), axis=1)
    dn = direc / np.maximum(np.linalg.norm(direc,axis=1,keepdims=True),1e-12)
    gn = agl / np.maximum(np.linalg.norm(agl,axis=1,keepdims=True),1e-12)
    h = np.degrees(np.arccos(np.clip(np.sum(dn*gn,axis=1),-1,1)))
    metrics = {
        "frames": int(len(err)),
        "Recall@1_pct": float(100*np.mean(recall)),
        "MLE_m": float(err.mean()), "MedLE_m": float(np.median(err)),
        "P90_m": float(np.percentile(err,90)), "P95_m": float(np.percentile(err,95)), "P99_m": float(np.percentile(err,99)),
        "LSR@5_pct": float(100*np.mean(err<=5)), "LSR@10_pct": float(100*np.mean(err<=10)),
        "LSR@15_pct": float(100*np.mean(err<=15)), "LSR@20_pct": float(100*np.mean(err<=20)),
        "HSR@15_pct": float(100*np.mean(h<=15)), "MHE_deg": float(h.mean()), "MedHE_deg": float(np.median(h)),
        "distance_errors_m": err.tolist(), "heading_errors_deg": h.tolist(),
    }
    return pred_m.astype(np.float32), gt_m.astype(np.float32), metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--repo-commit", default="unknown")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    args=p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1,args.cpu_threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda:0"); torch.backends.cudnn.benchmark=True

    prepared=Path(args.prepared_root).resolve()
    dataset_root=Path(args.dataset_root).resolve()
    cache_dir=Path(args.cache_root).resolve()/"bearinguav_native"
    outdir=Path(args.output_root).resolve()/"bearinguav_route_adapted"/args.city
    outdir.mkdir(parents=True, exist_ok=True)
    fp,names=_build_native_cache(dataset_root,prepared,args.city,cache_dir,args.repo_commit,args.force)

    model, kwargs=_load_official_model(Path(args.official_root),device)
    train_fp=hashlib.sha256((fp+json.dumps({"epochs":args.epochs,"batch":args.batch_size,"seed":args.seed},sort_keys=True)).encode()).hexdigest()
    _train(model,names,outdir/"checkpoint.pt",train_fp,device,args.epochs,args.batch_size,args.force,args.seed)

    mpp=float(_json(prepared/"bearing_satellite.json")["mpp"])
    routes={}; allerr=[]; allhead=[]; total_recall=0.0
    for route in ("test_01","test_02"):
        pred,gt,met=_evaluate_route(model,names,route,device,args.batch_size,mpp)
        routes[route]=met; allerr.extend(met["distance_errors_m"]); allhead.extend(met["heading_errors_deg"])
        total_recall += met["Recall@1_pct"]*met["frames"]/100
        _figure(prepared,route,pred,gt,met,outdir/f"{route}_final_result.jpg","Bearing-UAV route-adapted")
        with (outdir/f"{route}_frames.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["frame_id","gt_x_m","gt_y_m","pred_x_m","pred_y_m","error_m"])
            for i,(g,q,e) in enumerate(zip(gt,pred,met["distance_errors_m"])): w.writerow([i,*map(float,g),*map(float,q),float(e)])
        print(f"[BEARING-ROUTE] {args.city}/{route}: MLE={met['MLE_m']:.3f}m LSR15={met['LSR@15_pct']:.2f}%",flush=True)

    e=np.asarray(allerr); h=np.asarray(allhead)
    aggregate={
        "frames":int(len(e)),"Recall@1_pct":float(100*total_recall/max(len(e),1)),
        "MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"P90_m":float(np.percentile(e,90)),
        "LSR@5_pct":float(100*np.mean(e<=5)),"LSR@10_pct":float(100*np.mean(e<=10)),
        "LSR@15_pct":float(100*np.mean(e<=15)),"LSR@20_pct":float(100*np.mean(e<=20)),
        "HSR@15_pct":float(100*np.mean(h<=15)),"MHE_deg":float(h.mean()),"MedHE_deg":float(np.median(h)),
    }
    payload={
        "method":"Bearing-UAV route-adapted","city":args.city,
        "kind":"route-adapted reproduction using official PARCASGM_v5a architecture/objective",
        "repo_commit":args.repo_commit,"model_kwargs":kwargs,
        "training_scope":"selected train_01 only","test_scope":"same selected test_01/test_02 rows as ours",
        "published_protocol_equivalent":False,
        "protocol_note":"trained from ImageNet-initialized official architecture on Route-A only; separate from authors' released full-dataset pretrained checkpoint",
        "routes":routes,"aggregate_two_routes":aggregate,
    }
    (outdir/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"[BEARING-ROUTE-DONE] {args.city}: MLE={aggregate['MLE_m']:.3f}m LSR15={aggregate['LSR@15_pct']:.2f}%",flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__=="__main__":
    main()
