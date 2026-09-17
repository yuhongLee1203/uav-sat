#!/usr/bin/env python3
"""Run public M2T methods with Bearing-UAV's native FOUR-adjacent-RST protocol.

The old adapter incorrectly searched all 256 city tiles.  Bearing-UAV's
supplement defines Recall@1 over the four adjacent RSTs of the current RSB, and
M2T methods use the retrieved RST centre as their final location.  This runner
implements that definition exactly for the selected route frames.

No waypoint, route centreline, previous position, temporal state, Kalman state,
or v39 local prior enters baseline inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from bearing_route_baseline_runner import (
    Adapter, _device, _figure, _normal_batches, _prep, _raw_batch, _seed,
    _sequential_batches, _sha, _unique_class_batches,
)

RUNNER_VERSION = "bearing_native_four_rst_v1"
METHODS = ("university1652", "sues200", "denseuav", "gtauav")


def _balanced_positive_batches(labels: np.ndarray, batch: int, rng: np.random.Generator):
    """DenseUAV hard triplet mining needs >=2 samples/class and >=2 classes/batch."""
    classes = np.unique(labels)
    if len(classes) < 2:
        raise RuntimeError("DenseUAV needs at least two training RST classes")
    per_class = 2
    classes_per_batch = max(2, batch // per_class)
    steps = max(1, int(math.ceil(len(labels) / max(batch, 1))))
    for _ in range(steps):
        chosen = rng.choice(classes, size=min(classes_per_batch, len(classes)), replace=False)
        idx = []
        for c in chosen:
            pool = np.flatnonzero(labels == c)
            take = rng.choice(pool, size=per_class, replace=(len(pool) < per_class))
            idx.extend(int(v) for v in take)
        if len(idx) >= 4:
            yield np.asarray(idx, dtype=np.int64)


def _train(adapter: Adapter, cache: Path, checkpoint: Path, fingerprint: str, force: bool, seed: int):
    if checkpoint.exists() and not force:
        ck = torch.load(checkpoint, map_location="cpu")
        if ck.get("fingerprint") == fingerprint:
            adapter.model.load_state_dict(ck["model_state_dict"], strict=True)
            print(f"[4RST] checkpoint hit {checkpoint}", flush=True)
            return

    uav = np.load(cache / "train_uav.npy", mmap_mode="r")
    sat = np.load(cache / "train_positive_sat.npy", mmap_mode="r")
    labels = np.load(cache / "train_class_index.npy")
    if not (len(uav) == len(sat) == len(labels)):
        raise RuntimeError("4RST training cache length mismatch")
    rng = np.random.default_rng(seed)
    steps = max(1, int(math.ceil(len(labels) / adapter.batch)))
    adapter.prepare_scheduler(steps)
    scaler = torch.cuda.amp.GradScaler(enabled=adapter.amp)
    best_loss = float("inf"); best_state = None

    for epoch in range(1, adapter.epochs + 1):
        adapter.model.train()
        if adapter.method == "denseuav":
            batches = _balanced_positive_batches(labels, adapter.batch, rng)
        elif adapter.method == "gtauav":
            batches = _unique_class_batches(labels, adapter.batch, rng)
        else:
            batches = _normal_batches(len(labels), adapter.batch, rng)
        total = 0.0; seen = 0
        for idx in batches:
            xu = _prep(_raw_batch(uav, idx, adapter.device), adapter.input_size,
                       adapter.mean, adapter.std, train=True, satellite=False)
            xs = _prep(_raw_batch(sat, idx, adapter.device), adapter.input_size,
                       adapter.mean, adapter.std, train=True, satellite=True)
            y = torch.as_tensor(labels[idx], device=adapter.device, dtype=torch.long)
            adapter.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=adapter.amp):
                loss = adapter.train_step(xs, xu, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{adapter.method}: non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(adapter.optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 100.0)
            scaler.step(adapter.optimizer); scaler.update()
            if adapter.scheduler_per_step and adapter.scheduler is not None:
                adapter.scheduler.step()
            total += float(loss.detach()) * len(idx); seen += len(idx)
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
    torch.save({"fingerprint": fingerprint, "model_state_dict": adapter.model.state_dict(),
                "config": adapter.config(), "best_train_loss": best_loss}, checkpoint)
    print(f"[4RST] saved {checkpoint}", flush=True)


def _features(adapter: Adapter, arr: np.ndarray, view: str, batch: int) -> torch.Tensor:
    adapter.model.eval(); out = []
    with torch.inference_mode():
        for idx in _sequential_batches(len(arr), batch):
            x = _prep(_raw_batch(arr, idx, adapter.device), adapter.input_size,
                      adapter.mean, adapter.std, train=False, satellite=(view == "sat"))
            with torch.cuda.amp.autocast(enabled=adapter.amp):
                z = adapter.embed(x, view)
            out.append(z.detach())
    return torch.cat(out, dim=0)


def _metrics(pred: np.ndarray, gt: np.ndarray, top1: np.ndarray, true_idx: np.ndarray):
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "frames": int(len(err)),
        "Recall@1_pct": float(100.0 * np.mean(top1 == true_idx)),
        "MLE_m": float(np.mean(err)),
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


def _evaluate(adapter: Adapter, cache: Path, prepared: Path, outdir: Path):
    results = {}
    for route in ("test_01", "test_02"):
        q = np.load(cache / f"{route}_uav.npy", mmap_mode="r")
        cand = np.load(cache / f"{route}_candidates.npy", mmap_mode="r")
        cxy = np.load(cache / f"{route}_candidate_xy_m.npy")
        gt = np.load(cache / f"{route}_gt_m.npy")
        true_idx = np.load(cache / f"{route}_true_index.npy")
        n = len(q)
        if cand.shape[:2] != (n, 4):
            raise RuntimeError(f"{route}: expected N x 4 RST candidates, got {cand.shape}")

        qf = _features(adapter, q, "uav", max(adapter.batch, 16))
        flat = cand.reshape(n * 4, *cand.shape[2:])
        sf = _features(adapter, flat, "sat", max(adapter.batch, 32)).reshape(n, 4, -1)
        sim = torch.sum(qf[:, None, :] * sf, dim=-1)
        top1 = torch.argmax(sim, dim=1).cpu().numpy().astype(np.int64)
        pred = cxy[np.arange(n), top1]
        met = _metrics(pred, gt, top1, true_idx)
        results[route] = met

        with (outdir / f"{route}_frames.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_id","gt_x_m","gt_y_m","pred_x_m","pred_y_m","true_candidate","pred_candidate","error_m"])
            for i, (g, p, t, r, e) in enumerate(zip(gt, pred, true_idx, top1, met["distance_errors_m"])):
                w.writerow([i, float(g[0]), float(g[1]), float(p[0]), float(p[1]), int(t), int(r), float(e)])
        _figure(prepared, route, pred, gt, met, outdir / f"{route}_final_result.jpg", adapter.method + " | four-RST")
        print(f"[4RST] {adapter.method}/{route}: R1={met['Recall@1_pct']:.2f}% MLE={met['MLE_m']:.3f}m LSR15={met['LSR@15_pct']:.2f}%", flush=True)
        del qf, sf, sim
        torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--repo-commit", default="unknown")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    torch.set_num_threads(max(1, a.cpu_threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.backends.cudnn.benchmark = True
    _seed(a.seed)
    cache = Path(a.cache_dir).resolve(); prepared = Path(a.prepared_root).resolve()
    outdir = Path(a.output_root).resolve() / a.method / a.city; outdir.mkdir(parents=True, exist_ok=True)
    cm = json.loads((cache / "cache_meta.json").read_text())
    if cm.get("candidate_policy", "").startswith("exact p1/p2/p3/p4") is False:
        raise RuntimeError("stale/wrong cache: not the four-RST protocol")
    nclasses = int(cm["train_class_count"])
    default_batch = {"university1652":16,"sues200":12,"denseuav":16,"gtauav":32}[a.method]
    batch = max(2, a.batch_size if a.batch_size > 0 else min(default_batch, int(cm["train_frames"])))
    adapter = Adapter(a.method, Path(a.repo_dir).resolve(), nclasses, _device(), batch)
    fp = _sha({"cache":cm["fingerprint"],"repo":a.repo_commit,"adapter":adapter.config(),"seed":a.seed,"runner":RUNNER_VERSION})
    _train(adapter, cache, outdir / "checkpoint.pt", fp, a.force, a.seed)
    routes = _evaluate(adapter, cache, prepared, outdir)

    allerr = np.asarray(routes["test_01"]["distance_errors_m"] + routes["test_02"]["distance_errors_m"], dtype=np.float64)
    n = sum(routes[r]["frames"] for r in ("test_01","test_02"))
    r1 = sum(routes[r]["Recall@1_pct"] * routes[r]["frames"] for r in ("test_01","test_02")) / max(n,1)
    pooled = {
        "frames": int(len(allerr)), "Recall@1_pct": float(r1), "MLE_m": float(allerr.mean()),
        "MedLE_m": float(np.median(allerr)), "P90_m": float(np.percentile(allerr,90)),
        "LSR@15_pct": float(100*np.mean(allerr<=15.0)),
    }
    published = {
        "university1652":{"Recall@1_pct":60.20,"MLE_m":33.15,"LSR@15_pct":15.11},
        "sues200":{"Recall@1_pct":66.60,"MLE_m":30.83,"LSR@15_pct":15.76},
        "denseuav":{"Recall@1_pct":73.43,"MLE_m":28.79,"LSR@15_pct":16.54},
        "gtauav":{"Recall@1_pct":70.71,"MLE_m":28.43,"LSR@15_pct":27.96},
    }[a.method]
    payload = {
        "method": a.method, "city": a.city,
        "kind": "route-adapted public-method reproduction with Bearing native four-RST M2T candidate protocol",
        "candidate_scope": "four adjacent p1/p2/p3/p4 RSTs from official metadata",
        "uses_waypoint_or_route_prior": False, "uses_temporal_history": False,
        "endpoint_completion_required": False,
        "training_scope": "selected train_01 observations only; therefore route results are not expected to numerically equal the paper full benchmark",
        "published_full_benchmark_reference": published,
        "routes": routes, "aggregate_two_routes": pooled,
        "cache_oracle": cm["routes"], "config": adapter.config(), "repo_commit": a.repo_commit,
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[4RST] DONE {a.method}/{a.city}: R1={r1:.2f}% MLE={pooled['MLE_m']:.3f}m", flush=True)


if __name__ == "__main__":
    main()
