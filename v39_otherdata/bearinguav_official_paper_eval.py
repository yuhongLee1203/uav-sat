#!/usr/bin/env python3
"""Verify the released Bearing-UAV VGG-16 checkpoint on the paper test split.

This is intentionally separate from our pseudo-flight route experiment.  The
public Bearing-UAV code splits the full metadata 85/5/10 with seed 42.  Running
the released checkpoint on that exact test split lets us check whether our local
dataset/environment reproduces the paper-scale localization numbers before any
route comparison is interpreted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split

import bearing_prepare as bearing
from bearinguav_official_route_eval import _load_model, _resolve

PUBLISHED = {
    "Recall@1_pct": 83.17,
    "LSR@15_pct": 89.36,
    "HSR@15_pct": 77.21,
    "MLE_m": 8.61,
    "MedLE_m": 7.30,
    "MHE_deg": 12.90,
    "MedHE_deg": 7.20,
}


def _prepare_test_csv(dataset_root: Path, metadata_path: Path, out_csv: Path) -> tuple[int, int]:
    df = pd.read_csv(metadata_path)
    n = len(df)
    tr = int(0.85 * n); va = int(0.05 * n); te = n - tr - va
    _, _, test_indices = random_split(range(n), [tr, va, te], generator=torch.Generator().manual_seed(42))
    test = df.iloc[list(test_indices.indices)].copy().reset_index(drop=True)
    for col in ("p1_path", "p2_path", "p3_path", "p4_path", "target_path"):
        if col not in test.columns:
            raise RuntimeError(f"official metadata missing {col}")
        test[col] = [_resolve(v, dataset_root) for v in test[col].tolist()]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_csv, index=False)
    return n, len(test)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu-threads", type=int, default=2)
    a = p.parse_args()

    torch.set_num_threads(max(1, a.cpu_threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(a.official_root).resolve(); weights = Path(a.weights_dir).resolve()
    data = Path(a.dataset_root).resolve(); out = Path(a.output).resolve()
    metadata = bearing._find_metadata(data)
    csv_path = out.parent / "official_paper_test_resolved.csv"
    total_n, test_n = _prepare_test_csv(data, metadata, csv_path)

    model, dscls, cls_name, model_kw = _load_model(root, weights, device)
    ds = dscls(str(csv_path), is_train=False)
    kw = dict(batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
              pin_memory=torch.cuda.is_available(), drop_last=False)
    if a.workers > 0: kw.update(persistent_workers=True, prefetch_factor=1)
    loader = DataLoader(ds, **kw)

    errs = []; heads = []; recalls = []
    with torch.inference_mode():
        for b in loader:
            x = b["patches"].to(device, non_blocking=True)
            pos, hd = model(x)
            pp = pos.detach().cpu().numpy().astype(np.float64)
            gp = b["coords"].numpy().astype(np.float64)
            pd = hd.detach().cpu().numpy().astype(np.float64)
            gd = b["agl_coords"].numpy().astype(np.float64)
            blocks = b["block_xy"].numpy().astype(np.float64)

            pred_px = blocks * bearing.PATCH_SIZE + bearing.PATCH_SIZE + pp * bearing.PATCH_SIZE
            gt_px = blocks * bearing.PATCH_SIZE + bearing.PATCH_SIZE + gp * bearing.PATCH_SIZE
            err = np.linalg.norm((pred_px - gt_px) * bearing.MPP, axis=1)
            recall = np.all(np.sign(pp) == np.sign(gp), axis=1)
            pn = pd / np.maximum(np.linalg.norm(pd, axis=1, keepdims=True), 1e-12)
            gn = gd / np.maximum(np.linalg.norm(gd, axis=1, keepdims=True), 1e-12)
            he = np.degrees(np.arccos(np.clip(np.sum(pn * gn, axis=1), -1.0, 1.0)))
            errs.extend(err.tolist()); heads.extend(he.tolist()); recalls.extend(recall.tolist())

    e = np.asarray(errs); h = np.asarray(heads); r = np.asarray(recalls, dtype=bool)
    metrics = {
        "frames": int(len(e)),
        "Recall@1_pct": float(100*np.mean(r)),
        "LSR@15_pct": float(100*np.mean(e <= 15.0)),
        "HSR@15_pct": float(100*np.mean(h <= 15.0)),
        "MLE_m": float(np.mean(e)),
        "MedLE_m": float(np.median(e)),
        "MHE_deg": float(np.mean(h)),
        "MedHE_deg": float(np.median(h)),
    }
    delta = {k: float(metrics[k] - v) for k, v in PUBLISHED.items()}
    # These are verification bands, not optimization targets.  A failure means
    # dataset/checkpoint/code mismatch should be investigated rather than tuned away.
    checks = {
        "MLE_within_3m": abs(delta["MLE_m"]) <= 3.0,
        "Recall_within_10pp": abs(delta["Recall@1_pct"]) <= 10.0,
        "LSR15_within_10pp": abs(delta["LSR@15_pct"]) <= 10.0,
    }
    payload = {
        "protocol": "official full metadata 85/5/10 split, seed=42, released VGG-16 checkpoint",
        "metadata_rows": total_n, "test_rows": test_n, "model_class": cls_name,
        "model_kwargs": model_kw, "measured": metrics, "published": PUBLISHED,
        "measured_minus_published": delta, "sanity_checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[PAPER-VERIFY] measured:", metrics, flush=True)
    print("[PAPER-VERIFY] published:", PUBLISHED, flush=True)
    print("[PAPER-VERIFY] checks:", checks, flush=True)
    if all(checks.values()):
        print("[PAPER-VERIFY] PASS: official Bearing-UAV checkpoint is in the expected paper-scale range", flush=True)
    else:
        print("[PAPER-VERIFY] WARNING: official paper-scale mismatch; inspect dataset/checkpoint before claiming reproduction", flush=True)


if __name__ == "__main__":
    main()
