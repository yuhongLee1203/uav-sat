#!/usr/bin/env python3
"""Evaluate the official Bearing-UAV VGG-16 checkpoint on OUR selected route frames.

This is the fairest directly-runnable baseline available from the public
Bearing-UAV release: it uses the official model code + official pretrained
checkpoint, but exactly the same Bearing-UAV image rows selected by our
city/test_01 and city/test_02 manifests.

Important protocol note:
- image frames / GT / MLE / MedLE / LSR thresholds are the same route subset;
- our v39 method is temporal controlled-local-prior refinement;
- official Bearing-UAV is a single-frame four-neighbour RST pose regressor.
So this is a much stronger same-data comparison, but still not an identical
algorithmic prior/protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bearing_prepare as bearing


def _read_manifest(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_dataset_path(raw: object, dataset_root: Path) -> str:
    text = str(raw).replace("\\", "/")
    p = Path(text)
    if p.exists():
        return str(p.resolve())

    marker = "Bearing_UAV_90K/"
    if marker in text:
        q = dataset_root / text.split(marker, 1)[1]
        if q.exists():
            return str(q.resolve())

    anchors = [
        "citya", "cityb", "cityc", "cityd", "city_rsi",
        "c4m_254k_96bc_b15_s100_v3d",
        "c4m_254k_96bc_b15_s100",
        "c1_254k_96bc_b15_s1_v3d",
        "c1_254k_96bc_b15_s1",
        "c1_254k_37bc_b15_s1_v3d",
    ]
    for anchor in anchors:
        token = f"/{anchor}/"
        if token in text:
            q = dataset_root / anchor / text.split(token, 1)[1]
            if q.exists():
                return str(q.resolve())

    raise FileNotFoundError(f"Cannot resolve Bearing-UAV path: {raw}")


def _route_metadata(
    dataset_root: Path,
    prepared_root: Path,
    city: str,
    route: str,
    city_rows: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    manifest = _read_manifest(prepared_root / "routes" / route / "manifest.csv")
    if not manifest:
        raise RuntimeError(f"{city}/{route}: empty manifest")

    selected = []
    for frame in manifest:
        idx = int(frame["source_index"])
        if idx < 0 or idx >= len(city_rows):
            raise RuntimeError(f"{city}/{route}: source_index out of range: {idx}")
        row = city_rows.iloc[idx].copy()
        # Make the official dataset class portable to the user's local dataset path.
        for col in ("p1_path", "p2_path", "p3_path", "p4_path", "target_path"):
            if col not in row.index:
                raise RuntimeError(f"metadata missing required column: {col}")
            row[col] = _resolve_dataset_path(row[col], dataset_root)
        row["__route"] = route
        row["__frame_id"] = int(frame["frame_id"])
        selected.append(row)

    df = pd.DataFrame(selected).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def _patch_torchvision_no_pretrained_download(bm) -> None:
    # The released checkpoint contains the complete backbone state.  The official
    # constructor nevertheless asks torchvision to download ImageNet VGG weights.
    # Disable that redundant network/IO step before instantiation; checkpoint
    # loading below is strict, so model parameters remain exactly checkpointed.
    original = bm.models.vgg16

    def no_download(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original(*args, **kwargs)

    bm.models.vgg16 = no_download


def _load_official_model(official_root: Path, weights_dir: Path, device: torch.device):
    sys.path.insert(0, str(official_root))
    try:
        from cvphr.models.posaglreg import models as bm
    finally:
        # Keep imported modules alive but do not permanently shadow project modules.
        try:
            sys.path.remove(str(official_root))
        except ValueError:
            pass

    _patch_torchvision_no_pretrained_download(bm)

    cfg_path = weights_dir / "training_configure.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cls_name = cfg.get("model_class", "PARCASGM_v5a")
        model_kwargs = cfg.get("model_kwargs", bm.model_kwargs_par_ca_sgm_v5a)
        model_class = getattr(bm, cls_name)
    else:
        model_class = bm.PARCASGM_v5a
        model_kwargs = dict(bm.model_kwargs_par_ca_sgm_v5a)

    model = model_class(**model_kwargs)
    checkpoint = torch.load(weights_dir / "best_model.pth", map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, bm.RSBlockDatasetPA_v3q, model_class.__name__, model_kwargs


def _evaluate_route(
    model,
    dataset_class,
    csv_path: Path,
    device: torch.device,
    mpp: float,
    workers: int,
    batch_size: int,
) -> dict:
    dataset = dataset_class(str(csv_path), is_train=False)
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=1)
    loader = DataLoader(dataset, **kwargs)

    pred_pos, gt_pos, pred_dir, gt_dir = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            patches = batch["patches"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            agl = batch["agl_coords"].to(device, non_blocking=True)
            pos, direction = model(patches)
            pred_pos.append(pos.detach().cpu().numpy())
            gt_pos.append(coords.detach().cpu().numpy())
            pred_dir.append(direction.detach().cpu().numpy())
            gt_dir.append(agl.detach().cpu().numpy())

    pred_pos = np.concatenate(pred_pos, axis=0).astype(np.float64)
    gt_pos = np.concatenate(gt_pos, axis=0).astype(np.float64)
    pred_dir = np.concatenate(pred_dir, axis=0).astype(np.float64)
    gt_dir = np.concatenate(gt_dir, axis=0).astype(np.float64)

    # Same global map coordinate metric used by our route evaluation.
    errors_m = np.linalg.norm((pred_pos - gt_pos) * float(bearing.PATCH_SIZE), axis=1) * float(mpp)

    recall = np.all(np.sign(pred_pos) == np.sign(gt_pos), axis=1)

    pred_norm = pred_dir / np.maximum(np.linalg.norm(pred_dir, axis=1, keepdims=True), 1e-12)
    gt_norm = gt_dir / np.maximum(np.linalg.norm(gt_dir, axis=1, keepdims=True), 1e-12)
    cos = np.clip(np.sum(pred_norm * gt_norm, axis=1), -1.0, 1.0)
    heading_deg = np.degrees(np.arccos(cos))

    result = {
        "frames": int(len(errors_m)),
        "Recall@1_pct": float(100.0 * np.mean(recall)),
        "MLE_m": float(np.mean(errors_m)),
        "MedLE_m": float(np.median(errors_m)),
        "P90_m": float(np.percentile(errors_m, 90)),
        "P95_m": float(np.percentile(errors_m, 95)),
        "P99_m": float(np.percentile(errors_m, 99)),
        "LSR@5_pct": float(100.0 * np.mean(errors_m <= 5.0)),
        "LSR@10_pct": float(100.0 * np.mean(errors_m <= 10.0)),
        "LSR@15_pct": float(100.0 * np.mean(errors_m <= 15.0)),
        "LSR@20_pct": float(100.0 * np.mean(errors_m <= 20.0)),
        "HSR@15_pct": float(100.0 * np.mean(heading_deg <= 15.0)),
        "MHE_deg": float(np.mean(heading_deg)),
        "MedHE_deg": float(np.median(heading_deg)),
        "distance_errors_m": errors_m.tolist(),
        "heading_errors_deg": heading_deg.tolist(),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--generated-root", required=True)
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--output-root", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    official_root = Path(args.official_root).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    generated_root = Path(args.generated_root).resolve()
    prepared_root = generated_root / args.city
    output_dir = Path(args.output_root).resolve() / args.city
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    mpp = float(json.loads((prepared_root / "bearing_satellite.json").read_text(encoding="utf-8"))["mpp"])
    metadata = pd.read_csv(bearing._find_metadata(dataset_root))
    city_rows = bearing._city_rows(metadata, args.city)

    model, dataset_class, model_class_name, model_kwargs = _load_official_model(
        official_root, weights_dir, device
    )

    routes: Dict[str, dict] = {}
    all_dist: List[float] = []
    all_head: List[float] = []
    weighted_recall_success = 0.0
    for route in ("test_01", "test_02"):
        csv_path = output_dir / f"{route}_selected_metadata.csv"
        _route_metadata(dataset_root, prepared_root, args.city, route, city_rows, csv_path)
        metrics = _evaluate_route(
            model, dataset_class, csv_path, device, mpp,
            workers=int(args.workers), batch_size=int(args.batch_size),
        )
        routes[route] = metrics
        all_dist.extend(metrics["distance_errors_m"])
        all_head.extend(metrics["heading_errors_deg"])
        weighted_recall_success += metrics["Recall@1_pct"] * metrics["frames"] / 100.0
        print(
            f"[BEARING-OFFICIAL-SAME-ROUTE] {args.city}/{route}: "
            f"MLE={metrics['MLE_m']:.3f}m MedLE={metrics['MedLE_m']:.3f}m "
            f"LSR15={metrics['LSR@15_pct']:.2f}% Recall1={metrics['Recall@1_pct']:.2f}%",
            flush=True,
        )

    d = np.asarray(all_dist, dtype=np.float64)
    h = np.asarray(all_head, dtype=np.float64)
    aggregate = {
        "frames": int(len(d)),
        "Recall@1_pct": float(100.0 * weighted_recall_success / max(len(d), 1)),
        "MLE_m": float(d.mean()),
        "MedLE_m": float(np.median(d)),
        "P90_m": float(np.percentile(d, 90)),
        "LSR@15_pct": float(100.0 * np.mean(d <= 15.0)),
        "HSR@15_pct": float(100.0 * np.mean(h <= 15.0)),
        "MHE_deg": float(h.mean()),
        "MedHE_deg": float(np.median(h)),
    }
    payload = {
        "method": "Bearing-UAV official VGG-16",
        "source": "official liukejia121/bearinguav code + official pretrained cross_view checkpoint",
        "city": args.city,
        "model_class": model_class_name,
        "model_kwargs": model_kwargs,
        "dataset_route_source": exp.get("test_route_source"),
        "evaluation_scope": "exact same selected test_01/test_02 Bearing frames used by our v39 route experiment",
        "metric_note": "position errors use the same 0.25m/px global-map coordinate metric as our route result",
        "protocol_note": "same frames/GT; official Bearing-UAV is single-frame four-neighbour RST regression, ours is temporal controlled-local-prior refinement",
        "routes": routes,
        "aggregate_two_routes": aggregate,
    }
    (output_dir / "official_bearinguav_same_route.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"[BEARING-OFFICIAL-SAME-ROUTE] DONE {args.city}: "
        f"MLE={aggregate['MLE_m']:.3f}m LSR15={aggregate['LSR@15_pct']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
