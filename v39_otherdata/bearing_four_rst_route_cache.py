#!/usr/bin/env python3
"""Build a low-CPU cache for Bearing-UAV paper-style four-RST M2T evaluation.

This cache fixes an important protocol error in the old baseline adapter.  The
Bearing-UAV supplement defines Recall@1 as selecting the closest RST from the
FOUR adjacent RSTs of the current RSB.  University-1652, SUES-200, DenseUAV and
GTA-UAV therefore must not search the whole 4096x4096 city map when reproducing
that localization comparison.

For each selected pseudo-flight frame we cache:
  * UAV image
  * p1/p2/p3/p4 RST images from the official Bearing metadata
  * the four RST centre coordinates
  * the nearest-RST target index (0..3)

No waypoint, route centreline, previous position, motion prior or test-GT based
candidate construction is used.  Waypoints remain visualization-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import bearing_prepare as bearing
from bearinguav_official_route_eval import _resolve

CACHE_VERSION = "bearing_native_four_rst_v1"
RST_PX = 256
OFFSETS = np.asarray([[-128.0, -128.0], [-128.0, 128.0], [128.0, -128.0], [128.0, 128.0]], dtype=np.float64)


def _rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _u8(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _centres_px(row: pd.Series) -> np.ndarray:
    # block_x/y identify the upper-left RST of the 2x2 RSB.  The four official
    # p1..p4 coordinates in the Bearing model are [-1,-1],[-1,+1],[+1,-1],[+1,+1].
    intersection = np.asarray([(float(row["block_x"]) + 1.0) * RST_PX,
                               (float(row["block_y"]) + 1.0) * RST_PX], dtype=np.float64)
    return intersection[None, :] + OFFSETS


def _selected_metadata(dataset_root: Path, prepared: Path, city: str, route: str, city_rows: pd.DataFrame):
    manifest = _rows(prepared / "routes" / route / "manifest.csv")
    rows = []
    for frame in manifest:
        idx = int(frame["source_index"])
        if idx < 0 or idx >= len(city_rows):
            raise RuntimeError(f"{city}/{route}: source_index out of range: {idx}")
        r = city_rows.iloc[idx].copy()
        # audit selected GT against the metadata formula
        gx = float(r["global_x_px"]); gy = float(r["global_y_px"])
        if abs(gx - float(frame["x_px"])) > 1e-3 or abs(gy - float(frame["y_px"])) > 1e-3:
            raise RuntimeError(f"{city}/{route}: manifest/metadata GT mismatch at frame {frame['frame_id']}")
        rows.append((frame, r))
    return rows


def build_city(dataset_root: Path, generated_root: Path, cache_root: Path, city: str,
               cache_size: int = 256, force: bool = False) -> Path:
    prepared = generated_root / city
    meta_path = bearing._find_metadata(dataset_root)
    meta = pd.read_csv(meta_path)
    city_rows = bearing._city_rows(meta, city)
    sat_meta = _json(prepared / "bearing_satellite.json")
    mpp = float(sat_meta["mpp"])

    source_files = [
        meta_path,
        prepared / "routes" / "train_01" / "manifest.csv",
        prepared / "routes" / "test_01" / "manifest.csv",
        prepared / "routes" / "test_02" / "manifest.csv",
    ]
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(city.encode())
    h.update(str(cache_size).encode())
    for p in source_files:
        h.update(str(p).encode()); h.update(p.read_bytes())
    fingerprint = h.hexdigest()

    out = cache_root / city
    cm = out / "cache_meta.json"
    expected = [out / "train_uav.npy", out / "train_positive_sat.npy", out / "train_class_index.npy"]
    for route in ("test_01", "test_02"):
        expected += [out / f"{route}_uav.npy", out / f"{route}_candidates.npy",
                     out / f"{route}_candidate_xy_m.npy", out / f"{route}_gt_m.npy",
                     out / f"{route}_true_index.npy"]
    if not force and cm.exists() and _json(cm).get("fingerprint") == fingerprint and all(p.exists() for p in expected):
        print(f"[4RST-CACHE] hit {city}: {out}", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)

    selected = {r: _selected_metadata(dataset_root, prepared, city, r, city_rows)
                for r in ("train_01", "test_01", "test_02")}

    # Training remains route-adapted for the route experiment, but each positive
    # is the native nearest one of p1..p4.  It never receives route geometry.
    tr = selected["train_01"]
    train_uav = np.lib.format.open_memmap(out / "train_uav.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(tr), cache_size, cache_size, 3))
    train_sat = np.lib.format.open_memmap(out / "train_positive_sat.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(tr), cache_size, cache_size, 3))
    train_gt = np.zeros((len(tr), 2), dtype=np.float32)
    train_center = np.zeros((len(tr), 2), dtype=np.float32)
    # Stable class IDs are absolute RST centre coordinates, not route indices.
    centre_keys = []
    for i, (frame, r) in enumerate(tr):
        centres = _centres_px(r)
        gt_px = np.asarray([float(frame["x_px"]), float(frame["y_px"])], dtype=np.float64)
        ti = int(np.argmin(np.linalg.norm(centres - gt_px[None, :], axis=1)))
        up = _resolve(r["target_path"], dataset_root)
        sp = _resolve(r[f"p{ti+1}_path"], dataset_root)
        train_uav[i] = _u8(up, cache_size)
        train_sat[i] = _u8(sp, cache_size)
        train_gt[i] = gt_px * mpp
        train_center[i] = centres[ti] * mpp
        centre_keys.append((round(float(centres[ti,0]), 3), round(float(centres[ti,1]), 3)))
        if (i + 1) % 100 == 0 or i + 1 == len(tr):
            print(f"[4RST-CACHE] {city}/train: {i+1}/{len(tr)}", flush=True)
    train_uav.flush(); train_sat.flush(); del train_uav, train_sat
    uniq = {k: j for j, k in enumerate(sorted(set(centre_keys)))}
    labels = np.asarray([uniq[k] for k in centre_keys], dtype=np.int64)
    np.save(out / "train_class_index.npy", labels)
    np.save(out / "train_gt_m.npy", train_gt)
    np.save(out / "train_positive_xy_m.npy", train_center)

    route_meta = {}
    for route in ("test_01", "test_02"):
        rows = selected[route]
        uav = np.lib.format.open_memmap(out / f"{route}_uav.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(rows), cache_size, cache_size, 3))
        cand = np.lib.format.open_memmap(out / f"{route}_candidates.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(rows), 4, cache_size, cache_size, 3))
        cxy = np.zeros((len(rows), 4, 2), dtype=np.float32)
        gt = np.zeros((len(rows), 2), dtype=np.float32)
        true = np.zeros((len(rows),), dtype=np.int64)
        oracle_err = []
        for i, (frame, r) in enumerate(rows):
            centres = _centres_px(r)
            gt_px = np.asarray([float(frame["x_px"]), float(frame["y_px"])], dtype=np.float64)
            ti = int(np.argmin(np.linalg.norm(centres - gt_px[None, :], axis=1)))
            uav[i] = _u8(_resolve(r["target_path"], dataset_root), cache_size)
            for k in range(4):
                cand[i, k] = _u8(_resolve(r[f"p{k+1}_path"], dataset_root), cache_size)
            cxy[i] = centres * mpp
            gt[i] = gt_px * mpp
            true[i] = ti
            oracle_err.append(float(np.linalg.norm(cxy[i, ti] - gt[i])))
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"[4RST-CACHE] {city}/{route}: {i+1}/{len(rows)}", flush=True)
        uav.flush(); cand.flush(); del uav, cand
        np.save(out / f"{route}_candidate_xy_m.npy", cxy)
        np.save(out / f"{route}_gt_m.npy", gt)
        np.save(out / f"{route}_true_index.npy", true)
        route_meta[route] = {
            "frames": len(rows),
            "oracle_nearest_RST_center_MLE_m": float(np.mean(oracle_err)),
            "oracle_nearest_RST_center_MedLE_m": float(np.median(oracle_err)),
        }

    payload = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "city": city,
        "mpp": mpp,
        "train_frames": len(tr),
        "train_class_count": int(len(uniq)),
        "routes": route_meta,
        "candidate_policy": "exact p1/p2/p3/p4 from official metadata; four adjacent RSTs only",
        "recall_definition": "top-1 candidate equals nearest RST among the four adjacent RSTs",
        "uses_waypoint_or_route_prior": False,
        "uses_temporal_history": False,
        "paper_note": "Bearing-UAV supplement defines Recall@1 over four adjacent RSTs; tile-center prediction is used for M2T localization error.",
    }
    cm.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[4RST-CACHE] PASS {city}: train={len(tr)} classes={len(uniq)} routes={route_meta}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--generated-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    for city in a.cities:
        build_city(Path(a.dataset_root).resolve(), Path(a.generated_root).resolve(),
                   Path(a.cache_root).resolve(), city, a.cache_size, a.force)


if __name__ == "__main__":
    main()
