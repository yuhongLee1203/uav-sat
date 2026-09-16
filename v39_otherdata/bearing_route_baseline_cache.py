#!/usr/bin/env python3
"""Build a low-CPU cache for native matching-to-tile baselines on Bearing-UAV.

Important protocol rule
-----------------------
University-1652, SUES-200, DenseUAV and GTA-UAV do NOT receive our waypoint
route, temporal state, local prior, previous-frame position, or a route-limited
candidate pool.  Their test gallery is the native Bearing-UAV 16x16 RST grid
covering the complete 4096x4096 city RSI (256 tiles).

Training uses only the UAV frames selected for our Route-A/train_01 experiment,
but each frame is paired with the ordinary 256x256 RST tile that contains it.
Thus the compared methods are route-adapted only in the sense that they see the
same training UAV observations; their localization mechanism remains their own
matching/retrieval paradigm.

CPU / I/O
---------
The selected train/test JPEGs are decoded once to uint8 .npy files.  The whole
satellite gallery is cut once from the city RSI.  Later model runs mmap these
arrays and therefore do not repeatedly decode source JPEGs with many workers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

CACHE_VERSION = "native_m2t_full_city_rst_v2"
RST_PX = 256


def _rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(paths: Iterable[Path], extra: dict) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode())
        h.update(path.read_bytes())
    h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()


def _u8_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _write_array(path: Path, shape, producer) -> None:
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    for i in range(shape[0]):
        arr[i] = producer(i)
        if (i + 1) % 250 == 0 or i + 1 == shape[0]:
            print(f"[M2T-CACHE] {path.name}: {i+1}/{shape[0]}", flush=True)
    arr.flush()
    del arr


def _tile_index_from_px(x: float, y: float, tiles_x: int, tiles_y: int) -> int:
    tx = int(np.clip(np.floor(float(x) / RST_PX), 0, tiles_x - 1))
    ty = int(np.clip(np.floor(float(y) / RST_PX), 0, tiles_y - 1))
    return ty * tiles_x + tx


def build_city(
    generated_root: Path,
    cache_root: Path,
    city: str,
    cache_size: int = 256,
    force: bool = False,
) -> Path:
    prepared = generated_root / city
    required = [
        prepared / "bearing_satellite.json",
        prepared / "routes" / "train_01" / "manifest.csv",
        prepared / "routes" / "test_01" / "manifest.csv",
        prepared / "routes" / "test_02" / "manifest.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    sat_meta = _json(required[0])
    sat_path = Path(sat_meta["satellite_image"]).resolve()
    mpp = float(sat_meta["mpp"])
    if not sat_path.exists():
        raise FileNotFoundError(sat_path)

    with Image.open(sat_path) as src:
        width, height = src.size
    if width % RST_PX or height % RST_PX:
        raise RuntimeError(f"{city}: RSI size {width}x{height} is not divisible by {RST_PX}")
    tiles_x, tiles_y = width // RST_PX, height // RST_PX
    if (tiles_x, tiles_y) != (16, 16):
        raise RuntimeError(f"{city}: expected Bearing 16x16 RST grid, got {tiles_x}x{tiles_y}")

    extra = {
        "cache_version": CACHE_VERSION,
        "city": city,
        "cache_size": int(cache_size),
        "mpp": mpp,
        "rst_px": RST_PX,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "satellite_image": str(sat_path),
    }
    fp = _fingerprint(required, extra)
    out = cache_root / city
    meta_path = out / "cache_meta.json"
    expected = [
        out / "tile_gallery_sat.npy",
        out / "tile_gallery_xy_m.npy",
        out / "train_uav.npy",
        out / "train_gt_m.npy",
        out / "train_tile_index.npy",
        out / "train_class_index.npy",
        out / "test_01_uav.npy",
        out / "test_01_gt_m.npy",
        out / "test_01_tile_index.npy",
        out / "test_02_uav.npy",
        out / "test_02_gt_m.npy",
        out / "test_02_tile_index.npy",
    ]
    if not force and meta_path.exists():
        old = _json(meta_path)
        if old.get("fingerprint") == fp and all(p.exists() for p in expected):
            print(f"[M2T-CACHE] hit {city}: {out}", flush=True)
            return out

    out.mkdir(parents=True, exist_ok=True)
    train = _rows(required[1])
    tests = {
        route: _rows(prepared / "routes" / route / "manifest.csv")
        for route in ("test_01", "test_02")
    }
    if not train:
        raise RuntimeError(f"{city}: empty train_01")

    # Whole-city native RST gallery.  It is independent of any test route.
    tile_xy_m = []
    with Image.open(sat_path) as src0:
        sat = src0.convert("RGB")
        gallery = np.lib.format.open_memmap(
            out / "tile_gallery_sat.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(tiles_x * tiles_y, cache_size, cache_size, 3),
        )
        k = 0
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                left, top = tx * RST_PX, ty * RST_PX
                patch = sat.crop((left, top, left + RST_PX, top + RST_PX)).convert("RGB")
                if patch.size != (cache_size, cache_size):
                    patch = patch.resize((cache_size, cache_size), Image.Resampling.BICUBIC)
                gallery[k] = np.asarray(patch, dtype=np.uint8)
                tile_xy_m.append(((left + RST_PX / 2.0) * mpp, (top + RST_PX / 2.0) * mpp))
                k += 1
        gallery.flush()
        del gallery
    tile_xy_m = np.asarray(tile_xy_m, dtype=np.float32)
    np.save(out / "tile_gallery_xy_m.npy", tile_xy_m)

    # Route-A observations become normal retrieval training samples.  The route
    # geometry itself is never read here and is never an input to a baseline.
    train_gt = np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in train], dtype=np.float32)
    train_tile = np.asarray([
        _tile_index_from_px(float(r["x_px"]), float(r["y_px"]), tiles_x, tiles_y)
        for r in train
    ], dtype=np.int64)
    unique_tiles = sorted(set(int(v) for v in train_tile.tolist()))
    tile_to_class = {tile: i for i, tile in enumerate(unique_tiles)}
    train_class = np.asarray([tile_to_class[int(v)] for v in train_tile], dtype=np.int64)
    np.save(out / "train_gt_m.npy", train_gt)
    np.save(out / "train_tile_index.npy", train_tile)
    np.save(out / "train_class_index.npy", train_class)
    _write_array(
        out / "train_uav.npy",
        (len(train), cache_size, cache_size, 3),
        lambda i: _u8_rgb(Path(train[i]["image_path"]), cache_size),
    )

    route_meta = {}
    for route, rows in tests.items():
        if not rows:
            raise RuntimeError(f"{city}/{route}: empty test route")
        gt = np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in rows], dtype=np.float32)
        tile_idx = np.asarray([
            _tile_index_from_px(float(r["x_px"]), float(r["y_px"]), tiles_x, tiles_y)
            for r in rows
        ], dtype=np.int64)
        np.save(out / f"{route}_gt_m.npy", gt)
        np.save(out / f"{route}_tile_index.npy", tile_idx)
        _write_array(
            out / f"{route}_uav.npy",
            (len(rows), cache_size, cache_size, 3),
            lambda i, rr=rows: _u8_rgb(Path(rr[i]["image_path"]), cache_size),
        )
        route_meta[route] = {
            "frames": len(rows),
            "unique_gt_tiles": int(len(set(tile_idx.tolist()))),
        }

    meta = {
        **extra,
        "fingerprint": fp,
        "train_frames": len(train),
        "train_class_count": len(unique_tiles),
        "train_positive_tile_ids": unique_tiles,
        "routes": route_meta,
        "gallery_candidates": int(tiles_x * tiles_y),
        "gallery_policy": (
            "native whole-city matching-to-tile gallery: all 16x16=256 fixed 256px RST tiles; "
            "NO waypoint/route prior, NO previous-frame prior, NO test-GT candidate placement"
        ),
        "training_policy": (
            "selected train_01 UAV observations paired only with the fixed RST tile containing the observation; "
            "route geometry/waypoints are not model inputs"
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[M2T-CACHE] built {city}: train={len(train)} classes={len(unique_tiles)} "
        f"gallery={tiles_x*tiles_y} test={route_meta}",
        flush=True,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for city in args.cities:
        build_city(
            Path(args.generated_root).resolve(),
            Path(args.cache_root).resolve(),
            city,
            cache_size=args.cache_size,
            force=args.force,
        )


if __name__ == "__main__":
    main()
