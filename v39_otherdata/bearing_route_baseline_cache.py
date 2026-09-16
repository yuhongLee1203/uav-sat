#!/usr/bin/env python3
"""Build one shared, low-CPU Bearing route cache for all public baselines.

The cache is deliberately method-agnostic.  JPEG/RSI files are decoded only
once, resized to uint8 256x256 arrays and stored as uncompressed .npy files so
all later training jobs can mmap the same bytes instead of repeatedly decoding
images with many CPU workers.

Training pairs
--------------
Each retained train_01 UAV observation is paired with a satellite crop centred
at that observation.  The pair id is the training class / positive-pair id used
by route-adapted reproductions of University-1652, SUES-200, DenseUAV and
GTA-UAV.

Test gallery
------------
The test gallery is NOT built from per-frame test GT.  It is sampled from the
pre-declared route centreline with a fixed 4.5 m longitudinal spacing and five
cross-track lanes.  Thus every method sees the same route prior/candidate pool
without leaking the exact test observation coordinate into the gallery.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

Point = Tuple[float, float]


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


def _waypoints(path: Path) -> List[Point]:
    payload = _json(path)
    items = sorted(payload["waypoints"], key=lambda x: int(x["waypoint_order"]))
    pts = [(float(x["pixel_x"]), float(x["pixel_y"])) for x in items]
    if len(pts) < 2:
        raise RuntimeError(f"fewer than two waypoints: {path}")
    return pts


def _sample_corridor(
    waypoints: Sequence[Point],
    mpp: float,
    spacing_m: float,
    cross_offsets_m: Sequence[float],
    width: int,
    height: int,
) -> np.ndarray:
    pts: List[Point] = []
    seen = set()

    def add(x: float, y: float):
        if not (0.0 <= x < width and 0.0 <= y < height):
            return
        key = (round(x, 2), round(y, 2))
        if key not in seen:
            seen.add(key)
            pts.append((x, y))

    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        lp = math.hypot(dx, dy)
        if lp <= 1e-6:
            continue
        length_m = lp * mpp
        steps = max(1, int(math.ceil(length_m / spacing_m)))
        nx, ny = -dy / lp, dx / lp
        for i in range(steps):
            t = i / steps
            cx, cy = a[0] + t * dx, a[1] + t * dy
            for off_m in cross_offsets_m:
                off_px = float(off_m) / mpp
                add(cx + nx * off_px, cy + ny * off_px)
    # include final endpoint lanes
    a, b = waypoints[-2], waypoints[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    lp = max(math.hypot(dx, dy), 1e-6)
    nx, ny = -dy / lp, dx / lp
    for off_m in cross_offsets_m:
        off_px = float(off_m) / mpp
        add(b[0] + nx * off_px, b[1] + ny * off_px)
    return np.asarray(pts, dtype=np.float32)


def _u8_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _sat_crop(base: Image.Image, x: float, y: float, crop_px: int, size: int) -> np.ndarray:
    half = crop_px / 2.0
    # PIL pads regions outside the image with black.  The route planner normally
    # stays well inside the RSI; padding is retained as a deterministic fallback.
    patch = base.crop((x - half, y - half, x + half, y + half))
    patch = patch.resize((size, size), Image.Resampling.BICUBIC).convert("RGB")
    return np.asarray(patch, dtype=np.uint8)


def _write_array(path: Path, shape, producer):
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    for i in range(shape[0]):
        arr[i] = producer(i)
        if (i + 1) % 250 == 0 or i + 1 == shape[0]:
            print(f"[CACHE] {path.name}: {i+1}/{shape[0]}", flush=True)
    arr.flush()
    del arr


def build_city(
    generated_root: Path,
    cache_root: Path,
    city: str,
    cache_size: int = 256,
    crop_fov_m: float = 45.0,
    spacing_m: float = 4.5,
    cross_offsets_m: Sequence[float] = (-13.5, -6.75, 0.0, 6.75, 13.5),
    force: bool = False,
) -> Path:
    prepared = generated_root / city
    required = [
        prepared / "bearing_satellite.json",
        prepared / "routes" / "train_01" / "manifest.csv",
        prepared / "routes" / "test_01" / "manifest.csv",
        prepared / "routes" / "test_02" / "manifest.csv",
        prepared / "routes" / "test_01" / "waypoints.json",
        prepared / "routes" / "test_02" / "waypoints.json",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    sat_meta = _json(required[0])
    sat_path = Path(sat_meta["satellite_image"]).resolve()
    mpp = float(sat_meta["mpp"])
    extra = {
        "city": city,
        "cache_size": cache_size,
        "crop_fov_m": crop_fov_m,
        "spacing_m": spacing_m,
        "cross_offsets_m": list(map(float, cross_offsets_m)),
        "mpp": mpp,
        "satellite_image": str(sat_path),
    }
    fp = _fingerprint(required, extra)
    out = cache_root / city
    meta_path = out / "cache_meta.json"
    if not force and meta_path.exists():
        old = _json(meta_path)
        expected = [
            out / "train_uav.npy", out / "train_sat.npy", out / "train_gt_m.npy",
            out / "test_01_uav.npy", out / "test_01_gt_m.npy",
            out / "test_01_gallery_sat.npy", out / "test_01_gallery_xy_m.npy",
            out / "test_02_uav.npy", out / "test_02_gt_m.npy",
            out / "test_02_gallery_sat.npy", out / "test_02_gallery_xy_m.npy",
        ]
        if old.get("fingerprint") == fp and all(p.exists() for p in expected):
            print(f"[CACHE] hit {city}: {out}", flush=True)
            return out

    out.mkdir(parents=True, exist_ok=True)
    train = _rows(required[1])
    test = {r: _rows(prepared / "routes" / r / "manifest.csv") for r in ("test_01", "test_02")}
    crop_px = max(8, int(round(crop_fov_m / mpp)))

    with Image.open(sat_path) as sat_src:
        sat = sat_src.convert("RGB")
        w, h = sat.size

        train_gt = np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in train], dtype=np.float32)
        np.save(out / "train_gt_m.npy", train_gt)
        _write_array(
            out / "train_uav.npy", (len(train), cache_size, cache_size, 3),
            lambda i: _u8_rgb(Path(train[i]["image_path"]), cache_size),
        )
        _write_array(
            out / "train_sat.npy", (len(train), cache_size, cache_size, 3),
            lambda i: _sat_crop(sat, float(train[i]["x_px"]), float(train[i]["y_px"]), crop_px, cache_size),
        )

        route_meta = {}
        for route in ("test_01", "test_02"):
            rows = test[route]
            gt = np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in rows], dtype=np.float32)
            np.save(out / f"{route}_gt_m.npy", gt)
            _write_array(
                out / f"{route}_uav.npy", (len(rows), cache_size, cache_size, 3),
                lambda i, rr=rows: _u8_rgb(Path(rr[i]["image_path"]), cache_size),
            )

            wp = _waypoints(prepared / "routes" / route / "waypoints.json")
            gallery_px = _sample_corridor(wp, mpp, spacing_m, cross_offsets_m, w, h)
            gallery_m = gallery_px * mpp
            np.save(out / f"{route}_gallery_xy_m.npy", gallery_m.astype(np.float32))
            _write_array(
                out / f"{route}_gallery_sat.npy",
                (len(gallery_px), cache_size, cache_size, 3),
                lambda i, gp=gallery_px: _sat_crop(sat, float(gp[i, 0]), float(gp[i, 1]), crop_px, cache_size),
            )
            route_meta[route] = {
                "frames": len(rows),
                "gallery_candidates": int(len(gallery_px)),
                "waypoints": len(wp),
            }

    meta = {
        **extra,
        "fingerprint": fp,
        "train_frames": len(train),
        "crop_px": crop_px,
        "routes": route_meta,
        "gallery_policy": "planned-route corridor; no per-frame test GT used to place gallery candidates",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[CACHE] built {city}: train={len(train)} routes={route_meta}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generated-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--crop-fov-m", type=float, default=45.0)
    p.add_argument("--spacing-m", type=float, default=4.5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for city in args.cities:
        build_city(
            Path(args.generated_root).resolve(), Path(args.cache_root).resolve(), city,
            cache_size=args.cache_size, crop_fov_m=args.crop_fov_m,
            spacing_m=args.spacing_m, force=args.force,
        )


if __name__ == "__main__":
    main()
