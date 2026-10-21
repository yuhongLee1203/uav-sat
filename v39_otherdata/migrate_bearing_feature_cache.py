#!/usr/bin/env python3
"""Migrate verified legacy size/mtime feature caches to model-state hashes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import torch


VISUAL_NAME = "visual_retrieval_A_only.pt"


def model_hash(path: Path) -> str:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("best_model") or payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"visual checkpoint has no model state: {path}")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def checkpoint_candidates(root: Path) -> list[Path]:
    paths = [root / "shared" / "frames3" / VISUAL_NAME]
    paths.extend(
        root / city / "train_frames3" / "checkpoints" / VISUAL_NAME
        for city in ("citya", "cityb", "cityc", "cityd")
    )
    return [path.resolve() for path in paths if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--base-suite-root", required=True)
    args = parser.parse_args()
    suite = Path(args.suite_root).resolve()
    base = Path(args.base_suite_root).resolve()

    checkpoints = []
    for root in (suite, base):
        checkpoints.extend(checkpoint_candidates(root))
    checkpoints = list(dict.fromkeys(checkpoints))
    if not checkpoints:
        raise RuntimeError("no visual checkpoints found")

    identities = {}
    hashes = {}
    for path in checkpoints:
        stat = path.stat()
        digest = model_hash(path)
        identities[(int(stat.st_size), int(stat.st_mtime_ns))] = digest
        hashes[path] = digest

    target = hashes[checkpoints[0]]
    if any(value != target for value in hashes.values()):
        raise RuntimeError("visual model states differ; refusing cache migration")

    cache_entries = []
    route_map = {"route_A": "train_01", "route_B": "test_01", "route_C": "test_02"}
    for city in ("citya", "cityb", "cityc", "cityd"):
        city_root = base / city
        visual_path = city_root / "train_frames3" / "checkpoints" / VISUAL_NAME
        visual = torch.load(visual_path, map_location="cpu")
        origin_lat = float(visual["origin_lat"])
        origin_lon = float(visual["origin_lon"])
        for route_name, prepared_name in route_map.items():
            cache_path = (city_root / "feature_cache" / f"{route_name}_uav_clip.pt").resolve()
            manifest_path = city_root / "prepared" / "routes" / prepared_name / "manifest.csv"
            if cache_path.is_file():
                cache_entries.append(
                    (cache_path, manifest_path.resolve(), origin_lat, origin_lon)
                )

    migrated = reused = skipped = 0
    for path, manifest_path, origin_lat, origin_lon in cache_entries:
        payload = torch.load(path, map_location="cpu")
        signature = payload.get("signature", {})
        legacy = (signature.get("size"), signature.get("mtime_ns"))
        signature_hash = signature.get("uav_model_sha256")
        if signature_hash not in (None, target):
            print("[CACHE MIGRATION SKIP] model hash mismatch", path)
            skipped += 1
            continue
        if signature_hash is None and identities.get(legacy) != target:
            print("[CACHE MIGRATION SKIP] unverified legacy identity", path, legacy)
            skipped += 1
            continue

        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        image_paths = [str(Path(row["image_path"])) for row in rows]
        if image_paths != [str(Path(value)) for value in payload["image_paths"]]:
            print("[CACHE MIGRATION SKIP] image order mismatch", path)
            skipped += 1
            continue
        gt_xy = torch.tensor(
            [
                [float(row["x_m"]) - origin_lon, float(row["y_m"]) - origin_lat]
                for row in rows
            ],
            dtype=payload["gt_xy"].dtype,
        )
        payload["gt_xy"] = gt_xy
        payload["signature"] = {
            "uav_model_sha256": target,
            "backbone": signature.get("backbone", "mobilenet_v3_small"),
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "route_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        }
        temporary = path.with_name(path.name + ".migrating")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        print("[CACHE MIGRATED]", path)
        migrated += 1

    print(
        "[CACHE MIGRATION DONE]",
        f"sha256={target}",
        f"migrated={migrated}",
        f"already_current={reused}",
        f"skipped={skipped}",
        flush=True,
    )
    if skipped:
        raise RuntimeError("some caches could not be cryptographically verified")


if __name__ == "__main__":
    main()
