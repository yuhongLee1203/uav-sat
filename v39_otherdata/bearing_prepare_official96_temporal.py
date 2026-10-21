#!/usr/bin/env python3
"""Adapt Bearing-UAV's official rsi_id=96 split to temporal route episodes.

The official 90K metadata and its seed-42 85/5/10 random split are preserved
exactly.  Bearing-UAV rows are independent observations, while the v39 model
requires ordered trajectories, so nearby observations are ordered into one
episode inside each split and city.  No episode crosses a city or split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import bearing_prepare as base
from bearing_multicity_routes import OFFICIAL_TEST_ROUTES

SPLIT_SEED = 42
SPLIT_RATIOS = (0.85, 0.05, 0.10)
ADAPTER_VERSION = "official_rsi96_seed42_temporal_v1"


def official_split_indices(size: int) -> dict[str, np.ndarray]:
    lengths = [int(SPLIT_RATIOS[0] * size), int(SPLIT_RATIOS[1] * size)]
    lengths.append(size - sum(lengths))
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(size, generator=generator).numpy()
    a, b = lengths[0], lengths[0] + lengths[1]
    return {
        "train": permutation[:a],
        "val": permutation[a:b],
        "test": permutation[b:],
    }


def _sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _angle_error(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - float(target) + 180.0) % 360.0 - 180.0)


def select_episode(
    rows: pd.DataFrame,
    planned: np.ndarray,
    *,
    step_m: float,
    max_target_distance_m: float,
    max_frame_step_m: float,
    used_paths: set[str],
) -> tuple[pd.DataFrame, dict]:
    targets, headings = base._dense_targets(planned, step_m)
    xy = rows[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    yaw = np.mod(rows["theta"].to_numpy(dtype=np.float64), 360.0)
    max_target_px = max_target_distance_m / base.MPP
    max_step_px = max_frame_step_m / base.MPP
    picked: list[int] = []
    picked_targets: list[int] = []
    local_used: set[int] = set()
    last_xy = None

    for target_id, (target, heading) in enumerate(zip(targets, headings)):
        spatial = np.linalg.norm(xy - target[None, :], axis=1)
        count = min(96, len(rows))
        candidates = np.argpartition(spatial, count - 1)[:count]
        candidates = candidates[np.argsort(spatial[candidates])]
        best = None
        best_cost = float("inf")
        for raw_idx in candidates:
            idx = int(raw_idx)
            identity = str(rows.iloc[idx]["target_path"])
            if idx in local_used or identity in used_paths:
                continue
            if spatial[idx] > max_target_px:
                continue
            if last_xy is not None:
                frame_step = float(np.linalg.norm(xy[idx] - last_xy))
                if frame_step > max_step_px:
                    continue
            else:
                frame_step = 0.0
            cost = (
                float(spatial[idx])
                + 0.04 * float(_angle_error(yaw[idx:idx + 1], heading)[0])
                + 0.15 * abs(frame_step - step_m / base.MPP)
            )
            if cost < best_cost:
                best, best_cost = idx, cost
        if best is None:
            continue
        picked.append(best)
        picked_targets.append(target_id)
        local_used.add(best)
        used_paths.add(str(rows.iloc[best]["target_path"]))
        last_xy = xy[best]

    if len(picked) < 24:
        raise RuntimeError(
            f"official split is too sparse for a temporal episode: {len(picked)} frames"
        )
    selected = rows.iloc[picked].copy()
    selected.index = selected["_city_source_index"].astype(int)
    selected_xy = selected[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    steps = np.linalg.norm(np.diff(selected_xy, axis=0), axis=1) * base.MPP
    target_error = np.asarray([
        np.linalg.norm(selected_xy[i] - targets[target_id]) * base.MPP
        for i, target_id in enumerate(picked_targets)
    ])
    audit = {
        "frames": int(len(selected)),
        "targets": int(len(targets)),
        "first_target": int(picked_targets[0]),
        "last_target": int(picked_targets[-1]),
        "target_span_ratio": float(
            (picked_targets[-1] - picked_targets[0]) / max(len(targets) - 1, 1)
        ),
        "target_error_mean_m": float(target_error.mean()),
        "target_error_max_m": float(target_error.max()),
        "frame_step_mean_m": float(steps.mean()),
        "frame_step_p90_m": float(np.percentile(steps, 90)),
        "frame_step_p95_m": float(np.percentile(steps, 95)),
        "frame_step_max_m": float(steps.max()),
    }
    if audit["target_span_ratio"] < 0.75:
        raise RuntimeError(f"episode covers too little of route: {audit}")
    return selected, audit


def select_spatial_episode(
    rows: pd.DataFrame,
    *,
    preferred_step_m: float,
    max_frame_step_m: float,
    target_frames: int,
    route_slot: int,
    used_paths: set[str],
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build a deterministic nearby-observation chain inside one split."""
    available = rows[
        ~rows["target_path"].astype(str).isin(used_paths)
    ].reset_index(drop=True)
    xy = available[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    if len(xy) < 24:
        raise RuntimeError("too few unused official rows for a temporal episode")
    preferred_px = preferred_step_m / base.MPP
    max_step_px = max_frame_step_m / base.MPP
    ordering = np.argsort(xy[:, 0] + (0.371 + 0.113 * route_slot) * xy[:, 1])
    fractions = np.linspace(0.05, 0.95, 24)
    fractions = np.roll(fractions, route_slot * 7)
    best_path: list[int] = []

    for fraction in fractions:
        start = int(ordering[min(int(fraction * (len(ordering) - 1)), len(ordering) - 1)])
        path = [start]
        used = {start}
        previous_direction = None
        while len(path) < target_frames:
            delta = xy - xy[path[-1]][None, :]
            distance = np.linalg.norm(delta, axis=1)
            valid = np.where((distance > 1e-6) & (distance <= max_step_px))[0]
            valid = [int(i) for i in valid if int(i) not in used]
            if not valid:
                break
            candidates = np.asarray(valid, dtype=np.int64)
            cost = np.abs(distance[candidates] - preferred_px)
            if previous_direction is not None:
                direction = delta[candidates] / np.maximum(
                    distance[candidates, None], 1e-9
                )
                cost = cost + 0.35 * preferred_px * (
                    1.0 - direction @ previous_direction
                )
            nxt = int(candidates[int(np.argmin(cost))])
            step = xy[nxt] - xy[path[-1]]
            previous_direction = step / max(float(np.linalg.norm(step)), 1e-9)
            path.append(nxt)
            used.add(nxt)
        if len(path) > len(best_path):
            best_path = path
        if len(best_path) >= target_frames:
            break

    if len(best_path) < 24:
        raise RuntimeError(
            f"official split cannot form a >=24-frame local chain: {len(best_path)}"
        )
    selected = available.iloc[best_path].copy()
    selected.index = selected["_city_source_index"].astype(int)
    for value in selected["target_path"].astype(str):
        used_paths.add(value)
    selected_xy = selected[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    steps = np.linalg.norm(np.diff(selected_xy, axis=0), axis=1) * base.MPP
    # A sparse set of chain anchors defines the known pseudo-route.  Every
    # observation remains its original official GT; no coordinate is moved.
    anchor_ids = list(range(0, len(selected_xy), 8))
    if anchor_ids[-1] != len(selected_xy) - 1:
        anchor_ids.append(len(selected_xy) - 1)
    planned = selected_xy[anchor_ids]
    audit = {
        "frames": int(len(selected)),
        "target_frames": int(target_frames),
        "waypoint_anchors": int(len(planned)),
        "frame_step_mean_m": float(steps.mean()),
        "frame_step_p90_m": float(np.percentile(steps, 90)),
        "frame_step_p95_m": float(np.percentile(steps, 95)),
        "frame_step_max_m": float(steps.max()),
        "construction": "deterministic nearest-neighbor chain within official split",
    }
    return selected, planned, audit


def _copy_route(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def prepare(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    city = args.city.lower()
    metadata_path = base._find_metadata(dataset_root)
    metadata = pd.read_csv(metadata_path)
    metadata["_official_index"] = np.arange(len(metadata), dtype=np.int64)
    splits = official_split_indices(len(metadata))
    split_lookup = {
        name: set(int(x) for x in values.tolist()) for name, values in splits.items()
    }

    city_rows = base._city_rows(metadata, city)
    city_rows["_city_source_index"] = np.arange(len(city_rows), dtype=np.int64)
    city_rows["official_split"] = ""
    for name, allowed in split_lookup.items():
        mask = city_rows["_official_index"].astype(int).isin(allowed)
        city_rows.loc[mask, "official_split"] = name
    if (city_rows["official_split"] == "").any():
        raise RuntimeError("some city rows were not assigned to the official split")

    sat_path = base._find_satellite(dataset_root, city)
    with Image.open(sat_path) as image:
        width, height = image.size
    if (width, height) != (base.REFERENCE_SIZE, base.REFERENCE_SIZE):
        raise ValueError(f"unexpected RSI size: {(width, height)}")

    specs = {
        "train_01": ("train", base.ROUTE_SPECS["train_01"]),
        "val_01": ("val", None),
        "test_01": ("test", None),
        "test_02": ("test", None),
    }
    basename_index = base._build_basename_index(dataset_root, city)
    used_by_split = {name: set() for name in ("train", "val", "test")}
    route_stats = {}
    route_arrays = {}
    output_root.mkdir(parents=True, exist_ok=True)

    for route_slot, (route_name, (split_name, points)) in enumerate(specs.items()):
        pool = city_rows[city_rows["official_split"] == split_name].reset_index(drop=True)
        if points is not None:
            planned = base._scale_route(points, width, height)
            selected, audit = select_episode(
                pool,
                planned,
                step_m=float(args.step_m),
                max_target_distance_m=float(args.max_target_distance_m),
                max_frame_step_m=float(args.max_frame_step_m),
                used_paths=used_by_split[split_name],
            )
        else:
            target_frames = (
                int(args.val_frames) if split_name == "val" else int(args.test_frames)
            )
            selected, planned, audit = select_spatial_episode(
                pool,
                preferred_step_m=float(args.sparse_preferred_step_m),
                max_frame_step_m=float(args.max_frame_step_m),
                target_frames=target_frames,
                route_slot=route_slot,
                used_paths=used_by_split[split_name],
            )
        image_paths = [
            base._resolve_image_path(value, dataset_root, city, basename_index)
            for value in selected["target_path"]
        ]
        base._write_route(
            output_root / "routes" / route_name,
            route_name,
            planned,
            selected,
            image_paths,
        )
        route_arrays[route_name] = planned
        route_stats[route_name] = {
            "official_split": split_name,
            "pool_rows": int(len(pool)),
            "sample_step_m": float(args.step_m),
            "actual_step_mean_m": float(audit["frame_step_mean_m"]),
            "actual_step_p90_m": float(audit["frame_step_p90_m"]),
            "actual_step_p95_m": float(audit["frame_step_p95_m"]),
            **audit,
        }
        print("[OFFICIAL96 EPISODE]", city, route_name, json.dumps(route_stats[route_name]), flush=True)

    old_train_routes = base.TRAIN_ROUTES
    try:
        base.TRAIN_ROUTES = ("train_01",)
        base._make_train_union(output_root)
    finally:
        base.TRAIN_ROUTES = old_train_routes
    _copy_route(output_root / "routes" / "val_01", output_root / "routes" / "route_V")

    split_audit = {
        "adapter_version": ADAPTER_VERSION,
        "official_dataset_id": 96,
        "metadata_csv": str(metadata_path),
        "metadata_rows": int(len(metadata)),
        "split_seed": SPLIT_SEED,
        "split_ratio": list(SPLIT_RATIOS),
        "split_counts_global": {k: int(len(v)) for k, v in splits.items()},
        "split_sha256_global": {k: _sha256_indices(v) for k, v in splits.items()},
        "city": city,
        "city_rows": int(len(city_rows)),
        "city_split_counts": {
            k: int((city_rows["official_split"] == k).sum())
            for k in ("train", "val", "test")
        },
        "route_stats": route_stats,
        "no_cross_split_samples": True,
        "temporal_adapter": (
            "Independent official rows are ordered into nearby per-city episodes; "
            "this is not an original recorded UAV video."
        ),
    }
    (output_root / "official96_split_audit.json").write_text(
        json.dumps(split_audit, indent=2), encoding="utf-8"
    )
    sat_meta = {
        "mode": "bearing_uav_pixel_meter",
        "mpp": base.MPP,
        "width": width,
        "height": height,
        "city": city,
        "rsi_id": base.CITY_TO_RSI[city][0],
        "official_merged_rsi_id": 96,
        "satellite_image": str(sat_path),
        "metadata_csv": str(metadata_path),
    }
    (output_root / "bearing_satellite.json").write_text(
        json.dumps(sat_meta, indent=2), encoding="utf-8"
    )
    experiment = {
        "dataset_root": str(dataset_root),
        "city": city,
        "satellite_image": str(sat_path),
        "metadata_csv": str(metadata_path),
        "mpp": base.MPP,
        "sequence_selection_version": ADAPTER_VERSION,
        "official_dataset_id": 96,
        "official_split_seed": SPLIT_SEED,
        "official_split_ratio": list(SPLIT_RATIOS),
        "train_routes": ["train_01"],
        "validation_routes": ["val_01"],
        "inference_routes": ["test_01", "test_02"],
        "route_stats": route_stats,
    }
    (output_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    preview_routes = {
        "train_01": route_arrays["train_01"],
        "train_02": route_arrays["val_01"],
        "test_01": route_arrays["test_01"],
        "test_02": route_arrays["test_02"],
    }
    base._draw_preview(
        sat_path, preview_routes, output_root / "route_plan_full_satellite.jpg"
    )
    print("[OFFICIAL96 PREPARED]", output_root, flush=True)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    parser.add_argument("--city", required=True, choices=sorted(base.CITY_TO_RSI))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step-m", type=float, default=8.0)
    parser.add_argument("--max-target-distance-m", type=float, default=30.0)
    parser.add_argument("--max-frame-step-m", type=float, default=42.0)
    parser.add_argument("--sparse-preferred-step-m", type=float, default=18.0)
    parser.add_argument("--val-frames", type=int, default=80)
    parser.add_argument("--test-frames", type=int, default=120)
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
