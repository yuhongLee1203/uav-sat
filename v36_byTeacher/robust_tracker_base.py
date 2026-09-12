"""Shared non-estimator utilities for v36_byTeacher.

This file intentionally contains no tracking policy, GT-centered prior,
teacher selection, speed/turn clamp, Kalman gate, or posterior projection.
The autonomous estimator lives in robust_tracker.py.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import config
from data import RouteDataset, meters_from_latlon


@dataclass
class RouteCache:
    route_name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    uav_clip: torch.Tensor
    image_paths: list

    def __len__(self):
        return int(self.gt_xy.shape[0])


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_dtype():
    name = str(getattr(config, "FEATURE_CACHE_DTYPE", "float16")).lower()
    return torch.float16 if name == "float16" else torch.float32


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def load_waypoint_xy(route_name, origin_lat, origin_lon):
    path = Path(config.WAYPOINT_FILES[route_name])
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    waypoints = sorted(
        payload["waypoints"], key=lambda item: int(item["waypoint_order"])
    )
    rows = []
    for waypoint in waypoints:
        x_m, y_m = meters_from_latlon(
            waypoint["latitude"],
            waypoint["longitude"],
            origin_lat,
            origin_lon,
        )
        rows.append([float(x_m), float(y_m)])
    if len(rows) < 2:
        raise RuntimeError("%s needs at least start + one following waypoint" % route_name)
    return np.asarray(rows, dtype=np.float64)


@torch.no_grad()
def build_route_cache(route_name, root, visual, device):
    """Cache frozen UAV backbone features and position labels.

    The position labels are stored for training targets and post-prediction
    metrics.  robust_tracker.py does not use gt_xy to choose an inference
    search center.
    """

    config.FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.FEATURE_CACHE_DIR / (route_name + "_uav_clip.pt")

    stat = config.VISUAL_CHECKPOINT.stat()
    signature = {
        "visual_size": int(stat.st_size),
        "visual_mtime_ns": int(stat.st_mtime_ns),
        "backbone": str(config.BACKBONE_KEY),
    }

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("signature") == signature:
            return RouteCache(
                route_name=route_name,
                frame_ids=payload["frame_ids"],
                gt_xy=payload["gt_xy"],
                uav_clip=payload["uav_clip"],
                image_paths=payload["image_paths"],
            )

    dataset = RouteDataset(
        Path(root),
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )
    frame_rows = []
    gt_rows = []
    clip_rows = []
    image_paths = []
    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        clip_rows.append(
            visual.encode_uav_clip(uav).detach().cpu().to(cache_dtype())
        )
        gt_rows.append(torch.stack([item["xy"].float() for item in items]))
        for item in items:
            frame_rows.append(parse_frame_id(item["frame_id"]))
            image_paths.append(str(item["image_path"]))
        if start == 0 or end == len(dataset) or (start // batch_size) % 10 == 0:
            print(
                "%s backbone cache: %d/%d" % (route_name, end, len(dataset)),
                flush=True,
            )

    result = RouteCache(
        route_name=route_name,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        gt_xy=torch.cat(gt_rows).float(),
        uav_clip=torch.cat(clip_rows),
        image_paths=image_paths,
    )
    torch.save(
        {
            "signature": signature,
            "frame_ids": result.frame_ids,
            "gt_xy": result.gt_xy,
            "uav_clip": result.uav_clip,
            "image_paths": result.image_paths,
        },
        cache_path,
    )
    return result


def metric_summary(errors):
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return {
            "MLE_m": float("inf"),
            "MedLE_m": float("inf"),
            "P90_m": float("inf"),
            "P95_m": float("inf"),
            "P99_m": float("inf"),
            "CVaR90_m": float("inf"),
            "LSR@5_pct": 0.0,
            "LSR@10_pct": 0.0,
            "LSR@15_pct": 0.0,
            "LSR@20_pct": 0.0,
        }
    p90 = float(np.quantile(values, 0.90))
    tail = values[values >= p90]
    return {
        "MLE_m": float(np.mean(values)),
        "MedLE_m": float(np.median(values)),
        "P90_m": p90,
        "P95_m": float(np.quantile(values, 0.95)),
        "P99_m": float(np.quantile(values, 0.99)),
        "CVaR90_m": float(np.mean(tail)) if tail.size else p90,
        "LSR@5_pct": float(np.mean(values <= 5.0) * 100.0),
        "LSR@10_pct": float(np.mean(values <= 10.0) * 100.0),
        "LSR@15_pct": float(np.mean(values <= 15.0) * 100.0),
        "LSR@20_pct": float(np.mean(values <= 20.0) * 100.0),
    }
