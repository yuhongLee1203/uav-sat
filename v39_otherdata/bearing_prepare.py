#!/usr/bin/env python3
"""Prepare Bearing-UAV-90K pseudo-flight routes for the v39 temporal tracker.

Bearing-UAV-90K contains independently sampled UAV observations rather than a
recorded video trajectory. This script plans continuous routes on one full city
RSI and converts nearest, heading-compatible UAV samples into ordered sequences.
Heading is used only offline for sequence construction and is never fed to the
localization model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

MPP = 0.25
PATCH_SIZE = 256
REFERENCE_SIZE = 4096
CITY_TO_RSI = {
    "citya": ("34bc", "35.67091338738739_139.69289911300856_1791.95_1024_1024_4326_city.jpg"),
    "cityb": ("36bc", "25.030947387387386_121.51462868800057_1791.95_1024_1024_4326_city.jpg"),
    "cityc": ("37bc", "1.2897673873873876_103.84197619336068_1791.95_1024_1024_4326_city.jpg"),
    "cityd": ("38bc", "37.75538738738739_-122.4533351740761_1791.95_1024_1024_4326_city.jpg"),
}

# Smooth long-leg pseudo-flight routes for temporal evaluation.  Keep each route
# in the same broad map region as the previous version, but remove the repeated
# short left/right zig-zags.  Each route now contains a few long straight legs
# joined by broad, interpretable bends so GT-vs-prediction plots are easier to
# read while still testing turns. Coordinates are canonical 4096x4096 px.
ROUTE_SPECS: Dict[str, List[Tuple[int, int]]] = {
    "train_01": [
        (330, 620),
        (980, 760),
        (1600, 910),
        (2200, 1110),
        (2800, 1290),
        (3330, 1480),
    ],
    "train_02": [
        (430, 3080),
        (1050, 2960),
        (1700, 2860),
        (2350, 2740),
        (2950, 2600),
        (3510, 2410),
    ],
    "train_03": [
        (3330, 430),
        (3210, 1060),
        (3350, 1710),
        (3260, 2360),
        (3400, 3010),
        (3310, 3610),
    ],
    "test_01": [
        (560, 1810),
        (1160, 1710),
        (1760, 1760),
        (2360, 1860),
        (2920, 1810),
        (3410, 2050),
    ],
    "test_02": [
        (900, 330),
        (1080, 960),
        (1020, 1610),
        (1230, 2260),
        (1320, 2960),
        (1440, 3690),
    ],
}
TRAIN_ROUTES = ("train_01", "train_02", "train_03")
TEST_ROUTES = ("test_01", "test_02")


def _scale_route(points: Sequence[Tuple[int, int]], width: int, height: int) -> np.ndarray:
    sx, sy = width / float(REFERENCE_SIZE), height / float(REFERENCE_SIZE)
    return np.asarray([(x * sx, y * sy) for x, y in points], dtype=np.float64)


def _route_length_px(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _dense_targets(points: np.ndarray, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a polyline approximately every step_m and return route headings."""
    step_px = float(step_m) / MPP
    rows, headings = [], []
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    distances = list(np.arange(0.0, total + 1e-6, step_px))
    if not distances or total - distances[-1] > 0.5 * step_px:
        distances.append(total)
    for s in distances:
        leg = int(np.clip(np.searchsorted(cumulative, s, side="right") - 1, 0, len(points) - 2))
        local = float(s - cumulative[leg])
        delta = points[leg + 1] - points[leg]
        length = max(float(seg_lengths[leg]), 1e-9)
        unit = delta / length
        rows.append(points[leg] + unit * local)
        headings.append((math.degrees(math.atan2(unit[1], unit[0])) + 360.0) % 360.0)
    return np.asarray(rows, dtype=np.float64), np.asarray(headings, dtype=np.float64)


def _find_metadata(dataset_root: Path) -> Path:
    preferred = dataset_root / "c4m_254k_96bc_b15_s100_v3d" / "metadata" / "metadata.csv"
    if preferred.exists():
        return preferred
    matches = sorted(dataset_root.glob("c4m*_v3d/metadata/metadata.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Cannot find c4m*_v3d/metadata/metadata.csv under %s" % dataset_root)


def _find_satellite(dataset_root: Path, city: str) -> Path:
    _, filename = CITY_TO_RSI[city]
    path = dataset_root / "city_rsi" / filename
    if path.exists():
        return path
    raise FileNotFoundError("Expected official %s RSI at %s" % (city, path))


def _city_rows(df: pd.DataFrame, city: str) -> pd.DataFrame:
    required = {"block_x", "block_y", "x_norm", "y_norm", "target_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("metadata.csv missing columns: %s" % sorted(missing))
    rsi_id, _ = CITY_TO_RSI[city]
    paths = df["target_path"].astype(str).str.replace("\\", "/", regex=False)
    mask = paths.str.contains(f"/{city}/", regex=False) | paths.str.contains(rsi_id, regex=False)
    rows = df.loc[mask].copy()
    if rows.empty:
        raise ValueError("No %s/%s rows found from target_path" % (city, rsi_id))
    # Same conversion used by Bearing-UAV's official cvphr/utils/utils.py.
    rows["global_x_px"] = rows["block_x"].astype(float) * PATCH_SIZE + PATCH_SIZE + rows["x_norm"].astype(float) * PATCH_SIZE
    rows["global_y_px"] = rows["block_y"].astype(float) * PATCH_SIZE + PATCH_SIZE + rows["y_norm"].astype(float) * PATCH_SIZE
    rows = rows[
        rows["global_x_px"].between(0, REFERENCE_SIZE - 1)
        & rows["global_y_px"].between(0, REFERENCE_SIZE - 1)
    ].reset_index(drop=True)
    return rows


def _build_basename_index(dataset_root: Path, city: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in (dataset_root / city).glob("uav_*/*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index.setdefault(p.name, p)
    return index


def _resolve_image_path(raw: str, dataset_root: Path, city: str, index: Dict[str, Path]) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p.resolve()
    normalized = str(raw).replace("\\", "/")
    if "Bearing_UAV_90K/" in normalized:
        q = dataset_root / normalized.split("Bearing_UAV_90K/", 1)[1]
        if q.exists():
            return q.resolve()
    marker = f"/{city}/"
    if marker in normalized:
        q = dataset_root / city / normalized.split(marker, 1)[1]
        if q.exists():
            return q.resolve()
    q = index.get(Path(normalized).name)
    if q is not None:
        return q.resolve()
    raise FileNotFoundError("Cannot resolve target_path: %s" % raw)


def _angular_error_deg(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - float(target) + 180.0) % 360.0 - 180.0)


def _select_sequence(rows, targets, headings, used_global, max_distance_m, heading_weight):
    xy = rows[["global_x_px", "global_y_px"]].to_numpy(dtype=np.float64)
    if "theta" in rows.columns:
        theta = np.mod(rows["theta"].to_numpy(dtype=np.float64), 360.0)
    elif {"x_cosa", "y_sina"}.issubset(rows.columns):
        theta = np.mod(np.degrees(np.arctan2(rows["y_sina"], rows["x_cosa"])), 360.0)
    else:
        theta = None
    max_px = float(max_distance_m) / MPP
    selected = []
    local_used = set()
    for target, heading in zip(targets, headings):
        spatial = np.linalg.norm(xy - target[None, :], axis=1)
        score = spatial.copy()
        if theta is not None:
            score = score + float(heading_weight) * _angular_error_deg(theta, heading)
        pick = None
        for idx in np.argsort(score)[:512]:
            idx = int(idx)
            identity = str(rows.iloc[idx]["target_path"])
            if idx in local_used or identity in used_global or spatial[idx] > max_px:
                continue
            pick = idx
            break
        if pick is None:
            raise RuntimeError(
                "No unused UAV sample within %.1fm of planned route point; nearest=%.2fm"
                % (max_distance_m, float(spatial.min()) * MPP)
            )
        selected.append(pick)
        local_used.add(pick)
        used_global.add(str(rows.iloc[pick]["target_path"]))
    return selected


def _write_route(route_dir: Path, route_name: str, planned: np.ndarray, selected, image_paths) -> None:
    route_dir.mkdir(parents=True, exist_ok=True)
    fields = ["frame_id", "image_path", "x_px", "y_px", "x_m", "y_m", "yaw_deg", "timestamp_ns", "source_index", "source_target_path"]
    with (route_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame_id, ((source_idx, row), image_path) in enumerate(zip(selected.iterrows(), image_paths)):
            if "theta" in selected.columns:
                yaw = float(row["theta"]) % 360.0
            elif {"x_cosa", "y_sina"}.issubset(selected.columns):
                yaw = math.degrees(math.atan2(float(row["y_sina"]), float(row["x_cosa"]))) % 360.0
            else:
                yaw = 0.0
            x_px, y_px = float(row["global_x_px"]), float(row["global_y_px"])
            writer.writerow({
                "frame_id": frame_id,
                "image_path": str(image_path),
                "x_px": f"{x_px:.6f}", "y_px": f"{y_px:.6f}",
                "x_m": f"{x_px * MPP:.6f}", "y_m": f"{y_px * MPP:.6f}",
                "yaw_deg": f"{yaw:.6f}",
                "timestamp_ns": frame_id * 1_000_000_000,
                "source_index": int(source_idx),
                "source_target_path": str(row["target_path"]),
            })
    payload = {
        "route_name": route_name,
        "coordinate_system": "bearing_rsi_map_meters",
        "mpp": MPP,
        "waypoints": [
            {
                "waypoint_order": i,
                "latitude": float(y * MPP),
                "longitude": float(x * MPP),
                "pixel_x": float(x),
                "pixel_y": float(y),
            }
            for i, (x, y) in enumerate(planned)
        ],
    }
    (route_dir / "waypoints.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_train_union(output_root: Path) -> None:
    out = output_root / "routes" / "route_A"
    out.mkdir(parents=True, exist_ok=True)
    all_rows, fields, frame_id = [], None, 0
    for name in TRAIN_ROUTES:
        with (output_root / "routes" / name / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for row in reader:
                row = dict(row)
                row["frame_id"] = str(frame_id)
                row["timestamp_ns"] = str(frame_id * 1_000_000_000)
                all_rows.append(row)
                frame_id += 1
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def _draw_preview(sat_path: Path, routes: Dict[str, np.ndarray], out_path: Path) -> None:
    image = Image.open(sat_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    palette = {
        "train_01": (0, 220, 80, 255), "train_02": (0, 145, 255, 255),
        "train_03": (255, 185, 0, 255), "test_01": (255, 60, 80, 255),
        "test_02": (190, 70, 255, 255),
    }
    for name, points in routes.items():
        pts = [(int(round(x)), int(round(y))) for x, y in points]
        draw.line(pts, fill=palette[name], width=max(5, image.width // 700), joint="curve")
        r = max(7, image.width // 500)
        for i, (x, y) in enumerate(pts):
            draw.ellipse((x-r, y-r, x+r, y+r), fill=palette[name], outline=(255,255,255,255), width=2)
            if i in (0, len(pts)-1):
                draw.text((x+r+2, y-r), f"{name}:{'S' if i == 0 else 'E'}", fill=(255,255,255,255))
    draw.rounded_rectangle((12, 12, 410, 186), radius=12, fill=(0,0,0,155))
    for i, name in enumerate(routes):
        y = 24 + i*30
        draw.line((24, y+8, 66, y+8), fill=palette[name], width=6)
        split = "TRAIN" if name in TRAIN_ROUTES else "TEST"
        draw.text((78, y), f"{name}  [{split}]", fill=(255,255,255,255))
    image.save(out_path, quality=94)


def prepare(args):
    dataset_root = Path(args.dataset_root).resolve()
    city = args.city.lower()
    output_root = Path(args.output_root).resolve() if args.output_root else Path(__file__).resolve().parent / "generated" / city
    output_root.mkdir(parents=True, exist_ok=True)
    sat_path = _find_satellite(dataset_root, city)
    metadata_path = _find_metadata(dataset_root)
    with Image.open(sat_path) as im:
        width, height = im.size
    if (width, height) != (REFERENCE_SIZE, REFERENCE_SIZE):
        raise ValueError("Expected official city RSI %dx%d, got %s" % (REFERENCE_SIZE, REFERENCE_SIZE, (width, height)))
    df = pd.read_csv(metadata_path)
    rows = _city_rows(df, city)
    basename_index = _build_basename_index(dataset_root, city)
    routes = {name: _scale_route(points, width, height) for name, points in ROUTE_SPECS.items()}
    used_global, stats = set(), {}
    for name in (*TRAIN_ROUTES, *TEST_ROUTES):
        planned = routes[name]
        targets, headings = _dense_targets(planned, args.step_m)
        ids = _select_sequence(rows, targets, headings, used_global, args.max_sample_distance_m, args.heading_weight_px_per_deg)
        selected = rows.iloc[ids].copy()
        paths = [_resolve_image_path(v, dataset_root, city, basename_index) for v in selected["target_path"]]
        _write_route(output_root / "routes" / name, name, planned, selected, paths)
        err = np.linalg.norm(selected[["global_x_px", "global_y_px"]].to_numpy() - targets, axis=1) * MPP
        stats[name] = {
            "split": "train" if name in TRAIN_ROUTES else "inference",
            "planned_length_m": _route_length_px(planned) * MPP,
            "waypoints": len(planned), "frames": len(selected),
            "sample_step_m": float(args.step_m),
            "mean_nearest_sample_error_m": float(err.mean()),
            "max_nearest_sample_error_m": float(err.max()),
        }
        print(name, stats[name], flush=True)
    _make_train_union(output_root)
    sat_meta = {
        "mode": "bearing_uav_pixel_meter", "mpp": MPP, "width": width, "height": height,
        "city": city, "rsi_id": CITY_TO_RSI[city][0], "satellite_image": str(sat_path),
        "metadata_csv": str(metadata_path),
    }
    (output_root / "bearing_satellite.json").write_text(json.dumps(sat_meta, indent=2), encoding="utf-8")
    experiment = {
        "dataset_root": str(dataset_root), "city": city, "satellite_image": str(sat_path),
        "metadata_csv": str(metadata_path), "mpp": MPP,
        "train_routes": list(TRAIN_ROUTES), "inference_routes": list(TEST_ROUTES),
        "route_stats": stats,
        "note": "Independent Bearing-UAV observations are selected into disjoint pseudo-flight sequences.",
    }
    (output_root / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    preview = output_root / "route_plan_full_satellite.jpg"
    _draw_preview(sat_path, routes, preview)
    print("preview:", preview, flush=True)
    return output_root


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", default="cityb", choices=sorted(CITY_TO_RSI))
    p.add_argument("--output-root", default=None)
    # Keep pseudo-frame displacement inside the unmodified v39 temporal motion
    # envelope instead of enlarging the model's speed/acceleration caps.
    p.add_argument("--step-m", type=float, default=8.0)
    p.add_argument("--max-sample-distance-m", type=float, default=15.0)
    p.add_argument("--heading-weight-px-per-deg", type=float, default=0.35)
    return p


if __name__ == "__main__":
    prepare(build_parser().parse_args())
