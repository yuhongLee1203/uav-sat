#!/usr/bin/env python3
"""Build one reusable uint8 image/candidate cache for all paper baselines."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "v36-exp"
BASE = ROOT / "outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
OUT = EXP / "cache/paper_common_256.pt"
IMAGE_SIZE = 256
SAT_CROP = 320
SAT_STRIDE = 32


def to_uint8(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)


def deterministic_jitter(frame_id, route_name, maximum=8.0):
    route_code = sum(ord(ch) for ch in route_name)
    angle = 0.11 * route_code + 0.035 * frame_id
    phase = 0.07 * route_code + 0.017 * frame_id
    fraction = 0.40 + (0.75 - 0.40) * (0.5 + 0.5 * math.sin(phase))
    radius = maximum * fraction
    return np.asarray([radius * math.cos(angle), radius * math.sin(angle)])


def main():
    if OUT.exists():
        print(f"reuse paper image cache: {OUT}")
        return
    checkpoint = torch.load(BASE / "checkpoints/visual_retrieval_A_only.pt", map_location="cpu")
    gallery = checkpoint["gallery"]
    gallery_xy = gallery["xy"].float().cpu().numpy()
    gallery_pixel = np.rint(gallery["pixel"].cpu().numpy()).astype(np.int64)
    pixel_index = {(int(x), int(y)): i for i, (x, y) in enumerate(gallery_pixel)}
    tree = cKDTree(gallery_xy)
    route_payloads = {}
    needed_sat = set()

    for route_name in ("route_A", "route_B", "route_C"):
        payload = torch.load(BASE / f"feature_cache/{route_name}_uav_clip.pt", map_location="cpu")
        gt_xy = payload["gt_xy"].float().cpu().numpy()
        frame_ids = payload["frame_ids"].long().cpu().numpy()
        candidate_rows = []
        positive_rows = []
        last_heading = 0.0
        for i, (frame_id, gt) in enumerate(zip(frame_ids, gt_xy)):
            if i > 0 and np.linalg.norm(gt - gt_xy[i - 1]) > 1e-5:
                delta = gt - gt_xy[i - 1]
                last_heading = math.atan2(float(delta[1]), float(delta[0]))
            elif i == 0 and len(gt_xy) > 1:
                delta = gt_xy[1] - gt_xy[0]
                last_heading = math.atan2(float(delta[1]), float(delta[0]))
            center_xy = gt + deterministic_jitter(int(frame_id), route_name)
            center_idx = int(tree.query(center_xy, k=1)[1])
            px, py = gallery_pixel[center_idx]
            full = []
            for oy in range(-3, 3):
                for ox in range(-3, 3):
                    idx = pixel_index.get((int(px + ox * SAT_STRIDE), int(py + oy * SAT_STRIDE)))
                    if idx is not None:
                        full.append(idx)
            if len(full) != 36:
                full = tree.query(center_xy, k=36)[1].tolist()
            full = np.asarray(full, dtype=np.int64)
            relative = gallery_xy[full] - center_xy[None, :]
            forward = relative @ np.asarray([math.cos(last_heading), math.sin(last_heading)])
            selected = full[np.argpartition(forward, -18)[-18:]]
            candidate_rows.append(selected)
            positive = int(tree.query(gt, k=1)[1])
            positive_rows.append(positive)
            needed_sat.update(int(v) for v in selected)
            needed_sat.add(positive)
        print(f"candidate geometry {route_name}: {len(frame_ids)} frames", flush=True)
        route_payloads[route_name] = {
            "frame_ids": torch.from_numpy(frame_ids.copy()),
            "gt_xy": torch.from_numpy(gt_xy.copy()).float(),
            "image_paths": list(payload["image_paths"]),
            "candidate_gallery": torch.from_numpy(np.stack(candidate_rows)),
            "positive_gallery": torch.tensor(positive_rows, dtype=torch.long),
        }

    sat_indices = np.asarray(sorted(needed_sat), dtype=np.int64)
    sat_lookup = {int(g): i for i, g in enumerate(sat_indices)}
    Image.MAX_IMAGE_PIXELS = None
    import config
    with Image.open(config.SAT_IMAGE) as source:
        source = source.convert("RGB")
        sat_images = []
        half = SAT_CROP // 2
        for n, gallery_index in enumerate(sat_indices):
            px, py = gallery_pixel[gallery_index]
            crop = source.crop((int(px - half), int(py - half), int(px + half), int(py + half)))
            crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            sat_images.append(torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1))
            if n == 0 or (n + 1) % 500 == 0 or n + 1 == len(sat_indices):
                print(f"satellite image cache: {n + 1}/{len(sat_indices)}", flush=True)
    sat_images = torch.stack(sat_images)

    routes = {}
    for route_name, payload in route_payloads.items():
        queries = []
        for n, path in enumerate(payload.pop("image_paths")):
            queries.append(to_uint8(path))
            if n == 0 or (n + 1) % 500 == 0 or n + 1 == len(payload["frame_ids"]):
                print(f"UAV image cache {route_name}: {n + 1}/{len(payload['frame_ids'])}", flush=True)
        payload["query_images"] = torch.stack(queries)
        payload["candidate_sat_rows"] = torch.tensor(
            [[sat_lookup[int(g)] for g in row] for row in payload.pop("candidate_gallery").numpy()],
            dtype=torch.long,
        )
        payload["positive_sat_rows"] = torch.tensor(
            [sat_lookup[int(g)] for g in payload.pop("positive_gallery").numpy()], dtype=torch.long
        )
        routes[route_name] = payload

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "image_size": IMAGE_SIZE,
            "protocol": "same Route-A train / Route-B,C test; controlled GT+jitter forward 3x6 local gallery",
            "sat_images": sat_images,
            "sat_xy": torch.from_numpy(gallery_xy[sat_indices]).float(),
            "sat_gallery_indices": torch.from_numpy(sat_indices),
            "routes": routes,
        },
        OUT,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
