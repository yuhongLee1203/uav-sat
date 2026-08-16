#!/usr/bin/env python3
"""Prepare the minimum class-paired dataset expected by native CVGL papers.

No local candidate list is produced.  Every test UAV query is compared by the
paper implementation against one fixed gallery containing all satellite
locations occurring in Route A/B/C.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "v36-exp"
BASE = ROOT / "outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
OUTPUT = EXP / "native-paper-data" / "U1652"
MANIFEST = EXP / "native-paper-data" / "manifest.json"
SAT_CROP = 320


def ensure_link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and not target.is_symlink():
        target.symlink_to(source.resolve())


def main():
    checkpoint = torch.load(
        BASE / "checkpoints/visual_retrieval_A_only.pt", map_location="cpu"
    )
    gallery = checkpoint["gallery"]
    gallery_xy = gallery["xy"].float().cpu().numpy()
    gallery_pixel = np.rint(gallery["pixel"].cpu().numpy()).astype(np.int64)
    tree = cKDTree(gallery_xy)

    routes = {}
    used_gallery = set()
    for route_name in ("route_A", "route_B", "route_C"):
        payload = torch.load(
            BASE / "feature_cache" / (route_name + "_uav_clip.pt"),
            map_location="cpu",
        )
        gt_xy = payload["gt_xy"].float().cpu().numpy()
        nearest = tree.query(gt_xy, k=1)[1].astype(np.int64)
        used_gallery.update(int(value) for value in nearest)
        routes[route_name] = {
            "frame_ids": payload["frame_ids"].long().cpu().numpy(),
            "gt_xy": gt_xy,
            "image_paths": payload["image_paths"],
            "gallery_rows": nearest,
        }

    ordered_gallery = sorted(used_gallery)
    class_for_gallery = {
        gallery_row: index for index, gallery_row in enumerate(ordered_gallery)
    }
    patch_root = EXP / "native-paper-data" / "satellite_patches"
    patch_root.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(__import__("config").SAT_IMAGE) as satellite:
        satellite = satellite.convert("RGB")
        half = SAT_CROP // 2
        for number, gallery_row in enumerate(ordered_gallery):
            class_id = class_for_gallery[gallery_row]
            patch_path = patch_root / ("%06d.jpg" % class_id)
            if not patch_path.exists():
                px, py = gallery_pixel[gallery_row]
                patch = satellite.crop(
                    (int(px - half), int(py - half), int(px + half), int(py + half))
                )
                patch.save(patch_path, quality=95)
            if number == 0 or (number + 1) % 500 == 0 or number + 1 == len(ordered_gallery):
                print("satellite gallery %d/%d" % (number + 1, len(ordered_gallery)), flush=True)

    query_rows = []
    for route_name, payload in routes.items():
        is_train = route_name == "route_A"
        for frame_id, gt_xy, image_path, gallery_row in zip(
            payload["frame_ids"], payload["gt_xy"], payload["image_paths"], payload["gallery_rows"]
        ):
            class_id = class_for_gallery[int(gallery_row)]
            class_name = "%06d" % class_id
            source = Path(image_path)
            output_name = "%s_f%06d%s" % (route_name, int(frame_id), source.suffix.lower())
            if is_train:
                ensure_link(source, OUTPUT / "train/drone" / class_name / output_name)
                ensure_link(
                    patch_root / (class_name + ".jpg"),
                    OUTPUT / "train/satellite" / class_name / (class_name + ".jpg"),
                )
            else:
                ensure_link(source, OUTPUT / "test/query_drone" / class_name / output_name)
                query_rows.append(
                    {
                        "route": route_name,
                        "frame_id": int(frame_id),
                        "file": output_name,
                        "class_id": class_id,
                        "gt_xy": [float(gt_xy[0]), float(gt_xy[1])],
                    }
                )

    # The test gallery is fixed once and is never selected per query.
    for gallery_row in ordered_gallery:
        class_id = class_for_gallery[gallery_row]
        class_name = "%06d" % class_id
        ensure_link(
            patch_root / (class_name + ".jpg"),
            OUTPUT / "test/gallery_satellite" / class_name / (class_name + ".jpg"),
        )

    manifest = {
        "protocol": "native class-paired training; fixed global gallery; no local prior",
        "train_route": "route_A",
        "test_routes": ["route_B", "route_C"],
        "gallery_count": len(ordered_gallery),
        "gallery": {
            str(class_for_gallery[row]): {
                "source_gallery_row": int(row),
                "xy": [float(gallery_xy[row, 0]), float(gallery_xy[row, 1])],
            }
            for row in ordered_gallery
        },
        "queries": query_rows,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("native dataset: %s" % OUTPUT)
    print("manifest: %s" % MANIFEST)


if __name__ == "__main__":
    main()
