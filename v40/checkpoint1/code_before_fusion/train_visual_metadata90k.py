"""Legacy GT-centered local-patch visual diagnostic on official metadata.

This script uses frame GT to choose *training and validation inputs*, so its
checkpoints do not satisfy the strict GT-loss-only protocol. Use
train_visual_global_gtloss.py for that protocol instead.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

import image_only_citya as exp
import visual_localizer as vl
from data import image_transform
from visual_model import AllMapGeoCLIP

META = Path("/yh/study/cvpr_data/Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv")
CVPR_PARENT = Path("/yh/study/cvpr_data")


@torch.no_grad()
def gallery_embeddings(model, gallery, device):
    """Mining pool is recomputed per epoch; only indices are detached."""
    model.eval()
    chunks = []
    for start in range(0, len(gallery["xy"]), 1024):
        end = min(start + 1024, len(gallery["xy"]))
        chunks.append(model.encode_sat_from_clip(
            gallery["clip_feat"][start:end].to(device).float(),
            gallery["xy"][start:end].to(device)))
    return torch.cat(chunks)


def mined_negative_loss(model, item, gallery, batch, local_logits,
                        mined_sat, count, device):
    """GT excludes nearby positives; retrieval scores select distant hard negatives."""
    uav = model.encode_uav_from_clip(item["uav_clip"][batch].to(device).float())
    gt = item["gt_xy"][batch].to(device)
    coords = gallery["xy"].to(device)
    with torch.no_grad():
        similarity = uav.detach() @ mined_sat.T
        # A nearby tile may still contain the UAV; do not label it negative.
        similarity.masked_fill_(torch.cdist(gt, coords) < 40.0, -float("inf"))
        negative_index = similarity.topk(count, dim=1).indices
    sat_clip = gallery["clip_feat"][negative_index.cpu()].to(device).float()
    sat_xy = coords[negative_index]
    z_sat = model.encode_sat_from_clip(
        sat_clip.reshape(-1, sat_clip.shape[-1]),
        sat_xy.reshape(-1, 2)).reshape(len(batch), count, -1)
    negative_logits = model.logit_scale.exp().clamp(max=100.0) * (
        uav[:, None] * z_sat).sum(dim=2)
    target = item["target_indices"][batch].to(device)
    return F.cross_entropy(torch.cat([local_logits, negative_logits], dim=1), target)


def load_rows(city, limit, seed, official96_train_only=False):
    exp.configure(city)
    official_train_indices = None
    if official96_train_only:
        with META.open(newline="") as f:
            total = sum(1 for _ in f) - 1
        if total != 90000:
            raise RuntimeError("official96 metadata must contain 90,000 rows")
        permutation = torch.randperm(total, generator=torch.Generator().manual_seed(42))
        official_train_indices = set(permutation[:int(0.85 * total)].tolist())
    test_paths = set()
    for route_root in exp.config.ROUTE_ROOTS[1:]:
        with (route_root / "manifest.csv").open(newline="") as f:
            test_paths.update(row["image_path"] for row in csv.DictReader(f))
    digest = hashlib.sha256("\n".join(sorted(test_paths)).encode()).hexdigest()
    rows = []
    with META.open(newline="") as f:
        for official_index, row in enumerate(csv.DictReader(f)):
            if official_train_indices is not None and official_index not in official_train_indices:
                continue
            if "/%s/" % city not in row["target_path"]:
                continue
            image_path = str((CVPR_PARENT / row["target_path"].lstrip("./")).resolve())
            if image_path in test_paths:
                continue
            x_px = (float(row["block_x"]) * 256.0 + 256.0
                    + float(row["x_norm"]) * 128.0)
            y_px = (float(row["block_y"]) * 256.0 + 256.0
                    + float(row["y_norm"]) * 128.0)
            rows.append((image_path, x_px * 0.25, y_px * 0.25))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:limit]
    if len(rows) < limit or any(not Path(p).is_file() for p, _, _ in rows):
        raise RuntimeError("missing/insufficient official metadata images for " + city)
    if any(p in test_paths for p, _, _ in rows):
        raise RuntimeError("held-out image leaked into visual training")
    return rows, digest


def cache_city(city, rows, gallery, origin_lat, origin_lon, model, device, out,
               front18=False):
    paths = [row[0] for row in rows]
    path_digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:12]
    mode_name = "front18" if front18 else "grid36"
    cache_path = out / (city + "_" + exp.config.BACKBONE_KEY + "_" + mode_name + "_" + path_digest
                        + "_metadata_features.pt")
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("image_paths") == paths:
            print("[VISUAL 90K CACHE] reuse", city, flush=True)
            return payload
    transform = image_transform(False, source="uav")
    features = []
    for start in range(0, len(rows), 64):
        images = []
        for path, _, _ in rows[start:start + 64]:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        with torch.no_grad():
            features.append(model.encode_clip_image(torch.stack(images).to(device)).cpu().half())
        if start % 640 == 0:
            print("[VISUAL 90K CACHE]", city, min(start + 64, len(rows)), "/", len(rows), flush=True)
    gt_xy = torch.tensor([[x - origin_lon, y - origin_lat] for _, x, y in rows], dtype=torch.float32)
    xy = gallery["xy"].numpy()
    pixels = gallery["pixel"].numpy()
    tree = cKDTree(xy)
    jitter = np.random.default_rng(1000 + len(rows)).uniform(-6.0, 6.0, (len(rows), 2))
    if front18:
        angles = np.random.default_rng(2000 + len(rows)).uniform(-np.pi, np.pi, len(rows))
        forward = np.column_stack((np.cos(angles), np.sin(angles)))
        grid_centers = (gt_xy.numpy() + jitter
                        - float(exp.config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M) * forward)
    else:
        forward = None
        grid_centers = gt_xy.numpy() + jitter
    nearest = tree.query(grid_centers)[1]
    lookup = {(int(round(x)), int(round(y))): index for index, (x, y) in enumerate(pixels)}
    stride = int(exp.config.SAT_STRIDE)
    candidate_rows = []
    for i, center_idx in enumerate(nearest):
        px, py = (int(round(v)) for v in pixels[center_idx])
        indices = [lookup.get((px + dx * stride, py + dy * stride))
                   for dy in range(-3, 3) for dx in range(-3, 3)]
        if any(index is None for index in indices):
            indices = tree.query(gt_xy[i].numpy(), k=36)[1].tolist()
        if front18:
            projection = (xy[indices] - grid_centers[i]) @ forward[i]
            selected = np.argsort(projection)[-18:]
            indices = [indices[j] for j in selected]
        candidate_rows.append(indices)
    candidate_indices = torch.tensor(candidate_rows, dtype=torch.long)
    centers = gallery["xy"][candidate_indices]
    distance = torch.linalg.norm(centers - gt_xy[:, None, :], dim=2)
    payload = {
        "image_paths": paths,
        "uav_clip": torch.cat(features),
        "gt_xy": gt_xy,
        "candidate_indices": candidate_indices,
        "target_indices": distance.argmin(dim=1),
        "capture": distance.min(dim=1).values <= float(exp.config.CANDIDATE_CAPTURE_RADIUS_M),
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print("[VISUAL 90K CACHE]", city, "capture", float(payload["capture"].float().mean()) * 100, flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--samples-per-city", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--coord-loss-weight", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hard-negatives", type=int, default=0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--front18-train", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        default=Path("/yh/study/uav-sat/v40/cache/visual90k"))
    parser.add_argument("--seed", type=int, default=2033)
    args = parser.parse_args()
    if exp.config.BACKBONE_KEY != "mobilenet_v3_small":
        parser.error("v39 architecture requires UAVSAT_BACKBONE=mobilenet_v3_small")
    if (args.samples_per_city < 100 or args.epochs < 1 or args.lr <= 0
            or args.coord_loss_weight < 0 or not 0 <= args.label_smoothing < 1
            or args.hard_negatives < 0 or args.hard_negative_weight < 0):
        parser.error("invalid training hyperparameters")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    vl.config.VISUAL_COORD_LOSS_WEIGHT = args.coord_loss_weight
    vl.config.VISUAL_LABEL_SMOOTHING = args.label_smoothing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp.configure(args.cities[0])
    base_checkpoint = torch.load(exp.config.VISUAL_CHECKPOINT, map_location="cpu")
    model = AllMapGeoCLIP().to(device)
    checkpoint_dim = int(base_checkpoint["model"]["uav_head.0.weight"].shape[1])
    if checkpoint_dim != exp.config.CLIP_DIM:
        raise RuntimeError("source visual backbone differs from v39 MobileNetV3-Small")
    vl._load_task_specific_state(model, base_checkpoint["model"])
    model.clip.eval()
    training = {}
    heldout_hashes = {}
    galleries = {}
    origins = {}
    for city in args.cities:
        exp.configure(city)
        checkpoint = torch.load(exp.config.VISUAL_CHECKPOINT, map_location="cpu")
        if checkpoint.get("gallery_city") != city:
            raise RuntimeError("gallery city mismatch: " + city)
        gallery = checkpoint["gallery"]
        rows, heldout_hashes[city] = load_rows(city, args.samples_per_city, args.seed)
        origin_lat, origin_lon = float(checkpoint["origin_lat"]), float(checkpoint["origin_lon"])
        training[city] = cache_city(
            city, rows, gallery, origin_lat, origin_lon, model, device,
            args.cache_root, front18=args.front18_train)
        galleries[city] = gallery
        origins[city] = (origin_lat, origin_lon)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        order = list(args.cities)
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for city in order:
            item = training[city]
            gallery = galleries[city]
            cut = int(len(item["image_paths"]) * 0.9)
            indices = list(range(cut))
            random.Random(args.seed + epoch + len(city)).shuffle(indices)
            cache = vl.VisualTrainCache(
                item["uav_clip"], item["gt_xy"], torch.arange(len(item["image_paths"])),
                item["candidate_indices"], item["target_indices"], item["capture"])
            mined_sat = (gallery_embeddings(model, gallery, device)
                         if args.hard_negatives else None)
            model.train()
            model.clip.eval()
            for offset in range(0, len(indices), 64):
                batch = torch.tensor(indices[offset:offset + 64])
                if not bool(item["capture"][batch].all()):
                    batch = batch[item["capture"][batch]]
                if len(batch) == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss, logits, _, _ = vl._visual_forward_batch(model, cache, gallery, batch, device)
                if mined_sat is not None:
                    loss = loss + args.hard_negative_weight * mined_negative_loss(
                        model, item, gallery, batch, logits, mined_sat,
                        args.hard_negatives, device)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite visual loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        vals = {}
        for city in args.cities:
            item = training[city]
            cut = int(len(item["image_paths"]) * 0.9)
            cache = vl.VisualTrainCache(
                item["uav_clip"], item["gt_xy"], torch.arange(len(item["image_paths"])),
                item["candidate_indices"], item["target_indices"], item["capture"])
            vals[city] = vl._evaluate_visual_model(
                model, cache, galleries[city], list(range(cut, len(item["image_paths"]))), device)
        score = float(np.mean([v["SoftMS_MLE_m"] for v in vals.values()]))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "val": vals,
               "val_candidate_capture_pct": {
                   city: float(training[city]["capture"][int(len(training[city]["image_paths"]) * 0.9):]
                               .float().mean().item() * 100.0)
                   for city in args.cities},
               "macro_val_mle": score}
        history.append(row)
        print("[VISUAL 90K]", json.dumps(row), flush=True)
        if score < best:
            best = score
            best_state = vl._task_specific_state_dict(model)
    for city in args.cities:
        exp.configure(city)
        source = torch.load(exp.config.VISUAL_CHECKPOINT, map_location="cpu")
        source.update({
            "model": best_state,
            "best_model": best_state,
            "gallery": galleries[city],
            "origin_lat": origins[city][0],
            "origin_lon": origins[city][1],
            "gallery_city": city,
            "training_protocol": "metadata90k_excluding_route_tests",
            "visual_train_routes": ["official_metadata_train"],
            "visual_validation_routes": ["official_metadata_validation"],
            "previous_task_checkpoint_loaded": True,
            "backbone_source": exp.config.BACKBONE_NAME,
            "excluded_test_manifest_sha256": heldout_hashes[city],
            "fine_tuned_cities": list(args.cities),
            "fine_tuned_samples_per_city": args.samples_per_city,
            "best_score": best,
        })
        path = args.output / "checkpoints" / city / exp.VISUAL_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(source, path)
    report = {"cities": args.cities, "samples_per_city": args.samples_per_city,
              "coord_loss_weight": args.coord_loss_weight,
              "label_smoothing": args.label_smoothing,
              "hard_negatives": args.hard_negatives,
              "hard_negative_weight": args.hard_negative_weight,
              "backbone": exp.config.BACKBONE_KEY,
              "front18_train": args.front18_train,
              "test_paths_excluded_sha256": heldout_hashes, "best_val_mle": best,
              "history": history, "checkpoint_root": str(args.output / "checkpoints")}
    (args.output / "visual_report.json").write_text(json.dumps(report, indent=2))
    print("[VISUAL 90K DONE]", args.output / "visual_report.json", flush=True)


if __name__ == "__main__":
    main()
