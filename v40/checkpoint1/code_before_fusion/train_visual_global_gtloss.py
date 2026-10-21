"""Fresh MobileNetV3-Small heads: whole-map input, GT appears only in loss.

The satellite gallery is fixed before any UAV GT is read. Every forward pass
scores every satellite location; labels only define a Gaussian target for the
loss. Custom held-out test-route images are excluded from training/validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import image_only_citya as exp
import train_visual_metadata90k as metadata
import visual_localizer as vl
from data import image_transform
from visual_model import AllMapGeoCLIP


@torch.no_grad()
def cache_uav_features(city, rows, origin_lat, origin_lon, model, device, root):
    """Cache image features and loss labels, never GT-selected candidates."""
    paths = [row[0] for row in rows]
    digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:12]
    cache_path = root / (city + "_mobilenet_global_" + digest + ".pt")
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("image_paths") == paths and cached["uav_clip"].shape[-1] == 576:
            print("[GLOBAL UAV CACHE] reuse", cache_path, flush=True)
            return cached
    transform = image_transform(False, source="uav")
    features = []
    for start in range(0, len(rows), 64):
        images = []
        for path, _, _ in rows[start:start + 64]:
            with Image.open(path) as picture:
                images.append(transform(picture.convert("RGB")))
        features.append(model.encode_clip_image(torch.stack(images).to(device)).cpu().half())
        if start % 640 == 0:
            print("[GLOBAL UAV CACHE]", city, min(start + 64, len(rows)), "/", len(rows), flush=True)
    payload = {
        "image_paths": paths,
        "uav_clip": torch.cat(features),
        "gt_xy": torch.tensor([[x - origin_lon, y - origin_lat] for _, x, y in rows],
                              dtype=torch.float32),
        "backbone": "mobilenet_v3_small",
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


@torch.no_grad()
def evaluate(model, uav, gt, sat_clip, sat_xy, device, batch_size):
    model.eval()
    sat = model.encode_sat_from_clip(sat_clip, sat_xy)
    errors = []
    for start in range(0, len(uav), batch_size):
        z_uav = model.encode_uav_from_clip(uav[start:start + batch_size])
        logits = model.logit_scale.exp().clamp(max=100.0) * (z_uav @ sat.T)
        xy = sat_xy[logits.argmax(dim=1)]
        errors.append(torch.linalg.norm(xy - gt[start:start + batch_size], dim=1).cpu())
    error = torch.cat(errors)
    return {"global_top1_MLE_m": float(error.mean()),
            "global_top1_P90_m": float(torch.quantile(error, 0.9)),
            "global_top1_LSR15_pct": float((error <= 15).float().mean() * 100)}


def local_window_table(gallery, stride, device):
    """Precompute a fixed 6x6 SAT window for each gallery center (no GT)."""
    pixels = gallery["pixel"].tolist()
    index = vl.build_pixel_index(gallery["pixel"])
    rows = []
    for center, (x, y) in enumerate(pixels):
        x, y = int(round(x)), int(round(y))
        rows.append([index.get((x + dx * stride, y + dy * stride), center)
                     for dy in range(-3, 3) for dx in range(-3, 3)])
    return torch.tensor(rows, dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--target-sigma-m", type=float, default=4.0)
    parser.add_argument("--distance-loss-weight", type=float, default=0.0,
                        help="GT-loss-only expected-distance penalty, enabled after warmup epoch")
    parser.add_argument("--local-loss-weight", type=float, default=0.0,
                        help="Extra GT loss on image-selected local SAT windows")
    parser.add_argument("--local-topk-windows", type=int, default=3,
                        help="Number of top image-only gallery centers for local loss")
    parser.add_argument("--official96-train-only", action="store_true",
                        help="Restrict source rows to c4m/ID96 seed-42 train split")
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Continue the same strict GT-loss-only visual task")
    parser.add_argument("--rebuild-satellite-gallery", action="store_true",
                        help="Re-encode SAT patches for a changed V40_SAT_CROP_SIZE_M")
    parser.add_argument("--cache-root", type=Path,
                        default=Path("/yh/study/uav-sat/v40/cache/visual90k"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2033)
    args = parser.parse_args()
    if exp.config.BACKBONE_KEY != "mobilenet_v3_small":
        parser.error("v39 architecture requires MobileNetV3-Small")
    if (args.samples < 100 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.distance_loss_weight < 0 or args.local_loss_weight < 0
            or args.local_topk_windows < 1):
        parser.error("invalid training sizes or learning rate")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp.configure(args.city)
    source = torch.load(exp.config.VISUAL_CHECKPOINT, map_location="cpu")
    rows, excluded_hash = metadata.load_rows(
        args.city, args.samples, args.seed,
        official96_train_only=args.official96_train_only)
    model = AllMapGeoCLIP().to(device)  # task heads are fresh; backbone remains pretrained/frozen
    resumed_best_mle = float("inf")
    resumed_best_state = None
    if args.resume_checkpoint:
        previous = torch.load(args.resume_checkpoint, map_location="cpu")
        if (previous.get("training_protocol") != "global_gallery_gt_loss_only"
                or previous.get("previous_task_checkpoint_loaded") is not False
                or previous.get("gallery_city") != args.city
                or previous.get("official96_train_only") != args.official96_train_only
                or previous.get("fine_tuned_samples_per_city") != args.samples):
            raise RuntimeError("resume checkpoint is not the same audited visual task")
        if previous.get("excluded_test_manifest_sha256") != excluded_hash:
            raise RuntimeError("resume checkpoint was trained against a different held-out test manifest")
        vl._load_task_specific_state(model, previous["model"])
        resumed_best_mle = float(previous["best_score"])
        resumed_best_state = previous["model"]
    if args.rebuild_satellite_gallery:
        source["gallery"] = vl._build_satellite_backbone_gallery(
            model, float(source["origin_lat"]), float(source["origin_lon"]), device)
        # A score obtained with a different crop cannot select these weights.
        resumed_best_mle = float("inf")
        resumed_best_state = None
    gallery = source["gallery"]
    cache = cache_uav_features(args.city, rows, float(source["origin_lat"]),
                               float(source["origin_lon"]), model, device,
                               args.cache_root)
    if cache["image_paths"] != [row[0] for row in rows]:
        raise RuntimeError("UAV feature cache does not match training rows")
    # No candidate indices are generated or read: every forward pass scores
    # the entire fixed satellite gallery.
    sat_clip = gallery["clip_feat"].to(device).float()
    sat_xy = gallery["xy"].to(device)
    window_table = (local_window_table(gallery, exp.config.SAT_STRIDE, device)
                    if args.local_loss_weight else None)
    uav = cache["uav_clip"].to(device).float()
    gt = cache["gt_xy"].to(device)
    cut = int(len(rows) * 0.9)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=1e-4)
    best_mle = resumed_best_mle
    best_state = resumed_best_state
    history = []
    for epoch in range(args.epochs):
        model.train()
        model.clip.eval()
        order = torch.randperm(cut, generator=torch.Generator().manual_seed(args.seed + epoch))
        losses = []
        local_captures = []
        for start in range(0, cut, args.batch_size):
            batch = order[start:start + args.batch_size].to(device)
            z_sat = model.encode_sat_from_clip(sat_clip, sat_xy)
            z_uav = model.encode_uav_from_clip(uav[batch])
            logits = model.logit_scale.exp().clamp(max=100.0) * (z_uav @ z_sat.T)
            # GT is used only to construct the soft target inside the loss.
            d2 = torch.cdist(gt[batch], sat_xy).square()
            target = torch.softmax(-d2 / (2.0 * args.target_sigma_m ** 2), dim=1)
            loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            if args.distance_loss_weight and epoch >= 1:
                # Every gallery item was already scored; GT only measures
                # how costly the predicted distribution is inside this loss.
                expected_distance = (
                    F.softmax(logits, dim=1) * torch.sqrt(d2 + 1.0)
                ).sum(dim=1).mean() / 100.0
                loss = loss + args.distance_loss_weight * expected_distance
            if args.local_loss_weight:
                # Window centers are selected from image logits only.  The GT
                # may mask a loss term, but never selects a forward candidate.
                count = min(args.local_topk_windows, logits.shape[1])
                selected_centers = logits.detach().topk(count, dim=1).indices
                window_indices = window_table[selected_centers]
                flattened = window_indices.reshape(len(batch), -1)
                local_logits = logits.gather(1, flattened).reshape(len(batch), count, 36)
                local_d2 = d2.gather(1, flattened).reshape(len(batch), count, 36)
                capture = (local_d2.min(dim=2).values.sqrt()
                           <= float(exp.config.CANDIDATE_CAPTURE_RADIUS_M))
                local_captures.append(float(capture.any(dim=1).float().mean().detach().cpu()))
                local_target = torch.softmax(
                    -local_d2 / (2.0 * args.target_sigma_m ** 2), dim=2)
                local_ce = -(local_target * F.log_softmax(local_logits, dim=2)).sum(dim=2)
                loss = loss + args.local_loss_weight * (
                    (local_ce * capture).sum() / capture.sum().clamp_min(1))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite global gallery loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (start // args.batch_size) % 10 == 0:
                print("[GLOBAL GTLOSS] epoch=%d samples=%d/%d loss=%.4f" % (
                    epoch + 1, min(start + len(batch), cut), cut, losses[-1]), flush=True)
        metrics = evaluate(model, uav[cut:], gt[cut:], sat_clip, sat_xy,
                           device, args.batch_size)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
               "image_selected_local_capture_pct": (
                   100.0 * sum(local_captures) / len(local_captures)
                   if local_captures else None),
               "validation": metrics}
        history.append(row)
        print("[GLOBAL GTLOSS]", json.dumps(row), flush=True)
        if metrics["global_top1_MLE_m"] < best_mle:
            best_mle = metrics["global_top1_MLE_m"]
            best_state = vl._task_specific_state_dict(model)
    final_state = vl._task_specific_state_dict(model)
    final_val_mle = float(history[-1]["validation"]["global_top1_MLE_m"])
    source.update({
        "model": best_state,
        "best_model": best_state,
        "gallery_city": args.city,
        "training_protocol": "global_gallery_gt_loss_only",
        "previous_task_checkpoint_loaded": False,
        "excluded_test_manifest_sha256": excluded_hash,
        "backbone_source": exp.config.BACKBONE_NAME,
        "visual_train_routes": ["official_metadata_train_global_gallery"],
        "visual_validation_routes": ["official_metadata_val_global_gallery"],
        "fine_tuned_samples_per_city": args.samples,
        "best_score": best_mle,
        "distance_loss_weight": args.distance_loss_weight,
        "local_loss_weight": args.local_loss_weight,
        "local_topk_windows": args.local_topk_windows,
        "official96_train_only": args.official96_train_only,
        "sat_crop_size_m": float(exp.config.SAT_CROP_SIZE) * float(json.loads((exp.PREPARED / "experiment.json").read_text())["mpp"]),
        "sat_crop_size_px": int(exp.config.SAT_CROP_SIZE),
        "satellite_gallery_rebuilt": args.rebuild_satellite_gallery,
        "same_task_checkpoint_resumed": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
    })
    checkpoint = args.output / "checkpoints" / args.city / exp.VISUAL_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, checkpoint)
    final_source = dict(source)
    final_source["model"] = final_state
    final_source["best_model"] = final_state
    final_source["best_score"] = final_val_mle
    final_source["checkpoint_selection"] = "last_epoch_for_route_validation_only"
    last_checkpoint = args.output / "checkpoints_last" / args.city / exp.VISUAL_NAME
    last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_source, last_checkpoint)
    report = {"city": args.city, "samples": args.samples,
              "training_GT_loss_only": True, "GT_selected_forward_inputs": False,
              "previous_task_checkpoint_loaded": False,
              "excluded_test_manifest_sha256": excluded_hash,
              "best_val_global_top1_MLE_m": best_mle, "history": history,
              "distance_loss_weight": args.distance_loss_weight,
              "local_loss_weight": args.local_loss_weight,
              "local_topk_windows": args.local_topk_windows,
              "official96_train_only": args.official96_train_only,
              "sat_crop_size_m": float(exp.config.SAT_CROP_SIZE) * float(json.loads((exp.PREPARED / "experiment.json").read_text())["mpp"]),
              "sat_crop_size_px": int(exp.config.SAT_CROP_SIZE),
              "satellite_gallery_rebuilt": args.rebuild_satellite_gallery,
              "same_task_checkpoint_resumed": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
              "checkpoint": str(checkpoint)}
    report["last_epoch_checkpoint"] = str(last_checkpoint)
    report["last_epoch_val_global_top1_MLE_m"] = final_val_mle
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("[GLOBAL GTLOSS DONE]", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
