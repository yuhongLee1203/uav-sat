"""Fine-tune MobileNetV3-Small weights without changing the v39 topology.

The full, fixed satellite gallery is always the model input; location GT
appears only when constructing the training loss.  Satellite backbone features
are refreshed after every epoch because the shared MobileNet weights change.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import image_only_citya as exp
import train_visual_metadata90k as metadata
import train_visual_global_gtloss as global_train
import visual_localizer as vl
from data import image_transform, SatPatchGallery
from visual_model import AllMapGeoCLIP


def image_batch(rows, indices, transform, device):
    pictures = []
    for index in indices:
        with Image.open(rows[int(index)][0]) as image:
            pictures.append(transform(image.convert("RGB")))
    return torch.stack(pictures).to(device)


@torch.no_grad()
def validation_features(model, rows, transform, device, batch_size):
    model.eval()
    values = []
    for start in range(0, len(rows), batch_size):
        indices = range(start, min(start + batch_size, len(rows)))
        images = image_batch(rows, indices, transform, device)
        values.append(model.clip.encode_image(images).float())
    return torch.cat(values)


def state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def live_satellite_features(model, z_uav, sat_clip, sat_xy, dataset,
                            topk, random_count, device, gt_nearest_indices=None):
    """Select SAT images for shared-backbone gradient updates."""
    with torch.no_grad():
        scores = z_uav.detach() @ model.encode_sat_from_clip(sat_clip, sat_xy).T
        selected = scores.topk(min(topk, len(sat_clip)), dim=1).indices.flatten()
        if random_count:
            selected = torch.cat((selected, torch.randperm(
                len(sat_clip), device=device)[:random_count]))
        if gt_nearest_indices is not None:
            selected = torch.cat((selected, gt_nearest_indices.reshape(-1)))
        selected = selected.unique()
    images = torch.stack([dataset[int(index)]["sat"]
                          for index in selected.cpu().tolist()]).to(device)
    live = model.clip.encode_image(images)
    return sat_clip.index_copy(0, selected, live), selected, live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-interim", type=Path,
                        help="Resume from a validated best_interim.pt after interruption")
    parser.add_argument("--samples", type=int, default=18000)
    parser.add_argument("--extra-train-routes-dir", type=Path,
                        help="Audited train-only CityA route episodes to oversample")
    parser.add_argument("--extra-route-oversample", type=int, default=0,
                        help="Repetitions of unique audited train route images")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-lr", type=float, default=0.00001)
    parser.add_argument("--head-lr", type=float, default=0.00005)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--target-sigma-m", type=float, default=4.0)
    parser.add_argument("--distance-loss-weight", type=float, default=5.0)
    parser.add_argument("--live-sat-topk", type=int, default=0,
                        help="Backpropagate through image-selected SAT candidates per UAV")
    parser.add_argument("--live-sat-random", type=int, default=16,
                        help="Additional uniformly selected SAT patches when live SAT is enabled")
    parser.add_argument("--live-sat-gt-nearest", type=int, default=0,
                        help="Training-only GT chooses nearest SAT patches for backbone gradients")
    parser.add_argument("--hard-negative-margin-weight", type=float, default=0.0,
                        help="GT-loss-only ranking against the hardest distant gallery match")
    parser.add_argument("--hard-negative-min-distance-m", type=float, default=20.0)
    parser.add_argument("--hard-negative-logit-margin", type=float, default=2.0)
    parser.add_argument("--hard-negative-temperature", type=float, default=4.0)
    parser.add_argument("--uav-quarter-turn-augment", action="store_true",
                        help="Randomly rotate training UAV images about their center; position labels unchanged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2033)
    args = parser.parse_args()
    if (args.samples < 100 or args.epochs < 1 or args.batch_size < 1
            or args.backbone_lr <= 0 or args.head_lr <= 0
            or args.unfreeze_last_blocks < 1 or args.target_sigma_m <= 0
            or args.live_sat_topk < 0 or args.live_sat_random < 0
            or args.live_sat_gt_nearest < 0 or args.extra_route_oversample < 0
            or args.hard_negative_margin_weight < 0
            or args.hard_negative_min_distance_m <= 0
            or args.hard_negative_temperature <= 0):
        parser.error("invalid training parameters")
    exp.configure(args.city)
    source = torch.load(args.source_checkpoint, map_location="cpu")
    if (source.get("training_protocol") not in (
            "global_gallery_gt_loss_only", "global_gallery_gt_selected_train_patches")
            or source.get("gallery_city") != args.city
            or source.get("official96_train_only") is not True
            or source.get("previous_task_checkpoint_loaded") is not False):
        parser.error("source must be a same-city strict official96 global-gallery checkpoint")
    rows, excluded_hash = metadata.load_rows(
        args.city, args.samples, args.seed, official96_train_only=True)
    if source.get("excluded_test_manifest_sha256") != excluded_hash:
        parser.error("source checkpoint and this run exclude different test manifests")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AllMapGeoCLIP().to(device)
    vl._load_task_specific_state(model, source["model"])
    if source.get("backbone_fine_tuned"):
        model.clip.load_state_dict(source["backbone_state"], strict=True)
    resume = None
    if args.resume_interim:
        resume = torch.load(args.resume_interim, map_location="cpu")
        for key in ("model", "backbone_state", "gallery", "validation_mle_m", "epoch"):
            if key not in resume:
                raise RuntimeError("incomplete interim backbone checkpoint: " + key)
        vl._load_task_specific_state(model, resume["model"])
        model.clip.load_state_dict(resume["backbone_state"], strict=True)
    network = model.clip.network
    blocks = network.features
    if args.unfreeze_last_blocks > len(blocks):
        parser.error("unfreeze-last-blocks exceeds MobileNet block count")
    for parameter in model.clip.parameters():
        parameter.requires_grad_(False)
    for block in list(blocks)[-args.unfreeze_last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    backbone_parameters = [parameter for parameter in model.clip.parameters()
                           if parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters()
                       if not name.startswith("clip.") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.backbone_lr},
        {"params": head_parameters, "lr": args.head_lr},
    ], weight_decay=1e-4)
    origin_lat = float(source["origin_lat"])
    origin_lon = float(source["origin_lon"])
    cut = int(len(rows) * 0.9)
    extra_unique = []
    if args.extra_route_oversample:
        if args.extra_train_routes_dir is None:
            parser.error("--extra-route-oversample requires --extra-train-routes-dir")
        city_dir = args.extra_train_routes_dir / args.city
        audit_path = city_dir / "audit.json"
        if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("no_cross_split_samples"):
            raise RuntimeError("extra route directory lacks a train-only audit")
        blocked = {row[0] for row in rows[cut:]}
        seen = set()
        for manifest in sorted(city_dir.glob("train_*/manifest.csv")):
            with manifest.open(newline="") as route_file:
                for route_row in csv.DictReader(route_file):
                    image_path = str(Path(route_row["image_path"]).resolve())
                    if image_path in blocked or image_path in seen:
                        continue
                    if not Path(image_path).is_file():
                        raise FileNotFoundError(image_path)
                    seen.add(image_path)
                    extra_unique.append((image_path, float(route_row["x_m"]),
                                         float(route_row["y_m"])))
        if not extra_unique:
            raise RuntimeError("no audited extra route training images")
        extra_rows = extra_unique * args.extra_route_oversample
        rows = rows[:cut] + extra_rows + rows[cut:]
        cut += len(extra_rows)
        print("[BACKBONE EXTRA TRAIN]", json.dumps({
            "unique_route_images": len(extra_unique),
            "repetitions": args.extra_route_oversample,
            "extra_training_rows": len(extra_rows),
            "validation_rows": len(rows) - cut,
        }), flush=True)
    gt = torch.tensor([[x - origin_lon, y - origin_lat] for _, x, y in rows],
                      dtype=torch.float32, device=device)
    transform = image_transform(False, source="uav")
    # The sampled train-only validation subset depends on the seed. Compare
    # continuation against the source evaluated on this *same* subset.
    gallery = resume["gallery"] if resume else source["gallery"]
    if resume:
        source_validation = None
        best = float(resume["validation_mle_m"])
    else:
        source_uav = validation_features(model, rows[cut:], transform, device,
                                         args.batch_size)
        source_validation = global_train.evaluate(
            model, source_uav, gt[cut:],
            gallery["clip_feat"].to(device).float(),
            gallery["xy"].to(device), device, args.batch_size)
        best = float(source_validation["global_top1_MLE_m"])
        del source_uav
        print("[BACKBONE SOURCE SAME-SPLIT]", json.dumps(source_validation), flush=True)
    improved_in_this_run = False
    best_model = resume["model"] if resume else source["model"]
    best_backbone = (resume["backbone_state"] if resume else
                     {key: value.detach().cpu().clone()
                      for key, value in model.clip.state_dict().items()})
    best_gallery = gallery
    live_dataset = None
    if args.live_sat_topk or args.live_sat_gt_nearest:
        live_dataset = SatPatchGallery(origin_lat=origin_lat, origin_lon=origin_lon)
        pixels = torch.tensor([sample[:2] for sample in live_dataset.samples])
        if pixels.shape != gallery["pixel"].shape or not torch.allclose(
                pixels, gallery["pixel"].float()):
            raise RuntimeError("live satellite image ordering differs from cached gallery")
    history = ([{"resumed_from": str(args.resume_interim),
                 "previous_best_val_global_top1_MLE_m": best,
                 "previous_epoch": int(resume["epoch"])}] if resume else [])
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        model.clip.eval()  # Keep pretrained BatchNorm statistics fixed.
        sat_clip = gallery["clip_feat"].to(device).float()
        sat_xy = gallery["xy"].to(device)
        order = torch.randperm(cut, generator=torch.Generator().manual_seed(
            args.seed + (int(resume["epoch"]) if resume else 0) + epoch))
        losses = []
        for start in range(0, cut, args.batch_size):
            indices = order[start:start + args.batch_size]
            images = image_batch(rows, indices.tolist(), transform, device)
            if args.uav_quarter_turn_augment:
                turns = torch.randint(4, (len(images),)).tolist()
                images = torch.stack([torch.rot90(image, turn, (-2, -1))
                                      for image, turn in zip(images, turns)])
            uav_clip = model.clip.encode_image(images)
            z_uav = model.encode_uav_from_clip(uav_clip)
            # Position labels are read only in the training step. The held-out
            # route and validation inference never call this selection path.
            d2 = torch.cdist(gt[indices.to(device)], sat_xy).square()
            training_sat = sat_clip
            if live_dataset is not None:
                gt_nearest = (d2.topk(args.live_sat_gt_nearest, dim=1,
                                      largest=False).indices
                              if args.live_sat_gt_nearest else None)
                training_sat, live_indices, live_features = live_satellite_features(
                    model, z_uav, sat_clip, sat_xy, live_dataset,
                    args.live_sat_topk, args.live_sat_random, device,
                    gt_nearest_indices=gt_nearest)
            z_sat = model.encode_sat_from_clip(training_sat, sat_xy)
            logits = model.logit_scale.exp().clamp(max=100.0) * (z_uav @ z_sat.T)
            target = torch.softmax(-d2 / (2 * args.target_sigma_m ** 2), dim=1)
            loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            if args.distance_loss_weight:
                expected_distance = (F.softmax(logits, dim=1)
                                     * torch.sqrt(d2 + 1.0)).sum(dim=1).mean() / 100.0
                loss = loss + args.distance_loss_weight * expected_distance
            if args.hard_negative_margin_weight:
                # GT is used only to build the training loss. No GT-selected
                # satellite image is added to the model input or inference.
                positive_index = d2.argmin(dim=1, keepdim=True)
                positive_logit = logits.gather(1, positive_index).squeeze(1)
                distant = d2 >= args.hard_negative_min_distance_m ** 2
                negative_logit = logits.masked_fill(~distant, -torch.inf).max(dim=1).values
                rank_gap = (negative_logit - positive_logit
                            + args.hard_negative_logit_margin)
                rank_loss = F.softplus(rank_gap / args.hard_negative_temperature).mean()
                loss = loss + args.hard_negative_margin_weight * rank_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite backbone loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_parameters + head_parameters, 1.0)
            optimizer.step()
            if live_dataset is not None:
                with torch.no_grad():
                    sat_clip.index_copy_(0, live_indices, live_features.detach())
            losses.append(float(loss.detach().cpu()))
            if start == 0 or (start // args.batch_size) % 50 == 0:
                print("[BACKBONE GTLOSS] epoch=%d samples=%d/%d loss=%.4f" % (
                    epoch + 1, min(start + len(indices), cut), cut, losses[-1]), flush=True)
        # Refresh the *entire* satellite gallery with the same shared backbone.
        gallery = vl._build_satellite_backbone_gallery(
            model, origin_lat, origin_lon, device)
        sat_clip = gallery["clip_feat"].to(device).float()
        sat_xy = gallery["xy"].to(device)
        val_uav = validation_features(model, rows[cut:], transform, device,
                                      args.batch_size)
        metrics = global_train.evaluate(model, val_uav, gt[cut:], sat_clip,
                                        sat_xy, device, args.batch_size)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
               "validation": metrics}
        history.append(row)
        print("[BACKBONE GTLOSS]", json.dumps(row), flush=True)
        if metrics["global_top1_MLE_m"] < best:
            improved_in_this_run = True
            best = metrics["global_top1_MLE_m"]
            best_model = vl._task_specific_state_dict(model)
            best_backbone = {key: value.detach().cpu().clone()
                             for key, value in model.clip.state_dict().items()}
            best_gallery = {key: value.detach().cpu().clone()
                            for key, value in gallery.items()}
            interim = args.output / "best_interim.pt"
            torch.save({"model": best_model, "backbone_state": best_backbone,
                        "gallery": best_gallery, "validation_mle_m": best,
                            "epoch": (int(resume["epoch"]) if resume else 0) + epoch + 1}, interim)
    if best_model is None:
        raise RuntimeError("no backbone checkpoint selected")
    # Save the trained epoch even when global-gallery selection prefers the
    # source. The held-out route can select among separately trained candidates.
    candidate_model = vl._task_specific_state_dict(model)
    candidate_backbone = {key: value.detach().cpu().clone()
                          for key, value in model.clip.state_dict().items()}
    candidate_gallery = {key: value.detach().cpu().clone()
                         for key, value in gallery.items()}
    candidate = dict(source)
    candidate.update({
        "model": candidate_model, "best_model": candidate_model,
        "backbone_state": candidate_backbone,
        "backbone_state_sha256": state_sha256(candidate_backbone),
        "backbone_fine_tuned": True, "gallery": candidate_gallery,
        "training_protocol": "global_gallery_gt_loss_only",
        "GT_selected_training_patches": False,
        "previous_task_checkpoint_loaded": False,
        "official96_train_only": True,
        "excluded_test_manifest_sha256": excluded_hash,
        "extra_route_oversample": args.extra_route_oversample,
        "extra_route_unique_train_images": len(extra_unique),
        "best_score": float(metrics["global_top1_MLE_m"]),
    })
    torch.save(candidate, args.output / "candidate_epoch_last.pt")
    del candidate
    source.update({
        "model": best_model,
        "best_model": best_model,
        "backbone_state": best_backbone,
        "backbone_state_sha256": state_sha256(best_backbone),
        "backbone_fine_tuned": True,
        "backbone_unfreeze_last_blocks": args.unfreeze_last_blocks,
        "live_sat_topk": args.live_sat_topk,
        "live_sat_random": args.live_sat_random if args.live_sat_topk else 0,
        "live_sat_gt_nearest": (args.live_sat_gt_nearest if improved_in_this_run
                                else source.get("live_sat_gt_nearest", 0)),
        "GT_selected_training_patches": (bool(args.live_sat_gt_nearest)
                                         if improved_in_this_run else bool(source.get(
                                             "GT_selected_training_patches", False))),
        "uav_quarter_turn_augment": args.uav_quarter_turn_augment,
        "gallery": best_gallery,
        "training_protocol": (("global_gallery_gt_selected_train_patches"
                               if args.live_sat_gt_nearest else "global_gallery_gt_loss_only")
                              if improved_in_this_run else source["training_protocol"]),
        "previous_task_checkpoint_loaded": False,
        "backbone_source": exp.config.BACKBONE_NAME,
        "official96_train_only": True,
        "excluded_test_manifest_sha256": excluded_hash,
        "fine_tuned_samples_per_city": args.samples,
        "best_score": best,
        "visual_validation_routes": ["official_metadata_val_global_gallery"],
    })
    checkpoint = args.output / "checkpoints" / args.city / exp.VISUAL_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, checkpoint)
    report = {"city": args.city, "samples": args.samples,
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "architecture": exp.config.ARCHITECTURE_NAME,
              "backbone": exp.config.BACKBONE_NAME,
              "backbone_unfreeze_last_blocks": args.unfreeze_last_blocks,
              "live_sat_topk": args.live_sat_topk,
              "live_sat_random": args.live_sat_random if args.live_sat_topk else 0,
              "live_sat_gt_nearest": args.live_sat_gt_nearest,
              "uav_quarter_turn_augment": args.uav_quarter_turn_augment,
              "resumed_from_interim": str(args.resume_interim) if args.resume_interim else None,
              "best_checkpoint_improved_in_this_run": improved_in_this_run,
              "best_checkpoint_training_protocol": source["training_protocol"],
              "best_checkpoint_GT_selected_patches": bool(source.get(
                  "GT_selected_training_patches", False)),
              "training_GT_loss_only": not bool(args.live_sat_gt_nearest),
              "GT_selected_forward_inputs": bool(args.live_sat_gt_nearest),
              "best_val_global_top1_MLE_m": best,
              "source_same_split_validation": source_validation,
              "extra_route_unique_train_images": len(extra_unique),
              "candidate_epoch_last": str(args.output / "candidate_epoch_last.pt"),
              "history": history,
              "checkpoint": str(checkpoint)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[BACKBONE GTLOSS DONE]", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
