"""2-train/1-validation orchestration for same-scene BearingUAV actual-pose routes."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker as rt
import robust_tracker_base as b
import visual_localizer as vl
from data import RouteDataset
from train_multirate_a import ARCHITECTURE_NAME, _slice_cache, _step_stats, _train_one_sequence
from visual_model import AllMapGeoCLIP, ThreeFrameRouteStateGRU


def validate_generated_protocol():
    """Fail early if stale cross-city, fixed-spacing or oversized routes are present."""
    summary_path = Path(config.GENERATED_ROOT) / "generation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing {summary_path}; run prepare_bearinguav_routes.py first"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    adapter = payload.get("adapter", {})
    split = payload.get("split", {})
    scene_policy = payload.get("scene_policy", {})

    if split.get("train") != ["train_1", "train_2"]:
        raise RuntimeError(f"wrong BearingUAV train split: {split.get('train')}")
    if split.get("validation") != ["val_1"] or split.get("test") not in ([], None):
        raise RuntimeError(f"wrong BearingUAV validation/test split: {split}")
    if not bool(scene_policy.get("same_satellite_scene_for_all_splits", False)):
        raise RuntimeError("BearingUAV routes are not marked as same-satellite-scene")
    if scene_policy.get("scene") != "citya":
        raise RuntimeError(f"expected one city-A scene, got {scene_policy.get('scene')}")
    if not bool(scene_policy.get("diagonal_segments_allowed", False)):
        raise RuntimeError("expected irregular routes with diagonal segments enabled")
    if adapter.get("position_labels") != "actual selected BearingUAV sample positions":
        raise RuntimeError("expected actual source-position labels")
    if not bool(adapter.get("variable_step", False)):
        raise RuntimeError("variable_step is not enabled")

    max_frames = int(adapter.get("max_route_frames", 600))
    min_frames = int(adapter.get("min_accepted_frames", 400))
    if max_frames > 600:
        raise RuntimeError(f"generated protocol allows {max_frames} frames; maximum must be 600")

    routes = payload.get("routes", [])
    if {r.get("route") for r in routes} != {"train_1", "train_2", "val_1"}:
        raise RuntimeError("generation summary does not contain exactly train_1/train_2/val_1")

    means = {}
    for route in routes:
        name = route.get("route", "unknown")
        if route.get("scene") != "citya":
            raise RuntimeError(f"{name}: expected citya, got {route.get('scene')}")
        frames = int(route.get("frames", 0))
        if frames > max_frames:
            raise RuntimeError(f"{name}: {frames} frames exceeds the {max_frames}-frame cap")
        if frames < min_frames:
            raise RuntimeError(f"{name}: only {frames} frames; minimum accepted is {min_frames}")
        if int(route.get("turn_waypoints", 0)) < 4:
            raise RuntimeError(f"{name}: too few segment-junction waypoints")
        label_mean = float(route.get("image_label_error_mean_m", float("inf")))
        label_p90 = float(route.get("image_label_error_p90_m", float("inf")))
        step_std = float(route.get("step_std_m", 0.0))
        step_mean = float(route.get("step_mean_m", 0.0))
        step_p10 = float(route.get("step_p10_m", 0.0))
        step_p90 = float(route.get("step_p90_m", 0.0))
        means[name] = step_mean
        if label_mean > 1e-6 or label_p90 > 1e-6:
            raise RuntimeError(
                f"{name}: image/position labels are misaligned "
                f"(mean={label_mean:.3f}m p90={label_p90:.3f}m)"
            )
        if step_std < 0.05 or abs(step_p90 - step_p10) < 0.10:
            raise RuntimeError(
                f"{name}: frame spacing is still effectively fixed "
                f"(mean={step_mean:.3f}m std={step_std:.3f}m "
                f"p10={step_p10:.3f}m p90={step_p90:.3f}m)"
            )
        print(
            f"{name} same-scene irregular-route check: scene=citya frames={frames} "
            f"turns={route['turn_waypoints']} step_mean={step_mean:.3f}m "
            f"std={step_std:.3f}m p10={step_p10:.3f}m p90={step_p90:.3f}m "
            f"image_label_error=0m",
            flush=True,
        )

    if not (means["train_1"] < means["val_1"] < means["train_2"]):
        print(
            "WARNING effective speed is not strictly train_1 < val_1 < train_2; "
            f"actual means={means}",
            flush=True,
        )


@torch.no_grad()
def build_visual_cache(model, gallery, root, route_index, origin_lat, origin_lon, device, jitter_m):
    dataset = RouteDataset(root, train=False, origin_lat=origin_lat, origin_lon=origin_lon)
    uav_rows, gt_rows, frame_rows = [], [], []
    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)
    model.eval()
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        uav_rows.append(model.encode_clip_image(uav).detach().cpu().to(torch.float16))
        gt_rows.append(torch.stack([item["xy"] for item in items]).float())
        frame_rows.extend(int(item["frame_id"]) for item in items)
        if start == 0 or end == len(dataset):
            print("%s visual backbone cache: %d/%d" % (Path(root).name, end, len(dataset)), flush=True)
    gt_xy = torch.cat(gt_rows)
    prior_xy = gt_xy + vl._deterministic_jitter(
        len(dataset), route_index=route_index, maximum_m=float(jitter_m)
    )
    candidate_indices = vl.regular_grid_indices(
        gallery["xy"], gallery["pixel"], vl.build_pixel_index(gallery["pixel"]),
        prior_xy, config.GRID_SIZE, config.SAT_STRIDE, torch.device("cpu"),
    )
    centers = gallery["xy"][candidate_indices]
    target_indices = vl._nearest_target(centers, gt_xy)
    capture = torch.linalg.norm(centers - gt_xy[:, None, :], dim=2).min(dim=1).values <= float(
        config.CANDIDATE_CAPTURE_RADIUS_M
    )
    cache = vl.VisualTrainCache(
        uav_clip=torch.cat(uav_rows), gt_xy=gt_xy,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        candidate_indices=candidate_indices, target_indices=target_indices, capture=capture,
    )
    print("%s visual candidate capture=%.2f%%" %
          (Path(root).name, 100.0 * capture.float().mean().item()), flush=True)
    return cache


def train_visual_multiroute(device, epochs, jitter_m):
    rt.set_seed(config.SEED)
    model = AllMapGeoCLIP().to(device)
    first = RouteDataset(config.ROUTE_ROOTS[0], train=False)
    origin_lat, origin_lon = float(first.origin_lat), float(first.origin_lon)
    gallery = vl._build_satellite_backbone_gallery(model, origin_lat, origin_lon, device)
    route_roots = dict(zip(config.ROUTE_NAMES, config.ROUTE_ROOTS))
    caches = {
        name: build_visual_cache(model, gallery, route_roots[name], index,
                                 origin_lat, origin_lon, device, jitter_m)
        for index, name in enumerate(config.TRAIN_ROUTE_NAMES + config.VALIDATION_ROUTE_NAMES)
    }
    train_rows = []
    for name in config.TRAIN_ROUTE_NAMES:
        cache = caches[name]
        train_rows.append((name, cache, [i for i in range(len(cache)) if bool(cache.capture[i])]))
    val_name = config.VALIDATION_ROUTE_NAMES[0]
    val_cache = caches[val_name]
    val_indices = list(range(len(val_cache)))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config.VISUAL_LR),
                                  weight_decay=float(config.VISUAL_WEIGHT_DECAY))
    best_score, best_state, patience = float("inf"), None, 0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.VISUAL_BATCH_SIZE)
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        model.train(); model.clip.eval(); losses = []
        order = list(train_rows); random.shuffle(order)
        for _name, cache, indices in order:
            shuffled = list(indices); random.shuffle(shuffled)
            for offset in range(0, len(shuffled), batch_size):
                batch = torch.tensor(shuffled[offset:offset + batch_size])
                optimizer.zero_grad(set_to_none=True)
                loss, _, _, _ = vl._visual_forward_batch(model, cache, gallery, batch, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite visual loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, float(config.GRAD_CLIP_NORM))
                optimizer.step(); losses.append(float(loss.detach().cpu()))
        validation = vl._evaluate_visual_model(model, val_cache, gallery, val_indices, device)
        score = float(validation["SoftMS_MLE_m"])
        if score < best_score:
            best_score = score; best_state = vl._task_specific_state_dict(model); patience = 0
        else:
            patience += 1
        torch.save({
            "model": best_state if best_state is not None else vl._task_specific_state_dict(model),
            "best_model": best_state, "model_format": "task_specific_only",
            "origin_lat": origin_lat, "origin_lon": origin_lon, "gallery": gallery,
            "epoch": epoch, "best_score": best_score,
            "visual_train_routes": list(config.TRAIN_ROUTE_NAMES),
            "visual_validation_routes": list(config.VALIDATION_ROUTE_NAMES),
            "visual_eval_routes": [],
            "previous_task_checkpoint_loaded": False,
            "backbone_source": config.BACKBONE_NAME,
            "task_specific_initialization": "random", "jitter_m": float(jitter_m),
            "bearing_data_protocol": str(config.BEARING_DATA_PROTOCOL),
        }, config.VISUAL_CHECKPOINT)
        elapsed = time.perf_counter() - started
        print("visual epoch=%03d/%d loss=%.5f val_softms_mle=%.3fm val_p90=%.3fm "
              "val_lsr10=%.2f%% time=%.1fs" %
              (epoch, epochs, np.mean(losses), validation["SoftMS_MLE_m"],
               validation["SoftMS_P90_m"], validation["LSR@10_pct"], elapsed), flush=True)
        if patience >= int(config.VISUAL_EARLY_STOPPING_PATIENCE):
            break
    if best_state is None:
        raise RuntimeError("visual training produced no checkpoint")


def load_multiroute_visual(device):
    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    if checkpoint.get("visual_train_routes") != list(config.TRAIN_ROUTE_NAMES):
        raise RuntimeError("visual checkpoint train split mismatch")
    if checkpoint.get("bearing_data_protocol") != str(config.BEARING_DATA_PROTOCOL):
        raise RuntimeError("visual checkpoint BearingUAV data protocol mismatch")
    original_validator = vl._validate_visual_provenance
    vl._validate_visual_provenance = lambda _checkpoint: None
    try:
        return vl.FrozenVisualLocalizer(device)
    finally:
        vl._validate_visual_provenance = original_validator


def build_route_objects(visual, device):
    roots = dict(zip(config.ROUTE_NAMES, config.ROUTE_ROOTS))
    result = {}
    for name in config.ROUTE_NAMES:
        cache = rt.build_route_cache(name, roots[name], visual, device)
        route = rt.WaypointRoute(rt.load_waypoint_xy(name, visual.origin_lat, visual.origin_lon))
        result[name] = (cache, route, b.build_gt_route_state(cache, route))
    return result


def train_temporal_multiroute(visual, objects, device, epochs, patience_limit, resume=False):
    stride = int(config.TEMPORAL_EXTRA_A_STRIDE)
    prepared = {}
    for name in config.TRAIN_ROUTE_NAMES:
        cache, route, gt_state = objects[name]
        stride_indices = np.arange(0, len(cache), stride, dtype=np.int64)
        stride_cache = _slice_cache(cache, stride_indices, name + "_stride%d" % stride)
        prepared[name] = (cache, route, gt_state, stride_cache,
                          b.build_gt_route_state(stride_cache, route))
        print("%s native=%d mean_step=%.3fm stride%d=%d mean_step=%.3fm" %
              (name, len(cache), _step_stats(cache)["mean"], stride,
               len(stride_cache), _step_stats(stride_cache)["mean"]), flush=True)
    val_name = config.VALIDATION_ROUTE_NAMES[0]
    val_cache, val_route, val_gt = objects[val_name]
    model = ThreeFrameRouteStateGRU().to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config.TEMPORAL_LR),
                                  weight_decay=float(config.TEMPORAL_WEIGHT_DECAY))
    start_epoch, best_score, best_state, patience = 1, float("inf"), None, 0
    if resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        if payload.get("bearing_data_protocol") != str(config.BEARING_DATA_PROTOCOL):
            raise RuntimeError("temporal checkpoint BearingUAV data protocol mismatch")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1; best_score = float(payload["best_score"])
        best_state = payload.get("best_model"); patience = int(payload.get("patience", 0))
    for epoch in range(start_epoch, int(epochs) + 1):
        started = time.perf_counter(); model.train(); native_losses, stride_losses = [], []
        name = config.TRAIN_ROUTE_NAMES[(epoch - 1) % len(config.TRAIN_ROUTE_NAMES)]
        cache, route, gt, stride_cache, stride_gt = prepared[name]
        native_losses.extend(_train_one_sequence(
            model, optimizer, params, visual, cache, route, gt,
            range(len(cache)), device))
        stride_losses.extend(_train_one_sequence(
            model, optimizer, params, visual, stride_cache, route, stride_gt,
            range(len(stride_cache)), device))
        val = rt.evaluate_closed_loop(model, visual, val_cache, val_route, val_gt,
                                      (0, len(val_cache)), device)
        score = float(val["score"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)
        if improved:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
            torch.save({"architecture": ARCHITECTURE_NAME, "model": best_state,
                        "epoch": epoch, "validation": val,
                        "train_routes": list(config.TRAIN_ROUTE_NAMES), "epoch_route": name,
                        "validation_routes": list(config.VALIDATION_ROUTE_NAMES),
                        "test_routes": [], "train_forward_rows": 3,
                        "bearing_data_protocol": str(config.BEARING_DATA_PROTOCOL)},
                       config.TEMPORAL_CHECKPOINT)
        else:
            patience += 1
        torch.save({"architecture": ARCHITECTURE_NAME, "model": model.state_dict(),
                    "best_model": best_state, "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_score": best_score, "patience": patience,
                    "train_routes": list(config.TRAIN_ROUTE_NAMES), "epoch_route": name,
                    "validation_routes": list(config.VALIDATION_ROUTE_NAMES),
                    "test_routes": [], "bearing_data_protocol": str(config.BEARING_DATA_PROTOCOL)},
                   config.LATEST_TEMPORAL_CHECKPOINT)
        elapsed = time.perf_counter() - started
        print("multiroute epoch=%03d/%d route=%s native_loss=%.5f stride%d_loss=%.5f "
              "val_mle=%.3fm val_p90=%.3fm score=%.3f best=%.3f time=%.1fs" %
              (epoch, epochs, name, np.mean(native_losses), stride, np.mean(stride_losses),
               val["mle"], val["p90"], score, best_score, elapsed), flush=True)
        if elapsed > 300.0:
            print("WARNING epoch exceeded requested 300s ceiling", flush=True)
        if epoch >= int(config.EARLY_STOP_MIN_EPOCH) and patience >= int(patience_limit):
            break
    if best_state is None:
        raise RuntimeError("temporal training produced no best checkpoint")
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS))
    parser.add_argument("--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS))
    parser.add_argument("--patience", type=int, default=int(config.EARLY_STOP_PATIENCE))
    parser.add_argument("--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    validate_generated_protocol()
    rt._set_forward_rows(3)
    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(args.jitter_m)
    rt.set_seed(config.SEED); device = rt.resolve_device()
    if not config.VISUAL_CHECKPOINT.exists():
        train_visual_multiroute(device, args.visual_epochs, args.jitter_m)
    visual = load_multiroute_visual(device)
    objects = build_route_objects(visual, device)
    model = train_temporal_multiroute(visual, objects, device, args.temporal_epochs,
                                      args.patience, args.resume)

    val_name = config.VALIDATION_ROUTE_NAMES[0]
    val_cache, val_route, val_gt = objects[val_name]
    validation = rt.evaluate_closed_loop(
        model, visual, val_cache, val_route, val_gt, (0, len(val_cache)), device
    )
    results = {
        "protocol": "same city-A satellite image; irregular train_1+train_2 train, val_1 validation",
        "validation_route": val_name,
        "validation": validation,
    }
    output = config.OUTPUT_DIR / "validation_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print("validation=" + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
