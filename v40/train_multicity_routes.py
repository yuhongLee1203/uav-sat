"""Shared temporal model trained over Bearing-UAV city route episodes.

The model is updated once across selected city episodes; GRU and Kalman state
reset at each route boundary. Inference never receives current-frame GT.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import torch

import image_only_citya as exp

CITIES = ("citya", "cityb", "cityc", "cityd")


def check_city(city):
    exp.configure(city)
    exp.verify_visual_checkpoint_split()
    manifests = []
    for name in ("train_01", "test_01", "test_02"):
        path = exp.config.ROUTE_ROOTS[("train_01", "test_01", "test_02").index(name)] / "manifest.csv"
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or any(not row["image_path"].startswith(
                "/yh/study/cvpr_data/Bearing_UAV_90K/" + city + "/") for row in rows):
            raise RuntimeError("wrong or empty Bearing-UAV 90K city route: " + str(path))
        manifests.append({row["image_path"] for row in rows})
    if any(manifests[i] & manifests[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("train/test image overlap in " + city)
    return [len(s) for s in manifests]


def city_context(city, device, cache_root):
    exp.configure(city)
    visual = exp.FrozenVisualLocalizer(device)
    cache = exp.prepare_cache(visual, device, cache_root / city)
    route = exp.tr.WaypointRoute(exp.tr.load_waypoint_xy(
        "route_A", visual.origin_lat, visual.origin_lon))
    return visual, cache, route


def training_targets(visual, route, device):
    cache = exp.tr.build_route_cache(
        "route_A", exp.config.ROUTE_ROOTS[0], visual, device)
    # next_step_se at frame t is applied to predict frame t+1. Match the
    # established v39 training target; the final frame reuses its last step.
    step_xy = torch.zeros_like(cache.gt_xy)
    step_xy[:-1] = cache.gt_xy[1:] - cache.gt_xy[:-1]
    if len(step_xy) > 1:
        step_xy[-1] = step_xy[-2]
    return {"xy": cache.gt_xy, "step_xy": step_xy}


def evaluate_city(city, model, device, output):
    exp.configure(city)
    if (exp.config.GT_FRONT18_ORACLE or exp.config.GT_FINAL_MS_ORACLE_CENTER
            or exp.config.GT_FRONT18_PATCH_ORACLE
            or exp.config.GT_FINAL_MS_PATCH_ORACLE):
        raise RuntimeError("held-out test forbids current-frame GT oracle switches")
    if not exp.config.NO_GT_INFERENCE or not exp.config.ROUTE_REFERENCE_ONLY or getattr(
            exp.config, "SCHEDULED_ROUTE_REFERENCE", False):
        raise RuntimeError("held-out evaluation must use causal route reference without frame GT")
    exp.config.OUTPUT_DIR = output / city
    exp.config.MEASURE_END_TO_END_LATENCY = True
    visual = exp.FrozenVisualLocalizer(device)
    rows = {}
    model.eval()
    for name, label in (("route_B", "test_01"), ("route_C", "test_02")):
        route_index = exp.config.ROUTE_NAMES.index(name)
        route = exp.tr.WaypointRoute(exp.tr.load_waypoint_xy(
            name, visual.origin_lat, visual.origin_lon))
        cache = exp.tr.build_route_cache(
            name, exp.config.ROUTE_ROOTS[route_index], visual, device)
        summary = exp.tr.run_route_inference(
            name, visual, model, cache, route, device)
        rows[label] = summary
    return rows


def evaluate_validation(city, model, device, output):
    """Independent official96 validation episode; no frame GT enters inference."""
    exp.configure(city)
    if not exp.config.NO_GT_INFERENCE or not exp.config.ROUTE_REFERENCE_ONLY:
        raise RuntimeError("validation must be causal and GT-free")
    val_root = exp.PREPARED / "routes" / "val_01"
    if not (val_root / "manifest.csv").is_file():
        raise FileNotFoundError("missing independent validation episode: " + str(val_root))
    exp.config.ROUTE_ROOTS.append(val_root)
    exp.config.ROUTE_NAMES.append("route_V")
    exp.config.WAYPOINT_FILES["route_V"] = val_root / "waypoints.json"
    exp.config.OUTPUT_DIR = output / city
    # Exercise the deployed path: current image runs through the backbone,
    # while the recurrent history reuses the previous two embeddings.
    exp.config.MEASURE_END_TO_END_LATENCY = True
    visual = exp.FrozenVisualLocalizer(device)
    route = exp.tr.WaypointRoute(exp.tr.load_waypoint_xy(
        "route_V", visual.origin_lat, visual.origin_lon))
    cache = exp.tr.build_route_cache("route_V", val_root, visual, device)
    model.eval()
    return exp.tr.run_route_inference("route_V", visual, model, cache, route, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cities", nargs="+", choices=CITIES, default=["citya"])
    parser.add_argument("--eval-cities", nargs="+", choices=CITIES, default=None)
    parser.add_argument("--supervision", choices=("image_only", "gt_train"), required=True)
    parser.add_argument("--init", choices=("shared", "random_temporal"), default="shared")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-frames", type=int, default=136)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--teacher-acquisition-ratio", type=float, default=0.0)
    parser.add_argument("--teacher-center-ratio", type=float, default=0.0)
    parser.add_argument("--teacher-state-ratio", type=float, default=0.0)
    parser.add_argument("--acquisition-loss-weight", type=float, default=1.0)
    parser.add_argument("--measurement-loss-weight", type=float, default=2.0)
    parser.add_argument("--motion-loss-weight", type=float, default=2.0)
    parser.add_argument("--dense-offbank-loss", action="store_true",
                        help="Keep GT loss active even when the causal search bank misses")
    parser.add_argument("--acquisition-target-sigma-m", type=float, default=0.0,
                        help="Positive value enables distance-based soft labels for overlapping windows")
    parser.add_argument("--extra-train-routes-dir", type=Path,
                        help="Train-only episode roots under DIR/<city>/train_*")
    parser.add_argument("--validate-every-epochs", type=int, default=0,
                        help="Select temporal checkpoint on separate val_01 episodes")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--skip-test", action="store_true",
                        help="Develop on validation only; evaluate held-out test once after selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--init-temporal-checkpoint", type=Path,
                        help="Continue a verified GT-loss-only temporal checkpoint")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_train_frames < 3 or args.lr <= 0:
        parser.error("epochs >= 1, max-train-frames >= 3 and lr > 0 required")
    if args.validate_every_epochs < 0 or args.early_stop_patience < 1:
        parser.error("invalid validation interval or patience")
    if (not 0 <= args.teacher_acquisition_ratio <= 1
            or not 0 <= args.teacher_center_ratio <= 1
            or not 0 <= args.teacher_state_ratio <= 1):
        parser.error("teacher ratios must be in [0,1]")
    if args.teacher_acquisition_ratio or args.teacher_center_ratio or args.teacher_state_ratio:
        parser.error("GT is permitted only in supervision losses; teacher inputs must all be zero")
    if min(args.acquisition_loss_weight, args.measurement_loss_weight,
           args.motion_loss_weight, args.acquisition_target_sigma_m) < 0:
        parser.error("loss weights must be nonnegative")
    if args.supervision == "gt_train" and args.init != "random_temporal":
        parser.error("strict GT-loss-only training requires --init random_temporal; shared is GT-prior trained")
    if args.init_temporal_checkpoint and args.supervision != "gt_train":
        parser.error("temporal warm start is only supported for gt_train")
    if args.supervision == "gt_train" and not os.environ.get("V40_VISUAL_CHECKPOINT_ROOT"):
        parser.error("strict GT-loss-only training requires fresh visual checkpoints via V40_VISUAL_CHECKPOINT_ROOT")
    if len(set(args.train_cities)) != len(args.train_cities):
        parser.error("duplicate train city")
    eval_cities = args.eval_cities or args.train_cities
    if len(set(eval_cities)) != len(eval_cities):
        parser.error("duplicate eval city")
    counts = {city: check_city(city) for city in set(args.train_cities + eval_cities)}
    if args.supervision == "gt_train":
        for city in set(args.train_cities + eval_cities):
            exp.configure(city)
            visual_source = torch.load(exp.config.VISUAL_CHECKPOINT, map_location="cpu")
            if (visual_source.get("training_protocol") != "global_gallery_gt_loss_only"
                    or visual_source.get("previous_task_checkpoint_loaded") is not False
                    or visual_source.get("backbone_source") != "torchvision:mobilenet_v3_small"):
                parser.error("strict GT-loss-only training requires a fresh MobileNet global-gallery visual checkpoint for " + city)
            del visual_source
    args.output.mkdir(parents=True, exist_ok=True)
    run_settings = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("V40_", "UAVSAT_", "MS_"))
                        or key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                   "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    (args.output / "run_settings.json").write_text(json.dumps(run_settings, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp.configure(args.train_cities[0])
    if args.init == "shared":
        model = exp.new_model(device)
    else:
        model = exp.ThreeFrameRouteStateGRU().to(device)
    if args.init_temporal_checkpoint:
        source_path = args.init_temporal_checkpoint.resolve()
        source_settings_path = source_path.parent / "run_settings.json"
        if not source_settings_path.is_file():
            parser.error("warm-start checkpoint lacks run_settings.json provenance")
        source_settings = json.loads(source_settings_path.read_text())
        source_args = source_settings.get("arguments", {})
        source_env = source_settings.get("environment", {})
        if (source_args.get("supervision") != "gt_train"
                or any(float(source_args.get(name, 1.0)) != 0.0 for name in (
                    "teacher_acquisition_ratio", "teacher_center_ratio",
                    "teacher_state_ratio"))
                or any(source_env.get(name, "0") != "0" for name in (
                    "V40_GT_FRONT18_ORACLE", "V40_GT_FRONT18_PATCH_ORACLE",
                    "V40_GT_FINAL_MS_ORACLE_CENTER",
                    "V40_GT_FINAL_MS_PATCH_ORACLE"))):
            parser.error("warm-start checkpoint is not verified GT-loss-only")
        loaded = torch.load(source_path, map_location="cpu")["model"]
        current = model.state_dict()
        if set(loaded) != set(current):
            parser.error("warm-start model keys differ")
        expanded = []
        for name, value in loaded.items():
            if value.shape == current[name].shape:
                current[name] = value
            elif (name == "acq_numeric_projection.0.weight"
                  and value.ndim == 2 and value.shape[0] == current[name].shape[0]
                  and value.shape[1] + 1 == current[name].shape[1]):
                padded = torch.zeros_like(current[name])
                padded[:, :value.shape[1]] = value
                current[name] = padded
                expanded.append(name)
            else:
                parser.error("warm-start tensor shape differs: " + name)
        model.load_state_dict(current, strict=True)
        print("[V40 GTLOSS WARM START]", json.dumps({
            "checkpoint": str(source_path), "expanded": expanded
        }), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_val_mle = float("inf")
    best_state = None
    best_epoch = None
    bad_validations = 0
    for epoch in range(args.epochs):
        order = list(args.train_cities)
        random.Random(args.seed + epoch).shuffle(order)
        for city in order:
            exp.configure(city)
            visual = exp.FrozenVisualLocalizer(device)
            route_roots = [exp.config.ROUTE_ROOTS[0]]
            if args.extra_train_routes_dir:
                extra_city = args.extra_train_routes_dir / city
                audit_path = extra_city / "audit.json"
                if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("no_cross_split_samples"):
                    raise RuntimeError("missing audited train-only extra episodes: " + str(extra_city))
                route_roots.extend(sorted(p for p in extra_city.glob("train_*")
                                          if p.is_dir()))
            random.Random(args.seed + epoch + len(city)).shuffle(route_roots)
            city_feature_cache_dir = exp.config.FEATURE_CACHE_DIR
            for route_root in route_roots:
                exp.config.ROUTE_ROOTS[0] = route_root
                exp.config.WAYPOINT_FILES["route_A"] = route_root / "waypoints.json"
                # Each independent episode needs its own persistent backbone
                # cache; otherwise route_A in the next episode overwrites it.
                exp.config.FEATURE_CACHE_DIR = city_feature_cache_dir / route_root.name
                cache = exp.prepare_cache(visual, device,
                    args.output / "feature_cache" / city / route_root.name)
                route = exp.tr.WaypointRoute(exp.tr.load_waypoint_xy(
                    "route_A", visual.origin_lat, visual.origin_lon))
                targets = (training_targets(visual, route, device)
                           if args.supervision == "gt_train" else None)
                print("[V40 ROUTE START]", json.dumps({
                    "epoch": epoch + 1, "epochs": args.epochs, "city": city,
                    "route": route_root.name,
                    "frames": min(args.max_train_frames, len(cache["image_paths"])),
                }), flush=True)
                loss = exp.train_epoch(
                    "visual_pseudo", model, visual, cache, route, optimizer,
                    device, args.max_train_frames, targets=targets,
                    loss_weights=(args.acquisition_loss_weight,
                                  args.measurement_loss_weight,
                                  args.motion_loss_weight),
                    dense_offbank=args.dense_offbank_loss,
                    acquisition_target_sigma_m=args.acquisition_target_sigma_m)
                row = {"epoch": epoch + 1, "city": city, "route": route_root.name,
                       "loss": loss, "training_GT_used": targets is not None}
                history.append(row)
                print("[V40 EPISODE]", json.dumps(row), flush=True)
                del cache, route, targets
            del visual
        if args.validate_every_epochs and ((epoch + 1) % args.validate_every_epochs == 0):
            validation = {}
            for city in eval_cities:
                summary = evaluate_validation(city, model, device,
                    args.output / "validation" / ("epoch_%03d" % (epoch + 1)))
                validation[city] = float(summary["MLE_m"])
            score = sum(validation.values()) / len(validation)
            history.append({"epoch": epoch + 1, "validation_mle_by_city": validation,
                            "validation_mean_mle_m": score})
            print("[V40 VALIDATION]", json.dumps(history[-1]), flush=True)
            if score < best_val_mle:
                best_val_mle = score
                best_epoch = epoch + 1
                best_state = {name: weight.detach().cpu().clone()
                              for name, weight in model.state_dict().items()}
                interim = args.output / "best_temporal_interim.pt"
                temporary = args.output / "best_temporal_interim.pt.tmp"
                torch.save({"model": best_state,
                            "architecture": exp.config.ARCHITECTURE_NAME,
                            "runtime_correction_bounds_m": exp.correction_bounds(),
                            "loss_coordinate_system": "raw_xy",
                            "validation_mean_mle_m": score,
                            "epoch": best_epoch}, temporary)
                os.replace(temporary, interim)
                bad_validations = 0
            else:
                bad_validations += 1
                if bad_validations >= args.early_stop_patience:
                    print("[V40 EARLY STOP] no validation MLE improvement", flush=True)
                    break
    if args.validate_every_epochs:
        if best_state is None:
            raise RuntimeError("validation did not select a temporal checkpoint")
        model.load_state_dict(best_state, strict=True)
    checkpoint = args.output / "shared_temporal.pt"
    torch.save({"model": model.state_dict(), "architecture": exp.config.ARCHITECTURE_NAME,
                "runtime_correction_bounds_m": exp.correction_bounds(),
                "loss_coordinate_system": "raw_xy",
                "supervision": args.supervision, "init": args.init,
                "train_cities": args.train_cities, "history": history}, checkpoint)
    exp.configure(args.train_cities[0])
    prepared_experiment = json.loads((exp.PREPARED / "experiment.json").read_text())
    heldout_gt_waypoints = bool(prepared_experiment.get(
        "heldout_waypoints_from_GT",
        prepared_experiment.get("sequence_selection_version")
        == "official_rsi96_seed42_temporal_v1"))
    extra_gt_waypoints = bool(args.extra_train_routes_dir) and not all(
        json.loads((args.extra_train_routes_dir / city / "audit.json").read_text())
        .get("predeclared_train_waypoints", False)
        for city in args.train_cities)
    report = {
        "architecture": exp.config.ARCHITECTURE_NAME,
        "runtime_correction_bounds_m": exp.correction_bounds(),
        "loss_coordinate_system": "raw_xy",
        "validation_current_frame_backbone": True,
        "source_dataset": "/yh/study/cvpr_data/Bearing_UAV_90K",
        "protocol": "custom pseudo-route episodes within the official96 metadata split; not official Table 6",
        "training_supervision": args.supervision,
        "initialization": args.init,
        "training_teacher_acquisition_ratio": args.teacher_acquisition_ratio,
        "training_teacher_center_ratio": args.teacher_center_ratio,
        "training_teacher_state_ratio": args.teacher_state_ratio,
        "training_GT_loss_only": args.supervision == "gt_train",
        "training_GT_model_inputs": extra_gt_waypoints,
        "extra_train_routes_dir": str(args.extra_train_routes_dir) if args.extra_train_routes_dir else None,
        "best_validation_mean_mle_m": best_val_mle if args.validate_every_epochs else None,
        "best_validation_epoch": best_epoch,
        "dense_offbank_loss": args.dense_offbank_loss,
        "acquisition_target_sigma_m": args.acquisition_target_sigma_m,
        "training_loss_weights_acquisition_measurement_motion": [
            args.acquisition_loss_weight, args.measurement_loss_weight,
            args.motion_loss_weight],
        "visual_weights": ("external visual checkpoints: verify training_protocol=global_gallery_gt_loss_only"
                           if os.environ.get("V40_VISUAL_CHECKPOINT_ROOT")
                           else "previously GT-supervised shared ABCD checkpoint"),
        "visual_checkpoint_root": os.environ.get("V40_VISUAL_CHECKPOINT_ROOT"),
        "train_cities": args.train_cities,
        "eval_cities": eval_cities,
        "image_counts_train_test01_test02": counts,
        "test_GT_used_by_model": False,
        "test_GT_derived_waypoints": heldout_gt_waypoints,
        "episode_construction_uses_GT_offline": prepared_experiment.get(
            "episode_construction_uses_GT_offline", True),
        "route_duration_prior": False,
        "checkpoint": str(checkpoint),
        "test_evaluated": not args.skip_test,
        "history": history,
        "results": {},
    }
    if not args.skip_test:
        for city in eval_cities:
            report["results"][city] = evaluate_city(city, model, device, args.output / "eval")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    route_caveat = (
        "Held-out waypoints were derived from GT; this is not a GT-free test."
        if heldout_gt_waypoints else
        "Held-out waypoints are predeclared independently of selected GT; "
        "GT was used offline to assemble synthetic episodes."
    )
    lines = ["# Bearing-UAV prepared-route model evaluation", "",
             "Training supervision: `%s`; temporal initialization: `%s`." % (args.supervision, args.init),
             "Inference does not read the current-frame GT column. " + route_caveat,
             "This is a custom route subset of Bearing-UAV 90K, not official Table 6.", "",
             "| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |",
             "|---|---|---:|---:|---:|---:|---:|"]
    if args.skip_test:
        lines.append("Held-out test was not evaluated in this training run.")
    else:
        for city in eval_cities:
            for name, idx in (("test_01", 1), ("test_02", 2)):
                s = report["results"][city][name]
                lines.append("| %s | %s | %d | %.3f | %.3f | %.2f%% | %.2f%% |" % (
                    city, name, counts[city][idx], s["MLE_m"], s["P90_m"],
                    s["LSR@15_pct"], s["SelectedCandidateCapture_pct"]))
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("[V40 DONE]", args.output / "report.md", flush=True)


if __name__ == "__main__":
    main()
