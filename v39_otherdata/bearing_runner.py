#!/usr/bin/env python3
"""Train the copied v36 architecture on Bearing-UAV routes and infer held-out routes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# This is the verbatim v36 config copied into v39_otherdata. Patch its dataset
# and protocol fields before importing visual_localizer/robust_tracker.
import config
from bearing_prepare import TEST_ROUTES, TRAIN_ROUTES, prepare as prepare_bearing


def _patch_config(args, prepared_root: Path) -> None:
    prepared_root = prepared_root.resolve()
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    output = prepared_root / "v39_output"
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    config.PROJECT_ROOT = HERE
    config.ARCHITECTURE_NAME = "V39_BearingUAV_" + str(config.ARCHITECTURE_NAME)
    config.OUTPUT_DIR = output
    config.CHECKPOINT_DIR = checkpoints
    config.VISUAL_CHECKPOINT = checkpoints / "visual_retrieval_Bearing_train_union.pt"
    config.TEMPORAL_CHECKPOINT = checkpoints / "temporal_Bearing_multiroute.pt"
    config.LATEST_TEMPORAL_CHECKPOINT = checkpoints / "temporal_Bearing_multiroute_latest.pt"

    # route_A is a union alias used only by inherited visual-head training.
    route_names = ["route_A", *TRAIN_ROUTES, *TEST_ROUTES]
    config.ROUTE_NAMES = list(route_names)
    config.ROUTE_ROOTS = [prepared_root / "routes" / name for name in route_names]
    config.WAYPOINT_FILES = {
        name: prepared_root / "routes" / name / "waypoints.json"
        for name in (*TRAIN_ROUTES, *TEST_ROUTES)
    }
    config.WAYPOINT_DIR = prepared_root / "routes"
    config.SAT_IMAGE = Path(exp["satellite_image"]).resolve()
    config.SAT_JSON = prepared_root / "bearing_satellite.json"
    config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Preserve v36 retrieval geometry. On Bearing-UAV 0.25 m/px, stride 32 px = 8 m.
    config.IMAGE_SIZE = 256
    config.UAV_CENTER_CROP_SIZE = 256
    config.UAV_RESIZE_AFTER_CROP = None
    config.TRAIN_UAV_AUGMENT = False
    config.SAT_CROP_SIZE = 320
    config.SAT_STRIDE = 32
    config.GRID_SIZE = 6
    config.CANDIDATE_COUNT = 36
    config.CANDIDATE_CAPTURE_RADIUS_M = 10.0
    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)

    # Explicitly use v36 route-reference mode: inference cannot read current-frame
    # GT coordinates or use a GT progress cap. GT remains training supervision.
    config.REFERENCE_PROTOCOL = "route_reference"
    config.FRAME_REFERENCE_SUPERVISION = True
    config.ROUTE_REFERENCE_ONLY = True
    config.SCHEDULED_ROUTE_REFERENCE = False
    config.NO_GT_INFERENCE = True
    config.CONTROLLED_FINAL_PROGRESS_CAP_TO_GT = False
    config.ACQ_HYPOTHESIS_COUNT = int(args.route_hypotheses)
    config.ACQ_RAW_VISUAL_EVIDENCE_WEIGHT = 2.0
    config.LOSS_ACQUISITION = 1.0
    config.LOSS_MEASUREMENT = 3.0
    config.LOSS_NEXT_STEP = 3.0
    config.LOSS_VELOCITY = 0.10

    # Pseudo-route frames are approximately 25 m apart, unlike the dense original
    # video. Raise only per-frame physical caps; the GRU/polynomial/Kalman design
    # itself is unchanged.
    step = float(args.step_m)
    config.MAX_FORWARD_SPEED_M_PER_FRAME = max(35.0, step * 1.6)
    config.MAX_CROSS_SPEED_M_PER_FRAME = max(15.0, step * 0.8)
    config.MAX_POLYNOMIAL_STEP_M_PER_FRAME = max(40.0, step * 1.8)
    config.ROUTE_STEP_SCALE_M = max(10.0, step)


def _reset_stage_patience() -> None:
    if not config.LATEST_TEMPORAL_CHECKPOINT.exists():
        return
    payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
    payload["patience"] = 0
    payload["best_score"] = float("inf")
    payload["best_model"] = payload.get("model")
    torch.save(payload, config.LATEST_TEMPORAL_CHECKPOINT)


def _promote_latest_to_final() -> None:
    payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
    torch.save(
        {
            "architecture": str(config.ARCHITECTURE_NAME),
            "model": payload["model"],
            "epoch": int(payload.get("epoch", 0)),
            "best_score": float(payload.get("best_score", float("nan"))),
            "v39_multiroute_curriculum": list(TRAIN_ROUTES),
            "inference_routes": list(TEST_ROUTES),
        },
        config.TEMPORAL_CHECKPOINT,
    )


def train_and_infer(args, prepared_root: Path) -> None:
    _patch_config(args, prepared_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import robust_tracker as tracker
    from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

    device = tracker.resolve_device()
    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume),
        )
    else:
        print("reuse visual checkpoint:", config.VISUAL_CHECKPOINT, flush=True)

    visual = FrozenVisualLocalizer(device)

    # Three-route curriculum: same model and optimizer continue across train_01/02/03.
    cumulative_epochs = 0
    for stage, route_name in enumerate(TRAIN_ROUTES):
        cumulative_epochs += int(args.epochs_per_route)
        if stage > 0:
            _reset_stage_patience()
        root = prepared_root / "routes" / route_name
        cache = tracker.build_route_cache(route_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
        )
        print("\n=== temporal stage %d/%d: %s ===" % (stage + 1, len(TRAIN_ROUTES), route_name), flush=True)
        tracker.train_temporal_model(
            visual=visual,
            cache=cache,
            route=route,
            device=device,
            epochs=cumulative_epochs,
            patience_limit=int(args.patience),
            resume=stage > 0 or bool(args.resume),
        )

    if not config.LATEST_TEMPORAL_CHECKPOINT.exists():
        raise RuntimeError("Temporal training produced no latest checkpoint")
    _promote_latest_to_final()
    model = tracker.load_temporal_model(device)

    summaries = {}
    for route_name in TEST_ROUTES:
        root = prepared_root / "routes" / route_name
        cache = tracker.build_route_cache(route_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
        )
        print("\n=== held-out inference: %s ===" % route_name, flush=True)
        summaries[route_name] = tracker.run_route_inference(
            route_name, visual, model, cache, route, device
        )
    summary_path = Path(config.OUTPUT_DIR) / "bearing_v39_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=float), encoding="utf-8")
    print("\n[DONE] summary:", summary_path, flush=True)
    print("[DONE] output :", config.OUTPUT_DIR, flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", default="cityb", choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--visual-epochs", type=int, default=30)
    p.add_argument("--epochs-per-route", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--jitter-m", type=float, default=12.0)
    p.add_argument("--route-hypotheses", type=int, default=13)
    p.add_argument("--step-m", type=float, default=25.0)
    p.add_argument("--max-sample-distance-m", type=float, default=20.0)
    p.add_argument("--heading-weight-px-per-deg", type=float, default=0.35)
    p.add_argument("--reuse-visual", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepared_root = HERE / "generated" / args.city
    if not (prepared_root / "experiment.json").exists():
        prep = argparse.Namespace(
            dataset_root=args.dataset_root,
            city=args.city,
            output_root=str(prepared_root),
            step_m=args.step_m,
            max_sample_distance_m=args.max_sample_distance_m,
            heading_weight_px_per_deg=args.heading_weight_px_per_deg,
        )
        prepare_bearing(prep)
    train_and_infer(args, prepared_root)


if __name__ == "__main__":
    main()
