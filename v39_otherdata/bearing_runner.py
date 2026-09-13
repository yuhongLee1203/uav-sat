#!/usr/bin/env python3
"""Run the canonical v39 DirectFinalMS model on prepared Bearing-UAV routes.

Bearing-specific code is restricted to dataset preparation / data.py.  The model
runtime is rebuilt from v39_DirectFinalMS/base_src on every run and receives the
same DirectFinalMS + Context-GRU patches as the canonical v39 run.sh.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CANONICAL_ROOT = REPO_ROOT / "v39_DirectFinalMS"
CANONICAL_BASE = CANONICAL_ROOT / "base_src"
CANONICAL_FINALMS_PATCH = CANONICAL_ROOT / "patch_direct_finalms.py"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bearing_prepare import TEST_ROUTES, TRAIN_ROUTES, prepare as prepare_bearing

FINAL_ARCH = "V39_Forward3x6_ContextGRU_FixedKalman_FinalMS5x5_BearingUAV"
RUNTIME_FILES = ("config.py", "robust_tracker.py", "visual_localizer.py", "visual_model.py")


def _prepared_matches(prepared_root: Path, args) -> bool:
    exp_path = prepared_root / "experiment.json"
    if not exp_path.exists():
        return False
    try:
        exp = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if Path(exp.get("dataset_root", "")).resolve() != Path(args.dataset_root).resolve():
        return False
    if exp.get("city") != args.city:
        return False
    stats = exp.get("route_stats", {})
    names = list(TRAIN_ROUTES) + list(TEST_ROUTES)
    if any(name not in stats for name in names):
        return False
    return all(
        abs(float(stats[name].get("sample_step_m", -1.0)) - float(args.step_m)) < 1e-6
        for name in names
    )


def _ensure_prepared(args, prepared_root: Path) -> None:
    if _prepared_matches(prepared_root, args) and not args.reprepare:
        return
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prep = argparse.Namespace(
        dataset_root=args.dataset_root,
        city=args.city,
        output_root=str(prepared_root),
        step_m=args.step_m,
        max_sample_distance_m=args.max_sample_distance_m,
        heading_weight_px_per_deg=args.heading_weight_px_per_deg,
    )
    print(
        "[PREP] rebuilding Bearing routes: city=%s step=%.2fm" % (args.city, args.step_m),
        flush=True,
    )
    prepare_bearing(prep)


def _patch_context_gru(runtime: Path) -> None:
    """Apply the exact Context-GRU + velocity-fusion patch from canonical run.sh."""
    p = runtime / "visual_model.py"
    s = p.read_text(encoding="utf-8")
    old = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
    new = "        self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)\n"
    if s.count(old) != 1:
        raise RuntimeError("canonical Context-GRU declaration patch did not match exactly once")
    s = s.replace(old, new, 1)
    old_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
    new_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
    if s.count(old_block) != 1:
        raise RuntimeError("canonical Context-GRU recurrent-input patch did not match exactly once")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, encoding="utf-8")
    compile(s, str(p), "exec")

    p = runtime / "robust_tracker.py"
    s = p.read_text(encoding="utf-8")
    old_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            step = velocity.copy()
'''
    new_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            if bool(getattr(config, "EXPERIMENT_DISABLE_GRU", False)):
                velocity = self.x[2:4].copy()
                step = velocity.copy()
            else:
                step = velocity.copy()
'''
    if s.count(old_motion) != 1:
        raise RuntimeError("canonical velocity-fusion patch did not match exactly once")
    s = s.replace(old_motion, new_motion, 1)
    p.write_text(s, encoding="utf-8")
    compile(s, str(p), "exec")


def _make_runtime(prepared_root: Path) -> Path:
    for filename in RUNTIME_FILES:
        source = CANONICAL_BASE / filename
        if not source.exists():
            raise FileNotFoundError("missing canonical v39 source: %s" % source)
    if not CANONICAL_FINALMS_PATCH.exists():
        raise FileNotFoundError("missing canonical DirectFinalMS patch: %s" % CANONICAL_FINALMS_PATCH)

    runtime = prepared_root / "runtime_v39_directfinalms"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)

    for filename in RUNTIME_FILES:
        shutil.copy2(CANONICAL_BASE / filename, runtime / filename)
    # Only the data adapter is Bearing-specific.  Architecture/training files are canonical v39.
    shutil.copy2(HERE / "data.py", runtime / "data.py")

    subprocess.run(
        [sys.executable, str(CANONICAL_FINALMS_PATCH), str(runtime / "robust_tracker.py")],
        check=True,
    )
    _patch_context_gru(runtime)
    print("[RUNTIME] canonical v39 DirectFinalMS source + patches: PASS", flush=True)
    return runtime


def _set_canonical_environment(args, prepared_root: Path) -> tuple[Path, Path, Path]:
    output = prepared_root / "v39_output_corrected"
    checkpoints = output / "checkpoints"
    feature_cache = prepared_root / "feature_cache_corrected"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)

    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))

    # These are the selected defaults in v39_DirectFinalMS/run.sh.  Set them BEFORE
    # config.py is imported so all derived flags/losses are internally consistent.
    env = {
        "UAVSAT_DEVICE": f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu",
        "UAVSAT_OUTPUT_DIR": str(output),
        "UAVSAT_CHECKPOINT_DIR": str(checkpoints),
        "UAVSAT_FEATURE_CACHE_DIR": str(feature_cache),
        "UAVSAT_DATA_ROOT": str(prepared_root),
        "UAVSAT_BACKBONE": str(args.backbone),
        "UAVSAT_ARCHITECTURE_NAME": FINAL_ARCH,
        "UAVSAT_REFERENCE_PROTOCOL": "controlled_gt_jitter",
        "UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid",
        "UAVSAT_EXPERIMENT_FRAME_COUNT": "3",
        "UAVSAT_EXPERIMENT_MOTION": "velocity",
        "UAVSAT_EXPERIMENT_KALMAN": "fixed",
        "UAVSAT_EXPERIMENT_DISABLE_GRU": "0",
        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1",
        "UAVSAT_SAT_IMAGE": str(Path(exp["satellite_image"]).resolve()),
        "UAVSAT_SAT_JSON": str((prepared_root / "bearing_satellite.json").resolve()),
        "MS_ENABLED": "1",
        "MS_GRID_SIZE": "5",
        "MS_BANDWIDTH_M": "7.0",
    }
    os.environ.update(env)
    return output, checkpoints, feature_cache


def _load_runtime_modules(runtime: Path):
    for name in ("robust_tracker", "visual_localizer", "visual_model", "data", "config"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(runtime))
    try:
        config = importlib.import_module("config")
        tracker = importlib.import_module("robust_tracker")
        visual_localizer = importlib.import_module("visual_localizer")
    finally:
        # Keep imported modules alive, but do not leave runtime ahead of the project forever.
        try:
            sys.path.remove(str(runtime))
        except ValueError:
            pass
    return config, tracker, visual_localizer


def _patch_bearing_paths(config, args, prepared_root: Path) -> None:
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
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

    # Preserve v39 image/candidate geometry. Bearing RSI is 0.25 m/px, therefore
    # SAT_STRIDE=32 means an 8 m lattice spacing.  Do not alter the temporal model
    # speed/acceleration caps to compensate for sparse pseudo frames.
    config.IMAGE_SIZE = 256
    config.UAV_CENTER_CROP_SIZE = 256
    config.UAV_RESIZE_AFTER_CROP = None
    config.TRAIN_UAV_AUGMENT = False
    config.SAT_CROP_SIZE = 320
    config.SAT_STRIDE = 32
    config.GRID_SIZE = 6
    config.CANDIDATE_COUNT = 36
    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(args.jitter_m)

    # route_A is a union used by the inherited visual-head trainer.  Cap only the
    # visual split guard when the external pseudo-route union is short.
    route_a_manifest = prepared_root / "routes" / "route_A" / "manifest.csv"
    with route_a_manifest.open("r", encoding="utf-8") as handle:
        route_a_length = max(0, sum(1 for _ in handle) - 1)
    train_end = int(route_a_length * float(config.TRAIN_FRACTION))
    val_end = int(route_a_length * (float(config.TRAIN_FRACTION) + float(config.VAL_FRACTION)))
    val_span = max(0, val_end - train_end)
    config.SPLIT_GUARD_FRAMES = min(int(config.SPLIT_GUARD_FRAMES), max(0, val_span // 4))


def _audit_canonical(config, runtime: Path, args, prepared_root: Path) -> None:
    expected = {
        "REFERENCE_PROTOCOL": "controlled_gt_jitter",
        "EXPERIMENT_ANCHOR": "weighted_centroid",
        "EXPERIMENT_FRAME_COUNT": 3,
        "EXPERIMENT_MOTION": "velocity",
        "EXPERIMENT_KALMAN": "fixed",
        "EXPERIMENT_DISABLE_GRU": False,
        "FORWARD_ONLY_LOCAL_SEARCH": True,
        "LOSS_ACQUISITION": 0.0,
        "LOSS_MEASUREMENT": 1.0,
        "LOSS_NEXT_STEP": 3.0,
        "LOSS_VELOCITY": 0.25,
    }
    for key, value in expected.items():
        actual = getattr(config, key)
        if actual != value:
            raise RuntimeError("v39 audit failed: %s=%r expected %r" % (key, actual, value))
    if str(config.BACKBONE_KEY) != str(args.backbone):
        raise RuntimeError("v39 audit failed: backbone=%s" % config.BACKBONE_KEY)

    robust_text = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    model_text = (runtime / "visual_model.py").read_text(encoding="utf-8")
    if "Weighted Centroid -> GRU -> Kalman -> ONE final MS -> Final" not in robust_text:
        raise RuntimeError("v39 audit failed: DirectFinalMS runtime patch missing")
    if "self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)" not in model_text:
        raise RuntimeError("v39 audit failed: 5-block Context-GRU patch missing")

    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    frame_counts = {name: int(exp["route_stats"][name]["frames"]) for name in (*TRAIN_ROUTES, *TEST_ROUTES)}
    audit = {
        "architecture": FINAL_ARCH,
        "canonical_source": "v39_DirectFinalMS/base_src + patch_direct_finalms.py + run.sh Context-GRU patch",
        "backbone": str(config.BACKBONE_KEY),
        "reference_protocol": str(config.REFERENCE_PROTOCOL),
        "visual_decoder": str(config.EXPERIMENT_ANCHOR),
        "frame_count": int(config.EXPERIMENT_FRAME_COUNT),
        "motion": str(config.EXPERIMENT_MOTION),
        "kalman": str(config.EXPERIMENT_KALMAN),
        "forward_only": bool(config.FORWARD_ONLY_LOCAL_SEARCH),
        "final_ms_grid": 5,
        "final_ms_bandwidth_m": 7.0,
        "losses": {
            "acquisition": float(config.LOSS_ACQUISITION),
            "measurement": float(config.LOSS_MEASUREMENT),
            "next_step": float(config.LOSS_NEXT_STEP),
            "velocity": float(config.LOSS_VELOCITY),
            "heading": float(config.LOSS_HEADING),
            "variance_nll": float(config.LOSS_VARIANCE_NLL),
        },
        "bearing_step_m": float(args.step_m),
        "route_frames": frame_counts,
        "train_routes": list(TRAIN_ROUTES),
        "test_routes": list(TEST_ROUTES),
    }
    audit_path = Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("[AUDIT] canonical v39 training settings: PASS", flush=True)
    print(json.dumps(audit, indent=2), flush=True)


def _reset_stage_patience(config) -> None:
    if not config.LATEST_TEMPORAL_CHECKPOINT.exists():
        return
    payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
    payload["patience"] = 0
    payload["best_score"] = float("inf")
    payload["best_model"] = payload.get("model")
    torch.save(payload, config.LATEST_TEMPORAL_CHECKPOINT)


def _promote_latest_to_final(config) -> None:
    payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
    torch.save(
        {
            "architecture": str(config.ARCHITECTURE_NAME),
            "model": payload["model"],
            "epoch": int(payload.get("epoch", 0)),
            "best_score": float(payload.get("best_score", float("nan"))),
            "bearing_train_routes": list(TRAIN_ROUTES),
            "bearing_inference_routes": list(TEST_ROUTES),
            "canonical_v39_training": True,
        },
        config.TEMPORAL_CHECKPOINT,
    )


def train_and_infer(args, prepared_root: Path) -> None:
    runtime = _make_runtime(prepared_root)
    _set_canonical_environment(args, prepared_root)
    config, tracker, visual_localizer = _load_runtime_modules(runtime)
    _patch_bearing_paths(config, args, prepared_root)
    _audit_canonical(config, runtime, args, prepared_root)

    device = tracker.resolve_device()
    if not args.resume:
        # A new external-dataset run must not silently inherit the old broken run.
        for path in (config.VISUAL_CHECKPOINT, config.TEMPORAL_CHECKPOINT, config.LATEST_TEMPORAL_CHECKPOINT):
            if Path(path).exists() or Path(path).is_symlink():
                Path(path).unlink()

    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        visual_localizer.train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume),
        )
    else:
        print("reuse visual checkpoint:", config.VISUAL_CHECKPOINT, flush=True)

    visual = visual_localizer.FrozenVisualLocalizer(device)

    # External Bearing data contains independent observations rather than one
    # recorded video. Treat each planned train route as a separate temporal
    # episode (hidden state resets at route boundary), while model+optimizer are
    # resumed across episodes. Total default epochs = 3 * 20 = canonical 60.
    cumulative_epochs = 0
    for stage, route_name in enumerate(TRAIN_ROUTES):
        cumulative_epochs += int(args.epochs_per_route)
        if stage > 0:
            _reset_stage_patience(config)
        root = prepared_root / "routes" / route_name
        cache = tracker.build_route_cache(route_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
        )
        print(
            "\n=== canonical v39 temporal episode %d/%d: %s ==="
            % (stage + 1, len(TRAIN_ROUTES), route_name),
            flush=True,
        )
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
    _promote_latest_to_final(config)
    model = tracker.load_temporal_model(device)

    summaries = {}
    for route_name in TEST_ROUTES:
        root = prepared_root / "routes" / route_name
        cache = tracker.build_route_cache(route_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
        )
        print("\n=== held-out canonical-v39 inference: %s ===" % route_name, flush=True)
        summaries[route_name] = tracker.run_route_inference(
            route_name, visual, model, cache, route, device
        )

    summary_path = Path(config.OUTPUT_DIR) / "bearing_v39_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=float), encoding="utf-8")
    print("\n[DONE] summary:", summary_path, flush=True)
    print("[DONE] audit  :", Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json", flush=True)
    print("[DONE] output :", config.OUTPUT_DIR, flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", default="cityb", choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobileclip2_s2", "resnet18", "resnet50", "mobilenet_v3_small", "vgg16"])
    p.add_argument("--visual-epochs", type=int, default=30)
    p.add_argument("--epochs-per-route", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--jitter-m", type=float, default=8.0)
    # 25 m/frame changed the dynamics and forced the previous adapter to alter
    # v39 speed caps.  8 m/frame fits the canonical 14 m/frame motion envelope.
    p.add_argument("--step-m", type=float, default=8.0)
    p.add_argument("--max-sample-distance-m", type=float, default=15.0)
    p.add_argument("--heading-weight-px-per-deg", type=float, default=0.35)
    p.add_argument("--reuse-visual", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reprepare", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepared_root = HERE / "generated" / args.city
    _ensure_prepared(args, prepared_root)
    train_and_infer(args, prepared_root)


if __name__ == "__main__":
    main()
