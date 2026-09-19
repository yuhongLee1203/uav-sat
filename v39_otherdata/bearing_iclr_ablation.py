#!/usr/bin/env python3
"""Train/evaluate the ICLR temporal ablation on prepared Bearing-UAV routes.

The paper-facing chain contains exactly one MeanShift decoder:

    6x6 geometry -> forward 3x6 visual scores -> 3-frame GRU
    -> fixed-R Kalman -> one final local MeanShift -> XY

Training is supervised on Route A. Evaluation preserves the existing
Bearing-UAV ``controlled_gt_jitter`` protocol and its GT/reference behavior;
this runner does not alter the dataset loader, route preparation, or labels.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

import bearing_runner_multicity_v39 as multi

exact = multi.exact
base = exact.base
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SIMPLE_GRU_PATCH = REPO_ROOT / "v39_DirectFinalMS" / "patch_simple_figure_gru.py"

ARCH = "ICLR_Forward18_Simple3FrameGRU_FixedKalman_OneFinalMS"
VARIANTS = {
    "full": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=6),
    "no_gru": dict(frames=3, disable_gru=True, kalman="fixed", ms=True, grid=6),
    "no_kalman": dict(frames=3, disable_gru=False, kalman="none", ms=True, grid=6),
    "no_ms": dict(frames=3, disable_gru=False, kalman="fixed", ms=False, grid=6),
    "frames1": dict(frames=1, disable_gru=False, kalman="fixed", ms=True, grid=6),
    "frames2": dict(frames=2, disable_gru=False, kalman="fixed", ms=True, grid=6),
    "grid4": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=4),
    "grid5": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=5),
    "grid7": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=7),
    "grid8": dict(frames=3, disable_gru=False, kalman="fixed", ms=True, grid=8),
}


def _prepared_root(args: argparse.Namespace) -> Path:
    return Path(args.suite_root).resolve() / args.city / "prepared"


def _train_root(args: argparse.Namespace) -> Path:
    return Path(args.suite_root).resolve() / args.city / "train_full"


def _variant_root(args: argparse.Namespace) -> Path:
    return Path(args.suite_root).resolve() / args.city / "variants" / args.variant


def _lock_prepared(args: argparse.Namespace, prepared_root: Path) -> None:
    """Validate existing data without imposing a different preparation recipe.

    Legacy packages can omit the selection version. Keep that fact in the
    output audit; do not label them v13 or regenerate their routes/GT.
    """
    if args.reprepare:
        raise RuntimeError("This ablation reuses existing prepared data; --reprepare is unsupported.")
    exp = exact._experiment(prepared_root)
    errors = []
    if not exp.get("dataset_root") or Path(exp["dataset_root"]).resolve() != Path(args.dataset_root).resolve():
        errors.append("dataset_root mismatch")
    if str(exp.get("city", "")).lower() != args.city.lower():
        errors.append("city mismatch")
    routes = ("train_01", "test_01", "test_02")
    steps = {}
    fingerprints = {}
    for name in routes:
        stats = exp.get("route_stats", {}).get(name, {})
        try:
            step = float(stats["sample_step_m"])
            if not math.isfinite(step) or step <= 0:
                raise ValueError
            steps[name] = step
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name}: missing/invalid sample_step_m")
        for filename in ("manifest.csv", "waypoints.json"):
            relative = f"routes/{name}/{filename}"
            path = prepared_root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty {path}")
            else:
                fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(steps) == 3 and max(steps.values()) - min(steps.values()) > 1e-6:
        errors.append(f"route sample steps differ: {steps}")
    step = steps.get("train_01")
    if args.step_m is not None and step is not None:
        if not math.isfinite(args.step_m) or abs(args.step_m - step) > 1e-6:
            errors.append(f"explicit --step-m={args.step_m} differs from prepared step={step}; omit --step-m to reuse it")
    train = exp.get("route_stats", {}).get("train_01", {})
    for key in ("actual_step_mean_m", "actual_step_p90_m", "actual_step_p95_m"):
        try:
            value = float(train[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"train_01: missing/invalid {key}")
    try:
        mpp = float(exp["mpp"])
        if not math.isfinite(mpp) or mpp <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("missing/invalid mpp")
    for path in (prepared_root / "bearing_satellite.json", Path(exp.get("satellite_image", ""))):
        if not path.is_file():
            errors.append(f"missing satellite file: {path}")
    if errors:
        raise RuntimeError("Bearing ICLR prepared-data validation failed:\n- " + "\n- ".join(errors))
    args.step_m = step
    args.epochs_per_route = int(args.temporal_epochs)
    fingerprints["experiment.json"] = hashlib.sha256((prepared_root / "experiment.json").read_bytes()).hexdigest()
    fingerprints["bearing_satellite.json"] = hashlib.sha256((prepared_root / "bearing_satellite.json").read_bytes()).hexdigest()
    args.prepared_data_audit = {
        "prepared_root": str(prepared_root.resolve()),
        "selection_version": exp.get("sequence_selection_version"),
        "selection_version_status": "recorded" if exp.get("sequence_selection_version") else "unrecorded_legacy",
        "sample_step_m": steps,
        "sha256": fingerprints,
        "data_modified": False,
    }
    lock = Path(args.suite_root).resolve() / args.city / "prepared_contract.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != args.prepared_data_audit:
        raise RuntimeError(f"Prepared data changed within this suite: {lock}; use a new suite directory.")
    if not lock.exists():
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(args.prepared_data_audit, indent=2), encoding="utf-8")
    print(f"[PREPARED] PASS | city={args.city} | step={step:g} m | "
          f"selection={exp.get('sequence_selection_version') or 'unrecorded_legacy'} | existing GT/routes unchanged", flush=True)


def _make_runtime(prepared_root: Path, runtime_root: Path) -> Path:
    """Build the same runtime for training and every ablation row."""
    del prepared_root  # Runtime code is independent of the prepared-data path.
    for filename in base.RUNTIME_FILES:
        source = base.CANONICAL_BASE / filename
        if not source.exists():
            raise FileNotFoundError(source)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    for filename in base.RUNTIME_FILES:
        shutil.copy2(base.CANONICAL_BASE / filename, runtime_root / filename)
    shutil.copy2(HERE / "data.py", runtime_root / "data.py")
    subprocess.run(
        [
            sys.executable,
            str(base.CANONICAL_FINALMS_PATCH),
            str(runtime_root / "robust_tracker.py"),
        ],
        check=True,
    )
    base._patch_context_gru(runtime_root)
    if not SIMPLE_GRU_PATCH.exists():
        raise FileNotFoundError(SIMPLE_GRU_PATCH)
    subprocess.run(
        [sys.executable, str(SIMPLE_GRU_PATCH), str(runtime_root / "visual_model.py")],
        check=True,
    )
    exact._patch_final_ms_reference(runtime_root)
    return runtime_root


def _set_environment(
    args: argparse.Namespace,
    prepared_root: Path,
    output: Path,
    variant: dict,
    *,
    training: bool,
) -> None:
    checkpoints = output / "checkpoints"
    feature_cache = Path(args.suite_root).resolve() / args.city / "feature_cache"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)
    exp = exact._experiment(prepared_root)
    protocol = "controlled_gt_jitter"
    os.environ.update({
        "UAVSAT_DEVICE": f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu",
        "UAVSAT_OUTPUT_DIR": str(output),
        "UAVSAT_CHECKPOINT_DIR": str(checkpoints),
        "UAVSAT_FEATURE_CACHE_DIR": str(feature_cache),
        "UAVSAT_DATA_ROOT": str(prepared_root),
        "UAVSAT_BACKBONE": str(args.backbone),
        "UAVSAT_ARCHITECTURE_NAME": ARCH,
        "UAVSAT_REFERENCE_PROTOCOL": protocol,
        # Forward-18 posterior is summarized without a front MeanShift.  The
        # only MeanShift in the paper chain is the post-Kalman decoder.
        "UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid",
        "UAVSAT_EXPERIMENT_FRAME_COUNT": str(int(variant["frames"])),
        "UAVSAT_EXPERIMENT_MOTION": "velocity",
        "UAVSAT_EXPERIMENT_KALMAN": str(variant["kalman"]),
        "UAVSAT_EXPERIMENT_DISABLE_GRU": "1" if variant["disable_gru"] else "0",
        "UAVSAT_EXPERIMENT_FORWARD_ONLY": "1",
        "UAVSAT_SAT_IMAGE": str(Path(exp["satellite_image"]).resolve()),
        "UAVSAT_SAT_JSON": str((prepared_root / "bearing_satellite.json").resolve()),
        "UAVSAT_MEASURE_LATENCY": "0" if training else "1",
        "UAVSAT_LATENCY_WARMUP": str(int(args.latency_warmup)),
        "MS_ENABLED": "1" if variant["ms"] else "0",
        "MS_GRID_SIZE": str(int(variant["grid"])),
        "MS_BANDWIDTH_M": str(float(args.ms_bandwidth_m)),
        "MS_MEASURE_LATENCY": "0" if training else "1",
        "MS_LATENCY_WARMUP": str(int(args.latency_warmup)),
        "UAVSAT_SEED": str(int(args.seed)),
    })


def _patch_paths(config, args: argparse.Namespace, prepared_root: Path) -> None:
    exact._patch_paths_and_scale(config, args, prepared_root)
    config.ARCHITECTURE_NAME = ARCH
    config.TEMPORAL_EPOCHS = int(args.temporal_epochs)


def _checkpoint_source(prepared_root: Path, name: str) -> Path | None:
    candidates = [
        prepared_root / "v39_output_bearing_adapted" / "checkpoints" / name,
        prepared_root / "v39_output_corrected" / "checkpoints" / name,
    ]
    return next((p for p in candidates if p.exists()), None)


def _reuse_visual_checkpoint(config, prepared_root: Path) -> None:
    dest = Path(config.VISUAL_CHECKPOINT)
    if dest.exists() or dest.is_symlink():
        return
    src = _checkpoint_source(prepared_root, dest.name)
    if src is None:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src.resolve())
    print(f"[REUSE] visual checkpoint -> {src}", flush=True)


def _link_full_checkpoints(config, train_root: Path) -> None:
    source = train_root / "checkpoints"
    for dest in (Path(config.VISUAL_CHECKPOINT), Path(config.TEMPORAL_CHECKPOINT)):
        src = source / dest.name
        if not src.exists():
            raise FileNotFoundError(f"missing trained full checkpoint: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())


def _audit_runtime(config, runtime: Path, variant: dict, training: bool) -> dict:
    tracker_text = (runtime / "robust_tracker.py").read_text(encoding="utf-8")
    model_text = (runtime / "visual_model.py").read_text(encoding="utf-8")
    checks = {
        "forward_18": int(config.FORWARD_SEARCH_CANDIDATE_COUNT) == 18,
        "base_geometry_6x6": int(config.ACQ_LOCAL_GRID_SIZE) == 6,
        "simple_six_block_gru": "nn.GRUCell(feature_dim * 6" in model_text,
        "one_final_ms_source": "online final path contains exactly one MeanShift" in tracker_text,
        "front_decoder_not_ms": str(config.EXPERIMENT_ANCHOR) == "weighted_centroid",
        "protocol": str(config.REFERENCE_PROTOCOL) == "controlled_gt_jitter",
        "controlled_gt_reference_enabled": not bool(config.NO_GT_INFERENCE),
        "frame_count": int(config.EXPERIMENT_FRAME_COUNT) == int(variant["frames"]),
        "gru_flag": bool(config.EXPERIMENT_DISABLE_GRU) == bool(variant["disable_gru"]),
        "kalman_flag": str(config.EXPERIMENT_KALMAN) == str(variant["kalman"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"[ARCH-AUDIT] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    if failed:
        raise RuntimeError("architecture audit failed: " + ", ".join(failed))
    return checks


def train_full(args: argparse.Namespace) -> None:
    prepared = _prepared_root(args)
    _lock_prepared(args, prepared)
    output = _train_root(args)
    runtime = _make_runtime(prepared, output / "runtime")
    _set_environment(args, prepared, output, VARIANTS["full"], training=True)
    config, tracker, visual_localizer = base._load_runtime_modules(runtime)
    _patch_paths(config, args, prepared)
    audit = _audit_runtime(config, runtime, VARIANTS["full"], training=True)
    _reuse_visual_checkpoint(config, prepared)

    device = tracker.resolve_device()
    if not Path(config.VISUAL_CHECKPOINT).exists():
        visual_localizer.train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=Path(config.VISUAL_CHECKPOINT).exists(),
        )
    else:
        print("[TRAIN] reuse visual checkpoint", config.VISUAL_CHECKPOINT, flush=True)

    final_ckpt = Path(config.TEMPORAL_CHECKPOINT)
    if final_ckpt.exists() and not args.force_train:
        try:
            payload = torch.load(final_ckpt, map_location="cpu")
            if payload.get("architecture") == ARCH:
                print("[TRAIN] reuse completed temporal checkpoint", final_ckpt, flush=True)
                _write_manifest(args, output, VARIANTS["full"], audit, training=True)
                return
        except Exception:
            pass

    visual = visual_localizer.FrozenVisualLocalizer(device)
    cache = tracker.build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route = tracker.WaypointRoute(
        tracker.load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon)
    )
    latest = Path(config.LATEST_TEMPORAL_CHECKPOINT)
    resume = latest.exists() and not args.force_train
    tracker.train_temporal_model(
        visual=visual,
        cache=cache,
        route=route,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=resume,
    )
    if not final_ckpt.exists():
        raise RuntimeError(f"training did not produce {final_ckpt}")
    _write_manifest(args, output, VARIANTS["full"], audit, training=True)


def evaluate(args: argparse.Namespace) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(args.variant)
    variant = VARIANTS[args.variant]
    prepared = _prepared_root(args)
    _lock_prepared(args, prepared)
    output = _variant_root(args)
    runtime = _make_runtime(prepared, output / "runtime")
    _set_environment(args, prepared, output, variant, training=False)
    config, tracker, visual_localizer = base._load_runtime_modules(runtime)
    _patch_paths(config, args, prepared)
    _link_full_checkpoints(config, _train_root(args))
    audit = _audit_runtime(config, runtime, variant, training=False)

    device = tracker.resolve_device()
    visual = visual_localizer.FrozenVisualLocalizer(device)
    model = tracker.load_temporal_model(device)
    cadence = exact._training_only_adaptation(prepared)
    geometry = exact._physical_sat_geometry(prepared)
    summaries = {}
    for external_name, canonical_name, root in (
        ("test_01", "route_B", config.ROUTE_ROOTS[1]),
        ("test_02", "route_C", config.ROUTE_ROOTS[2]),
    ):
        cache = tracker.build_route_cache(canonical_name, root, visual, device)
        route = tracker.WaypointRoute(
            tracker.load_waypoint_xy(canonical_name, visual.origin_lat, visual.origin_lon)
        )
        print(f"\n=== {args.city} {args.variant}: {external_name} ===", flush=True)
        result = tracker.run_route_inference(
            external_name, visual, model, cache, route, device
        )
        result["ICLRProtocol"] = {
            "uses_controlled_gt_reference": True,
            "reference_protocol": "controlled_gt_jitter",
            "training_route": "train_01",
            "held_out_route": external_name,
            "base_candidate_geometry": "6x6",
            "scored_forward_candidates": 18,
            "front_decoder": "posterior_weighted_centroid",
            "online_meanshift_count": 1 if variant["ms"] else 0,
            "final_ms_grid": int(variant["grid"]),
            "training_only_cadence": cadence,
            "physical_sat_geometry": geometry,
        }
        summaries[external_name] = result

    summary_path = output / "bearing_v39_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, default=float), encoding="utf-8"
    )
    _write_manifest(args, output, variant, audit, training=False)
    print("[DONE]", summary_path, flush=True)


def _write_manifest(args, output: Path, variant: dict, audit: dict, training: bool) -> None:
    manifest = {
        "architecture": ARCH,
        "city": args.city,
        "phase": "training" if training else "held_out_evaluation",
        "variant": "full" if training else args.variant,
        "variant_settings": variant,
        "seed": int(args.seed),
        "temporal_epochs": int(args.temporal_epochs),
        "patience": int(args.patience),
        "reference_protocol": "controlled_gt_jitter",
        "uses_controlled_gt_reference": True,
        "candidate_geometry": "6x6 local geometry; heading-guided forward 3x6 = 18 scored patches",
        "paper_chain": "Forward18 posterior -> 3-frame GRU -> fixed-R Kalman -> one final MeanShift -> XY",
        "architecture_audit": audit,
        "prepared_data": args.prepared_data_audit,
        "integrity": "Measured outputs are never edited, clipped, or reordered to force Full to win.",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "train", "eval"])
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--suite-root", required=True)
    p.add_argument("--dataset-root", default="/yh/study/cvpr_data/Bearing_UAV_90K")
    p.add_argument("--city", required=True, choices=["citya", "cityb", "cityc", "cityd"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--visual-epochs", type=int, default=30)
    p.add_argument("--temporal-epochs", type=int, default=80)
    p.add_argument("--epochs-per-route", type=int, default=80)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--jitter-m", type=float, default=8.0)
    p.add_argument("--step-m", type=float, default=None,
                   help="Optional assertion; by default use the existing prepared sample_step_m.")
    p.add_argument("--max-sample-distance-m", type=float, default=15.0)
    p.add_argument("--heading-weight-px-per-deg", type=float, default=0.0)
    p.add_argument("--ms-bandwidth-m", type=float, default=7.0)
    p.add_argument("--latency-warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=2033)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--reprepare", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "check":
        _lock_prepared(args, _prepared_root(args))
    elif args.mode == "train":
        train_full(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
