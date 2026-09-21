#!/usr/bin/env python3
"""Train/evaluate the ICLR temporal ablation on prepared Bearing-UAV routes.

The paper-facing chain uses an explicit front SoftMS and final MeanShift:

    6x6 geometry -> forward 3x6 = 18 visual scores -> front SoftMS
    -> residual temporal GRU -> constrained Kalman -> final MeanShift -> XY

Training is supervised on Route A. Evaluation preserves the existing
Bearing-UAV ``controlled_gt_jitter`` protocol and its GT/reference behavior;
this runner does not alter the dataset loader, route preparation, or labels.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import bearing_runner_multicity_v39 as multi

exact = multi.exact
base = exact.base
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SIMPLE_GRU_PATCH = REPO_ROOT / "v39_DirectFinalMS" / "patch_simple_figure_gru.py"

ARCH = "ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS"
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


def _train_root(args: argparse.Namespace, frames=None) -> Path:
    frame_count = int(frames if frames is not None else getattr(args, "train_frames", 3))
    return Path(args.suite_root).resolve() / args.city / f"train_frames{frame_count}"


def _variant_root(args: argparse.Namespace) -> Path:
    return Path(args.suite_root).resolve() / args.city / "variants" / args.variant


def _training_step_statistics(prepared_root: Path):
    """Read cadence metadata, deriving absent legacy fields from Route A GT.

    manifest.csv x_m/y_m are the metric coordinates consumed by RouteDataset.
    Preserve CSV order, including zero-length steps; never sort, smooth, resample,
    read test routes, or write back to the prepared package.
    """
    exp = exact._experiment(prepared_root)
    train = exp["route_stats"]["train_01"]
    keys = ("actual_step_mean_m", "actual_step_p90_m", "actual_step_p95_m")
    missing = [key for key in keys if train.get(key) is None]
    values = {}
    for key in keys:
        if key in missing:
            continue
        try:
            value = float(train[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"train_01: invalid {key}={train[key]!r}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"train_01: invalid {key}={value}")
        values[key] = value
    if missing:
        manifest = prepared_root / "routes" / "train_01" / "manifest.csv"
        coordinates = []
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"x_m", "y_m"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Route A GT requires x_m/y_m columns: {manifest}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    xy = (float(row["x_m"]), float(row["y_m"]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid Route A GT at {manifest}:{line_number}") from exc
                if not all(math.isfinite(v) for v in xy):
                    raise ValueError(f"Non-finite Route A GT at {manifest}:{line_number}")
                coordinates.append(xy)
        if len(coordinates) < 2:
            raise ValueError(f"Need at least two Route A GT rows for cadence: {manifest}")
        steps = np.linalg.norm(np.diff(np.asarray(coordinates, dtype=np.float64), axis=0), axis=1)
        if not np.isfinite(steps).all():
            raise ValueError(f"Non-finite Route A GT step distances: {manifest}")
        derived = dict(zip(keys, (float(steps.mean()), float(np.percentile(steps, 90)),
                                  float(np.percentile(steps, 95)))))
        values.update({key: derived[key] for key in missing})
    return values, {
        "source": "train_01_manifest_gt" if missing else "experiment_route_stats_train_01",
        "derived_fields": missing,
    }


def _existing_training_adaptation(prepared_root: Path):
    """Derive all cadence values from train_01 only. Never inspect B/C stats."""
    train, provenance = _training_step_statistics(prepared_root)
    p90 = train["actual_step_p90_m"]
    p95 = train["actual_step_p95_m"]
    mean = train["actual_step_mean_m"]

    # Longitudinal limits only. Cross-track limits remain canonical so the
    # external adapter does not gain extra freedom to wiggle sideways.
    values = {
        "source": "train_01_only",
        "statistics_provenance": provenance,
        "train_step_mean_m": mean,
        "train_step_p90_m": p90,
        "train_step_p95_m": p95,
        "max_forward_speed_m_per_frame": max(14.0, min(30.0, 1.15 * p95)),
        "max_polynomial_step_m_per_frame": max(14.0, min(30.0, 1.15 * p95)),
        "max_measurement_correction_parallel_m": max(6.0, min(14.0, 0.80 * p90)),
        "kalman_max_measurement_innovation_progress_m": max(8.0, min(24.0, 1.05 * p90)),
        "kalman_max_posterior_correction_progress_m": max(6.0, min(18.0, 0.80 * p90)),
        "kalman_max_velocity_correction_m_per_frame": max(2.0, min(6.0, 0.30 * p90)),
        "kalman_final_step_max_m": max(10.0, min(30.0, 1.15 * p95)),
    }
    return values



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
    try:
        cadence = _existing_training_adaptation(prepared_root)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"train_01 cadence: {exc}")
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
    args.training_cadence_audit = cadence
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
    print("[CADENCE] source=%s | mean=%.6f m | p90=%.6f m | p95=%.6f m" % (
        cadence["statistics_provenance"]["source"], cadence["train_step_mean_m"],
        cadence["train_step_p90_m"], cadence["train_step_p95_m"]), flush=True)


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
    # Legacy 5-block Context-GRU prepatch intentionally disabled here.
    # patch_simple_figure_gru.py owns the complete temporal architecture.
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
        # Forward-18 is always decoded by Soft MeanShift.
        "UAVSAT_EXPERIMENT_ANCHOR": "softms",
        "UAVSAT_EXPERIMENT_FRAME_COUNT": str(int(variant["frames"])),
        "UAVSAT_EXPERIMENT_MOTION": "quadratic",
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
    # Only this ablation process installs the legacy-metadata adapter.
    # The existing runner files, GT loader and prepared data stay unchanged.
    exact._training_only_adaptation = _existing_training_adaptation
    exact._patch_paths_and_scale(config, args, prepared_root)
    geometry = getattr(config, "BEARING_PHYSICAL_SAT_GEOMETRY", None)
    if not isinstance(geometry, dict) or "sat_stride_m" not in geometry:
        raise RuntimeError("missing audited Bearing physical SAT geometry")
    # Cover the bounded local-prior jitter before retaining only the
    # heading-forward 18 cells.  Derived only from protocol geometry.
    config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = (
        float(config.CONTROLLED_GT_PRIOR_JITTER_M)
        + 0.5 * float(geometry["sat_stride_m"])
    )
    # Initialize recurrent + Kalman motion from this city training cadence.
    config.INIT_FORWARD_SPEED_M_PER_FRAME = float(
        args.training_cadence_audit["train_step_mean_m"]
    )
    # Apply a Kalman profile selected on this city training validation only.
    calibration_path = _train_root(args, 3) / "kalman_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        best = calibration.get("best", {})
        for key, attr in (
            ("fixed_variance_m2", "EXPERIMENT_FIXED_VARIANCE_M2"),
            ("q_progress", "KALMAN_Q_PROGRESS"),
            ("q_cross", "KALMAN_Q_CROSS"),
            ("q_velocity", "KALMAN_Q_VELOCITY"),
            ("confidence_power", "KALMAN_CONFIDENCE_POWER"),
            ("temporal_3frame_scale", "TEMPORAL_ADAPTER_3FRAME_SCALE"),
            ("delta2_scale", "TEMPORAL_DELTA2_SCALE"),
            ("prior_blend_base", "KALMAN_PRIOR_BLEND_BASE"),
            ("prior_blend_lowconf_gain", "KALMAN_PRIOR_BLEND_LOWCONF_GAIN"),
            ("prior_blend_max", "KALMAN_PRIOR_BLEND_MAX"),
            ("prior_blend_cutoff", "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF"),
            ("step_relax_confidence", "KALMAN_STEP_RELAX_CONFIDENCE"),
            ("step_relax_width", "KALMAN_STEP_RELAX_WIDTH"),
            ("step_visual_slack_m", "KALMAN_STEP_VISUAL_SLACK_M"),
        ):
            if key in best:
                setattr(config, attr, float(best[key]))
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
        "front_softms_source": (
            "anchor_xy_all = candidate.softms_xy" in tracker_text
            and tracker_text.count("soft_mean_shift(") == 3
            and 'getattr(config, "EXPERIMENT_ANCHOR"' not in tracker_text
        ),
        "one_final_ms_source": "exactly one final local Soft MeanShift after the Kalman estimator" in tracker_text,
        "front_decoder_softms": str(config.EXPERIMENT_ANCHOR) == "softms",
        "simple_seven_block_gru": "nn.GRUCell(feature_dim * 7" in model_text,
        "residual_motion_head": "MOTION_RESIDUAL_FORWARD_M" in model_text,
        "temporal_reliability_gate": "self.temporal_reliability_head" in model_text,
        "causal_visual_displacement": "self.visual_motion_projection(visual_motion)" in model_text,
        "cadence_initialized_tracker": "INIT_FORWARD_SPEED_M_PER_FRAME" in tracker_text,
        "motion_training_inference_aligned": str(config.EXPERIMENT_MOTION) == "quadratic",
        "training_city_motion_scale_init": float(getattr(config, "INIT_FORWARD_SPEED_M_PER_FRAME", 0.0)) > 0.0,
        "forward_origin_backshift_covers_jitter": (
            float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M)
            >= float(config.CONTROLLED_GT_PRIOR_JITTER_M)
        ),
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



def _calibrate_kalman_on_training_validation(args, config, tracker, visual, model, cache, route):
    """V5 train-validation selection for direct second-order motion + Kalman."""
    if int(args.train_frames) != 3:
        return None

    gt_state = tracker.build_gt_route_state(cache, route)
    split = tracker.split_ranges(len(cache))
    val_range = split["val"]
    device = tracker.resolve_device()
    profiles = []

    fields = (
        ("fixed_variance_m2", "EXPERIMENT_FIXED_VARIANCE_M2"),
        ("q_progress", "KALMAN_Q_PROGRESS"),
        ("q_cross", "KALMAN_Q_CROSS"),
        ("q_velocity", "KALMAN_Q_VELOCITY"),
        ("confidence_power", "KALMAN_CONFIDENCE_POWER"),
        ("temporal_3frame_scale", "TEMPORAL_ADAPTER_3FRAME_SCALE"),
        ("delta2_scale", "TEMPORAL_DELTA2_SCALE"),
        ("prior_blend_base", "KALMAN_PRIOR_BLEND_BASE"),
        ("prior_blend_lowconf_gain", "KALMAN_PRIOR_BLEND_LOWCONF_GAIN"),
        ("prior_blend_max", "KALMAN_PRIOR_BLEND_MAX"),
        ("prior_blend_cutoff", "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF"),
        ("step_relax_confidence", "KALMAN_STEP_RELAX_CONFIDENCE"),
        ("step_relax_width", "KALMAN_STEP_RELAX_WIDTH"),
        ("step_visual_slack_m", "KALMAN_STEP_VISUAL_SLACK_M"),
    )

    def snapshot():
        return {key: float(getattr(config, attr)) for key, attr in fields}

    def apply_values(values):
        for key, attr in fields:
            if key in values:
                setattr(config, attr, float(values[key]))

    def evaluate(stage, updates):
        apply_values(updates)
        result = tracker.evaluate_closed_loop(
            model, visual, cache, route, gt_state, val_range, device
        )
        row = snapshot()
        row.update({
            "stage": stage,
            "val_mle_m": float(result["mle"]),
            "val_p90_m": float(result["p90"]),
            "val_speed_mae": float(result["speed_mae"]),
            "val_progress_mae": float(result["progress_mae"]),
        })
        row["objective"] = float(
            row["val_mle_m"]
            + 0.08 * row["val_p90_m"]
            + 0.02 * row["val_speed_mae"]
            + 0.01 * row["val_progress_mae"]
        )
        profiles.append(row)
        return row

    def choose(rows):
        best = min(rows, key=lambda r: (
            r["objective"], r["val_mle_m"], r["val_p90_m"]
        ))
        apply_values(best)
        return best

    baseline = evaluate("baseline", {})

    temporal_rows = []
    # The original search started at 0.90 and CityA selected that lower
    # boundary while disabling delta2 entirely.  Include conservative residual
    # strengths so train-validation can reject noisy pseudo-temporal context
    # without consulting either held-out test route.
    for temporal_scale in (0.00, 0.25, 0.50, 0.75, 0.90, 1.00, 1.10):
        for delta2_scale in (0.00, 0.25, 0.50, 0.75, 1.00):
            temporal_rows.append(evaluate("direct_delta2", {
                "temporal_3frame_scale": temporal_scale,
                "delta2_scale": delta2_scale,
            }))
    temporal_best = choose(temporal_rows + [baseline])

    blend_rows = []
    for base in (0.00, 0.02, 0.05):
        for gain in (0.10, 0.20, 0.30):
            for cutoff in (0.60, 0.70):
                blend_rows.append(evaluate("blend", {
                    "prior_blend_base": base,
                    "prior_blend_lowconf_gain": gain,
                    "prior_blend_max": 0.30,
                    "prior_blend_cutoff": cutoff,
                }))
    blend_best = choose(blend_rows + [temporal_best])

    step_rows = []
    for center, width, slack in (
        (0.00, 0.05, 10.0),
        (0.35, 0.08, 8.0),
        (0.50, 0.08, 6.0),
    ):
        step_rows.append(evaluate("step", {
            "step_relax_confidence": center,
            "step_relax_width": width,
            "step_visual_slack_m": slack,
        }))
    step_best = choose(step_rows + [blend_best])

    filter_rows = []
    for fixed_r in (6.0, 9.0):
        for q_scale in (0.75, 1.00):
            for conf_power in (0.75, 1.00):
                filter_rows.append(evaluate("filter", {
                    "fixed_variance_m2": fixed_r,
                    "q_progress": 1.50 * q_scale,
                    "q_cross": 0.40 * q_scale,
                    "q_velocity": 1.00 * q_scale,
                    "confidence_power": conf_power,
                }))
    best = choose(filter_rows + [step_best])

    selected = snapshot()
    kalman_mode = str(config.EXPERIMENT_KALMAN)
    config.EXPERIMENT_KALMAN = "none"
    no_k = tracker.evaluate_closed_loop(
        model, visual, cache, route, gt_state, val_range, device
    )
    config.EXPERIMENT_KALMAN = kalman_mode
    apply_values(selected)

    payload = {
        "selection_source": "current_city_training_validation_only",
        "city": args.city,
        "validation_range": [int(val_range[0]), int(val_range[1])],
        "criterion": "mle + .08*p90 + .02*speed_mae + .01*progress_mae",
        "search": "direct_delta2_second_order_v5",
        "best": best,
        "validation_no_kalman_diagnostic": {
            "mle_m": float(no_k["mle"]),
            "p90_m": float(no_k["p90"]),
        },
        "profiles": profiles,
        "held_out_navigation_read": False,
    }
    out = _train_root(args, 3) / "kalman_calibration.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[TRAIN-ONLY V5 CALIBRATION]", json.dumps(best, sort_keys=True), flush=True)
    print("[TRAIN-ONLY V5 NO-KALMAN DIAGNOSTIC]", json.dumps(payload["validation_no_kalman_diagnostic"], sort_keys=True), flush=True)
    return payload

def train_full(args: argparse.Namespace) -> None:
    prepared = _prepared_root(args)
    _lock_prepared(args, prepared)
    train_variant = dict(VARIANTS["full"])
    train_variant["frames"] = int(args.train_frames)
    output = _train_root(args, args.train_frames)
    runtime = _make_runtime(prepared, output / "runtime")
    _set_environment(args, prepared, output, train_variant, training=True)
    config, tracker, visual_localizer = base._load_runtime_modules(runtime)
    _patch_paths(config, args, prepared)
    audit = _audit_runtime(config, runtime, train_variant, training=True)
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
                _write_manifest(args, output, train_variant, audit, training=True)
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
    if int(args.train_frames) == 3:
        calibrated_model = tracker.load_temporal_model(device)
        _calibrate_kalman_on_training_validation(
            args, config, tracker, visual, calibrated_model, cache, route
        )
    _write_manifest(args, output, train_variant, audit, training=True)


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
    checkpoint_frames = int(variant["frames"]) if args.variant in {"frames1", "frames2"} else 3
    _link_full_checkpoints(config, _train_root(args, checkpoint_frames))
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
            canonical_name, visual, model, cache, route, device
        )
        result["ICLRProtocol"] = {
            "uses_controlled_gt_reference": True,
            "reference_protocol": "controlled_gt_jitter",
            "training_sequence": "current_city_training_sequence",
            "held_out_navigation": external_name,
            "base_candidate_geometry": "6x6",
            "scored_forward_candidates": 18,
            "dataset_domain": args.city,
            "held_out_navigation": external_name,
            "front_decoder": "forward18_softms",
            "front_meanshift_count": 1,
            "motion_predictor": "residual_heading_aware_quadratic_next_step",
            "motion_init_source": "current_city_training_cadence",
            "motion_init_m_per_frame": float(config.INIT_FORWARD_SPEED_M_PER_FRAME),
            "final_meanshift_count": 1 if variant["ms"] else 0,
            "online_meanshift_count": 2 if variant["ms"] else 1,
            "forward_origin_backshift_m": float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M),
            "controlled_prior_jitter_m": float(config.CONTROLLED_GT_PRIOR_JITTER_M),
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
        "paper_chain": "Forward18 SoftMS -> residual temporal GRU -> constrained Kalman -> final MeanShift -> XY",
        "dataset_protocol": "Bearing-UAV citya/cityb/cityc/cityd with held-out test_01/test_02 reporting",
        "architecture_audit": audit,
        "prepared_data": args.prepared_data_audit,
        "training_cadence": args.training_cadence_audit,
        "integrity": "Measured outputs are never edited, clipped, or reordered to force Full to win.",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "train", "eval"])
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--train-frames", type=int, default=3, choices=[1, 2, 3])
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
