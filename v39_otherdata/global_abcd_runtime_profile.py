#!/usr/bin/env python3
"""One shared ABCD inference base for Table-6-style temporal ablations.

Bearing-UAV can concatenate static samples from four maps into merge_c4.  A
causal temporal tracker cannot carry recurrent/Kalman state across unrelated
cities, so city routes remain separate episodes.  This module makes the
*learned weights and every tunable inference constant global*, while state is
reset at each city/route boundary.  Final metrics can then be pooled over all
frames exactly like one merged benchmark.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CITIES = ("citya", "cityb", "cityc", "cityd")


def _route_xy(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append([float(row["x_m"]), float(row["y_m"])])
    x = np.asarray(rows, dtype=np.float64)
    if x.shape[0] < 2:
        raise RuntimeError(f"need >=2 Route-A rows: {path}")
    return x


def pooled_route_a_cadence(suite_root: str | Path) -> dict:
    root = Path(suite_root).resolve()
    all_steps = []
    per_city = {}
    for city in CITIES:
        manifest = root / city / "prepared" / "routes" / "train_01" / "manifest.csv"
        xy = _route_xy(manifest)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if not np.isfinite(steps).all():
            raise RuntimeError(f"non-finite Route-A cadence: {manifest}")
        all_steps.append(steps)
        per_city[city] = {
            "frames": int(xy.shape[0]),
            "steps": int(steps.size),
            "mean_m": float(steps.mean()),
            "p90_m": float(np.percentile(steps, 90)),
            "p95_m": float(np.percentile(steps, 95)),
        }
    steps = np.concatenate(all_steps)
    mean = float(steps.mean())
    p90 = float(np.percentile(steps, 90))
    p95 = float(np.percentile(steps, 95))
    return {
        "source": "pooled ABCD Route-A training sequences",
        "cities": list(CITIES),
        "steps": int(steps.size),
        "mean_m": mean,
        "p90_m": p90,
        "p95_m": p95,
        "per_city_audit": per_city,
    }


def global_base_values(suite_root: str | Path) -> dict:
    c = pooled_route_a_cadence(suite_root)
    mean, p90, p95 = c["mean_m"], c["p90_m"], c["p95_m"]
    return {
        # Pooled cadence: identical for A/B/C/D.
        "INIT_FORWARD_SPEED_M_PER_FRAME": mean,
        "MAX_FORWARD_SPEED_M_PER_FRAME": max(14.0, min(30.0, 1.15 * p95)),
        "MAX_POLYNOMIAL_STEP_M_PER_FRAME": max(14.0, min(30.0, 1.15 * p95)),
        "MAX_MEASUREMENT_CORRECTION_PARALLEL_M": max(6.0, min(14.0, 0.80 * p90)),
        "KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M": max(8.0, min(24.0, 1.05 * p90)),
        "KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M": max(6.0, min(18.0, 0.80 * p90)),
        "KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME": max(2.0, min(6.0, 0.30 * p90)),
        "KALMAN_FINAL_STEP_MAX_M": max(10.0, min(30.0, 1.15 * p95)),
        # Canonical post-GRU-patch Kalman base.  These explicitly erase any
        # city-local kalman_calibration.json that _patch_paths may have loaded.
        "EXPERIMENT_FIXED_VARIANCE_M2": 25.0,
        "KALMAN_Q_PROGRESS": 1.50,
        "KALMAN_Q_CROSS": 0.40,
        "KALMAN_Q_VELOCITY": 1.00,
        "KALMAN_CONFIDENCE_POWER": 0.50,
        "TEMPORAL_ADAPTER_3FRAME_SCALE": 1.00,
        "TEMPORAL_DELTA2_SCALE": 1.00,
        "KALMAN_PRIOR_BLEND_BASE": 0.00,
        "KALMAN_PRIOR_BLEND_LOWCONF_GAIN": 0.18,
        "KALMAN_PRIOR_BLEND_MAX": 0.30,
        "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF": 0.60,
        "KALMAN_STEP_RELAX_CONFIDENCE": 0.55,
        "KALMAN_STEP_RELAX_WIDTH": 0.08,
        "KALMAN_STEP_VISUAL_SLACK_M": 3.0,
    }


def apply_global_base(config, suite_root: str | Path) -> dict:
    cadence = pooled_route_a_cadence(suite_root)
    values = global_base_values(suite_root)
    missing = [k for k in values if not hasattr(config, k)]
    if missing:
        raise RuntimeError("runtime missing global ABCD fields: " + repr(missing))
    for key, value in values.items():
        setattr(config, key, float(value))
    audit = {"cadence": cadence, "runtime_base": values}
    print("[GLOBAL-ABCD-BASE]", json.dumps(audit, sort_keys=True), flush=True)
    return audit
