#!/usr/bin/env python3
"""Reject prepared routes that are not valid temporal navigation episodes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _manifest(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 24:
        raise RuntimeError(f"too few temporal frames ({len(rows)}): {path}")
    return np.asarray(
        [[float(row["x_m"]), float(row["y_m"])] for row in rows],
        dtype=np.float64,
    )


def _waypoints(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("waypoints", payload) if isinstance(payload, dict) else payload
    points = np.asarray(
        [[float(row["longitude"]), float(row["latitude"])] for row in rows],
        dtype=np.float64,
    )
    if len(points) < 2:
        raise RuntimeError(f"too few waypoints: {path}")
    return points


def _cross_track(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    distances = []
    for start, end in zip(route[:-1], route[1:]):
        delta = end - start
        denom = float(delta @ delta)
        if denom <= 1e-12:
            continue
        t = np.clip(((points - start) @ delta) / denom, 0.0, 1.0)
        projection = start[None, :] + t[:, None] * delta[None, :]
        distances.append(np.linalg.norm(points - projection, axis=1))
    if not distances:
        raise RuntimeError("route has no non-zero waypoint leg")
    return np.min(np.stack(distances, axis=0), axis=0)


def audit_city(root: Path, city: str) -> dict:
    prepared = root / city / "prepared" / "routes"
    train = _manifest(prepared / "train_01" / "manifest.csv")
    train_steps = np.linalg.norm(np.diff(train, axis=0), axis=1)
    train_mean = float(train_steps.mean())
    report = {"city": city, "train_step_mean_m": train_mean, "tests": {}}
    failures = []

    for route_name in ("test_01", "test_02"):
        route_root = prepared / route_name
        points = _manifest(route_root / "manifest.csv")
        route = _waypoints(route_root / "waypoints.json")
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cross = _cross_track(points, route)
        step_mean = float(steps.mean())
        step_p95 = float(np.percentile(steps, 95))
        cross_p95 = float(np.percentile(cross, 95))
        ratio = step_mean / max(train_mean, 1e-9)
        report["tests"][route_name] = {
            "frames": int(len(points)),
            "step_mean_m": step_mean,
            "step_p95_m": step_p95,
            "train_test_cadence_ratio": ratio,
            "cross_track_p95_m": cross_p95,
        }
        if not 0.60 <= ratio <= 1.50:
            failures.append(
                f"{route_name} cadence ratio {ratio:.3f} outside [0.60, 1.50]"
            )
        if cross_p95 > 12.0:
            failures.append(
                f"{route_name} cross-track P95 {cross_p95:.3f} m exceeds 12 m"
            )

    if failures:
        raise RuntimeError(f"{city}: invalid temporal route: " + "; ".join(failures))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True)
    parser.add_argument(
        "--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"]
    )
    args = parser.parse_args()
    root = Path(args.suite_root).resolve()
    reports = [audit_city(root, city) for city in args.cities]
    print("[TEMPORAL ROUTE AUDIT PASS]", json.dumps(reports, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
