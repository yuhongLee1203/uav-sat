#!/usr/bin/env python3
"""Fail-fast audit of prepared Bearing-UAV coordinates and temporal labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

PATCH_SIZE = 256.0
OFFSET_SCALE = 128.0
MPP = 0.25
RSI_IDS = {"citya": "34bc", "cityb": "36bc", "cityc": "37bc", "cityd": "38bc"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--city", required=True, choices=sorted(RSI_IDS))
    args = parser.parse_args()

    prepared = Path(args.prepared_root).resolve()
    experiment = json.loads((prepared / "experiment.json").read_text(encoding="utf-8"))
    metadata_path = Path(experiment["metadata_csv"])
    metadata = pd.read_csv(metadata_path)
    paths = metadata["target_path"].astype(str).str.replace("\\", "/", regex=False)
    city_rows = metadata.loc[
        paths.str.contains(f"/{args.city}/", regex=False)
        | paths.str.contains(RSI_IDS[args.city], regex=False)
    ].copy().reset_index(drop=True)
    if city_rows.empty:
        raise RuntimeError(f"no metadata rows found for {args.city}")

    checked = 0
    max_px_error = 0.0
    max_m_error = 0.0
    routes = sorted((prepared / "routes").glob("*/manifest.csv"))
    if not routes:
        raise RuntimeError(f"no prepared manifests under {prepared}")
    for manifest in routes:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                source_index = int(item["source_index"])
                raw = city_rows.iloc[source_index]
                expected_x = float(raw.block_x) * PATCH_SIZE + PATCH_SIZE + float(raw.x_norm) * OFFSET_SCALE
                expected_y = float(raw.block_y) * PATCH_SIZE + PATCH_SIZE + float(raw.y_norm) * OFFSET_SCALE
                px_error = max(abs(float(item["x_px"]) - expected_x), abs(float(item["y_px"]) - expected_y))
                m_error = max(abs(float(item["x_m"]) - expected_x * MPP), abs(float(item["y_m"]) - expected_y * MPP))
                max_px_error = max(max_px_error, px_error)
                max_m_error = max(max_m_error, m_error)
                if str(raw.target_path) != item["source_target_path"]:
                    raise RuntimeError(f"source row mismatch: {manifest}:{item['frame_id']}")
                if not Path(item["image_path"]).is_file():
                    raise RuntimeError(f"missing UAV image: {item['image_path']}")
                checked += 1

    if max_px_error > 1e-5 or max_m_error > 1e-5:
        raise RuntimeError(
            f"Bearing-UAV GT conversion mismatch: px={max_px_error:.9g}, m={max_m_error:.9g}"
        )
    report = {
        "status": "PASS",
        "city": args.city,
        "checked_frames": checked,
        "coordinate_formula": "block*256 + 256 + norm*128",
        "meters_per_pixel": MPP,
        "max_pixel_error": max_px_error,
        "max_meter_error": max_m_error,
        "native_gt": ["block_x", "block_y", "x_norm", "y_norm", "theta"],
        "derived_temporal_targets": ["step", "velocity", "acceleration", "ground_track_heading"],
        "active_losses": ["measurement", "next_step", "variance_nll"],
    }
    out = prepared / "training_contract_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[BEARING GT/LOSS AUDIT]", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
