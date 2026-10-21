#!/usr/bin/env python3
"""Export Bearing-UAV-style metrics from existing v39 frame-level results.

Headline paper metrics:
  R@1* ↑, LSR@15 ↑, HSR@15 ↑, MLE ↓, MHE ↓

R@1* uses Bearing-UAV's official same-quadrant/sign rule, but is derived from
this tracker's continuous final XY instead of a dedicated four-adjacent-RST
retrieval head. HSR/MHE use the estimated absolute heading already stored by
the current v39 evaluation runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import bearing_prepare as bearing


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_result_csv(output_dir: Path, route: str, summary: dict) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.exists():
        return p
    q = output_dir / p.name
    if q.exists():
        return q
    matches = sorted(output_dir.glob(f"{route}_*_frames.csv"))
    if not matches:
        matches = sorted(output_dir.glob("*_frames.csv"))
    if len(matches) == 1:
        return matches[0]
    explicit = [x for x in matches if route in x.name]
    if len(explicit) == 1:
        return explicit[0]
    raise RuntimeError(f"Cannot resolve unique result CSV for {route}: {matches}")


def _same_quadrant_recall(result_rows, manifest_rows, city_rows, origin_x_m, origin_y_m):
    """Bearing-UAV R@1/PHR same-sign criterion applied to continuous final XY."""
    if len(result_rows) != len(manifest_rows):
        raise RuntimeError("result/manifest length mismatch")
    good = 0
    for pred, man in zip(result_rows, manifest_rows):
        source_index = int(man["source_index"])
        meta = city_rows.iloc[source_index]
        center_x_px = float(meta["block_x"]) * bearing.PATCH_SIZE + bearing.PATCH_SIZE
        center_y_px = float(meta["block_y"]) * bearing.PATCH_SIZE + bearing.PATCH_SIZE
        final_abs_x_px = (float(pred["final_x"]) + origin_x_m) / bearing.MPP
        final_abs_y_px = (float(pred["final_y"]) + origin_y_m) / bearing.MPP
        pred_rel = np.asarray([
            (final_abs_x_px - center_x_px) / bearing.OFFICIAL_OFFSET_SCALE_PX,
            (final_abs_y_px - center_y_px) / bearing.OFFICIAL_OFFSET_SCALE_PX,
        ])
        gt_rel = np.asarray([float(meta["x_norm"]), float(meta["y_norm"])])
        good += int(np.array_equal(np.sign(pred_rel), np.sign(gt_rel)))
    return 100.0 * good / max(len(result_rows), 1)


def _navigation_keys(summaries):
    if "test_01" in summaries and "test_02" in summaries:
        return [("test_01", "test_01"), ("test_02", "test_02")]
    if "nav50" in summaries and "nav51" in summaries:
        return [("test_01", "nav50"), ("test_02", "nav51")]
    raise KeyError(f"Expected test_01/test_02 or nav50/nav51, got {sorted(summaries)}")


def compute(prepared_root: Path, output_dir: Path) -> dict:
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    summaries = json.loads((output_dir / "bearing_v39_summary.json").read_text(encoding="utf-8"))
    city = str(exp["city"])
    city_rows = bearing._city_rows(pd.read_csv(exp["metadata_csv"]), city)
    train = _read_csv(prepared_root / "routes" / "train_01" / "manifest.csv")
    origin_x_m, origin_y_m = float(train[0]["x_m"]), float(train[0]["y_m"])

    route_metrics = {}
    flat_rows = []
    all_errors, all_headings = [], []
    recall_good_equivalent = 0.0
    total_frames = 0

    for route, stored_key in _navigation_keys(summaries):
        result = _read_csv(_find_result_csv(output_dir, route, summaries[stored_key]))
        manifest = _read_csv(prepared_root / "routes" / route / "manifest.csv")
        if len(result) != len(manifest):
            raise RuntimeError(f"{route}: result/manifest length mismatch")
        if result and "heading_error_deg" not in result[0]:
            raise RuntimeError(
                "Result CSV has no heading_error_deg. Re-run evaluation only with the current v39 runtime; retraining is not required."
            )

        errors = np.asarray([
            math.hypot(float(r["final_x"]) - float(r["gt_x"]), float(r["final_y"]) - float(r["gt_y"]))
            for r in result
        ], dtype=np.float64)
        headings = np.asarray([abs(float(r["heading_error_deg"])) for r in result], dtype=np.float64)
        recall = _same_quadrant_recall(result, manifest, city_rows, origin_x_m, origin_y_m)
        row = {
            "frames": len(result),
            "R@1*_pct": float(recall),
            "LSR@15_pct": 100.0 * float(np.mean(errors <= 15.0)),
            "HSR@15_pct": 100.0 * float(np.mean(headings <= 15.0)),
            "MLE_m": float(errors.mean()),
            "MHE_deg": float(headings.mean()),
        }
        route_metrics[route] = row
        flat_rows.append({"city": city, "route": route, **row})
        all_errors.extend(errors.tolist())
        all_headings.extend(headings.tolist())
        total_frames += len(result)
        recall_good_equivalent += recall * len(result) / 100.0

    all_errors = np.asarray(all_errors, dtype=np.float64)
    all_headings = np.asarray(all_headings, dtype=np.float64)
    aggregate = {
        "frames": total_frames,
        "R@1*_pct": 100.0 * recall_good_equivalent / max(total_frames, 1),
        "LSR@15_pct": 100.0 * float(np.mean(all_errors <= 15.0)),
        "HSR@15_pct": 100.0 * float(np.mean(all_headings <= 15.0)),
        "MLE_m": float(all_errors.mean()),
        "MHE_deg": float(all_headings.mean()),
    }
    payload = {
        "city": city,
        "metric_set": ["R@1*_pct", "LSR@15_pct", "HSR@15_pct", "MLE_m", "MHE_deg"],
        "R@1*_note": "Bearing-UAV same-quadrant/sign criterion derived from continuous final XY.",
        "routes": route_metrics,
        "aggregate_two_routes": aggregate,
    }
    (output_dir / "bearing_paper_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "bearing_paper_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print("[BEARING-METRICS]", city, json.dumps(aggregate, indent=2), flush=True)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    compute(Path(a.prepared_root).resolve(), Path(a.output_dir).resolve())


if __name__ == "__main__":
    main()
