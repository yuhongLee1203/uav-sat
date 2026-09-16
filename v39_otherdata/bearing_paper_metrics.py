#!/usr/bin/env python3
"""Export paper-ready Bearing-UAV comparison metrics.

Same mathematical metric definitions as Bearing-UAV:
  MLE, MedLE, LSR@15

IMPORTANT protocol note: this v39 experiment is controlled-local-prior temporal
refinement on pseudo-flight sequences, whereas Bearing-UAV's paper evaluates its
four-adjacent-RST pose-regression protocol.  Therefore equal metric names do NOT
by themselves imply a fully apples-to-apples experimental protocol.

Recall@1 is additionally derived using the SAME four-adjacent-RST quadrant
criterion used by the official Bearing-UAV test code, but from our continuous
final position.  It is exported with an explicit ``derived`` label.

HSR/MHE/MedHE are NOT silently fabricated: v39's temporal heading represents
route/motion heading, whereas Bearing-UAV supervises UAV camera heading.
SR@20/SPL/NE are also left N/A because the current experiment is offline
localization replay, not Bearing-Naver closed-loop flight.
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

OFFICIAL_UAV_REFERENCES = {
    "Bearing-UAV Mobile-V3S": {
        "Recall@1_pct": 79.76,
        "LSR@15_pct": 81.20,
        "HSR@15_pct": 64.94,
        "MLE_m": 10.34,
        "MedLE_m": 8.72,
        "MHE_deg": 19.53,
        "MedHE_deg": 10.14,
        "source": "CVPR 2026 supplementary Table 2, UAV view",
    },
    "Bearing-UAV VGG-16": {
        "Recall@1_pct": 83.17,
        "LSR@15_pct": 89.36,
        "HSR@15_pct": 77.21,
        "MLE_m": 8.61,
        "MedLE_m": 7.30,
        "MHE_deg": 12.90,
        "MedHE_deg": 7.20,
        "source": "CVPR 2026 supplementary Table 2, UAV view",
    },
}


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
        raise FileNotFoundError(f"No result CSV for {route}")
    return matches[-1]


def _same_quadrant_recall(
    result_rows, manifest_rows, city_rows: pd.DataFrame,
    origin_x_m: float, origin_y_m: float,
) -> float:
    if len(result_rows) != len(manifest_rows):
        raise RuntimeError("result/manifest length mismatch")
    good = 0
    total = 0
    for pred, man in zip(result_rows, manifest_rows):
        source_index = int(man["source_index"])
        if source_index < 0 or source_index >= len(city_rows):
            raise RuntimeError(f"source_index out of range: {source_index}")
        meta = city_rows.iloc[source_index]

        # Official Bearing global coordinate conversion:
        # block*256 + 256 + normalized_offset*256.
        center_x_px = float(meta["block_x"]) * bearing.PATCH_SIZE + bearing.PATCH_SIZE
        center_y_px = float(meta["block_y"]) * bearing.PATCH_SIZE + bearing.PATCH_SIZE
        final_abs_x_px = (float(pred["final_x"]) + origin_x_m) / bearing.MPP
        final_abs_y_px = (float(pred["final_y"]) + origin_y_m) / bearing.MPP
        pred_rel = np.asarray([
            final_abs_x_px - center_x_px,
            final_abs_y_px - center_y_px,
        ], dtype=np.float64)
        gt_rel = np.asarray([
            float(meta["x_norm"]), float(meta["y_norm"])
        ], dtype=np.float64)
        # Scale does not affect sign. Match official RECALL_AT_K_PHR sign rule.
        if np.array_equal(np.sign(pred_rel), np.sign(gt_rel)):
            good += 1
        total += 1
    return 100.0 * good / max(total, 1)


def compute(prepared_root: Path, output_dir: Path) -> dict:
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    summaries = json.loads((output_dir / "bearing_v39_summary.json").read_text(encoding="utf-8"))
    city = str(exp["city"])
    metadata = pd.read_csv(exp["metadata_csv"])
    city_rows = bearing._city_rows(metadata, city)
    train = _read_csv(prepared_root / "routes" / "train_01" / "manifest.csv")
    if not train:
        raise RuntimeError("empty train_01 manifest")
    origin_x_m = float(train[0]["x_m"])
    origin_y_m = float(train[0]["y_m"])

    route_metrics = {}
    flat_rows = []
    all_errors = []
    total_frames = 0
    recall_good_equivalent = 0.0

    for route in ("test_01", "test_02"):
        s = summaries[route]
        result_path = _find_result_csv(output_dir, route, s)
        result = _read_csv(result_path)
        manifest = _read_csv(prepared_root / "routes" / route / "manifest.csv")
        if len(result) != len(manifest):
            raise RuntimeError(f"{route}: result/manifest length mismatch")
        errors = np.asarray([
            math.hypot(
                float(r["final_x"]) - float(r["gt_x"]),
                float(r["final_y"]) - float(r["gt_y"]),
            ) for r in result
        ], dtype=np.float64)
        recall = _same_quadrant_recall(
            result, manifest, city_rows, origin_x_m, origin_y_m
        )
        metrics = {
            "frames": len(result),
            "Recall@1_derived_same_quadrant_pct": float(recall),
            "MLE_m": float(errors.mean()),
            "MedLE_m": float(np.median(errors)),
            "LSR@5_pct": float(100.0 * np.mean(errors <= 5.0)),
            "LSR@10_pct": float(100.0 * np.mean(errors <= 10.0)),
            "LSR@15_pct": float(100.0 * np.mean(errors <= 15.0)),
            "LSR@20_pct": float(100.0 * np.mean(errors <= 20.0)),
            "P90_m": float(np.percentile(errors, 90)),
            "P95_m": float(np.percentile(errors, 95)),
            "P99_m": float(np.percentile(errors, 99)),
            "HSR@15_pct": None,
            "MHE_deg": None,
            "MedHE_deg": None,
            "SR@20_pct": None,
            "SPL_pct": None,
            "NE_m": None,
        }
        route_metrics[route] = metrics
        flat_rows.append({"city": city, "route": route, **metrics})
        all_errors.extend(errors.tolist())
        total_frames += len(result)
        recall_good_equivalent += recall * len(result) / 100.0

    all_errors_arr = np.asarray(all_errors, dtype=np.float64)
    aggregate = {
        "frames": int(total_frames),
        "Recall@1_derived_same_quadrant_pct": float(100.0 * recall_good_equivalent / max(total_frames, 1)),
        "MLE_m": float(all_errors_arr.mean()),
        "MedLE_m": float(np.median(all_errors_arr)),
        "LSR@5_pct": float(100.0 * np.mean(all_errors_arr <= 5.0)),
        "LSR@10_pct": float(100.0 * np.mean(all_errors_arr <= 10.0)),
        "LSR@15_pct": float(100.0 * np.mean(all_errors_arr <= 15.0)),
        "LSR@20_pct": float(100.0 * np.mean(all_errors_arr <= 20.0)),
        "P90_m": float(np.percentile(all_errors_arr, 90)),
        "P95_m": float(np.percentile(all_errors_arr, 95)),
        "P99_m": float(np.percentile(all_errors_arr, 99)),
        "HSR@15_pct": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": None,
        "SPL_pct": None,
        "NE_m": None,
    }

    payload = {
        "city": city,
        "protocol": "v39 controlled-local-prior temporal refinement on Bearing pseudo-flight sequences",
        "bearing_uav_reference_protocol": "four-adjacent-RST pose regression plus separate closed-loop Bearing-Naver navigation",
        "same_metric_definition_but_protocol_requires_footnote": ["MLE_m", "MedLE_m", "LSR@15_pct"],
        "protocol_footnote": (
            "Metric formulas match Bearing-UAV, but the evaluation protocol differs: this v39 experiment uses a "
            "controlled local prior and temporal pseudo-flight refinement, so the values must not be described as "
            "a fully apples-to-apples replacement for Bearing-UAV's four-RST pose-regression benchmark."
        ),
        "derived_same_decision_criterion": {
            "Recall@1_derived_same_quadrant_pct": (
                "Uses the official Bearing-UAV four-adjacent-RST sign/quadrant criterion, "
                "but is derived from this method's continuous final position rather than an RST retrieval head."
            )
        },
        "not_available_under_current_protocol": {
            "HSR@15_pct/MHE_deg/MedHE_deg": (
                "Bearing-UAV evaluates supervised camera heading; v39 temporal heading is route/motion heading."
            ),
            "SR@20_pct/SPL_pct/NE_m": (
                "Bearing-UAV computes these in closed-loop navigation; this experiment is offline localization replay."
            ),
        },
        "routes": route_metrics,
        "aggregate_two_routes": aggregate,
        "official_uav_reference_rows": OFFICIAL_UAV_REFERENCES,
    }
    (output_dir / "bearing_paper_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "bearing_paper_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print("[PAPER-METRICS]", city, json.dumps(aggregate, indent=2), flush=True)
    print("[PAPER-METRICS] same metric definitions: MLE, MedLE, LSR@15 (protocol footnote REQUIRED)", flush=True)
    print("[PAPER-METRICS] Recall@1*: derived with official same-quadrant criterion", flush=True)
    print("[PAPER-METRICS] heading/navigation fields intentionally N/A (different task/protocol)", flush=True)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    compute(Path(a.prepared_root).resolve(), Path(a.output_dir).resolve())


if __name__ == "__main__":
    main()
