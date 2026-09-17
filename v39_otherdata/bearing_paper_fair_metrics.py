#!/usr/bin/env python3
"""Export publication-safe localization metrics for the NO-GT Bearing runner.

The metric formulas match Bearing-UAV's released evaluator:
  Recall@1 : same sign/quadrant criterion relative to the four-RST block
  LSR@15   : percentage with metric localization error <= 15 m
  MLE       : mean Euclidean localization error in metres
  MedLE     : median Euclidean localization error in metres

The full temporal v39 method does not regress the Bearing-UAV camera-heading
vector, so HSR/MHE/MedHE are intentionally N/A.  Likewise, offline route replay
is not Bearing-Naver closed-loop navigation, so SR/SPL/NE are intentionally N/A.
Never fabricate those columns from route heading or plotted trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import bearing_paper_metrics as base


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export(prepared_root: Path, output_dir: Path) -> dict:
    audit_path = output_dir / "paper_fair_inference_audit.json"
    if not audit_path.exists():
        raise RuntimeError(
            "Missing paper_fair_inference_audit.json. Refusing to label an older "
            "GT-prior run as paper-fair."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "paper_fair": True,
        "reference_protocol": "route_reference",
        "no_gt_inference": True,
        "uses_current_frame_gt_coordinate_at_inference": False,
        "uses_true_route_progress_at_inference": False,
        "uses_gt_motion_or_progress_cap_at_inference": False,
    }
    bad = {k: {"actual": audit.get(k), "required": v} for k, v in required.items() if audit.get(k) != v}
    if bad:
        raise RuntimeError("Paper-fair audit mismatch: " + json.dumps(bad, indent=2))

    # Reuse the already audited Bearing coordinate conversion and exact metric
    # formulas, then relabel the output with the stricter no-GT protocol.
    raw = base.compute(prepared_root, output_dir)

    route_rows = []
    routes = {}
    for route in ("test_01", "test_02"):
        src = raw["routes"][route]
        row = {
            "route": route,
            "frames": int(src["frames"]),
            "Recall@1_pct": float(src["Recall@1_derived_same_quadrant_pct"]),
            "LSR@15_pct": float(src["LSR@15_pct"]),
            "MLE_m": float(src["MLE_m"]),
            "MedLE_m": float(src["MedLE_m"]),
            "P90_m": float(src["P90_m"]),
            "HSR@15_pct": None,
            "MHE_deg": None,
            "MedHE_deg": None,
            "SR@20_pct": None,
            "SPL_pct": None,
            "NE_m": None,
        }
        routes[route] = dict(row)
        route_rows.append(dict(row))

    agg_src = raw["aggregate_two_routes"]
    aggregate = {
        "frames": int(agg_src["frames"]),
        "Recall@1_pct": float(agg_src["Recall@1_derived_same_quadrant_pct"]),
        "LSR@15_pct": float(agg_src["LSR@15_pct"]),
        "MLE_m": float(agg_src["MLE_m"]),
        "MedLE_m": float(agg_src["MedLE_m"]),
        "P90_m": float(agg_src["P90_m"]),
        "HSR@15_pct": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": None,
        "SPL_pct": None,
        "NE_m": None,
    }

    payload = {
        "paper_fair": True,
        "city": raw["city"],
        "evaluation_scope": "official Bearing-UAV navigation-route selected frames",
        "inference_protocol": "v39 temporal route-reference refinement; NO current-frame GT/true-progress access",
        "known_test_time_information": audit["known_at_inference"],
        "direct_metric_definitions_matching_bearinguav": [
            "Recall@1_pct",
            "LSR@15_pct",
            "MLE_m",
            "MedLE_m",
        ],
        "important_comparison_rule": (
            "These localization metrics may be compared directly only against methods rerun on the "
            "same selected route frames. Bearing-UAV's published Table 2/8 values use its paper test "
            "split and are included below as reference-only, not as an identical-sample comparison."
        ),
        "unsupported_metrics_reason": {
            "HSR@15/MHE/MedHE": "v39 route heading is not Bearing-UAV camera heading",
            "SR@20/SPL/NE": "current experiment is offline localization replay, not closed-loop Bearing-Naver navigation",
        },
        "routes": routes,
        "aggregate_two_routes": aggregate,
        "bearing_uav_published_uav_reference_only": {
            "Recall@1_pct": 83.17,
            "LSR@15_pct": 89.36,
            "MLE_m": 8.61,
            "MedLE_m": 7.30,
            "HSR@15_pct": 77.21,
            "MHE_deg": 12.90,
            "MedHE_deg": 7.20,
            "note": "Published VGG-16 UAV-view reference; different sample scope from this route evaluation.",
        },
    }

    out_json = output_dir / "paper_fair_localization_metrics.json"
    out_csv = output_dir / "paper_fair_localization_metrics.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_csv, route_rows)

    print("[PAPER-FAIR-METRICS] PASS", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)
    print("[PAPER-FAIR-METRICS] Comparable columns: Recall@1 / LSR@15 / MLE / MedLE", flush=True)
    print("[PAPER-FAIR-METRICS] Heading/navigation columns remain N/A unless the method actually predicts/runs them", flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    export(Path(a.prepared_root).resolve(), Path(a.output_dir).resolve())


if __name__ == "__main__":
    main()
