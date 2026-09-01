"""Collect one MKG ablation suite into purpose-specific thesis tables.

The collector deliberately does NOT create one giant mixed ranking table.
Each CSV answers one experimental question. Full-pipeline latency/FPS are not
reported as decoder timing; decoder aggregation timing is handled separately by
ms_vs_weighted_aggregation_timing.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config

ACCURACY_METRICS = (
    "MLE_m",
    "MedLE_m",
    "P90_m",
    "P95_m",
    "P99_m",
    "CVaR90_m",
    "LSR@5_pct",
    "LSR@10_pct",
    "LSR@15_pct",
    "LSR@20_pct",
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_row(row, setting=None):
    out = {}
    if setting is not None:
        out["setting"] = setting
    for key in (
        "run_id",
        "pipeline",
        "grid_size",
        "candidate_count",
        "decoder",
        "gru_ablation",
        "bandwidth_m",
        "tau",
        "seed",
        "jitter_m",
    ):
        if key in row:
            out[key] = row[key]
    for metric in ACCURACY_METRICS:
        if metric in row:
            out[metric] = row[metric]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-tag", required=True)
    args = parser.parse_args()

    root = (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "mkg_final_ablation_suite"
        / args.suite_tag
    )
    if not root.exists():
        raise FileNotFoundError(root)

    route_rows = []
    average_rows = []
    summaries = sorted(root.glob("*/summary.json"))
    if not summaries:
        raise RuntimeError("no completed run summaries found in %s" % root)

    for summary_path in summaries:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = {
            "run_id": payload["run_id"],
            "pipeline": payload["pipeline"],
            "grid_size": payload["grid_size"],
            "candidate_count": payload["candidate_count"],
            "decoder": payload["decoder"],
            "gru_ablation": payload["gru_ablation"],
            "bandwidth_m": payload["meanshift_bandwidth_m"],
            "tau": payload["meanshift_score_tau"],
            "seed": payload["seed"],
            "training_from_scratch": payload["training_from_scratch"],
        }

        for jitter_key, route_map in payload["results"].items():
            current = []
            for route_name in ("route_B", "route_C"):
                if route_name not in route_map:
                    continue
                s = route_map[route_name]
                row = dict(meta)
                row["jitter_m"] = float(jitter_key)
                row["route"] = route_name
                for metric in ACCURACY_METRICS:
                    if metric in s:
                        row[metric] = s[metric]
                route_rows.append(row)
                current.append(s)

            if len(current) == 2:
                avg = dict(meta)
                avg["jitter_m"] = float(jitter_key)
                avg["route"] = "B+C average"
                for metric in ACCURACY_METRICS:
                    values = [float(s[metric]) for s in current if metric in s]
                    if values:
                        avg[metric] = float(np.mean(values))
                average_rows.append(avg)

    write_csv(root / "all_route_accuracy_results.csv", route_rows)
    write_csv(root / "bc_average_accuracy_results.csv", average_rows)

    clean = {
        r["run_id"]: r
        for r in average_rows
        if float(r.get("jitter_m", 0.0)) == 0.0
    }

    # 01: Search-window size. Only grid size changes.
    window_spec = [
        ("window_4x4", "4x4 (16 patches)"),
        ("window_5x5", "5x5 (25 patches)"),
        ("baseline_mkg_6x6", "6x6 (36 patches)"),
        ("window_7x7", "7x7 (49 patches)"),
        ("window_8x8", "8x8 (64 patches)"),
    ]
    write_csv(
        root / "table_01_window_size_ablation.csv",
        [clean_row(clean[k], label) for k, label in window_spec if k in clean],
    )

    # 02: Component ablation. M/K/G presence only.
    component_spec = [
        ("component_m", "M"),
        ("component_mk", "M+K"),
        ("component_mg", "M+G"),
        ("baseline_mkg_6x6", "M+K+G (MKG)"),
    ]
    write_csv(
        root / "table_02_component_ablation.csv",
        [clean_row(clean[k], label) for k, label in component_spec if k in clean],
    )

    # 03: GRU input-branch ablation.
    gru_spec = [
        ("baseline_mkg_6x6", "Full GRU"),
        ("gru_no_xy", "w/o stage XY"),
        ("gru_no_variance", "w/o visual variance"),
        ("gru_no_temporal_mean", "w/o temporal mean"),
        ("gru_no_first_difference", "w/o first difference"),
    ]
    write_csv(
        root / "table_03_gru_input_ablation.csv",
        [clean_row(clean[k], label) for k, label in gru_spec if k in clean],
    )

    # 04: MeanShift bandwidth sensitivity.
    bandwidth_spec = [
        ("bandwidth_4m", "4 m"),
        ("baseline_mkg_6x6", "8 m"),
        ("bandwidth_12m", "12 m"),
    ]
    write_csv(
        root / "table_04_meanshift_bandwidth.csv",
        [clean_row(clean[k], label) for k, label in bandwidth_spec if k in clean],
    )

    # 05: Similarity/SoftMS temperature sensitivity.
    tau_spec = [
        ("tau_0p20", "tau=0.20"),
        ("baseline_mkg_6x6", "tau=0.30"),
        ("tau_0p50", "tau=0.50"),
    ]
    write_csv(
        root / "table_05_score_temperature.csv",
        [clean_row(clean[k], label) for k, label in tau_spec if k in clean],
    )

    # 06: Decoder ACCURACY only. Top-1 is intentionally excluded.
    decoder_spec = [
        ("decoder_weighted", "Weighted Centroid (36 patches)"),
        ("baseline_mkg_6x6", "SoftMS"),
    ]
    write_csv(
        root / "table_06_decoder_accuracy.csv",
        [clean_row(clean[k], label) for k, label in decoder_spec if k in clean],
    )

    # 07: Robustness to known search-center error.
    jitter_rows = []
    for row in average_rows:
        if row["run_id"] == "baseline_mkg_6x6":
            jitter_rows.append(clean_row(row, "%g m" % float(row["jitter_m"])))
    jitter_rows.sort(key=lambda r: float(r["jitter_m"]))
    write_csv(root / "table_07_reference_jitter_robustness.csv", jitter_rows)

    # 08: Seed stability.
    seed_spec = [
        ("baseline_mkg_6x6", "seed 2026"),
        ("seed_123", "seed 123"),
        ("seed_456", "seed 456"),
    ]
    seed_rows = [clean_row(clean[k], label) for k, label in seed_spec if k in clean]
    write_csv(root / "table_08_seed_stability.csv", seed_rows)
    if seed_rows:
        seed_summary = {
            "suite_tag": args.suite_tag,
            "MLE_mean_m": float(np.mean([r["MLE_m"] for r in seed_rows])),
            "MLE_std_m": float(np.std([r["MLE_m"] for r in seed_rows])),
            "P90_mean_m": float(np.mean([r["P90_m"] for r in seed_rows])),
            "P90_std_m": float(np.std([r["P90_m"] for r in seed_rows])),
        }
        (root / "table_08_seed_stability_summary.json").write_text(
            json.dumps(seed_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 09: Main six-order architecture comparison is intentionally LAST.
    arch_root = Path(config.BACKBONE_OUTPUT_DIR) / "six_architecture_gt_center_center6x6"
    arch_rows = []
    for arch in ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG"):
        path = arch_root / arch.lower() / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rb = payload["results"]["route_B"]
        rc = payload["results"]["route_C"]
        arch_rows.append(
            {
                "architecture": arch,
                "RouteB_MLE_m": float(rb["MLE_m"]),
                "RouteC_MLE_m": float(rc["MLE_m"]),
                "BC_Avg_MLE_m": float((rb["MLE_m"] + rc["MLE_m"]) / 2.0),
                "RouteB_P90_m": float(rb["P90_m"]),
                "RouteC_P90_m": float(rc["P90_m"]),
                "BC_Avg_P90_m": float((rb["P90_m"] + rc["P90_m"]) / 2.0),
                "RouteB_LSR15_pct": float(rb["LSR@15_pct"]),
                "RouteC_LSR15_pct": float(rc["LSR@15_pct"]),
                "BC_Avg_LSR15_pct": float((rb["LSR@15_pct"] + rc["LSR@15_pct"]) / 2.0),
            }
        )
    write_csv(root / "table_09_main_architecture_comparison_LAST.csv", arch_rows)

    manifest = {
        "suite_tag": args.suite_tag,
        "completed_run_count": len(summaries),
        "tables": [
            "table_01_window_size_ablation.csv",
            "table_02_component_ablation.csv",
            "table_03_gru_input_ablation.csv",
            "table_04_meanshift_bandwidth.csv",
            "table_05_score_temperature.csv",
            "table_06_decoder_accuracy.csv",
            "table_07_reference_jitter_robustness.csv",
            "table_08_seed_stability.csv",
            "table_09_main_architecture_comparison_LAST.csv",
        ],
        "decoder_timing": "decoder_aggregation_timing.json (generated separately; aggregation only)",
        "note": "No full-pipeline FPS is used as the Weighted-vs-SoftMS timing comparison.",
    }
    (root / "collection_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
