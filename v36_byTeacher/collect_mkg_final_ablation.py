"""Collect one MKG ablation suite into thesis-friendly CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config

METRICS = (
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
    "VisualMLE_m",
    "Latency_ms_per_frame",
    "FPS",
    "PeakCUDAAllocated_MB",
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
                for metric in METRICS:
                    if metric in s:
                        row[metric] = s[metric]
                route_rows.append(row)
                current.append(s)

            if len(current) == 2:
                avg = dict(meta)
                avg["jitter_m"] = float(jitter_key)
                avg["route"] = "B+C average"
                for metric in METRICS:
                    values = [float(s[metric]) for s in current if metric in s]
                    if values:
                        avg[metric] = float(np.mean(values))
                average_rows.append(avg)

    average_rows.sort(
        key=lambda r: (
            float(r.get("jitter_m", 0.0)),
            float(r.get("MLE_m", float("inf"))),
        )
    )
    write_csv(root / "all_route_results.csv", route_rows)
    write_csv(root / "bc_average_results.csv", average_rows)

    clean = [r for r in average_rows if float(r.get("jitter_m", 0.0)) == 0.0]
    ranking = sorted(clean, key=lambda r: float(r.get("MLE_m", float("inf"))))
    write_csv(root / "jitter0_ranking.csv", ranking)

    seed_rows = [
        r for r in clean
        if r["run_id"] in ("baseline_mkg_6x6", "seed_123", "seed_456")
    ]
    if seed_rows:
        seed_summary = {
            "suite_tag": args.suite_tag,
            "runs": [r["run_id"] for r in seed_rows],
            "MLE_mean_m": float(np.mean([r["MLE_m"] for r in seed_rows])),
            "MLE_std_m": float(np.std([r["MLE_m"] for r in seed_rows])),
            "P90_mean_m": float(np.mean([r["P90_m"] for r in seed_rows])),
            "P90_std_m": float(np.std([r["P90_m"] for r in seed_rows])),
        }
        (root / "seed_stability.json").write_text(
            json.dumps(seed_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    manifest = {
        "suite_tag": args.suite_tag,
        "completed_run_count": len(summaries),
        "route_csv": str(root / "all_route_results.csv"),
        "average_csv": str(root / "bc_average_results.csv"),
        "ranking_csv": str(root / "jitter0_ranking.csv"),
    }
    (root / "collection_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
