"""Collect unified experiment outputs into separate thesis-purpose tables.

No mixed all-in-one ranking is produced. Main architecture comparison is always
written last.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import config

ROOT = Path(config.BACKBONE_OUTPUT_DIR) / "unified_fixed8m_v1"
MAIN = ROOT / "main_architectures"
ABL = ROOT / "ablations"
TABLES = ROOT / "tables"

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
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def avg_routes(route_map):
    routes = [route_map.get("route_B"), route_map.get("route_C")]
    routes = [x for x in routes if x]
    if len(routes) != 2:
        raise RuntimeError("B/C results missing")
    row = {}
    for metric in METRICS:
        vals = [float(x[metric]) for x in routes if metric in x]
        if vals:
            row[metric] = float(np.mean(vals))
    for extra in (
        "VisualMLE_m",
        "BaseVisualMS_MLE_m",
        "CandidateCaptureRate_pct",
        "FormalCandidateCaptureRate_pct",
        "ProvisionalNextMLE_m",
        "FinalMS_MLE_m",
        "FinalNextCandidateCaptureRate_pct",
        "FormalCurrentCandidateCaptureRate_pct",
    ):
        vals = [float(x[extra]) for x in routes if x.get(extra) is not None]
        if vals:
            row[extra] = float(np.mean(vals))
    return row


def main_avg(arch):
    payload = read_json(MAIN / arch.lower() / "summary.json")
    row = avg_routes(payload["results"])
    row["Architecture"] = arch
    return row


def ablation_avg(run_id):
    payload = read_json(ABL / run_id / "summary.json")
    row = avg_routes(payload["results"])
    changed = payload["changed_factor"]
    row.update(
        {
            "Run": run_id,
            "Pipeline": changed["pipeline"],
            "GridSize": changed["grid_size"],
            "CandidateCount": int(changed["grid_size"]) ** 2,
            "Decoder": changed["decoder"],
            "GRUAblation": changed["gru_ablation"],
            "Bandwidth_m": changed["bandwidth_m"],
            "Tau": changed["tau"],
            "ModelSeed": changed["model_seed"],
            "Jitter_m": payload["formal_jitter_m"],
            "TrainCapture_pct": payload["train_capture"][
                "CandidateCaptureRate_pct"
            ],
        }
    )
    return row


def selected(row, setting):
    keys = [
        "MLE_m", "MedLE_m", "P90_m", "P95_m", "P99_m", "CVaR90_m",
        "LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct",
    ]
    out = {"Setting": setting}
    for key in keys:
        if key in row:
            out[key] = row[key]
    for key in (
        "CandidateCaptureRate_pct",
        "TrainCapture_pct",
        "CandidateCount",
        "ProvisionalNextMLE_m",
        "FinalNextCandidateCaptureRate_pct",
    ):
        if key in row:
            out[key] = row[key]
    if "CandidateCaptureRate_pct" in out:
        out["CaptureValid_ge95"] = bool(
            float(out["CandidateCaptureRate_pct"]) >= 95.0
        )
    return out


def baseline_row():
    row = main_avg("MKG")
    row["CandidateCount"] = 36
    row["CandidateCaptureRate_pct"] = float(
        np.mean(
            [
                read_json(MAIN / "mkg" / "summary.json")["results"][r][
                    "FormalCandidateCaptureRate_pct"
                ]
                for r in ("route_B", "route_C")
            ]
        )
    )
    return row


def build_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    base = baseline_row()

    rows = []
    for size in (4, 5):
        r = ablation_avg("window_%dx%d" % (size, size))
        rows.append(selected(r, "%dx%d" % (size, size)))
    rows.append(selected(base, "6x6"))
    for size in (7, 8):
        r = ablation_avg("window_%dx%d" % (size, size))
        rows.append(selected(r, "%dx%d" % (size, size)))
    write_csv(TABLES / "table_01_window_size_ablation.csv", rows)

    rows = [
        selected(ablation_avg("component_m"), "M"),
        selected(ablation_avg("component_mk"), "M+K"),
        selected(ablation_avg("component_mg"), "M+G"),
        selected(base, "M+K+G"),
    ]
    write_csv(TABLES / "table_02_component_ablation.csv", rows)

    rows = [selected(base, "Full GRU")]
    for run_id, label in (
        ("gru_no_xy", "w/o Stage XY"),
        ("gru_no_variance", "w/o Variance"),
        ("gru_no_temporal_mean", "w/o Temporal Mean"),
        ("gru_no_first_difference", "w/o First Difference"),
    ):
        rows.append(selected(ablation_avg(run_id), label))
    write_csv(TABLES / "table_03_gru_input_ablation.csv", rows)

    rows = [
        selected(ablation_avg("bandwidth_4m"), "4 m"),
        selected(base, "8 m"),
        selected(ablation_avg("bandwidth_12m"), "12 m"),
    ]
    write_csv(TABLES / "table_04_meanshift_bandwidth.csv", rows)

    rows = [
        selected(ablation_avg("tau_0p20"), "0.20"),
        selected(base, "0.30"),
        selected(ablation_avg("tau_0p50"), "0.50"),
    ]
    write_csv(TABLES / "table_05_score_temperature.csv", rows)

    rows = [
        selected(ablation_avg("decoder_weighted"), "Weighted Centroid"),
        selected(base, "SoftMS"),
    ]
    write_csv(TABLES / "table_06_decoder_accuracy.csv", rows)

    timing_path = ROOT / "decoder_timing" / "decoder_aggregation_timing.json"
    timing = read_json(timing_path)
    rows = [
        {
            "Setting": "Weighted Centroid",
            "AggregationInput": "%d patch positions + weights"
            % timing["patch_count_N"],
            "Aggregation_ms": timing["weighted_centroid_aggregation_ms"],
        },
        {
            "Setting": "SoftMS",
            "AggregationInput": "converged active modes (mean K=%.3f)"
            % timing["mean_active_ms_modes_K"],
            "Aggregation_ms": timing["softms_final_mode_aggregation_ms"],
        },
    ]
    write_csv(TABLES / "table_07_decoder_aggregation_time_ms.csv", rows)

    robust = read_json(ABL / "robustness_fixed8_model" / "summary.json")
    rows = []
    for level_text, item in robust["levels"].items():
        level = float(level_text)
        cap_values = [
            float(item["Capture"][r]["CandidateCaptureRate_pct"])
            for r in ("route_B", "route_C")
        ]
        row = {
            "PriorError_m": level,
            "CandidateCaptureRate_pct": float(np.mean(cap_values)),
            "ValidForLocalizationComparison": not bool(item["Skipped"]),
        }
        if not item["Skipped"]:
            avg = avg_routes(item["Results"])
            for metric in METRICS:
                if metric in avg:
                    row[metric] = avg[metric]
        rows.append(row)
    rows.sort(key=lambda x: x["PriorError_m"])
    write_csv(TABLES / "table_08_prior_error_robustness.csv", rows)

    rows = [
        selected(base, "Seed 2026"),
        selected(ablation_avg("seed_123"), "Seed 123"),
        selected(ablation_avg("seed_456"), "Seed 456"),
    ]
    write_csv(TABLES / "table_09_seed_stability.csv", rows)

    rows = []
    for arch in (
        "MKG", "MGK", "GMK", "GKM", "KGM", "KMG", "delayKG", "delayGK"
    ):
        r = main_avg(arch)
        row = {"Architecture": arch}
        for metric in METRICS:
            if metric in r:
                row[metric] = r[metric]
        if "ProvisionalNextMLE_m" in r:
            row["ProvisionalNextMLE_m"] = r["ProvisionalNextMLE_m"]
        if "FinalNextCandidateCaptureRate_pct" in r:
            row["FinalNextCandidateCaptureRate_pct"] = r[
                "FinalNextCandidateCaptureRate_pct"
            ]
        rows.append(row)
    write_csv(TABLES / "table_10_main_8_architecture_comparison_LAST.csv", rows)

    manifest = {
        "protocol": read_json(MAIN / "mkg" / "summary.json")["protocol"],
        "tables_in_order": [
            "table_01_window_size_ablation.csv",
            "table_02_component_ablation.csv",
            "table_03_gru_input_ablation.csv",
            "table_04_meanshift_bandwidth.csv",
            "table_05_score_temperature.csv",
            "table_06_decoder_accuracy.csv",
            "table_07_decoder_aggregation_time_ms.csv",
            "table_08_prior_error_robustness.csv",
            "table_09_seed_stability.csv",
            "table_10_main_8_architecture_comparison_LAST.csv",
        ],
        "mixed_all_in_one_table": False,
        "oracle_zero_m_in_formal_tables": False,
        "main_architecture_table_is_last": True,
    }
    (TABLES / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    build_tables()
