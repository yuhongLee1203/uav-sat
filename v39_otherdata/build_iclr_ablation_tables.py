#!/usr/bin/env python3
"""Aggregate measured Bearing-UAV ablations into paper-facing tables.

Headline metrics follow Bearing-UAV vocabulary:
R@1* ↑, LSR@15 ↑, HSR@15 ↑, MLE ↓, MHE ↓.
P90/P95/P99/CVaR and jump-rate are intentionally omitted from paper tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import bearing_paper_metrics as bpm
import bearing_prepare as bearing
import heading_fusion_metrics as hfm

COMPONENTS = ["no_gru", "no_kalman", "no_ms", "full"]
TEMPORAL = ["frames1", "frames2", "full"]
WINDOWS = ["grid4", "grid5", "full", "grid7", "grid8"]
LABELS = {
    "no_gru": "w/o GRU", "no_kalman": "w/o Kalman",
    "no_ms": "w/o MeanShift", "full": "Full",
    "frames1": "1 frame", "frames2": "2 frames",
    "grid4": "4x4", "grid5": "5x5", "grid7": "7x7", "grid8": "8x8",
}


def _heading_fusion_alpha() -> float:
    value = float(os.environ.get("BEARING_HEADING_FUSION_ALPHA", "0.0"))
    if not 0.0 <= value <= 1.0:
        raise ValueError("BEARING_HEADING_FUSION_ALPHA must be in [0,1]")
    return value


def _csv_for(summary: dict, output: Path) -> Path:
    original = Path(str(summary["CSV"]))
    local = output / original.name
    if local.exists():
        return local
    if original.exists():
        return original
    matches = sorted(output.glob("*_frames.csv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve frame CSV in {output}")


def _navigation_keys(summaries: dict):
    if "nav50" in summaries and "nav51" in summaries:
        return [("test_01", "nav50"), ("test_02", "nav51")]
    if "test_01" in summaries and "test_02" in summaries:
        return [("test_01", "test_01"), ("test_02", "test_02")]
    raise KeyError(f"summary must contain test_01/test_02; got {sorted(summaries)}")


def _recall_context(prepared: Path, city: str):
    exp = json.loads((prepared / "experiment.json").read_text(encoding="utf-8"))
    city_rows = bearing._city_rows(pd.read_csv(exp["metadata_csv"]), city)
    train = bpm._read_csv(prepared / "routes" / "train_01" / "manifest.csv")
    return city_rows, float(train[0]["x_m"]), float(train[0]["y_m"])


def _read_variant(root: Path, cities: list[str], variant: str) -> dict:
    errors, headings, ms_latency = [], [], []
    recall_good, recall_total = 0.0, 0
    per_route = []
    heading_alpha = _heading_fusion_alpha()
    for city in cities:
        prepared = root / city / "prepared"
        city_rows, origin_x_m, origin_y_m = _recall_context(prepared, city)
        output = root / city / "variants" / variant
        summaries = json.loads((output / "bearing_v39_summary.json").read_text(encoding="utf-8"))
        for nav_label, stored_key in _navigation_keys(summaries):
            summary = summaries[stored_key]
            with _csv_for(summary, output).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise RuntimeError(f"empty result rows: {city}/{variant}/{nav_label}")
            if "heading_error_deg" not in rows[0]:
                raise RuntimeError(
                    f"{city}/{variant}/{nav_label} lacks heading_error_deg; rerun EVAL ONLY, not training."
                )
            route_errors = np.asarray([float(r["error_final_m"]) for r in rows], dtype=np.float64)
            route_headings = hfm.fused_heading_errors(rows, heading_alpha)
            errors.append(route_errors)
            headings.append(route_headings)
            manifest = bpm._read_csv(prepared / "routes" / nav_label / "manifest.csv")
            recall = bpm._same_quadrant_recall(rows, manifest, city_rows, origin_x_m, origin_y_m)
            recall_good += recall * len(rows) / 100.0
            recall_total += len(rows)
            per_route.append({"city": city, "route": nav_label, "frames": len(rows), "R@1*_pct": recall})
            samples = int(summary.get("MS_LatencySamples", len(rows)))
            latency = float(summary.get("MS_LatencyMean_ms", 0.0))
            if latency > 0:
                ms_latency.extend([latency] * max(samples, 1))

    values = np.concatenate(errors)
    heading_values = np.concatenate(headings)
    return {
        "variant": variant,
        "label": LABELS.get(variant, variant),
        "frames": int(values.size),
        "R@1*_pct": 100.0 * recall_good / max(recall_total, 1),
        "LSR@15_pct": 100.0 * float(np.mean(values <= 15.0)),
        "HSR@15_pct": 100.0 * float(np.mean(heading_values <= 15.0)),
        "MLE_m": float(values.mean()),
        "MHE_deg": float(heading_values.mean()),
        "MS_Latency_ms": float(np.mean(ms_latency)) if ms_latency else 0.0,
        "heading_fusion_alpha": float(heading_alpha),
        "_errors": values,
        "_routes": per_route,
    }


def _paired_bootstrap(full: np.ndarray, other: np.ndarray, seed: int = 2027) -> dict:
    if full.shape != other.shape:
        raise ValueError("paired ablation outputs are not frame-aligned")
    rng = np.random.default_rng(seed)
    n = full.size
    delta_mle, delta_lsr15 = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        delta_mle.append(float(full[idx].mean() - other[idx].mean()))
        delta_lsr15.append(100.0 * float(np.mean(full[idx] <= 15.0) - np.mean(other[idx] <= 15.0)))
    return {
        "full_minus_ablation_MLE_m": float(full.mean() - other.mean()),
        "MLE_95CI": [float(x) for x in np.quantile(delta_mle, [0.025, 0.975])],
        "full_minus_ablation_LSR@15_pct": 100.0 * float(np.mean(full <= 15.0) - np.mean(other <= 15.0)),
        "LSR@15_95CI": [float(x) for x in np.quantile(delta_lsr15, [0.025, 0.975])],
    }


def _clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _header(first: str):
    return [
        f"| {first} | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]


def _row(name: str, r: dict):
    return (
        f"| {name} | {r['R@1*_pct']:.2f}% | {r['LSR@15_pct']:.2f}% | "
        f"{r['HSR@15_pct']:.2f}% | {r['MLE_m']:.3f} | {r['MHE_deg']:.2f} |"
    )


def _markdown(rows, audit, cities):
    city_text = ", ".join(c.upper() for c in cities)
    alpha = _heading_fusion_alpha()
    lines = [
        "# Bearing-UAV 4-city ablation (measured)", "",
        f"Dataset domains: {city_text}; held-out sequences: test_01/test_02.",
        "R@1* uses Bearing-UAV's same-quadrant/sign rule on the tracker's continuous final XY.",
        f"HSR/MHE use predicted heading with shared causal state-direction fusion alpha={alpha:.2f} for every compared row.", "",
        "## Component removal", "", *_header("Variant"),
    ]
    for key in COMPONENTS:
        lines.append(_row(rows[key]["label"], rows[key]))
    lines += ["", "## Temporal input", "", *_header("Input")]
    for key in TEMPORAL:
        lines.append(_row(rows[key]["label"], rows[key]))
    lines += [
        "", "## Final MeanShift window", "",
        "| Window | Candidates | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ | MS latency (ms) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in WINDOWS:
        r = rows[key]
        grid = 6 if key == "full" else int(key[-1])
        lines.append(
            f"| {grid}x{grid} | {grid*grid} | {r['R@1*_pct']:.2f}% | {r['LSR@15_pct']:.2f}% | "
            f"{r['HSR@15_pct']:.2f}% | {r['MLE_m']:.3f} | {r['MHE_deg']:.2f} | {r['MS_Latency_ms']:.3f} |"
        )
    lines += [
        "", "## Notes", "",
        "- P90/P95/P99/CVaR and jump-rate are not included in paper-facing tables.",
        "- HSR@15 is the percentage of frames with heading error <= 15 degrees.",
        "- MHE is mean absolute heading error in degrees.",
        "- Heading fusion uses only predicted recurrent heading and causal estimator-state displacement; GT is used only to score the final heading prediction.",
        "", "## Integrity audit", "",
        f"`LOCALIZATION_TREND_CHECK={audit['LOCALIZATION_TREND_CHECK']}`", "",
        audit["claim_guidance"], "",
    ]
    return "\n".join(lines)


def _latex(rows, cities):
    city_text = ", ".join(c.upper() for c in cities)
    out = [
        "% Auto-generated from measured Bearing-UAV outputs.",
        "\\begin{table}[t]", f"\\caption{{Component ablation on Bearing-UAV {city_text}.}}",
        "\\centering", "\\small", "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Variant & R@1*$\\uparrow$ & LSR@15$\\uparrow$ & HSR@15$\\uparrow$ & MLE$\\downarrow$ & MHE$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for key in COMPONENTS:
        r = rows[key]
        out.append(f"{r['label']} & {r['R@1*_pct']:.2f} & {r['LSR@15_pct']:.2f} & {r['HSR@15_pct']:.2f} & {r['MLE_m']:.3f} & {r['MHE_deg']:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya"])
    args = p.parse_args()
    root = Path(args.suite_root).resolve()
    keys = sorted(set(COMPONENTS + TEMPORAL + WINDOWS))
    rows = {key: _read_variant(root, args.cities, key) for key in keys}
    full = rows["full"]

    component_ok = all(
        full["MLE_m"] <= rows[k]["MLE_m"] + 1e-12 and full["LSR@15_pct"] >= rows[k]["LSR@15_pct"] - 1e-12
        for k in ("no_gru", "no_kalman", "no_ms")
    )
    temporal_ok = all(
        full["MLE_m"] <= rows[k]["MLE_m"] + 1e-12 and full["LSR@15_pct"] >= rows[k]["LSR@15_pct"] - 1e-12
        for k in ("frames1", "frames2")
    )
    localization_ok = component_ok and temporal_ok
    audit = {
        "LOCALIZATION_TREND_CHECK": "PASS" if localization_ok else "FAIL",
        "component_full_localization_best": component_ok,
        "three_frame_full_localization_best": temporal_ok,
        "heading_fusion_alpha": _heading_fusion_alpha(),
        "paired_bootstrap": {
            key: _paired_bootstrap(full["_errors"], rows[key]["_errors"])
            for key in ("no_gru", "no_kalman", "no_ms", "frames1", "frames2")
        },
        "claim_guidance": (
            "Full is numerically best on MLE and LSR@15 across the checked component/temporal ablations; inspect confidence intervals before claiming significance."
            if localization_ok else
            "At least one measured ablation is better on MLE or LSR@15; report that trade-off rather than forcing a preferred ranking."
        ),
        "integrity": "All values are recomputed from measured frame-level outputs; shared heading fusion is applied identically to every row.",
    }

    payload = {
        "cities": args.cities,
        "metric_set": ["R@1*_pct", "LSR@15_pct", "HSR@15_pct", "MLE_m", "MHE_deg"],
        "heading_fusion_alpha": _heading_fusion_alpha(),
        "rows": {k: _clean(v) for k, v in rows.items()},
        "audit": audit,
    }
    (root / "paper_ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = list(_clean(next(iter(rows.values()))).keys())
    with (root / "paper_ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_clean(rows[k]) for k in keys)
    (root / "paper_ablation_tables.md").write_text(_markdown(rows, audit, args.cities), encoding="utf-8")
    (root / "paper_ablation_tables.tex").write_text(_latex(rows, args.cities), encoding="utf-8")
    (root / "paper_trend_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(_markdown(rows, audit, args.cities))


if __name__ == "__main__":
    main()
