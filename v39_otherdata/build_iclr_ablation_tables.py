#!/usr/bin/env python3
"""Aggregate measured Bearing-UAV four-city ablations into paper tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COMPONENTS = ["no_gru", "no_kalman", "no_ms", "full"]
TEMPORAL = ["frames1", "frames2", "full"]
WINDOWS = ["grid4", "grid5", "full", "grid7", "grid8"]
LABELS = {
    "no_gru": "w/o GRU",
    "no_kalman": "w/o Kalman",
    "no_ms": "w/o MeanShift",
    "full": "Full",
    "frames1": "1 frame",
    "frames2": "2 frames",
    "grid4": "4x4",
    "grid5": "5x5",
    "grid7": "7x7",
    "grid8": "8x8",
}


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


def _navigation_keys(summaries: dict) -> list[tuple[str, str]]:
    """Return paper label + stored key, supporting old result folders too."""
    if "nav50" in summaries and "nav51" in summaries:
        return [("nav50", "nav50"), ("nav51", "nav51")]
    if "test_01" in summaries and "test_02" in summaries:
        return [("nav50", "test_01"), ("nav51", "test_02")]
    raise KeyError(f"summary must contain nav50/nav51; got {sorted(summaries)}")


def _read_variant(root: Path, cities: list[str], variant: str) -> dict:
    errors, per_route, ms_latency = [], [], []
    for city in cities:
        output = root / city / "variants" / variant
        summary_path = output / "bearing_v39_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        for nav_label, stored_key in _navigation_keys(summaries):
            summary = summaries[stored_key]
            csv_path = _csv_for(summary, output)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            route_errors = np.asarray(
                [float(row["error_final_m"]) for row in rows], dtype=np.float64
            )
            errors.append(route_errors)
            per_route.append({"city": city, "navigation": nav_label, "frames": len(rows)})
            samples = int(summary.get("MS_LatencySamples", len(rows)))
            latency = float(summary.get("MS_LatencyMean_ms", 0.0))
            if latency > 0:
                ms_latency.extend([latency] * max(samples, 1))
    values = np.concatenate(errors)
    tail = values[values >= np.quantile(values, 0.90)]
    return {
        "variant": variant,
        "label": LABELS.get(variant, variant),
        "frames": int(values.size),
        "MLE_m": float(values.mean()),
        "MedLE_m": float(np.median(values)),
        "P90_m": float(np.quantile(values, 0.90)),
        "P95_m": float(np.quantile(values, 0.95)),
        "P99_m": float(np.quantile(values, 0.99)),
        "CVaR90_m": float(tail.mean()),
        "LSR@3_pct": 100.0 * float(np.mean(values <= 3.0)),
        "LSR@5_pct": 100.0 * float(np.mean(values <= 5.0)),
        "LSR@10_pct": 100.0 * float(np.mean(values <= 10.0)),
        "LSR@15_pct": 100.0 * float(np.mean(values <= 15.0)),
        "LSR@20_pct": 100.0 * float(np.mean(values <= 20.0)),
        "MS_Latency_ms": float(np.mean(ms_latency)) if ms_latency else 0.0,
        "errors": values,
        "routes": per_route,
    }


def _paired_bootstrap(full: np.ndarray, other: np.ndarray, seed: int = 2027) -> dict:
    if full.shape != other.shape:
        raise ValueError("paired ablation outputs are not frame-aligned")
    rng = np.random.default_rng(seed)
    n = full.size
    delta_mle, delta_lsr5 = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        delta_mle.append(float(full[idx].mean() - other[idx].mean()))
        delta_lsr5.append(
            100.0 * float(np.mean(full[idx] <= 5.0) - np.mean(other[idx] <= 5.0))
        )
    return {
        "full_minus_ablation_MLE_m": float(full.mean() - other.mean()),
        "MLE_95CI": [float(x) for x in np.quantile(delta_mle, [0.025, 0.975])],
        "full_minus_ablation_LSR@5_pct": 100.0 * float(
            np.mean(full <= 5.0) - np.mean(other <= 5.0)
        ),
        "LSR@5_95CI": [float(x) for x in np.quantile(delta_lsr5, [0.025, 0.975])],
    }


def _clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in {"errors", "routes"}}


def _markdown(rows: dict[str, dict], audit: dict, cities: list[str]) -> str:
    city_text = ", ".join(c.upper() for c in cities)
    lines = [
        "# Bearing-UAV four-city ICLR ablation (measured)", "",
        f"Dataset domains: {city_text}; held-out navigation trajectories are reported as nav50/nav51.",
        "The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.", "",
        "## Component removal", "",
        "| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for key in COMPONENTS:
        r = rows[key]
        lines.append(f"| {r['label']} | {r['MLE_m']:.3f} | {r['P90_m']:.3f} | {r['LSR@3_pct']:.2f}% | {r['LSR@5_pct']:.2f}% | {r['LSR@10_pct']:.2f}% |")
    lines += ["", "## Temporal input", "", "| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |", "|---|---:|---:|---:|---:|"]
    for key in TEMPORAL:
        r = rows[key]
        lines.append(f"| {r['label']} | {r['MLE_m']:.3f} | {r['P90_m']:.3f} | {r['LSR@3_pct']:.2f}% | {r['LSR@5_pct']:.2f}% |")
    lines += ["", "## Final MeanShift window", "", "| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for key in WINDOWS:
        r = rows[key]
        grid = 6 if key == "full" else int(key[-1])
        lines.append(f"| {grid}x{grid} | {grid*grid} | {r['MLE_m']:.3f} | {r['P90_m']:.3f} | {r['LSR@5_pct']:.2f}% | {r['MS_Latency_ms']:.3f} |")
    lines += ["", "## Integrity audit", "", f"`FULL_TREND_CHECK={audit['FULL_TREND_CHECK']}`", "", audit["claim_guidance"], ""]
    return "\n".join(lines)


def _latex(rows: dict[str, dict]) -> str:
    out = [
        "% Auto-generated from measured Bearing-UAV outputs. Do not edit values manually.",
        "\\begin{table}[t]", "\\caption{Component ablation across Bearing-UAV Cities A--D.}", "\\label{tab:component-ablation}", "\\centering", "\\small", "\\begin{tabular}{lrrrrr}", "\\toprule", "Variant & MLE$\\downarrow$ & P90$\\downarrow$ & LSR@3$\\uparrow$ & LSR@5$\\uparrow$ & LSR@10$\\uparrow$ \\\\", "\\midrule",
    ]
    for key in COMPONENTS:
        r = rows[key]
        out.append(f"{r['label']} & {r['MLE_m']:.3f} & {r['P90_m']:.3f} & {r['LSR@3_pct']:.2f} & {r['LSR@5_pct']:.2f} & {r['LSR@10_pct']:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "", "\\begin{table}[t]", "\\caption{Temporal input and final MeanShift window ablations.}", "\\label{tab:temporal-window-ablation}", "\\centering", "\\small", "\\begin{tabular}{lrrrr}", "\\toprule", "Temporal input & MLE$\\downarrow$ & P90$\\downarrow$ & LSR@3$\\uparrow$ & LSR@5$\\uparrow$ \\\\", "\\midrule"]
    for key in TEMPORAL:
        r = rows[key]
        out.append(f"{r['label']} & {r['MLE_m']:.3f} & {r['P90_m']:.3f} & {r['LSR@3_pct']:.2f} & {r['LSR@5_pct']:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\vspace{0.8em}", "\\begin{tabular}{lrrrr}", "\\toprule", "MS window & Candidates & MLE$\\downarrow$ & LSR@5$\\uparrow$ & Latency (ms)$\\downarrow$ \\\\", "\\midrule"]
    for key in WINDOWS:
        r = rows[key]
        grid = 6 if key == "full" else int(key[-1])
        out.append(f"{grid}$\\times${grid} & {grid*grid} & {r['MLE_m']:.3f} & {r['LSR@5_pct']:.2f} & {r['MS_Latency_ms']:.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    args = p.parse_args()
    root = Path(args.suite_root).resolve()
    keys = sorted(set(COMPONENTS + TEMPORAL + WINDOWS))
    rows = {key: _read_variant(root, args.cities, key) for key in keys}
    full = rows["full"]
    component_ok = all(
        full["MLE_m"] <= rows[k]["MLE_m"] + 1e-12
        and full["LSR@5_pct"] >= rows[k]["LSR@5_pct"] - 1e-12
        for k in ("no_gru", "no_kalman", "no_ms")
    )
    temporal_ok = all(
        full["MLE_m"] <= rows[k]["MLE_m"] + 1e-12
        and full["LSR@5_pct"] >= rows[k]["LSR@5_pct"] - 1e-12
        for k in ("frames1", "frames2")
    )
    audit = {
        "FULL_TREND_CHECK": "PASS" if component_ok and temporal_ok else "FAIL",
        "component_full_best": component_ok,
        "three_frame_full_best": temporal_ok,
        "paired_bootstrap": {
            key: _paired_bootstrap(full["errors"], rows[key]["errors"])
            for key in ("no_gru", "no_kalman", "no_ms", "frames1", "frames2")
        },
        "claim_guidance": (
            "Full is numerically best on the predeclared primary metrics; inspect paired confidence intervals before claiming significance."
            if component_ok and temporal_ok else
            "At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results."
        ),
        "integrity": "No result is modified, hidden, or selectively discarded to force a preferred ranking.",
    }
    payload = {"cities": args.cities, "rows": {k: _clean(v) for k, v in rows.items()}, "audit": audit}
    (root / "paper_ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = list(_clean(next(iter(rows.values()))).keys())
    with (root / "paper_ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_clean(rows[k]) for k in keys)
    (root / "paper_ablation_tables.md").write_text(_markdown(rows, audit, args.cities), encoding="utf-8")
    (root / "paper_ablation_tables.tex").write_text(_latex(rows), encoding="utf-8")
    (root / "paper_trend_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(_markdown(rows, audit, args.cities))


if __name__ == "__main__":
    main()
