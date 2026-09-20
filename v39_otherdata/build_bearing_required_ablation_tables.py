#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CITIES = ("citya", "cityb", "cityc", "cityd")
ROUTE_ALIAS = {"test_01": "nav50", "test_02": "nav51", "nav50": "nav50", "nav51": "nav51"}
GROUPS = {
    "core_components": ["no_gru", "no_kalman", "no_ms", "no_heading_feedback", "full"],
    "temporal_context": ["frames1", "frames2", "full"],
    "search_policy": ["full36", "full"],
    "visual_anchor": ["front_top1", "full", "front_softms"],
    "prior_jitter": ["jitter0", "jitter4", "full", "jitter12", "jitter16"],
    "final_ms_grid": ["grid4", "grid5", "full", "grid7", "grid8"],
}
LABELS = {
    "full": "Full: Forward-18 + 3f GRU + Kalman + final SoftMS",
    "no_gru": "w/o GRU",
    "no_kalman": "w/o Kalman",
    "no_ms": "w/o final MeanShift",
    "no_heading_feedback": "w/o learned heading feedback",
    "frames1": "1 frame",
    "frames2": "2 frames",
    "full36": "Full 6x6 scoring (36 candidates)",
    "front_top1": "Top-1 visual anchor",
    "front_softms": "Front SoftMS visual anchor",
    "jitter0": "Prior jitter 0 m",
    "jitter4": "Prior jitter 4 m",
    "jitter12": "Prior jitter 12 m",
    "jitter16": "Prior jitter 16 m",
    "grid4": "Final MS grid 4x4",
    "grid5": "Final MS grid 5x5",
    "grid7": "Final MS grid 7x7",
    "grid8": "Final MS grid 8x8",
}


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_csv(full: Path, route_key: str, summary: dict) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.is_file():
        return p
    if p.name and (full / p.name).is_file():
        return full / p.name
    nav = ROUTE_ALIAS.get(route_key, route_key)
    prefix = "route_B" if nav == "nav50" else "route_C"
    matches = sorted(full.glob(prefix + "_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"{full}: no frames CSV for {route_key}/{nav}")
    return matches[-1]


def _float(rows, key):
    return np.asarray([float(r[key]) for r in rows if r.get(key, "") not in ("", None)], dtype=np.float64)


def pooled_metrics(rows):
    err = _float(rows, "error_final_m")
    he = np.abs(_float(rows, "heading_error_deg"))
    step = _float(rows, "final_step_m")
    latency = _float(rows, "end_to_end_latency_ms")
    jumps = _float(rows, "abnormal_jump")
    cap = _float(rows, "selected_candidate_capture")
    if len(err) == 0:
        raise RuntimeError("empty error_final_m")
    out = {
        "Frames": int(len(err)),
        "MLE_m": float(err.mean()),
        "MedLE_m": float(np.median(err)),
        "P90_m": float(np.percentile(err, 90)),
        "LSR@5_pct": float(100.0 * np.mean(err <= 5.0)),
        "LSR@10_pct": float(100.0 * np.mean(err <= 10.0)),
        "LSR@15_pct": float(100.0 * np.mean(err <= 15.0)),
        "MHE_deg": float(he.mean()) if len(he) else None,
        "MedHE_deg": float(np.median(he)) if len(he) else None,
        "HSR@15_pct": float(100.0 * np.mean(he <= 15.0)) if len(he) else None,
        "JumpRate_pct": float(100.0 * np.mean(jumps != 0)) if len(jumps) else None,
        "MaxFinalStep_m": float(step.max()) if len(step) else None,
        "SelectedCapture_pct": float(100.0 * cap.mean()) if len(cap) else None,
        "InferenceMean_ms": float(latency.mean()) if len(latency) else None,
        "FPS": float(1000.0 / latency.mean()) if len(latency) and latency.mean() > 0 else None,
    }
    return out


def collect_variant(root: Path, variant: str):
    all_rows = []
    sources = []
    for city in CITIES:
        full = root / city / "variants" / variant
        summary_path = full / "bearing_v39_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, summary in summaries.items():
            if key not in ROUTE_ALIAS:
                continue
            cp = find_csv(full, key, summary)
            rows = read_csv(cp)
            all_rows.extend(rows)
            sources.append({"city": city, "route": ROUTE_ALIAS[key], "csv": str(cp)})
    result = {"Variant": variant, "Label": LABELS.get(variant, variant), **pooled_metrics(all_rows)}
    result["AblationProtocol"] = "inference/component ablation; full trained checkpoint reused"
    result["Sources"] = sources
    return result


def write_csv(path: Path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k == "Sources":
                continue
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def md(headers, rows):
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h)) for h in headers) + " |")
    return "\n".join(lines)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--suite-root", required=True)
    a.add_argument("--output-dir")
    args = a.parse_args()
    root = Path(args.suite_root).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else root / "paper_ablation"
    out.mkdir(parents=True, exist_ok=True)

    ordered = []
    seen = set()
    for variants in GROUPS.values():
        for v in variants:
            if v not in seen:
                ordered.append(v); seen.add(v)
    results = {v: collect_variant(root, v) for v in ordered}

    payload = {
        "suite": str(root),
        "protocol": "Inference/component ablations reusing the trained Full checkpoint; no held-out metrics are used to select settings.",
        "groups": {},
    }
    headers = ["Label", "MLE_m", "MedLE_m", "P90_m", "LSR@5_pct", "LSR@15_pct", "MHE_deg", "HSR@15_pct", "JumpRate_pct", "MaxFinalStep_m", "SelectedCapture_pct", "InferenceMean_ms", "FPS"]
    markdown = [
        "# Paper-required ablation tables",
        "",
        "> Protocol: inference/component ablation. All rows reuse the same trained Full checkpoint unless explicitly stated otherwise.",
        "> These tables isolate runtime contribution/sensitivity; do not describe them as independently retrained architecture variants.",
        "",
    ]
    for group, variants in GROUPS.items():
        rows = [results[v] for v in variants]
        payload["groups"][group] = rows
        write_csv(out / f"ablation_{group}.csv", rows)
        markdown += [f"## {group.replace('_',' ').title()}", "", md(headers, rows), ""]

    all_rows = [results[v] for v in ordered]
    write_csv(out / "ablation_all.csv", all_rows)
    (out / "ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "ABLATION_TABLES.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("[ABLATION TABLES DONE]", out)
    for v in ordered:
        r = results[v]
        print("%-22s MLE=%6.3f LSR15=%6.2f MHE=%6.2f HSR15=%6.2f jump=%5.2f" % (
            v, r["MLE_m"], r["LSR@15_pct"], r["MHE_deg"] or float("nan"),
            r["HSR@15_pct"] or 0.0, r["JumpRate_pct"] or 0.0))


if __name__ == "__main__":
    main()
