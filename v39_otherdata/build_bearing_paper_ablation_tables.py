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
    "temporal_context_retrained": ["frames1", "frames2", "full"],
    "search_policy": ["full36", "full"],
    "visual_anchor": ["front_top1", "front_weighted", "full"],
    "prior_jitter_sensitivity": ["jitter0", "jitter4", "full", "jitter12", "jitter16"],
    "final_ms_grid": ["grid4", "grid5", "full", "grid7", "grid8"],
}
LABELS = {
    "full": "SoftMS visual anchor (Full)",
    "no_gru": "w/o GRU (inference removal)",
    "no_kalman": "w/o Kalman",
    "no_ms": "w/o final MeanShift",
    "no_heading_feedback": "w/o learned heading feedback",
    "frames1": "1 frame (retrained temporal)",
    "frames2": "2 frames (retrained temporal)",
    "full36": "Full 6x6 scoring (36 candidates)",
    "front_top1": "Top-1 visual anchor",
    "front_weighted": "Posterior-weighted visual anchor",
    "jitter0": "Prior jitter 0 m",
    "jitter4": "Prior jitter 4 m",
    "jitter12": "Prior jitter 12 m",
    "jitter16": "Prior jitter 16 m",
    "grid4": "Final MS grid 4x4",
    "grid5": "Final MS grid 5x5",
    "grid7": "Final MS grid 7x7",
    "grid8": "Final MS grid 8x8",
}


def protocol_label(v: str) -> str:
    if v in {"frames1", "frames2"}:
        return "retrained temporal-context ablation"
    if v == "full":
        return "trained Full baseline"
    return "inference/component or sensitivity ablation using Full checkpoint"


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


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows if r.get(key, "") not in ("", None)], dtype=np.float64)


def pooled(rows):
    err = arr(rows, "error_final_m")
    he = np.abs(arr(rows, "heading_error_deg"))
    step = arr(rows, "final_step_m")
    latency = arr(rows, "end_to_end_latency_ms")
    jumps = arr(rows, "abnormal_jump")
    capture = arr(rows, "selected_candidate_capture")
    if not len(err):
        raise RuntimeError("empty error_final_m")
    return {
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
        "SelectedCapture_pct": float(100.0 * capture.mean()) if len(capture) else None,
        "InferenceMean_ms": float(latency.mean()) if len(latency) else None,
        "FPS": float(1000.0 / latency.mean()) if len(latency) and latency.mean() > 0 else None,
    }


def collect(root: Path, variant: str):
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
    return {
        "Variant": variant,
        "Label": LABELS.get(variant, variant),
        "AblationProtocol": protocol_label(variant),
        **pooled(all_rows),
        "Sources": sources,
    }


def add_deltas(row, full):
    row = dict(row)
    row["DeltaMLE_m_vsFull"] = row["MLE_m"] - full["MLE_m"]
    row["DeltaLSR15_pp_vsFull"] = row["LSR@15_pct"] - full["LSR@15_pct"]
    row["DeltaMHE_deg_vsFull"] = row["MHE_deg"] - full["MHE_deg"] if row["MHE_deg"] is not None else None
    row["DeltaHSR15_pp_vsFull"] = row["HSR@15_pct"] - full["HSR@15_pct"] if row["HSR@15_pct"] is not None else None
    row["DeltaLatency_ms_vsFull"] = row["InferenceMean_ms"] - full["InferenceMean_ms"] if row["InferenceMean_ms"] is not None and full["InferenceMean_ms"] is not None else None
    return row


def write_csv(path: Path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k != "Sources" and k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in keys})


def md(headers, rows):
    def fmt(v):
        if v is None: return "—"
        if isinstance(v, float): return f"{v:.3f}"
        return str(v)
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ] + ["| " + " | ".join(fmt(r.get(h)) for h in headers) + " |" for r in rows])


def main():
    a = argparse.ArgumentParser(); a.add_argument("--suite-root", required=True); a.add_argument("--output-dir")
    x = a.parse_args(); root = Path(x.suite_root).resolve(); out = Path(x.output_dir).resolve() if x.output_dir else root / "paper_ablation"
    out.mkdir(parents=True, exist_ok=True)
    ordered = []
    for vv in GROUPS.values():
        for v in vv:
            if v not in ordered: ordered.append(v)
    raw = {v: collect(root, v) for v in ordered}
    full = raw["full"]
    results = {v: add_deltas(raw[v], full) for v in ordered}

    payload = {
        "suite": str(root),
        "protocol_notes": {
            "temporal_context": "frames1/frames2 are separately trained temporal checkpoints on the same training split; Full is the 3-frame trained checkpoint.",
            "decoder": "Top-1 and posterior-weighted rows change only the front visual anchor; Full preserves the formal front SoftMS.",
            "other_rows": "Other rows are inference/component or sensitivity ablations reusing the Full checkpoint unless the row explicitly says otherwise.",
            "selection": "No nav50/nav51 metric is used to select ablation settings.",
            "controlled_prior": "This remains a controlled local-prior/jitter experiment and must not be described as fully GT-free deployment.",
        },
        "groups": {},
    }
    headers = ["Label", "AblationProtocol", "MLE_m", "DeltaMLE_m_vsFull", "P90_m", "LSR@5_pct", "LSR@15_pct", "DeltaLSR15_pp_vsFull", "MHE_deg", "DeltaMHE_deg_vsFull", "HSR@15_pct", "DeltaHSR15_pp_vsFull", "JumpRate_pct", "MaxFinalStep_m", "SelectedCapture_pct", "InferenceMean_ms", "FPS"]
    text = [
        "# Paper-facing ablation tables", "",
        "> 1/2/3-frame temporal-context rows use separately trained temporal checkpoints.",
        "> Kalman / MeanShift / search-policy / decoder / heading-feedback / jitter / grid rows are one-factor inference-component or sensitivity ablations using the Full checkpoint.",
        "> All held-out nav50/nav51 results are measured outputs; no row is edited to force Full to win.", "",
    ]
    for group, variants in GROUPS.items():
        rows = [results[v] for v in variants]
        payload["groups"][group] = rows
        write_csv(out / f"ablation_{group}.csv", rows)
        text += [f"## {group.replace('_', ' ').title()}", "", md(headers, rows), ""]
    write_csv(out / "ablation_all.csv", [results[v] for v in ordered])
    (out / "ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "ABLATION_TABLES.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("[PAPER ABLATION TABLES DONE]", out)
    for v in ordered:
        r = results[v]
        print("%-22s MLE=%6.3f dMLE=%+6.3f LSR15=%6.2f MHE=%6.2f HSR15=%6.2f jump=%5.2f" % (
            v, r["MLE_m"], r["DeltaMLE_m_vsFull"], r["LSR@15_pct"],
            r["MHE_deg"] if r["MHE_deg"] is not None else float("nan"),
            r["HSR@15_pct"] if r["HSR@15_pct"] is not None else 0.0,
            r["JumpRate_pct"] if r["JumpRate_pct"] is not None else 0.0,
        ))

if __name__ == "__main__": main()
