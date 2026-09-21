#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CITIES = ("citya", "cityb", "cityc", "cityd")
ROUTE_ALIAS = {"test_01": "nav50", "test_02": "nav51", "nav50": "nav50", "nav51": "nav51"}
CORE = ["corev4_no_gru", "corev4_no_kalman", "corev4_no_ms", "corev4_full"]
TEMPORAL = ["corev4_ctx1", "corev4_ctx2", "corev4_full"]
LABELS = {
    "corev4_no_gru": "w/o GRU",
    "corev4_no_kalman": "w/o Kalman",
    "corev4_no_ms": "w/o Final MeanShift",
    "corev4_full": "Full",
    "corev4_ctx1": "1 frame context",
    "corev4_ctx2": "2 frame context",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows if r.get(key, "") not in ("", None)], dtype=float)


def find_csv(folder: Path, route_key: str, summary: dict) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.is_file():
        return p
    if p.name and (folder / p.name).is_file():
        return folder / p.name
    nav = ROUTE_ALIAS.get(route_key, route_key)
    prefix = "route_B" if nav == "nav50" else "route_C"
    m = sorted(folder.glob(prefix + "_*_frames.csv"))
    if not m:
        raise FileNotFoundError(f"{folder}: missing frames CSV for {route_key}")
    return m[-1]


def pooled(rows):
    e = arr(rows, "error_final_m")
    jump = arr(rows, "abnormal_jump")
    step = arr(rows, "final_step_m")
    lat = arr(rows, "end_to_end_latency_ms")
    if not len(e):
        raise RuntimeError("No error_final_m rows")
    return {
        "Frames": int(len(e)),
        "MLE_m": float(e.mean()),
        "MedLE_m": float(np.median(e)),
        "P90_m": float(np.percentile(e, 90)),
        "P95_m": float(np.percentile(e, 95)),
        "LSR@5_pct": float(100*np.mean(e <= 5)),
        "LSR@10_pct": float(100*np.mean(e <= 10)),
        "LSR@15_pct": float(100*np.mean(e <= 15)),
        "JumpRate_pct": float(100*np.mean(jump != 0)) if len(jump) else None,
        "MaxFinalStep_m": float(step.max()) if len(step) else None,
        "InferenceMean_ms": float(lat.mean()) if len(lat) else None,
        "FPS": float(1000/lat.mean()) if len(lat) and lat.mean() > 0 else None,
    }


def collect(root: Path, variant: str):
    all_rows=[]; sources=[]; validation=[]
    for city in CITIES:
        folder=root/city/"variants_core_v4"/variant
        sp=folder/"bearing_v39_summary.json"
        if not sp.is_file(): raise FileNotFoundError(sp)
        summaries=json.loads(sp.read_text(encoding="utf-8"))
        for k,s in summaries.items():
            if k not in ROUTE_ALIAS: continue
            cp=find_csv(folder,k,s); rr=read_csv(cp); all_rows.extend(rr)
            sources.append({"city":city,"route":ROUTE_ALIAS[k],"csv":str(cp)})
        mp=folder/"core_v4_manifest.json"
        if mp.is_file():
            m=json.loads(mp.read_text(encoding="utf-8"))
            validation.append({
                "city":city,
                "validation_kalman_gain_m":m.get("kalman_profile_selection",{}).get("validation_kalman_gain_m"),
                "selected":m.get("kalman_profile_selection",{}).get("selected"),
            })
    return {"Variant":variant,"Label":LABELS.get(variant,variant),**pooled(all_rows),"Sources":sources,"ValidationAudit":validation}


def write_csv(path, rows, keys):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in keys})


def md(headers, rows):
    def fmt(v):
        if v is None:return "—"
        if isinstance(v,float):return f"{v:.3f}"
        return str(v)
    return "\n".join([
        "| "+" | ".join(headers)+" |",
        "| "+" | ".join(["---"]*len(headers))+" |",
        *["| "+" | ".join(fmt(r.get(h)) for h in headers)+" |" for r in rows],
    ])


def best_check(full, competitors):
    checks={
        "MLE": all(full["MLE_m"] <= r["MLE_m"] + 1e-12 for r in competitors),
        "P90": all(full["P90_m"] <= r["P90_m"] + 1e-12 for r in competitors),
        "LSR5": all(full["LSR@5_pct"] >= r["LSR@5_pct"] - 1e-12 for r in competitors),
        "LSR15": all(full["LSR@15_pct"] >= r["LSR@15_pct"] - 1e-12 for r in competitors),
        "JumpRate": all(full["JumpRate_pct"] <= r["JumpRate_pct"] + 1e-12 for r in competitors),
    }
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument("--suite-root",required=True);p.add_argument("--output-dir");a=p.parse_args()
    root=Path(a.suite_root).resolve();out=Path(a.output_dir).resolve() if a.output_dir else root/"paper_core_v4";out.mkdir(parents=True,exist_ok=True)
    variants=[]
    for v in CORE+TEMPORAL:
        if v not in variants:variants.append(v)
    R={v:collect(root,v) for v in variants}
    core=[R[v] for v in CORE]; temporal=[R[v] for v in TEMPORAL]
    core_check=best_check(R["corev4_full"],[R[v] for v in CORE if v!="corev4_full"])
    temporal_check=best_check(R["corev4_full"],[R["corev4_ctx1"],R["corev4_ctx2"]])
    core_keys=["Label","MLE_m","MedLE_m","P90_m","LSR@5_pct","LSR@15_pct","JumpRate_pct","MaxFinalStep_m"]
    temp_keys=["Label","MLE_m","P90_m","LSR@5_pct","LSR@15_pct","JumpRate_pct"]
    write_csv(out/"table_core_components.csv",core,core_keys)
    write_csv(out/"table_temporal_context_same_checkpoint.csv",temporal,temp_keys)
    payload={
        "suite":str(root),
        "protocol":{
            "core":"same trained 3-frame Full checkpoint; remove exactly one component at inference",
            "temporal":"same trained 3-frame Full checkpoint; truncate explicit temporal context to 1f/2f",
            "kalman_profile_selection":"train_01 validation calibration only; held-out nav50/nav51 never used for profile selection",
        },
        "core_components":core,
        "temporal_context":temporal,
        "full_best_core_check":core_check,
        "full_best_temporal_check":temporal_check,
    }
    (out/"core_v4_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    text=[
        "# Core V4 paper tables","",
        "## Table 1. Component ablation","",
        md(core_keys,core),"",
        "Protocol: every row uses the same trained 3-frame Full checkpoint; only the named component is disabled.","",
        "## Table 2. Temporal-context truncation","",
        md(temp_keys,temporal),"",
        "Protocol: 1f/2f/3f use the same trained 3-frame Full checkpoint. No 1f/2f retraining is used in this table.","",
        "## Audit","",
        f"- Full-best core check: {core_check}",
        f"- Full-best temporal check: {temporal_check}",
        "- Kalman/runtime profile is selected only from train_01 validation calibration.",
        "- Held-out nav50/nav51 metrics are never used for profile selection or automatic retuning.",
    ]
    (out/"PAPER_CORE_V4_TABLES.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("[CORE-V4 TABLES DONE]",out)
    print("[FULL-BEST CORE]", "PASS" if all(core_check.values()) else "NOT_ALL_METRICS")
    print("[FULL-BEST TEMPORAL]", "PASS" if all(temporal_check.values()) else "NOT_ALL_METRICS")
    for r in core:
        print("%-20s MLE=%7.4f P90=%7.4f LSR5=%6.2f Jump=%6.3f"%(r["Label"],r["MLE_m"],r["P90_m"],r["LSR@5_pct"],r["JumpRate_pct"]))
    for r in temporal:
        print("%-20s MLE=%7.4f P90=%7.4f LSR5=%6.2f Jump=%6.3f"%(r["Label"],r["MLE_m"],r["P90_m"],r["LSR@5_pct"],r["JumpRate_pct"]))

if __name__=="__main__":main()
