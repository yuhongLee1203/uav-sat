#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

CITIES=("citya","cityb","cityc","cityd")
ROUTE_ALIAS={"test_01":"nav50","test_02":"nav51","nav50":"nav50","nav51":"nav51"}

PAPER_VARIANTS=("no_gru","no_ms","frames1","frames2","full36","full","grid4","grid5","grid7","grid8")

LABELS={
    "no_gru":"w/o temporal GRU",
    "no_ms":"w/o final MeanShift",
    "frames1":"1 frame",
    "frames2":"2 frames",
    "full":"Full (3-frame)",
    "full36":"Full 6x6 search",
    "grid4":"4x4",
    "grid5":"5x5",
    "grid7":"7x7",
    "grid8":"8x8",
}


def read_csv(path:Path):
    with path.open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_frames_csv(folder:Path, route_key:str, summary:dict)->Path:
    p=Path(str(summary.get("CSV","")))
    if p.is_file(): return p
    if p.name and (folder/p.name).is_file(): return folder/p.name
    nav=ROUTE_ALIAS.get(route_key,route_key)
    prefix="route_B" if nav=="nav50" else "route_C"
    matches=sorted(folder.glob(prefix+"_*_frames.csv"))
    if not matches: raise FileNotFoundError(f"{folder}: no frames CSV for {route_key}")
    return matches[-1]


def arr(rows,key):
    return np.asarray([float(r[key]) for r in rows if r.get(key,"") not in ("",None)],dtype=np.float64)


def pooled(rows):
    err=arr(rows,"error_final_m")
    latency=arr(rows,"end_to_end_latency_ms")
    jumps=arr(rows,"abnormal_jump")
    capture=arr(rows,"selected_candidate_capture")
    step=arr(rows,"final_step_m")
    if not len(err): raise RuntimeError("empty error_final_m")
    return {
        "Frames":int(len(err)),
        "MLE_m":float(err.mean()),
        "MedLE_m":float(np.median(err)),
        "P90_m":float(np.percentile(err,90)),
        "LSR@5_pct":float(100*np.mean(err<=5)),
        "LSR@15_pct":float(100*np.mean(err<=15)),
        "JumpRate_pct":float(100*np.mean(jumps!=0)) if len(jumps) else None,
        "MaxFinalStep_m":float(step.max()) if len(step) else None,
        "SelectedCapture_pct":float(100*capture.mean()) if len(capture) else None,
        "InferenceMean_ms":float(latency.mean()) if len(latency) else None,
        "FPS":float(1000/latency.mean()) if len(latency) and latency.mean()>0 else None,
    }


def collect(root:Path,variant:str):
    rows=[]
    for city in CITIES:
        folder=root/city/"variants"/variant
        sp=folder/"bearing_v39_summary.json"
        if not sp.is_file(): raise FileNotFoundError(sp)
        summaries=json.loads(sp.read_text(encoding="utf-8"))
        for key,summary in summaries.items():
            if key in ROUTE_ALIAS:
                rows.extend(read_csv(find_frames_csv(folder,key,summary)))
    return {"Variant":variant,"Label":LABELS[variant],**pooled(rows)}


def write_csv(path:Path,rows,fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields})


def md(fields,rows):
    def fmt(v):
        if v is None:return "—"
        return f"{v:.3f}" if isinstance(v,float) else str(v)
    return "\n".join([
        "| "+" | ".join(fields)+" |",
        "| "+" | ".join(["---"]*len(fields))+" |",
        *["| "+" | ".join(fmt(r.get(k)) for k in fields)+" |" for r in rows]
    ])


def main():
    p=argparse.ArgumentParser();p.add_argument("--suite-root",required=True);p.add_argument("--output-dir")
    a=p.parse_args();root=Path(a.suite_root).resolve();out=Path(a.output_dir).resolve() if a.output_dir else root/"paper_core"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)

    r={v:collect(root,v) for v in PAPER_VARIANTS}

    core=[r["no_gru"],r["no_ms"],r["full"]]
    search=[]
    for v,cands in (("full36",36),("full",18)):
        x=dict(r[v]);x["Candidates"]=cands
        x["Search"]="Full 6x6" if v=="full36" else "Forward 3x6"
        search.append(x)
    temporal=[]
    for v,n in (("frames1",1),("frames2",2),("full",3)):
        x=dict(r[v]);x["FramesInput"]=n;temporal.append(x)
    grid=[]
    for v,n in (("grid4",4),("grid5",5),("full",6),("grid7",7),("grid8",8)):
        x=dict(r[v]);x["Grid"] = f"{n}x{n}";grid.append(x)

    core_fields=["Label","MLE_m","P90_m","LSR@5_pct","LSR@15_pct","JumpRate_pct"]
    search_fields=["Search","Candidates","MLE_m","P90_m","SelectedCapture_pct","InferenceMean_ms","FPS"]
    temporal_fields=["FramesInput","MLE_m","P90_m","LSR@5_pct","LSR@15_pct","JumpRate_pct"]
    grid_fields=["Grid","MLE_m","P90_m","LSR@5_pct","InferenceMean_ms"]

    write_csv(out/"table_core_components.csv",core,core_fields)
    write_csv(out/"table_search_efficiency.csv",search,search_fields)
    write_csv(out/"table_temporal_context_single_seed.csv",temporal,temporal_fields)
    write_csv(out/"table_final_ms_grid.csv",grid,grid_fields)

    payload={
        "suite":str(root),
        "core_components":core,
        "search_efficiency":search,
        "temporal_context_single_seed":temporal,
        "final_ms_grid":grid,
        "paper_excluded":{
            "top1_decoder":"Not part of the proposed architecture; removed from paper-facing tables.",
            "prior_jitter_sensitivity":"Kept only in the historical audit suite; not a paper-facing ablation.",
            "kalman_and_heading_feedback":"Historical measurements are preserved, but current results do not support claiming an accuracy gain, so they are not presented as positive component ablations.",
        },
        "integrity":"Measured historical outputs are preserved. This exporter changes presentation only and does not edit metric values.",
        "protocol":"Current Full localization remains a controlled local-prior/jitter experiment; paper wording must not claim fully GT-free deployment."
    }
    (out/"paper_core_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

    text=[
        "# Paper Core Tables","",
        "## Table 1. Core component ablation","",md(core_fields,core),"",
        "## Table 2. Search-region efficiency","",md(search_fields,search),"",
        "## Table 3. Temporal context (single seed; multi-seed table should be used for the final paper)","",md(temporal_fields,temporal),"",
        "## Supplementary. Final MeanShift grid size","",md(grid_fields,grid),"",
        "## Notes","",
        "- Top-1 and prior-jitter sensitivity are not included in paper-facing tables.",
        "- The 1/2/3-frame rows use separately trained temporal checkpoints.",
        "- Do not change or select settings using nav50/nav51 to force a preferred ordering.",
        "- Current Full results still use the controlled local-prior/jitter protocol; removing a sensitivity table does not make the inference GT-free.",
    ]
    (out/"PAPER_CORE_TABLES.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("[PAPER CORE TABLES DONE]",out)

if __name__=="__main__":main()
