#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CITIES=("citya","cityb","cityc","cityd")
VARIANTS=(("frames1",1),("frames2",2),("full",3))
ROUTE_ALIAS={"test_01":"nav50","test_02":"nav51","nav50":"nav50","nav51":"nav51"}


def read_csv(p):
    with Path(p).open("r",newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def find_csv(folder,key,summary):
    p=Path(str(summary.get("CSV","")))
    if p.is_file():return p
    if p.name and (folder/p.name).is_file():return folder/p.name
    nav=ROUTE_ALIAS.get(key,key);prefix="route_B" if nav=="nav50" else "route_C"
    m=sorted(folder.glob(prefix+"_*_frames.csv"))
    if not m:raise FileNotFoundError(f"{folder}: missing {nav} frames csv")
    return m[-1]


def metrics(rows):
    e=np.asarray([float(r["error_final_m"]) for r in rows],dtype=float)
    j=np.asarray([float(r.get("abnormal_jump",0) or 0) for r in rows],dtype=float)
    return {
        "MLE_m":float(e.mean()),
        "P90_m":float(np.percentile(e,90)),
        "LSR@5_pct":float(100*np.mean(e<=5)),
        "LSR@15_pct":float(100*np.mean(e<=15)),
        "JumpRate_pct":float(100*np.mean(j!=0)),
    }


def collect_seed(seed_root:Path,variant:str):
    rows=[]
    for city in CITIES:
        folder=seed_root/city/"variants"/variant
        sp=folder/"bearing_v39_summary.json"
        if not sp.is_file():raise FileNotFoundError(sp)
        summaries=json.loads(sp.read_text(encoding="utf-8"))
        for key,s in summaries.items():
            if key in ROUTE_ALIAS:rows.extend(read_csv(find_csv(folder,key,s)))
    return metrics(rows)


def fmt(v):return f"{v:.3f}" if isinstance(v,float) else str(v)


def md(rows):
    h=["Frames","MLE mean±std","P90 mean±std","LSR@5 mean±std","LSR@15 mean±std","Jump mean±std"]
    lines=["| "+" | ".join(h)+" |","| "+" | ".join(["---"]*len(h))+" |"]
    for r in rows:
        lines.append("| %d | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |"%(
            r["Frames"],r["MLE_mean"],r["MLE_std"],r["P90_mean"],r["P90_std"],
            r["LSR5_mean"],r["LSR5_std"],r["LSR15_mean"],r["LSR15_std"],r["Jump_mean"],r["Jump_std"]))
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output-dir");a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.output_dir).resolve() if a.output_dir else root/"aggregate";out.mkdir(parents=True,exist_ok=True)
    seed_dirs=sorted([d for d in root.glob("seed_*") if d.is_dir()])
    if len(seed_dirs)<2:raise RuntimeError("Need at least two completed seed_* directories")
    per_seed=[];summary=[]
    for variant,n in VARIANTS:
        vals=[]
        for d in seed_dirs:
            m=collect_seed(d,variant);seed=int(d.name.split("_",1)[1]);row={"Seed":seed,"Frames":n,"Variant":variant,**m};per_seed.append(row);vals.append(m)
        def stat(k):
            x=np.asarray([v[k] for v in vals],float);return float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.0
        mle,mles=stat("MLE_m");p90,p90s=stat("P90_m");l5,l5s=stat("LSR@5_pct");l15,l15s=stat("LSR@15_pct");j,js=stat("JumpRate_pct")
        summary.append({"Frames":n,"Variant":variant,"Seeds":len(vals),"MLE_mean":mle,"MLE_std":mles,"P90_mean":p90,"P90_std":p90s,"LSR5_mean":l5,"LSR5_std":l5s,"LSR15_mean":l15,"LSR15_std":l15s,"Jump_mean":j,"Jump_std":js})
    with (out/"temporal_per_seed.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
    with (out/"temporal_multiseed_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    payload={"root":str(root),"seeds":[int(d.name.split("_",1)[1]) for d in seed_dirs],"summary":summary,"selection_rule":"No held-out nav50/nav51 metric is used to tune or select a frame count.","interpretation":"Use mean±std. Do not claim 3-frame is best unless the multi-seed results support it."}
    (out/"temporal_multiseed.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    (out/"TEMPORAL_MULTI_SEED.md").write_text("# Temporal context multi-seed evaluation\n\n"+md(summary)+"\n\nNo held-out result was used to tune the ordering.\n",encoding="utf-8")
    print("[TEMPORAL MULTISEED DONE]",out)
    print(md(summary))

if __name__=="__main__":main()
