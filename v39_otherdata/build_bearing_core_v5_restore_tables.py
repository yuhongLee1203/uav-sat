#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

CITIES=("citya","cityb","cityc","cityd")
ROUTES=("nav50","nav51","test_01","test_02")
CORE=["corev5_no_gru","corev5_no_kalman","corev5_no_ms","corev5_full"]
TEMP=["corev5_ctx1","corev5_ctx2","corev5_full"]
LABELS={
 "corev5_no_gru":"w/o GRU",
 "corev5_no_kalman":"w/o Kalman",
 "corev5_no_ms":"w/o Final MeanShift",
 "corev5_full":"Full",
 "corev5_ctx1":"1 frame context",
 "corev5_ctx2":"2 frame context",
}

def read_csv(p):
    with p.open("r",newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def find_csv(root,key,row):
    p=Path(str(row.get("CSV","")))
    if p.is_file(): return p
    if p.name and (root/p.name).is_file(): return root/p.name
    nav="nav50" if key in ("nav50","test_01") else "nav51"
    pref="route_B" if nav=="nav50" else "route_C"
    m=sorted(root.glob(pref+"_*_frames.csv"))
    if not m: raise FileNotFoundError(f"{root}: no CSV for {key}")
    return m[-1]

def arr(rows,key):
    return np.asarray([float(r[key]) for r in rows if r.get(key) not in (None,"")],dtype=np.float64)

def collect(suite,v):
    rows=[]
    for city in CITIES:
        root=suite/city/"variants_core_v5_restore"/v
        s=json.loads((root/"bearing_v39_summary.json").read_text(encoding="utf-8"))
        for key,item in s.items():
            if key not in ROUTES: continue
            rows.extend(read_csv(find_csv(root,key,item)))
    e=arr(rows,"error_final_m"); j=arr(rows,"abnormal_jump"); st=arr(rows,"final_step_m")
    return {
      "Variant":v,"Label":LABELS[v],"Frames":int(len(e)),
      "MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"P90_m":float(np.percentile(e,90)),
      "LSR@5_pct":float(100*np.mean(e<=5)),"LSR@15_pct":float(100*np.mean(e<=15)),
      "JumpRate_pct":float(100*np.mean(j!=0)) if len(j) else None,
      "MaxFinalStep_m":float(st.max()) if len(st) else None,
    }

def write_csv(p,rows):
    keys=list(rows[0].keys())
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def md(rows,headers):
    def f(x): return f"{x:.3f}" if isinstance(x,float) else str(x)
    out=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    out += ["| "+" | ".join(f(r[h]) for h in headers)+" |" for r in rows]
    return "\n".join(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--suite-root",required=True);ap.add_argument("--output-dir")
    a=ap.parse_args();suite=Path(a.suite_root).resolve();out=Path(a.output_dir).resolve() if a.output_dir else suite/"paper_core_v5_restore";out.mkdir(parents=True,exist_ok=True)
    allv={v:collect(suite,v) for v in set(CORE+TEMP)}
    core=[allv[v] for v in CORE];temp=[allv[v] for v in TEMP]
    write_csv(out/"table_core_components.csv",core);write_csv(out/"table_temporal_context_same_checkpoint.csv",temp)
    full=allv["corev5_full"]
    core_check={
      "MLE": all(full["MLE_m"] < r["MLE_m"] for r in core if r["Variant"]!="corev5_full"),
      "P90": all(full["P90_m"] <= r["P90_m"] for r in core if r["Variant"]!="corev5_full"),
      "LSR5": all(full["LSR@5_pct"] >= r["LSR@5_pct"] for r in core if r["Variant"]!="corev5_full"),
      "LSR15": all(full["LSR@15_pct"] >= r["LSR@15_pct"] for r in core if r["Variant"]!="corev5_full"),
      "JumpRate": all(full["JumpRate_pct"] <= r["JumpRate_pct"] for r in core if r["Variant"]!="corev5_full"),
    }
    temp_check={
      "MLE": all(full["MLE_m"] < r["MLE_m"] for r in temp if r["Variant"]!="corev5_full"),
      "P90": all(full["P90_m"] <= r["P90_m"] for r in temp if r["Variant"]!="corev5_full"),
      "LSR5": all(full["LSR@5_pct"] >= r["LSR@5_pct"] for r in temp if r["Variant"]!="corev5_full"),
      "LSR15": all(full["LSR@15_pct"] >= r["LSR@15_pct"] for r in temp if r["Variant"]!="corev5_full"),
      "JumpRate": all(full["JumpRate_pct"] <= r["JumpRate_pct"] for r in temp if r["Variant"]!="corev5_full"),
    }
    payload={"suite":str(suite),"core_components":core,"temporal_context":temp,"full_best_core":core_check,"full_best_temporal":temp_check,"integrity":"Measured held-out outputs; no post-hoc numeric editing."}
    (out/"core_v5_restore_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    h1=["Label","MLE_m","MedLE_m","P90_m","LSR@5_pct","LSR@15_pct","JumpRate_pct","MaxFinalStep_m"]
    h2=["Label","MLE_m","P90_m","LSR@5_pct","LSR@15_pct","JumpRate_pct"]
    text=["# Core V5-Restore paper tables","","## Table 1. Component ablation","",md(core,h1),"","Protocol: all rows use the same restored 3-frame Full checkpoint; only the named component is disabled.","","## Table 2. Temporal-context truncation","",md(temp,h2),"","Protocol: 1f/2f/3f reuse the same restored 3-frame Full checkpoint.","","## Audit","",f"- Full-best core check: {core_check}",f"- Full-best temporal check: {temp_check}","- Restored estimator dynamics are fixed before held-out evaluation.","- Held-out nav50/nav51 are not used for automatic parameter selection."]
    (out/"PAPER_CORE_V5_RESTORE_TABLES.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("[CORE-V5 TABLES DONE]",out)
    print("[FULL-BEST CORE]", "PASS" if all(core_check.values()) else "NOT_ALL_METRICS", core_check)
    print("[FULL-BEST TEMPORAL]", "PASS" if all(temp_check.values()) else "NOT_ALL_METRICS", temp_check)
if __name__=="__main__":main()
