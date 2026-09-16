#!/usr/bin/env python3
"""Aggregate ours + all public route-adapted baselines on the same 8 routes."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Dict, List
import numpy as np

CITIES=("citya","cityb","cityc","cityd")
ROUTES=("test_01","test_02")
RETRIEVAL=("university1652","sues200","denseuav","gtauav")
DISPLAY={
 "university1652":"University-1652 route-adapted",
 "sues200":"SUES-200 route-adapted",
 "denseuav":"DenseUAV route-adapted",
 "gtauav":"GTA-UAV route-adapted",
}

def _read_csv(p):
    with Path(p).open("r",newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def _write_csv(path,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def _find_ours_csv(out,route,summary):
    p=Path(str(summary.get("CSV","")))
    if p.exists():return p
    q=out/p.name
    if q.exists():return q
    m=sorted(out.glob(f"{route}_*_frames.csv"))
    if not m:raise FileNotFoundError(f"ours csv {out}/{route}")
    return m[-1]

def _ours(groot,city,route):
    out=groot/city/"v39_output_bearing_adapted"
    s=json.loads((out/"bearing_v39_summary.json").read_text())[route]
    rows=_read_csv(_find_ours_csv(out,route,s))
    e=np.asarray([math.hypot(float(r["final_x"])-float(r["gt_x"]),float(r["final_y"])-float(r["gt_y"])) for r in rows])
    recall=None
    pp=out/"bearing_paper_metrics.json"
    if pp.exists():
        p=json.loads(pp.read_text())
        recall=p.get("routes",{}).get(route,{}).get("Recall@1_derived_same_quadrant_pct")
    return e,{
      "frames":len(e),"NativeRecall@1_pct":recall,
      "MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"P90_m":float(np.percentile(e,90)),
      "LSR@5_pct":float(100*np.mean(e<=5)),"LSR@10_pct":float(100*np.mean(e<=10)),
      "LSR@15_pct":float(100*np.mean(e<=15)),"LSR@20_pct":float(100*np.mean(e<=20)),
    }

def _metrics_from_errors(e):
    e=np.asarray(e,dtype=float)
    return {"frames":len(e),"MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"P90_m":float(np.percentile(e,90)),
      "LSR@5_pct":float(100*np.mean(e<=5)),"LSR@10_pct":float(100*np.mean(e<=10)),
      "LSR@15_pct":float(100*np.mean(e<=15)),"LSR@20_pct":float(100*np.mean(e<=20))}

def export(groot:Path,baseline_root:Path,official_root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    pools:Dict[str,List[float]]={}
    native_recall:Dict[str,List[tuple]]={}
    gallery_recall:Dict[str,List[tuple]]={}
    headings:Dict[str,List[float]]={}

    def add(city,route,method,source,protocol,metrics,errors,native=None,gallery=None,herrors=None):
        row={"city":city,"route":route,"method":method,"result_source":source,"protocol":protocol,
             "frames":metrics["frames"],"NativeRecall@1_pct":native if native is not None else "N/A",
             "RouteGalleryRecall@1_pct":gallery if gallery is not None else "N/A",
             "MLE_m":metrics["MLE_m"],"MedLE_m":metrics["MedLE_m"],"P90_m":metrics["P90_m"],
             "LSR@5_pct":metrics["LSR@5_pct"],"LSR@10_pct":metrics["LSR@10_pct"],
             "LSR@15_pct":metrics["LSR@15_pct"],"LSR@20_pct":metrics["LSR@20_pct"]}
        rows.append(row);pools.setdefault(method,[]).extend(map(float,errors))
        if native is not None:native_recall.setdefault(method,[]).append((float(native),int(metrics["frames"])))
        if gallery is not None:gallery_recall.setdefault(method,[]).append((float(gallery),int(metrics["frames"])))
        if herrors is not None:headings.setdefault(method,[]).extend(map(float,herrors))

    for city in CITIES:
        # Ours
        for route in ROUTES:
            e,m=_ours(groot,city,route)
            add(city,route,"Ours v39 Bearing-adapted","our rerun",
                "temporal controlled-local-prior refinement",m,e,m.get("NativeRecall@1_pct"))

        # Four route-adapted retrieval baselines
        for key in RETRIEVAL:
            p=baseline_root/key/city/"result.json"
            data=json.loads(p.read_text())
            for route in ROUTES:
                m=data["routes"][route]
                add(city,route,DISPLAY[key],"public official model/objective retrained on Route-A",
                    "route-adapted retrieval on common planned-route gallery",m,m["distance_errors_m"],
                    gallery=m.get("RouteGalleryRecall@1_pct"))

        # Bearing-UAV retrained on our Route-A
        data=json.loads((baseline_root/"bearinguav_route_adapted"/city/"result.json").read_text())
        for route in ROUTES:
            m=data["routes"][route]
            add(city,route,"Bearing-UAV route-adapted","official architecture/objective retrained on Route-A",
                "single-frame four-neighbour RST pose regression; Route-A-only training",m,m["distance_errors_m"],
                native=m.get("Recall@1_pct"),herrors=m.get("heading_errors_deg"))

        # Authors' released full-data checkpoint on same selected frames
        data=json.loads((official_root/city/"official_bearinguav_same_route.json").read_text())
        for route in ROUTES:
            m=data["routes"][route]
            add(city,route,"Bearing-UAV official pretrained VGG-16","authors' checkpoint rerun on same selected frames",
                "single-frame four-neighbour RST pose regression; pretrained on authors' full benchmark training split",
                m,m["distance_errors_m"],native=m.get("Recall@1_pct"),herrors=m.get("heading_errors_deg"))

    pooled=[]
    order=["Ours v39 Bearing-adapted"]+[DISPLAY[k] for k in RETRIEVAL]+["Bearing-UAV route-adapted","Bearing-UAV official pretrained VGG-16"]
    for method in order:
        m=_metrics_from_errors(pools[method])
        nr=native_recall.get(method,[]);gr=gallery_recall.get(method,[]);hh=np.asarray(headings.get(method,[]),dtype=float)
        row={"method":method,"evaluation":"same 4 cities / 8 selected test routes",
             "frames":m["frames"],"NativeRecall@1_pct":("N/A" if not nr else sum(v*n for v,n in nr)/sum(n for _,n in nr)),
             "RouteGalleryRecall@1_pct":("N/A" if not gr else sum(v*n for v,n in gr)/sum(n for _,n in gr)),
             **{k:v for k,v in m.items() if k!="frames"},
             "HSR@15_pct":("N/A" if hh.size==0 else float(100*np.mean(hh<=15))),
             "MHE_deg":("N/A" if hh.size==0 else float(hh.mean())),
             "MedHE_deg":("N/A" if hh.size==0 else float(np.median(hh)))}
        pooled.append(row)

    _write_csv(out/"all_methods_same_routes_route_level.csv",rows)
    _write_csv(out/"all_methods_same_routes_pooled.csv",pooled)
    payload={
      "route_level":rows,"pooled":pooled,
      "comparison_rule":"MLE/MedLE/P90/LSR use identical selected test frames and metre-space GT. Retrieval baselines share one route gallery built only from planned waypoints, never exact per-frame test GT.",
      "recall_warning":"RouteGalleryRecall@1 is not the papers' native identity Recall@1. Native Recall is reported only where the method has the corresponding native decision rule.",
      "protocol_warning":"Same data/GT improves fairness, but priors and output parameterizations still differ; report route-adapted values separately from published benchmark values.",
    }
    (out/"all_methods_same_routes.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("[ALL-METHODS-COMPARISON] PASS",flush=True)
    for r in pooled:print(f"  {r['method']}: MLE={r['MLE_m']:.3f}m MedLE={r['MedLE_m']:.3f}m LSR15={r['LSR@15_pct']:.2f}%",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--generated-root",required=True);p.add_argument("--baseline-root",required=True);p.add_argument("--official-result-root",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args()
    export(Path(a.generated_root).resolve(),Path(a.baseline_root).resolve(),Path(a.official_result_root).resolve(),Path(a.output_dir).resolve())
if __name__=="__main__":main()
