#!/usr/bin/env python3
"""Aggregate ours, official Bearing-UAV, and four native M2T reproductions."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

CITIES=("citya","cityb","cityc","cityd")
ROUTES=("test_01","test_02")
M2T=("university1652","sues200","denseuav","gtauav")


def J(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def val(d,k):
    v=d.get(k)
    return "" if v is None else v


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--generated-root",required=True);ap.add_argument("--baseline-root",required=True);ap.add_argument("--official-root",required=True);ap.add_argument("--output-dir",required=True);a=ap.parse_args()
    gen=Path(a.generated_root);base=Path(a.baseline_root);off=Path(a.official_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Ours
    for city in CITIES:
        p=J(gen/city/"v39_output_bearing_adapted"/"bearing_paper_metrics.json")
        for route in ROUTES:
            m=p["routes"][route]
            rows.append({"method":"Ours-v39","family":"temporal-local-refinement","city":city,"route":route,"frames":m["frames"],"Recall@1_pct":m.get("Recall@1_derived_same_quadrant_pct"),"MLE_m":m["MLE_m"],"MedLE_m":m["MedLE_m"],"P90_m":m["P90_m"],"LSR@5_pct":m["LSR@5_pct"],"LSR@10_pct":m["LSR@10_pct"],"LSR@15_pct":m["LSR@15_pct"],"LSR@20_pct":m["LSR@20_pct"],"HSR@15_pct":None,"MHE_deg":None,"note":"controlled local prior + temporal refinement; Recall@1 is derived quadrant criterion"})
    # Official Bearing-UAV
    for city in CITIES:
        p=J(off/city/"official_bearinguav_same_route.json")
        for route in ROUTES:
            m=p["routes"][route]
            rows.append({"method":"Bearing-UAV-official-VGG16","family":"four-RST-pose-regression","city":city,"route":route,"frames":m["frames"],"Recall@1_pct":m.get("Recall@1_pct"),"MLE_m":m["MLE_m"],"MedLE_m":m["MedLE_m"],"P90_m":m["P90_m"],"LSR@5_pct":m["LSR@5_pct"],"LSR@10_pct":m["LSR@10_pct"],"LSR@15_pct":m["LSR@15_pct"],"LSR@20_pct":m["LSR@20_pct"],"HSR@15_pct":m.get("HSR@15_pct"),"MHE_deg":m.get("MHE_deg"),"note":"official pretrained model, same selected UAV frames"})
    # Native matching-to-tile reproductions
    for method in M2T:
        for city in CITIES:
            p=J(base/method/city/"result.json")
            for route in ROUTES:
                m=p["routes"][route]
                rows.append({"method":method,"family":"native-matching-to-tile","city":city,"route":route,"frames":m["frames"],"Recall@1_pct":m["Recall@1_pct"],"MLE_m":m["MLE_m"],"MedLE_m":m["MedLE_m"],"P90_m":m["P90_m"],"LSR@5_pct":m["LSR@5_pct"],"LSR@10_pct":m["LSR@10_pct"],"LSR@15_pct":m["LSR@15_pct"],"LSR@20_pct":m["LSR@20_pct"],"HSR@15_pct":None,"MHE_deg":None,"note":"no waypoint/local/temporal prior; independent full-city 16x16 RST retrieval"})
    fields=["method","family","city","route","frames","Recall@1_pct","MLE_m","MedLE_m","P90_m","LSR@5_pct","LSR@10_pct","LSR@15_pct","LSR@20_pct","HSR@15_pct","MHE_deg","note"]
    with (out/"same_frames_route_level.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # pooled using frame-weighted means for percentages/MLE and concatenation unavailable for Med/P90 -> route-frame weighted descriptive values
    pooled=[]
    for method in sorted(set(r["method"] for r in rows)):
        rr=[r for r in rows if r["method"]==method];n=sum(int(r["frames"]) for r in rr)
        def W(k):
            good=[r for r in rr if r.get(k) not in (None,"")]
            if not good:return None
            den=sum(int(r["frames"]) for r in good)
            return sum(float(r[k])*int(r["frames"]) for r in good)/den
        pooled.append({"method":method,"family":rr[0]["family"],"frames":n,"Recall@1_pct":W("Recall@1_pct"),"MLE_m":W("MLE_m"),"MedLE_route_weighted_m":W("MedLE_m"),"P90_route_weighted_m":W("P90_m"),"LSR@5_pct":W("LSR@5_pct"),"LSR@10_pct":W("LSR@10_pct"),"LSR@15_pct":W("LSR@15_pct"),"LSR@20_pct":W("LSR@20_pct"),"HSR@15_pct":W("HSR@15_pct"),"MHE_deg":W("MHE_deg"),"note":rr[0]["note"]})
    pf=list(pooled[0].keys())
    with (out/"same_frames_pooled.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=pf);w.writeheader();w.writerows(pooled)
    manifest={"methods":[p["method"] for p in pooled],"cities":list(CITIES),"test_routes_per_city":2,"comparison_rule":"All rows use the same selected test UAV frames and GT metrics. M2T baselines search the entire 16x16 city RST gallery independently per frame and receive no route/waypoint/temporal prior. Ours and Bearing-UAV retain their native priors/protocols, so method-family differences must still be stated.","published_numbers":"kept separately by bearing_published_reference.py; not mixed with rerun rows"}
    (out/"comparison_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"[COMPARE] wrote {out/'same_frames_route_level.csv'}")
    print(f"[COMPARE] wrote {out/'same_frames_pooled.csv'}")
if __name__=="__main__":main()
