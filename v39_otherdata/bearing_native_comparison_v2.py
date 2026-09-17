#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

CITIES=("citya","cityb","cityc","cityd")
ROUTES=("test_01","test_02")
M2T=("university1652","sues200","denseuav","gtauav")

def J(p): return json.loads(Path(p).read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--generated-root',required=True);p.add_argument('--baseline-root',required=True);p.add_argument('--official-root',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
 gen=Path(a.generated_root);base=Path(a.baseline_root);off=Path(a.official_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for city in CITIES:
  q=J(gen/city/'v39_output_bearing_adapted'/'bearing_paper_metrics.json')
  for route in ROUTES:
   m=q['routes'][route];rows.append({'method':'Ours-v39','family':'temporal-local-refinement','city':city,'route':route,'frames':m['frames'],'Recall@1_pct':m.get('Recall@1_derived_same_quadrant_pct'),'MLE_m':m['MLE_m'],'MedLE_m':m['MedLE_m'],'P90_m':m['P90_m'],'LSR@5_pct':m['LSR@5_pct'],'LSR@10_pct':m['LSR@10_pct'],'LSR@15_pct':m['LSR@15_pct'],'LSR@20_pct':m['LSR@20_pct'],'HSR@15_pct':None,'MHE_deg':None,'note':'our controlled local-prior temporal refinement protocol'})
 for city in CITIES:
  q=J(off/city/'official_bearinguav_same_route.json')
  for route in ROUTES:
   m=q['routes'][route];rows.append({'method':'Bearing-UAV-official-pretrained','family':'four-RST-pose-regression','city':city,'route':route,'frames':m['frames'],'Recall@1_pct':m['Recall@1_pct'],'MLE_m':m['MLE_m'],'MedLE_m':m['MedLE_m'],'P90_m':m['P90_m'],'LSR@5_pct':m['LSR@5_pct'],'LSR@10_pct':m['LSR@10_pct'],'LSR@15_pct':m['LSR@15_pct'],'LSR@20_pct':m['LSR@20_pct'],'HSR@15_pct':m.get('HSR@15_pct'),'MHE_deg':m.get('MHE_deg'),'note':'authors released VGG-16 checkpoint; native p1/p2/p3/p4 pose regression; same selected route UAV frames'})
 for method in M2T:
  for city in CITIES:
   q=J(base/method/city/'result.json')
   if q.get('candidate_scope')!='four adjacent p1/p2/p3/p4 RSTs from official metadata': raise RuntimeError(f'{method}/{city}: stale wrong candidate protocol')
   for route in ROUTES:
    m=q['routes'][route];rows.append({'method':method,'family':'native-four-RST-matching-to-tile','city':city,'route':route,'frames':m['frames'],'Recall@1_pct':m['Recall@1_pct'],'MLE_m':m['MLE_m'],'MedLE_m':m['MedLE_m'],'P90_m':m['P90_m'],'LSR@5_pct':m['LSR@5_pct'],'LSR@10_pct':m['LSR@10_pct'],'LSR@15_pct':m['LSR@15_pct'],'LSR@20_pct':m['LSR@20_pct'],'HSR@15_pct':None,'MHE_deg':None,'note':'each UAV independently matches ONLY its official four adjacent RSTs; prediction is retrieved RST centre; no waypoint/temporal/local prior'})
 fields=list(rows[0].keys())
 with (out/'same_routes_route_level.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 pooled=[]
 for method in sorted(set(r['method'] for r in rows)):
  rr=[r for r in rows if r['method']==method];n=sum(int(r['frames']) for r in rr)
  def W(k):
   g=[r for r in rr if r.get(k) not in (None,'')]
   if not g:return None
   d=sum(int(r['frames']) for r in g);return sum(float(r[k])*int(r['frames']) for r in g)/d
  pooled.append({'method':method,'family':rr[0]['family'],'frames':n,'Recall@1_pct':W('Recall@1_pct'),'MLE_m':W('MLE_m'),'MedLE_route_weighted_m':W('MedLE_m'),'P90_route_weighted_m':W('P90_m'),'LSR@5_pct':W('LSR@5_pct'),'LSR@10_pct':W('LSR@10_pct'),'LSR@15_pct':W('LSR@15_pct'),'LSR@20_pct':W('LSR@20_pct'),'HSR@15_pct':W('HSR@15_pct'),'MHE_deg':W('MHE_deg'),'note':rr[0]['note']})
 with (out/'same_routes_pooled.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(pooled[0].keys()));w.writeheader();w.writerows(pooled)
 manifest={'candidate_protocol_for_M2T':'four adjacent RSTs p1/p2/p3/p4 from official metadata','paper_recall_definition':'retrieve the RST closest to UVP from the four adjacent RSTs','route_note':'the 8 routes are the official Bearing-UAV navigation routes; route results are not numerically identical to the full static localization benchmark','methods':[x['method'] for x in pooled]}
 (out/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('[COMPARE-V2] wrote',out/'same_routes_pooled.csv')
if __name__=='__main__':main()
