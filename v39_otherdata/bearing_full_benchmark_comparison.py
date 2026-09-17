#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
METHODS=('university1652','sues200','denseuav','gtauav')
CITIES=('citya','cityb','cityc','cityd'); ROUTES=('test_01','test_02')
def J(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=list(rows[0].keys())
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def pooled_route_from_payload(p):
 errs=[]; hits=0; n=0
 for c in CITIES:
  for r in ROUTES:
   m=p['route_results'][c][r]; errs.extend(m['distance_errors_m']); hits+=m['Recall@1_pct']*m['frames']/100.; n+=m['frames']
 e=np.asarray(errs,dtype=np.float64)
 return {'frames':int(len(e)),'Recall@1_pct':100*hits/max(n,1),'MLE_m':float(e.mean()),'MedLE_m':float(np.median(e)),'P90_m':float(np.percentile(e,90)),'LSR@15_pct':float(100*np.mean(e<=15))}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--generated-root',required=True); ap.add_argument('--baseline-root',required=True); ap.add_argument('--official-route-root',required=True); ap.add_argument('--official-paper-json',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
 gen=Path(a.generated_root); base=Path(a.baseline_root); off=Path(a.official_route_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 # Paper protocol reproduction table.
 rows=[]
 for m in METHODS:
  p=J(base/m/'full_benchmark_and_routes.json'); q=p['paper_benchmark']; pub=p['published_reference']
  rows.append({'method':m,'training':'official Bearing-UAV 85% split','test':'official Bearing-UAV 10% split','Recall@1_measured':q['Recall@1_pct'],'Recall@1_published':pub['Recall@1_pct'],'MLE_measured_m':q['MLE_m'],'MLE_published_m':pub['MLE_m'],'LSR15_measured':q['LSR@15_pct'],'LSR15_published':pub['LSR@15_pct'],'MLE_delta_m':q['MLE_m']-pub['MLE_m']})
 bp=J(a.official_paper_json); q=bp['measured']; pub=bp['published']
 rows.append({'method':'Bearing-UAV-official-VGG16','training':'authors released checkpoint','test':'official Bearing-UAV 10% split','Recall@1_measured':q['Recall@1_pct'],'Recall@1_published':pub['Recall@1_pct'],'MLE_measured_m':q['MLE_m'],'MLE_published_m':pub['MLE_m'],'LSR15_measured':q['LSR@15_pct'],'LSR15_published':pub['LSR@15_pct'],'MLE_delta_m':q['MLE_m']-pub['MLE_m']})
 write_csv(out/'paper_benchmark_reproduction.csv',rows)

 # Same 8 route frames: ours, official Bearing, then four full-split baselines.
 route_rows=[]
 ours_errors=[]; ours_hits=0; ours_n=0
 for c in CITIES:
  p=J(gen/c/'v39_output_bearing_adapted'/'bearing_paper_metrics.json')
  for r in ROUTES:
   m=p['routes'][r]; ours_n+=m['frames']; ours_hits+=m.get('Recall@1_derived_same_quadrant_pct',0)*m['frames']/100
   # per-frame raw errors are not stored in this file, so weighted route MLE is used for pooled MLE below.
   route_rows.append({'method':'Ours-v39','city':c,'route':r,'frames':m['frames'],'Recall@1_pct':m.get('Recall@1_derived_same_quadrant_pct'),'MLE_m':m['MLE_m'],'MedLE_m':m['MedLE_m'],'P90_m':m['P90_m'],'LSR@15_pct':m['LSR@15_pct'],'protocol':'temporal controlled-local-prior'})
 for c in CITIES:
  p=J(off/c/'official_bearinguav_same_route.json')
  for r in ROUTES:
   m=p['routes'][r]; route_rows.append({'method':'Bearing-UAV-official','city':c,'route':r,'frames':m['frames'],'Recall@1_pct':m['Recall@1_pct'],'MLE_m':m['MLE_m'],'MedLE_m':m['MedLE_m'],'P90_m':m['P90_m'],'LSR@15_pct':m['LSR@15_pct'],'protocol':'native four-RST pose regression'})
 for method in METHODS:
  p=J(base/method/'full_benchmark_and_routes.json')
  for c in CITIES:
   for r in ROUTES:
    m=p['route_results'][c][r]; route_rows.append({'method':method,'city':c,'route':r,'frames':m['frames'],'Recall@1_pct':m['Recall@1_pct'],'MLE_m':m['MLE_m'],'MedLE_m':m['MedLE_m'],'P90_m':m['P90_m'],'LSR@15_pct':m['LSR@15_pct'],'protocol':'full-split trained, native four-RST tile matching'})
 write_csv(out/'same_routes_route_level.csv',route_rows)
 pooled=[]
 for method in sorted(set(x['method'] for x in route_rows)):
  rr=[x for x in route_rows if x['method']==method]; n=sum(x['frames'] for x in rr)
  W=lambda k: sum(float(x[k])*x['frames'] for x in rr)/n
  pooled.append({'method':method,'frames':n,'Recall@1_pct':W('Recall@1_pct'),'MLE_m':W('MLE_m'),'MedLE_route_weighted_m':W('MedLE_m'),'P90_route_weighted_m':W('P90_m'),'LSR@15_pct':W('LSR@15_pct'),'protocol':rr[0]['protocol']})
 write_csv(out/'same_routes_pooled.csv',pooled)
 (out/'comparison_manifest.json').write_text(json.dumps({'paper_table':'paper_benchmark_reproduction.csv','route_table':'same_routes_pooled.csv','rule':'Paper reproduction and route experiment are separate. External baselines are trained on the full official 85% split and receive only their native four RST candidates; no v39 waypoint/temporal/local prior is injected.'},indent=2),encoding='utf-8')
 print('[COMPARE-FULL] wrote corrected paper + route tables',flush=True)
if __name__=='__main__': main()
