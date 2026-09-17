#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

EXPECTED={
 "citya":{"test_01":771.721426809399,"test_02":1119.7335563431486},
 "cityb":{"test_01":644.3947002838131,"test_02":757.4896581660938},
 "cityc":{"test_01":524.3000707227156,"test_02":821.7919470879535},
 "cityd":{"test_01":721.1506052477557,"test_02":1115.449659747041},
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--generated-root',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 gen=Path(a.generated_root);cache=Path(a.cache_root);report={"paper_navigation_step_m":25.0,"paper_waypoint_arrival_threshold_m":20.0,"paper_route_length_range_m":[524.0,1119.0],"cities":{}}
 for city,rr in EXPECTED.items():
  report["cities"][city]={}
  cm=json.loads((cache/city/'cache_meta.json').read_text())
  if not cm.get('candidate_policy','').startswith('exact p1/p2/p3/p4'):
   raise RuntimeError(f'{city}: not native four-RST cache')
  for route,expected in rr.items():
   wp=json.loads((gen/city/'routes'/route/'waypoints.json').read_text())['waypoints']
   pts=np.asarray([[float(x['pixel_x']),float(x['pixel_y'])] for x in sorted(wp,key=lambda z:int(z['waypoint_order']))])
   length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*0.25)
   delta=abs(length-expected)
   oracle=float(cm['routes'][route]['oracle_nearest_RST_center_MLE_m'])
   if delta>0.1: raise RuntimeError(f'{city}/{route}: route length mismatch {length:.3f} vs official {expected:.3f}')
   if oracle>50.0: raise RuntimeError(f'{city}/{route}: four-RST geometry looks wrong; oracle tile-center MLE={oracle:.2f}m')
   report['cities'][city][route]={"computed_length_m":length,"official_length_m":expected,"length_delta_m":delta,"waypoints":len(pts),"nearest_RST_center_oracle_MLE_m":oracle}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
 print('[ROUTE-PROTOCOL-AUDIT] PASS: 8/8 routes exactly match official Bearing-UAV waypoint lengths')
 print('[ROUTE-PROTOCOL-AUDIT] paper navigation: step=25m, arrival threshold=20m, routes=524..1119m')
 print('[ROUTE-PROTOCOL-AUDIT] four-RST candidate geometry: PASS')
if __name__=='__main__':main()
