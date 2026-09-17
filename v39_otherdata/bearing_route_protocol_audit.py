#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

# total_length copied from the official Bearing-UAV waypoint JSON files.  These
# are the paper/navigation lengths; they are not identical to simply summing
# pixel distances * 0.25 because the official files derive distance in the
# geographic coordinate system.
EXPECTED={
 "citya":{"test_01":(771.721426809399,13),"test_02":(1119.7335563431486,10)},
 "cityb":{"test_01":(644.3947002838131,13),"test_02":(757.4896581660938,11)},
 "cityc":{"test_01":(524.3000707227156,13),"test_02":(821.7919470879535,11)},
 "cityd":{"test_01":(721.1506052477557,13),"test_02":(1115.449659747041,11)},
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--generated-root',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 gen=Path(a.generated_root);cache=Path(a.cache_root)
 report={"paper_navigation_step_m":25.0,"paper_waypoint_arrival_threshold_m":20.0,"official_route_length_range_m":[524.3000707227156,1119.7335563431486],"length_source":"official Bearing-UAV waypoint JSON total_length","cities":{}}
 for city,rr in EXPECTED.items():
  report['cities'][city]={};cm=json.loads((cache/city/'cache_meta.json').read_text())
  if not cm.get('candidate_policy','').startswith('exact p1/p2/p3/p4'):raise RuntimeError(f'{city}: not native four-RST cache')
  for route,(official_len,expected_wp) in rr.items():
   wp=json.loads((gen/city/'routes'/route/'waypoints.json').read_text())['waypoints']
   pts=np.asarray([[float(x['pixel_x']),float(x['pixel_y'])] for x in sorted(wp,key=lambda z:int(z['waypoint_order']))])
   if len(pts)!=expected_wp:raise RuntimeError(f'{city}/{route}: waypoint count {len(pts)} != official {expected_wp}')
   map_plane=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*0.25)
   oracle=float(cm['routes'][route]['oracle_nearest_RST_center_MLE_m'])
   if oracle>50.0:raise RuntimeError(f'{city}/{route}: four-RST geometry looks wrong; oracle tile-center MLE={oracle:.2f}m')
   report['cities'][city][route]={"official_total_length_m":official_len,"map_plane_0p25mpp_polyline_m":map_plane,"waypoints":len(pts),"nearest_RST_center_oracle_MLE_m":oracle}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
 print('[ROUTE-PROTOCOL-AUDIT] PASS: official route definitions/counts present for all 8 routes')
 print('[ROUTE-PROTOCOL-AUDIT] official lengths: 524.30..1119.73m; paper step=25m; arrival threshold=20m')
 print('[ROUTE-PROTOCOL-AUDIT] four-RST candidate geometry: PASS')
if __name__=='__main__':main()
