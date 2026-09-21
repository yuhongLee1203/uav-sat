#!/usr/bin/env python3
"""Summarize Bearing-Naver route CSVs as SR@20, SPL and NE."""
from __future__ import annotations
import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd


def hav(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    x = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371000 * 2 * math.atan2(math.sqrt(x), math.sqrt(max(0, 1-x)))


def path_len(points): return sum(hav(a, b) for a, b in zip(points, points[1:]))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',required=True); p.add_argument('--output-dir',required=True)
    a=p.parse_args(); root=Path(a.repo_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in sorted(root.glob('nav_*/*_uav_traj_records.csv')):
        d=pd.read_csv(f)
        if d.empty: continue
        m=re.search(r'nav_(34bc|36bc|37bc|38bc)(\d\d)_.*_d(2d|3d)', f.parent.name)
        if not m: continue
        actual=list(zip(d.cur_lon_real.astype(float),d.cur_lat_real.astype(float)))
        actual.append((float(d.iloc[-1].next_lon_real),float(d.iloc[-1].next_lat_real)))
        planned=list(zip(d.cur_lon_name.astype(float),d.cur_lat_name.astype(float)))
        planned.append((float(d.iloc[-1].next_lon_name),float(d.iloc[-1].next_lat_name)))
        ne=float(d.iloc[-1].distance_ep); success=ne<=20.0
        lp=path_len(planned); la=path_len(actual)
        rows.append({'City':{'34bc':'CityA','36bc':'CityB','37bc':'CityC','38bc':'CityD'}[m.group(1)],
                     'Route':m.group(2),'View':'Sat' if m.group(3)=='2d' else 'UAV',
                     'Success@20':int(success),'SPL':(lp/max(lp,la,1e-9) if success else 0.0),
                     'NE_m':ne,'PlannedPath_m':lp,'ActualPath_m':la,'Source':str(f)})
    if not rows: raise SystemExit('no completed Bearing-Naver route CSVs found')
    summary=[]
    for view in ('Sat','UAV'):
        for city in ('CityA','CityB','CityC','CityD','ALL'):
            rr=[r for r in rows if r['View']==view and (city=='ALL' or r['City']==city)]
            if rr: summary.append({'City':city,'View':view,'Routes':len(rr),
                'SR@20_pct':100*np.mean([r['Success@20'] for r in rr]),
                'SPL_pct':100*np.mean([r['SPL'] for r in rr]),'NE_m':np.mean([r['NE_m'] for r in rr])})
    for name,data in (('official_navigation_routes',rows),('official_navigation_summary',summary)):
        with (out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
        (out/f'{name}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    md=['# Official Bearing-Naver navigation','','| City | View | Routes | SR@20 | SPL | NE m |','|---|---|---:|---:|---:|---:|']
    for r in summary: md.append(f"| {r['City']} | {r['View']} | {r['Routes']} | {r['SR@20_pct']:.3f} | {r['SPL_pct']:.3f} | {r['NE_m']:.3f} |")
    (out/'OFFICIAL_NAVIGATION_TABLE.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('[OFFICIAL NAVIGATION TABLE]',out)


if __name__=='__main__': main()
