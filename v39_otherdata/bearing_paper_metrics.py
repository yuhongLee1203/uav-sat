#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

CITIES=('citya','cityb','cityc','cityd')
LITERATURE=[
 {'Method':'University-1652','Recall@1_UAV_pct':60.20,'LSR@15_UAV_pct':15.11,'MLE_UAV_m':33.15,'MHE_UAV_deg':None,'SR@20_UAV_pct':0.0,'SPL_UAV_pct':None,'NE_UAV_m':602.96},
 {'Method':'SUES-200','Recall@1_UAV_pct':66.60,'LSR@15_UAV_pct':15.76,'MLE_UAV_m':30.83,'MHE_UAV_deg':None,'SR@20_UAV_pct':0.0,'SPL_UAV_pct':None,'NE_UAV_m':618.85},
 {'Method':'DenseUAV','Recall@1_UAV_pct':73.43,'LSR@15_UAV_pct':16.54,'MLE_UAV_m':28.79,'MHE_UAV_deg':None,'SR@20_UAV_pct':0.0,'SPL_UAV_pct':None,'NE_UAV_m':651.93},
 {'Method':'GTA-UAV','Recall@1_UAV_pct':70.71,'LSR@15_UAV_pct':27.96,'MLE_UAV_m':28.43,'MHE_UAV_deg':None,'SR@20_UAV_pct':0.0,'SPL_UAV_pct':None,'NE_UAV_m':661.91},
 {'Method':'Bearing-UAV (VGG-16)','Recall@1_UAV_pct':83.17,'LSR@15_UAV_pct':89.36,'MLE_UAV_m':8.61,'MHE_UAV_deg':12.90,'SR@20_UAV_pct':50.0,'SPL_UAV_pct':None,'NE_UAV_m':275.61},
]

def q(a,p): return float(np.percentile(np.asarray(a,float),p))
def plen(x,y): return float(np.hypot(np.diff(x),np.diff(y)).sum())
def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def find_csv(full,nav):
    prefix='route_B' if nav=='nav50' else 'route_C'
    m=sorted(full.glob(prefix+'_*_frames.csv'))
    if not m: raise FileNotFoundError(f'{full}: {nav} frames csv missing')
    return m[-1]
def metrics(rows):
    err=np.asarray([float(r['error_final_m']) for r in rows])
    he=np.asarray([abs(float(r['heading_error_deg'])) for r in rows if r.get('heading_error_deg','')!=''])
    gx=np.asarray([float(r['gt_x']) for r in rows]); gy=np.asarray([float(r['gt_y']) for r in rows])
    px=np.asarray([float(r['final_x']) for r in rows]); py=np.asarray([float(r['final_y']) for r in rows])
    lat=np.asarray([float(r['end_to_end_latency_ms']) for r in rows if r.get('end_to_end_latency_ms','')!=''])
    ne=float(math.hypot(px[-1]-gx[-1],py[-1]-gy[-1])); sr=float(ne<=20.0)
    gl=plen(gx,gy); pl=plen(px,py); spl=sr*gl/max(gl,pl,1e-9)
    return {
      'Frames':len(rows),'MLE_m':float(err.mean()),'MedLE_m':float(np.median(err)),
      'P90_m':q(err,90),'P95_m':q(err,95),'P99_m':q(err,99),
      'LSR@5_pct':float((err<=5).mean()*100),'LSR@10_pct':float((err<=10).mean()*100),
      'LSR@15_pct':float((err<=15).mean()*100),'LSR@20_pct':float((err<=20).mean()*100),
      'MHE_deg':float(he.mean()) if len(he) else None,'MedHE_deg':float(np.median(he)) if len(he) else None,
      'HSR@15_pct':float((he<=15).mean()*100) if len(he) else None,
      'NE_m_route_replay':ne,'SR@20_pct_route_replay':100*sr,'SPL_pct_route_replay':100*spl,
      'GTPath_m':gl,'PredPath_m':pl,'JumpRate_pct':100*sum(int(float(r.get('abnormal_jump','0') or 0))!=0 for r in rows)/len(rows),
      'MaxFinalStep_m':max(float(r['final_step_m']) for r in rows),
      'InferenceMean_ms':float(lat.mean()) if len(lat) else None,'FPS':float(1000/lat.mean()) if len(lat) and lat.mean()>0 else None,
    }
def md(headers,rows):
    def f(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.3f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(f(r.get(h)) for h in headers)+' |' for r in rows])
def write_csv(p,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    a=argparse.ArgumentParser();a.add_argument('--suite-root',required=True);a.add_argument('--output-dir');x=a.parse_args()
    root=Path(x.suite_root).resolve();out=Path(x.output_dir or root/'paper_benchmark');out.mkdir(parents=True,exist_ok=True)
    routes=[]; pooled=[]
    for city in CITIES:
        full=root/city/'variants'/'full'
        if not (full/'bearing_v39_summary.json').is_file(): raise RuntimeError(f'missing summary: {city}')
        for nav in ('nav50','nav51'):
            cp=find_csv(full,nav); rows=read_csv(cp); pooled+=rows
            r={'City':city,'Route':nav,**metrics(rows),'CSV':str(cp)};routes.append(r)
    pm=metrics(pooled)
    ours={'Method':'Yours (Forward-18 + GRU + Kalman + SoftMS)','Recall@1_UAV_pct':None,'LSR@15_UAV_pct':pm['LSR@15_pct'],'MLE_UAV_m':pm['MLE_m'],'MedLE_UAV_m':pm['MedLE_m'],'MHE_UAV_deg':pm['MHE_deg'],'MedHE_UAV_deg':pm['MedHE_deg'],'HSR@15_UAV_pct':pm['HSR@15_pct'],'SR@20_UAV_pct':float(np.mean([r['SR@20_pct_route_replay'] for r in routes])),'SPL_UAV_pct':float(np.mean([r['SPL_pct_route_replay'] for r in routes])),'NE_UAV_m':float(np.mean([r['NE_m_route_replay'] for r in routes]))}
    cities=[]
    for c in CITIES:
        rr=[r for r in routes if r['City']==c]
        cities.append({'City':c,'MLE_m':float(np.mean([r['MLE_m'] for r in rr])),'MedLE_m':float(np.mean([r['MedLE_m'] for r in rr])),'LSR@15_pct':float(np.mean([r['LSR@15_pct'] for r in rr])),'MHE_deg':float(np.mean([r['MHE_deg'] for r in rr])),'MedHE_deg':float(np.mean([r['MedHE_deg'] for r in rr])),'HSR@15_pct':float(np.mean([r['HSR@15_pct'] for r in rr]))})
    comparison=LITERATURE+[ours]
    payload={'suite':str(root),'ours_main':ours,'per_route':routes,'per_city':cities,'literature_comparison':comparison,'fairness':{'localization_heading':'MLE/MedLE/LSR/heading are computed from raw frame predictions. Heading uses the model recurrent heading output already logged as heading_error_deg.','Recall@1':'NOT filled for ours because Bearing-UAV Recall@1 is a four-adjacent-RST retrieval decision; Forward-18 top-1 is not substituted.','navigation':'SR@20/SPL/NE are exported as route-replay diagnostics only. They are NOT claimed as Bearing-Naver closed-loop equivalents.'}}
    (out/'bearing_paper_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    write_csv(out/'table_route_metrics.csv',routes);write_csv(out/'table_city_metrics.csv',cities);write_csv(out/'table_literature_comparison.csv',comparison)
    lines=['# Bearing-UAV aligned paper tables','','## A. Localization + heading','',md(['Method','MLE_UAV_m','MedLE_UAV_m','LSR@15_UAV_pct','MHE_UAV_deg','MedHE_UAV_deg','HSR@15_UAV_pct'],[ours]),'','## B. Multi-city results','',md(['City','MLE_m','MedLE_m','LSR@15_pct','MHE_deg','MedHE_deg','HSR@15_pct'],cities),'','## C. Route-level navigation diagnostics','',md(['City','Route','NE_m_route_replay','SR@20_pct_route_replay','SPL_pct_route_replay','JumpRate_pct','MaxFinalStep_m'],routes),'','## D. Literature comparison','',md(['Method','Recall@1_UAV_pct','LSR@15_UAV_pct','MLE_UAV_m','MHE_UAV_deg','SR@20_UAV_pct','SPL_UAV_pct','NE_UAV_m'],comparison),'','## Protocol notes','','- Do not fill our Recall@1 with Forward-18 top-1. Bearing-UAV Recall@1 uses four adjacent RSTs.','- SR@20/SPL/NE here are route-replay diagnostics. They are not Bearing-Naver closed-loop results.','- MLE/MedLE/LSR@15/MHE/MedHE/HSR@15 are calculated directly from raw per-frame outputs.']
    (out/'PAPER_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[PAPER BENCHMARK DONE]',out);print(json.dumps(ours,indent=2))
if __name__=='__main__':main()
