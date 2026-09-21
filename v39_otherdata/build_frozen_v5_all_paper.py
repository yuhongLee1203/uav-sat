#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd

PATCH_SIZE=256.0
OFFSET_SCALE=128.0
MPP=0.25
ROUTE_MAP={"test_01":"test_01","test_02":"test_02"}
VARIANTS=("full","no_gru","no_kalman","no_ms","frames1","frames2","grid4","grid5","grid7","grid8")
CITIES=("citya","cityb","cityc","cityd")
RSI_IDS={"citya":"34bc","cityb":"36bc","cityc":"37bc","cityd":"38bc"}
LITERATURE=[
 {"Method":"University-1652","Recall@1_UAV_pct":60.20,"LSR@15_UAV_pct":15.11,"HSR@15_UAV_pct":None,"MLE_UAV_m":33.15,"MedLE_UAV_m":None,"MHE_UAV_deg":None,"MedHE_UAV_deg":None,"SR@20_UAV_pct":0.0,"SPL_UAV_pct":None,"NE_UAV_m":602.96},
 {"Method":"SUES-200","Recall@1_UAV_pct":66.60,"LSR@15_UAV_pct":15.76,"HSR@15_UAV_pct":None,"MLE_UAV_m":30.83,"MedLE_UAV_m":None,"MHE_UAV_deg":None,"MedHE_UAV_deg":None,"SR@20_UAV_pct":0.0,"SPL_UAV_pct":None,"NE_UAV_m":618.85},
 {"Method":"DenseUAV","Recall@1_UAV_pct":73.43,"LSR@15_UAV_pct":16.54,"HSR@15_UAV_pct":None,"MLE_UAV_m":28.79,"MedLE_UAV_m":None,"MHE_UAV_deg":None,"MedHE_UAV_deg":None,"SR@20_UAV_pct":0.0,"SPL_UAV_pct":None,"NE_UAV_m":651.93},
 {"Method":"GTA-UAV","Recall@1_UAV_pct":70.71,"LSR@15_UAV_pct":27.96,"HSR@15_UAV_pct":None,"MLE_UAV_m":28.43,"MedLE_UAV_m":None,"MHE_UAV_deg":None,"MedHE_UAV_deg":None,"SR@20_UAV_pct":0.0,"SPL_UAV_pct":None,"NE_UAV_m":661.91},
 {"Method":"Bearing-UAV (VGG-16)","Recall@1_UAV_pct":83.17,"LSR@15_UAV_pct":89.36,"HSR@15_UAV_pct":77.21,"MLE_UAV_m":8.61,"MedLE_UAV_m":7.30,"MHE_UAV_deg":12.90,"MedHE_UAV_deg":7.20,"SR@20_UAV_pct":50.0,"SPL_UAV_pct":None,"NE_UAV_m":275.61},
]

def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def result_csv(vroot,route):
    patt='route_B_*_frames.csv' if route=='test_01' else 'route_C_*_frames.csv'
    xs=sorted(Path(vroot).glob(patt)) or sorted(Path(vroot).glob(f'{route}_*_frames.csv'))
    if not xs: raise FileNotFoundError(f'no frame csv: {vroot} {route}')
    return xs[-1]

def city_metadata(exp):
    df=pd.read_csv(exp['metadata_csv'])
    for col in ('city','City','city_name'):
        if col in df.columns:
            s=df[col].astype(str).str.lower(); sub=df[s==str(exp['city']).lower()]
            if len(sub): return sub.reset_index(drop=True)
    city=str(exp['city']).lower(); paths=df['target_path'].astype(str).str.replace('\\','/',regex=False)
    sub=df[paths.str.contains(f'/{city}/',regex=False)|paths.str.contains(RSI_IDS[city],regex=False)]
    if not len(sub): raise RuntimeError(f'no metadata rows for {city}')
    return sub.reset_index(drop=True)

def recall_same_quadrant(rows,manifest,meta,origin):
    good=0
    for r,m in zip(rows,manifest):
        idx=int(m['source_index']); q=meta.iloc[idx]
        cx=float(q['block_x'])*PATCH_SIZE+PATCH_SIZE; cy=float(q['block_y'])*PATCH_SIZE+PATCH_SIZE
        px=((float(r['final_x'])+origin[0])/MPP-cx)/OFFSET_SCALE
        py=((float(r['final_y'])+origin[1])/MPP-cy)/OFFSET_SCALE
        gt=np.asarray([float(q['x_norm']),float(q['y_norm'])]); pred=np.asarray([px,py])
        good+=int(np.array_equal(np.sign(pred),np.sign(gt)))
    return 100.0*good/max(len(rows),1)

def metrics_from_rows(rows):
    e=np.asarray([math.hypot(float(r['final_x'])-float(r['gt_x']),float(r['final_y'])-float(r['gt_y'])) for r in rows])
    out={"Frames":len(rows),"MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"P90_m":float(np.percentile(e,90)),"P95_m":float(np.percentile(e,95)),"P99_m":float(np.percentile(e,99)),"LSR@5_pct":100*float(np.mean(e<=5)),"LSR@10_pct":100*float(np.mean(e<=10)),"LSR@15_pct":100*float(np.mean(e<=15)),"LSR@20_pct":100*float(np.mean(e<=20))}
    if rows and 'heading_error_deg' in rows[0] and rows[0].get('heading_error_deg','')!='':
        h=np.asarray([abs(float(r['heading_error_deg'])) for r in rows if r.get('heading_error_deg','')!=''])
        if len(h): out.update({"Motion_MHE_deg":float(h.mean()),"Motion_MedHE_deg":float(np.median(h)),"Motion_HSR@15_pct":100*float(np.mean(h<=15))})
    return out

def replay_nav(rows):
    gt=np.asarray([[float(r['gt_x']),float(r['gt_y'])] for r in rows]); pr=np.asarray([[float(r['final_x']),float(r['final_y'])] for r in rows])
    ne=float(np.linalg.norm(pr[-1]-gt[-1])); success=ne<=20.0
    gtl=float(np.linalg.norm(np.diff(gt,axis=0),axis=1).sum()); prl=float(np.linalg.norm(np.diff(pr,axis=0),axis=1).sum())
    spl=100.0*(gtl/max(gtl,prl,1e-9)) if success else 0.0
    return {"NE_m_route_replay":ne,"SR@20_pct_route_replay":100.0 if success else 0.0,"SPL_pct_route_replay":spl,"GTPath_m":gtl,"PredPath_m":prl}

def write_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def fmt(v,n=3):
    return '—' if v is None else f'{v:.{n}f}' if isinstance(v,(int,float,np.floating)) else str(v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite-root',required=True);a=ap.parse_args()
    root=Path(a.suite_root).resolve(); out=root/'paper_bundle';out.mkdir(parents=True,exist_ok=True)
    per_city=[]; route_rows=[]; nav_rows=[]; all_full_rows=[]; ablation_rows=[]
    for city in CITIES:
        croot=root/city; prep=croot/'prepared'; exp=json.loads((prep/'experiment.json').read_text()); meta=city_metadata(exp)
        train=read_csv(prep/'routes/train_01/manifest.csv'); origin=(float(train[0]['x_m']),float(train[0]['y_m']))
        city_full=[]; recall_good=0.0; total=0
        for route in ROUTE_MAP:
            rr=read_csv(result_csv(croot/'variants/full',route)); man=read_csv(prep/f'routes/{route}/manifest.csv')
            m=metrics_from_rows(rr); rec=recall_same_quadrant(rr,man,meta,origin); m['Recall@1_4RST_derived_pct']=rec
            route_rows.append({'City':city,'Route':ROUTE_MAP[route],**m}); nav_rows.append({'City':city,'Route':ROUTE_MAP[route],**replay_nav(rr)})
            city_full+=rr; all_full_rows+=rr; recall_good+=rec*len(rr)/100.0; total+=len(rr)
        cm=metrics_from_rows(city_full);cm['Recall@1_4RST_derived_pct']=100*recall_good/max(total,1);per_city.append({'City':city,**cm})
        for variant in VARIANTS:
            vr=[]
            for route in ROUTE_MAP: vr+=read_csv(result_csv(croot/f'variants/{variant}',route))
            ablation_rows.append({'City':city,'Variant':variant,**metrics_from_rows(vr)})
    overall=metrics_from_rows(all_full_rows)
    overall['Recall@1_4RST_derived_pct']=sum(x['Recall@1_4RST_derived_pct']*x['Frames'] for x in per_city)/sum(x['Frames'] for x in per_city)
    per_city.append({'City':'ALL_4_CITIES',**overall})
    ours={"Method":"Yours (Frozen V5 Forward18 + 3f GRU + Kalman + FinalMS)","Recall@1_UAV_pct":overall['Recall@1_4RST_derived_pct'],"LSR@15_UAV_pct":overall['LSR@15_pct'],"HSR@15_UAV_pct":None,"MLE_UAV_m":overall['MLE_m'],"MedLE_UAV_m":overall['MedLE_m'],"MHE_UAV_deg":None,"MedHE_UAV_deg":None,"SR@20_UAV_pct":None,"SPL_UAV_pct":None,"NE_UAV_m":None}
    comparison=LITERATURE+[ours]
    # pooled ablations across all cities
    pooled=[]
    for v in VARIANTS:
        rr=[]
        for city in CITIES:
            for route in ROUTE_MAP: rr+=read_csv(result_csv(root/city/f'variants/{v}',route))
        pooled.append({'Variant':v,**metrics_from_rows(rr)})
    core=[r for r in pooled if r['Variant'] in ('no_gru','no_kalman','no_ms','full')]
    temporal=[r for r in pooled if r['Variant'] in ('frames1','frames2','full')]
    grid=[r for r in pooled if r['Variant'] in ('grid4','grid5','full','grid7','grid8')]
    write_csv(out/'table_main_per_city.csv',per_city);write_csv(out/'table_route_metrics.csv',route_rows);write_csv(out/'table_bearinguav_comparison.csv',comparison);write_csv(out/'table_motion_heading_diagnostic.csv',[{k:v for k,v in r.items() if k in ('City','Route','Motion_MHE_deg','Motion_MedHE_deg','Motion_HSR@15_pct')} for r in route_rows]);write_csv(out/'table_route_replay_navigation.csv',nav_rows);write_csv(out/'table_ablation_all_cities.csv',ablation_rows);write_csv(out/'table_ablation_core_pooled.csv',core);write_csv(out/'table_temporal_context_pooled.csv',temporal);write_csv(out/'table_grid_pooled.csv',grid)
    payload={'protocol':'Frozen V5 controlled_gt_jitter local-prior sequential evaluation','important_notes':{'Recall@1':'4-RST same-quadrant decision derived from continuous final position; not a native retrieval head.','Heading':'Current tracker heading is motion/route heading, not Bearing-UAV camera yaw; official HSR/MHE/MedHE remain N/A for Ours.','Navigation':'test_01/test_02 route replay is not Bearing-Naver closed-loop; SR/SPL/NE replay diagnostics are kept separate.','Comparison':'MLE/MedLE/LSR formulas align, but protocol differs from Bearing-UAV four-RST pose regression.'},'overall':overall,'per_city':per_city,'literature':comparison}
    (out/'paper_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    md=['# Frozen V5 paper tables','','## Main localization by city','', '| City | Recall@1* | MLE m | MedLE m | P90 m | LSR@5 | LSR@15 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in per_city: md.append(f"| {r['City']} | {fmt(r.get('Recall@1_4RST_derived_pct'))} | {fmt(r['MLE_m'])} | {fmt(r['MedLE_m'])} | {fmt(r['P90_m'])} | {fmt(r['LSR@5_pct'])} | {fmt(r['LSR@15_pct'])} |")
    md += ['','*Recall@1 is derived with the Bearing-UAV four-RST same-quadrant decision criterion from continuous localization output.','','## Bearing-UAV-aligned comparison','','| Method | Recall@1 | LSR@15 | HSR@15 | MLE m | MedLE m | MHE deg | MedHE deg | SR@20 | SPL | NE m |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison: md.append('| '+ ' | '.join([str(r['Method']),fmt(r.get('Recall@1_UAV_pct')),fmt(r.get('LSR@15_UAV_pct')),fmt(r.get('HSR@15_UAV_pct')),fmt(r.get('MLE_UAV_m')),fmt(r.get('MedLE_UAV_m')),fmt(r.get('MHE_UAV_deg')),fmt(r.get('MedHE_UAV_deg')),fmt(r.get('SR@20_UAV_pct')),fmt(r.get('SPL_UAV_pct')),fmt(r.get('NE_UAV_m'))])+' |')
    md += ['','**Protocol:** Ours uses controlled local-prior temporal refinement. Bearing-UAV uses four-adjacent-RST pose regression; navigation values are closed-loop Bearing-Naver. Therefore Ours camera-heading and closed-loop navigation cells are intentionally not fabricated.','', '## Core ablation pooled across four cities','', '| Variant | MLE m | P90 m | LSR@5 | LSR@15 |','|---|---:|---:|---:|---:|']
    for r in core: md.append(f"| {r['Variant']} | {fmt(r['MLE_m'])} | {fmt(r['P90_m'])} | {fmt(r['LSR@5_pct'])} | {fmt(r['LSR@15_pct'])} |")
    md += ['','## Temporal ablation pooled across four cities','', '| Variant | MLE m | P90 m | LSR@5 | LSR@15 |','|---|---:|---:|---:|---:|']
    for r in temporal: md.append(f"| {r['Variant']} | {fmt(r['MLE_m'])} | {fmt(r['P90_m'])} | {fmt(r['LSR@5_pct'])} | {fmt(r['LSR@15_pct'])} |")
    md += ['','## Output files','', '- `table_bearinguav_comparison.csv`: paper comparison table','- `table_main_per_city.csv`: City A/B/C/D main results','- `table_ablation_core_pooled.csv`: GRU/Kalman/MeanShift/Full','- `table_temporal_context_pooled.csv`: 1f/2f/3f','- `table_grid_pooled.csv`: final MeanShift window','- `table_motion_heading_diagnostic.csv`: motion-heading only','- `table_route_replay_navigation.csv`: offline replay diagnostic only']
    (out/'PAPER_TABLES.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('[PAPER BUNDLE DONE]',out); print(json.dumps(overall,indent=2))
if __name__=='__main__':main()
