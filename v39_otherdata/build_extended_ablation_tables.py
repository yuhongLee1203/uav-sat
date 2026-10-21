#!/usr/bin/env python3
"""Build grid/search/decoder tables from a completed frozen V5 suite."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd

CITIES=('citya','cityb','cityc','cityd')

def files(root, variant):
    out=[]
    for c in CITIES:
        p=root/c/'variants'/variant
        out += sorted(p.glob('*_frames.csv'))
    if not out: raise FileNotFoundError(f'no CSV for {variant}')
    return out

def aggregate(root,variant):
    d=pd.concat([pd.read_csv(p) for p in files(root,variant)],ignore_index=True)
    e=d.error_final_m.to_numpy(float)
    return {'MLE_m':e.mean(),'P90_m':np.percentile(e,90),'LSR@5_pct':100*(e<=5).mean(),
            'Capture_pct':100*d.selected_candidate_capture.astype(float).mean(),
            'EndToEndLatency_ms':d.end_to_end_latency_ms.astype(float).mean(),
            'FinalMSLatency_ms':d.ms_latency_ms.astype(float).mean()}

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite-root',required=True);a=p.parse_args()
    root=Path(a.suite_root); out=root/'paper_bundle';out.mkdir(parents=True,exist_ok=True)
    grid=[]
    for n,v in ((4,'grid4'),(5,'grid5'),(6,'full'),(7,'grid7'),(8,'grid8')):
        grid.append({'Grid':f'{n}x{n}',**aggregate(root,v)})
    search=[{'Search':'Full 6x6','Candidates':36,**aggregate(root,'search_full6x6')},
            {'Search':'Forward 3x6','Candidates':18,**aggregate(root,'full')}]
    decoder=[{'Decoder':'Weighted','Candidates':36,**aggregate(root,'decoder_weighted')},
             {'Decoder':'MeanShift','Candidates':36,**aggregate(root,'search_full6x6')}]
    write(out/'table_grid_with_latency.csv',grid);write(out/'table_search_geometry.csv',search);write(out/'table_visual_decoder.csv',decoder)
    md=['# Extended frozen V5 ablations','','## Final MeanShift grid',
        '| Grid | MLE | P90 | LSR@5 | Final-MS latency ms |','|---|---:|---:|---:|---:|']
    for r in grid:md.append(f"| {r['Grid']} | {r['MLE_m']:.4f} | {r['P90_m']:.4f} | {r['LSR@5_pct']:.3f} | {r['FinalMSLatency_ms']:.4f} |")
    md += ['','## Search geometry','| Search | Candidates | MLE | P90 | Capture | End-to-end latency ms |','|---|---:|---:|---:|---:|---:|']
    for r in search:md.append(f"| {r['Search']} | {r['Candidates']} | {r['MLE_m']:.4f} | {r['P90_m']:.4f} | {r['Capture_pct']:.3f} | {r['EndToEndLatency_ms']:.4f} |")
    md += ['','## Visual decoder (accuracy; aggregation-only timing is in decoder_microbenchmark/)','| Decoder | MLE | P90 | LSR@5 |','|---|---:|---:|---:|']
    for r in decoder:md.append(f"| {r['Decoder']} | {r['MLE_m']:.4f} | {r['P90_m']:.4f} | {r['LSR@5_pct']:.3f} |")
    (out/'EXTENDED_ABLATIONS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (out/'extended_ablation_results.json').write_text(json.dumps({'grid':grid,'search':search,'decoder':decoder},indent=2),encoding='utf-8')
    print('[EXTENDED TABLES]',out)
if __name__=='__main__':main()
