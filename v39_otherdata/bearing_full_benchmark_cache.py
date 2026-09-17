#!/usr/bin/env python3
"""Build one shared low-CPU cache for paper-scale baseline reproduction.

The old baseline experiment trained each method only on the selected train_01
route.  That made all four methods nearly random among four RST candidates and
artificially clustered their MLE around 60 m.  This cache instead follows the
released Bearing-UAV split: full metadata 85/5/10 with seed 42.

Large uint8 arrays stay local under generated/ and are git-ignored.  Source JPEGs
are decoded once; all baseline methods mmap the same arrays afterwards.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import bearing_prepare as bearing
from bearinguav_official_route_eval import _resolve

CACHE_VERSION='bearing_full_benchmark_v1'
RST_PX=256
OFFSETS=np.asarray([[-128.,-128.],[-128.,128.],[128.,-128.],[128.,128.]],dtype=np.float64)

def _u8(path:str,size:int=256):
    with Image.open(path) as im:
        im=im.convert('RGB')
        if im.size!=(size,size): im=im.resize((size,size),Image.Resampling.BICUBIC)
        return np.asarray(im,dtype=np.uint8)

def _centres_px(row):
    inter=np.asarray([(float(row['block_x'])+1.0)*RST_PX,(float(row['block_y'])+1.0)*RST_PX],dtype=np.float64)
    return inter[None,:]+OFFSETS

def _global_px(row):
    if 'global_x_px' in row and 'global_y_px' in row:
        return np.asarray([float(row['global_x_px']),float(row['global_y_px'])],dtype=np.float64)
    # Bearing metadata normalized coordinates use UNI_PIXEL=128.
    return np.asarray([(float(row['block_x'])+1.0)*RST_PX+float(row['x_norm'])*128.0,
                       (float(row['block_y'])+1.0)*RST_PX+float(row['y_norm'])*128.0],dtype=np.float64)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--force',action='store_true'); p.add_argument('--cache-size',type=int,default=256); a=p.parse_args()
    root=Path(a.dataset_root).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    meta_path=bearing._find_metadata(root); df=pd.read_csv(meta_path); n=len(df)
    h=hashlib.sha256(); h.update(CACHE_VERSION.encode()); h.update(str(meta_path).encode()); h.update(str(meta_path.stat().st_size).encode()); h.update(str(a.cache_size).encode()); fp=h.hexdigest()
    mp=out/'cache_meta.json'
    expected=[out/'sat_gallery.npy',out/'train_uav.npy',out/'train_positive_sat_index.npy',out/'train_labels.npy',out/'test_uav.npy',out/'test_candidate_sat_index.npy',out/'test_candidate_xy_m.npy',out/'test_gt_m.npy',out/'test_true_index.npy']
    if not a.force and mp.exists() and json.loads(mp.read_text()).get('fingerprint')==fp and all(x.exists() for x in expected):
        print(f'[FULL-CACHE] hit: {out}',flush=True); return

    # Exact released split semantics: torch random_split with seed 42.
    tr=int(.85*n); va=int(.05*n); te=n-tr-va
    train_idx,val_idx,test_idx=torch.utils.data.random_split(range(n),[tr,va,te],generator=torch.Generator().manual_seed(42))
    train_ids=np.asarray(list(train_idx.indices),dtype=np.int64); test_ids=np.asarray(list(test_idx.indices),dtype=np.int64)
    np.save(out/'train_indices.npy',train_ids); np.save(out/'test_indices.npy',test_ids)

    # Resolve each unique RST image once and build a single shared satellite gallery.
    sat_paths=[]
    for col in ('p1_path','p2_path','p3_path','p4_path'):
        sat_paths.extend(_resolve(v,root) for v in df[col].astype(str).tolist())
    uniq=sorted(set(sat_paths)); path_to_idx={x:i for i,x in enumerate(uniq)}
    (out/'sat_paths.json').write_text(json.dumps(uniq,indent=2),encoding='utf-8')
    sat=np.lib.format.open_memmap(out/'sat_gallery.npy',mode='w+',dtype=np.uint8,shape=(len(uniq),a.cache_size,a.cache_size,3))
    for i,path in enumerate(uniq):
        sat[i]=_u8(path,a.cache_size)
        if (i+1)%100==0 or i+1==len(uniq): print(f'[FULL-CACHE] satellite {i+1}/{len(uniq)}',flush=True)
    sat.flush(); del sat

    # Training UAVs + nearest native RST positive.  Class id is the actual RST path.
    tu=np.lib.format.open_memmap(out/'train_uav.npy',mode='w+',dtype=np.uint8,shape=(len(train_ids),a.cache_size,a.cache_size,3))
    pos=np.zeros(len(train_ids),dtype=np.int64); labels=np.zeros(len(train_ids),dtype=np.int64)
    for j,idx in enumerate(train_ids):
        r=df.iloc[int(idx)]; gt=_global_px(r); centres=_centres_px(r); ti=int(np.argmin(np.linalg.norm(centres-gt[None,:],axis=1)))
        up=_resolve(r['target_path'],root); sp=_resolve(r[f'p{ti+1}_path'],root); si=path_to_idx[sp]
        tu[j]=_u8(up,a.cache_size); pos[j]=si; labels[j]=si
        if (j+1)%1000==0 or j+1==len(train_ids): print(f'[FULL-CACHE] train UAV {j+1}/{len(train_ids)}',flush=True)
    tu.flush(); del tu; np.save(out/'train_positive_sat_index.npy',pos); np.save(out/'train_labels.npy',labels)

    # Official paper test split. Metric coordinates remain map-plane metres at 0.25 m/px;
    # Bearing-UAV regression itself uses UNI_PIXEL=128 for normalized deltas.
    q=np.lib.format.open_memmap(out/'test_uav.npy',mode='w+',dtype=np.uint8,shape=(len(test_ids),a.cache_size,a.cache_size,3))
    cand=np.zeros((len(test_ids),4),dtype=np.int64); cxy=np.zeros((len(test_ids),4,2),dtype=np.float32); gt_m=np.zeros((len(test_ids),2),dtype=np.float32); true=np.zeros(len(test_ids),dtype=np.int64)
    for j,idx in enumerate(test_ids):
        r=df.iloc[int(idx)]; gt=_global_px(r); centres=_centres_px(r); ti=int(np.argmin(np.linalg.norm(centres-gt[None,:],axis=1)))
        q[j]=_u8(_resolve(r['target_path'],root),a.cache_size)
        for k in range(4): cand[j,k]=path_to_idx[_resolve(r[f'p{k+1}_path'],root)]
        cxy[j]=centres*float(bearing.MPP); gt_m[j]=gt*float(bearing.MPP); true[j]=ti
        if (j+1)%1000==0 or j+1==len(test_ids): print(f'[FULL-CACHE] test UAV {j+1}/{len(test_ids)}',flush=True)
    q.flush(); del q
    np.save(out/'test_candidate_sat_index.npy',cand); np.save(out/'test_candidate_xy_m.npy',cxy); np.save(out/'test_gt_m.npy',gt_m); np.save(out/'test_true_index.npy',true)

    payload={'cache_version':CACHE_VERSION,'fingerprint':fp,'metadata_rows':n,'split_seed':42,'split':[tr,va,te],'train_rows':len(train_ids),'test_rows':len(test_ids),'satellite_unique_images':len(uniq),'mpp':float(bearing.MPP),'training_scope':'official full metadata 85% split','test_scope':'official 10% split','candidate_scope':'native p1/p2/p3/p4 four adjacent RSTs','uses_route_prior':False}
    mp.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('[FULL-CACHE] PASS',payload,flush=True)
if __name__=='__main__': main()
