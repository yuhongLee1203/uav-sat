#!/usr/bin/env python3
"""Verify released Bearing-UAV VGG-16 on the official paper test split.

Important: Bearing-UAV position labels are normalized by UNI_PIXEL=128, not by
PATCH_SIZE=256.  The official MLE implementation computes
    UNI_PIXEL * (pred - gt) * meter_per_pixel
so this evaluator follows that definition exactly.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import bearing_prepare as bearing
from bearinguav_official_route_eval import _load_model,_resolve

UNI_PIXEL=128.0
PUBLISHED={"Recall@1_pct":83.17,"LSR@15_pct":89.36,"HSR@15_pct":77.21,"MLE_m":8.61,"MedLE_m":7.30,"MHE_deg":12.90,"MedHE_deg":7.20}

def _prepare_test_csv(dataset_root:Path,metadata_path:Path,out_csv:Path):
    df=pd.read_csv(metadata_path); n=len(df); tr=int(.85*n); va=int(.05*n); te=n-tr-va
    _,_,idx=random_split(range(n),[tr,va,te],generator=torch.Generator().manual_seed(42))
    test=df.iloc[list(idx.indices)].copy().reset_index(drop=True)
    for col in ("p1_path","p2_path","p3_path","p4_path","target_path"):
        test[col]=[_resolve(v,dataset_root) for v in test[col].tolist()]
    out_csv.parent.mkdir(parents=True,exist_ok=True); test.to_csv(out_csv,index=False)
    return n,len(test)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--official-root',required=True); p.add_argument('--weights-dir',required=True); p.add_argument('--dataset-root',required=True); p.add_argument('--output',required=True); p.add_argument('--workers',type=int,default=1); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--cpu-threads',type=int,default=2); a=p.parse_args()
    torch.set_num_threads(max(1,a.cpu_threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark=True
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root=Path(a.official_root).resolve(); weights=Path(a.weights_dir).resolve(); data=Path(a.dataset_root).resolve(); out=Path(a.output).resolve()
    metadata=bearing._find_metadata(data); csvp=out.parent/'official_paper_test_resolved.csv'; total_n,test_n=_prepare_test_csv(data,metadata,csvp)
    model,dscls,cls_name,model_kw=_load_model(root,weights,dev); ds=dscls(str(csvp),is_train=False)
    kw=dict(batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=torch.cuda.is_available(),drop_last=False)
    if a.workers>0: kw.update(persistent_workers=True,prefetch_factor=1)
    loader=DataLoader(ds,**kw); errs=[]; heads=[]; recalls=[]
    with torch.inference_mode():
        for b in loader:
            x=b['patches'].to(dev,non_blocking=True); pos,hd=model(x)
            pp=pos.detach().cpu().numpy().astype(np.float64); gp=b['coords'].numpy().astype(np.float64)
            pd=hd.detach().cpu().numpy().astype(np.float64); gd=b['agl_coords'].numpy().astype(np.float64)
            # Official definition: normalized-coordinate delta * UNI_PIXEL * mpp.
            err=np.linalg.norm((pp-gp)*UNI_PIXEL*float(bearing.MPP),axis=1)
            recall=np.all(np.sign(pp)==np.sign(gp),axis=1)
            pn=pd/np.maximum(np.linalg.norm(pd,axis=1,keepdims=True),1e-12); gn=gd/np.maximum(np.linalg.norm(gd,axis=1,keepdims=True),1e-12)
            he=np.degrees(np.arccos(np.clip(np.sum(pn*gn,axis=1),-1,1)))
            errs.extend(err.tolist()); heads.extend(he.tolist()); recalls.extend(recall.tolist())
    e=np.asarray(errs); h=np.asarray(heads); r=np.asarray(recalls,dtype=bool)
    met={"frames":int(len(e)),"Recall@1_pct":float(100*np.mean(r)),"LSR@15_pct":float(100*np.mean(e<=15)),"HSR@15_pct":float(100*np.mean(h<=15)),"MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"MHE_deg":float(h.mean()),"MedHE_deg":float(np.median(h))}
    delta={k:float(met[k]-v) for k,v in PUBLISHED.items()}
    checks={"MLE_within_3m":abs(delta['MLE_m'])<=3,"Recall_within_10pp":abs(delta['Recall@1_pct'])<=10,"LSR15_within_10pp":abs(delta['LSR@15_pct'])<=10}
    payload={"protocol":"official full metadata 85/5/10 split, seed=42, released VGG-16 checkpoint","metric_conversion":"official UNI_PIXEL=128 normalized-coordinate scale","metadata_rows":total_n,"test_rows":test_n,"model_class":cls_name,"model_kwargs":model_kw,"measured":met,"published":PUBLISHED,"measured_minus_published":delta,"sanity_checks":checks}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('[PAPER-VERIFY] measured:',met,flush=True); print('[PAPER-VERIFY] published:',PUBLISHED,flush=True); print('[PAPER-VERIFY] checks:',checks,flush=True)
    print('[PAPER-VERIFY] PASS' if all(checks.values()) else '[PAPER-VERIFY] WARNING: paper-scale mismatch remains',flush=True)
if __name__=='__main__': main()
