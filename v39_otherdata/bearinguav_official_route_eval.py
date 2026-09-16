#!/usr/bin/env python3
"""Evaluate official Bearing-UAV VGG-16 on the selected test route frames.

This uses the authors' released model/checkpoint and its native four-neighbour
RST input.  No v39 waypoint/local/temporal prior is injected.  Output figures
show true per-frame GT (purple dashed) and the raw Bearing-UAV prediction (red).
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from typing import Dict,List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import bearing_prepare as bearing
from bearing_route_baseline_runner import _figure


def _manifest(path:Path)->List[dict]:
    with path.open("r",newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def _resolve(raw,dataset_root:Path)->str:
    text=str(raw).replace("\\","/");p=Path(text)
    if p.exists():return str(p.resolve())
    if "Bearing_UAV_90K/" in text:
        q=dataset_root/text.split("Bearing_UAV_90K/",1)[1]
        if q.exists():return str(q.resolve())
    for a in ("citya","cityb","cityc","cityd","city_rsi","c4m_254k_96bc_b15_s100_v3d","c4m_254k_96bc_b15_s100","c1_254k_96bc_b15_s1_v3d","c1_254k_96bc_b15_s1","c1_254k_37bc_b15_s1_v3d"):
        t=f"/{a}/"
        if t in text:
            q=dataset_root/a/text.split(t,1)[1]
            if q.exists():return str(q.resolve())
    raise FileNotFoundError(f"Cannot resolve Bearing path: {raw}")

def _route_csv(dataset_root,prepared,city,route,city_rows,out):
    man=_manifest(prepared/"routes"/route/"manifest.csv");sel=[]
    for frame in man:
        idx=int(frame["source_index"])
        if idx<0 or idx>=len(city_rows):raise RuntimeError(f"{city}/{route}: bad source_index {idx}")
        row=city_rows.iloc[idx].copy()
        for col in ("p1_path","p2_path","p3_path","p4_path","target_path"):
            row[col]=_resolve(row[col],dataset_root)
        row["__frame_id"]=int(frame["frame_id"]);row["__gt_x_m"]=float(frame["x_m"]);row["__gt_y_m"]=float(frame["y_m"])
        sel.append(row)
    df=pd.DataFrame(sel).reset_index(drop=True);out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False);return df

def _load_model(root:Path,weights:Path,device):
    sys.path.insert(0,str(root))
    try:from cvphr.models.posaglreg import models as bm
    finally:
        try:sys.path.remove(str(root))
        except ValueError:pass
    orig=bm.models.vgg16
    def nodl(*args,**kwargs):kwargs.pop("pretrained",None);kwargs["weights"]=None;return orig(*args,**kwargs)
    bm.models.vgg16=nodl
    cfg=json.loads((weights/"training_configure.json").read_text()) if (weights/"training_configure.json").exists() else {}
    cls_name=cfg.get("model_class","PARCASGM_v5a");kw=cfg.get("model_kwargs",bm.model_kwargs_par_ca_sgm_v5a)
    model=getattr(bm,cls_name)(**kw);ck=torch.load(weights/"best_model.pth",map_location="cpu");model.load_state_dict(ck.get("model_state_dict",ck),strict=True);model.to(device).eval()
    return model,bm.RSBlockDatasetPA_v3q,cls_name,kw

def _eval(model,dataset_class,csv_path,device,mpp,workers,batch_size):
    ds=dataset_class(str(csv_path),is_train=False);kw=dict(batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available(),drop_last=False)
    if workers>0:kw.update(persistent_workers=True,prefetch_factor=1)
    loader=DataLoader(ds,**kw);pp=[];gp=[];pd=[];gd=[];blocks=[]
    with torch.inference_mode():
        for b in loader:
            x=b["patches"].to(device,non_blocking=True);p,d=model(x)
            pp.append(p.detach().cpu().numpy());gp.append(b["coords"].numpy());pd.append(d.detach().cpu().numpy());gd.append(b["agl_coords"].numpy());blocks.append(b["block_xy"].numpy())
    pp=np.concatenate(pp).astype(np.float64);gp=np.concatenate(gp).astype(np.float64);pd=np.concatenate(pd).astype(np.float64);gd=np.concatenate(gd).astype(np.float64);blocks=np.concatenate(blocks).astype(np.float64)
    pred_px=blocks*bearing.PATCH_SIZE+bearing.PATCH_SIZE+pp*bearing.PATCH_SIZE
    gt_px=blocks*bearing.PATCH_SIZE+bearing.PATCH_SIZE+gp*bearing.PATCH_SIZE
    pred_m=pred_px*float(mpp);gt_m=gt_px*float(mpp);err=np.linalg.norm(pred_m-gt_m,axis=1)
    recall=np.all(np.sign(pp)==np.sign(gp),axis=1)
    pnorm=pd/np.maximum(np.linalg.norm(pd,axis=1,keepdims=True),1e-12);gnorm=gd/np.maximum(np.linalg.norm(gd,axis=1,keepdims=True),1e-12);head=np.degrees(np.arccos(np.clip(np.sum(pnorm*gnorm,axis=1),-1,1)))
    met={"frames":int(len(err)),"Recall@1_pct":float(100*np.mean(recall)),"MLE_m":float(err.mean()),"MedLE_m":float(np.median(err)),"P90_m":float(np.percentile(err,90)),"P95_m":float(np.percentile(err,95)),"P99_m":float(np.percentile(err,99)),"LSR@5_pct":float(100*np.mean(err<=5)),"LSR@10_pct":float(100*np.mean(err<=10)),"LSR@15_pct":float(100*np.mean(err<=15)),"LSR@20_pct":float(100*np.mean(err<=20)),"HSR@15_pct":float(100*np.mean(head<=15)),"MHE_deg":float(head.mean()),"MedHE_deg":float(np.median(head)),"distance_errors_m":err.tolist(),"heading_errors_deg":head.tolist()}
    return met,pred_m,gt_m

def main():
    p=argparse.ArgumentParser();p.add_argument("--official-root",required=True);p.add_argument("--weights-dir",required=True);p.add_argument("--dataset-root",required=True);p.add_argument("--generated-root",required=True);p.add_argument("--city",required=True,choices=["citya","cityb","cityc","cityd"]);p.add_argument("--output-root",required=True);p.add_argument("--workers",type=int,default=1);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--cpu-threads",type=int,default=2);a=p.parse_args()
    torch.set_num_threads(max(1,a.cpu_threads))
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    if torch.cuda.is_available():torch.backends.cudnn.benchmark=True
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");root=Path(a.official_root).resolve();weights=Path(a.weights_dir).resolve();data=Path(a.dataset_root).resolve();gen=Path(a.generated_root).resolve();prepared=gen/a.city;out=Path(a.output_root).resolve()/a.city;out.mkdir(parents=True,exist_ok=True)
    exp=json.loads((prepared/"experiment.json").read_text());mpp=float(json.loads((prepared/"bearing_satellite.json").read_text())["mpp"]);meta=pd.read_csv(bearing._find_metadata(data));city_rows=bearing._city_rows(meta,a.city)
    model,dscls,cls_name,model_kw=_load_model(root,weights,device);routes={};all_d=[];all_h=[];rec_hits=0.0
    for route in ("test_01","test_02"):
        csvp=out/f"{route}_selected_metadata.csv";_route_csv(data,prepared,a.city,route,city_rows,csvp)
        met,pred_m,gt_m=_eval(model,dscls,csvp,device,mpp,a.workers,a.batch_size)
        # hard audit against our selected-frame manifest GT
        man=_manifest(prepared/"routes"/route/"manifest.csv");manifest_gt=np.asarray([[float(r["x_m"]),float(r["y_m"])] for r in man],dtype=np.float64)
        mx=float(np.linalg.norm(manifest_gt-gt_m,axis=1).max())
        if mx>1e-2:raise RuntimeError(f"{a.city}/{route}: official metadata GT mismatch {mx:.6f}m")
        with (out/f"{route}_frames.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["frame_id","gt_x_m","gt_y_m","pred_x_m","pred_y_m","error_m"])
            for i,(g,q,e) in enumerate(zip(gt_m,pred_m,met["distance_errors_m"])):w.writerow([i,*map(float,g),*map(float,q),float(e)])
        _figure(prepared,route,pred_m,gt_m,met,out/f"{route}_final_result.jpg","Bearing-UAV official VGG-16")
        routes[route]=met;all_d+=met["distance_errors_m"];all_h+=met["heading_errors_deg"];rec_hits+=met["Recall@1_pct"]*met["frames"]/100
        print(f"[BEARING-OFFICIAL] {a.city}/{route}: R1={met['Recall@1_pct']:.2f}% MLE={met['MLE_m']:.3f}m LSR15={met['LSR@15_pct']:.2f}%",flush=True)
    d=np.asarray(all_d);h=np.asarray(all_h);agg={"frames":int(len(d)),"Recall@1_pct":float(100*rec_hits/max(len(d),1)),"MLE_m":float(d.mean()),"MedLE_m":float(np.median(d)),"P90_m":float(np.percentile(d,90)),"LSR@15_pct":float(100*np.mean(d<=15)),"HSR@15_pct":float(100*np.mean(h<=15)),"MHE_deg":float(h.mean()),"MedHE_deg":float(np.median(h))}
    payload={"method":"Bearing-UAV official VGG-16","source":"official liukejia121/bearinguav code + official pretrained cross_view checkpoint","city":a.city,"model_class":cls_name,"model_kwargs":model_kw,"dataset_route_source":exp.get("test_route_source"),"evaluation_scope":"same selected test frames; native official four-neighbour RST pose regression","uses_v39_route_prior":False,"routes":routes,"aggregate_two_routes":agg}
    (out/"official_bearinguav_same_route.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");print(f"[BEARING-OFFICIAL] DONE {a.city}: MLE={agg['MLE_m']:.3f}m",flush=True)
if __name__=="__main__":main()
