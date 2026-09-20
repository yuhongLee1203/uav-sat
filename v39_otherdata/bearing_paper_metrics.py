#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import bearing_prepare as bearing

CITIES=("citya","cityb","cityc","cityd")
NAV_TO_INTERNAL={"nav50":"test_01","nav51":"test_02"}

# Bearing-UAV UAV-view reference values. Camera-heading values are intentionally
# kept separate from our motion-heading diagnostic.
LOCALIZATION_REF=[
 {"Method":"University-1652","Recall@1_UAV_pct":60.20,"LSR@15_UAV_pct":15.11,"MLE_UAV_m":33.15,"MedLE_UAV_m":None},
 {"Method":"SUES-200","Recall@1_UAV_pct":66.60,"LSR@15_UAV_pct":15.76,"MLE_UAV_m":30.83,"MedLE_UAV_m":None},
 {"Method":"DenseUAV","Recall@1_UAV_pct":73.43,"LSR@15_UAV_pct":16.54,"MLE_UAV_m":28.79,"MedLE_UAV_m":None},
 {"Method":"GTA-UAV","Recall@1_UAV_pct":70.71,"LSR@15_UAV_pct":27.96,"MLE_UAV_m":28.43,"MedLE_UAV_m":None},
 {"Method":"Bearing-UAV (VGG-16)","Recall@1_UAV_pct":83.17,"LSR@15_UAV_pct":89.36,"MLE_UAV_m":8.61,"MedLE_UAV_m":7.30},
]
CAMERA_HEADING_REF=[
 {"Method":"University-1652","HSR@15_camera_pct":None,"MHE_camera_deg":None,"MedHE_camera_deg":None},
 {"Method":"SUES-200","HSR@15_camera_pct":None,"MHE_camera_deg":None,"MedHE_camera_deg":None},
 {"Method":"DenseUAV","HSR@15_camera_pct":None,"MHE_camera_deg":None,"MedHE_camera_deg":None},
 {"Method":"GTA-UAV","HSR@15_camera_pct":None,"MHE_camera_deg":None,"MedHE_camera_deg":None},
 {"Method":"Bearing-UAV (VGG-16)","HSR@15_camera_pct":77.21,"MHE_camera_deg":12.90,"MedHE_camera_deg":7.20},
 {"Method":"Yours (current model)","HSR@15_camera_pct":None,"MHE_camera_deg":None,"MedHE_camera_deg":None},
]
NAV_REF=[
 {"Method":"University-1652","SR@20_UAV_pct":0.0,"SPL_UAV_pct":0.0,"NE_UAV_m":602.96},
 {"Method":"SUES-200","SR@20_UAV_pct":0.0,"SPL_UAV_pct":0.0,"NE_UAV_m":618.85},
 {"Method":"DenseUAV","SR@20_UAV_pct":0.0,"SPL_UAV_pct":0.0,"NE_UAV_m":651.93},
 {"Method":"GTA-UAV","SR@20_UAV_pct":0.0,"SPL_UAV_pct":0.0,"NE_UAV_m":661.91},
 {"Method":"Bearing-UAV (VGG-16)","SR@20_UAV_pct":50.0,"SPL_UAV_pct":29.82,"NE_UAV_m":275.61},
 {"Method":"Yours (current offline replay)","SR@20_UAV_pct":None,"SPL_UAV_pct":None,"NE_UAV_m":None},
]

def read_csv(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def path_len(x,y):return float(np.hypot(np.diff(x),np.diff(y)).sum())
def find_csv(full,nav):
    prefix="route_B" if nav=="nav50" else "route_C"
    m=sorted(Path(full).glob(prefix+"_*_frames.csv"))
    if not m:raise FileNotFoundError(f"{full}: {nav} frames csv missing")
    return m[-1]
def metrics(rows):
    err=np.asarray([float(r["error_final_m"]) for r in rows],float)
    # IMPORTANT: this is ground-track/motion heading, not Bearing camera yaw.
    mhe=np.asarray([abs(float(r["heading_error_deg"])) for r in rows if r.get("heading_error_deg","")!=""],float)
    gx=np.asarray([float(r["gt_x"]) for r in rows]);gy=np.asarray([float(r["gt_y"]) for r in rows])
    px=np.asarray([float(r["final_x"]) for r in rows]);py=np.asarray([float(r["final_y"]) for r in rows])
    lat=np.asarray([float(r["end_to_end_latency_ms"]) for r in rows if r.get("end_to_end_latency_ms","")!=""],float)
    ne=float(math.hypot(px[-1]-gx[-1],py[-1]-gy[-1]));sr=float(ne<=20.0)
    gl=path_len(gx,gy);pl=path_len(px,py);spl=sr*gl/max(gl,pl,1e-9)
    return {
      "Frames":len(rows),"MLE_m":float(err.mean()),"MedLE_m":float(np.median(err)),
      "P90_m":float(np.percentile(err,90)),"P95_m":float(np.percentile(err,95)),"P99_m":float(np.percentile(err,99)),
      "LSR@5_pct":float(100*np.mean(err<=5)),"LSR@10_pct":float(100*np.mean(err<=10)),
      "LSR@15_pct":float(100*np.mean(err<=15)),"LSR@20_pct":float(100*np.mean(err<=20)),
      "MotionMHE_deg":float(mhe.mean()) if len(mhe) else None,
      "MotionMedHE_deg":float(np.median(mhe)) if len(mhe) else None,
      "MotionHSR@15_pct":float(100*np.mean(mhe<=15)) if len(mhe) else None,
      "NE_m_route_replay":ne,"SR@20_pct_route_replay":100*sr,"SPL_pct_route_replay":100*spl,
      "JumpRate_pct":100*sum(int(float(r.get("abnormal_jump","0") or 0))!=0 for r in rows)/len(rows),
      "MaxFinalStep_m":max(float(r["final_step_m"]) for r in rows),
      "InferenceMean_ms":float(lat.mean()) if len(lat) else None,
      "FPS":float(1000/lat.mean()) if len(lat) and lat.mean()>0 else None,
    }

def four_rst_recall(prepared,nav,rows):
    prepared=Path(prepared);exp=json.loads((prepared/"experiment.json").read_text())
    metadata=pd.read_csv(exp["metadata_csv"]);city_rows=bearing._city_rows(metadata,exp["city"])
    manifest=read_csv(prepared/"routes"/NAV_TO_INTERNAL[nav]/"manifest.csv")
    train=read_csv(prepared/"routes"/"train_01"/"manifest.csv")
    if len(manifest)!=len(rows):raise RuntimeError(f"{exp['city']} {nav}: manifest/result length mismatch")
    origin_x=float(train[0]["x_m"]);origin_y=float(train[0]["y_m"]);mpp=float(exp["mpp"])
    good=0
    for man,pred in zip(manifest,rows):
        meta=city_rows.iloc[int(man["source_index"])]
        cx=float(meta["block_x"])*bearing.PATCH_SIZE+bearing.PATCH_SIZE
        cy=float(meta["block_y"])*bearing.PATCH_SIZE+bearing.PATCH_SIZE
        gtq=(float(meta["x_norm"])>=0.0,float(meta["y_norm"])>=0.0)
        pred_abs_x=(float(pred["final_x"])+origin_x)/mpp
        pred_abs_y=(float(pred["final_y"])+origin_y)/mpp
        predq=(pred_abs_x-cx>=0.0,pred_abs_y-cy>=0.0)
        good+=int(gtq==predq)
    return 100.0*good/max(len(rows),1)

def md(headers,rows):
    def f(v):
        if v is None:return "—"
        return f"{v:.3f}" if isinstance(v,float) else str(v)
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(f(r.get(h)) for h in headers)+" |" for r in rows])
def write_csv(p,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(p).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def checkpoint_mb(root):
    vals=[]
    for c in CITIES:
        d=Path(root)/c/"train_frames3"/"checkpoints"
        total=sum((d/n).stat().st_size for n in ["visual_retrieval_A_only.pt","controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"] if (d/n).is_file())
        if total:vals.append(total/(1024**2))
    return float(np.mean(vals)) if vals else None

def main():
    p=argparse.ArgumentParser();p.add_argument("--suite-root",required=True);p.add_argument("--output-dir");a=p.parse_args()
    root=Path(a.suite_root).resolve();out=Path(a.output_dir or root/"paper_benchmark");out.mkdir(parents=True,exist_ok=True)
    routes=[];pooled=[];rec_num=0.0;rec_den=0
    for city in CITIES:
        full=root/city/"variants"/"full";prepared=root/city/"prepared"
        for nav in ("nav50","nav51"):
            cp=find_csv(full,nav);rows=read_csv(cp);pooled+=rows;rec=four_rst_recall(prepared,nav,rows)
            rec_num+=rec*len(rows);rec_den+=len(rows)
            routes.append({"City":city,"Route":nav,**metrics(rows),"Recall@1_4RST_derived_pct":rec,"CSV":str(cp)})
    pm=metrics(pooled);recall=rec_num/max(rec_den,1)
    ours_loc={"Method":"Yours (Forward-18 + GRU + Kalman + SoftMS)","Recall@1_UAV_pct":recall,"LSR@15_UAV_pct":pm["LSR@15_pct"],"MLE_UAV_m":pm["MLE_m"],"MedLE_UAV_m":pm["MedLE_m"]}
    loc_table=LOCALIZATION_REF+[ours_loc]
    motion=[{"Method":"Yours (motion/ground-track heading diagnostic)","MotionMHE_deg":pm["MotionMHE_deg"],"MotionMedHE_deg":pm["MotionMedHE_deg"],"MotionHSR@15_pct":pm["MotionHSR@15_pct"]}]
    cities=[]
    for c in CITIES:
        rr=[r for r in routes if r["City"]==c];w=np.asarray([r["Frames"] for r in rr],float)
        cities.append({"City":c,"Recall@1_4RST_derived_pct":float(np.average([r["Recall@1_4RST_derived_pct"] for r in rr],weights=w)),"MLE_m":float(np.average([r["MLE_m"] for r in rr],weights=w)),"MedLE_m":float(np.average([r["MedLE_m"] for r in rr],weights=w)),"LSR@15_pct":float(np.average([r["LSR@15_pct"] for r in rr],weights=w)),"MotionMHE_deg":float(np.average([r["MotionMHE_deg"] for r in rr],weights=w)),"MotionHSR@15_pct":float(np.average([r["MotionHSR@15_pct"] for r in rr],weights=w))})
    replay=[{"Method":"Yours route-replay diagnostic","SR@20_pct_route_replay":float(np.mean([r["SR@20_pct_route_replay"] for r in routes])),"SPL_pct_route_replay":float(np.mean([r["SPL_pct_route_replay"] for r in routes])),"NE_m_route_replay":float(np.mean([r["NE_m_route_replay"] for r in routes]))}]
    efficiency=[{"Method":"Yours","OnDiskCheckpointSize_MB_per_city":checkpoint_mb(root),"EndToEndInferenceMean_ms":pm["InferenceMean_ms"],"FPS":pm["FPS"],"GFLOPs":None}]
    payload={"suite":str(root),"localization_comparison":loc_table,"camera_heading_reference":CAMERA_HEADING_REF,"motion_heading_diagnostic":motion,"per_city":cities,"per_route":routes,"bearing_naver_reference":NAV_REF,"route_replay_diagnostic":replay,"efficiency":efficiency,"protocol_notes":{"controlled_prior":"Current results use controlled_gt_jitter local prior and are not fully GT-free deployment results.","Recall@1":"Ours is a 4-RST-derived decision from continuous XY, not a retrieval-head output.","heading":"Current heading_error_deg is causal ground-track/motion heading derived from GT(t)-GT(t-1), NOT Bearing-UAV camera yaw. Therefore current MotionMHE/HSR are not inserted into the Bearing camera-heading comparison.","navigation":"Route-replay SR/SPL/NE are not Bearing-Naver closed-loop navigation metrics."}}
    (out/"bearing_paper_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    write_csv(out/"table_localization_comparison.csv",loc_table);write_csv(out/"table_camera_heading_reference.csv",CAMERA_HEADING_REF);write_csv(out/"table_motion_heading_diagnostic.csv",motion);write_csv(out/"table_city_metrics.csv",cities);write_csv(out/"table_route_metrics.csv",routes);write_csv(out/"table_navigation_reference.csv",NAV_REF);write_csv(out/"table_route_replay_navigation.csv",replay);write_csv(out/"table_efficiency.csv",efficiency)
    lines=["# Bearing-UAV aligned paper tables","","## A. Localization comparison (paper-facing)","",md(["Method","Recall@1_UAV_pct","LSR@15_UAV_pct","MLE_UAV_m","MedLE_UAV_m"],loc_table),"","## B. Camera-heading comparison (Bearing definition)","",md(["Method","HSR@15_camera_pct","MHE_camera_deg","MedHE_camera_deg"],CAMERA_HEADING_REF),"","## C. Current model motion-heading diagnostic (NOT camera yaw)","",md(["Method","MotionMHE_deg","MotionMedHE_deg","MotionHSR@15_pct"],motion),"","## D. Per-city localization + motion-heading diagnostic","",md(["City","Recall@1_4RST_derived_pct","MLE_m","MedLE_m","LSR@15_pct","MotionMHE_deg","MotionHSR@15_pct"],cities),"","## E. Bearing-Naver closed-loop reference","",md(["Method","SR@20_UAV_pct","SPL_UAV_pct","NE_UAV_m"],NAV_REF),"","## F. Offline route-replay diagnostic","",md(["City","Route","NE_m_route_replay","SR@20_pct_route_replay","SPL_pct_route_replay","JumpRate_pct","MaxFinalStep_m"],routes),"","## G. Efficiency","",md(["Method","OnDiskCheckpointSize_MB_per_city","EndToEndInferenceMean_ms","FPS","GFLOPs"],efficiency),"","## Required protocol notes","","- Current localization is a controlled local-prior/jitter experiment; do not claim fully GT-free deployment.","- Our Recall@1 is explicitly 4-RST-derived from continuous XY.","- Current MHE/HSR are ground-track motion-heading metrics, not Bearing-UAV camera-heading metrics.","- Route-replay SR/SPL/NE are not closed-loop Bearing-Naver metrics."]
    (out/"PAPER_TABLES.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("[PAPER BENCHMARK DONE]",out);print(json.dumps({"localization":ours_loc,"motion_heading":motion[0]},indent=2))
if __name__=="__main__":main()
