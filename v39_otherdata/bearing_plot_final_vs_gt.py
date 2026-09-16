#!/usr/bin/env python3
"""Render clear paper-style final Bearing-v39 result figures.

No numerical trajectory is smoothed, projected, or cosmetically moved.
Visual semantics are explicit:
  purple dashed = true per-frame GT observations;
  red solid     = final prediction;
  gray dotted   = planned waypoint/reference route only.
A white halo, thick strokes and a dimmed satellite background keep both GT and
prediction visible even when they overlap closely.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Dict,List,Mapping,Sequence,Tuple
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFont
Point=Tuple[float,float]
PRED=(238,45,45,255); GT=(176,78,245,255); REF=(215,215,215,180); HALO=(255,255,255,238)

def _rows(p:Path)->List[Dict[str,str]]:
    with p.open("r",newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def _origin(root:Path):
    r=_rows(root/"routes"/"train_01"/"manifest.csv")
    if not r:raise RuntimeError("train_01 manifest is empty")
    return float(r[0]["x_m"]),float(r[0]["y_m"])
def _find_csv(route,out,summary):
    p=Path(str(summary.get("CSV","")))
    if p.exists():return p
    q=out/p.name
    if q.exists():return q
    m=sorted(out.glob(f"{route}_*_frames.csv"))
    if not m:raise FileNotFoundError(f"No inference CSV for {route}")
    return m[-1]
def _waypoints(root,route):
    p=json.loads((root/"routes"/route/"waypoints.json").read_text())
    return [(float(x["pixel_x"]),float(x["pixel_y"])) for x in sorted(p["waypoints"],key=lambda x:int(x["waypoint_order"]))]
def _abs_px(x,y,ox,oy,mpp):return ((x+ox)/mpp,(y+oy)/mpp)

def _audit_points(route,root,out,summary,mpp,size,ox,oy):
    rows=_rows(_find_csv(route,out,summary)); man=_rows(root/"routes"/route/"manifest.csv")
    if not rows or len(rows)!=len(man):raise RuntimeError(f"{route}: CSV/manifest count mismatch")
    pred=[];gt=[];errs=[];mx=0.;w,h=size
    for i,(r,m) in enumerate(zip(rows,man)):
        gx_rel,gy_rel=float(r["gt_x"]),float(r["gt_y"])
        gx_abs,gy_abs=gx_rel+ox,gy_rel+oy
        ex=math.hypot(gx_abs-float(m["x_m"]),gy_abs-float(m["y_m"]))
        mx=max(mx,ex)
        fx,fy=float(r["final_x"]),float(r["final_y"])
        errs.append(math.hypot(fx-gx_rel,fy-gy_rel))
        pp=_abs_px(fx,fy,ox,oy,mpp); gg=(gx_abs/mpp,gy_abs/mpp)
        if not(-1<=pp[0]<=w and -1<=pp[1]<=h):raise RuntimeError(f"{route}: pred {i} outside RSI")
        pred.append(pp);gt.append(gg)
    if mx>1e-3:raise RuntimeError(f"{route}: GT coordinate contract mismatch {mx:.6f}m")
    mle=float(np.mean(errs))
    if abs(mle-float(summary["MLE_m"]))>1e-5:raise RuntimeError(f"{route}: MLE mismatch")
    print(f"[FINAL-PLOT-AUDIT] {route}: PASS frames={len(rows)} MLE={mle:.3f}m",flush=True)
    return pred,gt

def _dash(draw,pts,fill,width,dash,gap):
    for a,b in zip(pts[:-1],pts[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];L=math.hypot(dx,dy)
        if L<=1e-9:continue
        ux,uy=dx/L,dy/L;s=0.
        while s<L:
            e=min(L,s+dash);draw.line((a[0]+ux*s,a[1]+uy*s,a[0]+ux*e,a[1]+uy*e),fill=fill,width=width);s+=dash+gap
def _font(size,bold=False):
    names=["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for n in names:
        try:return ImageFont.truetype(n,size)
        except OSError:pass
    return ImageFont.load_default()
def _circle(draw,p,r,fill):
    draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=fill,outline=HALO,width=4)
def _bounds(groups,w,h):
    p=[q for g in groups for q in g];xs=[q[0] for q in p];ys=[q[1] for q in p]
    m=max(220,int(.06*max(max(xs)-min(xs),max(ys)-min(ys),1)))
    return(max(0,int(min(xs))-m),max(0,int(min(ys))-m),min(w,int(max(xs))+m),min(h,int(max(ys))+m))

def _legend(img,route,s):
    ov=Image.new("RGBA",img.size,(0,0,0,0));d=ImageDraw.Draw(ov,"RGBA")
    sc=max(1.,min(img.size)/1200.);tf=_font(max(22,int(28*sc)),True);bf=_font(max(18,int(21*sc)))
    pad=max(18,int(22*sc));lh=max(30,int(35*sc));bw=min(img.width-2*pad,max(700,int(790*sc)));bh=pad*2+lh*5
    d.rounded_rectangle((pad,pad,pad+bw,pad+bh),radius=14,fill=(0,0,0,185),outline=(255,255,255,180),width=2)
    x=pad+20;y=pad+14
    d.text((x,y),f"{route} - Bearing-v39 final result",font=tf,fill="white");y+=lh
    d.text((x,y),f"MLE {float(s['MLE_m']):.2f} m   P90 {float(s['P90_m']):.2f} m   LSR@15 {float(s['LSR@15_pct']):.1f}%",font=bf,fill="white");y+=lh
    sw=max(110,int(125*sc));lw=max(8,int(10*sc))
    d.line((x,y+10,x+sw,y+10),fill=HALO,width=lw+6);d.line((x,y+10,x+sw,y+10),fill=PRED,width=lw);d.text((x+sw+18,y-3),"Final prediction",font=bf,fill="white");y+=lh
    _dash(d,[(x,y+10),(x+sw,y+10)],HALO,lw+6,24*sc,13*sc);_dash(d,[(x,y+10),(x+sw,y+10)],GT,lw,24*sc,13*sc);d.text((x+sw+18,y-3),"Per-frame GT trajectory",font=bf,fill="white");y+=lh
    d.text((x,y),"Gray dotted line = planned waypoint route (context only)",font=bf,fill=(225,225,225,255))
    img.alpha_composite(ov)

def render(route,root,out,summary):
    sm=json.loads((root/"bearing_satellite.json").read_text());mpp=float(sm["mpp"])
    src=Image.open(sm["satellite_image"]).convert("RGB");base=ImageEnhance.Brightness(src).enhance(.84).convert("RGBA")
    ox,oy=_origin(root);ref=_waypoints(root,route);pred,gt=_audit_points(route,root,out,summary,mpp,base.size,ox,oy)
    d=ImageDraw.Draw(base,"RGBA");sc=max(1.,base.width/4096.)
    # planned route is deliberately subordinate
    _dash(d,ref,(255,255,255,130),6,18*sc,16*sc);_dash(d,ref,REF,3,18*sc,16*sc)
    # true GT first, with strong halo
    _dash(d,gt,HALO,max(19,int(20*sc)),36*sc,18*sc);_dash(d,gt,GT,max(12,int(13*sc)),36*sc,18*sc)
    # prediction above GT so even sub-metre overlap remains visible
    d.line(pred,fill=HALO,width=max(22,int(23*sc)),joint="curve");d.line(pred,fill=PRED,width=max(15,int(16*sc)),joint="curve")
    for p,c in ((gt[0],GT),(gt[-1],GT),(pred[0],PRED),(pred[-1],PRED)):_circle(d,p,max(12,int(14*sc)),c)
    for p in ref:_circle(d,p,max(5,int(6*sc)),(205,205,205,210))
    crop=base.crop(_bounds((ref,gt,pred),*base.size)).convert("RGBA");_legend(crop,route,summary)
    dest=out/f"{route}_final_result.jpg";crop.convert("RGB").save(dest,quality=98,subsampling=0)
    print(f"[FINAL-PLOT] {dest} | purple=true GT red=prediction gray=planned route",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared-root",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--routes",nargs="+",default=["test_01","test_02"]);a=p.parse_args()
    root=Path(a.prepared_root).resolve();out=Path(a.output_dir).resolve();s=json.loads((out/"bearing_v39_summary.json").read_text())
    for r in a.routes:render(r,root,out,s[r])
if __name__=="__main__":main()
