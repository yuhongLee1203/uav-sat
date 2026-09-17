#!/usr/bin/env python3
"""Render readable paper-style Bearing-v39 route figures.

No numerical trajectory is smoothed, projected, shifted, or cosmetically moved.
Visual semantics:
  red solid        = final prediction (dominant foreground trajectory)
  cyan thin dashed = true per-frame GT trajectory
  cyan rings       = sparse GT samples, so overlap remains readable
  gray dotted      = planned waypoint/reference route (context only)

The previous renderer used two very thick haloed trajectories.  Because v39 is
usually only a few metres from GT, those strokes visually merged.  This version
keeps the prediction clearly visible and encodes GT with a different line style
and sparse hollow markers instead of another heavy stroke.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFont

Point=Tuple[float,float]
PRED=(238,45,45,255)
GT=(0,215,255,255)
REF=(225,225,225,165)
HALO=(255,255,255,225)
DARK=(0,0,0,185)


def _rows(p:Path)->List[Dict[str,str]]:
    with p.open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _origin(root:Path):
    r=_rows(root/"routes"/"train_01"/"manifest.csv")
    if not r: raise RuntimeError("train_01 manifest is empty")
    return float(r[0]["x_m"]),float(r[0]["y_m"])


def _find_csv(route,out,summary):
    p=Path(str(summary.get("CSV","")))
    if p.exists(): return p
    q=out/p.name
    if q.exists(): return q
    m=sorted(out.glob(f"{route}_*_frames.csv"))
    if not m: raise FileNotFoundError(f"No inference CSV for {route}")
    return m[-1]


def _waypoints(root,route):
    p=json.loads((root/"routes"/route/"waypoints.json").read_text())
    return [(float(x["pixel_x"]),float(x["pixel_y"])) for x in sorted(p["waypoints"],key=lambda x:int(x["waypoint_order"]))]


def _abs_px(x,y,ox,oy,mpp):
    return ((x+ox)/mpp,(y+oy)/mpp)


def _audit_points(route,root,out,summary,mpp,size,ox,oy):
    rows=_rows(_find_csv(route,out,summary)); man=_rows(root/"routes"/route/"manifest.csv")
    if not rows or len(rows)!=len(man): raise RuntimeError(f"{route}: CSV/manifest count mismatch")
    pred=[];gt=[];errs=[];mx=0.;w,h=size
    for i,(r,m) in enumerate(zip(rows,man)):
        gx_rel,gy_rel=float(r["gt_x"]),float(r["gt_y"])
        gx_abs,gy_abs=gx_rel+ox,gy_rel+oy
        ex=math.hypot(gx_abs-float(m["x_m"]),gy_abs-float(m["y_m"]))
        mx=max(mx,ex)
        fx,fy=float(r["final_x"]),float(r["final_y"])
        errs.append(math.hypot(fx-gx_rel,fy-gy_rel))
        pp=_abs_px(fx,fy,ox,oy,mpp); gg=(gx_abs/mpp,gy_abs/mpp)
        if not(-1<=pp[0]<=w and -1<=pp[1]<=h): raise RuntimeError(f"{route}: pred {i} outside RSI")
        pred.append(pp); gt.append(gg)
    if mx>1e-3: raise RuntimeError(f"{route}: GT coordinate contract mismatch {mx:.6f}m")
    mle=float(np.mean(errs))
    if abs(mle-float(summary["MLE_m"]))>1e-5: raise RuntimeError(f"{route}: MLE mismatch")
    print(f"[FINAL-PLOT-AUDIT] {route}: PASS frames={len(rows)} MLE={mle:.3f}m",flush=True)
    return pred,gt


def _dash(draw,pts,fill,width,dash,gap):
    for a,b in zip(pts[:-1],pts[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]; L=math.hypot(dx,dy)
        if L<=1e-9: continue
        ux,uy=dx/L,dy/L; s=0.
        while s<L:
            e=min(L,s+dash)
            draw.line((a[0]+ux*s,a[1]+uy*s,a[0]+ux*e,a[1]+uy*e),fill=fill,width=width)
            s+=dash+gap


def _font(size,bold=False):
    names=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for n in names:
        try: return ImageFont.truetype(n,size)
        except OSError: pass
    return ImageFont.load_default()


def _filled_circle(draw,p,r,fill,outline=HALO,width=3):
    draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=fill,outline=outline,width=width)


def _ring(draw,p,r,outline,width=3):
    draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=None,outline=outline,width=width)


def _bounds(groups,w,h):
    p=[q for g in groups for q in g]; xs=[q[0] for q in p]; ys=[q[1] for q in p]
    m=max(220,int(.06*max(max(xs)-min(xs),max(ys)-min(ys),1)))
    return(max(0,int(min(xs))-m),max(0,int(min(ys))-m),min(w,int(max(xs))+m),min(h,int(max(ys))+m))


def _legend(img,route,s):
    ov=Image.new("RGBA",img.size,(0,0,0,0)); d=ImageDraw.Draw(ov,"RGBA")
    sc=max(1.,min(img.size)/1200.); tf=_font(max(21,int(26*sc)),True); bf=_font(max(17,int(20*sc)))
    pad=max(16,int(20*sc)); lh=max(28,int(32*sc)); bw=min(img.width-2*pad,max(650,int(735*sc))); bh=pad*2+lh*5
    d.rounded_rectangle((pad,pad,pad+bw,pad+bh),radius=13,fill=(0,0,0,175),outline=(255,255,255,150),width=2)
    x=pad+18; y=pad+12
    d.text((x,y),f"{route} - Bearing-v39",font=tf,fill="white"); y+=lh
    d.text((x,y),f"MLE {float(s['MLE_m']):.2f} m   P90 {float(s['P90_m']):.2f} m   LSR@15 {float(s['LSR@15_pct']):.1f}%",font=bf,fill="white"); y+=lh
    sw=max(95,int(105*sc)); pw=max(6,int(7*sc)); gw=max(3,int(4*sc))
    d.line((x,y+9,x+sw,y+9),fill=HALO,width=pw+4); d.line((x,y+9,x+sw,y+9),fill=PRED,width=pw)
    d.text((x+sw+16,y-3),"Prediction",font=bf,fill="white"); y+=lh
    _dash(d,[(x,y+9),(x+sw,y+9)],GT,gw,18*sc,12*sc); _ring(d,(x+sw*.5,y+9),max(4,int(5*sc)),GT,max(2,int(2*sc)))
    d.text((x+sw+16,y-3),"Ground truth",font=bf,fill="white"); y+=lh
    _dash(d,[(x,y+9),(x+sw,y+9)],REF,max(2,int(2*sc)),12*sc,12*sc)
    d.text((x+sw+16,y-3),"Waypoints (context)",font=bf,fill=(225,225,225,255))
    img.alpha_composite(ov)


def render(route,root,out,summary):
    sm=json.loads((root/"bearing_satellite.json").read_text()); mpp=float(sm["mpp"])
    src=Image.open(sm["satellite_image"]).convert("RGB")
    base=ImageEnhance.Brightness(src).enhance(.80).convert("RGBA")
    ox,oy=_origin(root); ref=_waypoints(root,route); pred,gt=_audit_points(route,root,out,summary,mpp,base.size,ox,oy)
    d=ImageDraw.Draw(base,"RGBA"); sc=max(1.,base.width/4096.)

    # Context route: thin and visually subordinate.
    _dash(d,ref,REF,max(2,int(2*sc)),15*sc,15*sc)

    # GT: thin cyan dashed line.  No heavy halo, so it cannot bury prediction.
    _dash(d,gt,DARK,max(4,int(5*sc)),28*sc,18*sc)
    _dash(d,gt,GT,max(2,int(3*sc)),28*sc,18*sc)

    # Prediction: foreground solid red.  Thin halo only for contrast with satellite imagery.
    pw=max(7,int(8*sc)); hw=pw+5
    d.line(pred,fill=HALO,width=hw,joint="curve")
    d.line(pred,fill=PRED,width=pw,joint="curve")

    # Sparse hollow GT markers are drawn last.  Their centres remain transparent, so
    # an overlapping red prediction remains visible through the rings.
    marker_step=max(1,len(gt)//18)
    rr=max(5,int(6*sc)); rw=max(2,int(2*sc))
    for p in gt[::marker_step]: _ring(d,p,rr,GT,rw)

    # Distinct endpoints.  Prediction is filled red; GT is a hollow cyan ring.
    _filled_circle(d,pred[0],max(8,int(9*sc)),PRED)
    _filled_circle(d,pred[-1],max(8,int(9*sc)),PRED)
    _ring(d,gt[0],max(10,int(11*sc)),GT,max(3,int(3*sc)))
    _ring(d,gt[-1],max(10,int(11*sc)),GT,max(3,int(3*sc)))

    # Small gray waypoint rings only; do not cover either trajectory.
    for p in ref: _ring(d,p,max(4,int(4*sc)),REF,max(2,int(2*sc)))

    crop=base.crop(_bounds((ref,gt,pred),*base.size)).convert("RGBA")
    _legend(crop,route,summary)
    dest=out/f"{route}_final_result.jpg"
    crop.convert("RGB").save(dest,quality=98,subsampling=0)
    print(f"[FINAL-PLOT] {dest} | red=prediction cyan-dashed/rings=GT gray=waypoints",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepared-root",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--routes",nargs="+",default=["test_01","test_02"])
    a=p.parse_args()
    root=Path(a.prepared_root).resolve(); out=Path(a.output_dir).resolve()
    s=json.loads((out/"bearing_v39_summary.json").read_text())
    for r in a.routes: render(r,root,out,s[r])


if __name__=="__main__": main()
