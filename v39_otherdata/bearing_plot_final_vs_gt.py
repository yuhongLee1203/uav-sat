#!/usr/bin/env python3
"""Render Bearing-v39 navigation figures without inventing a continuous GT flight path.

Bearing-UAV-90K UAV samples are independent observations selected along an official
navigation route.  Connecting every selected GT observation with a polyline makes
an artificial zig-zag that is easy to misread as the real flight trajectory.

Visual semantics used here:
  green solid line + waypoint markers = official Bearing-UAV waypoint route
  blue dots                           = per-frame GT UAV observations (NOT connected)
  red solid line                      = final v39 prediction trajectory

No prediction or GT coordinate is smoothed, projected, shifted or cosmetically moved.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFont

Point=Tuple[float,float]
PRED=(230,45,45,255)
GT=(40,125,255,235)
ROUTE=(35,190,90,245)
HALO=(255,255,255,220)
TEXTBG=(0,0,0,170)


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
    return [(float(x["pixel_x"]),float(x["pixel_y"]))
            for x in sorted(p["waypoints"],key=lambda x:int(x["waypoint_order"]))]


def _abs_px(x,y,ox,oy,mpp):
    return ((x+ox)/mpp,(y+oy)/mpp)


def _audit_points(route,root,out,summary,mpp,size,ox,oy):
    rows=_rows(_find_csv(route,out,summary)); man=_rows(root/"routes"/route/"manifest.csv")
    if not rows or len(rows)!=len(man): raise RuntimeError(f"{route}: CSV/manifest count mismatch")
    pred=[]; gt=[]; errs=[]; mx=0.; w,h=size
    for i,(r,m) in enumerate(zip(rows,man)):
        gx_rel,gy_rel=float(r["gt_x"]),float(r["gt_y"])
        gx_abs,gy_abs=gx_rel+ox,gy_rel+oy
        mx=max(mx,math.hypot(gx_abs-float(m["x_m"]),gy_abs-float(m["y_m"])))
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


def _font(size,bold=False):
    names=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for n in names:
        try:return ImageFont.truetype(n,size)
        except OSError:pass
    return ImageFont.load_default()


def _circle(draw,p,r,fill,outline=None,width=2):
    draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=fill,outline=outline,width=width)


def _ring(draw,p,r,outline,width=2):
    draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=None,outline=outline,width=width)


def _bounds(groups,w,h):
    pts=[q for g in groups for q in g]
    xs=[q[0] for q in pts]; ys=[q[1] for q in pts]
    margin=max(180,int(.05*max(max(xs)-min(xs),max(ys)-min(ys),1)))
    return (max(0,int(min(xs))-margin),max(0,int(min(ys))-margin),
            min(w,int(max(xs))+margin),min(h,int(max(ys))+margin))


def _legend(img,route,s):
    ov=Image.new("RGBA",img.size,(0,0,0,0)); d=ImageDraw.Draw(ov,"RGBA")
    sc=max(1.,min(img.size)/1200.); title=_font(max(20,int(25*sc)),True); body=_font(max(16,int(19*sc)))
    pad=max(14,int(18*sc)); lh=max(27,int(31*sc)); bw=min(img.width-2*pad,max(625,int(700*sc))); bh=pad*2+lh*5
    d.rounded_rectangle((pad,pad,pad+bw,pad+bh),radius=12,fill=TEXTBG,outline=(255,255,255,130),width=2)
    x=pad+17; y=pad+11
    d.text((x,y),f"{route} - Bearing-v39",font=title,fill="white"); y+=lh
    d.text((x,y),f"MLE {float(s['MLE_m']):.2f} m   P90 {float(s['P90_m']):.2f} m   LSR@15 {float(s['LSR@15_pct']):.1f}%",font=body,fill="white"); y+=lh
    sw=max(90,int(100*sc))
    d.line((x,y+9,x+sw,y+9),fill=ROUTE,width=max(4,int(5*sc))); _circle(d,(x+sw*.5,y+9),max(4,int(5*sc)),ROUTE,HALO,1)
    d.text((x+sw+14,y-3),"Official waypoint route",font=body,fill="white"); y+=lh
    for xx in np.linspace(x,x+sw,6): _circle(d,(float(xx),y+9),max(2,int(3*sc)),GT,HALO,1)
    d.text((x+sw+14,y-3),"GT observations (points only)",font=body,fill="white"); y+=lh
    d.line((x,y+9,x+sw,y+9),fill=HALO,width=max(7,int(8*sc))); d.line((x,y+9,x+sw,y+9),fill=PRED,width=max(4,int(5*sc)))
    d.text((x+sw+14,y-3),"Prediction trajectory",font=body,fill="white")
    img.alpha_composite(ov)


def render(route,root,out,summary):
    sm=json.loads((root/"bearing_satellite.json").read_text()); mpp=float(sm["mpp"])
    src=Image.open(sm["satellite_image"]).convert("RGB")
    base=ImageEnhance.Brightness(src).enhance(.82).convert("RGBA")
    ox,oy=_origin(root); ref=_waypoints(root,route); pred,gt=_audit_points(route,root,out,summary,mpp,base.size,ox,oy)
    d=ImageDraw.Draw(base,"RGBA"); sc=max(1.,base.width/4096.)

    # 1) Official Bearing-UAV navigation route: the only continuous reference line.
    rw=max(4,int(5*sc))
    d.line(ref,fill=HALO,width=rw+4,joint="curve")
    d.line(ref,fill=ROUTE,width=rw,joint="curve")
    for i,p in enumerate(ref):
        _circle(d,p,max(6,int(7*sc)),ROUTE,HALO,max(1,int(2*sc)))

    # 2) Per-frame GT observations are independent samples.  Plot points only.
    #    Use every point, but keep markers compact so density is visible without a fake zig-zag.
    gr=max(2,int(3*sc))
    for p in gt:
        _circle(d,p,gr,GT,HALO,max(1,int(1*sc)))

    # 3) v39 prediction is temporal, therefore it is the only estimated continuous trajectory.
    pw=max(4,int(5*sc))
    d.line(pred,fill=HALO,width=pw+5,joint="curve")
    d.line(pred,fill=PRED,width=pw,joint="curve")

    # Start/end markers: route is green, prediction red; GT remains observations only.
    _circle(d,ref[0],max(9,int(10*sc)),ROUTE,HALO,max(2,int(2*sc)))
    _ring(d,ref[-1],max(10,int(11*sc)),ROUTE,max(3,int(3*sc)))
    _circle(d,pred[0],max(7,int(8*sc)),PRED,HALO,max(2,int(2*sc)))
    _ring(d,pred[-1],max(8,int(9*sc)),PRED,max(3,int(3*sc)))

    crop=base.crop(_bounds((ref,gt,pred),*base.size)).convert("RGBA")
    _legend(crop,route,summary)
    dest=out/f"{route}_final_result.jpg"
    crop.convert("RGB").save(dest,quality=98,subsampling=0)
    print(f"[FINAL-PLOT] {dest} | green=official route blue=GT observations red=prediction",flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--prepared-root",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--routes",nargs="+",default=["test_01","test_02"]); a=p.parse_args()
    root=Path(a.prepared_root).resolve(); out=Path(a.output_dir).resolve(); s=json.loads((out/"bearing_v39_summary.json").read_text())
    for r in a.routes:render(r,root,out,s[r])


if __name__=="__main__":main()
