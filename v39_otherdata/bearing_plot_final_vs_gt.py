#!/usr/bin/env python3
"""Render Bearing-v39 navigation figures in the same trajectory sense as Bearing-UAV Fig. 6.

Bearing-UAV Fig. 6 does NOT connect independent UAV-view samples and call that the
navigation GT.  Its purple dashed line is the predefined navigation trajectory
(Route #1/#2), i.e. the polyline defined by the official waypoint JSON files.

This renderer therefore shows only two trajectories:
  purple dashed = official predefined Bearing-UAV navigation trajectory
  red solid     = final Bearing-v39 prediction trajectory

The per-frame sample GT is still used internally to audit the already-reported
localization metrics, but it is deliberately not drawn because connecting those
independent samples creates the artificial zig-zag seen in the previous figures.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFont

Point=Tuple[float,float]
PRED=(228,44,52,255)
GT=(143,76,190,245)
HALO=(255,255,255,215)
TEXTBG=(0,0,0,170)


def _rows(p:Path)->List[Dict[str,str]]:
    with p.open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _origin(root:Path):
    r=_rows(root/"routes"/"train_01"/"manifest.csv")
    if not r:
        raise RuntimeError("train_01 manifest is empty")
    return float(r[0]["x_m"]),float(r[0]["y_m"])


def _find_csv(route,out,summary):
    p=Path(str(summary.get("CSV","")))
    if p.exists(): return p
    q=out/p.name
    if q.exists(): return q
    m=sorted(out.glob(f"{route}_*_frames.csv"))
    if not m:
        raise FileNotFoundError(f"No inference CSV for {route}")
    return m[-1]


def _official_trajectory(root:Path,route:str)->List[Point]:
    """Exact predefined trajectory points from the prepared copy of official wps*.json."""
    p=json.loads((root/"routes"/route/"waypoints.json").read_text(encoding="utf-8"))
    pts=[(float(x["pixel_x"]),float(x["pixel_y"]))
         for x in sorted(p["waypoints"],key=lambda x:int(x["waypoint_order"]))]
    if len(pts)<2:
        raise RuntimeError(f"{route}: official trajectory has <2 waypoints")
    return pts


def _abs_px(x,y,ox,oy,mpp):
    return ((x+ox)/mpp,(y+oy)/mpp)


def _audit_and_prediction(route,root,out,summary,mpp,size,ox,oy):
    """Keep the numerical metric audit, but return only the prediction for plotting."""
    rows=_rows(_find_csv(route,out,summary))
    man=_rows(root/"routes"/route/"manifest.csv")
    if not rows or len(rows)!=len(man):
        raise RuntimeError(f"{route}: CSV/manifest count mismatch")
    pred=[]; errs=[]; mx=0.; w,h=size
    for i,(r,m) in enumerate(zip(rows,man)):
        gx_rel,gy_rel=float(r["gt_x"]),float(r["gt_y"])
        gx_abs,gy_abs=gx_rel+ox,gy_rel+oy
        mx=max(mx,math.hypot(gx_abs-float(m["x_m"]),gy_abs-float(m["y_m"])))
        fx,fy=float(r["final_x"]),float(r["final_y"])
        errs.append(math.hypot(fx-gx_rel,fy-gy_rel))
        pp=_abs_px(fx,fy,ox,oy,mpp)
        if not(-1<=pp[0]<=w and -1<=pp[1]<=h):
            raise RuntimeError(f"{route}: pred {i} outside RSI")
        pred.append(pp)
    if mx>1e-3:
        raise RuntimeError(f"{route}: sample-GT coordinate contract mismatch {mx:.6f}m")
    mle=float(np.mean(errs))
    if abs(mle-float(summary["MLE_m"]))>1e-5:
        raise RuntimeError(f"{route}: MLE mismatch")
    print(f"[FINAL-PLOT-AUDIT] {route}: PASS frames={len(rows)} sample-MLE={mle:.3f}m",flush=True)
    return pred


def _dash(draw,pts,fill,width,dash,gap):
    for a,b in zip(pts[:-1],pts[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        if length<=1e-9: continue
        ux,uy=dx/length,dy/length
        s=0.0
        while s<length:
            e=min(length,s+dash)
            draw.line((a[0]+ux*s,a[1]+uy*s,a[0]+ux*e,a[1]+uy*e),fill=fill,width=width)
            s+=dash+gap


def _font(size,bold=False):
    names=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for n in names:
        try:return ImageFont.truetype(n,size)
        except OSError:pass
    return ImageFont.load_default()


def _bounds(groups,w,h):
    pts=[q for g in groups for q in g]
    xs=[q[0] for q in pts]; ys=[q[1] for q in pts]
    margin=max(180,int(.05*max(max(xs)-min(xs),max(ys)-min(ys),1)))
    return (max(0,int(min(xs))-margin),max(0,int(min(ys))-margin),
            min(w,int(max(xs))+margin),min(h,int(max(ys))+margin))


def _city_traj_title(root:Path,route:str)->str:
    city_map={"citya":"City A","cityb":"City B","cityc":"City C","cityd":"City D"}
    traj="#1" if route=="test_01" else "#2"
    return f"{city_map.get(root.name,root.name)} / Traj. {traj}"


def _legend(img,root,route,s):
    ov=Image.new("RGBA",img.size,(0,0,0,0)); d=ImageDraw.Draw(ov,"RGBA")
    sc=max(1.,min(img.size)/1200.)
    title=_font(max(20,int(25*sc)),True); body=_font(max(16,int(19*sc)))
    pad=max(14,int(18*sc)); lh=max(27,int(31*sc)); bw=min(img.width-2*pad,max(600,int(670*sc))); bh=pad*2+lh*4
    d.rounded_rectangle((pad,pad,pad+bw,pad+bh),radius=12,fill=TEXTBG,outline=(255,255,255,120),width=2)
    x=pad+17; y=pad+11
    d.text((x,y),_city_traj_title(root,route),font=title,fill="white"); y+=lh
    d.text((x,y),f"MLE {float(s['MLE_m']):.2f} m   P90 {float(s['P90_m']):.2f} m   LSR@15 {float(s['LSR@15_pct']):.1f}%",font=body,fill="white"); y+=lh
    sw=max(95,int(105*sc)); gw=max(3,int(4*sc)); pw=max(4,int(5*sc))
    _dash(d,[(x,y+9),(x+sw,y+9)],GT,gw,18*sc,11*sc)
    d.text((x+sw+15,y-3),"GT (predefined trajectory)",font=body,fill="white"); y+=lh
    d.line((x,y+9,x+sw,y+9),fill=HALO,width=pw+3)
    d.line((x,y+9,x+sw,y+9),fill=PRED,width=pw)
    d.text((x+sw+15,y-3),"Prediction",font=body,fill="white")
    img.alpha_composite(ov)


def render(route,root,out,summary):
    sm=json.loads((root/"bearing_satellite.json").read_text(encoding="utf-8")); mpp=float(sm["mpp"])
    src=Image.open(sm["satellite_image"]).convert("RGB")
    base=ImageEnhance.Brightness(src).enhance(.84).convert("RGBA")
    ox,oy=_origin(root)
    gt=_official_trajectory(root,route)
    pred=_audit_and_prediction(route,root,out,summary,mpp,base.size,ox,oy)
    d=ImageDraw.Draw(base,"RGBA"); sc=max(1.,base.width/4096.)

    # Bearing-UAV Fig. 6 convention: predefined trajectory is purple dashed.
    gw=max(4,int(5*sc))
    _dash(d,gt,HALO,gw+3,30*sc,17*sc)
    _dash(d,gt,GT,gw,30*sc,17*sc)

    # Our final temporal prediction: red solid, drawn on top.
    pw=max(4,int(5*sc))
    d.line(pred,fill=HALO,width=pw+4,joint="curve")
    d.line(pred,fill=PRED,width=pw,joint="curve")

    crop=base.crop(_bounds((gt,pred),*base.size)).convert("RGBA")
    _legend(crop,root,route,summary)
    dest=out/f"{route}_final_result.jpg"
    crop.convert("RGB").save(dest,quality=98,subsampling=0)
    print(f"[FINAL-PLOT] {dest} | purple-dashed=official predefined trajectory red=prediction",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepared-root",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--routes",nargs="+",default=["test_01","test_02"])
    a=p.parse_args()
    root=Path(a.prepared_root).resolve(); out=Path(a.output_dir).resolve()
    s=json.loads((out/"bearing_v39_summary.json").read_text(encoding="utf-8"))
    for r in a.routes:
        render(r,root,out,s[r])


if __name__=="__main__":
    main()
