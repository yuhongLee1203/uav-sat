import argparse, math
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
import config
from data import RouteDataset
GT_COLOR=(0,255,0); PRED_COLOR=(255,0,255)
def meters_to_latlon(x,y,olat,olon):
    r=6378137.0; return float(olat)+math.degrees(float(y)/r), float(olon)+math.degrees(float(x)/(r*math.cos(math.radians(float(olat)))))
def pix(xy,ds,olat,olon):
    a=[]
    for x,y in np.asarray(xy): lat,lon=meters_to_latlon(x,y,olat,olon); px,py=ds.mapper.latlon_to_pixel(lat,lon); a.append([px,py])
    return np.asarray(a,float)
def fit(img,w,h):
    sh,sw=img.shape[:2]; sc=min(w/sw,h/sh); dw=max(1,int(sw*sc)); dh=max(1,int(sh*sc)); rs=cv2.resize(img,(dw,dh)); p=np.zeros((h,w,3),np.uint8); ox=(w-dw)//2; oy=(h-dh)//2; p[oy:oy+dh,ox:ox+dw]=rs; return p
def render(name):
    rows=pd.read_csv(config.OUTPUT_DIR/(name+"_recurrent_visual_measurement_frames.csv")); cp=torch.load(config.VISUAL_CHECKPOINT,map_location="cpu"); olat=float(cp["origin_lat"]); olon=float(cp["origin_lon"]); ds=RouteDataset(Path(config.ROUTE_ROOTS[config.ROUTE_NAMES.index(name)]),train=False,origin_lat=olat,origin_lon=olon)
    with Image.open(config.SAT_IMAGE) as im: sw,sh=im.size; mh=config.VIDEO_HEIGHT; mw=min(int(sw*mh/sh),int(config.VIDEO_WIDTH*.63)); mp=cv2.cvtColor(np.asarray(im.convert("RGB").resize((mw,mh))),cv2.COLOR_RGB2BGR)
    W,H=config.VIDEO_WIDTH,config.VIDEO_HEIGHT; uw=W-mw; sx=mw/sw; sy=mh/sh
    def cvxy(xy): s=pix(xy,ds,olat,olon); o=np.empty_like(s); o[:,0]=s[:,0]*sx+uw; o[:,1]=s[:,1]*sy; return o
    g=cvxy(rows[["gt_x","gt_y"]].values); p=cvxy(rows[["final_x","final_y"]].values); out=config.OUTPUT_DIR/(name+"_synchronized_inference.mp4"); wr=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),config.VIDEO_FPS,(W,H))
    for i,row in rows.iterrows():
        c=np.zeros((H,W,3),np.uint8); c[:,uw:]=mp; img=cv2.imread(str(row["image_path"]));
        if img is not None: c[:,:uw]=fit(img,uw,H)
        if i: cv2.polylines(c,[g[:i+1].astype(np.int32)],False,GT_COLOR,3); cv2.polylines(c,[p[:i+1].astype(np.int32)],False,PRED_COLOR,3)
        gp=g[i]; pp=p[i]; cv2.drawMarker(c,(int(gp[0]),int(gp[1])),GT_COLOR,cv2.MARKER_CROSS,20,3); cv2.drawMarker(c,(int(pp[0]),int(pp[1])),PRED_COLOR,cv2.MARKER_STAR,22,3)
        if int(row["heading_valid"]):
            a=math.radians(row["estimated_heading_deg_enu"]); m=config.HEADING_RENDER_ARROW_LENGTH_M; tip=cvxy([[row["final_x"]+m*math.cos(a),row["final_y"]+m*math.sin(a)]])[0]; cv2.arrowedLine(c,(int(pp[0]),int(pp[1])),(int(tip[0]),int(tip[1])),PRED_COLOR,3,tipLength=.25)
        for j,t in enumerate(["GT = GREEN    FINAL PRED = MAGENTA",f"frame={int(row['frame_id'])}",f"KF speed={row['kf_speed_m_per_frame']:.2f} m/frame",f"visual err={row['error_measurement_m']:.2f}m final err={row['error_final_m']:.2f}m"]): cv2.putText(c,t,(18,30+j*29),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),1)
        wr.write(c)
    wr.release(); print("rendered",out)
def main():
    a=argparse.ArgumentParser(); a.add_argument("--route",choices=["route_B","route_C","all"],default="all"); x=a.parse_args(); [render(n) for n in (["route_B","route_C"] if x.route=="all" else [x.route])]
if __name__=="__main__": main()
