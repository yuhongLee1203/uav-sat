import argparse, csv, json, math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import config
from data import RouteDataset, meters_from_latlon
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import RecurrentVisualMeasurementRNN
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None
ARCHITECTURE_NAME = "RecurrentVisualMeasurementKalman_v15"
@dataclass
class RouteCache:
    route_name: str; frame_ids: torch.Tensor; gt_xy: torch.Tensor; uav_clip: torch.Tensor; image_paths: list
    def __len__(self): return int(self.gt_xy.shape[0])
@dataclass
class CandidateSet:
    centers: torch.Tensor; z_sat: torch.Tensor; raw_logits: torch.Tensor
def set_seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed));
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
def cache_dtype(): return torch.float16 if config.FEATURE_CACHE_DTYPE=="float16" else torch.float32
def parse_frame_id(v): return int(v.item()) if isinstance(v,torch.Tensor) else int(str(v))
def load_start_waypoint(route_name,olat,olon):
    payload=json.loads(Path(config.WAYPOINT_FILES[route_name]).read_text(encoding="utf-8")); first=sorted(payload["waypoints"],key=lambda x:int(x["waypoint_order"]))[0]; x,y=meters_from_latlon(first["latitude"],first["longitude"],olat,olon); return torch.tensor([x,y],dtype=torch.float32)
@torch.no_grad()
def build_route_cache(route_name,root,visual,device):
    st=config.VISUAL_CHECKPOINT.stat(); sig={"size":int(st.st_size),"mtime_ns":int(st.st_mtime_ns),"architecture":ARCHITECTURE_NAME}; cp=config.OUTPUT_DIR/"feature_cache"/(route_name+"_uav_clip.pt")
    if cp.exists():
        p=torch.load(cp,map_location="cpu")
        if p.get("signature")==sig: return RouteCache(route_name,p["frame_ids"],p["gt_xy"],p["uav_clip"],p["image_paths"])
    ds=RouteDataset(Path(root),train=False,origin_lat=visual.origin_lat,origin_lon=visual.origin_lon); fr=[]; gt=[]; clips=[]; paths=[]; bs=int(config.VISUAL_CACHE_BATCH_SIZE)
    for s in range(0,len(ds),bs):
        e=min(s+bs,len(ds)); items=[ds[i] for i in range(s,e)]; uav=torch.stack([x["uav"] for x in items]).to(device); clip=visual.encode_uav_clip(uav); clips.append(clip.detach().cpu().to(cache_dtype())); gt.append(torch.stack([x["xy"].float() for x in items]));
        for x in items: fr.append(parse_frame_id(x["frame_id"])); paths.append(str(x["image_path"]))
        if s==0 or e==len(ds) or (s//bs)%10==0: print(f"{route_name} backbone cache: {e}/{len(ds)}",flush=True)
    r=RouteCache(route_name,torch.tensor(fr,dtype=torch.long),torch.cat(gt).float(),torch.cat(clips),paths); cp.parent.mkdir(parents=True,exist_ok=True); torch.save({"signature":sig,"frame_ids":r.frame_ids,"gt_xy":r.gt_xy,"uav_clip":r.uav_clip,"image_paths":r.image_paths},cp); return r
@torch.no_grad()
def build_candidate_set(visual,uav_clip,center_xy):
    b=visual.candidate_batch(uav_clip,center_xy,grid_size=int(config.GRID_SIZE));
    if int(b.centers.shape[1])!=int(config.CANDIDATE_COUNT): raise RuntimeError("candidate count mismatch")
    return CandidateSet(b.centers,b.z_sat,b.raw_logits)
def candidate_target(c,gt):
    d=torch.linalg.norm(c.centers-gt[:,None,:],dim=2); n,i=d.min(dim=1); return i,n<=float(config.CANDIDATE_CAPTURE_RADIUS_M),n
def teacher_center_ratio(epoch):
    e=epoch+1; w=int(config.TEACHER_CENTER_WARMUP_EPOCHS); end=int(config.TEACHER_CENTER_END_EPOCH)
    if e<=w:return 1.0
    if e>=end:return 0.0
    return max(0.0,1.0-float(e-w)/float(max(1,end-w)))
def visual_measurement(out,cand,training):
    p=out.candidate_probability; idx=out.refined_logits.argmax(dim=1); hard=F.one_hot(idx,num_classes=out.refined_logits.shape[1]).to(p.dtype); w=hard+p-p.detach() if training else hard; anchor=(w.unsqueeze(-1)*cand.centers).sum(dim=1); return anchor+out.residual_xy,p,idx
class ConstantVelocityKalman2D:
    def __init__(self,xy):
        if KalmanFilter is None: raise ImportError("pip install filterpy")
        self.kf=KalmanFilter(dim_x=4,dim_z=2); self.kf.x=np.asarray([xy[0],xy[1],0,0],dtype=np.float64); self.kf.F=np.asarray([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=np.float64); self.kf.H=np.asarray([[1,0,0,0],[0,1,0,0]],dtype=np.float64); self.kf.P=np.diag([config.KF_INIT_POSITION_VAR,config.KF_INIT_POSITION_VAR,config.KF_INIT_VELOCITY_VAR,config.KF_INIT_VELOCITY_VAR]); self.kf.Q=np.diag([config.KF_PROCESS_POSITION_VAR,config.KF_PROCESS_POSITION_VAR,config.KF_PROCESS_VELOCITY_VAR,config.KF_PROCESS_VELOCITY_VAR]); self.kf.R=np.eye(2)*4
    @staticmethod
    def bound(v,m):
        v=np.asarray(v,dtype=np.float64).reshape(2); n=float(np.linalg.norm(v)); return v if n<=m else v*(m/max(n,1e-9))
    def position(self): return np.asarray(self.kf.x[:2],dtype=np.float64).reshape(2).copy()
    def velocity(self): return np.asarray(self.kf.x[2:4],dtype=np.float64).reshape(2).copy()
    def predict(self):
        prev=self.position(); self.kf.predict(); self.kf.x[2:4]=self.bound(self.kf.x[2:4],config.KF_MAX_SPEED_M_PER_FRAME); pred=self.position(); self.kf.x[:2]=prev+self.bound(pred-prev,config.MAX_STEP_M_PER_FRAME); return self.position()
    def update(self,z,var):
        prev=self.position(); self.kf.R=np.diag(np.clip(np.asarray(var).reshape(2),config.KF_MEASUREMENT_MIN_VAR,config.KF_MEASUREMENT_MAX_VAR)); self.kf.update(np.asarray(z).reshape(2)); self.kf.x[2:4]=self.bound(self.kf.x[2:4],config.KF_MAX_SPEED_M_PER_FRAME); upd=self.position(); self.kf.x[:2]=prev+self.bound(upd-prev,config.MAX_STEP_M_PER_FRAME); return self.position()
def loss_fn(out,meas,target,capture,gt):
    coord=F.smooth_l1_loss(meas,gt); ce=F.cross_entropy(out.refined_logits,target.to(out.refined_logits.device)) if bool(capture[0]) else meas.sum()*0; res=out.residual_xy.square().mean(); var=out.measurement_variance.clamp(config.KF_MEASUREMENT_MIN_VAR,config.KF_MEASUREMENT_MAX_VAR); err=meas-gt; nll=(0.5*(err.square()/var+var.log())).mean(); total=config.LOSS_COORD*coord+config.LOSS_CANDIDATE_CE*ce+config.LOSS_RESIDUAL*res+config.LOSS_UNCERTAINTY_NLL*nll; return total,coord,ce,res,nll
def train_one_epoch(model,opt,visual,cache,start_xy,train_end,epoch,device):
    model.train(); teacher=teacher_center_ratio(epoch); hidden=prev_uav=prev_score=None; kf=ConstantVelocityKalman2D(start_xy.cpu().numpy()); pend=None; cnt=0; rows=[]; caps=[]; tops=[]; errs=[]; opt.zero_grad(set_to_none=True)
    for i in range(train_end):
        pred_np=kf.predict() if i else kf.position(); pred=torch.tensor(pred_np,dtype=torch.float32,device=device).reshape(1,2); tc=(start_xy if i==0 else cache.gt_xy[i]).to(device).reshape(1,2); tc=tc+torch.randn_like(tc)*float(config.TEACHER_CENTER_JITTER_M); center=teacher*tc+(1-teacher)*pred; u=cache.uav_clip[i:i+1].to(device).float(); cand=build_candidate_set(visual,u,center); zu=visual.model.encode_uav_from_clip(u); out=model.forward_step(zu,cand.z_sat,cand.raw_logits,hidden,prev_uav,prev_score); meas,p,idx=visual_measurement(out,cand,True); gt=cache.gt_xy[i:i+1].to(device); target,capture,nearest=candidate_target(cand,gt); total,coord,ce,res,nll=loss_fn(out,meas,target,capture,gt); pend=total if pend is None else pend+total; cnt+=1; rows.append([float(x.detach().cpu()) for x in [total,coord,ce,res,nll]]+[float(out.measurement_variance.mean().detach().cpu()),float(torch.linalg.norm(out.residual_xy,dim=1).mean().detach().cpu()),float(nearest.mean().detach().cpu())]); caps.append(float(capture.float().mean().cpu())); tops.append(float((idx==target.to(idx.device)).float().mean().cpu())); errs.append(float(torch.linalg.norm(meas.detach()-gt,dim=1).mean().cpu())); kf.update(meas[0].detach().cpu().numpy(),out.measurement_variance[0].detach().cpu().numpy()); hidden=out.hidden; prev_uav=out.uav_state; prev_score=out.score_state
        if cnt>=config.TBPTT_STEPS or i==train_end-1:
            obj=pend/cnt; obj.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP_NORM); opt.step(); opt.zero_grad(set_to_none=True); hidden=hidden.detach(); prev_uav=prev_uav.detach(); prev_score=prev_score.detach(); pend=None; cnt=0
    a=np.asarray(rows); return {"teacher":teacher,"loss":a[:,0].mean(),"coord":a[:,1].mean(),"candidate":a[:,2].mean(),"residual":a[:,3].mean(),"nll":a[:,4].mean(),"variance":a[:,5].mean(),"residual_mag":a[:,6].mean(),"candidate_distance":a[:,7].mean(),"capture_pct":100*np.mean(caps),"top1_pct":100*np.mean(tops),"measurement_mle":np.mean(errs)}
@torch.no_grad()
def rollout(model,visual,cache,start_xy,device,collect=False):
    model.eval(); hidden=prev_uav=prev_score=None; kf=ConstantVelocityKalman2D(start_xy.cpu().numpy()); ma=[]; fa=[]; rows=[]
    for i in range(len(cache)):
        pred=kf.predict() if i else kf.position(); center=torch.tensor(pred,dtype=torch.float32,device=device).reshape(1,2); u=cache.uav_clip[i:i+1].to(device).float(); cand=build_candidate_set(visual,u,center); zu=visual.model.encode_uav_from_clip(u); out=model.forward_step(zu,cand.z_sat,cand.raw_logits,hidden,prev_uav,prev_score); meas,p,idx=visual_measurement(out,cand,False); mn=meas[0].cpu().numpy(); fn=kf.update(mn,out.measurement_variance[0].cpu().numpy()); vel=kf.velocity(); speed=float(np.linalg.norm(vel)); heading=math.degrees(math.atan2(vel[1],vel[0])) if speed>=config.HEADING_MIN_SPEED_M_PER_FRAME else float("nan"); gt=cache.gt_xy[i].numpy(); ma.append(mn); fa.append(fn)
        if collect:
            pc=p[0].cpu(); t2=pc.topk(k=min(2,pc.shape[0])).values; margin=float(t2[0]-(t2[1] if len(t2)>1 else 0)); sp=cand.centers[0,int(idx[0])].cpu().numpy(); rows.append({"sequence_index":i,"frame_id":int(cache.frame_ids[i]),"image_path":cache.image_paths[i],"gt_x":float(gt[0]),"gt_y":float(gt[1]),"kf_predict_x":float(pred[0]),"kf_predict_y":float(pred[1]),"visual_measurement_x":float(mn[0]),"visual_measurement_y":float(mn[1]),"final_x":float(fn[0]),"final_y":float(fn[1]),"kf_velocity_x_m_per_frame":float(vel[0]),"kf_velocity_y_m_per_frame":float(vel[1]),"kf_speed_m_per_frame":speed,"estimated_heading_deg_enu":heading,"heading_valid":int(np.isfinite(heading)),"selected_patch_x":float(sp[0]),"selected_patch_y":float(sp[1]),"rnn_residual_x_m":float(out.residual_xy[0,0]),"rnn_residual_y_m":float(out.residual_xy[0,1]),"measurement_variance_x":float(out.measurement_variance[0,0]),"measurement_variance_y":float(out.measurement_variance[0,1]),"candidate_probability_max":float(pc.max()),"candidate_probability_margin":margin,"candidate_count":36,"error_measurement_m":float(np.linalg.norm(mn-gt)),"error_final_m":float(np.linalg.norm(fn-gt))})
        hidden=out.hidden; prev_uav=out.uav_state; prev_score=out.score_state
    return np.asarray(ma),np.asarray(fa),rows
@torch.no_grad()
def eval_val(model,visual,cache,start,val_start,device):
    m,f,_=rollout(model,visual,cache,start,device); gt=cache.gt_xy.numpy(); me=np.linalg.norm(m[val_start:]-gt[val_start:],axis=1); fe=np.linalg.norm(f[val_start:]-gt[val_start:],axis=1); return {"measurement_mle":me.mean(),"mle":fe.mean(),"p90":np.quantile(fe,.9),"lsr15":100*np.mean(fe<=15)}
def train_temporal_model(model,visual,cache,start,device,epochs):
    train_end=max(8,int(len(cache)*config.TEMPORAL_TRAIN_FRACTION)); opt=torch.optim.AdamW(model.parameters(),lr=config.TEMPORAL_LR,weight_decay=config.TEMPORAL_WEIGHT_DECAY); best=float("inf"); bs=None; be=-1; patience=0; config.CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    for e in range(epochs):
        tr=train_one_epoch(model,opt,visual,cache,start,train_end,e,device); va=eval_val(model,visual,cache,start,train_end,device)
        if va["mle"]<best: best=va["mle"]; be=e+1; bs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        torch.save({"architecture":ARCHITECTURE_NAME,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"best_model":bs,"best_epoch":be,"best_val_mle":best,"current_gt_as_model_input":False,"previous_gt_as_model_input":False,"test_gt_as_model_input":False,"test_waypoint_frame_index_used":False,"rnn_type":"nn.RNNCell","search_grid":"full_6x6","candidate_count":36,"external_filter":"constant_velocity_kalman_xyvxvy","max_speed_m_per_frame":10.0,"teacher_center_ratio":tr["teacher"]},config.TEMPORAL_CHECKPOINT)
        print("epoch=%03d/%d loss=%.4f coord=%.4f cand=%.4f residual=%.4f nll=%.4f teacher=%.2f cap=%.1f%% top1=%.1f%% trainMeasMLE=%.3fm candDist=%.2fm var=%.2f residualMag=%.2fm valMeasMLE=%.3fm valMLE=%.3fm valP90=%.3fm valLSR15=%.2f%% best=%03d@%.3fm patience=%d"%(e+1,epochs,tr["loss"],tr["coord"],tr["candidate"],tr["residual"],tr["nll"],tr["teacher"],tr["capture_pct"],tr["top1_pct"],tr["measurement_mle"],tr["candidate_distance"],tr["variance"],tr["residual_mag"],va["measurement_mle"],va["mle"],va["p90"],va["lsr15"],be,best,patience),flush=True)
        if tr["teacher"]<=1e-6 and e+1>=config.TEMPORAL_MIN_EPOCHS_BEFORE_STOP and patience>=config.TEMPORAL_EARLY_STOPPING_PATIENCE: break
    cp=torch.load(config.TEMPORAL_CHECKPOINT,map_location="cpu"); cp["model"]=bs; cp["best_model"]=bs; torch.save(cp,config.TEMPORAL_CHECKPOINT); model.load_state_dict(bs)
def load_temporal_model(model,device):
    cp=torch.load(config.TEMPORAL_CHECKPOINT,map_location="cpu"); model.load_state_dict(cp.get("best_model") or cp["model"]); model.to(device).eval(); return cp
def metric(pred,gt):
    e=np.linalg.norm(pred-gt,axis=1); ps=np.linalg.norm(np.diff(pred,axis=0),axis=1) if len(pred)>1 else np.array([0.]); gs=np.linalg.norm(np.diff(gt,axis=0),axis=1) if len(gt)>1 else np.array([0.]); return {"MLE_m":float(e.mean()),"MedLE_m":float(np.median(e)),"P90_m":float(np.quantile(e,.9)),"P95_m":float(np.quantile(e,.95)),"LSR@5_pct":100*float(np.mean(e<=5)),"LSR@10_pct":100*float(np.mean(e<=10)),"LSR@15_pct":100*float(np.mean(e<=15)),"LSR@20_pct":100*float(np.mean(e<=20)),"RPE_step_mean_m":float(np.mean(np.abs(ps-gs))),"JumpRate_pct":100*float(np.mean(ps>(gs+config.JUMP_TOLERANCE_M))),"MaxError_m":float(e.max())}
def run_inference(model,visual,cache,start,device,csv_path):
    m,f,rows=rollout(model,visual,cache,start,device,True); csv_path.parent.mkdir(parents=True,exist_ok=True); fh=open(csv_path,"w",newline="",encoding="utf-8"); w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows); fh.close(); gt=cache.gt_xy.numpy(); return {"VisualMeasurement":metric(m,gt),"FinalKalman":metric(f,gt)},rows
def ensure_visual_checkpoint(device,epochs,reuse):
    if reuse and config.VISUAL_CHECKPOINT.exists(): return
    train_visual_retrieval_a_only(device=device,epochs=int(epochs),jitter_m=float(config.LOCAL_PRIOR_JITTER_M),resume=False)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["train","eval","train_eval"],default="train_eval"); p.add_argument("--visual-epochs",type=int,default=config.VISUAL_EPOCHS); p.add_argument("--temporal-epochs",type=int,default=config.TEMPORAL_EPOCHS); p.add_argument("--reuse-visual",action="store_true"); a=p.parse_args(); set_seed(config.SEED); device=torch.device(config.DEVICE if torch.cuda.is_available() else "cpu"); ensure_visual_checkpoint(device,a.visual_epochs,bool(a.reuse_visual)) if a.mode in ("train","train_eval") else None; visual=FrozenVisualLocalizer(device); routes={n:Path(r) for n,r in zip(config.ROUTE_NAMES,config.ROUTE_ROOTS)}
    if a.mode in ("train","train_eval"): c=build_route_cache("route_A",routes["route_A"],visual,device); s=load_start_waypoint("route_A",visual.origin_lat,visual.origin_lon); model=RecurrentVisualMeasurementRNN().to(device); train_temporal_model(model,visual,c,s,device,a.temporal_epochs)
    if a.mode in ("eval","train_eval"):
        model=RecurrentVisualMeasurementRNN().to(device); load_temporal_model(model,device); results={}
        for n in ["route_B","route_C"]: c=build_route_cache(n,routes[n],visual,device); s=load_start_waypoint(n,visual.origin_lat,visual.origin_lon); sm,_=run_inference(model,visual,c,s,device,config.OUTPUT_DIR/(n+"_recurrent_visual_measurement_frames.csv")); results[n]=sm; print(n,sm["FinalKalman"],flush=True)
        config.OUTPUT_DIR.mkdir(parents=True,exist_ok=True); (config.OUTPUT_DIR/"robust_tracker_summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
if __name__=="__main__": main()
