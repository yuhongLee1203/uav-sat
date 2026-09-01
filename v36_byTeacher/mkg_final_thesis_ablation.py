"""MKG thesis ablation suite. Route A train; Routes B/C eval."""
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import robust_tracker as rt
import six_architecture_autoref_experiment as core
from visual_localizer import FrozenVisualLocalizer
from six_architecture_model import PositionRefinementGRU

ABLS=("full","no_xy","no_variance","no_temporal_mean","no_first_difference","no_hidden")
DECS=("softms","weighted","top1")
COMPS=("M","MK","MG","MKG")
METRICS=("MLE_m","MedLE_m","P90_m","P95_m","P99_m","CVaR90_m","LSR@5_pct","LSR@10_pct","LSR@15_pct","LSR@20_pct")

class MaskedGRU(PositionRefinementGRU):
    def __init__(self, ablation):
        super().__init__(int(config.RNN_FEATURE_DIM),int(config.RNN_HIDDEN_DIM),float(config.RNN_DROPOUT))
        self.ablation=ablation
    def forward_step(self,stage_xy,variance_xy,z_uav,previous_z_uav,hidden):
        if previous_z_uav is None: previous_z_uav=z_uav
        if hidden is None or self.ablation=="no_hidden": hidden=self.initial_hidden(z_uav.shape[0],z_uav.device,z_uav.dtype)
        mean=.5*(z_uav+previous_z_uav); diff=z_uav-previous_z_uav
        xy=stage_xy.float()/max(self.position_scale_m,1e-6)
        var=torch.log1p(variance_xy.float().clamp_min(0)/max(self.variance_scale_m2,1e-6))
        if self.ablation=="no_xy": xy=torch.zeros_like(xy)
        if self.ablation=="no_variance": var=torch.zeros_like(var)
        if self.ablation=="no_temporal_mean": mean=torch.zeros_like(mean)
        if self.ablation=="no_first_difference": diff=torch.zeros_like(diff)
        inp=torch.cat([self.xy_projector(xy),self.var_projector(var),self.mean_projector(mean),self.diff_projector(diff)],1)
        h=self.gru(inp,hidden); corr=self.position_head(self.dropout(h))
        class O: pass
        o=O(); o.corrected_xy=stage_xy.float()+corr; o.correction_xy=corr; o.hidden=h
        return o

def root(): return Path(config.BACKBONE_OUTPUT_DIR)/"mkg_final_thesis_ablation"
def ckpt(tag): return Path(config.CHECKPOINT_DIR)/f"mkg_final_ablation_{tag}_{config.BACKBONE_KEY}.pt"
def rdir(tag): return root()/"runs"/tag

def cache(name,visual,device):
    i=config.ROUTE_NAMES.index(name); return rt.build_route_cache(name,config.ROUTE_ROOTS[i],visual,device)

def state(name,visual):
    _,xy,_=rt.planned_route_start(name,visual.origin_lat,visual.origin_lon)
    return {"kalman":core.StandardXYKalman(xy),"hidden":None,"previous_z":None}

def measure(visual,uav,center,grid,decoder):
    c=torch.as_tensor(center,dtype=torch.float32,device=visual.device).reshape(1,2)
    b=visual.candidate_batch(uav_clip=uav,center_xy=c,grid_size=grid)
    p=torch.softmax(b.raw_logits/float(config.MEANSHIFT_SCORE_TAU),1)
    if decoder=="softms": xy=b.softms_xy
    elif decoder=="weighted": xy=(p[:,:,None]*b.centers).sum(1)
    else: xy=b.raw_top1_xy
    d=b.centers-xy[:,None,:]; var=(p[:,:,None]*d.square()).sum(1).clamp_min(1e-3)
    return xy,var,b.z_uav,b.centers

def frame(comp,model,visual,uav,center,st,device,grid,decoder):
    xy,var,z,centers=measure(visual,uav,center,grid,decoder); base=xy
    gout=None
    for s in comp[1:]:
        if s=="K":
            v=st["kalman"].step(xy[0].detach().cpu().numpy(),var[0].detach().cpu().numpy())
            xy=torch.as_tensor(v,dtype=torch.float32,device=device).reshape(1,2)
        elif s=="G":
            gout=model.forward_step(xy,var,z,st["previous_z"],None if model.ablation=="no_hidden" else st["hidden"])
            xy=gout.corrected_xy; st["hidden"]=None if model.ablation=="no_hidden" else gout.hidden
    st["previous_z"]=z.detach(); return xy,base,var,centers,gout

def model_for(ablation,device): return MaskedGRU(ablation).to(device)

def train(tag,args,visual,device):
    if "G" not in args.component: return None,None
    m=model_for(args.gru_ablation,device); a=cache("route_A",visual,device)
    opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=float(config.TEMPORAL_WEIGHT_DECAY)); best=float("inf"); bs=None
    for ep in range(1,args.epochs+1):
        m.train(); st=state("route_A",visual); opt.zero_grad(set_to_none=True); loss=None; n=0; chunks=[]
        for i in range(len(a)):
            u=a.uav_clip[i:i+1].to(device).float(); ref=a.gt_xy[i].cpu().numpy().astype(np.float64)
            _,_,_,_,g=frame(args.component,m,visual,u,ref,st,device,args.grid_size,args.decoder)
            target=a.gt_xy[i:i+1].to(device).float(); l=F.smooth_l1_loss(g.corrected_xy,target); loss=l if loss is None else loss+l; n+=1
            if n>=args.tbptt or i==len(a)-1:
                x=loss/n; x.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),float(config.GRAD_CLIP_NORM)); opt.step(); opt.zero_grad(set_to_none=True)
                chunks.append(float(x.detach().cpu())); loss=None; n=0
                if st["hidden"] is not None: st["hidden"]=st["hidden"].detach()
                st["previous_z"]=st["previous_z"].detach()
        ml=float(np.mean(chunks))
        if ml<best:
            best=ml; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; ckpt(tag).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":bs,"ablation":args.gru_ablation,"loss":best},ckpt(tag))
        print(f"[{tag}] epoch={ep:03d}/{args.epochs} loss={ml:.6f} best={best:.6f}",flush=True)
    m.load_state_dict(bs); m.eval(); return m,best

def load(args,device):
    if "G" not in args.component: return None
    p=ckpt(args.load_tag or args.tag); q=torch.load(p,map_location="cpu"); m=model_for(args.gru_ablation,device); m.load_state_dict(q["model"]); m.eval(); return m

def offset(name,i,r,seed):
    if r<=0:return np.zeros(2)
    rid=config.ROUTE_NAMES.index(name); rng=np.random.default_rng(seed+100003*rid+9176*i); a=rng.uniform(0,2*math.pi)
    return r*np.array([math.cos(a),math.sin(a)])

@torch.no_grad()
def evaluate(name,args,visual,m,device):
    c=cache(name,visual,device); st=state(name,visual); errs=[]; baseerrs=[]; caps=[]; radii=[]; times=[]
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for i in range(len(c)):
        u=c.uav_clip[i:i+1].to(device).float(); ref=c.gt_xy[i].cpu().numpy().astype(np.float64); center=ref+offset(name,i,args.prior_error_m,args.seed)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        t=time.perf_counter(); out,base,_,centers,_=frame(args.component,m,visual,u,center,st,device,args.grid_size,args.decoder)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        times.append((time.perf_counter()-t)*1000)
        o=out[0].cpu().numpy(); b=base[0].cpu().numpy(); cc=centers[0].cpu().numpy()
        errs.append(float(np.linalg.norm(o-ref))); baseerrs.append(float(np.linalg.norm(b-ref)))
        d=np.linalg.norm(cc-ref[None,:],axis=1); caps.append(float(d.min()<=float(config.CANDIDATE_CAPTURE_RADIUS_M))); radii.append(float(np.linalg.norm(cc-center[None,:],axis=1).max()))
    s=rt.metric_summary(errs); s.update({"BaseVisualMLE_m":float(np.mean(baseerrs)),"CandidateCapture_pct":100*float(np.mean(caps)),"MeanCandidateMaxRadius_m":float(np.mean(radii)),"Latency_ms_per_frame":float(np.mean(times)),"FPS":1000/float(np.mean(times)),"PeakAllocatedGPU_MB":float(torch.cuda.max_memory_allocated(device)/1048576) if torch.cuda.is_available() else 0.0}); return s

def run(args):
    if args.ms_bandwidth is not None: config.MEANSHIFT_BANDWIDTH_M=float(args.ms_bandwidth)
    rt.set_seed(args.seed); device=rt.resolve_device(args.device); visual=FrozenVisualLocalizer(device)
    m,loss=(train(args.tag,args,visual,device) if args.mode=="train-eval" else (load(args,device),None))
    obj={"tag":args.tag,"group":args.group,"component":args.component,"grid":args.grid_size,"decoder":args.decoder,"gru":args.gru_ablation,"bandwidth":float(config.MEANSHIFT_BANDWIDTH_M),"prior_error":args.prior_error_m,"train_loss":loss,"results":{}}
    for n in ("route_B","route_C"):
        obj["results"][n]=evaluate(n,args,visual,m,device); print(json.dumps(obj["results"][n]),flush=True)
    rdir(args.tag).mkdir(parents=True,exist_ok=True); json.dump(obj,(rdir(args.tag)/"summary.json").open("w"),indent=2)

def collect():
    rows=[]
    for p in sorted((root()/"runs").glob("*/summary.json")):
        o=json.load(p.open()); rr=o["results"]; row={k:o[k] for k in ("tag","group","component","grid","decoder","gru","bandwidth","prior_error")}
        for k in METRICS+("BaseVisualMLE_m","CandidateCapture_pct","Latency_ms_per_frame","FPS","PeakAllocatedGPU_MB"):
            row["Avg_"+k]=float(np.mean([rr["route_B"][k],rr["route_C"][k]]))
        rows.append(row)
    t=root()/"tables"; t.mkdir(parents=True,exist_ok=True)
    def write(name,data):
        if not data:return
        with (t/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    write("all_runs_bc_average.csv",rows)
    for g,n in (("component","component_ablation.csv"),("window","window_size_ablation.csv"),("decoder","decoder_ablation.csv"),("gru","gru_ablation.csv"),("bandwidth","bandwidth_ablation.csv"),("robustness","coarse_prior_robustness.csv")):
        write(n,[r for r in rows if r["group"] in ("baseline",g)])
    old=Path(config.BACKBONE_OUTPUT_DIR)/"six_architecture_gt_center_center6x6"; order=[]
    for a in ("MKG","MGK","GMK","GKM","KGM","KMG"):
        p=old/a.lower()/"summary.json"
        if p.exists():
            q=json.load(p.open())["results"]; order.append({"Architecture":a,"Avg_MLE_m":float(np.mean([q["route_B"]["MLE_m"],q["route_C"]["MLE_m"]]))})
    write("architecture_order_existing.csv",order)
    json.dump({"final_method":"MKG","new_runs":len(rows),"windows":[4,5,6,7,8],"prior_errors_m":[0,5,10,15,20,25,30],"tables":[p.name for p in t.glob("*.csv")]},(root()/"manifest.json").open("w"),indent=2)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("train-eval","eval","collect"),default="train-eval"); p.add_argument("--tag",default=""); p.add_argument("--group",default="baseline"); p.add_argument("--component",choices=COMPS,default="MKG"); p.add_argument("--grid-size",type=int,default=6); p.add_argument("--decoder",choices=DECS,default="softms"); p.add_argument("--gru-ablation",choices=ABLS,default="full"); p.add_argument("--prior-error-m",type=float,default=0); p.add_argument("--ms-bandwidth",type=float,default=None); p.add_argument("--load-tag",default=""); p.add_argument("--device",default="cuda:0"); p.add_argument("--epochs",type=int,default=int(config.TEMPORAL_EPOCHS)); p.add_argument("--lr",type=float,default=float(config.TEMPORAL_LR)); p.add_argument("--tbptt",type=int,default=int(config.TBPTT_STEPS)); p.add_argument("--seed",type=int,default=int(config.SEED)); a=p.parse_args()
    collect() if a.mode=="collect" else run(a)
if __name__=="__main__": main()
