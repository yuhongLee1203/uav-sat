#!/usr/bin/env python3
"""Train public M2T baselines on Bearing-UAV's full 85% training split.

This replaces the invalid train_01-only baseline experiment.  Each method keeps
its own public architecture/objective, but all four are trained on the released
Bearing-UAV 85% train split and evaluated on the released 10% test split with
the native four-adjacent-RST candidate protocol.  The same trained checkpoint is
then evaluated on the eight official navigation-route frame sets without giving
it waypoint/temporal/local-prior information.
"""
from __future__ import annotations
import argparse,csv,json,math,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from bearing_route_baseline_runner import Adapter,_figure

PUBLISHED={
 'university1652':{'Recall@1_pct':60.20,'MLE_m':33.15,'LSR@15_pct':15.11},
 'sues200':{'Recall@1_pct':66.60,'MLE_m':30.83,'LSR@15_pct':15.76},
 'denseuav':{'Recall@1_pct':73.43,'MLE_m':28.79,'LSR@15_pct':16.54},
 'gtauav':{'Recall@1_pct':70.71,'MLE_m':28.43,'LSR@15_pct':27.96},
}
METHODS=tuple(PUBLISHED)

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def raw(arr,idx,dev):
 x=torch.from_numpy(np.asarray(arr[idx],dtype=np.uint8)).to(dev,non_blocking=True)
 return x.permute(0,3,1,2).contiguous().float().div_(255.)
def prep(x,size,mean,std,train=False,satellite=False):
 if x.shape[-2:]!=(size,size): x=F.interpolate(x,size=(size,size),mode='bicubic',align_corners=False,antialias=True)
 if train:
  if torch.rand((),device=x.device)<.5: x=torch.flip(x,[3])
  if satellite:
   k=int(torch.randint(0,4,(),device=x.device)); x=torch.rot90(x,k,(2,3)) if k else x
 m=torch.tensor(mean,device=x.device,dtype=x.dtype).view(1,3,1,1); s=torch.tensor(std,device=x.device,dtype=x.dtype).view(1,3,1,1)
 return (x-m)/s

def embed(adapter,x,view):
 with torch.cuda.amp.autocast(enabled=adapter.amp): return adapter.embed(x,view)
def batches(n,batch,rng):
 order=rng.permutation(n); end=(len(order)//batch)*batch
 for s in range(0,end,batch): yield order[s:s+batch]

def metrics(pred,gt,choice,true):
 e=np.linalg.norm(pred-gt,axis=1)
 return {'frames':int(len(e)),'Recall@1_pct':float(100*np.mean(choice==true)),'MLE_m':float(e.mean()),'MedLE_m':float(np.median(e)),'P90_m':float(np.percentile(e,90)),'LSR@5_pct':float(100*np.mean(e<=5)),'LSR@10_pct':float(100*np.mean(e<=10)),'LSR@15_pct':float(100*np.mean(e<=15)),'LSR@20_pct':float(100*np.mean(e<=20)),'distance_errors_m':e.tolist()}

def train(adapter,cache,ckpt,force,seed):
 if ckpt.exists() and not force:
  obj=torch.load(ckpt,map_location='cpu'); adapter.model.load_state_dict(obj['model_state_dict'],strict=True); print(f'[FULL-{adapter.method}] checkpoint hit {ckpt}',flush=True); return
 u=np.load(cache/'train_uav.npy',mmap_mode='r'); g=np.load(cache/'sat_gallery.npy',mmap_mode='r'); pos=np.load(cache/'train_positive_sat_index.npy'); lab=np.load(cache/'train_labels.npy'); rng=np.random.default_rng(seed)
 adapter.prepare_scheduler(max(1,len(u)//adapter.batch)); scaler=torch.cuda.amp.GradScaler(enabled=adapter.amp)
 start=1
 last=ckpt.with_name(ckpt.stem+'_resume.pt')
 if last.exists() and not force:
  obj=torch.load(last,map_location='cpu'); adapter.model.load_state_dict(obj['model']); adapter.optimizer.load_state_dict(obj['optimizer']); start=int(obj['epoch'])+1; print(f'[FULL-{adapter.method}] resume epoch {start}',flush=True)
 for ep in range(start,adapter.epochs+1):
  adapter.model.train(); total=0.; seen=0
  for idx in batches(len(u),adapter.batch,rng):
   xu=prep(raw(u,idx,adapter.device),adapter.input_size,adapter.mean,adapter.std,True,False); xs=prep(raw(g,pos[idx],adapter.device),adapter.input_size,adapter.mean,adapter.std,True,True); y=torch.as_tensor(lab[idx],device=adapter.device,dtype=torch.long)
   adapter.optimizer.zero_grad(set_to_none=True)
   with torch.cuda.amp.autocast(enabled=adapter.amp): loss=adapter.train_step(xs,xu,y)
   if not torch.isfinite(loss): raise RuntimeError(f'{adapter.method}: non-finite loss')
   scaler.scale(loss).backward(); scaler.unscale_(adapter.optimizer); torch.nn.utils.clip_grad_norm_(adapter.model.parameters(),100.); scaler.step(adapter.optimizer); scaler.update()
   if adapter.scheduler_per_step and adapter.scheduler is not None: adapter.scheduler.step()
   total+=float(loss.detach())*len(idx); seen+=len(idx)
  if not adapter.scheduler_per_step and adapter.scheduler is not None: adapter.scheduler.step()
  avg=total/max(seen,1); print(f'[FULL-{adapter.method}] epoch={ep:03d}/{adapter.epochs} loss={avg:.6f}',flush=True)
  if ep%5==0 or ep==adapter.epochs:
   torch.save({'epoch':ep,'model':adapter.model.state_dict(),'optimizer':adapter.optimizer.state_dict()},last)
 adapter.model.eval(); ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':adapter.model.state_dict(),'config':adapter.config(),'training_scope':'Bearing-UAV official 85% split'},ckpt)
 if last.exists(): last.unlink()

def gallery_features(adapter,arr,batch=64):
 out=[]; adapter.model.eval()
 with torch.inference_mode():
  for s in range(0,len(arr),batch):
   idx=np.arange(s,min(s+batch,len(arr))); x=prep(raw(arr,idx,adapter.device),adapter.input_size,adapter.mean,adapter.std,False,True); out.append(embed(adapter,x,'sat').detach())
 return torch.cat(out,0)
def query_features(adapter,arr,batch=64):
 out=[]; adapter.model.eval()
 with torch.inference_mode():
  for s in range(0,len(arr),batch):
   idx=np.arange(s,min(s+batch,len(arr))); x=prep(raw(arr,idx,adapter.device),adapter.input_size,adapter.mean,adapter.std,False,False); out.append(embed(adapter,x,'uav').detach())
 return torch.cat(out,0)
def eval_full(adapter,cache):
 gallery=np.load(cache/'sat_gallery.npy',mmap_mode='r'); q=np.load(cache/'test_uav.npy',mmap_mode='r'); cand=np.load(cache/'test_candidate_sat_index.npy'); cxy=np.load(cache/'test_candidate_xy_m.npy'); gt=np.load(cache/'test_gt_m.npy'); true=np.load(cache/'test_true_index.npy')
 gf=gallery_features(adapter,gallery,max(32,adapter.batch)); qf=query_features(adapter,q,max(16,adapter.batch)); cf=gf[torch.as_tensor(cand,device=gf.device)]; sim=torch.einsum('nd,nkd->nk',qf,cf); choice=sim.argmax(1).cpu().numpy(); pred=cxy[np.arange(len(choice)),choice]; return metrics(pred,gt,choice,true)
def eval_route(adapter,cache,prepared,route,outdir):
 q=np.load(cache/f'{route}_uav.npy',mmap_mode='r'); cand=np.load(cache/f'{route}_candidates.npy',mmap_mode='r'); cxy=np.load(cache/f'{route}_candidate_xy_m.npy'); gt=np.load(cache/f'{route}_gt_m.npy'); true=np.load(cache/f'{route}_true_index.npy')
 qf=query_features(adapter,q,max(16,adapter.batch)); flat=cand.reshape((-1,)+cand.shape[2:]); sf=query_features(adapter,flat,max(16,adapter.batch)).reshape(len(q),4,-1) if False else None
 # Satellite branch must be used for RST candidates.
 sat=[]
 with torch.inference_mode():
  for s in range(0,len(flat),max(16,adapter.batch)):
   idx=np.arange(s,min(s+max(16,adapter.batch),len(flat))); x=prep(raw(flat,idx,adapter.device),adapter.input_size,adapter.mean,adapter.std,False,True); sat.append(embed(adapter,x,'sat').detach())
 sf=torch.cat(sat,0).reshape(len(q),4,-1); sim=torch.einsum('nd,nkd->nk',qf,sf); choice=sim.argmax(1).cpu().numpy(); pred=cxy[np.arange(len(choice)),choice]; met=metrics(pred,gt,choice,true)
 outdir.mkdir(parents=True,exist_ok=True)
 with (outdir/f'{route}_frames.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['frame_id','gt_x_m','gt_y_m','pred_x_m','pred_y_m','candidate','true_candidate','error_m'])
  for i,(g,p,c,t,e) in enumerate(zip(gt,pred,choice,true,met['distance_errors_m'])): w.writerow([i,float(g[0]),float(g[1]),float(p[0]),float(p[1]),int(c),int(t),float(e)])
 _figure(prepared,route,pred,gt,met,outdir/f'{route}_final_result.jpg',adapter.method+' full-split')
 return met

def main():
 p=argparse.ArgumentParser(); p.add_argument('--method',required=True,choices=METHODS); p.add_argument('--repo-dir',required=True); p.add_argument('--repo-commit',default='unknown'); p.add_argument('--full-cache',required=True); p.add_argument('--route-cache-root',required=True); p.add_argument('--generated-root',required=True); p.add_argument('--output-root',required=True); p.add_argument('--cpu-threads',type=int,default=2); p.add_argument('--batch-size',type=int,default=0); p.add_argument('--force',action='store_true'); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
 torch.set_num_threads(max(1,a.cpu_threads))
 try: torch.set_num_interop_threads(1)
 except RuntimeError: pass
 torch.backends.cudnn.benchmark=True; seed_all(a.seed); dev=torch.device('cuda:0')
 cache=Path(a.full_cache).resolve(); cm=json.loads((cache/'cache_meta.json').read_text()); gallery=np.load(cache/'sat_gallery.npy',mmap_mode='r'); default={'university1652':32,'sues200':16,'denseuav':16,'gtauav':32}[a.method]; batch=a.batch_size or default
 adapter=Adapter(a.method,Path(a.repo_dir).resolve(),len(gallery),dev,batch)
 out=Path(a.output_root).resolve()/a.method; out.mkdir(parents=True,exist_ok=True); ckpt=out/'checkpoint.pt'; train(adapter,cache,ckpt,a.force,a.seed)
 paper=eval_full(adapter,cache); pub=PUBLISHED[a.method]; delta={k:float(paper[k]-v) for k,v in pub.items()}; print(f"[FULL-{a.method}] PAPER R1={paper['Recall@1_pct']:.2f}% MLE={paper['MLE_m']:.2f}m LSR15={paper['LSR@15_pct']:.2f}%",flush=True)
 routes={}
 for city in ('citya','cityb','cityc','cityd'):
  routes[city]={}; rc=Path(a.route_cache_root).resolve()/city; prep=Path(a.generated_root).resolve()/city; od=out/city
  for route in ('test_01','test_02'):
   m=eval_route(adapter,rc,prep,route,od); routes[city][route]=m; print(f"[FULL-{a.method}] {city}/{route}: R1={m['Recall@1_pct']:.2f}% MLE={m['MLE_m']:.2f}m",flush=True)
 payload={'method':a.method,'repo_commit':a.repo_commit,'training_scope':'Bearing-UAV official full metadata 85% split','paper_test_scope':'Bearing-UAV official 10% split seed=42','candidate_scope':'native p1/p2/p3/p4 only','uses_waypoint_or_temporal_prior':False,'paper_benchmark':paper,'published_reference':pub,'measured_minus_published':delta,'route_results':routes}
 (out/'full_benchmark_and_routes.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
if __name__=='__main__': main()
