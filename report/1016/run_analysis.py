#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,json,csv,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/yh/study/uav-sat')
SUITE=ROOT/'v39_otherdata/output/original3759_inference_20260925_224113_1642368'
OUT=ROOT/'report/1016'
sys.path.insert(0,str(ROOT/'v39_otherdata'))
import bearing_prepare as bp
sys.path.insert(0,str(SUITE/'citya/variants/decoder_probe/runtime'))
from visual_localizer import soft_mean_shift
VARIANTS=['full','no_gru','no_kalman','no_ms','frames1','frames2','grid4','grid5','grid7','grid8']
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def md(p,rows,note=''):
 cols=list(rows[0]);lines=['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
 lines+=['| '+' | '.join(str(row[k]) for k in cols)+' |' for row in rows]
 p.write_text('\n'.join(lines)+'\n\n'+note+'\n')
def metric(e):return {'MLE (m)':float(np.mean(e)),**{'LSR@%s (%%)'%k:100*float(np.mean(e<=k)) for k in (3,5,15)}}
def rounded(d):return {k:f'{v:.3f}' if 'MLE' in k else f'{v:.2f}' for k,v in d.items()}
def xy(rows,p):return np.array([[float(r[p+'_x']),float(r[p+'_y'])] for r in rows])
def localization():
 meta=pd.read_csv(json.loads((SUITE/'citya/prepared/experiment.json').read_text())['metadata_csv'])
 route_rows=[];totals={v:[] for v in VARIANTS};audit=[]
 for c in 'abcd':
  city='city'+c;prep=SUITE/city/'prepared';citymeta=bp._city_rows(meta,city)
  train=read(prep/'routes/train_01/manifest.csv');origin=np.array([float(train[0]['x_m']),float(train[0]['y_m'])])
  for route in ('test_01','test_02'):
   man=read(prep/'routes'/route/'manifest.csv');labels=[];centers=[]
   for m in man:
    row=citymeta.iloc[int(m['source_index'])]
    assert Path(row['target_path']).name==Path(m['image_path']).name
    center=np.array([float(row['block_x'])*256+256,float(row['block_y'])*256+256])*.25
    gt_abs=center+np.array([float(row['x_norm']),float(row['y_norm'])])*32
    assert np.linalg.norm(gt_abs-np.array([float(m['x_m']),float(m['y_m'])]))<1e-4
    labels.append(np.sign([float(row['x_norm']),float(row['y_norm'])]));centers.append(center-origin)
   centers=np.array(centers);labels=np.array(labels)
   for variant in VARIANTS:
    folder=SUITE/city/'variants'/variant;s=json.loads((folder/'bearing_v39_summary.json').read_text())[route]
    rows=read(folder/Path(s['CSV']).name);assert len(rows)==len(man)
    assert all(r['image_path']==m['image_path'] and r['frame_id']==m['frame_id'] for r,m in zip(rows,man))
    gt,pred=xy(rows,'gt'),xy(rows,'final');e=np.linalg.norm(pred-gt,axis=1)
    assert np.max(np.abs(e-np.array([float(r['error_final_m']) for r in rows])))<1e-4
    assert np.array_equal(np.sign(gt-centers),labels) and not np.any(labels==0)
    correct=np.all(np.sign(pred-centers)==labels,axis=1);totals[variant].append((e,correct))
    if variant=='full':
     route_rows.append({'City':c.upper(),'Route':route,'N':len(e),'Four-tile coordinate R@1 (%)':f'{100*correct.mean():.2f}',**rounded(metric(e))})
     audit.extend([{'city':city,'route':route,'frame_id':r['frame_id'],'four_tile_correct':bool(correct[i]),'error_m':float(e[i])} for i,r in enumerate(rows)])
 ablation=[]
 for v,items in totals.items():
  e=np.concatenate([x[0] for x in items]);good=np.concatenate([x[1] for x in items]);assert len(e)==1482
  row={'Variant':v,'N':len(e),'Four-tile coordinate R@1 (%)':f'{100*good.mean():.2f}',**rounded(metric(e))};ablation.append(row)
  if v=='full':
   assert abs(e.mean()-3.7591276968535423)<1e-6
   route_rows.append({'City':'Overall','Route':'—',**{k:v for k,v in row.items() if k!='Variant'}})
 note='四 tile 座標衍生 R@1：final XY 相對已知 RSB 中心的象限，依 Bearing-UAV 原始 RECALL_AT_K_PHR sign 規則計算。沒有全 gallery 影像排名。原始 3.759 m 使用 current-frame GT 輔助搜尋／hypothesis selection；本表不能宣稱與原論文同協定。LSR 以距離 ≤ threshold 計算。'
 md(OUT/'01_main_localization.md',route_rows,note)
 md(OUT/'02_ablation_localization.md',ablation,note+'\n\n歷史文字差異：本次保存 replay 的 no_kalman LSR@5=73.82%，不是先前稿件的73.68%；MLE同為3.763m。此處保留逐幀重算結果。')
 (OUT/'localization_frame_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
def sensitivity():
 torch.set_num_threads(2);assert torch.cuda.is_available()
 a=np.load(ROOT/'report/add_more/thesis_experiments/raw/final_decoder_predictions.npz');N=len(a['gt_xy'])
 C=torch.tensor(a['candidate_centers_xy'],dtype=torch.float32,device='cuda');S=torch.tensor(a['candidate_regularized_logits'],dtype=torch.float32,device='cuda')
 assert C.shape==(1482,36,2)
 @torch.inference_mode()
 def decode(tau,bw,it,beta):
  return np.vstack([soft_mean_shift(S[i:i+64],C[i:i+64],tau,bw,it,beta)[0].cpu().numpy() for i in range(0,N,64)])
 baseline=decode(.3,7.,3,12.);delta=np.linalg.norm(baseline-a['mean_shift_xy'],axis=1)
 if delta.max()>.003:raise RuntimeError(f'Baseline mismatch {delta.max()}')
 ids=[np.flatnonzero(a['route_id']==r)[len(np.flatnonzero(a['route_id']==r))//2] for r in np.unique(a['route_id'])]
 @torch.inference_mode()
 def latency(key):
  for j in ids:soft_mean_shift(S[j:j+1],C[j:j+1],*key)
  times=[]
  for rep in range(3):
   for j in ids:
    torch.cuda.synchronize();t=time.perf_counter();soft_mean_shift(S[j:j+1],C[j:j+1],*key);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
  return float(np.median(times))
 settings=[]
 for v in [2,4,5,7,10,14,20]:settings.append(('bandwidth_m',v,(.3,v,3,12.)))
 for v in [.1,.2,.3,.5,.7,1.]:settings.append(('temperature',v,(v,7.,3,12.)))
 for v in [1,2,3,5,8]:settings.append(('iterations',v,(.3,7.,v,12.)))
 for v in [0,4,8,12,20]:settings.append(('mode_beta',v,(.3,7.,3,v)))
 results=[];cache={};predictions={}
 for name,value,key in settings:
  if key not in cache:
   pred=baseline if key==(.3,7.,3,12.) else decode(*key)
   cache[key]={**metric(np.linalg.norm(pred-a['gt_xy'],axis=1)),'Decoder median latency (ms)':latency(key)}
   predictions['_'.join(map(str,key))]=pred
  row={'Parameter':name,'Value':value,'Default':'yes' if key==(.3,7.,3,12.) else '',**cache[key]};results.append(row);print('[SWEEP]',json.dumps(row),flush=True)
 note='條件式 final-decoder sensitivity：只重算保存的同一組 6×6／36 candidates 與 regularized logits。前段模型、GRU、Kalman、候選位置及後續時序狀態固定；不是完整 tracker 重播，也未重新訓練。Logits 含 GT-conditioned reference/Kalman priors。一次只變一個參數；baseline τ=0.30、bandwidth=7 m、iterations=3、β=12。測試集最佳列僅作探索性分析，不作新模型選擇。Latency：GPU 0、batch=1、8條 route 各取中間一幀，重複3次共24次，CUDA同步，原 decoder 含 basin consolidation 的中位數；排除 candidate scoring。'
 for param,stem in [('bandwidth_m','03_bandwidth'),('temperature','04_temperature'),('iterations','05_iterations'),('mode_beta','06_mode_beta')]:
  selected=[r for r in results if r['Parameter']==param]
  rows=[{k:(f'{v:.4f}' if isinstance(v,float) and k!='Value' else v) for k,v in r.items() if k!='Parameter'} for r in selected]
  md(OUT/(stem+'.md'),rows,note)
  fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
  for ax,m in zip(axes,['MLE (m)','LSR@5 (%)']):ax.plot([r['Value'] for r in selected],[r[m] for r in selected],'-o');ax.set_xlabel(param);ax.set_ylabel(m);ax.grid(alpha=.25)
  fig.suptitle('Fixed 6×6 final-decoder sensitivity (GT-assisted replay)');fig.savefig(OUT/(stem+'.png'),dpi=200);fig.savefig(OUT/(stem+'.pdf'));plt.close(fig)
 np.savez_compressed(OUT/'sensitivity_predictions.npz',**predictions)
 nearest=np.linalg.norm(a['candidate_centers_xy']-a['gt_xy'][:,None],axis=2).argmin(1);top=a['candidate_regularized_logits'].argmax(1)
 payload={'baseline_max_position_difference_m':float(delta.max()),'gpu':torch.cuda.get_device_name(0),'settings':results,'local_nearest_center_top1_pct':float(100*np.mean(nearest==top))}
 (OUT/'sensitivity_results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':
 OUT.mkdir(exist_ok=True,parents=True);localization();sensitivity();print('[DONE]',OUT,flush=True)
