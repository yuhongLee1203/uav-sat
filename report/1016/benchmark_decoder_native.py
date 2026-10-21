#!/usr/bin/env python3
"""Actual post-iteration implementation timing, identical wall-clock measurement."""
import csv
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
from benchmark_decoder_fair import converge, density_weights, original_merge_function
from benchmark_decoder_latency import FIXTURE, OUT, SOURCE, argmax_xy, weighted_xy, summary, digest, soft_mean_shift

DEST=OUT.parent/'decoder_latency_native'
REPEATS=50
TRIALS=3


@torch.inference_mode()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    a=np.load(FIXTURE)
    c=torch.tensor(a['candidate_centers_xy'],device='cuda:0',dtype=torch.float32)
    logits=torch.tensor(a['candidate_regularized_logits'],device='cuda:0',dtype=torch.float32)
    p=torch.softmax(logits/.30,dim=1)
    seed_parts=[]
    for start in range(0,len(c),64):seed_parts.append(converge(c[start:start+64],p[start:start+64]))
    seeds=torch.cat(seed_parts)
    merge,body=original_merge_function()
    (DEST/'original_merge_timed_source.txt').write_text(body)
    def ms_tail(c,p,seeds):
        density,weights=density_weights(c,p,seeds)
        return merge(seeds,density,weights)
    preds={'Argmax':[], 'Weighted Average':[], 'Mean Shift (original post-iteration tail)':[]}
    for start in range(0,len(c),64):
        cb,lb,pb,sb=c[start:start+64],logits[start:start+64],p[start:start+64],seeds[start:start+64]
        preds['Argmax'].append(cb[torch.arange(len(cb),device=c.device),lb.argmax(1)])
        preds['Weighted Average'].append((cb*torch.softmax(lb/.30,dim=1)[...,None]).sum(1))
        preds['Mean Shift (original post-iteration tail)'].append(ms_tail(cb,pb,sb))
    preds={name:torch.cat(values).cpu().numpy() for name,values in preds.items()}
    refs={'Argmax':'argmax_xy','Weighted Average':'weighted_average_xy','Mean Shift (original post-iteration tail)':'mean_shift_xy'}
    checks={name:float(np.linalg.norm(v-a[refs[name]],axis=1).max()) for name,v in preds.items()}
    assert max(checks.values())<.003,checks
    print('All 1482 predictions validated:',checks,flush=True)
    # Bring GPU out of idle before timing any method.
    scratch=torch.randn(512,512,device='cuda:0')
    for _ in range(100): scratch@scratch
    torch.cuda.synchronize()
    raw=[];by_route=[]
    for trial in range(TRIALS):
        for ri,route in enumerate(np.unique(a['route_id'])):
            ids=np.flatnonzero(a['route_id']==route);j=int(ids[len(ids)//2])
            cj,lj,pj,sj=c[j:j+1],logits[j],p[j:j+1],seeds[j:j+1]
            cases=[('Argmax',lambda:argmax_xy(cj[0],lj)),
                   ('Weighted Average',lambda:weighted_xy(cj[0],torch.softmax(lj/.30,dim=0))),
                   ('Mean Shift (original post-iteration tail)',lambda:ms_tail(cj,pj,sj))]
            shift=(trial+ri)%3;cases=cases[shift:]+cases[:shift]
            for name,fn in cases:
                for _ in range(10):fn()
                torch.cuda.synchronize()
                values=[]
                for repeat in range(REPEATS):
                    torch.cuda.synchronize()
                    start=time.perf_counter_ns()
                    result=fn()
                    torch.cuda.synchronize()
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    values.append(elapsed)
                    raw.append({'trial':trial+1,'route':str(route),'sample_key':str(a['sample_key'][j]),'decoder':name,'repeat':repeat,'ms':elapsed})
                by_route.append({'trial':trial+1,'route':str(route),'decoder':name,'wall_clock':summary(values)})
            print('Trial',trial+1,route,'done',flush=True)
    rows=[]
    for name,pred in preds.items():
        values=[r['ms'] for r in raw if r['decoder']==name]
        errors=np.linalg.norm(pred-a['gt_xy'],axis=1)
        rows.append({'decoder':name,'samples':len(values),'wall_clock':summary(values),'MLE_m':float(errors.mean()),'LSR3_pct':float(100*(errors<=3).mean()),'LSR5_pct':float(100*(errors<=5).mean())})
    payload={'gpu':torch.cuda.get_device_name(0),'physical_gpu':0,'torch':torch.__version__,'cuda':torch.version.cuda,'batch_size':1,'dtype':'float32','trials':TRIALS,'repeats_per_route_trial':REPEATS,'latency_frames':8,'accuracy_frames':len(c),'source_suite':str(SOURCE),'fixture_sha256':digest(FIXTURE),'runtime_sha256':digest(Path(inspect.getsourcefile(soft_mean_shift))),'weight_reference':str(OUT/'weights_manifest.json'),'max_xy_difference_from_saved_m':checks,'scope':'All inputs resident on GPU. Argmax: original candidate logits -> argmax -> gather XY. WA: logits -> divide tau -> softmax -> weighted XY. MS: original 3 configured iterations already finished; original probabilities and 36 iterated seeds resident -> density -> beta softmax -> original Python connected basin merge -> XY. Omit only iterations, pre-iteration steps, and post-XY support/padding/compact outputs. No equivalent-output replacement and no pre-merge shortcut. Same synchronous perf_counter_ns wall clock for all three including host control and GPU completion. No CUDA graphs, no per-batch division. This is conditional post-convergence implementation latency, not full MS algorithm runtime.','rows':rows,'per_route_trial':by_route}
    (DEST/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (DEST/'timing_raw.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(raw[0]));wr.writeheader();wr.writerows(raw)
    np.savez_compressed(DEST/'predictions.npz',**{k.replace(' ','_'):v for k,v in preds.items()},gt_xy=a['gt_xy'])
    lines=['# 原實作比較：MS只排除迭代（2026-10-17）','','本次不把MS改寫成直接加權平均，也不先合併成1點。GPU0 RTX3090、batch=1、FP32；8條route各1幀、3輪輪替順序、每輪每方法每幀50次，共每方法1,200次。誤差在全部1,482筆核對。','','## 明確起訖','','- Argmax：36候選座標與原logits已在GPU → 找最大索引 → 讀取XY。','- Weighted Average：相同座標／logits已在GPU → τ縮放／softmax → 36點加權加總 → XY。','- MS：原版設定的3次迭代已完成，36個收斂seed及迭代用候選機率已在GPU → density → β softmax → 原碼basin合併 → 最終XY。','- MS原始機率是迭代已必需的中間狀態，沒有重算；density及seed權重不預先計算。','- 三者統一使用perf_counter_ns，前後CUDA同步；包含實際Python發射／控制流程與GPU完成時間。沒有使用CUDA graph或除以batch數。排除模型特徵與candidate scoring；不計輸出XY以後的padding／compact等metadata。','','| 方法 | 實際耗時 median (ms) | P10–P90 (ms) | MLE (m) |','| --- | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['decoder']} | {r['wall_clock']['median_ms']:.6f} | {r['wall_clock']['p10_ms']:.6f}–{r['wall_clock']['p90_ms']:.6f} | {r['MLE_m']:.6f} |")
    lines+=['','這是「MS迭代已完成」條件下的原實作剩餘解碼時間；與舊graph純GPU加總數值口徑不同，不能直接混成同一表。MS合併原碼含逐seed的nonzero／tolist及Python集合遍歷，涉及GPU到CPU同步，因此實際延遲明顯較高。此數字描述目前原碼，不表示所有MS GPU實作都有相同成本。','','原版同一批資料固定執行3次迭代，沒有額外檢查數學收斂容差；此處「收斂後」沿用原版迭代已完成的狀態。權重、候選與分數來源維持1016原版本；沒有重訓，也沒有更改輸出以符合速度或精度排序。','','完整數值：results.json；逐次時間：timing_raw.csv；被計時的原合併片段：original_merge_timed_source.txt。','','重跑：`python3 report/1016/benchmark_decoder_native.py`。','']
    (DEST/'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
