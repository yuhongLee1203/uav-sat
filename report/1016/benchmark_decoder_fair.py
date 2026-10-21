#!/usr/bin/env python3
"""Fair prepared-input XY benchmark and explicit post-convergence controls."""
import csv
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
from benchmark_decoder_latency import (FIXTURE, OUT, SOURCE, argmax_xy, weighted_xy, measure, summary, soft_mean_shift, digest)
import visual_localizer

DEST=OUT.parent/'decoder_latency_fair'


def converge(c, p):
    seeds=c.clone()
    for _ in range(3):
        k=torch.exp(-torch.cdist(seeds,c).square()/(2*7.**2))
        weights=k*p[:,None,:]
        seeds=weights@c/weights.sum(2,keepdim=True).clamp_min(1e-8)
    return seeds


def density_weights(c,p,seeds):
    density=(torch.exp(-torch.cdist(seeds,c).square()/(2*7.**2))*p[:,None,:]).sum(2)
    return density,torch.softmax(12.*density,dim=1)


def timed_wall(fn, repeats=20):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize()
        start=time.perf_counter()
        result=fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return values


def original_merge_function():
    # Reuse the original source up to its final XY; omit later support/padding/diagnostics.
    src=inspect.getsource(soft_mean_shift)
    body=src[src.index('    merge_radius ='):src.index('        merged_soft_support.append')]
    env={'torch':torch,'config':visual_localizer.config}
    exec('def merge_xy(modes, density, seed_weights):\n    epsilon=1e-8\n'+body+'    return torch.stack(merged_soft_xy)\n',env)
    return env['merge_xy'],body


@torch.inference_mode()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    a=np.load(FIXTURE)
    c=torch.tensor(a['candidate_centers_xy'],dtype=torch.float32,device='cuda:0')
    logits=torch.tensor(a['candidate_regularized_logits'],dtype=torch.float32,device='cuda:0')
    p=torch.softmax(logits/.30,dim=1)
    seeds_parts=[];density_parts=[];sw_parts=[]
    for first in range(0,len(c),64):
        cb,pb=c[first:first+64],p[first:first+64]
        seeds=converge(cb,pb)
        d,sw=density_weights(cb,pb,seeds)
        seeds_parts.append(seeds);density_parts.append(d);sw_parts.append(sw)
    seeds=torch.cat(seeds_parts);density=torch.cat(density_parts);sw=torch.cat(sw_parts)
    pred={'Argmax':c[torch.arange(len(c),device=c.device),p.argmax(1)],
          'Mean Shift (36 converged seeds)':(seeds*sw[...,None]).sum(1),
          'Weighted Average':(c*p[...,None]).sum(1)}
    reference={'Argmax':'argmax_xy','Mean Shift (36 converged seeds)':'mean_shift_xy','Weighted Average':'weighted_average_xy'}
    verify={name:float(np.linalg.norm(x.cpu().numpy()-a[reference[name]],axis=1).max()) for name,x in pred.items()}
    assert max(verify.values())<.003,verify
    print('All 1482 frame checks passed:',verify,flush=True)
    merge,merge_source=original_merge_function()
    (DEST/'original_merge_timed_source.txt').write_text(merge_source)
    raw=[];route_rows=[];supplement=[];merge_checks=[]
    routes=np.unique(a['route_id'])
    for trial in range(3):
        for ri,route in enumerate(routes):
            ids=np.flatnonzero(a['route_id']==route);j=int(ids[len(ids)//2])
            # Exact same shapes, strides, dtype, coordinate operation for the weighted methods.
            cases=[('Argmax',argmax_xy,c[j],p[j]),('Mean Shift (36 converged seeds)',weighted_xy,seeds[j],sw[j]),('Weighted Average',weighted_xy,c[j],p[j])]
            shift=(trial+ri)%3;cases=cases[shift:]+cases[:shift]
            for name,fn,coords,weights in cases:
                assert coords.shape==(36,2) and coords.is_contiguous() and weights.shape==(36,)
                graph,single=measure(fn,coords,weights,graph_reps=100,single_reps=50)
                route_rows.append({'trial':trial+1,'route':str(route),'decoder':name,'gpu_only':summary(graph),'single_call':summary(single)})
                for kind,values in [('gpu_only',graph),('single_call',single)]:
                    raw.extend({'trial':trial+1,'route':str(route),'decoder':name,'timer':kind,'repeat':k,'ms':v} for k,v in enumerate(values))
            if trial==0:
                sj=seeds[j:j+1];dj=density[j:j+1];wj=sw[j:j+1];cj=c[j:j+1];pj=p[j:j+1]
                got=merge(sj,dj,wj)
                delta=float(torch.linalg.vector_norm(got-pred['Mean Shift (36 converged seeds)'][j]).cpu())
                assert delta<.003
                merge_checks.append(delta)
                # Actual Python/GPU implementation must be timed with synchronized wall clock.
                merged=timed_wall(lambda:merge(sj,dj,wj))
                def post_convergence_xy():
                    d,w=density_weights(cj,pj,sj)
                    return merge(sj,d,w)
                tail=timed_wall(post_convergence_xy)
                def direct_post_convergence(coords,unused):
                    _,w=density_weights(cj,pj,sj)
                    return weighted_xy(sj[0],w[0])
                graph,single=measure(direct_post_convergence,sj[0],wj[0],graph_reps=100,single_reps=50)
                supplement.append({'route':str(route),'original_merge_after_weights_ready_wall_ms':summary(merged),'original_full_tail_after_iterations_wall_ms':summary(tail),'direct_xy_after_iterations_including_density_and_weights_gpu_ms':summary(graph)})
                for kind,values in [('original_merge_ready_weights_wall',merged),('original_tail_post_iterations_wall',tail),('direct_tail_including_density_gpu',graph)]:
                    raw.extend({'trial':1,'route':str(route),'decoder':'MS supplemental','timer':kind,'repeat':k,'ms':v} for k,v in enumerate(values))
            print('Trial',trial+1,'route',route,'complete',flush=True)
    rows=[]
    for name,x in pred.items():
        e=np.linalg.norm(x.cpu().numpy()-a['gt_xy'],axis=1)
        gv=[r['ms'] for r in raw if r['decoder']==name and r['timer']=='gpu_only']
        sv=[r['ms'] for r in raw if r['decoder']==name and r['timer']=='single_call']
        rows.append({'decoder':name,'N':len(e),'timed_inputs':36,'MLE_m':float(e.mean()),'LSR3_pct':float(100*(e<=3).mean()),'LSR5_pct':float(100*(e<=5).mean()),'gpu_only':summary(gv),'single_call':summary(sv),'gpu_measurements':len(gv),'single_measurements':len(sv)})
    payload={'physical_gpu':0,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'batch_size':1,'dtype':'float32','trials':3,'latency_frames':8,'accuracy_frames':1482,'source_suite':str(SOURCE),'weights_manifest':str(OUT/'weights_manifest.json'),'fixture_sha256':digest(FIXTURE),'original_runtime_sha256':digest(Path(inspect.getsourcefile(soft_mean_shift))),'verify_max_xy_delta_m':verify,'merged_vs_direct_max_xy_delta_m_on_timed_frames':max(merge_checks),'scope':'Primary: all 36 coordinates and normalized weights ready on GPU. MS uses 36 iterated seeds, BEFORE merge, with density/beta weights already prepared. Same multiply/sum implementation as WA; no K=1 shortcut. Argmax searches 36 prepared probabilities and gathers XY. Convergence, scoring/weight preparation excluded for all methods. XY-only MS mathematically eliminates merge, validated against original. This is equivalent-output implementation, not original merging runtime. GPU CUDA graph:128 serial B=1 ops/replay; 100 replays per route/trial/method.','rows':rows,'per_route_trial':route_rows,'supplemental':supplement}
    (DEST/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (DEST/'timing_raw.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(raw[0]));wr.writeheader();wr.writerows(raw)
    np.savez_compressed(DEST/'predictions.npz',**{k.replace(' ','_'):v.cpu().numpy() for k,v in pred.items()},gt_xy=a['gt_xy'])
    lines=['# 公平比較：權重備妥後到最終 XY（2026-10-17）','','共同起點：batch=1、FP32，36個座標及正規化權重均已在GPU0。Argmax搜尋36個分數並讀XY；WA加總36個候選；MS加總36個完成原版3次迭代的seed，未先合併成1點。MS的density／β權重亦於計時前備妥。','','| 方法 | GPU median (ms) | GPU P10–P90 (ms) | 單次呼叫 median (ms) | MLE (m) |','| --- | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['decoder']} | {r['gpu_only']['median_ms']:.8f} | {r['gpu_only']['p10_ms']:.8f}–{r['gpu_only']['p90_ms']:.8f} | {r['single_call']['median_ms']:.8f} | {r['MLE_m']:.6f} |")
    lines+=['','GPU0 RTX3090、8條route各1真實幀、3輪輪替順序；每方法2,400次graph量測（每次128次序列batch=1運算後除128）、1,200次單次呼叫。誤差使用全部1,482筆。GPU時間為graph device時間；單次呼叫包含host發射間隙，不能混用。','','為何可以省略合併：每群mode位置 `μg=Σ(wi·mi)/Wg`、群權重 `Wg=Σwi`，最終 `Σ(Wg·μg)=Σ(wi·mi)`。本資料權重皆為正且群權重大於原epsilon，因此可直接加總36個收斂seed，無須產出群數等metadata；誤差及與原版位置差異已逐筆核對。這是明確採用等價XY輸出的實作，不能當作原Python合併流程的計時。','','這個公平切點下，MS與WA使用同樣形狀、同一個乘法與加總函式，應視為相同量級。微秒級小差異不支持固定宣稱MS一定比WA快。Argmax也不是零成本，需reduction再gather。','','## 補充：若「收斂後」只代表迭代完成','','下表保留原碼的合併路徑，只執行至XY（不含之後padding、compact等metadata），同步wall clock包含Python控制與GPU同步；不可拿來與graph純GPU欄直接比較。density/β是計算seed權重的後處理，若只排除迭代，它們也應納入。','','| Route | 權重已備妥，原版合併→XY wall (ms) | 剛完成迭代，density/β＋原版合併→XY wall (ms) | 剛完成迭代，density/β＋等價直接加總 GPU (ms) |','| --- | ---: | ---: | ---: |']
    for r in supplement:lines.append(f"| {r['route']} | {r['original_merge_after_weights_ready_wall_ms']['median_ms']:.6f} | {r['original_full_tail_after_iterations_wall_ms']['median_ms']:.6f} | {r['direct_xy_after_iterations_including_density_and_weights_gpu_ms']['median_ms']:.6f} |")
    lines+=['','只排除迭代與「所有權重均已備妥」是不同前提。主表回答原需求中的權重已分配好、到XY的運算成本；補充表展示真正保留原MS後處理時的成本。沒有要求時間小於0.01ms或指定排序來改動量測。','','原始權重：[固定參考](../WEIGHTS_REFERENCE.md)。完整結果／逐次時間：results.json、timing_raw.csv。原始歷史結果仍保留於../decoder_latency/；本次應優先引用此表並註明prepared-weights、post-convergence XY-only。','','重跑：`python3 report/1016/benchmark_decoder_fair.py`。','']
    (DEST/'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
