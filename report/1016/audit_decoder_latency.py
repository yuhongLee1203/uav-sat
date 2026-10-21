#!/usr/bin/env python3
"""Audit timer scope, GPU kernels and repeatability without replacing original results."""
import json
from pathlib import Path
import numpy as np
import torch
from benchmark_decoder_latency import (FIXTURE, OUT, argmax_xy, weighted_xy, measure, summary, soft_mean_shift)

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    a=np.load(FIXTURE)
    ids=np.flatnonzero(a['route_id']=='a/test_01'); j=int(ids[len(ids)//2])
    c=torch.tensor(a['candidate_centers_xy'][j],device='cuda:0',dtype=torch.float32)
    logits=torch.tensor(a['candidate_regularized_logits'][j],device='cuda:0',dtype=torch.float32)
    w=torch.softmax(logits/.3,dim=0)
    xy,_,m,_,mw,_=soft_mean_shift(logits[None],c[None],.3,7.,3,12.)
    active=mw[0]>0; mc=m[0][active].contiguous(); aw=mw[0][active].contiguous()
    # Original converged seeds, before basin consolidation (outside timed section).
    seeds=c[None].clone()
    for _ in range(3):
        kernel=torch.exp(-torch.cdist(seeds,c[None]).square()/(2*7.**2))
        weights=kernel*w[None,None,:]
        seeds=weights@c[None]/weights.sum(2,keepdim=True).clamp_min(1e-8)
    density=(torch.exp(-torch.cdist(seeds,c[None]).square()/(2*7.**2))*w[None,None,:]).sum(2)
    sw=torch.softmax(12.*density,dim=1)[0]
    assert torch.allclose(weighted_xy(seeds[0],sw),xy[0],atol=.001,rtol=0)
    cases=[('Argmax: search 36 + gather XY',argmax_xy,c,w),
           ('MS: final aggregation after merging (1 mode)',weighted_xy,mc,aw),
           ('Weighted Average: aggregate 36 candidates',weighted_xy,c,w),
           ('MS control: aggregate 36 converged seeds BEFORE merging',weighted_xy,seeds[0],sw),
           ('Argmax control: search 36 only (NO XY gather)',lambda c,w:w.argmax(),c,w)]
    results=[]
    for trial in range(3):
        order=cases[trial:]+cases[:trial]
        for name,fn,coords,weights in order:
            graph,single=measure(fn,coords,weights,graph_reps=150,single_reps=50)
            results.append({'trial':trial+1,'case':name,'gpu_only':summary(graph),'single_call':summary(single)})
        print('Finished trial',trial+1,flush=True)
    profiles=[]
    for name,fn,coords,weights in cases:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(coords,weights)
            torch.cuda.synchronize()
        gpu=[{'kernel':e.name,'duration_us':float(e.cuda_time_total)} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        profiles.append({'case':name,'cuda_kernel_count':len(gpu),'kernels':gpu})
    payload={'gpu':torch.cuda.get_device_name(0),'physical_gpu':0,'sample_key':str(a['sample_key'][j]),'active_mode_count':int(active.sum()),'merged_mode_weights':aw.cpu().tolist(),'trials':results,'profiles':profiles,'scope_note':'Original MS timing starts AFTER basin merging, so it measures 1 mode. 36-seed control starts after iterations but BEFORE merging and times only reduction; it does not time merging. Controls are diagnostic, not replacements for original method results.'}
    (OUT/'audit_results.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()
