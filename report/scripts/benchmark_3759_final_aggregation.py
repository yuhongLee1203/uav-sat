#!/usr/bin/env python3
"""Benchmark final XY reduction from real 3.759 m decoder-probe outputs.

Scope: Weighted Average of 36 scored centers versus Mean Shift's final
aggregation after convergence/merging. The 1,482 saved frames all have one
active Mean Shift mode. Model inference, candidate scoring, mode convergence,
and basin consolidation are outside this timer.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368')
OUT=Path('/yh/study/uav-sat/report/picture')


def reduce_xy(coords,weights):
    return (coords*weights[:,None]).sum(dim=0)


def examples(root,device):
    count=Counter();samples=[]
    for city in ('citya','cityb','cityc','cityd'):
        folder=root/city/'variants'/'decoder_probe'
        summary=json.loads((folder/'bearing_v39_summary.json').read_text())
        for route in ('test_01','test_02'):
            with (folder/Path(summary[route]['CSV']).name).open(newline='') as f:
                rows=list(csv.DictReader(f))
            count.update(int(r['ms_mode_count']) for r in rows)
            r=rows[len(rows)//2]
            centers=torch.tensor(json.loads(r['decoder_candidate_centers_json']),dtype=torch.float32,device=device)
            logits=torch.tensor(json.loads(r['decoder_candidate_logits_json']),dtype=torch.float32,device=device)
            weights=torch.softmax(logits/.30,dim=0)
            assert centers.shape==(36,2)
            w=reduce_xy(centers,weights)
            saved_w=torch.tensor([float(r['decoder_weighted_x']),float(r['decoder_weighted_y'])],device=device)
            if not torch.allclose(w,saved_w,atol=1e-4,rtol=0):
                raise RuntimeError(f'{city} {route}: weighted coordinate mismatch')
            ms=torch.tensor([[float(r['ms_x']),float(r['ms_y'])]],dtype=torch.float32,device=device)
            mw=torch.ones(1,device=device)
            samples.append((city,route,(centers,weights),(ms,mw)))
    assert sum(count.values())==1482
    if set(count)!={1}:
        raise RuntimeError(f'Actual mode count is not uniformly 1: {count}')
    return samples,dict(count)


def single_call(coords,weights,repeats,warmup):
    for _ in range(warmup): reduce_xy(coords,weights)
    torch.cuda.synchronize()
    times=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();reduce_xy(coords,weights);end.record();end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def graphed_device_only(coords,weights,repeats,warmup,ops_per_replay):
    stream=torch.cuda.Stream(device=coords.device)
    stream.wait_stream(torch.cuda.current_stream(coords.device))
    with torch.cuda.stream(stream):
        for _ in range(warmup): reduce_xy(coords,weights)
    torch.cuda.current_stream(coords.device).wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(ops_per_replay):
            result=reduce_xy(coords,weights)
    for _ in range(30): graph.replay()
    torch.cuda.synchronize()
    pairs=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record()
        pairs.append((start,end))
        if len(pairs)==50:
            pairs[-1][1].synchronize()
            yield from (float(a.elapsed_time(b))/ops_per_replay for a,b in pairs)
            pairs=[]
    if pairs:
        pairs[-1][1].synchronize()
        yield from (float(a.elapsed_time(b))/ops_per_replay for a,b in pairs)
    del result


def summarize(values):
    return {'mean_ms':float(statistics.mean(values)),
            'median_ms':float(statistics.median(values)),
            'p90_ms':float(np.percentile(values,90)),
            'samples':len(values)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--single-repeats',type=int,default=150)
    p.add_argument('--graph-repeats',type=int,default=200)
    p.add_argument('--ops-per-replay',type=int,default=64)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for latency benchmark')
    device=torch.device(args.device)
    names=['Weighted Average (36 patches)','Mean Shift final aggregation (1 mode)']
    cases,counts=examples(args.source.resolve(),device)
    data={name:{'single':[],'device_only':[]} for name in names}
    for city,route,weighted,ms in cases:
        for name,(coords,weights) in zip(names,(weighted,ms)):
            data[name]['single'].extend(single_call(coords,weights,args.single_repeats,200))
            data[name]['device_only'].extend(graphed_device_only(coords,weights,args.graph_repeats,200,args.ops_per_replay))
        print(city,route,'measured',flush=True)
    rows=[]
    for name,inputs in zip(names,(36,1)):
        rows.append({'decoder':name,'inputs':inputs,
                     'single_call':summarize(data[name]['single']),
                     'device_only':summarize(data[name]['device_only'])})
    payload={'gpu':torch.cuda.get_device_name(device),'device':str(device),
             'mode_count_histogram':counts,'frames':sum(counts.values()),
             'actual_route_examples':8,'ops_per_graph_replay':args.ops_per_replay,
             'scope':'final coordinate weighted reduction only, after scoring and after Mean Shift convergence/merging',
             'rows':rows}
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'final_aggregation_latency.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Final coordinate aggregation latency','',
           f"GPU: {payload['gpu']} ({args.device}). All {payload['frames']} test frames had exactly one final Mean Shift mode. Eight real mid-route frames (one per route) supply the timed inputs.",'',
           '| Decoder | Inputs | Single-call median (ms) | Single-call P90 (ms) | GPU-only median (ms) | GPU-only P90 (ms) |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        a=row['single_call'];b=row['device_only']
        lines.append(f"| {row['decoder']} | {row['inputs']} | {a['median_ms']:.6f} | {a['p90_ms']:.6f} | {b['median_ms']:.6f} | {b['p90_ms']:.6f} |")
    lines += ['',
        'Timing starts after the 36 candidate coordinates and regularized logits are available. For Mean Shift it starts after all iterations and basin merging, using the already converged one-mode coordinate and weight. Both rows compute only `(weights[:, None] * coordinates).sum(0)`. The single-call GPU event includes the gap between Python kernel launches; the CUDA graph replay amortizes host launch overhead over 64 reductions and reports device execution time per reduction. Neither number is full decoder or end-to-end inference latency.','']
    (out/'final_aggregation_latency.md').write_text('\n'.join(lines))
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()
