#!/usr/bin/env python3
"""GPU 0 final-coordinate benchmark; no inference/convergence in timers."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/yh/study/uav-sat')
OUT = ROOT / 'report/1016/decoder_latency'
SOURCE = ROOT / 'v39_otherdata/output/original3759_inference_20260925_224113_1642368'
FIXTURE = ROOT / 'report/add_more/thesis_experiments/raw/final_decoder_predictions.npz'
sys.path.insert(0, str(SOURCE / 'citya/variants/decoder_probe/runtime'))
from visual_localizer import soft_mean_shift


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def argmax_xy(c, w):
    return torch.index_select(c, 0, w.argmax().reshape(1)).squeeze(0)


def weighted_xy(c, w):
    return (c * w[:, None]).sum(0)


def summary(a):
    return {k: float(v) for k, v in zip(
        ('mean_ms', 'median_ms', 'p10_ms', 'p90_ms'),
        (np.mean(a), np.median(a), np.percentile(a, 10), np.percentile(a, 90)))}


def measure(fn, c, w, graph_reps=200, single_reps=150, operations=128):
    for _ in range(200):
        fn(c, w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(operations):
            result = fn(c, w)
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    graph_times = []
    for _ in range(graph_reps):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        graph_times.append(start.elapsed_time(end) / operations)
    single_times = []
    for _ in range(single_reps):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = fn(c, w)
        end.record()
        end.synchronize()
        single_times.append(start.elapsed_time(end))
    return graph_times, single_times


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    a = np.load(FIXTURE)
    c = torch.tensor(a['candidate_centers_xy'], device='cuda:0', dtype=torch.float32)
    logits = torch.tensor(a['candidate_regularized_logits'], device='cuda:0', dtype=torch.float32)
    w = torch.softmax(logits / .30, dim=1)
    assert c.shape == (1482, 36, 2)
    assert np.max(np.abs(w.cpu().numpy() - a['candidate_weights'])) < 1e-5
    modes, weights, ms = [], [], []
    for first in range(0, len(c), 64):
        xy, _, m, _, mw, _ = soft_mean_shift(logits[first:first+64], c[first:first+64], .30, 7., 3, 12.)
        modes.append(m)
        weights.append(mw)
        ms.append(xy)
    modes, weights, ms = torch.cat(modes), torch.cat(weights), torch.cat(ms)
    counts = (weights > 0).sum(1)
    histogram = dict(Counter(counts.cpu().tolist()))
    preds = {
        'Argmax': c[torch.arange(len(c), device=c.device), logits.argmax(1)].cpu().numpy(),
        'Mean Shift': ms.cpu().numpy(),
        'Weighted Average': (c * w[..., None]).sum(1).cpu().numpy(),
    }
    checks = {}
    for name, key in [('Argmax', 'argmax_xy'), ('Mean Shift', 'mean_shift_xy'), ('Weighted Average', 'weighted_average_xy')]:
        checks[name] = float(np.linalg.norm(preds[name] - a[key], axis=1).max())
        assert checks[name] < .003, (name, checks[name])
    print('Recomputed all 1482 predictions; mode histogram:', histogram, 'max differences:', checks, flush=True)
    route_ids = np.unique(a['route_id'])
    timing, raw, route_metrics = [], [], []
    pools = {name: {'graph': [], 'single': []} for name in preds}
    for rindex, route in enumerate(route_ids):
        ids = np.flatnonzero(a['route_id'] == route)
        j = int(ids[len(ids)//2])
        active = weights[j] > 0
        mc, mw = modes[j][active].contiguous(), weights[j][active].contiguous()
        cases = [('Argmax', argmax_xy, c[j], w[j]), ('Mean Shift', weighted_xy, mc, mw), ('Weighted Average', weighted_xy, c[j], w[j])]
        # Rotate execution order across routes to reduce systematic order effects.
        cases = cases[rindex % 3:] + cases[:rindex % 3]
        for name, fn, coords, weight in cases:
            graph, single = measure(fn, coords, weight)
            pools[name]['graph'].extend(graph)
            pools[name]['single'].extend(single)
            timing.append({'route': str(route), 'sample_key': str(a['sample_key'][j]), 'decoder': name, 'inputs': len(coords), 'gpu_only': summary(graph), 'single_call': summary(single)})
            for kind, values in [('gpu_only', graph), ('single_call', single)]:
                raw.extend({'route': str(route), 'decoder': name, 'timer': kind, 'repeat': k, 'ms': t} for k, t in enumerate(values))
        for name, p in preds.items():
            errors = np.linalg.norm(p[ids] - a['gt_xy'][ids], axis=1)
            route_metrics.append({'route': str(route), 'decoder': name, 'N': len(ids), 'MLE_m': float(errors.mean()), 'LSR3_pct': float(100*(errors<=3).mean()), 'LSR5_pct': float(100*(errors<=5).mean())})
        print('Measured:', route, flush=True)
    rows = []
    for name, p in preds.items():
        e = np.linalg.norm(p-a['gt_xy'], axis=1)
        rows.append({'decoder': name, 'N': len(e), 'MLE_m': float(e.mean()), 'LSR3_pct': float(100*(e<=3).mean()), 'LSR5_pct': float(100*(e<=5).mean()), 'gpu_only': summary(pools[name]['graph']), 'single_call': summary(pools[name]['single'])})
    checkpoints = []
    for city in ('citya', 'cityb', 'cityc', 'cityd'):
        for name in ('visual_retrieval_A_only.pt', 'controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt'):
            p = SOURCE/city/'train_frames3/checkpoints'/name
            checkpoints.append({'city': city, 'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size})
        p = SOURCE/city/'train_frames3/kalman_calibration.json'
        checkpoints.append({'city': city, 'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size})
    provenance = {'source_suite': str(SOURCE), 'checkpoints_and_calibration': checkpoints, 'fixture': str(FIXTURE), 'fixture_sha256': digest(FIXTURE), 'protocol': 'Original GT-assisted v13, 4m, frames3, full; frozen candidates/logits; conditional final decoder comparison', 'parameters': {'tau': .30, 'bandwidth_m': 7., 'iterations': 3, 'beta': 12., 'grid': '6x6'}}
    (OUT/'weights_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    payload = {'created_at': datetime.now().astimezone().isoformat(), 'gpu': torch.cuda.get_device_name(0), 'physical_gpu': 0, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'batch_size': 1, 'operations_per_graph': 128, 'graph_repeats_per_route_method': 200, 'single_repeats_per_route_method': 150, 'accuracy_frames': 1482, 'latency_route_examples': 8, 'mode_count_histogram': histogram, 'reproduction_max_xy_difference_m': checks, 'scope': 'Inputs and normalized weights ready on GPU. Argmax includes max-index search and coordinate gather; weighted average includes multiply and sum; Mean Shift includes actual active-mode multiply and sum after convergence AND basin consolidation. Excludes scoring, softmax, allocation of inputs, model, convergence, consolidation, CPU transfers. CUDA graph outputs allocated during capture, reused during replay. GPU-only is per B=1 operation averaged over 128 serial graph operations, not throughput divided by batch size. Single-call includes host dispatch gaps. No forced ordering.', 'rows': rows, 'per_route_latency': timing, 'per_route_accuracy': route_metrics, 'provenance': provenance}
    (OUT/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    with (OUT/'timing_raw.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    with (OUT/'accuracy_by_route.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(route_metrics[0])); writer.writeheader(); writer.writerows(route_metrics)
    np.savez_compressed(OUT/'predictions.npz', **{name.replace(' ', '_'): p for name, p in preds.items()}, gt_xy=a['gt_xy'], sample_key=a['sample_key'], route_id=a['route_id'])
    lines = ['# 三種解碼：最終座標運算時間與誤差', '', 'GPU 0：RTX 3090；batch=1、FP32。誤差使用全部1,482筆；時間使用8條route各一筆中間幀，每方法各1,600次CUDA graph與1,200次單次呼叫量測。', '', '| 方法 | GPU純運算 median (ms) | GPU P90 (ms) | 單次呼叫 median (ms) | MLE (m) | LSR@3 (%) | LSR@5 (%) |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['decoder']} | {row['gpu_only']['median_ms']:.8f} | {row['gpu_only']['p90_ms']:.8f} | {row['single_call']['median_ms']:.8f} | {row['MLE_m']:.6f} | {row['LSR3_pct']:.2f} | {row['LSR5_pct']:.2f} |")
    speed = sorted(rows, key=lambda x: x['gpu_only']['median_ms'])
    error = sorted(rows, key=lambda x: x['MLE_m'])
    lines += ['', 'GPU純運算耗時（小→大）：'+' < '.join(r['decoder'] for r in speed)+'。', 'MLE（小→大、越小越準）：'+' < '.join(r['decoder'] for r in error)+'。', '', '計時起點：所有座標、正規化權重都已放在GPU。Argmax包含找最大索引和取出XY；Weighted Average包含36點乘權重後加總；Mean Shift從完成迭代與basin合併後的實際active modes開始乘權重與加總。全部1,482筆實際皆為1個mode，沒有預設或捏造K。MS modes及權重由原保存runtime重新計算，與保存位置最大差異見results.json。', '', 'GPU純運算使用CUDA graph：每次replay內128次batch=1運算，CUDA events量整段後除128；排除Python逐kernel發射的空隙，並重用capture時分配的輸出buffer。單次呼叫另列，含host發射間隔。這不是完整decoder、端到端推論時間，也不是MS收斂時間；不能用此欄宣稱整體MS推論較快。', '', '誤差為同一組保存36候選／regularized logits的條件式比較，不回灌tracker；原始協定使用GT-assisted reference。權重路徑及SHA-256保存於weights_manifest.json。沒有因預期排序修改結果。', '', '重跑：`python3 /yh/study/uav-sat/report/1016/benchmark_decoder_latency.py`。原始逐次時間：timing_raw.csv；逐route誤差：accuracy_by_route.csv；完整結果：results.json。', '']
    (OUT/'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
