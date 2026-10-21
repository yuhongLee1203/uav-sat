#!/usr/bin/env python3
"""Paper tables/trajectory figures from the audited v39 3.759 m replay."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

CITIES=('citya','cityb','cityc','cityd')
ROUTES=('test_01','test_02')
DEFAULT=Path('/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368')
OUT=Path('/yh/study/uav-sat/report/picture')


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def route_rows(root, city, variant, route):
    folder=root/city/'variants'/variant
    summary=json.loads((folder/'bearing_v39_summary.json').read_text())[route]
    path=folder/Path(summary['CSV']).name
    rows=csv_rows(path)
    assert rows, path
    return rows, summary, path


def stats(errors):
    a=np.asarray(errors,dtype=float)
    assert a.size and np.isfinite(a).all()
    return len(a),float(a.mean()),float(100*np.mean(a<=3)),float(100*np.mean(a<=15))


def save_table(path, header, rows, notes=()):
    text=['| '+' | '.join(header)+' |','| '+' | '.join('---' for _ in header)+' |']
    text += ['| '+' | '.join(map(str,row))+' |' for row in rows]
    if notes: text += ['',*notes]
    path.write_text('\n'.join(text)+'\n')


def route_table(root,out):
    entries=[]; all_errors=[]; latency=[]; panels={}
    for city in CITIES:
        for route in ROUTES:
            rows,summary,path=route_rows(root,city,'full',route)
            manifest=csv_rows(root/city/'prepared'/'routes'/route/'manifest.csv')
            assert len(rows)==len(manifest)
            errors=[]
            for row,man in zip(rows,manifest):
                e=math.hypot(float(row['final_x'])-float(row['gt_x']),
                             float(row['final_y'])-float(row['gt_y']))
                assert abs(e-float(row['error_final_m']))<1e-4
                errors.append(e)
            n,mle,l3,l15=stats(errors)
            assert abs(mle-summary['MLE_m'])<1e-6
            entries.append((city,route,n,l3,l15,mle))
            all_errors.extend(errors)
            timing=summary['EndToEndTiming']
            latency.append((city,route,timing['samples'],timing['mean_ms']))
            panels[(city,route)]=(rows,manifest,n,mle,l3)
    n,mle,l3,l15=stats(all_errors)
    assert n==1482 and abs(mle-3.7591276968535423)<1e-6
    body=[(c.upper().replace('CITY','City '),r,str(n),f'{l3:.2f}%',f'{l15:.2f}%',f'{m:.3f}')
          for c,r,n,l3,l15,m in entries]
    body.append(('Overall','—',str(n),f'{l3:.2f}%',f'{l15:.2f}%',f'{mle:.3f}'))
    save_table(out/'route_wise_results.md',
        ('City','Route','Samples','LSR@3 ↑','LSR@15 ↑','MLE (m) ↓'),body,
        ('Metrics use each frame’s recorded GT coordinate. Overall is pooled across 1,482 observations, not an unweighted average of routes.',
         'Protocol: v13 synthetic 4 m routes; current-frame GT centers the local search and selects the acquisition hypothesis.'))
    total_timing=sum(x[2] for x in latency)
    mean_ms=sum(x[2]*x[3] for x in latency)/total_timing
    save_table(out/'inference_runtime.md',('Item','Value'),[
        ('GPU','NVIDIA GPU 0/5/6 across parallel city evaluation'),
        ('End-to-end mean latency',f'{mean_ms:.2f} ms/frame'),
        ('Throughput equivalent',f'{1000/mean_ms:.2f} FPS'),
        ('Timing samples',str(total_timing)),
        ('Final 6×6 Mean Shift latency',f"{sum(len(route_rows(root,c,'full',r)[0])*route_rows(root,c,'full',r)[1]['MS_LatencyMean_ms'] for c in CITIES for r in ROUTES)/1482:.2f} ms/frame"),
    ],('End-to-end timing starts at a prepared UAV tensor and excludes disk I/O, preprocessing, model loading, and satellite gallery construction. GPU models and concurrent load are in the replay environment; this is a measured replay, not a deployment benchmark.',))
    return panels,entries,(n,mle,l3,l15)


def waypoint_xy(root,city,route):
    source=json.loads((root/city/'prepared'/'routes'/route/'waypoints.json').read_text())
    points=sorted(source['waypoints'],key=lambda w:int(w['waypoint_order']))
    manifest=csv_rows(root/city/'prepared'/'routes'/'train_01'/'manifest.csv')
    ox=float(manifest[0]['x_m']); oy=float(manifest[0]['y_m'])
    mpp=float(source['mpp'])
    return np.asarray([(float(w['pixel_x'])*mpp-ox,float(w['pixel_y'])*mpp-oy) for w in points])


def trajectory_figures(root,out,panels):
    """Use one map scale and one square crop size across all eight routes."""
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.dpi':220})
    route_data={}
    max_span_m=0.0
    map_sizes=[]
    for city in CITIES:
        for route in ROUTES:
            rows=panels[(city,route)][0]
            waypoint=waypoint_xy(root,city,route)
            pred=np.asarray([(float(r['final_x']),float(r['final_y'])) for r in rows])
            points=np.vstack((waypoint,pred))
            route_data[(city,route)]=(waypoint,pred,points)
            max_span_m=max(max_span_m,float(np.ptp(points,axis=0).max()))
            source=json.loads((root/city/'prepared'/'experiment.json').read_text())
            with Image.open(source['satellite_image']) as im:
                map_sizes.append(im.size)
    assert len(set(map_sizes))==1, f'Input satellite maps differ in size: {map_sizes}'
    map_width,map_height=map_sizes[0]
    mpp_values=[float(json.loads((root/c/'prepared'/'experiment.json').read_text())['mpp']) for c in CITIES]
    assert len(set(mpp_values))==1, mpp_values
    mpp=mpp_values[0]
    # The input maps all have the same square size: show the complete source
    # satellite map in every route panel, at exactly the same physical scale.
    side_px=min(map_width,map_height)
    if side_px>min(map_width,map_height):
        raise RuntimeError(f'Uniform map crop {side_px}px exceeds source {map_width}x{map_height}px')
    audit={'source_map_size_px':[map_width,map_height], 'mpp':mpp,
           'largest_route_axis_span_m':max_span_m,'crop_size_px':[side_px,side_px],
           'crop_size_m':[side_px*mpp,side_px*mpp], 'routes':{}}
    fig,axes=plt.subplots(2,2,figsize=(12,12),constrained_layout=True)
    for city,ax in zip(CITIES,axes.flat):
        plot_route(root,panels,route_data,city,'test_01',ax,side_px,audit)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,
               bbox_to_anchor=(0.5,-0.005))
    fig.suptitle('GT waypoint route and raw FieldAnchor predictions — test_01',fontsize=13)
    fig.savefig(out/'trajectories_four_cities_test01.png',bbox_inches='tight')
    plt.close(fig)
    for city in CITIES:
        for route in ROUTES:
            fig,ax=plt.subplots(figsize=(8,8),constrained_layout=True)
            plot_route(root,panels,route_data,city,route,ax,side_px,audit)
            ax.legend(loc='best',framealpha=.9)
            fig.savefig(out/f'trajectory_{city}_{route}.png')
            plt.close(fig)
    (out/'trajectory_crop_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def plot_route(root,panels,route_data,city,route,ax,side_px,audit):
    rows,manifest,n,mle,l3=panels[(city,route)]
    waypoint,pred,both=route_data[(city,route)]
    source=json.loads((root/city/'prepared'/'experiment.json').read_text())
    satellite=Path(source['satellite_image'])
    mpp=float(source['mpp'])
    train=csv_rows(root/city/'prepared'/'routes'/'train_01'/'manifest.csv')
    ox=float(train[0]['x_m']); oy=float(train[0]['y_m'])
    with Image.open(satellite) as whole:
        width,height=whole.size
        pixel=(both+np.asarray([ox,oy]))/mpp
        center=(pixel.min(axis=0)+pixel.max(axis=0))/2
        left=int(np.clip(round(center[0]-side_px/2),0,width-side_px))
        top=int(np.clip(round(center[1]-side_px/2),0,height-side_px))
        box=(left,top,left+side_px,top+side_px)
        if np.any(pixel[:,0]<left) or np.any(pixel[:,0]>=box[2]) or np.any(pixel[:,1]<top) or np.any(pixel[:,1]>=box[3]):
            raise RuntimeError(f'{city} {route}: path falls outside uniform crop {box}')
        crop=np.asarray(whole.crop(box).convert('RGB'))
    key=f'{city}/{route}'
    audit['routes'][key]={'map':str(satellite),'crop_box_px':list(box),
                          'path_bbox_px':[float(pixel[:,0].min()),float(pixel[:,1].min()),
                                          float(pixel[:,0].max()),float(pixel[:,1].max())],
                          'all_path_points_inside':True}
    extent=(box[0]*mpp-ox,box[2]*mpp-ox,box[3]*mpp-oy,box[1]*mpp-oy)
    ax.imshow(crop,extent=extent,origin='upper',alpha=.82,zorder=0)
    outline=[pe.Stroke(linewidth=4.0,foreground='white',alpha=.9),pe.Normal()]
    gt_line,=ax.plot(waypoint[:,0],waypoint[:,1],color='#11934e',lw=2.1,
                     label='GT waypoint route',zorder=2)
    gt_line.set_path_effects(outline)
    pred_line,=ax.plot(pred[:,0],pred[:,1],color='#e01b24',lw=1.45,
                       label='Raw prediction',zorder=3)
    pred_line.set_path_effects([pe.Stroke(linewidth=3.0,foreground='white',alpha=.65),pe.Normal()])
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
    ax.set_title(f'{city.upper()} · {route} · N={n} · MLE={mle:.2f} m')
    ax.set_xlabel('Map X (m)');ax.set_ylabel('Map Y (m)')
    ax.set_aspect('equal',adjustable='box')

def decoder_table(root,out,full_overall):
    entries={'weighted':[],'mean_shift':[]}
    route_diffs=[]
    max_ms_difference=0.;frames=0
    for city in CITIES:
        for route in ROUTES:
            probe,_,_=route_rows(root,city,'decoder_probe',route)
            full,_,_=route_rows(root,city,'full',route)
            assert len(probe)==len(full)
            route_difference=0.0
            for a,b in zip(probe,full):
                assert a['frame_id']==b['frame_id']
                centers=json.loads(a['decoder_candidate_centers_json'])
                logits=json.loads(a['decoder_candidate_logits_json'])
                assert len(centers)==len(logits)==36
                gt=np.asarray([float(a['gt_x']),float(a['gt_y'])])
                for key,cols in [('weighted',('decoder_weighted_x','decoder_weighted_y')),
                                 ('mean_shift',('final_x','final_y'))]:
                    xy=np.asarray([float(a[cols[0]]),float(a[cols[1]])])
                    entries[key].append(float(np.linalg.norm(xy-gt)))
                route_difference += entries['mean_shift'][-1] - entries['weighted'][-1]
                max_ms_difference=max(max_ms_difference,math.hypot(float(a['final_x'])-float(b['final_x']),float(a['final_y'])-float(b['final_y'])))
                frames+=1
            route_diffs.append((len(probe),route_difference))
    assert frames==1482
    if max_ms_difference>1e-3:
        raise RuntimeError(f'Probe changed Full Mean Shift output: max XY delta {max_ms_difference:.6f} m')
    data=[]
    for key,label in [('weighted','Weighted Average (36 patches)'),('mean_shift','Mean Shift (1 mode)')]:
        n,mle,l3,l15=stats(entries[key])
        data.append((label,f'{l3:.2f}%',f'{l15:.2f}%',f'{mle:.3f}'))
    assert abs(stats(entries['mean_shift'])[1]-full_overall[1])<1e-6
    counts=np.asarray([r[0] for r in route_diffs]); sums=np.asarray([r[1] for r in route_diffs])
    rng=np.random.default_rng(2033)
    sampled=rng.integers(0,8,size=(20000,8))
    paired=sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
    ci=np.quantile(paired,[.025,.975])
    timing=json.loads((out/'final_aggregation_latency.json').read_text())
    if timing['mode_count_histogram']!={'1':1482}:
        raise RuntimeError('Latency benchmark mode distribution differs from this decoder probe')
    bench={row['inputs']:row for row in timing['rows']}
    data=[(*row, f"{bench[36 if i==0 else 1]['device_only']['median_ms']:.6f}",
                  f"{bench[36 if i==0 else 1]['single_call']['median_ms']:.6f}")
          for i,row in enumerate(data)]
    save_table(out/'final_decoder_comparison.md',
        ('Final decoder','LSR@3 ↑','LSR@15 ↑','MLE (m) ↓','GPU-only aggregation (ms) ↓','Single-call aggregation (ms) ↓'),data,
        ('Both rows start from the same 36 final 6×6 candidate centers and regularized logits after identical front Mean Shift, GRU, and Kalman outputs. Weighted Average reduces all 36 softmax(logit/τ)-weighted centers. Mean Shift first iterates and merges its 36 seeds into active modes; all 1,482 frames have exactly one active mode, which its final aggregation reduces.',
         f'Probe-vs-original Full maximum XY difference: {max_ms_difference:.9f} m across {frames} frames. The logits include Kalman and current-frame GT reference priors.',
         'Latency uses real probe inputs on an RTX 3090. GPU-only is CUDA-graph device time per final reduction; single-call is a synchronized CUDA-event measurement. Feature extraction, scoring, Mean Shift iterations and mode merging are excluded. The complete final Mean Shift stage takes about 7.24 ms/frame in the original replay.',
         f'Mean Shift minus Weighted Average paired MLE: {sums.sum()/counts.sum():+.3f} m (route bootstrap 95% interval [{ci[0]:+.3f}, {ci[1]:+.3f}] m; Mean Shift better on {int(np.sum(sums<0))}/8 routes). The observed result favors Weighted Average slightly; this experiment does not show a unique Mean Shift advantage.',))
    return data


def dataset_counts(root,out):
    table=[]
    for city in CITIES:
        n=[len(csv_rows(root/city/'prepared'/'routes'/r/'manifest.csv'))
           for r in ('train_01','test_01','test_02')]
        table.append((city.upper().replace('CITY','City '),*map(str,n),str(sum(n))))
    totals=[sum(int(row[j]) for row in table) for j in (1,2,3)]
    table.append(('Overall',*map(str,totals),str(sum(totals))))
    save_table(out/'dataset_route_counts.md',
        ('City','train_01','test_01','test_02','Route observations total'),table,
        ('These are counts of sampled observations on synthetic routes. Model training internally splits train_01 into train/validation segments; test_01 and test_02 are the eight evaluation routes.',))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();root=args.source.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    panels,entries,overall=route_table(root,out)
    dataset_counts(root,out)
    crop_audit=trajectory_figures(root,out,panels)
    decoder=decoder_table(root,out,overall)
    crop_px=crop_audit['crop_size_px'][0]
    crop_m=crop_audit['crop_size_m'][0]
    table=lambda name:(out/name).read_text().strip()
    lines=[
        '# FieldAnchor 3.759 m 重現結果與圖片',
        '',
        '本頁集中整理這次推論的資料、數值、圖片和測速定義。下面各表也另存成獨立 `.md`，方便複製進論文。',
        '',
        '## 實驗來源與適用範圍',
        '',
        f'- 原始推論：`{root}`。使用歷史 v39 權重重跑 inference，**沒有重新訓練**。',
        '- 資料：四個城市、每城兩條測試路線，共 8 條路線與 1,482 筆觀測。這些路線由各張帶地理標記的 UAV 影像組成序列。',
        '- 設定：**GT-assisted**。當前影格 GT 用於局部搜尋中心及 acquisition hypothesis 選擇，因此 3.759 m 不能寫成無 GT 推論成績，也不能直接當作開放式全圖 retrieval 的成績。',
        '- 指標：MLE 是每筆定位誤差（公尺）的平均；LSR@3、LSR@15 是誤差不超過 3 m、15 m 的比例。Overall 先合併 1,482 筆誤差再計算。',
        '',
        '## 八條路線定位結果',
        '',
        table('route_wise_results.md'),
        '',
        'City C 的 test_02 MLE 最高；主結果 Overall 為 3.759 m、LSR@3 41.90%、LSR@15 100.00%。[獨立表格](route_wise_results.md)',
        '',
        '## 衛星地圖上的軌跡',
        '',
        f'所有圖都直接使用各城市 **完整的 {crop_px} × {crop_px} 像素（{crop_m:.0f} × {crop_m:.0f} m）衛星圖**，不再依路線縮小背景。八張圖的地圖物理範圍和顯示比例完全相同，且路徑全部留在圖內。',
        '',
        '綠線是原始 waypoint 所定義的 GT 路徑，紅線依影格順序連接原始模型預測；沒有繪製每筆 GT 位置點。紅線可能與綠線重疊，請放大 PNG 檢視。四城總覽顯示各城市的 test_01；下列連結可開啟全部八條路線。',
        '',
        '![四城市 test_01 衛星地圖軌跡](trajectories_four_cities_test01.png)',
        '',
        '| City | test_01 | test_02 |',
        '| --- | --- | --- |',
    ]
    for city in CITIES:
        lines.append(f'| {city.upper().replace("CITY","City ")} | [PNG](trajectory_{city}_test_01.png) | [PNG](trajectory_{city}_test_02.png) |')
    lines += [
        '',
        '[地圖裁切與路徑完整性紀錄](trajectory_crop_audit.json)；八張圖的裁切邊長一致，所有 GT waypoint 與預測點皆在圖內。',
        '',
        '## 最後座標解碼比較',
        '',
        table('final_decoder_comparison.md'),
        '',
        '兩列使用完全相同的 6 × 6＝36 個 candidate 位置與 scores，前段的 Mean Shift、GRU、Kalman 也相同。Weighted Average 直接聚合 36 個點；Mean Shift 先完成迭代與 mode 合併，再聚合最後 mode。這 1,482 筆最後都只剩 1 個 mode。**Weighted Average 的 MLE 略低**，所以這組結果不能寫成「Mean Shift 的最後聚合精度最高」。[獨立表格](final_decoder_comparison.md)',
        '',
        '表內 GPU-only 約 0.003920 / 0.002640 ms 是最後一次座標加權聚合的 device 時間；single-call 約 0.041984 / 0.042480 ms 包含單次呼叫的 launch 間隔。兩者均不含 candidate scoring，也不含 Mean Shift 迭代與合併。完整 final Mean Shift stage 約 7.24 ms/frame。請勿把微基準數字稱為完整 decoder latency。[測速方法與 P90](final_aggregation_latency.md)',
        '',
        '## 整體推論速度',
        '',
        table('inference_runtime.md'),
        '',
        '平均 39.72 ms/frame 對應約 25.18 FPS。計時從已備妥的 UAV tensor 開始；不含磁碟 I/O、前處理、模型載入及 satellite gallery 建構。這是原始 replay 環境的量測，並非實機部署速度。[獨立表格](inference_runtime.md)',
        '',
        '## 路線資料量',
        '',
        table('dataset_route_counts.md'),
        '',
        '這裡計的是合成路線上抽樣的觀測數。`train_01` 在訓練流程中又分為 train/validation；八條評估路線是各城市的 `test_01` 和 `test_02`。[獨立表格](dataset_route_counts.md)',
        '',
        '## 重現檔案',
        '',
        '- 表格和圖片生成：[`../scripts/generate_3759_additions.py`](../scripts/generate_3759_additions.py)。',
        '- 最後聚合測速：[`../scripts/benchmark_3759_final_aggregation.py`](../scripts/benchmark_3759_final_aggregation.py)。',
        '- 36 個候選點及分數由 `decoder_probe` 推論 CSV 記錄；主結果則取自原始 `full` 推論 CSV。probe 的 Mean Shift 座標與原始 Full 在逐影格檢查中一致。',
        '',
    ]
    (out/'README.md').write_text('\n'.join(lines))
    print('overall',overall,'decoder',decoder,'output',out,flush=True)

if __name__=='__main__':main()
