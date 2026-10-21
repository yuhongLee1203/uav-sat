#!/usr/bin/env python3
"""Plot saved GT-free held-out predictions on each city's actual satellite image."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('NO_GT_PICTURE_ROOT', str(HERE.parent/'report/add_more/picture_noGT')))
PREPARED=HERE/'data/official96_fixed_routes'
CITIES='abcd'
ROUTES=('test_01','test_02')


def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def xy(rows,p):return np.asarray([[float(r[f'{p}_x']),float(r[f'{p}_y'])] for r in rows])


def rdp(points,epsilon=5.0):
    """Straighten only the displayed GT path; metrics always use raw GT."""
    if len(points)<3:return points
    a,b=points[0],points[-1]
    delta=b-a;norm=float(np.linalg.norm(delta))
    if norm<1e-9:dist=np.linalg.norm(points[1:-1]-a,axis=1)
    else:dist=np.abs(np.cross(delta,points[1:-1]-a))/norm
    i=int(np.argmax(dist))+1
    if dist[i-1]<=epsilon:return points[[0,-1]]
    return np.vstack([rdp(points[:i+1],epsilon)[:-1],rdp(points[i:],epsilon)])


def draw_map(ax,bg,extent,gt,pred,zoom=False):
    ax.imshow(bg,extent=extent,origin='upper',interpolation='nearest',zorder=0)
    simplified=rdp(gt,30.0)
    g,=ax.plot(simplified[:,0],simplified[:,1],lw=2.2,color='#00a86b',
               label='GT route (display simplified)',zorder=3)
    display_pred=rdp(pred,40.0)
    p,=ax.plot(display_pred[:,0],display_pred[:,1],lw=1.55,color='#df2e2e',
               label='No-GT prediction (display simplified)',zorder=4)
    g.set_path_effects([pe.Stroke(linewidth=4,foreground='white',alpha=.9),pe.Normal()])
    p.set_path_effects([pe.Stroke(linewidth=3,foreground='white',alpha=.75),pe.Normal()])
    if zoom:
        pts=np.vstack([gt,pred]);mn=pts.min(axis=0);mx=pts.max(axis=0)
        side=max(float(np.max(mx-mn))+40,140)
        center=(mn+mx)/2
        ax.set_xlim(center[0]-side/2,center[0]+side/2)
        ax.set_ylim(center[1]+side/2,center[1]-side/2)
    else:
        ax.set_xlim(extent[0],extent[1]);ax.set_ylim(extent[2],extent[3])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Map X (m)');ax.set_ylabel('Map Y (m)')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows_summary=[];plot_items={};satellite_hashes=set();frame_error_hashes=set()
    import hashlib
    for city in CITIES:
        full=f'city{city}'
        report=json.loads((OUT/'inference'/full/'report.json').read_text())
        assert report.get('gallery_city',full)==full
        assert report.get('current_frame_GT_model_input',report.get('current_frame_gt_model_input')) is False
        assert int(report.get('final_ms_grid_size',report.get('environment',{}).get('MS_GRID_SIZE','6')))==6
        prep=PREPARED/full/'prepared'
        experiment=json.loads((prep/'experiment.json').read_text())
        sat=Path(experiment['satellite_image'])
        satellite_hashes.add(hashlib.sha256(sat.read_bytes()).hexdigest())
        train=readcsv(prep/'routes/train_01/manifest.csv')
        ox,oy=float(train[0]['x_m']),float(train[0]['y_m'])
        with Image.open(sat) as im:
            size=im.size
            im.thumbnail((1600,1600),Image.Resampling.LANCZOS)
            bg=np.asarray(im.convert('RGB'))
        mpp=float(experiment['mpp'])
        extent=(-ox,size[0]*mpp-ox,size[1]*mpp-oy,-oy)
        for route in ROUTES:
            item=report['results'][route]
            path=Path(item['CSV'])
            route_rows=readcsv(path)
            assert route_rows and all(r['image_path'].startswith(f'/yh/study/cvpr_data/Bearing_UAV_90K/{full}/') for r in route_rows)
            gt,pred=xy(route_rows,'gt'),xy(route_rows,'final')
            e=np.linalg.norm(pred-gt,axis=1)
            saved=np.asarray([float(r['error_final_m']) for r in route_rows])
            assert np.max(np.abs(e-saved))<1e-4
            assert abs(e.mean()-float(item['MLE_m']))<1e-4
            frame_error_hashes.add(hashlib.sha256(e.tobytes()).hexdigest())
            mle=float(e.mean());p90=float(np.quantile(e,.9));lsr15=float(np.mean(e<=15)*100)
            rows_summary.append({'City':city.upper(),'Route':route,'Frames':len(e),
                                 'MLE (m)':mle,'P90 (m)':p90,'LSR@15 (%)':lsr15,
                                 'Prediction CSV':str(path)})
            plot_items[(city,route)]=(bg,extent,gt,pred,e)
            fig,axes=plt.subplots(1,3,figsize=(16,5.4),layout='constrained',
                                  gridspec_kw={'width_ratios':[1,1,.9]})
            draw_map(axes[0],bg,extent,gt,pred)
            axes[0].set_title('Complete city satellite map (1,024 × 1,024 m)')
            draw_map(axes[1],bg,extent,gt,pred,zoom=True)
            axes[1].set_title('Route detail on satellite map')
            axes[2].plot(np.arange(len(e)),e,color='#df2e2e',lw=1.1)
            axes[2].axhline(mle,ls='--',color='#313131',lw=1,label=f'MLE {mle:.2f} m')
            axes[2].set(xlabel='Observation',ylabel='Localization error (m)',
                        title=f'{full.upper()} {route}: {len(e)} frames')
            axes[2].grid(alpha=.25);axes[2].legend(frameon=False)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.015))
            fig.suptitle(f'GT-free inference · City {city.upper()} {route} · MLE {mle:.2f} m · P90 {p90:.2f} m · LSR@15 {lsr15:.1f}%',fontsize=13)
            stem=f'city{city}_{route}_noGT'
            fig.savefig(OUT/f'{stem}.png',dpi=180,bbox_inches='tight')
            fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight')
            plt.close(fig)
    assert len(rows_summary)==8 and len(satellite_hashes)==4 and len(frame_error_hashes)==8
    fig,axes=plt.subplots(4,2,figsize=(12,23),layout='constrained')
    for i,city in enumerate(CITIES):
        for j,route in enumerate(ROUTES):
            bg,extent,gt,pred,e=plot_items[(city,route)]
            ax=axes[i,j]
            draw_map(ax,bg,extent,gt,pred)
            ax.set_title(f'City {city.upper()} {route} · MLE {e.mean():.2f} m · P90 {np.quantile(e,.9):.2f} m')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.008))
    fig.suptitle('GT-free FieldAnchor · eight held-out test routes · city-specific satellite maps',fontsize=14)
    fig.savefig(OUT/'all_eight_routes_noGT.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'all_eight_routes_noGT.pdf',bbox_inches='tight')
    plt.close(fig)
    all_e=np.concatenate([plot_items[(c,r)][4] for c in CITIES for r in ROUTES])
    pooled={'Frames':len(all_e),'MLE (m)':float(all_e.mean()),'P90 (m)':float(np.quantile(all_e,.9)),
            'LSR@15 (%)':float(np.mean(all_e<=15)*100)}
    table=['| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 (%) |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows_summary:
        table.append(f"| {r['City']} | {r['Route']} | {r['Frames']} | {r['MLE (m)']:.2f} | {r['P90 (m)']:.2f} | {r['LSR@15 (%)']:.2f} |")
    table.append(f"| Overall | — | {pooled['Frames']} | {pooled['MLE (m)']:.2f} | {pooled['P90 (m)']:.2f} | {pooled['LSR@15 (%)']:.2f} |")
    retrained=os.environ.get('NO_GT_RETRAINED','0')=='1'
    model_note=('City A 使用先前已在 train split 以 GT-loss 訓練的視覺權重；City B/C/D 在本輪各自重新訓練視覺模型。四城市的共用時序模型在本輪從隨機初始化重新訓練，checkpoint 依四城市 validation MLE 選定，最後才評估 test。' if retrained else 'B/C/D 是 City A 權重的跨城市直接測試；每座城市的衛星 gallery 由該城市底圖重建，沒有重新訓練。已核對模型與 backbone 權重張量完全相同。結果不能稱為四城市各自訓練的模型。')
    note=['# 無 GT 模型：四城市全部 test route 結果圖','',
          ('本次使用四城市 validation 選定的固定 6×6 模型；八條 test route 僅在設定固定後推論，沒有用 test GT 選模型或搜尋位置。GT 只用於事後計算誤差及畫綠色顯示路徑。' if retrained else '本次使用 City A 驗證集所選的固定 6×6 模型；八條 test route 僅在設定固定後推論，沒有用 test GT 選模型或搜尋位置。GT 只用於事後計算誤差及畫綠色顯示路徑。'),
          '模型仍使用外部提供的 route/path reference；「無 GT」指當前幀 GT 不進模型。',
          model_note,
          '綠色 GT 路徑僅供顯示，以 30 m Ramer–Douglas–Peucker 簡化成較直的線段；所有數值用未簡化的逐幀 GT 座標。紅色預測線僅供顯示，以 40 m 容差保留主要轉向並移除逐幀鋸齒；逐幀誤差與全部指標使用未簡化的原始預測。四城市 test_01 的官方 waypoint 像素路線相同，但衛星底圖、測試 UAV 影像與預測不同。',
          '這次使用 v40 official96 固定 test split，共 613 幀；它與 v39 的 1,482 幀 GT-assisted replay 是不同資料／推論設定，不能直接以 3.76 m 對照。',
          '', '## 測試數值','',*table,'',
          '## 圖片','',
          '- [八條路線合圖](all_eight_routes_noGT.png)（每格是對應城市的完整衛星圖）',
          '- 各 route 圖有完整衛星圖、路線局部衛星圖和逐幀誤差：', '']
    for c in CITIES:
        note.append(f"- City {c.upper()}: [test_01](city{c}_test_01_noGT.png) · [test_02](city{c}_test_02_noGT.png)")
    note += ['', '## 可稽核來源','',
             '- 每個城市的 `inference/city*/report.json` 記錄 checkpoint、gallery 來源與測試摘要；`inference/city*/city*/route_*_frames.csv`（實際檔名見 report.json 的 `CSV`）保留每幀 GT 與預測。',
             ('- 每個城市的 `run.log` 保留執行輸出；train/validation/test 各階段權重來源見正式報告。' if retrained else '- 每個城市的 `run.log` 保留執行輸出。`visual_city_gallery.pt`（B/C/D）只將 City A 視覺權重搭配該城市底圖重新編碼的特徵庫，沒有更新模型權重。'),
             '- 所有八條路線的誤差序列雜湊互不相同；四座城市的衛星圖雜湊互不相同。',
             '']
    (OUT/'README.md').write_text('\n'.join(note))
    (OUT/'results.json').write_text(json.dumps({'routes':rows_summary,'overall':pooled,
              'distinct_satellite_hashes':len(satellite_hashes),'distinct_error_hashes':len(frame_error_hashes)},indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'overall':pooled,'routes':len(rows_summary)},indent=2),flush=True)

if __name__=='__main__':main()
