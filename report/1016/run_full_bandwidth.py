import os,json,csv,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path('/yh/study/uav-sat');OUT=ROOT/'report/1016';SOURCE=ROOT/'v39_otherdata/output/original3759_inference_20260925_224113_1642368'
e=os.environ.copy();e.update({'CUDA_VISIBLE_DEVICES':'0','BEARING_V40_RELAX_LIMITS':'0','BEARING_FRAME_ISOLATED_CACHE':'1','BEARING_HEADING_FUSION_ALPHA':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
records=[]
for bw in (7,4,14):
 suite=OUT/'full_tracker_bandwidth'/f'bw{bw}';suite.mkdir(parents=True,exist_ok=True)
 for city in ('citya','cityb','cityc','cityd'):
  folder=suite/city;folder.mkdir(exist_ok=True)
  for name in ('prepared','train_frames3'):
   target=folder/name
   if not target.exists():target.symlink_to(SOURCE/city/name,target_is_directory=True)
  cmd=[sys.executable,'-u',str(ROOT/'v39_otherdata/bearing_iclr_ablation.py'),'eval','--suite-root',str(suite),'--city',city,'--variant','full','--train-frames','3','--gpu','0','--seed','2033','--ms-bandwidth-m',str(bw)]
  print('[START]',bw,city,flush=True)
  with (folder/'run.log').open('w') as f:subprocess.run(cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
  summary=json.loads((folder/'variants/full/bearing_v39_summary.json').read_text())
  if bw==7:
   source=json.loads((SOURCE/city/'variants/full/bearing_v39_summary.json').read_text())
   for route in ('test_01','test_02'):
    assert abs(summary[route]['MLE_m']-source[route]['MLE_m'])<1e-5,(city,route,'baseline failed')
  es=[]
  for route in ('test_01','test_02'):
   rows=list(csv.DictReader(open(summary[route]['CSV'])))
   es.extend([float(r['error_final_m']) for r in rows])
  records.append({'bandwidth_m':bw,'city':city,'errors':es})
  (OUT/'full_tracker_bandwidth_partial.json').write_text(json.dumps(records)+'\n')
  print('[DONE]',bw,city,'MLE',np.mean(es),flush=True)
rows=[]
for bw in (7,4,14):
 errors=np.concatenate([r['errors'] for r in records if r['bandwidth_m']==bw]);assert len(errors)==1482
 rows.append({'bandwidth_m':bw,'N':len(errors),'MLE_m':float(errors.mean()),'LSR3_pct':float(100*np.mean(errors<=3)),'LSR5_pct':float(100*np.mean(errors<=5)),'LSR15_pct':float(100*np.mean(errors<=15))})
(OUT/'full_tracker_bandwidth_results.json').write_text(json.dumps(rows,indent=2)+'\n')
text=['# 完整 tracker 重播：Final MS bandwidth','', '| Bandwidth (m) | N | MLE (m) | LSR@3 (%) | LSR@5 (%) | LSR@15 (%) |','| --- | ---: | ---: | ---: | ---: | ---: |']
for r in rows:text.append(f"| {r['bandwidth_m']} | {r['N']} | {r['MLE_m']:.4f} | {r['LSR3_pct']:.2f} | {r['LSR5_pct']:.2f} | {r['LSR15_pct']:.2f} |")
text+=['','這三組重新執行完整 tracker：18 patch、三幀 GRU、Kalman、最終6×6MS。每條route從新狀態開始。使用相同歷史 checkpoint／prepared資料／calibration，只有 bandwidth 改變，GPU0。7m baseline逐條 route 與原3.759m run之MLE差異<1e-5m。這是原 GT-assisted test 上的探索性 sensitivity，沒有重新訓練，也不能用 test最佳列替換未獨立驗證的新模型主結果。','', 'raw outputs / logs: `full_tracker_bandwidth/bw*/city*/variants/full/`。']
(OUT/'09_full_tracker_bandwidth.md').write_text('\n'.join(text)+'\n')
print('[ALL DONE]',rows,flush=True)
