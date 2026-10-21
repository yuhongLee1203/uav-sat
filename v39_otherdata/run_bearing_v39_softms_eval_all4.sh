#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITIES="${CITIES:-citya cityb cityc cityd}"

for city in ${CITIES}; do
  echo "================================================================================"
  echo "V39 SOFTMS EVAL ONLY -- ${city}"
  echo "Forward 3x6 -> SoftMS -> 3-frame GRU -> fixed Kalman -> final 6x6 SoftMS"
  echo "NO RETRAINING: reuse existing ${city} checkpoints"
  echo "================================================================================"

  CITY="${city}" python3 v39_otherdata/bearing_runner_softms_eval_v39.py \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --gpu "${GPU}" \
    --backbone mobilenet_v3_small \
    --epochs-per-route 60 \
    --patience 10 \
    --jitter-m 8 \
    --step-m 4 \
    --max-sample-distance-m 15 \
    --heading-weight-px-per-deg 0 \
    --reuse-visual \
    --resume

  ROOT="v39_otherdata/generated/${city}"
  OUT="${ROOT}/v39_output_bearing_softms_eval"
  python3 v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "${ROOT}" \
    --output-dir "${OUT}" \
    --routes test_01 test_02
  python3 v39_otherdata/bearing_paper_metrics.py \
    --prepared-root "${ROOT}" \
    --output-dir "${OUT}"
done

python3 - "v39_otherdata/generated" ${CITIES} <<'PY'
import csv, json, math, sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]); cities=sys.argv[2:]
rows=[]; all_errors=[]
for city in cities:
    out=root/city/'v39_output_bearing_softms_eval'
    summary=json.loads((out/'bearing_v39_summary.json').read_text(encoding='utf-8'))
    for route in ('test_01','test_02'):
        matches=sorted(out.glob(f'{route}_*_frames.csv'))
        if len(matches)!=1:
            raise SystemExit(f'{city}/{route}: expected one frame CSV, got {len(matches)}')
        with matches[0].open(newline='',encoding='utf-8') as f:
            errors=np.asarray([float(r['error_final_m']) for r in csv.DictReader(f)],dtype=float)
        all_errors.extend(errors.tolist())
        row={
            'city':city,'route':route,'frames':int(errors.size),
            'MLE_m':float(errors.mean()),'MedLE_m':float(np.median(errors)),
            'P90_m':float(np.quantile(errors,.90)),'P95_m':float(np.quantile(errors,.95)),
            'P99_m':float(np.quantile(errors,.99)),
            'LSR@5_pct':float((errors<=5).mean()*100),
            'LSR@10_pct':float((errors<=10).mean()*100),
            'LSR@15_pct':float((errors<=15).mean()*100),
            'LSR@20_pct':float((errors<=20).mean()*100),
            'front_decoder':'forward_3x6_soft_mean_shift',
            'checkpoint_retraining':False,
        }
        rows.append(row)
        print(f"{city} {route}: frames={row['frames']} MLE={row['MLE_m']:.3f}m MedLE={row['MedLE_m']:.3f}m P90={row['P90_m']:.3f}m LSR@15={row['LSR@15_pct']:.2f}%")

err=np.asarray(all_errors,dtype=float)
overall={
    'frames':int(err.size),'MLE_m':float(err.mean()),'MedLE_m':float(np.median(err)),
    'P90_m':float(np.quantile(err,.90)),'P95_m':float(np.quantile(err,.95)),
    'P99_m':float(np.quantile(err,.99)),
    'LSR@5_pct':float((err<=5).mean()*100),'LSR@10_pct':float((err<=10).mean()*100),
    'LSR@15_pct':float((err<=15).mean()*100),'LSR@20_pct':float((err<=20).mean()*100),
    'front_decoder':'forward_3x6_soft_mean_shift','checkpoint_retraining':False,
}
payload={'method':'V39 Forward3x6 SoftMS eval-only decoder swap','routes':rows,'overall':overall}
(root/'paper_all4_softms_eval_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
with (root/'paper_all4_softms_eval_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('-'*80)
print(f"ALL {len(rows)} ROUTES: frames={overall['frames']} MLE={overall['MLE_m']:.3f}m MedLE={overall['MedLE_m']:.3f}m P90={overall['P90_m']:.3f}m LSR@15={overall['LSR@15_pct']:.2f}%")
print('Summary JSON:',root/'paper_all4_softms_eval_summary.json')
print('Summary CSV :',root/'paper_all4_softms_eval_summary.csv')
PY
