#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

FROZEN_BRANCH="bearing-v5-citya-pass-20260920"
FROZEN_SHA="6911ac1dbccbfc162b3bf77a253c235063fbd60c"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${FROZEN_CITYA_SUITE_ROOT:-${ROOT}/v39_otherdata/frozen_citya_paper_repro_${TS}}"
WT="$(mktemp -d /tmp/uavsat-frozen-citya.XXXXXX)"

cleanup(){
  git -C "${ROOT}" worktree remove --force "${WT}" >/dev/null 2>&1 || true
  rm -rf "${WT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[FROZEN REPRO] fetch ${FROZEN_BRANCH}"
git fetch origin "${FROZEN_BRANCH}" >/dev/null
ACTUAL="$(git rev-parse "origin/${FROZEN_BRANCH}")"
if [[ "${ACTUAL}" != "${FROZEN_SHA}" ]]; then
  echo "ERROR: frozen branch moved: expected ${FROZEN_SHA}, got ${ACTUAL}" >&2
  exit 10
fi

echo "[FROZEN REPRO] frozen SHA: ${FROZEN_SHA} PASS"
git worktree add --detach "${WT}" "${FROZEN_SHA}" >/dev/null

# Verify the exact V5 ingredients that produced the recorded PASS result.
python3 - "${WT}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
fixed = (root/'v39_otherdata/run_bearing_iclr_ablation_fixed.sh').read_text(encoding='utf-8')
patch = (root/'v39_DirectFinalMS/patch_simple_figure_gru.py').read_text(encoding='utf-8')
checks = {
    'separate_1_2_3_checkpoints': 'separate_1_2_3_checkpoints' in fixed,
    'direct_delta2_acceleration': 'TEMPORAL_DIRECT_ACCEL_FORWARD_M' in fixed,
    'direct_delta2_next_step': 'TEMPORAL_DIRECT_STEP_FORWARD_M' in fixed,
    'v5_validation_search': 'direct_delta2_second_order_v5' in fixed,
    'patience_4': 'PATIENCE="${PATIENCE:-4}"' in fixed,
    'next_step_loss_3': "UAVSAT_LOSS_NEXT_STEP:-2.5', 'UAVSAT_LOSS_NEXT_STEP:-3.0" in fixed,
    'accel_loss_0p50': "UAVSAT_LOSS_ACCELERATION:-0.25', 'UAVSAT_LOSS_ACCELERATION:-0.50" in fixed,
    'seven_block_gru': 'feature_dim * 7' in patch,
}
for k,v in checks.items():
    print(f'[FROZEN REPRO PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('ERROR: frozen V5 precheck failed')
PY

mkdir -p "${SUITE_ROOT}"
echo "================================================================================"
echo "FROZEN CITYA PAPER REPRO"
echo "commit          : ${FROZEN_SHA}"
echo "dataset         : ${DATASET_ROOT}"
echo "suite           : ${SUITE_ROOT}"
echo "temporal        : separate 1f/2f/3f checkpoints"
echo "3f-only branch  : direct delta2 -> acceleration + next_step"
echo "selection       : train/validation only"
echo "upload          : disabled during reproduction"
echo "================================================================================"

(
  cd "${WT}"
  BEARING_DATASET_ROOT="${DATASET_ROOT}" \
  CITY=citya \
  ICLR_SUITE_ROOT="${SUITE_ROOT}" \
  TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}" \
  VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}" \
  PATIENCE=4 \
  SEED="${SEED:-2033}" \
  UPLOAD_RESULTS=0 \
  RESUME_EVAL=0 \
  bash v39_otherdata/run_bearing_iclr_ablation_fixed.sh
)

AUDIT="${SUITE_ROOT}/paper_trend_audit.json"
TABLE="${SUITE_ROOT}/paper_ablation_tables.md"
[[ -s "${AUDIT}" ]] || { echo "ERROR: missing ${AUDIT}" >&2; exit 20; }
[[ -s "${TABLE}" ]] || { echo "ERROR: missing ${TABLE}" >&2; exit 21; }

python3 - "${AUDIT}" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p,'r',encoding='utf-8'))
print('[FROZEN REPRO AUDIT] FULL_TREND_CHECK =', d.get('FULL_TREND_CHECK'))
print('[FROZEN REPRO AUDIT] component_full_best =', d.get('component_full_best'))
print('[FROZEN REPRO AUDIT] three_frame_full_best =', d.get('three_frame_full_best'))
if d.get('FULL_TREND_CHECK') != 'PASS':
    raise SystemExit('ERROR: reproduced result did not pass frozen trend audit')
PY

echo "================================================================================"
echo "FROZEN CITYA REPRO COMPLETE"
echo "Table : ${TABLE}"
echo "Audit : ${AUDIT}"
echo "NOTE  : controlled_gt_jitter protocol remains unchanged and must be disclosed."
echo "================================================================================"
