#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SUITE="${1:-}"
if [[ -z "${SUITE}" ]]; then
  SUITE="$(find "${ROOT}/v39_otherdata" -maxdepth 1 -type d -name 'formal_bearing_v5_smooth_*' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
[[ -n "${SUITE}" && -d "${SUITE}" ]] || { echo "ERROR: smooth suite not found" >&2; exit 2; }
SUITE="$(readlink -f "${SUITE}")"
CITY=cityd
GPU="${GPU:-0}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-4}"
SEED="${SEED:-2033}"
CPU_THREADS_PER_CITY="${CPU_THREADS_PER_CITY:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
CPU_NICE="${CPU_NICE:-5}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS_PER_CITY}"
export BLIS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${UAVSAT_VISUAL_CACHE_BATCH_SIZE:-${CACHE_BATCH_SIZE}}"
export UAVSAT_SAT_CACHE_LOCK="${UAVSAT_SAT_CACHE_LOCK:-${SUITE}/.sat_backbone_cache.lock}"

# Exact smooth-v1 estimator profile.
export UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME="${UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME:-12.0}"
export UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME="${UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME:-3.0}"
export UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2="${UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2:-2.0}"
export UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME="${UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME:-12.0}"
export UAVSAT_HEADING_STATE_EMA_ALPHA="${UAVSAT_HEADING_STATE_EMA_ALPHA:-0.22}"
export UAVSAT_TURN_RATE_EMA_ALPHA="${UAVSAT_TURN_RATE_EMA_ALPHA:-0.20}"
export UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME="${UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME:-3.0}"
export UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2="${UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2:-3.0}"
export UAVSAT_LOSS_CROSS_MOTION_REG="${UAVSAT_LOSS_CROSS_MOTION_REG:-0.03}"
export UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M="${UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M:-1.25}"
export UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME="${UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME:-0.90}"
export UAVSAT_KALMAN_FINAL_STEP_MAX_M="${UAVSAT_KALMAN_FINAL_STEP_MAX_M:-6.0}"
export UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M="${UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M:-36.0}"
export UAVSAT_CORR_PARALLEL_M="${UAVSAT_CORR_PARALLEL_M:-0.70}"
export UAVSAT_CORR_CROSS_M="${UAVSAT_CORR_CROSS_M:-0.45}"

# Exact Formal-V5 training / Kalman profile.
export UAVSAT_LOSS_MEASUREMENT="${UAVSAT_LOSS_MEASUREMENT:-2.0}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-3.0}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-0.40}"
export UAVSAT_LOSS_ACCELERATION="${UAVSAT_LOSS_ACCELERATION:-0.50}"
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-8e-5}"
export UAVSAT_RNN_DROPOUT="${UAVSAT_RNN_DROPOUT:-0.05}"
export UAVSAT_EARLY_MIN_EPOCH="${UAVSAT_EARLY_MIN_EPOCH:-10}"
export UAVSAT_MOTION_VEL_ALPHA="${UAVSAT_MOTION_VEL_ALPHA:-0.65}"
export UAVSAT_MOTION_STEP_ALPHA="${UAVSAT_MOTION_STEP_ALPHA:-0.70}"
export UAVSAT_MOTION_RESIDUAL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_FORWARD_M:-2.5}"
export UAVSAT_MOTION_RESIDUAL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_CROSS_M:-1.25}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M:-1.25}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M:-0.75}"
export UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE:-0.45}"
export UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE:-1.00}"
export UAVSAT_TEMPORAL_DELTA2_SCALE="${UAVSAT_TEMPORAL_DELTA2_SCALE:-1.00}"
export UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M="${UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M:-1.25}"
export UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M="${UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M:-0.75}"
export UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M="${UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M:-2.00}"
export UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M="${UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M:-1.00}"
export UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="${UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2:-6.0}"
export UAVSAT_KALMAN_Q_PROGRESS="${UAVSAT_KALMAN_Q_PROGRESS:-1.50}"
export UAVSAT_KALMAN_Q_CROSS="${UAVSAT_KALMAN_Q_CROSS:-0.40}"
export UAVSAT_KALMAN_Q_VELOCITY="${UAVSAT_KALMAN_Q_VELOCITY:-1.00}"
export UAVSAT_KALMAN_CONFIDENCE_POWER="${UAVSAT_KALMAN_CONFIDENCE_POWER:-0.50}"
export UAVSAT_KALMAN_PRIOR_BLEND_BASE="${UAVSAT_KALMAN_PRIOR_BLEND_BASE:-0.00}"
export UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN="${UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN:-0.18}"
export UAVSAT_KALMAN_PRIOR_BLEND_MAX="${UAVSAT_KALMAN_PRIOR_BLEND_MAX:-0.30}"
export UAVSAT_KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF="${UAVSAT_KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF:-0.60}"
export UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE="${UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE:-0.55}"
export UAVSAT_KALMAN_STEP_RELAX_WIDTH="${UAVSAT_KALMAN_STEP_RELAX_WIDTH:-0.08}"
export UAVSAT_KALMAN_STEP_VISUAL_SLACK_M="${UAVSAT_KALMAN_STEP_VISUAL_SLACK_M:-3.0}"

PREPARED="${SUITE}/${CITY}/prepared"
TRAIN_DIR="${SUITE}/${CITY}/train_frames3"
FULL_DIR="${SUITE}/${CITY}/variants/full"
FIG_DIR="${FULL_DIR}/formal_figures"
CK="${TRAIN_DIR}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
SUMMARY="${FULL_DIR}/bearing_v39_summary.json"

[[ -s "${PREPARED}/experiment.json" ]] || { echo "ERROR: ${CITY} prepared package missing: ${PREPARED}" >&2; exit 10; }
for done_city in citya cityb cityc; do
  [[ -s "${SUITE}/${done_city}/variants/full/bearing_v39_summary.json" ]] || {
    echo "ERROR: ${done_city} is also incomplete; this CityD-only resume intentionally refuses to alter it" >&2
    exit 11
  }
done

python3 -m py_compile v39_otherdata/bearing_iclr_ablation.py v39_otherdata/bearing_plot_final_vs_gt.py
python3 - <<'PY'
from pathlib import Path
runner = Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
base = Path('v39_DirectFinalMS/base_src/config.py').read_text(encoding='utf-8')
checks = {
    'formal_v5_runner': 'formal_v5_direct_delta2_residual_kalman' in runner,
    'smooth_base_config': 'UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME' in base,
    'smooth_kalman_limit': 'UAVSAT_KALMAN_FINAL_STEP_MAX_M' in base,
    'canonical_v5_cross_bound': 'MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0' in base,
}
for k,v in checks.items(): print(f'[CITYD PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('ERROR: local generated Formal-V5 Smooth runner/base config is not in the expected state; do not retrain until refetched/repatched')
PY

COMMON=(
  --suite-root "${SUITE}"
  --dataset-root "${DATASET_ROOT}"
  --city "${CITY}"
  --gpu "${GPU}"
  --backbone mobilenet_v3_small
  --visual-epochs "${VISUAL_EPOCHS}"
  --temporal-epochs "${TEMPORAL_EPOCHS}"
  --epochs-per-route "${TEMPORAL_EPOCHS}"
  --patience "${PATIENCE}"
  --jitter-m 8
  --max-sample-distance-m 15
  --heading-weight-px-per-deg 0
  --ms-bandwidth-m 7
  --seed "${SEED}"
)

printf '%s\n' \
  "================================================================================" \
  "FORMAL V5 SMOOTH-v1 — CITYD RESUME ONLY" \
  "Suite      : ${SUITE}" \
  "GPU        : ${GPU}" \
  "Checkpoint : ${CK}" \
  "Summary    : ${SUMMARY}" \
  "================================================================================"

if [[ ! -s "${SUMMARY}" ]]; then
  if [[ -s "${CK}" ]]; then
    echo "[CITYD TRAIN SKIP] completed checkpoint exists"
  else
    echo "[CITYD TRAIN RESUME] final checkpoint missing; training CityD only"
    nice -n "${CPU_NICE}" python3 -u v39_otherdata/bearing_iclr_ablation.py train \
      "${COMMON[@]}" --train-frames 3 \
      2>&1 | tee "${SUITE}/logs/cityd_train_resume_only_gpu${GPU}.log"
    [[ -s "${CK}" ]] || { echo "ERROR: CityD checkpoint still missing after training" >&2; exit 20; }
  fi

  echo "[CITYD EVAL] nav50/nav51"
  nice -n "${CPU_NICE}" python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
    "${COMMON[@]}" --variant full \
    2>&1 | tee "${SUITE}/logs/cityd_test_resume_only_gpu${GPU}.log"
  [[ -s "${SUMMARY}" ]] || { echo "ERROR: CityD summary still missing after eval" >&2; exit 21; }
else
  echo "[CITYD EVAL SKIP] summary already exists"
fi

if [[ ! -s "${FIG_DIR}/nav50_result.jpg" || ! -s "${FIG_DIR}/nav51_result.jpg" || ! -s "${FIG_DIR}/plot_source_audit.json" ]]; then
  echo "[CITYD PLOT] raw final_x/final_y; no display smoothing"
  python3 -u v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "${PREPARED}" \
    --output-dir "${FULL_DIR}" \
    --routes test_01 test_02 \
    2>&1 | tee "${SUITE}/logs/cityd_plot_resume_only.log"
  mkdir -p "${FIG_DIR}"
  cp "${FULL_DIR}/paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg" "${FIG_DIR}/nav50_result.jpg"
  cp "${FULL_DIR}/paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg" "${FIG_DIR}/nav51_result.jpg"
  cp "${FULL_DIR}/paper_figures_waypoint_gt/plot_source_audit.json" "${FIG_DIR}/plot_source_audit.json"
fi

python3 - "${SUITE}" <<'PY'
from pathlib import Path
import json, sys
root=Path(sys.argv[1]); cities=['citya','cityb','cityc','cityd']; rows=[]
out={'run_type':'formal_smooth_v1','cities':{},'figures':{},'macro_average_over_8_held_out_routes':{}}
for city in cities:
    full=root/city/'variants'/'full'
    p=full/'bearing_v39_summary.json'
    if not p.is_file(): raise SystemExit(f'ERROR missing summary: {p}')
    data=json.loads(p.read_text())
    if len(data)!=2: raise SystemExit(f'ERROR {city} expected 2 route summaries, got {list(data)}')
    out['cities'][city]=data
    rows.extend((city,k,v) for k,v in data.items())
    fig=full/'formal_figures'
    for n in ['nav50_result.jpg','nav51_result.jpg','plot_source_audit.json']:
        q=fig/n
        if not q.is_file() or q.stat().st_size==0: raise SystemExit(f'ERROR missing output: {q}')
    out['figures'][city]={'nav50':str(fig/'nav50_result.jpg'),'nav51':str(fig/'nav51_result.jpg'),'plot_source_audit':str(fig/'plot_source_audit.json')}
for key in ['MLE_m','P90_m','LSR@3_pct','LSR@5_pct','LSR@10_pct','LSR@15_pct']:
    vals=[float(m[key]) for _,_,m in rows if key in m]
    if vals: out['macro_average_over_8_held_out_routes'][key]=sum(vals)/len(vals)
out['held_out_route_count']=len(rows); out['figure_count']=8
out['note']='Smooth-v1 raw model-output figures; no display trajectory smoothing.'
(root/'formal_allcities_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('[CITYD RESUME VERIFY] routes=',len(rows),'figures=8 PASS')
print(json.dumps(out['macro_average_over_8_held_out_routes'],indent=2))
PY

bash v39_otherdata/upload_existing_smooth_results.sh "${SUITE}"
echo "[CITYD RESUME + UPLOAD DONE] ${SUITE}"
