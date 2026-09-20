#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${FORMAL_SUITE_ROOT:?Set FORMAL_SUITE_ROOT to the interrupted formal suite}"
SUITE_ROOT="$(readlink -f "${FORMAL_SUITE_ROOT}")"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-4}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
CPU_THREADS_PER_CITY="${CPU_THREADS_PER_CITY:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
CPU_NICE="${CPU_NICE:-5}"
CITIES=(citya cityb cityc cityd)

[[ -d "${SUITE_ROOT}" ]] || { echo "ERROR: suite not found: ${SUITE_ROOT}" >&2; exit 2; }
mkdir -p "${SUITE_ROOT}/logs"

# Resource-safe limits. These do not change model math/protocol.
export OMP_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS_PER_CITY}"
export BLIS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${UAVSAT_VISUAL_CACHE_BATCH_SIZE:-${CACHE_BATCH_SIZE}}"
export UAVSAT_SAT_CACHE_LOCK="${UAVSAT_SAT_CACHE_LOCK:-${SUITE_ROOT}/.sat_backbone_cache.lock}"

# Same formal V5 settings as the interrupted run.
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

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/bearing_plot_final_vs_gt.py

# The interrupted formal command already generated the V5-aligned local runner.
# Refuse to continue with an accidentally reverted base file.
python3 - <<'PY'
from pathlib import Path
runner = Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
checks = {
    'formal_v5_calibration': 'formal_v5_direct_delta2_residual_kalman' in runner,
    'legacy_context_gru_disabled': 'base._patch_context_gru(runtime_root)' not in runner,
}
for k, v in checks.items():
    print(f'[RESUME PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit(
        'Local bearing_iclr_ablation.py is not the generated Formal V5 runner. '
        'Do not overwrite it with the branch base file before resume.'
    )
PY

common_args(){
  local city="$1" gpu="$2"
  COMMON=(
    --suite-root "${SUITE_ROOT}"
    --dataset-root "${DATASET_ROOT}"
    --city "${city}"
    --gpu "${gpu}"
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
}

plot_city(){
  local city="$1"
  local prepared="${SUITE_ROOT}/${city}/prepared"
  local full_dir="${SUITE_ROOT}/${city}/variants/full"
  local fig_dir="${full_dir}/formal_figures"

  echo "[RESUME PLOT] ${city}"
  python3 -u v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "${prepared}" \
    --output-dir "${full_dir}" \
    --routes test_01 test_02 \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_plot_full_resume.log"

  mkdir -p "${fig_dir}"
  cp "${full_dir}/paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg" \
     "${fig_dir}/nav50_result.jpg"
  cp "${full_dir}/paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg" \
     "${fig_dir}/nav51_result.jpg"
  cp "${full_dir}/paper_figures_waypoint_gt/plot_source_audit.json" \
     "${fig_dir}/plot_source_audit.json"

  [[ -s "${fig_dir}/nav50_result.jpg" ]] || return 22
  [[ -s "${fig_dir}/nav51_result.jpg" ]] || return 22
  [[ -s "${fig_dir}/plot_source_audit.json" ]] || return 22
  echo "[RESUME FIGURES DONE] ${city} nav50/nav51"
}

resume_city(){
  local city="$1" gpu="$2"
  local prepared="${SUITE_ROOT}/${city}/prepared"
  local ck="${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  local summary="${SUITE_ROOT}/${city}/variants/full/bearing_v39_summary.json"
  local nav50="${SUITE_ROOT}/${city}/variants/full/formal_figures/nav50_result.jpg"
  local nav51="${SUITE_ROOT}/${city}/variants/full/formal_figures/nav51_result.jpg"

  [[ -s "${prepared}/experiment.json" ]] || {
    echo "ERROR: ${city} prepared data missing; refusing to silently change the interrupted suite" >&2
    return 10
  }

  common_args "${city}" "${gpu}"

  if [[ ! -s "${summary}" ]]; then
    if [[ ! -s "${ck}" ]]; then
      echo "[RESUME TRAIN] ${city} GPU${gpu} checkpoint incomplete -> resume training"
      nice -n "${CPU_NICE}" python3 -u v39_otherdata/bearing_iclr_ablation.py train \
        "${COMMON[@]}" --train-frames 3 \
        2>&1 | tee "${SUITE_ROOT}/logs/${city}_train_resume_gpu${gpu}.log"
      [[ -s "${ck}" ]] || { echo "ERROR: ${city} checkpoint still missing" >&2; return 20; }
    else
      echo "[RESUME TRAIN SKIP] ${city} checkpoint already complete"
    fi

    echo "[RESUME EVAL] ${city} nav50/nav51 GPU${gpu}"
    nice -n "${CPU_NICE}" python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      "${COMMON[@]}" --variant full \
      2>&1 | tee "${SUITE_ROOT}/logs/${city}_test_resume_gpu${gpu}.log"
    [[ -s "${summary}" ]] || { echo "ERROR: ${city} summary missing after eval" >&2; return 21; }
  else
    echo "[RESUME EVAL SKIP] ${city} summary already exists"
  fi

  if [[ ! -s "${nav50}" || ! -s "${nav51}" ]]; then
    plot_city "${city}"
  else
    echo "[RESUME PLOT SKIP] ${city} figures already complete"
  fi

  echo "[RESUME CITY DONE] ${city} GPU${gpu}"
}

echo "================================================================================"
echo "FORMAL V5 SAFE RESUME"
echo "Suite                  : ${SUITE_ROOT}"
echo "CPU threads per process: ${CPU_THREADS_PER_CITY}"
echo "SAT cache batch        : ${UAVSAT_VISUAL_CACHE_BATCH_SIZE}"
echo "SAT cache lock         : ${UAVSAT_SAT_CACHE_LOCK}"
echo "================================================================================"

# Keep the same three-GPU scheduling, but already-complete cities finish almost
# immediately and do not retrain/re-evaluate.
declare -A PID_GPU=()
declare -A PID_CITY=()
launch(){
  local city="$1" gpu="$2"
  ( resume_city "${city}" "${gpu}" ) &
  local pid=$!
  PID_GPU["${pid}"]="${gpu}"
  PID_CITY["${pid}"]="${city}"
  echo "[RESUME SCHEDULER] ${city} -> GPU${gpu} pid=${pid}"
}

launch citya 0
launch cityb 5
launch cityc 6
remaining="cityd"
failed=0
while ((${#PID_GPU[@]} > 0)); do
  done_pid=""
  set +e
  wait -n -p done_pid
  rc=$?
  set -e
  [[ -n "${done_pid}" ]] || { failed=1; break; }
  gpu="${PID_GPU[${done_pid}]}"
  city="${PID_CITY[${done_pid}]}"
  unset 'PID_GPU['"${done_pid}"']'
  unset 'PID_CITY['"${done_pid}"']'
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: resume failed for ${city} on GPU${gpu} rc=${rc}" >&2
    failed=1
  else
    echo "[RESUME SCHEDULER] ${city} complete on GPU${gpu}"
  fi
  if [[ -n "${remaining}" ]]; then
    next="${remaining}"
    remaining=""
    launch "${next}" "${gpu}"
  fi
done
[[ "${failed}" -eq 0 ]] || exit 30

# Verify and aggregate all 8 held-out routes + all 8 figures.
python3 - "${SUITE_ROOT}" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
cities = ['citya','cityb','cityc','cityd']
out = {
    'run_type': 'formal_full_only_resume',
    'method': 'Bearing V5 frozen checkpoint architecture',
    'cities': {},
    'figures': {},
    'macro_average_over_8_held_out_routes': {},
}
rows = []
for city in cities:
    full = root / city / 'variants' / 'full'
    summary_path = full / 'bearing_v39_summary.json'
    if not summary_path.is_file():
        raise RuntimeError(f'missing summary: {summary_path}')
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    out['cities'][city] = data
    for route, metrics in data.items():
        rows.append((city, route, metrics))
    nav50 = full / 'formal_figures' / 'nav50_result.jpg'
    nav51 = full / 'formal_figures' / 'nav51_result.jpg'
    audit = full / 'formal_figures' / 'plot_source_audit.json'
    for p in (nav50, nav51, audit):
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f'missing formal output: {p}')
    out['figures'][city] = {
        'nav50': str(nav50),
        'nav51': str(nav51),
        'plot_source_audit': str(audit),
    }

for key in ('MLE_m','P90_m','LSR@3_pct','LSR@5_pct','LSR@10_pct','LSR@15_pct'):
    vals = [float(m[key]) for _,_,m in rows if key in m]
    if vals:
        out['macro_average_over_8_held_out_routes'][key] = sum(vals)/len(vals)
out['held_out_route_count'] = len(rows)
out['figure_count'] = 8
out['note'] = 'P90_m is a macro-average of per-route P90 values, not pooled-error P90.'
(root / 'formal_allcities_results.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print('[RESUME AGGREGATE] routes=', len(rows), 'figures=', out['figure_count'])
print(json.dumps(out['macro_average_over_8_held_out_routes'], indent=2))
PY

# Upload the completed resumed suite.
if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-formal-resume-upload-XXXXXX")"
  upload_branch="formal-v5-resume-upload-${TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin bearing-v5-formal-allcities
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/bearing-v5-formal-allcities
  dest="paper_results/formal_bearing_v5_allcities_resume_${TS}"
  mkdir -p "${upload_wt}/${dest}"
  cp "${SUITE_ROOT}/formal_allcities_results.json" "${upload_wt}/${dest}/"
  for city in "${CITIES[@]}"; do
    src="${SUITE_ROOT}/${city}"
    dst="${upload_wt}/${dest}/${city}"
    mkdir -p "${dst}/train_frames3" "${dst}/full/formal_figures"
    cp "${src}/prepared/experiment.json" "${dst}/prepared_experiment.json"
    cp "${src}/train_frames3/experiment_manifest.json" "${dst}/train_frames3/" 2>/dev/null || true
    cp "${src}/train_frames3/kalman_calibration.json" "${dst}/train_frames3/" 2>/dev/null || true
    cp "${src}/variants/full/bearing_v39_summary.json" "${dst}/full/"
    cp "${src}/variants/full/experiment_manifest.json" "${dst}/full/" 2>/dev/null || true
    cp "${src}/variants/full"/*_frames.csv "${dst}/full/" 2>/dev/null || true
    cp "${src}/variants/full/formal_figures/nav50_result.jpg" "${dst}/full/formal_figures/"
    cp "${src}/variants/full/formal_figures/nav51_result.jpg" "${dst}/full/formal_figures/"
    cp "${src}/variants/full/formal_figures/plot_source_audit.json" "${dst}/full/formal_figures/"
  done
  (
    cd "${upload_wt}"
    git add "${dest}"
    git commit -m "Add resumed formal Bearing V5 all-city results ${TS}"
    git fetch origin bearing-v5-formal-allcities
    git rebase origin/bearing-v5-formal-allcities
    git push origin HEAD:bearing-v5-formal-allcities
  )
  echo "[RESUME UPLOAD] ${dest}"
fi

echo "================================================================================"
echo "DONE: FORMAL V5 SAFE RESUME"
echo "Suite   : ${SUITE_ROOT}"
echo "Summary : ${SUITE_ROOT}/formal_allcities_results.json"
echo "Figures : 8 total (nav50/nav51 x 4 cities)"
echo "================================================================================"
