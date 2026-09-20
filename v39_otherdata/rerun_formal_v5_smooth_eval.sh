#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${FORMAL_SUITE_ROOT:?Set FORMAL_SUITE_ROOT to the completed Formal V5 suite}"
SUITE_ROOT="$(readlink -f "${FORMAL_SUITE_ROOT}")"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SEED="${SEED:-2033}"
CPU_THREADS_PER_CITY="${CPU_THREADS_PER_CITY:-2}"
CPU_NICE="${CPU_NICE:-5}"
MS_OUTPUT_BLEND="${MS_OUTPUT_BLEND:-0.35}"
MS_OUTPUT_RESIDUAL_CAP_M="${MS_OUTPUT_RESIDUAL_CAP_M:-3.0}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
CITIES=(citya cityb cityc cityd)

[[ -d "${SUITE_ROOT}" ]] || { echo "ERROR: suite not found: ${SUITE_ROOT}" >&2; exit 2; }
[[ -s v39_otherdata/bearing_iclr_ablation.py ]] || { echo "ERROR: local generated V5 runner missing" >&2; exit 3; }
[[ -s v39_otherdata/patch_formal_v5_smooth_output.py ]] || { echo "ERROR: smooth patch missing" >&2; exit 4; }

mkdir -p "${SUITE_ROOT}/logs"

export OMP_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS_PER_CITY}"
export BLIS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export TOKENIZERS_PARALLELISM=false

export MS_OUTPUT_BLEND
export MS_OUTPUT_RESIDUAL_CAP_M

# Preserve the same Formal V5 inference configuration used by the completed run.
export UAVSAT_LOSS_MEASUREMENT="${UAVSAT_LOSS_MEASUREMENT:-2.0}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-3.0}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-0.40}"
export UAVSAT_LOSS_ACCELERATION="${UAVSAT_LOSS_ACCELERATION:-0.50}"
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-8e-5}"
export UAVSAT_RNN_DROPOUT="${UAVSAT_RNN_DROPOUT:-0.05}"
export UAVSAT_EARLY_MIN_EPOCH="${UAVSAT_EARLY_MIN_EPOCH:-10}"
export UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE:-0.45}"
export UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE:-1.00}"
export UAVSAT_TEMPORAL_DELTA2_SCALE="${UAVSAT_TEMPORAL_DELTA2_SCALE:-1.00}"
export UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M="${UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M:-1.25}"
export UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M="${UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M:-0.75}"
export UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M="${UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M:-2.00}"
export UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M="${UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M:-1.00}"

# Patch the LOCAL generated Formal V5 runner so every newly built runtime applies
# patch_direct_finalms.py first, then the bounded-output smoother. Do not replace
# the generated runner with the branch-base bearing_iclr_ablation.py.
python3 - <<'PY'
from pathlib import Path

p = Path('v39_otherdata/bearing_iclr_ablation.py')
s = p.read_text(encoding='utf-8')

required = {
    'formal_v5_calibration': 'formal_v5_direct_delta2_residual_kalman' in s,
    'legacy_context_gru_disabled': 'base._patch_context_gru(runtime_root)' not in s,
}
for key, ok in required.items():
    print(f'[SMOOTH PRECHECK] {key}: {"PASS" if ok else "FAIL"}')
if not all(required.values()):
    raise SystemExit('Local bearing_iclr_ablation.py is not the generated Formal V5 runner.')

if 'patch_formal_v5_smooth_output.py' not in s:
    anchor = '''    subprocess.run(
        [
            sys.executable,
            str(base.CANONICAL_FINALMS_PATCH),
            str(runtime_root / "robust_tracker.py"),
        ],
        check=True,
    )
'''
    if s.count(anchor) != 1:
        raise SystemExit(f'final-MS runtime patch anchor matches={s.count(anchor)}')
    addition = anchor + '''    smooth_output_patch = HERE / "patch_formal_v5_smooth_output.py"
    if not smooth_output_patch.exists():
        raise FileNotFoundError(smooth_output_patch)
    subprocess.run(
        [
            sys.executable,
            str(smooth_output_patch),
            str(runtime_root / "robust_tracker.py"),
        ],
        check=True,
    )
'''
    s = s.replace(anchor, addition, 1)
    compile(s, str(p), 'exec')
    backup = p.with_suffix('.py.pre_smooth_output')
    if not backup.exists():
        backup.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    p.write_text(s, encoding='utf-8')
    print('[SMOOTH RUNNER PATCH] installed')
else:
    print('[SMOOTH RUNNER PATCH] already installed')
PY

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/patch_formal_v5_smooth_output.py \
  v39_otherdata/bearing_plot_final_vs_gt.py

common_args(){
  local city="$1" gpu="$2"
  COMMON=(
    --suite-root "${SUITE_ROOT}"
    --dataset-root "${DATASET_ROOT}"
    --city "${city}"
    --gpu "${gpu}"
    --backbone mobilenet_v3_small
    --visual-epochs 30
    --temporal-epochs 100
    --epochs-per-route 100
    --patience 4
    --jitter-m 8
    --max-sample-distance-m 15
    --heading-weight-px-per-deg 0
    --ms-bandwidth-m 7
    --seed "${SEED}"
  )
}

run_eval(){
  local city="$1" gpu="$2"
  local train_root="${SUITE_ROOT}/${city}/train_frames3"
  local ck="${train_root}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  local full="${SUITE_ROOT}/${city}/variants/full"
  local backup="${SUITE_ROOT}/${city}/variants/full_before_smooth_${RUN_TS}"
  local figures="${full}/formal_figures"

  [[ -s "${ck}" ]] || { echo "ERROR: ${city} checkpoint missing: ${ck}" >&2; return 20; }
  [[ -s "${SUITE_ROOT}/${city}/prepared/experiment.json" ]] || { echo "ERROR: ${city} prepared data missing" >&2; return 21; }

  if [[ -d "${full}" && ! -e "${backup}" ]]; then
    cp -a "${full}" "${backup}"
    echo "[SMOOTH BACKUP] ${city} -> ${backup}"
  fi

  common_args "${city}" "${gpu}"
  echo "================================================================================"
  echo "[SMOOTH EVAL] ${city} GPU${gpu} | blend=${MS_OUTPUT_BLEND} cap=${MS_OUTPUT_RESIDUAL_CAP_M}m"
  echo "================================================================================"
  nice -n "${CPU_NICE}" python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
    "${COMMON[@]}" --variant full \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_smooth_eval_gpu${gpu}.log"

  [[ -s "${full}/bearing_v39_summary.json" ]] || { echo "ERROR: ${city} smooth summary missing" >&2; return 22; }

  python3 -u v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "${SUITE_ROOT}/${city}/prepared" \
    --output-dir "${full}" \
    --routes test_01 test_02 \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_smooth_plot.log"

  mkdir -p "${figures}"
  cp "${full}/paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg" "${figures}/nav50_result.jpg"
  cp "${full}/paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg" "${figures}/nav51_result.jpg"
  cp "${full}/paper_figures_waypoint_gt/plot_source_audit.json" "${figures}/plot_source_audit.json"
  [[ -s "${figures}/nav50_result.jpg" && -s "${figures}/nav51_result.jpg" ]] || return 23

  python3 - "${full}/bearing_v39_summary.json" "${city}" <<'PY'
import json, sys
p, city = sys.argv[1], sys.argv[2]
data = json.load(open(p))
for route, m in data.items():
    print(
        f"[SMOOTH RESULT] {city} {route}: "
        f"MLE={float(m.get('MLE_m', float('nan'))):.3f}m "
        f"P90={float(m.get('P90_m', float('nan'))):.3f}m "
        f"jump={float(m.get('JumpRate_pct', float('nan'))):.2f}% "
        f"max_step={float(m.get('MaxFinalStep_m', float('nan'))):.2f}m"
    )
PY
}

# A/B/C in parallel on GPU 0/5/6; D starts on the first GPU that becomes free.
declare -A PID_GPU=()
declare -A PID_CITY=()
launch(){
  local city="$1" gpu="$2"
  ( run_eval "${city}" "${gpu}" ) &
  local pid=$!
  PID_GPU["${pid}"]="${gpu}"
  PID_CITY["${pid}"]="${city}"
  echo "[SMOOTH SCHEDULER] ${city} -> GPU${gpu} pid=${pid}"
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
    echo "ERROR: ${city} smooth eval failed on GPU${gpu} rc=${rc}" >&2
    failed=1
  else
    echo "[SMOOTH SCHEDULER] ${city} complete on GPU${gpu}"
  fi
  if [[ -n "${remaining}" ]]; then
    next="${remaining}"
    remaining=""
    launch "${next}" "${gpu}"
  fi
done
[[ "${failed}" -eq 0 ]] || exit 30

python3 - "${SUITE_ROOT}" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
cities = ['citya','cityb','cityc','cityd']
out = {
    'run_type': 'formal_v5_smooth_eval_only',
    'output_refinement': 'Kalman + bounded ambiguity-aware MeanShift residual',
    'cities': {},
    'macro_average_over_8_held_out_routes': {},
    'figure_count': 0,
}
rows=[]
for city in cities:
    full = root / city / 'variants' / 'full'
    data = json.loads((full/'bearing_v39_summary.json').read_text())
    out['cities'][city]=data
    rows.extend(data.values())
    for nav in ('nav50','nav51'):
        p=full/'formal_figures'/f'{nav}_result.jpg'
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f'missing figure: {p}')
        out['figure_count'] += 1
for key in ('MLE_m','P90_m','LSR@5_pct','LSR@10_pct','LSR@15_pct','JumpRate_pct','MaxFinalStep_m'):
    vals=[float(m[key]) for m in rows if key in m]
    if vals:
        out['macro_average_over_8_held_out_routes'][key]=sum(vals)/len(vals)
(root/'formal_allcities_results_smooth.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('[SMOOTH AGGREGATE] routes=',len(rows),'figures=',out['figure_count'])
print(json.dumps(out['macro_average_over_8_held_out_routes'],indent=2))
PY

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-smooth-upload-XXXXXX")"
  upload_branch="formal-v5-smooth-upload-${RUN_TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin bearing-v5-formal-allcities
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/bearing-v5-formal-allcities
  dest="paper_results/formal_bearing_v5_smooth_${RUN_TS}"
  mkdir -p "${upload_wt}/${dest}"
  cp "${SUITE_ROOT}/formal_allcities_results_smooth.json" "${upload_wt}/${dest}/"
  for city in "${CITIES[@]}"; do
    src="${SUITE_ROOT}/${city}/variants/full"
    dst="${upload_wt}/${dest}/${city}"
    mkdir -p "${dst}/formal_figures"
    cp "${src}/bearing_v39_summary.json" "${dst}/"
    cp "${src}"/*_frames.csv "${dst}/" 2>/dev/null || true
    cp "${src}/formal_figures/nav50_result.jpg" "${dst}/formal_figures/"
    cp "${src}/formal_figures/nav51_result.jpg" "${dst}/formal_figures/"
    cp "${src}/formal_figures/plot_source_audit.json" "${dst}/formal_figures/"
  done
  (
    cd "${upload_wt}"
    git add "${dest}"
    git commit -m "Add smooth Formal V5 eval results ${RUN_TS}"
    git fetch origin bearing-v5-formal-allcities
    git rebase origin/bearing-v5-formal-allcities
    git push origin HEAD:bearing-v5-formal-allcities
  )
  echo "[SMOOTH UPLOAD] ${dest}"
fi

echo "================================================================================"
echo "DONE: FORMAL V5 SMOOTH EVAL ONLY"
echo "No training was repeated. Existing checkpoints were reused."
echo "Summary: ${SUITE_ROOT}/formal_allcities_results_smooth.json"
echo "================================================================================"
