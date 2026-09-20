#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SUITE="${1:-}"
if [[ -z "${SUITE}" ]]; then
  SUITE="$(find "${ROOT}/v39_otherdata" -maxdepth 1 -type d -name 'formal_bearing_v5_smooth_*' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
[[ -n "${SUITE}" && -d "${SUITE}" ]] || { echo "ERROR: completed smooth suite not found" >&2; exit 2; }
SUITE="$(readlink -f "${SUITE}")"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
BACKBONE="${BACKBONE:-mobilenet_v3_small}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-4}"
SEED="${SEED:-2033}"
FORCE_ABLATIONS="${FORCE_ABLATIONS:-0}"
UPLOAD_ABLATIONS="${UPLOAD_ABLATIONS:-1}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_CITY:-2}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_CITY:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_CITY:-2}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_CITY:-2}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS_PER_CITY:-2}"
export BLIS_NUM_THREADS="${CPU_THREADS_PER_CITY:-2}"
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"

python3 -m py_compile \
  v39_otherdata/bearing_required_ablation.py \
  v39_otherdata/build_bearing_required_ablation_tables.py

# Do not silently evaluate the unpatched legacy source.  The completed smooth
# run leaves the generated Formal-V5 runner in this workspace.
python3 - <<'PY'
from pathlib import Path
p=Path('v39_otherdata/bearing_iclr_ablation.py')
s=p.read_text(encoding='utf-8')
checks={
 'formal_v5_calibration':'formal_v5_direct_delta2_residual_kalman' in s,
 'legacy_context_gru_disabled':'base._patch_context_gru(runtime_root)' not in s,
}
for k,v in checks.items(): print(f'[ABLATION PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('ERROR: local bearing_iclr_ablation.py is not the generated Formal-V5 runner from the completed smooth run. Do not overwrite it from the branch before running this suite.')
PY

for city in citya cityb cityc cityd; do
  [[ -s "${SUITE}/${city}/variants/full/bearing_v39_summary.json" ]] || {
    echo "ERROR: missing completed Full result for ${city}" >&2; exit 3;
  }
  [[ -s "${SUITE}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" || \
     -s "${SUITE}/${city}/train_full/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" ]] || {
    echo "ERROR: missing Full temporal checkpoint for ${city}" >&2; exit 4;
  }
done

VARIANTS=(
  no_gru no_kalman no_ms
  frames1 frames2
  grid4 grid5 grid7 grid8
  full36
  front_top1 front_softms
  no_heading_feedback
  jitter0 jitter4 jitter12 jitter16
)

run_city(){
  local city="$1" gpu="$2" variant summary
  echo "================================================================================"
  echo "[ABLATION CITY START] ${city} GPU${gpu}"
  echo "================================================================================"
  for variant in "${VARIANTS[@]}"; do
    summary="${SUITE}/${city}/variants/${variant}/bearing_v39_summary.json"
    if [[ "${FORCE_ABLATIONS}" != "1" && -s "${summary}" ]]; then
      echo "[ABLATION SKIP] ${city} ${variant} already exists"
      continue
    fi
    echo "------------------------------------------------------------------------"
    echo "[ABLATION RUN] ${city} ${variant} GPU${gpu}"
    echo "------------------------------------------------------------------------"
    python3 -u v39_otherdata/bearing_required_ablation.py \
      --suite-root "${SUITE}" \
      --dataset-root "${DATASET_ROOT}" \
      --city "${city}" \
      --gpu "${gpu}" \
      --variant "${variant}" \
      --backbone "${BACKBONE}" \
      --visual-epochs "${VISUAL_EPOCHS}" \
      --temporal-epochs "${TEMPORAL_EPOCHS}" \
      --epochs-per-route "${TEMPORAL_EPOCHS}" \
      --patience "${PATIENCE}" \
      --max-sample-distance-m 15 \
      --heading-weight-px-per-deg 0 \
      --ms-bandwidth-m 7 \
      --seed "${SEED}" \
      2>&1 | tee "${SUITE}/logs/${city}_paper_ablation_${variant}_gpu${gpu}.log"
    [[ -s "${summary}" ]] || { echo "ERROR: summary missing after ${city} ${variant}" >&2; return 20; }
  done
  echo "[ABLATION CITY DONE] ${city} GPU${gpu}"
}

# Three independent cities in parallel; CityD starts after the first batch to
# keep GPU/CPU pressure predictable.
run_city citya 0 & PIDA=$!
run_city cityb 5 & PIDB=$!
run_city cityc 6 & PIDC=$!
failed=0
wait "${PIDA}" || failed=1
wait "${PIDB}" || failed=1
wait "${PIDC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: A/B/C ablation failed" >&2; exit 30; }
run_city cityd 0

python3 -u v39_otherdata/build_bearing_required_ablation_tables.py \
  --suite-root "${SUITE}"

echo "================================================================================"
echo "[PAPER ABLATION COMPLETE]"
echo "Suite  : ${SUITE}"
echo "Tables : ${SUITE}/paper_ablation/ABLATION_TABLES.md"
echo "JSON   : ${SUITE}/paper_ablation/ablation_results.json"
echo "================================================================================"

if [[ "${UPLOAD_ABLATIONS}" == "1" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  DEST="paper_results/formal_bearing_v5_ablation_${STAMP}"
  TMP="$(mktemp -d "${ROOT%/*}/uav-sat-ablation-upload-XXXXXX")"
  BR="ablation-upload-${STAMP}-$$"
  cleanup(){
    git -C "${ROOT}" worktree remove --force "${TMP}" >/dev/null 2>&1 || true
    git -C "${ROOT}" branch -D "${BR}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin bearing-v5-formal-smooth-v1
  git worktree add -b "${BR}" "${TMP}" origin/bearing-v5-formal-smooth-v1
  mkdir -p "${TMP}/${DEST}"
  cp -r "${SUITE}/paper_ablation" "${TMP}/${DEST}/"
  for city in citya cityb cityc cityd; do
    for variant in full "${VARIANTS[@]}"; do
      src="${SUITE}/${city}/variants/${variant}"
      [[ -s "${src}/bearing_v39_summary.json" ]] || continue
      dst="${TMP}/${DEST}/${city}/${variant}"
      mkdir -p "${dst}"
      cp "${src}/bearing_v39_summary.json" "${dst}/"
      cp "${src}/experiment_manifest.json" "${dst}/" 2>/dev/null || true
    done
  done
  git -C "${TMP}" add "${DEST}"
  git -C "${TMP}" -c user.name='OpenAI Results Uploader' -c user.email='results@local' \
    commit -m "Upload Bearing V5 paper ablations ${STAMP}"
  git -C "${TMP}" push origin "HEAD:bearing-v5-formal-smooth-v1"
  echo "[ABLATION UPLOAD DONE] ${DEST}"
fi
