#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SUITE="${1:-}"
[[ -n "${SUITE}" && -d "${SUITE}" ]] || { echo "ERROR: pass completed Smooth-V1 suite" >&2; exit 2; }
SUITE="$(readlink -f "${SUITE}")"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SEED="${SEED:-2033}"
THREADS="${CPU_THREADS_PER_CITY:-2}"
FORCE="${FORCE_CORE_V5:-1}"
UPLOAD="${UPLOAD_CORE_V5:-1}"
BRANCH="bearing-v5-formal-smooth-v1"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
export MALLOC_ARENA_MAX=2

python3 v39_otherdata/patch_core_v5_restore_cadence.py
python3 -m py_compile \
  v39_otherdata/bearing_core_v5_restore.py \
  v39_otherdata/build_bearing_core_v5_restore_tables.py \
  v39_otherdata/bearing_paper_ablation.py

python3 - <<'PY'
from pathlib import Path
s=Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
checks={
 'formal_v5_calibration':'formal_v5_direct_delta2_residual_kalman' in s,
 'frames':'EXPERIMENT_FRAME_COUNT' in s,
 'no_kalman':'no_kalman' in s,
 'train':'train_temporal_model' in s,
}
for k,v in checks.items(): print(f'[CORE-V5 PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()): raise SystemExit('ERROR: local bearing_iclr_ablation.py is not the completed Formal-V5 runner; do not overwrite it.')
PY

mkdir -p "${SUITE}/logs"
COMMON=(
  --suite-root "${SUITE}" --dataset-root "${DATASET_ROOT}"
  --backbone mobilenet_v3_small --visual-epochs 30 --temporal-epochs 100
  --epochs-per-route 100 --patience 4 --jitter-m 8 --max-sample-distance-m 15
  --heading-weight-px-per-deg 0 --ms-bandwidth-m 7 --seed "${SEED}"
)

train_city(){
  local city
  local gpu
  local train_root
  city="$1"
  gpu="$2"
  train_root="${SUITE}/${city}/train_core_v5_restore"
  if [[ "${FORCE}" == "1" ]]; then rm -rf "${train_root}"; fi
  echo "================================================================================"
  echo "[CORE-V5 TRAIN RESTORED 3F] ${city} GPU${gpu}"
  echo "================================================================================"
  python3 -u v39_otherdata/bearing_core_v5_restore.py train "${COMMON[@]}" --city "${city}" --gpu "${gpu}" \
    2>&1 | tee "${SUITE}/logs/core_v5_${city}_train.log"
}
train_city citya 0 & PA=$!
train_city cityb 5 & PB=$!
train_city cityc 6 & PC=$!
failed=0; wait "${PA}" || failed=1; wait "${PB}" || failed=1; wait "${PC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: citya/b/c restored training failed" >&2; exit 30; }
train_city cityd 0

VARIANTS=(corev5_full corev5_no_gru corev5_no_kalman corev5_no_ms corev5_ctx1 corev5_ctx2)
run_city(){
  local city
  local gpu
  local variant
  local out
  city="$1"
  gpu="$2"
  for variant in "${VARIANTS[@]}"; do
    out="${SUITE}/${city}/variants_core_v5_restore/${variant}"
    if [[ "${FORCE}" == "1" ]]; then rm -rf "${out}"; fi
    echo "================================================================================"
    echo "[CORE-V5 EVAL] ${city} ${variant} GPU${gpu}"
    echo "================================================================================"
    python3 -u v39_otherdata/bearing_core_v5_restore.py eval "${COMMON[@]}" --city "${city}" --gpu "${gpu}" --variant "${variant}" \
      2>&1 | tee "${SUITE}/logs/core_v5_${city}_${variant}.log"
    [[ -s "${out}/bearing_v39_summary.json" ]] || { echo "ERROR: missing ${out}/bearing_v39_summary.json" >&2; return 40; }
  done
}
run_city citya 0 & PA=$!
run_city cityb 5 & PB=$!
run_city cityc 6 & PC=$!
failed=0; wait "${PA}" || failed=1; wait "${PB}" || failed=1; wait "${PC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: citya/b/c Core V5 eval failed" >&2; exit 50; }
run_city cityd 0

rm -rf "${SUITE}/paper_core_v5_restore"
python3 -u v39_otherdata/build_bearing_core_v5_restore_tables.py --suite-root "${SUITE}" --output-dir "${SUITE}/paper_core_v5_restore"
cat "${SUITE}/paper_core_v5_restore/PAPER_CORE_V5_RESTORE_TABLES.md"

echo "================================================================================"
echo "CORE V5-RESTORE LOCAL COMPLETE"
echo "Tables: ${SUITE}/paper_core_v5_restore/PAPER_CORE_V5_RESTORE_TABLES.md"
echo "Architecture unchanged; pre-Smooth-V1 lateral/heading dynamics restored."
echo "Longitudinal constraints remain train_01 cadence-derived exactly as old V5."
echo "1f/2f are same-checkpoint context truncations; no multi-seed run."
echo "================================================================================"

if [[ "${UPLOAD}" != "1" ]]; then exit 0; fi
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST_REL="paper_results/formal_bearing_v5_core_v5_restore_${STAMP}"
TMP="$(mktemp -d /tmp/uavsat-corev5-upload.XXXXXX)"
cleanup(){ git worktree remove --force "${TMP}" >/dev/null 2>&1 || true; rm -rf "${TMP}"; }
trap cleanup EXIT

git fetch origin "${BRANCH}"
git worktree add --detach "${TMP}" "origin/${BRANCH}" >/dev/null
mkdir -p "${TMP}/${DEST_REL}/paper_core_v5_restore"
cp -a "${SUITE}/paper_core_v5_restore/." "${TMP}/${DEST_REL}/paper_core_v5_restore/"
cat > "${TMP}/${DEST_REL}/PROTOCOL.txt" <<'EOF'
Core V5-Restore:
- Same architecture: Forward-18 SoftMS -> 3-frame recurrent GRU -> constrained Kalman -> final MeanShift -> XY.
- Restores pre-Smooth-V1 lateral/heading estimator dynamics and retrains the 3-frame temporal checkpoint.
- Longitudinal motion/Kalman limits remain owned by train_01-only cadence adaptation, matching old V5.
- Existing visual checkpoint is reused; temporal/estimator state is retrained on train_01.
- Component table: w/o GRU, w/o Kalman, w/o Final MeanShift, Full.
- Temporal table: same restored 3-frame checkpoint with 1f/2f context truncation.
- Held-out nav50/nav51 are not used for automatic parameter selection or numeric editing.
- Localization protocol remains controlled_gt_jitter.
EOF
for city in citya cityb cityc cityd; do
  cp -a "${SUITE}/${city}/train_core_v5_restore/core_v5_restore_manifest.json" "${TMP}/${DEST_REL}/${city}_train_manifest.json"
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE}/${city}/variants_core_v5_restore/${variant}"
    dst="${TMP}/${DEST_REL}/${city}/${variant}"
    mkdir -p "${dst}"
    cp "${src}/bearing_v39_summary.json" "${dst}/"
    cp "${src}/core_v5_restore_manifest.json" "${dst}/"
  done
done
cd "${TMP}"
git add "${DEST_REL}"
git -c user.name="OpenAI" -c user.email="noreply@openai.com" commit -m "Upload Bearing V5 Core V5 restored results ${STAMP}" >/dev/null
git push origin "HEAD:${BRANCH}"
echo "[CORE V5 RESTORE UPLOAD DONE] ${DEST_REL}"
