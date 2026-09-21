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
FORCE="${FORCE_CORE_V4:-1}"
UPLOAD="${UPLOAD_CORE_V4:-1}"
BRANCH="bearing-v5-formal-smooth-v1"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
export MALLOC_ARENA_MAX=2

python3 -m py_compile \
  v39_otherdata/bearing_core_v4_ablation.py \
  v39_otherdata/build_bearing_core_v4_tables.py \
  v39_otherdata/bearing_paper_ablation.py

python3 - <<'PY'
from pathlib import Path
p=Path('v39_otherdata/bearing_iclr_ablation.py')
s=p.read_text(encoding='utf-8')
checks={
 'formal_v5_calibration':'formal_v5_direct_delta2_residual_kalman' in s,
 'frames':'EXPERIMENT_FRAME_COUNT' in s,
 'no_kalman':'no_kalman' in s,
 'nav50':'nav50' in s or 'test_01' in s,
}
for k,v in checks.items(): print(f'[CORE-V4 PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('ERROR: local bearing_iclr_ablation.py is not the completed Formal-V5 runner; do not overwrite it with the GitHub template.')
PY

VARIANTS=(corev4_full corev4_no_gru corev4_no_kalman corev4_no_ms corev4_ctx1 corev4_ctx2)

run_city(){
  local city="$1" gpu="$2" variant out
  for variant in "${VARIANTS[@]}"; do
    out="${SUITE}/${city}/variants_core_v4/${variant}"
    if [[ "${FORCE}" == "1" ]]; then rm -rf "${out}"; fi
    if [[ -s "${out}/bearing_v39_summary.json" ]]; then
      echo "[CORE-V4 SKIP] ${city} ${variant}"
      continue
    fi
    echo "================================================================================"
    echo "[CORE-V4 EVAL] ${city} ${variant} GPU${gpu}"
    echo "================================================================================"
    python3 -u v39_otherdata/bearing_core_v4_ablation.py \
      --suite-root "${SUITE}" \
      --dataset-root "${DATASET_ROOT}" \
      --city "${city}" \
      --gpu "${gpu}" \
      --variant "${variant}" \
      --backbone mobilenet_v3_small \
      --visual-epochs 30 \
      --temporal-epochs 100 \
      --epochs-per-route 100 \
      --patience 4 \
      --jitter-m 8 \
      --max-sample-distance-m 15 \
      --heading-weight-px-per-deg 0 \
      --ms-bandwidth-m 7 \
      --seed "${SEED}" \
      2>&1 | tee "${SUITE}/logs/core_v4_${city}_${variant}.log"
    [[ -s "${out}/bearing_v39_summary.json" ]] || { echo "ERROR: missing ${out}/bearing_v39_summary.json" >&2; return 20; }
  done
}

mkdir -p "${SUITE}/logs"
run_city citya 0 & PA=$!
run_city cityb 5 & PB=$!
run_city cityc 6 & PC=$!
failed=0
wait "${PA}" || failed=1
wait "${PB}" || failed=1
wait "${PC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: citya/b/c Core V4 failed" >&2; exit 30; }
run_city cityd 0

rm -rf "${SUITE}/paper_core_v4"
python3 -u v39_otherdata/build_bearing_core_v4_tables.py \
  --suite-root "${SUITE}" \
  --output-dir "${SUITE}/paper_core_v4"

echo "================================================================================"
echo "CORE V4 LOCAL COMPLETE"
echo "Tables: ${SUITE}/paper_core_v4/PAPER_CORE_V4_TABLES.md"
echo "Temporal protocol: same trained 3f Full checkpoint; 1f/2f are context truncations."
echo "Core protocol: w/o GRU / w/o Kalman / w/o Final MS / Full."
echo "================================================================================"

if [[ "${UPLOAD}" != "1" ]]; then exit 0; fi

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST_REL="paper_results/formal_bearing_v5_core_v4_${STAMP}"
TMP="$(mktemp -d /tmp/uavsat-corev4-upload.XXXXXX)"
cleanup(){ git worktree remove --force "${TMP}" >/dev/null 2>&1 || true; rm -rf "${TMP}"; }
trap cleanup EXIT

git fetch origin "${BRANCH}"
git worktree add --detach "${TMP}" "origin/${BRANCH}" >/dev/null
mkdir -p "${TMP}/${DEST_REL}/paper_core_v4"
cp -a "${SUITE}/paper_core_v4/." "${TMP}/${DEST_REL}/paper_core_v4/"
cat > "${TMP}/${DEST_REL}/PROTOCOL.txt" <<'EOF'
Core V4:
- Core component table includes w/o GRU, w/o Kalman, w/o Final MeanShift, Full.
- Temporal table restores same-checkpoint context truncation: 1f/2f/3f all reuse the trained 3-frame Full checkpoint.
- Kalman/runtime profile is selected from train_01 validation calibration only.
- Held-out nav50/nav51 metrics are never used for automatic profile selection or retuning.
- Overall chain remains Forward-18 SoftMS -> recurrent GRU -> constrained Kalman -> final MeanShift -> XY.
- Current localization protocol remains controlled_gt_jitter.
EOF

for city in citya cityb cityc cityd; do
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE}/${city}/variants_core_v4/${variant}"
    dst="${TMP}/${DEST_REL}/${city}/${variant}"
    mkdir -p "${dst}"
    cp "${src}/bearing_v39_summary.json" "${dst}/"
    cp "${src}/core_v4_manifest.json" "${dst}/"
  done
done

cd "${TMP}"
git add "${DEST_REL}"
git -c user.name="OpenAI" -c user.email="noreply@openai.com" commit -m "Upload Bearing V5 Core V4 results ${STAMP}" >/dev/null
git push origin "HEAD:${BRANCH}"
echo "[CORE V4 UPLOAD DONE] ${DEST_REL}"
