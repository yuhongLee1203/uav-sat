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
THREADS="${CPU_THREADS_PER_CITY:-2}"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"

python3 -m py_compile \
  v39_otherdata/bearing_required_ablation.py \
  v39_otherdata/build_bearing_paper_ablation_tables.py

# IMPORTANT: do not overwrite bearing_iclr_ablation.py from GitHub before this
# run. The completed Smooth-V1 run leaves the generated Formal-V5 runner in the
# local workspace; the repository source is the pre-generation template.
python3 - <<'PY'
from pathlib import Path
s=Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
checks={
 'formal_v5_calibration':'formal_v5_direct_delta2_residual_kalman' in s,
 'legacy_context_gru_disabled':'base._patch_context_gru(runtime_root)' not in s,
 'train_frames_cli':'--train-frames' in s,
 'frame_specific_checkpoints':'checkpoint_frames = int(variant["frames"])' in s,
}
for k,v in checks.items(): print(f'[PAPER ABLATION PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('ERROR: local bearing_iclr_ablation.py is not the generated Formal-V5 runner from the completed Smooth-V1 run. Restore the completed-run workspace; do NOT replace this file with the branch template.')
PY

for city in citya cityb cityc cityd; do
  [[ -s "${SUITE}/${city}/variants/full/bearing_v39_summary.json" ]] || {
    echo "ERROR: missing completed Full summary for ${city}" >&2; exit 3;
  }
  CK3="${SUITE}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  VIS3="${SUITE}/${city}/train_frames3/checkpoints/visual_retrieval_A_only.pt"
  [[ -s "${CK3}" ]] || { echo "ERROR: missing train_frames3 temporal checkpoint for ${city}: ${CK3}" >&2; exit 4; }
  [[ -e "${VIS3}" ]] || { echo "ERROR: missing train_frames3 visual checkpoint for ${city}: ${VIS3}" >&2; exit 5; }
done

common_args(){
  local city="$1" gpu="$2"
  COMMON=(
    --suite-root "${SUITE}"
    --dataset-root "${DATASET_ROOT}"
    --city "${city}"
    --gpu "${gpu}"
    --backbone "${BACKBONE}"
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

ensure_temporal_context_checkpoints(){
  local city="$1" gpu="$2" frames ck dstvis srcvis
  srcvis="${SUITE}/${city}/train_frames3/checkpoints/visual_retrieval_A_only.pt"
  for frames in 1 2; do
    ck="${SUITE}/${city}/train_frames${frames}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
    dstvis="${SUITE}/${city}/train_frames${frames}/checkpoints/visual_retrieval_A_only.pt"
    if [[ -s "${ck}" && "${FORCE_ABLATIONS}" != "1" ]]; then
      echo "[TEMPORAL TRAIN SKIP] ${city} frames=${frames} checkpoint exists"
      continue
    fi
    mkdir -p "$(dirname "${dstvis}")"
    if [[ ! -e "${dstvis}" ]]; then
      ln -s "$(readlink -f "${srcvis}")" "${dstvis}"
    fi
    common_args "${city}" "${gpu}"
    echo "================================================================================"
    echo "[TEMPORAL RETRAIN] ${city} ${frames}-frame GPU${gpu}"
    echo "Visual checkpoint reused; temporal model trained separately"
    echo "================================================================================"
    python3 -u v39_otherdata/bearing_iclr_ablation.py train \
      "${COMMON[@]}" --train-frames "${frames}" \
      2>&1 | tee "${SUITE}/logs/${city}_paper_train_frames${frames}_gpu${gpu}.log"
    [[ -s "${ck}" ]] || { echo "ERROR: ${city} frames=${frames} checkpoint missing after training" >&2; return 11; }
  done
}

VARIANTS=(
  no_gru no_kalman no_ms
  frames1 frames2
  grid4 grid5 grid7 grid8
  full36 front_top1 front_softms no_heading_feedback
  jitter0 jitter4 jitter12 jitter16
)

patch_temporal_metadata(){
  local city="$1" variant="$2"
  python3 - "${SUITE}" "${city}" "${variant}" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]); city=sys.argv[2]; variant=sys.argv[3]
p=root/city/'variants'/variant/'bearing_v39_summary.json'
d=json.loads(p.read_text())
if variant in {'frames1','frames2'}:
    for row in d.values():
        proto=row.setdefault('PaperAblationProtocol',{})
        proto['ablation_type']='retrained_temporal_context_ablation'
        proto['retrained_for_variant']=True
        proto['training_checkpoint']=f'train_{variant.replace("frames", "frames")}'
p.write_text(json.dumps(d,indent=2,default=float))
PY
}

run_city(){
  local city="$1" gpu="$2" variant summary
  ensure_temporal_context_checkpoints "${city}" "${gpu}"
  for variant in "${VARIANTS[@]}"; do
    summary="${SUITE}/${city}/variants/${variant}/bearing_v39_summary.json"
    if [[ "${FORCE_ABLATIONS}" != "1" && -s "${summary}" ]]; then
      echo "[ABLATION SKIP] ${city} ${variant} already exists"
      patch_temporal_metadata "${city}" "${variant}"
      continue
    fi
    common_args "${city}" "${gpu}"
    echo "------------------------------------------------------------------------"
    echo "[PAPER ABLATION] ${city} ${variant} GPU${gpu}"
    echo "------------------------------------------------------------------------"
    python3 -u v39_otherdata/bearing_required_ablation.py \
      "${COMMON[@]}" --variant "${variant}" \
      2>&1 | tee "${SUITE}/logs/${city}_paper_ablation_${variant}_gpu${gpu}.log"
    [[ -s "${summary}" ]] || { echo "ERROR: summary missing after ${city} ${variant}" >&2; return 20; }
    patch_temporal_metadata "${city}" "${variant}"
  done
  echo "[PAPER ABLATION CITY DONE] ${city} GPU${gpu}"
}

# Keep three-city parallelism, then CityD, matching the resource-safe formal run.
run_city citya 0 & PIDA=$!
run_city cityb 5 & PIDB=$!
run_city cityc 6 & PIDC=$!
failed=0
wait "${PIDA}" || failed=1
wait "${PIDB}" || failed=1
wait "${PIDC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: citya/b/c paper ablation failed" >&2; exit 30; }
run_city cityd 0

python3 -u v39_otherdata/build_bearing_paper_ablation_tables.py --suite-root "${SUITE}"
python3 -u v39_otherdata/bearing_paper_metrics.py --suite-root "${SUITE}"

echo "================================================================================"
echo "PAPER EXPERIMENTS COMPLETE"
echo "Main tables : ${SUITE}/paper_benchmark/PAPER_TABLES.md"
echo "Ablations   : ${SUITE}/paper_ablation/ABLATION_TABLES.md"
echo "All CSVs    : ${SUITE}/paper_ablation/*.csv"
echo "================================================================================"

if [[ "${UPLOAD_ABLATIONS}" == "1" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  DEST="paper_results/formal_bearing_v5_paper_suite_${STAMP}"
  TMP="$(mktemp -d "${ROOT%/*}/uav-sat-paper-upload-XXXXXX")"
  BR="paper-suite-upload-${STAMP}-$$"
  cleanup(){
    git -C "${ROOT}" worktree remove --force "${TMP}" >/dev/null 2>&1 || true
    git -C "${ROOT}" branch -D "${BR}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin bearing-v5-formal-smooth-v1
  git worktree add -b "${BR}" "${TMP}" origin/bearing-v5-formal-smooth-v1
  mkdir -p "${TMP}/${DEST}"
  cp -r "${SUITE}/paper_benchmark" "${TMP}/${DEST}/"
  cp -r "${SUITE}/paper_ablation" "${TMP}/${DEST}/"
  for city in citya cityb cityc cityd; do
    for frames in 1 2 3; do
      src="${SUITE}/${city}/train_frames${frames}"
      [[ -d "${src}" ]] || continue
      dst="${TMP}/${DEST}/${city}/train_frames${frames}"
      mkdir -p "${dst}"
      cp "${src}/experiment_manifest.json" "${dst}/" 2>/dev/null || true
      cp "${src}/kalman_calibration.json" "${dst}/" 2>/dev/null || true
    done
    for variant in full "${VARIANTS[@]}"; do
      src="${SUITE}/${city}/variants/${variant}"
      [[ -s "${src}/bearing_v39_summary.json" ]] || continue
      dst="${TMP}/${DEST}/${city}/variants/${variant}"
      mkdir -p "${dst}"
      cp "${src}/bearing_v39_summary.json" "${dst}/"
      cp "${src}/experiment_manifest.json" "${dst}/" 2>/dev/null || true
    done
  done
  git -C "${TMP}" add "${DEST}"
  git -C "${TMP}" -c user.name='OpenAI Results Uploader' -c user.email='results@local' \
    commit -m "Upload Bearing V5 paper experiment suite ${STAMP}"
  git -C "${TMP}" push origin "HEAD:bearing-v5-formal-smooth-v1"
  echo "[PAPER SUITE UPLOAD DONE] ${DEST}"
fi
