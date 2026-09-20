#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_SUITE="${1:-}"
[[ -n "${BASE_SUITE}" && -d "${BASE_SUITE}" ]] || { echo "ERROR: pass completed Smooth-V1 suite" >&2; exit 2; }
BASE_SUITE="$(readlink -f "${BASE_SUITE}")"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SEEDS_STR="${TEMPORAL_SEEDS:-2033 2034 2035}"
THREADS="${CPU_THREADS_PER_CITY:-2}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-4}"
OUTROOT="${BASE_SUITE}/paper_temporal_multiseed_v3"
mkdir -p "${OUTROOT}" "${BASE_SUITE}/logs"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
export MALLOC_ARENA_MAX=2

# Keep the completed Smooth-v1 model-side profile. These are not selected on held-out data.
export UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME="${UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME:-12.0}"
export UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME="${UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME:-3.0}"
export UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2="${UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2:-2.0}"
export UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME="${UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME:-12.0}"
export UAVSAT_CORR_PARALLEL_M="${UAVSAT_CORR_PARALLEL_M:-0.70}"
export UAVSAT_CORR_CROSS_M="${UAVSAT_CORR_CROSS_M:-0.45}"
export UAVSAT_HEADING_STATE_EMA_ALPHA="${UAVSAT_HEADING_STATE_EMA_ALPHA:-0.22}"
export UAVSAT_TURN_RATE_EMA_ALPHA="${UAVSAT_TURN_RATE_EMA_ALPHA:-0.20}"
export UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME="${UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME:-3.0}"
export UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2="${UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2:-3.0}"
export UAVSAT_LOSS_CROSS_MOTION_REG="${UAVSAT_LOSS_CROSS_MOTION_REG:-0.03}"
export UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M="${UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M:-1.25}"
export UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME="${UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME:-0.90}"
export UAVSAT_KALMAN_FINAL_STEP_MAX_M="${UAVSAT_KALMAN_FINAL_STEP_MAX_M:-6.0}"
export UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M="${UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M:-36.0}"
export UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="${UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2:-25.0}"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/bearing_paper_ablation.py \
  v39_otherdata/build_bearing_paper_ablation_tables.py \
  v39_otherdata/build_bearing_temporal_multiseed.py

python3 - <<'PY'
from pathlib import Path
s=Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
checks={
 'formal_v5':'formal_v5_direct_delta2_residual_kalman' in s,
 'train_frames':'--train-frames' in s,
 'frame_ckpt':'checkpoint_frames = int(variant["frames"])' in s,
 'nav50':'("nav50", "route_B"' in s,
}
for k,v in checks.items(): print(f'[CORE-V3 PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()): raise SystemExit('ERROR: local bearing_iclr_ablation.py is not the generated Formal-V5 runner')
PY

# Clean paper-facing tables. Historical raw variants remain untouched for audit.
python3 -u v39_otherdata/build_bearing_paper_ablation_tables.py \
  --suite-root "${BASE_SUITE}" \
  --output-dir "${BASE_SUITE}/paper_core"

common_args(){
  local suite="$1" city="$2" gpu="$3" seed="$4"
  COMMON=(
    --suite-root "${suite}"
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
    --seed "${seed}"
  )
}

prepare_seed_city(){
  local seedroot="$1" city="$2" frames srcvis dstvis
  mkdir -p "${seedroot}/${city}"
  ln -sfn "${BASE_SUITE}/${city}/prepared" "${seedroot}/${city}/prepared"
  if [[ -d "${BASE_SUITE}/${city}/feature_cache" ]]; then
    ln -sfn "${BASE_SUITE}/${city}/feature_cache" "${seedroot}/${city}/feature_cache"
  fi
  srcvis="${BASE_SUITE}/${city}/train_frames3/checkpoints/visual_retrieval_A_only.pt"
  [[ -e "${srcvis}" ]] || { echo "ERROR: missing visual checkpoint ${srcvis}" >&2; return 4; }
  for frames in 1 2 3; do
    mkdir -p "${seedroot}/${city}/train_frames${frames}/checkpoints"
    dstvis="${seedroot}/${city}/train_frames${frames}/checkpoints/visual_retrieval_A_only.pt"
    [[ -e "${dstvis}" ]] || ln -s "$(readlink -f "${srcvis}")" "${dstvis}"
  done
}

run_seed_city(){
  local seed="$1" city="$2" gpu="$3" seedroot frames ck variant summary
  seedroot="${OUTROOT}/seed_${seed}"
  prepare_seed_city "${seedroot}" "${city}"

  # Train all frame-count models under the same seed/split. No held-out result is used for selection.
  for frames in 1 2 3; do
    ck="${seedroot}/${city}/train_frames${frames}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
    if [[ -s "${ck}" ]]; then
      echo "[TEMPORAL MULTISEED TRAIN SKIP] seed=${seed} city=${city} frames=${frames}"
    else
      common_args "${seedroot}" "${city}" "${gpu}" "${seed}"
      echo "================================================================================"
      echo "[TEMPORAL MULTISEED TRAIN] seed=${seed} city=${city} frames=${frames} GPU${gpu}"
      echo "================================================================================"
      python3 -u v39_otherdata/bearing_iclr_ablation.py train \
        "${COMMON[@]}" --train-frames "${frames}" \
        2>&1 | tee "${BASE_SUITE}/logs/temporal_seed${seed}_${city}_${frames}f_train.log"
      [[ -s "${ck}" ]] || { echo "ERROR: checkpoint missing: ${ck}" >&2; return 10; }
    fi
  done

  for variant in frames1 frames2 full; do
    summary="${seedroot}/${city}/variants/${variant}/bearing_v39_summary.json"
    if [[ -s "${summary}" ]]; then
      echo "[TEMPORAL MULTISEED EVAL SKIP] seed=${seed} city=${city} variant=${variant}"
      continue
    fi
    common_args "${seedroot}" "${city}" "${gpu}" "${seed}"
    echo "[TEMPORAL MULTISEED EVAL] seed=${seed} city=${city} variant=${variant} GPU${gpu}"
    python3 -u v39_otherdata/bearing_paper_ablation.py \
      "${COMMON[@]}" --variant "${variant}" \
      2>&1 | tee "${BASE_SUITE}/logs/temporal_seed${seed}_${city}_${variant}_eval.log"
    [[ -s "${summary}" ]] || { echo "ERROR: summary missing: ${summary}" >&2; return 11; }
  done
}

for seed in ${SEEDS_STR}; do
  echo "################################################################################"
  echo "TEMPORAL MULTI-SEED ${seed}"
  echo "################################################################################"
  run_seed_city "${seed}" citya 0 & PA=$!
  run_seed_city "${seed}" cityb 5 & PB=$!
  run_seed_city "${seed}" cityc 6 & PC=$!
  failed=0
  wait "${PA}" || failed=1
  wait "${PB}" || failed=1
  wait "${PC}" || failed=1
  [[ "${failed}" -eq 0 ]] || { echo "ERROR: seed ${seed} citya/b/c failed" >&2; exit 20; }
  run_seed_city "${seed}" cityd 0
done

python3 -u v39_otherdata/build_bearing_temporal_multiseed.py --root "${OUTROOT}"

echo "================================================================================"
echo "PAPER CORE V3 COMPLETE"
echo "Core tables     : ${BASE_SUITE}/paper_core/PAPER_CORE_TABLES.md"
echo "Temporal seeds  : ${OUTROOT}/aggregate/TEMPORAL_MULTI_SEED.md"
echo "IMPORTANT       : 3-frame is only claimed best if the multi-seed result supports it."
echo "Protocol        : Full remains controlled_gt_jitter; this run does not make it GT-free."
echo "================================================================================"
