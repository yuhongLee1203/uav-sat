#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SUITE_ROOT="${ABCD_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SEED="${SEED:-2033}"
GPUS_STR="${EVAL_GPUS:-0 5 6}"
read -r -a GPUS <<< "${GPUS_STR}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_kalman)

# Important protocol correction:
# The final MeanShift implementation can include a frame-reference prior derived
# from cache.gt_xy[index].  That is ground-truth-derived information at held-out
# inference time and also overwhelms the Kalman contribution.  For the paper
# evaluation we disable ONLY that reference term for every variant.
# The architecture remains:
# Forward-18 SoftMS -> GRU -> Kalman (or ablated) -> final MeanShift -> XY.
export MS_REFERENCE_PRIOR_WEIGHT="${MS_REFERENCE_PRIOR_WEIGHT:-0.0}"
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}"
export MS_REFERENCE_SIGMA_M="${MS_REFERENCE_SIGMA_M:-4.0}"
export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}"

if [[ ! -d "${SUITE_ROOT}" ]]; then
  echo "ERROR: suite not found: ${SUITE_ROOT}" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "ERROR: EVAL_GPUS is empty" >&2
  exit 2
fi

for city in "${CITIES[@]}"; do
  ckpt="${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  if [[ ! -s "${ckpt}" ]]; then
    echo "ERROR: missing shared frames=3 checkpoint: ${ckpt}" >&2
    exit 3
  fi
done

# Keep the previous measured pair for audit instead of deleting unfavorable data.
BACKUP_ROOT="${SUITE_ROOT}/audit_before_no_gt_reference"
mkdir -p "${BACKUP_ROOT}"
for city in "${CITIES[@]}"; do
  mkdir -p "${BACKUP_ROOT}/${city}"
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE_ROOT}/${city}/variants/${variant}"
    dst="${BACKUP_ROOT}/${city}/${variant}"
    if [[ -d "${src}" && ! -e "${dst}" ]]; then
      cp -a "${src}" "${dst}"
    fi
  done
done

run_one() {
  local city="$1"
  local variant="$2"
  local gpu="$3"
  local out="${SUITE_ROOT}/${city}/variants/${variant}"
  rm -rf "${out}"
  echo "[EVAL] city=${city} variant=${variant} gpu=${gpu} REF_W=${MS_REFERENCE_PRIOR_WEIGHT} KF_W=${MS_KF_PRIOR_WEIGHT}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
    --suite-root "${SUITE_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --variant "${variant}" \
    --train-frames 3 \
    --gpu 0 \
    --seed "${SEED}"
}

pids=()
labels=()
job=0
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
    run_one "${city}" "${variant}" "${gpu}" &
    pids+=("$!")
    labels+=("${city}/${variant}/gpu${gpu}")
    job=$((job + 1))
    if (( ${#pids[@]} >= ${#GPUS[@]} )); then
      for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
          echo "ERROR: evaluation failed: ${labels[$i]}" >&2
          exit 4
        fi
      done
      pids=()
      labels=()
    fi
  done
done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: evaluation failed: ${labels[$i]}" >&2
    exit 4
  fi
done

python3 v39_otherdata/build_kalman_pair_table.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd

cat > "${SUITE_ROOT}/no_gt_reference_protocol.json" <<EOF
{
  "protocol": "held-out final MeanShift without GT-derived frame-reference prior",
  "MS_REFERENCE_PRIOR_WEIGHT": ${MS_REFERENCE_PRIOR_WEIGHT},
  "MS_KF_PRIOR_WEIGHT": ${MS_KF_PRIOR_WEIGHT},
  "MS_REFERENCE_SIGMA_M": ${MS_REFERENCE_SIGMA_M},
  "MS_KF_SIGMA_M": ${MS_KF_SIGMA_M},
  "seed": ${SEED},
  "architecture_changed": false,
  "table_schema_changed": false
}
EOF

echo "[DONE] fair Full vs w/o Kalman rerun with GT-derived final reference prior disabled"
echo "[PAIR] ${SUITE_ROOT}/kalman_pair_table.md"
echo "[JSON] ${SUITE_ROOT}/kalman_pair_results.json"
echo "[AUDIT] ${BACKUP_ROOT}"
