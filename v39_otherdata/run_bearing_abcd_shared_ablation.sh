#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export OMP_NUM_THREADS="${CPU_THREADS:-2}"
export MKL_NUM_THREADS="${CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS:-2}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${ABCD_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_${STAMP}}"
GPU="${TRAIN_GPU:-0}"
ROUNDS="${ABCD_ROUNDS:-2}"
VISUAL_PER_CITY="${VISUAL_EPOCHS_PER_CITY:-5}"
TEMPORAL_PER_CITY="${TEMPORAL_EPOCHS_PER_CITY:-12}"
PATIENCE="${PATIENCE:-20}"
CITIES=(citya cityb cityc cityd)
FRAMES=(1 2 3)
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)

mkdir -p "${SUITE_ROOT}/logs" "${SUITE_ROOT}/shared"
printf '%s\n' "ABCD shared episodic training" \
  "One model is updated by A/B/C/D route episodes; recurrent/filter state resets at every episode." \
  "suite=${SUITE_ROOT}" | tee "${SUITE_ROOT}/PROTOCOL.txt"

# Idempotently align the checked-in driver before generating any runtime.
python3 v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_iclr_ablation.py

python3 -m py_compile v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/bearing_prepare_multicity.py v39_otherdata/build_iclr_ablation_tables.py

for city in "${CITIES[@]}"; do
  prepared="${SUITE_ROOT}/${city}/prepared"
  if [[ ! -s "${prepared}/experiment.json" ]]; then
    python3 -u v39_otherdata/bearing_prepare_multicity.py \
      --dataset-root "${DATASET_ROOT}" --city "${city}" --output-root "${prepared}" \
      2>&1 | tee "${SUITE_ROOT}/logs/${city}_prepare.log"
  fi
  python3 -u v39_otherdata/audit_bearing_training_contract.py \
    --dataset-root "${DATASET_ROOT}" --prepared-root "${prepared}" --city "${city}"
done

common_args() {
  local city="$1" frames="$2" visual_epochs="$3" temporal_epochs="$4"
  COMMON=(--suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}"
    --gpu "${GPU}" --backbone mobilenet_v3_small --train-frames "${frames}"
    --visual-epochs "${visual_epochs}" --temporal-epochs "${temporal_epochs}"
    --epochs-per-route "${temporal_epochs}" --patience "${PATIENCE}" --jitter-m 8
    --max-sample-distance-m 15 --heading-weight-px-per-deg 0 --ms-bandwidth-m 7
    --seed "${SEED:-2033}" --no-reuse-visual)
}

# Each frame setting gets exactly one shared model. A city is an episode, not
# an independent model. Checkpoints travel A->B->C->D and across rounds.
for frames in "${FRAMES[@]}"; do
  shared="${SUITE_ROOT}/shared/frames${frames}"
  mkdir -p "${shared}"
  latest_name="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"
  completed=0
  if [[ -s "${shared}/${latest_name}" && -s "${shared}/visual_retrieval_A_only.pt" ]]; then
    # The shared visual checkpoint is copied only after the whole city episode
    # (visual + temporal) succeeds. Its epoch therefore remains an exact
    # completion marker even when temporal early stopping ends at e.g. 93/96.
    completed="$(python3 - "${shared}/visual_retrieval_A_only.pt" "${VISUAL_PER_CITY}" <<'PY'
import sys, torch
x=torch.load(sys.argv[1], map_location='cpu')
print(int(x.get('epoch', 0)) // int(sys.argv[2]))
PY
)"
    echo "[RESUME] frames=${frames} completed_city_episodes=${completed}"
  fi
  episode=0
  for ((round=1; round<=ROUNDS; round++)); do
    for city in "${CITIES[@]}"; do
      episode=$((episode + 1))
      target_visual=$((episode * VISUAL_PER_CITY))
      target_temporal=$((episode * TEMPORAL_PER_CITY))
      if (( episode <= completed )); then
        echo "[RESUME SKIP] frames=${frames} episode=${episode} ${city} already complete"
        continue
      fi
      dest="${SUITE_ROOT}/${city}/train_frames${frames}/checkpoints"
      mkdir -p "${dest}"
      if [[ -s "${shared}/visual_retrieval_A_only.pt" ]]; then
        cp -f "${shared}/visual_retrieval_A_only.pt" "${dest}/visual_retrieval_A_only.pt"
        cp -f "${shared}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt" \
          "${dest}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"
        rm -f "${dest}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
        continuation=(--continue-train)
      else
        continuation=()
      fi
      common_args "${city}" "${frames}" "${target_visual}" "${target_temporal}"
      python3 -u v39_otherdata/bearing_iclr_ablation.py train "${COMMON[@]}" "${continuation[@]}" \
        2>&1 | tee "${SUITE_ROOT}/logs/shared_f${frames}_r${round}_${city}.log"

      # Promote the latest (current shared weights), rather than a best score
      # selected on only one city's validation split.
      python3 - "${dest}" <<'PY'
import sys, torch
from pathlib import Path
d=Path(sys.argv[1]); latest=torch.load(d/'controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt', map_location='cpu')
torch.save({'architecture': latest['architecture'], 'model': latest['model'],
            'training_protocol': 'ABCD_shared_city_episodic'}, d/'controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt')
PY
      cp -f "${dest}/visual_retrieval_A_only.pt" "${shared}/visual_retrieval_A_only.pt"
      cp -f "${dest}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt" "${shared}/"
      cp -f "${dest}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" "${shared}/"
    done
  done

  # Bind the same learned weights to each city's non-learned satellite gallery.
  for city in "${CITIES[@]}"; do
    dest="${SUITE_ROOT}/${city}/train_frames${frames}/checkpoints"
    mkdir -p "${dest}"
    # Temporal weights are cheap to refresh. Preserve an already-correct city
    # gallery: rebuilding 47,961 satellite features takes several minutes and
    # is unnecessary on a result-only resume.
    cp -f "${shared}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only"*.pt "${dest}/"
    gallery_ready="$(python3 - "${dest}/visual_retrieval_A_only.pt" "${city}" <<'PY'
import sys, torch
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    print(0)
else:
    x=torch.load(p, map_location='cpu')
    print(int(x.get('gallery_city') == sys.argv[2] and x.get('shared_abcd_weights') is True))
PY
)"
    if [[ "${gallery_ready}" == "1" ]]; then
      echo "[REBIND SKIP] frames=${frames} ${city} gallery already matches shared weights"
      continue
    fi
    cp -f "${shared}/visual_retrieval_A_only.pt" "${dest}/visual_retrieval_A_only.pt"
    common_args "${city}" "${frames}" "$((ROUNDS*4*VISUAL_PER_CITY))" "$((ROUNDS*4*TEMPORAL_PER_CITY))"
    python3 -u v39_otherdata/bearing_iclr_ablation.py rebind "${COMMON[@]}"
  done
done

# Evaluation is read-only and may use the three available GPUs concurrently.
gpu_list=(0 5 6); jobs=()
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    summary="${SUITE_ROOT}/${city}/variants/${variant}/bearing_v39_summary.json"
    if [[ -s "${summary}" ]]; then
      echo "[EVAL SKIP] ${city} ${variant} already complete"
      continue
    fi
    case "${variant}" in frames1) frames=1;; frames2) frames=2;; *) frames=3;; esac
    gpu="${gpu_list[${#jobs[@]}]}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}" \
      --variant "${variant}" --train-frames "${frames}" --gpu "${gpu}" --seed "${SEED:-2033}" \
      >"${SUITE_ROOT}/logs/eval_${city}_${variant}.log" 2>&1 &
    jobs+=("$!")
    if (( ${#jobs[@]} == 3 )); then
      for pid in "${jobs[@]}"; do wait "${pid}"; done
      jobs=()
    fi
  done
done
for pid in "${jobs[@]}"; do wait "${pid}"; done

missing=0
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    summary="${SUITE_ROOT}/${city}/variants/${variant}/bearing_v39_summary.json"
    if [[ ! -s "${summary}" ]]; then
      echo "ERROR: evaluation output missing: ${summary}" >&2
      missing=1
    fi
  done
done
(( missing == 0 )) || exit 31

python3 v39_otherdata/build_iclr_ablation_tables.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd | tee "${SUITE_ROOT}/logs/table.log"
printf '%s\n' "[DONE] ${SUITE_ROOT}/paper_ablation_tables.md" \
  "[RAW]  ${SUITE_ROOT}/paper_ablation_results.json"
