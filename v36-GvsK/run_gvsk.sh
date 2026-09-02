#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

GPU="${GPU:-0}"
CPU_THREADS="${CPU_THREADS:-2}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
FROM_SCRATCH="${FROM_SCRATCH:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

if [[ "$FROM_SCRATCH" == "1" ]]; then
  echo "[GvsK] removing v36-GvsK/output so the original V36 is retrained from scratch"
  rm -rf output
fi
mkdir -p output/checkpoints

{
  echo "[GvsK] GPU=$GPU"
  echo "[GvsK] original snapshot=e732045cacc6d2bff152663e8b5966ee1b49b98b"
  echo "[GvsK] visual_epochs=$VISUAL_EPOCHS temporal_epochs=$TEMPORAL_EPOCHS jitter_m=$JITTER_M"

  if [[ "$FROM_SCRATCH" == "1" || ! -f output/checkpoints/controlled_referenceprior_forward3x6_teacher_feedback_state_gru_A_only.pt ]]; then
    python3 -u robust_tracker.py \
      --mode train \
      --visual-epochs "$VISUAL_EPOCHS" \
      --temporal-epochs "$TEMPORAL_EPOCHS" \
      --patience "$PATIENCE" \
      --jitter-m "$JITTER_M"
  else
    echo "[GvsK] existing temporal checkpoint found; skipping training"
  fi

  python3 -u compare_g_vs_k.py --jitter-m "$JITTER_M"
} 2>&1 | tee output/run_gvsk.log

echo "[GvsK] done"
echo "[GvsK] summary: $HERE/output/GvsK_summary.json"
echo "[GvsK] table  : $HERE/output/GvsK_comparison_table.csv"
echo "[GvsK] frames : $HERE/output/route_B_GvsK_frames.csv"
echo "[GvsK] frames : $HERE/output/route_C_GvsK_frames.csv"
