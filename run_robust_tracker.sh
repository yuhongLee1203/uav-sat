#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
MODE="${MODE:-train_eval}"; GPU="${GPU:-0}"; VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"; TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"; REUSE_VISUAL="${REUSE_VISUAL:-1}"; FORCE_FULL_RETRAIN="${FORCE_FULL_RETRAIN:-0}"
OUT="outputs/recurrent_visual_measurement_kalman_v15"; mkdir -p "$OUT/checkpoints"; V="$OUT/checkpoints/visual_retrieval_A_only.pt"; T="$OUT/checkpoints/recurrent_visual_measurement_A_only.pt"
find_visual(){ for f in "$V" outputs/stable_visual_inertial_rnn_v14/checkpoints/visual_retrieval_A_only.pt outputs/timestamp_velocity_visual_rnn_v13/checkpoints/visual_retrieval_A_only.pt outputs/direct_displacement_visual_rnn_v12/checkpoints/visual_retrieval_A_only.pt outputs/continuous_progress_visual_rnn_v11/checkpoints/visual_retrieval_A_only.pt; do [[ -s "$f" ]] && { echo "$f"; return 0; }; done; return 1; }
prepare_visual(){ [[ "$FORCE_FULL_RETRAIN" == "1" || "$REUSE_VISUAL" != "1" ]] && { rm -f "$V"; return 1; }; f="$(find_visual)" || return 1; [[ "$f" != "$V" ]] && cp -p "$f" "$V"; return 0; }
run(){ CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python3 -u robust_tracker.py --mode "$1" --visual-epochs "$VISUAL_EPOCHS" --temporal-epochs "$TEMPORAL_EPOCHS" ${2:-}; }
case "$MODE" in
 train) rm -f "$T"; prepare_visual && run train --reuse-visual || run train ;;
 train_eval) rm -f "$T"; prepare_visual && run train_eval --reuse-visual || run train_eval; python3 -u render_results_video.py --route all ;;
 eval) [[ -s "$V" ]] || prepare_visual; [[ -s "$T" ]] || { echo "missing v15 temporal checkpoint" >&2; exit 23; }; run eval; python3 -u render_results_video.py --route all ;;
 *) echo "MODE must be train/eval/train_eval" >&2; exit 2 ;;
esac
