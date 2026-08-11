#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
MODE="${MODE:-train_eval}"; GPU="${GPU:-0}"; VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"; TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-40}"; REUSE_VISUAL="${REUSE_VISUAL:-1}"; FORCE_FULL_RETRAIN="${FORCE_FULL_RETRAIN:-0}"
OUTPUT_DIR="outputs/reversible_topology_recovery_lstm_v10"; CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"; VISUAL_CKPT="${CHECKPOINT_DIR}/visual_retrieval_A_only.pt"; TEMPORAL_CKPT="${CHECKPOINT_DIR}/reversible_topology_recovery_lstm_A_only.pt"
ROUTE_B_CSV="${OUTPUT_DIR}/route_B_reversible_topology_frames.csv"; ROUTE_C_CSV="${OUTPUT_DIR}/route_C_reversible_topology_frames.csv"; SUMMARY_JSON="${OUTPUT_DIR}/robust_tracker_summary.json"; mkdir -p "${CHECKPOINT_DIR}"
case "${MODE}" in train|eval|train_eval) ;; *) echo "ERROR: MODE must be train, eval, or train_eval" >&2; exit 2;; esac
python3 - <<'PYCHK'
import config, robust_tracker
print("config OUTPUT_DIR:", config.OUTPUT_DIR); print("tracker architecture:", robust_tracker.ARCHITECTURE_NAME)
assert robust_tracker.ARCHITECTURE_NAME == "ReversibleTopologyRecoveryLSTM_v10"
for name in ("torch","filterpy","matplotlib","pandas","cv2"): __import__(name); print("import OK:", name)
PYCHK
find_visual(){ local candidates=("${VISUAL_CKPT}" "outputs/image_causal_forward_lstm_v9/checkpoints/visual_retrieval_A_only.pt" "outputs/route_tangent_forward_lstm_v8/checkpoints/visual_retrieval_A_only.pt" "outputs/heading_forward_lstm_v7/checkpoints/visual_retrieval_A_only.pt" "outputs/route_bounded_hypothesis_lstm_v6/checkpoints/visual_retrieval_A_only.pt" "outputs/strict_train_A_test_BC_t2only_w5/checkpoints/visual_retrieval_A_only.pt"); local candidate; for candidate in "${candidates[@]}"; do [[ -s "${candidate}" ]] && { printf '%s
' "${candidate}"; return 0; }; done; return 1; }
prepare_visual(){ [[ "${FORCE_FULL_RETRAIN}" == "1" || "${REUSE_VISUAL}" != "1" ]] && return 1; local found=""; if found="$(find_visual)"; then if [[ "${found}" != "${VISUAL_CKPT}" ]]; then cp -p "${found}" "${VISUAL_CKPT}"; echo "Reuse visual checkpoint: ${found} -> ${VISUAL_CKPT}"; else echo "Reuse v10 visual checkpoint: ${VISUAL_CKPT}"; fi; return 0; fi; return 1; }
run_tracker(){ local tracker_mode="$1"; shift || true; PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python3 -u robust_tracker.py --mode "${tracker_mode}" --visual-epochs "${VISUAL_EPOCHS}" --temporal-epochs "${TEMPORAL_EPOCHS}" "$@"; }
verify_eval(){ local failed=0; for path in "${TEMPORAL_CKPT}" "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"; do [[ -s "${path}" ]] && echo "OK: ${path}" || { echo "MISSING: ${path}" >&2; failed=1; }; done; [[ "${failed}" == "0" ]] || exit 30; }
fresh_train_eval(){ rm -f "${TEMPORAL_CKPT}" "${TEMPORAL_CKPT}.tmp" "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"; if prepare_visual; then run_tracker train_eval --reuse-visual; else rm -f "${VISUAL_CKPT}"; run_tracker train_eval; fi; verify_eval; }
case "${MODE}" in
 train) rm -f "${TEMPORAL_CKPT}" "${TEMPORAL_CKPT}.tmp"; if prepare_visual; then run_tracker train --reuse-visual; else rm -f "${VISUAL_CKPT}"; run_tracker train; fi; [[ -s "${TEMPORAL_CKPT}" ]] || { echo "ERROR: temporal checkpoint missing after train" >&2; exit 23; };;
 train_eval) fresh_train_eval; python3 -u render_results_video.py --route all;;
 eval) if [[ -s "${TEMPORAL_CKPT}" && -s "${VISUAL_CKPT}" ]]; then rm -f "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"; run_tracker eval; verify_eval; else echo "v10 checkpoint missing; training v10 before evaluation."; fresh_train_eval; fi; python3 -u render_results_video.py --route all;;
esac
echo "DONE: ${OUTPUT_DIR}"
