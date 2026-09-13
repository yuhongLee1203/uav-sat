#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
BACKBONE="${UAVSAT_BACKBONE:-mobilenet_v3_small}"
GPU="${UAVSAT_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-5}"
DEFAULT_MOTION="velocity"
DEFAULT_KALMAN="fixed"
DEFAULT_MS_GRID="${MS_GRID_SIZE:-5}"
DEFAULT_MS_BANDWIDTH="${MS_BANDWIDTH_M:-7.0}"
TRAIN_FROM_SCRATCH="${UAVSAT_TRAIN_FROM_SCRATCH:-0}"
RUN_MODE="${UAVSAT_RUN_MODE:-eval}"

FINAL_ARCH="V39_Forward3x6_ContextGRU_FixedKalman_FinalMS5x5"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
FIELDANCHOR_VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
FIELDANCHOR_TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/${CKPT_NAME}"

OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output_v39_current}"
RUNTIME="${UAVSAT_RUNTIME_DIR:-${ROOT}/runtime_v39_current}"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${OUT}/feature_cache}"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing ${ROOT}/patch_direct_finalms.py" >&2; exit 2; }
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || {
    echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2
    echo "UAVSAT_DATA_ROOT must point to the PREPARED dataset root containing routes/route_A, route_B, route_C." >&2
    exit 2
  }
done

# The old all-ablation suite is kept in its dedicated script. run.sh is now the
# canonical train/eval entry for the selected final model.
if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  exec bash "${ROOT}/run_corrected_ablation.sh"
fi

patch_context_gru() {
  local runtime="$1"
  python3 - "${runtime}" <<'PY'
from pathlib import Path
import sys
runtime = Path(sys.argv[1])

# Match the final formal model: temporal mean + first diff + second diff
# + posterior-weighted SAT context + previous state => 5 projected GRU blocks.
p = runtime / "visual_model.py"
s = p.read_text(encoding="utf-8")
old = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
new = "        self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)\n"
if s.count(old) != 1:
    raise SystemExit(f"ERROR: expected one 4-block GRU declaration, got {s.count(old)}")
s = s.replace(old, new, 1)
old_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
new_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
if s.count(old_block) != 1:
    raise SystemExit(f"ERROR: expected one 4-block recurrent input, got {s.count(old_block)}")
s = s.replace(old_block, new_block, 1)
p.write_text(s, encoding="utf-8")
compile(s, str(p), "exec")

# Match the corrected velocity-fusion behavior used by the final formal suite.
p = runtime / "robust_tracker.py"
s = p.read_text(encoding="utf-8")
old_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            step = velocity.copy()
'''
new_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            if bool(getattr(config, "EXPERIMENT_DISABLE_GRU", False)):
                velocity = self.x[2:4].copy()
                step = velocity.copy()
            else:
                step = velocity.copy()
'''
if s.count(old_motion) != 1:
    raise SystemExit(f"ERROR: expected one velocity block, got {s.count(old_motion)}")
s = s.replace(old_motion, new_motion, 1)
p.write_text(s, encoding="utf-8")
compile(s, str(p), "exec")
print("[PATCH] Context-GRU + corrected velocity fusion: PASS")
PY
}

make_runtime() {
  rm -rf "${RUNTIME}"
  mkdir -p "${RUNTIME}" "${OUT}/checkpoints" "${FEATURE_CACHE_DIR}"
  cp -a "${BASE_SRC}/." "${RUNTIME}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${RUNTIME}/robust_tracker.py"
  patch_context_gru "${RUNTIME}"
}

fresh_audit() {
  local visual_ckpt="${OUT}/checkpoints/visual_retrieval_A_only.pt"
  local temporal_ckpt="${OUT}/checkpoints/${CKPT_NAME}"
  python3 - "${visual_ckpt}" "${temporal_ckpt}" "${FINAL_ARCH}" <<'PY'
import sys, torch
from pathlib import Path
v = Path(sys.argv[1]); t = Path(sys.argv[2]); arch = sys.argv[3]
if not v.is_file() or v.is_symlink():
    raise SystemExit(f"AUDIT FAILED: visual checkpoint is not a freshly written regular file: {v}")
if not t.is_file() or t.is_symlink():
    raise SystemExit(f"AUDIT FAILED: temporal checkpoint is not a freshly written regular file: {t}")
vp = torch.load(v, map_location="cpu")
tp = torch.load(t, map_location="cpu")
if vp.get("visual_train_routes") != ["route_A"]:
    raise SystemExit(f"AUDIT FAILED: visual train routes={vp.get('visual_train_routes')}")
if vp.get("visual_validation_routes") != ["route_A"]:
    raise SystemExit(f"AUDIT FAILED: visual validation routes={vp.get('visual_validation_routes')}")
if vp.get("previous_task_checkpoint_loaded") is not False:
    raise SystemExit("AUDIT FAILED: visual checkpoint does not prove random task-head initialization")
if tp.get("architecture") != arch:
    raise SystemExit(f"AUDIT FAILED: temporal architecture={tp.get('architecture')!r} expected={arch!r}")
if tp.get("train_routes") != ["route_A"]:
    raise SystemExit(f"AUDIT FAILED: temporal train routes={tp.get('train_routes')}")
print("[AUDIT] fresh visual + fresh temporal checkpoints: PASS")
print(f"[AUDIT] architecture: {arch}")
PY
}

make_runtime

ARGS=(--mode "${RUN_MODE}" --jitter-m "${JITTER_M}")

if [[ "${TRAIN_FROM_SCRATCH}" == "1" ]]; then
  if [[ "${RUN_MODE}" != "train" && "${RUN_MODE}" != "train_eval" ]]; then
    echo "ERROR: UAVSAT_TRAIN_FROM_SCRATCH=1 requires UAVSAT_RUN_MODE=train or train_eval" >&2
    exit 3
  fi

  # Critical: do NOT link/reuse FieldAnchor task-specific checkpoints.
  # The public backbone may still use its normal pretrained weights; only the
  # UAV/SAT task heads and temporal model are trained from scratch on DATA_ROOT.
  rm -rf "${OUT}"
  mkdir -p "${OUT}/checkpoints" "${FEATURE_CACHE_DIR}"
  [[ ! -e "${OUT}/checkpoints/visual_retrieval_A_only.pt" ]] || {
    echo "ERROR: stale visual checkpoint survived output reset" >&2; exit 4;
  }
  [[ ! -e "${OUT}/checkpoints/${CKPT_NAME}" ]] || {
    echo "ERROR: stale temporal checkpoint survived output reset" >&2; exit 4;
  }
  ARGS+=(--visual-epochs "${VISUAL_EPOCHS}" --temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
else
  # Evaluation / legacy reuse path only. This is intentionally impossible in
  # scratch mode so an external dataset cannot silently inherit FieldAnchor heads.
  [[ -s "${FIELDANCHOR_VISUAL_CKPT}" ]] || { echo "ERROR: missing ${FIELDANCHOR_VISUAL_CKPT}" >&2; exit 5; }
  ln -sfn "${FIELDANCHOR_VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
  if [[ "${RUN_MODE}" == "eval" ]]; then
    CKPT_SOURCE="${UAVSAT_TEMPORAL_CKPT_SOURCE:-${FIELDANCHOR_TEMPORAL_CKPT}}"
    [[ -s "${CKPT_SOURCE}" ]] || { echo "ERROR: missing temporal checkpoint ${CKPT_SOURCE}" >&2; exit 5; }
    ln -sfn "${CKPT_SOURCE}" "${OUT}/checkpoints/${CKPT_NAME}"
  fi
  ARGS+=(--reuse-visual)
  if [[ "${RUN_MODE}" == "train" || "${RUN_MODE}" == "train_eval" ]]; then
    ARGS+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
  fi
fi

echo "============================================================================================================"
echo "V39 FINAL TRAIN/EVAL"
echo "data root      : ${DATA_ROOT}"
echo "output         : ${OUT}"
echo "gpu            : ${GPU}"
echo "backbone       : ${BACKBONE}"
echo "scratch        : ${TRAIN_FROM_SCRATCH}"
echo "mode           : ${RUN_MODE}"
echo "visual epochs  : ${VISUAL_EPOCHS}"
echo "temporal epochs: ${TEMPORAL_EPOCHS}"
echo "patience       : ${PATIENCE}"
echo "jitter         : ${JITTER_M} m"
echo "front search   : 6x6 geometry -> causal forward 3x6 scoring"
echo "GRU            : 3-frame Context GRU (mean, d1, d2, SAT context, previous state)"
echo "Kalman         : fixed-R external Kalman"
echo "Final MS       : ${DEFAULT_MS_GRID}x${DEFAULT_MS_GRID}, bandwidth=${DEFAULT_MS_BANDWIDTH} m"
echo "============================================================================================================"

(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${FINAL_ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
  UAVSAT_EXPERIMENT_KALMAN="${DEFAULT_KALMAN}" \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  MS_ENABLED=1 \
  MS_GRID_SIZE="${DEFAULT_MS_GRID}" \
  MS_BANDWIDTH_M="${DEFAULT_MS_BANDWIDTH}" \
  python3 -u robust_tracker.py "${ARGS[@]}" 2>&1 | tee "${OUT}/${RUN_MODE}.log"
)

if [[ "${TRAIN_FROM_SCRATCH}" == "1" ]]; then
  fresh_audit
fi

echo "[DONE] summary: ${OUT}/robust_tracker_summary.json"
