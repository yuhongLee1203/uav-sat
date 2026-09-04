#!/usr/bin/env bash
# V36 ablation: remove ONLY satellite context from the MAIN GRU input.
# Acquisition scorer satellite context is intentionally preserved.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/original_forNX"
SRC="${ROOT}/no_sat_context"
FORNX="${ROOT}/../forNX"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${ROOT}/v36_training_data}"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output/no_sat_context}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
BACKBONE="${UAVSAT_BACKBONE:-mobileclip2_s2}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${ROOT}/output/meanshift_gru/feature_cache}"
ARCH="V36_NoSatelliteContext_MainGRU_Forward3x6_PolynomialKalman"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE}/${f}" ]] || { echo "ERROR: missing ${BASE}/${f}" >&2; exit 2; }
done
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

ORIG_VISUAL="${FORNX}/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
[[ -s "${ORIG_VISUAL}" ]] || { echo "ERROR: missing original forNX visual checkpoint: ${ORIG_VISUAL}" >&2; exit 2; }

# Always rebuild the ablation source from the CURRENT original_forNX copy so no
# stale experimental edits can survive between runs.
rm -rf "${SRC}"
mkdir -p "${SRC}"
cp -a "${BASE}/." "${SRC}/"

# Change exactly two things in visual_model.py:
#   1) main GRU input blocks: 5 -> 4 (640 -> 512 for feature_dim=128)
#   2) remove sat_projection(sat_context) ONLY from recurrent_input
# score_hypotheses() keeps satellite context unchanged.
python3 - "${SRC}/visual_model.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old_gru = "self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)"
new_gru = "self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)"
if s.count(old_gru) != 1:
    raise SystemExit(f"ERROR: expected exactly one original GRU declaration, found {s.count(old_gru)}")
s = s.replace(old_gru, new_gru, 1)

old_input = """                self.delta_accel_projection(delta_accel),\n                self.sat_projection(sat_context),\n                self.numeric_projection(numeric),"""
new_input = """                self.delta_accel_projection(delta_accel),\n                self.numeric_projection(numeric),"""
if s.count(old_input) != 1:
    raise SystemExit(f"ERROR: expected exactly one main-GRU satellite-context input, found {s.count(old_input)}")
s = s.replace(old_input, new_input, 1)

# The acquisition scorer MUST still use satellite context. This keeps the
# ablation isolated to the main recurrent state estimator.
if "sat_h = self.sat_projection(sat_context)" not in s:
    raise SystemExit("ERROR: acquisition scorer satellite context was unexpectedly removed")
if new_gru not in s:
    raise SystemExit("ERROR: GRU input dimension patch failed")
p.write_text(s, encoding="utf-8")
PY

# Audit that every source file except visual_model.py is byte-identical.
for f in config.py data.py robust_tracker.py visual_localizer.py; do
  cmp -s "${BASE}/${f}" "${SRC}/${f}" || { echo "ERROR: unintended code difference in ${f}" >&2; exit 3; }
done

echo "=== V36 SATELLITE-CONTEXT ABLATION ==="
echo "base source       : ${BASE}"
echo "ablation source   : ${SRC}"
echo "output            : ${OUT}"
echo "device            : ${DEVICE}"
echo "backbone          : ${BACKBONE}"
echo "visual checkpoint : ${ORIG_VISUAL} (REUSED; visual model is NOT retrained)"
echo "feature cache     : ${CACHE_DIR}"
echo "main GRU input    : 5 x 128 = 640 -> 4 x 128 = 512"
echo "removed           : sat_projection(sat_context) from main GRU only"
echo "preserved         : acquisition scorer satellite context, SoftMS, 3 frames, quadratic motion, learned Kalman, forward 3x6, 8m smooth jitter"

rm -rf "${OUT}"
mkdir -p "${OUT}/checkpoints" "${CACHE_DIR}"
ln -sfn "${ORIG_VISUAL}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

export TORCH_HOME="${ROOT}/pretrained_cache/torch"
export HF_HOME="${ROOT}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=quadratic \
  UAVSAT_EXPERIMENT_KALMAN=learned \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  python3 -u robust_tracker.py \
    --mode train_eval \
    --reuse-visual \
    --temporal-epochs "${TEMPORAL_EPOCHS}" \
    --patience "${PATIENCE}" \
    --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/train_eval.log"

echo "[DONE] no-satellite-context result: ${OUT}/robust_tracker_summary.json"
