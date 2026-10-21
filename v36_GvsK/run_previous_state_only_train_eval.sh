#!/usr/bin/env bash
# V36 ablation built from CURRENT original_forNX:
# - remove satellite context from MAIN GRU
# - replace the old 8D numeric state with a 4D Previous State only:
#     previous velocity [v_s, v_e] + previous heading residual + previous turn rate
# - keep SoftMS, 3-frame temporal features, quadratic motion and learned Kalman unchanged.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/original_forNX"
SRC="${ROOT}/previous_state_only"
FORNX="${ROOT}/../forNX"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${ROOT}/v36_training_data}"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output/previous_state_only}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
BACKBONE="${UAVSAT_BACKBONE:-mobileclip2_s2}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${ROOT}/output/meanshift_gru/feature_cache}"
ARCH="V36_NoSatContext_PreviousStateOnly_Forward3x6_PolynomialKalman"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE}/${f}" ]] || { echo "ERROR: missing ${BASE}/${f}" >&2; exit 2; }
done
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

ORIG_VISUAL="${FORNX}/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
[[ -s "${ORIG_VISUAL}" ]] || { echo "ERROR: missing original forNX visual checkpoint: ${ORIG_VISUAL}" >&2; exit 2; }

# Always rebuild from current original_forNX so stale experiment edits cannot survive.
rm -rf "${SRC}"
mkdir -p "${SRC}"
cp -a "${BASE}/." "${SRC}/"

python3 - "${SRC}/config.py" "${SRC}/visual_model.py" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
model_path = Path(sys.argv[2])

# ------------------------------------------------------------------
# config.py: the MAIN GRU state block is now explicitly Previous State.
# ------------------------------------------------------------------
c = config_path.read_text(encoding="utf-8")
old_cfg = '''# Only simplify the main GRU input. Everything else remains the v34 protocol:\n# response variance(2) + visual innovation(2) + previous velocity(2)\n# + previous heading residual/turn-rate(2).\nRNN_NUMERIC_DIM = 8'''
new_cfg = '''# Main GRU Previous State only:\n# previous velocity [v_s, v_e] (2) + previous heading residual (1)\n# + previous turn rate (1). No response variance, visual innovation,\n# satellite context, or Kalman final position is fed to the main GRU.\nRNN_PREVIOUS_STATE_DIM = 4'''
if c.count(old_cfg) != 1:
    raise SystemExit(f"ERROR: expected one original RNN numeric-state config block, found {c.count(old_cfg)}")
c = c.replace(old_cfg, new_cfg, 1)
config_path.write_text(c, encoding="utf-8")

# ------------------------------------------------------------------
# visual_model.py
# ------------------------------------------------------------------
s = model_path.read_text(encoding="utf-8")

# Rename the main-GRU projection from numeric -> Previous State and reduce 8D -> 4D.
old_proj = '''        self.numeric_projection = nn.Sequential(\n            nn.Linear(int(config.RNN_NUMERIC_DIM), feature_dim),\n            nn.GELU(),\n            nn.Linear(feature_dim, feature_dim),\n            nn.GELU(),\n            nn.LayerNorm(feature_dim),\n        )'''
new_proj = '''        self.previous_state_projection = nn.Sequential(\n            nn.Linear(int(config.RNN_PREVIOUS_STATE_DIM), feature_dim),\n            nn.GELU(),\n            nn.Linear(feature_dim, feature_dim),\n            nn.GELU(),\n            nn.LayerNorm(feature_dim),\n        )'''
if s.count(old_proj) != 1:
    raise SystemExit(f"ERROR: expected one main numeric projection, found {s.count(old_proj)}")
s = s.replace(old_proj, new_proj, 1)

# Remove satellite context from MAIN GRU: 5 projected blocks -> 4 projected blocks.
old_gru = "self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)"
new_gru = "self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)"
if s.count(old_gru) != 1:
    raise SystemExit(f"ERROR: expected one original GRU declaration, found {s.count(old_gru)}")
s = s.replace(old_gru, new_gru, 1)

# Replace the old response-variance + innovation + previous-state 8D block
# with ONLY previous velocity + previous heading information (4D).
pattern = re.compile(
    r'''        if previous_measurement_se is None:\n            previous_measurement_se = visual_anchor_se\.detach\(\)\n\n        innovation_se = visual_anchor_se - predicted_se\n\n        numeric = torch\.cat\(\n            \[\n                torch\.log1p\(response_variance_se\.clamp_min\(0\.0\)\) / 7\.0,\n                torch\.cat\(\n                    \[\n                        innovation_se\[:, 0:1\] / float\(config\.ROUTE_STEP_SCALE_M\),\n                        innovation_se\[:, 1:2\] / float\(config\.ROUTE_CROSS_TRACK_SCALE_M\),\n                    \],\n                    dim=1,\n                \),\n                previous_velocity_se / float\(config\.ROUTE_STEP_SCALE_M\),\n                previous_heading_state\[:, 0:1\] / math\.radians\(float\(config\.MAX_HEADING_RESIDUAL_DEG\)\),\n                previous_heading_state\[:, 1:2\] / math\.radians\(float\(config\.MAX_TURN_RATE_DEG_PER_FRAME\)\),\n            \],\n            dim=1,\n        \)\n        if int\(numeric\.shape\[1\]\) != int\(config\.RNN_NUMERIC_DIM\):\n            raise RuntimeError\(\n                "RNN numeric dimension mismatch: got %d expected %d"\n                % \(int\(numeric\.shape\[1\]\), int\(config\.RNN_NUMERIC_DIM\)\)\n            \)\n''',
    re.MULTILINE,
)
replacement = '''        previous_state = torch.cat(\n            [\n                previous_velocity_se / float(config.ROUTE_STEP_SCALE_M),\n                previous_heading_state[:, 0:1] / math.radians(float(config.MAX_HEADING_RESIDUAL_DEG)),\n                previous_heading_state[:, 1:2] / math.radians(float(config.MAX_TURN_RATE_DEG_PER_FRAME)),\n            ],\n            dim=1,\n        )\n        if int(previous_state.shape[1]) != int(config.RNN_PREVIOUS_STATE_DIM):\n            raise RuntimeError(\n                "RNN Previous State dimension mismatch: got %d expected %d"\n                % (int(previous_state.shape[1]), int(config.RNN_PREVIOUS_STATE_DIM))\n            )\n'''
s, n = pattern.subn(replacement, s, count=1)
if n != 1:
    raise SystemExit(f"ERROR: expected one main GRU 8D numeric-state block, replaced {n}")

old_input = '''                self.delta_accel_projection(delta_accel),\n                self.sat_projection(sat_context),\n                self.numeric_projection(numeric),'''
new_input = '''                self.delta_accel_projection(delta_accel),\n                self.previous_state_projection(previous_state),'''
if s.count(old_input) != 1:
    raise SystemExit(f"ERROR: expected one original main-GRU input block, found {s.count(old_input)}")
s = s.replace(old_input, new_input, 1)

# Acquisition scorer still owns its own local SAT/numeric features; this experiment
# changes only the MAIN recurrent state estimator.
if "sat_h = self.sat_projection(sat_context)" not in s:
    raise SystemExit("ERROR: acquisition scorer satellite context was unexpectedly removed")
if "self.previous_state_projection(previous_state)" not in s:
    raise SystemExit("ERROR: Previous State projection patch failed")
if "innovation_se = visual_anchor_se - predicted_se" in s:
    raise SystemExit("ERROR: visual innovation still enters the main GRU")

model_path.write_text(s, encoding="utf-8")
PY

# Outside config.py + visual_model.py, source must remain identical to original forNX.
for f in data.py robust_tracker.py visual_localizer.py; do
  cmp -s "${BASE}/${f}" "${SRC}/${f}" || { echo "ERROR: unintended code difference in ${f}" >&2; exit 3; }
done

echo "=== V36 PREVIOUS-STATE-ONLY MAIN GRU ABLATION ==="
echo "base source       : ${BASE}"
echo "ablation source   : ${SRC}"
echo "output            : ${OUT}"
echo "device            : ${DEVICE}"
echo "backbone          : ${BACKBONE}"
echo "visual checkpoint : ${ORIG_VISUAL} (REUSED)"
echo "feature cache     : ${CACHE_DIR}"
echo "main GRU current input:"
echo "  - temporal mean          128D"
echo "  - first difference       128D"
echo "  - second difference      128D"
echo "  - Previous State         128D projection"
echo "Previous State raw = [previous v_s, previous v_e, previous heading residual, previous turn rate] = 4D"
echo "REMOVED from main GRU: satellite context, response variance, visual innovation"
echo "NO Kalman final position is fed to the main GRU in this ablation"
echo "PRESERVED: SoftMS, acquisition path, 3 frames, quadratic motion, learned Kalman, forward 3x6, 8m smooth jitter"

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

echo "[DONE] Previous-State-only result: ${OUT}/robust_tracker_summary.json"
