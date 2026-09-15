#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"

cd "${REPO_ROOT}"

prepare_sequence() {
  local safety_cap="$1"
  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] Bearing data adapter: preserve BIG turns, smooth only same-leg GT wobble; safety cap ${safety_cap} m"
  python3 v39_otherdata/bearing_prepare_sequence_v3.py \
    --dataset-root "${DATASET_ROOT}" \
    --city "${CITY}" \
    --step-m 4 \
    --max-sample-distance-m 10 \
    --preferred-step-m 4 \
    --safety-max-step-m "${safety_cap}" \
    --candidate-limit 128 \
    --beam-width 256 \
    --skip-penalty 36 \
    --continuity-weight 2.0 \
    --large-step-weight 0.75 \
    --cross-weight 1.25 \
    --backward-weight 10.0 \
    --point-cross-weight 3.0 \
    --lateral-smooth-weight 3.0 \
    --big-turn-threshold-deg 45 \
    --min-selected-ratio 0.75
}

# DATA ADAPTER ONLY. The original large planned turns are restored.  The added
# cost applies only inside a route leg (planned heading change <45 deg), so real
# large corners are not flattened. The GT coordinates remain the true selected
# Bearing observation coordinates; no coordinate projection or fake smoothing.
if ! prepare_sequence 14; then
  echo "[PREP] 14 m data safety cap too sparse; retrying 18 m"
  if ! prepare_sequence 18; then
    echo "[PREP] 18 m data safety cap too sparse; final retry 22 m"
    prepare_sequence 22
  fi
fi

# Audit density and the two new small-wobble diagnostics before model training.
python3 - "${PREPARED_ROOT}/experiment.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
print("[BIG-TURN/SMOOTH-LEG] route audit")
failed = []
for name, s in d["route_stats"].items():
    mean = float(s["actual_step_mean_m"])
    p90 = float(s["actual_step_p90_m"])
    ratio = float(s["selected_ratio"])
    back = float(s["backward_step_pct"])
    center_p90 = float(s.get("centerline_cross_p90_m", 0.0))
    wobble_p90 = float(s.get("same_leg_lateral_delta_p90_m", 0.0))
    print(
        f"  {name}: frames={s['frames']} selected={ratio*100:.1f}% "
        f"step_mean={mean:.3f}m step_p90={p90:.3f}m "
        f"centerline_p90={center_p90:.3f}m same_leg_wobble_p90={wobble_p90:.3f}m "
        f"backward={back:.2f}%"
    )
    if ratio < 0.75:
        failed.append(f"{name}: selected_ratio={ratio:.3f}")
    if back > 0.5:
        print(f"  [WARN] {name}: backward_step_pct={back:.3f}% (diagnostic only; continuing)")
if failed:
    raise SystemExit("[BIG-TURN/SMOOTH-LEG] audit failed: " + "; ".join(failed))
print("[BIG-TURN/SMOOTH-LEG] route audit: PASS")
PY

# IMPORTANT: exact canonical-v39 estimator. No Bearing cadence adaptation.
python3 v39_otherdata/bearing_runner_exact_v39.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --gpu "${GPU}" \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 20 \
  --patience 10 \
  --jitter-m 8 \
  --step-m 4 \
  --max-sample-distance-m 10 \
  --heading-weight-px-per-deg 0

# Visualization uses true metric coordinates. Green = GT/reference, red = final.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing exact-v39 BIG-turn / smooth-leg experiment finished"
echo "Route policy: original BIG turns preserved; only same-leg sample wobble is penalized"
echo "Data cadence target: 4.0 m/frame"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Canonical Kalman final-step cap: 7.0 m"
echo "Route preview: ${PREPARED_ROOT}/route_plan_full_satellite.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Audit  : ${OUTPUT_DIR}/v39_bearing_training_audit.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
