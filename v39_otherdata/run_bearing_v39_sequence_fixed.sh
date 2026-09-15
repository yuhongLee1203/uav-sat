#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"
USED_SAMPLE_DISTANCE="10"

cd "${REPO_ROOT}"

prepare_sequence() {
  local safety_cap="$1"
  local sample_distance="$2"
  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] piecewise route = long straight -> BIG turn -> long straight; safety=${safety_cap}m sample_radius=${sample_distance}m"

  if python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" "${safety_cap}" "${sample_distance}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# IMPORTANT: explicit piecewise-linear navigation routes. Every two waypoints
# define one long straight leg. Intermediate waypoints are deliberate large
# corners, not gradual small bends.
seq.PIECEWISE_ROUTE_SPECS = {
    "train_01": [
        (330, 620), (1300, 620), (1300, 1100),
        (2400, 1100), (2400, 1550), (3330, 1550),
    ],
    "train_02": [
        (430, 3080), (1400, 3080), (1400, 2600),
        (2600, 2600), (2600, 3100), (3510, 3100),
    ],
    "train_03": [
        (3330, 430), (3330, 1300), (3000, 1300),
        (3000, 2400), (3550, 2400), (3550, 3610),
    ],
    "test_01": [
        (560, 1810), (1500, 1810), (1500, 1450),
        (2600, 1450), (2600, 2050), (3410, 2050),
    ],
    "test_02": [
        (900, 330), (900, 1150), (1300, 1150),
        (1300, 2300), (900, 2300), (900, 3300), (1440, 3690),
    ],
}
seq.SELECTION_VERSION = "soft_sequence_v6_piecewise_straight_big_turns"

args = argparse.Namespace(
    dataset_root=sys.argv[1],
    city=sys.argv[2],
    output_root=sys.argv[3],
    step_m=4.0,
    max_sample_distance_m=float(sys.argv[5]),
    preferred_step_m=4.0,
    safety_max_step_m=float(sys.argv[4]),
    candidate_limit=128,
    beam_width=256,
    skip_penalty=36.0,
    continuity_weight=2.0,
    large_step_weight=0.75,
    cross_weight=1.25,
    backward_weight=10.0,
    point_cross_weight=8.0,
    lateral_smooth_weight=12.0,
    big_turn_threshold_deg=45.0,
    min_selected_ratio=0.75,
)
seq.prepare(args)
PY
  then
    USED_SAMPLE_DISTANCE="${sample_distance}"
    return 0
  fi
  return 1
}

# First try a tighter 8 m observation radius so the real GT samples stay closer
# to the long straight centreline. Only relax to 10 m if Bearing is too sparse.
if ! prepare_sequence 14 8; then
  echo "[PREP] 14m step cap / 8m sample radius too sparse; retrying 18m / 8m"
  if ! prepare_sequence 18 8; then
    echo "[PREP] 18m / 8m too sparse; retrying 22m / 8m"
    if ! prepare_sequence 22 8; then
      echo "[PREP] 8m observation radius too sparse; final fallback 22m / 10m"
      prepare_sequence 22 10
    fi
  fi
fi

python3 - "${PREPARED_ROOT}/experiment.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
print("[PIECEWISE-STRAIGHT] route audit")
failed = []
for name, s in d["route_stats"].items():
    mean = float(s["actual_step_mean_m"])
    p90 = float(s["actual_step_p90_m"])
    ratio = float(s["selected_ratio"])
    back = float(s["backward_step_pct"])
    center_p90 = float(s.get("centerline_cross_p90_m", 0.0))
    wobble_p90 = float(s.get("same_leg_lateral_delta_p90_m", 0.0))
    print(
        f"  {name}: waypoints={s['waypoints']} frames={s['frames']} selected={ratio*100:.1f}% "
        f"step_mean={mean:.3f}m step_p90={p90:.3f}m "
        f"centerline_p90={center_p90:.3f}m same_leg_wobble_p90={wobble_p90:.3f}m "
        f"backward={back:.2f}%"
    )
    if ratio < 0.75:
        failed.append(f"{name}: selected_ratio={ratio:.3f}")
if failed:
    raise SystemExit("[PIECEWISE-STRAIGHT] audit failed: " + "; ".join(failed))
print("[PIECEWISE-STRAIGHT] route audit: PASS")
PY

# IMPORTANT: the localization model itself remains exact canonical v39.
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
  --max-sample-distance-m "${USED_SAMPLE_DISTANCE}" \
  --heading-weight-px-per-deg 0

python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing exact-v39 piecewise-straight experiment finished"
echo "Route policy: LONG STRAIGHT -> explicit BIG TURN -> LONG STRAIGHT"
echo "GT sample radius used: ${USED_SAMPLE_DISTANCE} m"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Route preview: ${PREPARED_ROOT}/route_plan_full_satellite.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
