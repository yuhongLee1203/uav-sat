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
  echo "[PREP] route = STRAIGHT -> BIG TURN -> STRAIGHT -> BIG TURN -> STRAIGHT; safety=${safety_cap}m sample_radius=${sample_distance}m"

  if python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" "${safety_cap}" "${sample_distance}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# EXACTLY three straight legs and two major corners per route.
# There are no intermediate bend waypoints.  The selected Bearing observations
# may sit a few metres to either side of a leg because they are independent real
# samples, but those samples are not interpreted as extra route turns.
seq.PIECEWISE_ROUTE_SPECS = {
    "train_01": [
        (330, 620),
        (2800, 620),
        (2800, 1500),
        (3330, 1500),
    ],
    "train_02": [
        (430, 3080),
        (3100, 3080),
        (3100, 2450),
        (3510, 2450),
    ],
    "train_03": [
        (3330, 430),
        (3330, 2800),
        (3000, 2800),
        (3000, 3610),
    ],
    "test_01": [
        (560, 1810),
        (3000, 1810),
        (3000, 1450),
        (3410, 1450),
    ],
    "test_02": [
        (900, 330),
        (900, 3000),
        (1500, 3000),
        (1500, 3690),
    ],
}
seq.SELECTION_VERSION = "soft_sequence_v7_three_straights_two_big_turns"

args = argparse.Namespace(
    dataset_root=sys.argv[1],
    city=sys.argv[2],
    output_root=sys.argv[3],
    step_m=4.0,
    max_sample_distance_m=float(sys.argv[5]),
    preferred_step_m=4.0,
    safety_max_step_m=float(sys.argv[4]),
    candidate_limit=160,
    beam_width=320,
    skip_penalty=36.0,
    continuity_weight=2.0,
    large_step_weight=0.75,
    cross_weight=1.5,
    backward_weight=10.0,
    # Strongly prefer true Bearing observations close to each straight leg and
    # with a stable same-leg lateral offset. This reduces sample scatter without
    # moving/relabeling any GT coordinate.
    point_cross_weight=12.0,
    lateral_smooth_weight=20.0,
    # The only planned turns are the two explicit major corners.
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

# First try a tight 6 m radius around the three straight legs. If the independent
# Bearing observations are too sparse, relax only the sample radius; the route
# geometry remains exactly three straight legs / two major turns.
if ! prepare_sequence 14 6; then
  echo "[PREP] 14m step cap / 6m sample radius too sparse; retrying 18m / 6m"
  if ! prepare_sequence 18 6; then
    echo "[PREP] 6m radius too sparse; retrying 18m / 8m"
    if ! prepare_sequence 18 8; then
      echo "[PREP] 18m / 8m too sparse; retrying 22m / 8m"
      if ! prepare_sequence 22 8; then
        echo "[PREP] final fallback: 22m step cap / 10m sample radius"
        prepare_sequence 22 10
      fi
    fi
  fi
fi

python3 - "${PREPARED_ROOT}/experiment.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
print("[THREE-STRAIGHTS/TWO-TURNS] route audit")
failed = []
for name, s in d["route_stats"].items():
    ratio = float(s["selected_ratio"])
    mean = float(s["actual_step_mean_m"])
    p90 = float(s["actual_step_p90_m"])
    center_p90 = float(s.get("centerline_cross_p90_m", 0.0))
    wobble_p90 = float(s.get("same_leg_lateral_delta_p90_m", 0.0))
    print(
        f"  {name}: waypoints={s['waypoints']} (=3 legs/2 turns) "
        f"frames={s['frames']} selected={ratio*100:.1f}% "
        f"step_mean={mean:.3f}m step_p90={p90:.3f}m "
        f"centerline_p90={center_p90:.3f}m sample_wobble_p90={wobble_p90:.3f}m"
    )
    if int(s["waypoints"]) != 4:
        failed.append(f"{name}: expected 4 waypoints, got {s['waypoints']}")
    if ratio < 0.75:
        failed.append(f"{name}: selected_ratio={ratio:.3f}")
if failed:
    raise SystemExit("[THREE-STRAIGHTS/TWO-TURNS] audit failed: " + "; ".join(failed))
print("[THREE-STRAIGHTS/TWO-TURNS] route audit: PASS")
PY

# IMPORTANT: localization model remains exact canonical v39. Nothing below
# changes GRU/Kalman/MeanShift.
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

# Green line = exact planned reference route (3 straight legs / 2 major turns).
# True Bearing GT samples remain metric truth, but are dots only and are never
# connected into the misleading caterpillar polyline.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing exact-v39 three-straight / two-turn experiment finished"
echo "Route geometry: STRAIGHT -> BIG TURN -> STRAIGHT -> BIG TURN -> STRAIGHT"
echo "Planned corners per route: 2"
echo "GT sample radius used: ${USED_SAMPLE_DISTANCE} m"
echo "Green line: planned reference route; true GT samples are unconnected dots"
echo "Metrics: still computed against true per-frame Bearing GT"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Route preview: ${PREPARED_ROOT}/route_plan_full_satellite.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
