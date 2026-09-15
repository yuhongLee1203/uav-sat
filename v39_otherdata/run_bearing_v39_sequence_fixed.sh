#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"
USED_SAMPLE_DISTANCE="8"

cd "${REPO_ROOT}"

prepare_sequence() {
  local safety_cap="$1"
  local sample_distance="$2"
  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] route = LONG STRAIGHTS + MULTIPLE IRREGULAR MAJOR TURNS; safety=${safety_cap}m sample_radius=${sample_distance}m"

  if python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" "${safety_cap}" "${sample_distance}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# v8 route policy:
# - 7 waypoints = 6 long straight legs and 5 explicit major turns per route.
# - Turn angles are intentionally irregular; they are NOT fixed at 90 degrees.
# - No intermediate micro-bend waypoint is inserted inside a straight leg.
# - Typical leg length is roughly 120-230 m on the 0.25 m/px Bearing map.
seq.PIECEWISE_ROUTE_SPECS = {
    "train_01": [
        (330, 620),
        (1050, 660),
        (1450, 1100),
        (2150, 980),
        (2550, 1500),
        (3200, 1420),
        (3450, 1950),
    ],
    "train_02": [
        (430, 3200),
        (1100, 3000),
        (1550, 2500),
        (2250, 2700),
        (2800, 2250),
        (3450, 2500),
        (3250, 3400),
    ],
    "train_03": [
        (3350, 430),
        (3000, 950),
        (3500, 1450),
        (3050, 2100),
        (3600, 2600),
        (3150, 3200),
        (3500, 3700),
    ],
    "test_01": [
        (560, 1850),
        (1200, 1550),
        (1800, 1750),
        (2250, 1300),
        (2850, 1600),
        (3200, 1250),
        (3500, 1900),
    ],
    "test_02": [
        (900, 330),
        (1250, 900),
        (850, 1500),
        (1450, 2050),
        (1050, 2700),
        (1650, 3150),
        (1450, 3700),
    ],
}
seq.SELECTION_VERSION = "soft_sequence_v8_irregular_major_turns_long_legs"

args = argparse.Namespace(
    dataset_root=sys.argv[1],
    city=sys.argv[2],
    output_root=sys.argv[3],
    step_m=4.0,
    max_sample_distance_m=float(sys.argv[5]),
    preferred_step_m=4.0,
    safety_max_step_m=float(sys.argv[4]),
    candidate_limit=192,
    beam_width=384,
    skip_penalty=38.0,
    continuity_weight=2.5,
    large_step_weight=0.9,
    cross_weight=2.0,
    backward_weight=12.0,
    # Bearing observations are independent.  These stronger data-selection
    # weights prefer true samples that stay close to each long straight segment
    # and do not alternate left/right every few frames.  GT coordinates are
    # never moved or projected onto the route.
    point_cross_weight=20.0,
    lateral_smooth_weight=35.0,
    # Smallest planned major turn is about 31 deg.  A 25-deg threshold therefore
    # preserves every deliberate corner while still smoothing only within a leg.
    big_turn_threshold_deg=25.0,
    min_selected_ratio=0.70,
)
seq.prepare(args)
PY
  then
    USED_SAMPLE_DISTANCE="${sample_distance}"
    return 0
  fi
  return 1
}

# Keep real Bearing observations much closer to the planned long legs than the
# previous 6-10 m setup.  Wider radii are fallbacks only when the independent
# image pool is too sparse.  The route geometry itself never changes in fallback.
if ! prepare_sequence 14 4; then
  echo "[PREP] 14m step cap / 4m sample radius too sparse; retrying 18m / 4m"
  if ! prepare_sequence 18 4; then
    echo "[PREP] 4m radius too sparse; retrying 18m / 5m"
    if ! prepare_sequence 18 5; then
      echo "[PREP] 18m / 5m too sparse; retrying 22m / 6m"
      if ! prepare_sequence 22 6; then
        echo "[PREP] final fallback: 22m step cap / 8m sample radius"
        prepare_sequence 22 8
      fi
    fi
  fi
fi

python3 - "${PREPARED_ROOT}/experiment.json" "${PREPARED_ROOT}" <<'PY'
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
root = Path(sys.argv[2])
d = json.loads(p.read_text(encoding="utf-8"))
print("[IRREGULAR-MAJOR-TURNS] route audit")
failed = []
for name, s in d["route_stats"].items():
    ratio = float(s["selected_ratio"])
    mean = float(s["actual_step_mean_m"])
    p90 = float(s["actual_step_p90_m"])
    center_p90 = float(s.get("centerline_cross_p90_m", 0.0))
    wobble_p90 = float(s.get("same_leg_lateral_delta_p90_m", 0.0))

    wp = json.loads((root / "routes" / name / "waypoints.json").read_text(encoding="utf-8"))["waypoints"]
    pts = [(float(x["pixel_x"]), float(x["pixel_y"])) for x in wp]
    headings = []
    leg_m = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        headings.append(math.degrees(math.atan2(dy, dx)))
        leg_m.append(math.hypot(dx, dy) * 0.25)
    turns = [abs((b-a+180.0)%360.0-180.0) for a,b in zip(headings, headings[1:])]

    print(
        f"  {name}: waypoints={len(pts)} legs={len(leg_m)} turns={len(turns)} "
        f"turn_deg={[round(v,1) for v in turns]} "
        f"leg_m={[round(v,1) for v in leg_m]} frames={s['frames']} "
        f"selected={ratio*100:.1f}% centerline_p90={center_p90:.3f}m "
        f"sample_wobble_p90={wobble_p90:.3f}m step_mean={mean:.3f}m step_p90={p90:.3f}m"
    )
    if len(pts) != 7 or len(turns) != 5:
        failed.append(f"{name}: expected 7 waypoints / 5 turns")
    if turns and min(turns) < 25.0:
        failed.append(f"{name}: micro-turn remained ({min(turns):.1f} deg)")
    if ratio < 0.70:
        failed.append(f"{name}: selected_ratio={ratio:.3f}")
if failed:
    raise SystemExit("[IRREGULAR-MAJOR-TURNS] audit failed: " + "; ".join(failed))
print("[IRREGULAR-MAJOR-TURNS] route audit: PASS")
PY

# IMPORTANT: localization model remains exact canonical v39.  This experiment
# changes Bearing route construction / sample selection only; GRU, Kalman and
# final MeanShift are not silently altered.
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

# Green solid line = planned/reference route.  Actual per-frame Bearing GT stays
# as unconnected dots.  Red = the real final v39 output (not route-projected).
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing exact-v39 irregular-major-turn experiment finished"
echo "Route geometry: 6 LONG straight legs + 5 IRREGULAR major turns per route"
echo "Turn policy: multiple turns; NOT fixed 90 degrees; no micro-bend waypoints"
echo "GT sample radius used: ${USED_SAMPLE_DISTANCE} m"
echo "Green solid line: planned/reference route"
echo "True per-frame Bearing GT: unconnected dots; metrics still use these real coordinates"
echo "Red line: raw exact-v39 final prediction; no cosmetic projection/smoothing"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Route preview: ${PREPARED_ROOT}/route_plan_full_satellite.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
