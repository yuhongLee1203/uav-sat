#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"
USED_SAMPLE_DISTANCE="10"
USED_MODE="smooth"

cd "${REPO_ROOT}"

prepare_sequence() {
  local safety_cap="$1"
  local sample_distance="$2"
  local point_cross="$3"
  local lateral_smooth="$4"
  local min_ratio="$5"
  local mode="$6"

  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] proven Bearing corridors + multiple irregular turns; safety=${safety_cap}m sample_radius=${sample_distance}m mode=${mode}"

  if python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" "${safety_cap}" "${sample_distance}" "${point_cross}" "${lateral_smooth}" "${min_ratio}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# IMPORTANT:
# These are the original Bearing route corridors that already produced dense,
# successful pseudo-flight sequences in the previous runs.  We do NOT invent a
# new path through regions where Bearing has no observations.  Every consecutive
# waypoint pair is a straight leg; the intermediate waypoints are the deliberate
# route turns.  The turns are multiple and irregular, not a fixed two-turn or
# fixed-90-degree pattern.
seq.PIECEWISE_ROUTE_SPECS = {
    "train_01": [
        (330, 620), (690, 850), (1060, 690), (1390, 1030), (1710, 880),
        (1990, 1210), (2240, 1090), (2510, 1450), (2780, 1290),
        (3070, 1620), (3330, 1480),
    ],
    "train_02": [
        (430, 3080), (770, 2780), (1120, 3060), (1460, 2700),
        (1800, 2970), (2110, 2600), (2460, 2910), (2800, 2510),
        (3170, 2780), (3510, 2410),
    ],
    "train_03": [
        (3330, 430), (3050, 760), (3410, 1110), (3100, 1480),
        (3510, 1810), (3200, 2180), (3560, 2530), (3260, 2900),
        (3610, 3260), (3310, 3610),
    ],
    "test_01": [
        (560, 1810), (900, 1510), (1260, 1840), (1610, 1540),
        (1980, 1900), (2320, 1610), (2680, 1970), (3250, 1450),
        (3410, 2050),
    ],
    "test_02": [
        (900, 330), (1160, 660), (900, 1010), (1270, 1320),
        (1010, 1660), (1370, 2010), (1090, 2360), (1500, 2660),
        (1240, 3010), (1660, 3360), (1440, 3690),
    ],
}
seq.SELECTION_VERSION = "soft_sequence_v9_proven_corridors_irregular_turns_straight_legs"

args = argparse.Namespace(
    dataset_root=sys.argv[1],
    city=sys.argv[2],
    output_root=sys.argv[3],
    step_m=4.0,
    max_sample_distance_m=float(sys.argv[5]),
    preferred_step_m=4.0,
    safety_max_step_m=float(sys.argv[4]),
    candidate_limit=256,
    beam_width=512,
    skip_penalty=40.0,
    continuity_weight=2.5,
    large_step_weight=0.85,
    cross_weight=2.0,
    backward_weight=12.0,
    # High weights are intentional: inside each straight leg, select real
    # Bearing observations that stay near the leg centreline and avoid rapidly
    # alternating left/right offsets.  The coordinates themselves are never
    # projected or relabelled.
    point_cross_weight=float(sys.argv[6]),
    lateral_smooth_weight=float(sys.argv[7]),
    # All planned turns on these proven corridors are well above 25 degrees.
    # Same-leg smoothing therefore switches off only at a deliberate corner.
    big_turn_threshold_deg=25.0,
    min_selected_ratio=float(sys.argv[8]),
)
seq.prepare(args)
PY
  then
    USED_SAMPLE_DISTANCE="${sample_distance}"
    USED_MODE="${mode}"
    return 0
  fi
  return 1
}

# First try aggressive straight-leg sample selection on the proven data corridor.
# Relax only the observation radius if needed.  Because the geometry is the
# previously successful corridor, 10-12 m should normally be sufficient.
if ! prepare_sequence 14 6 24 45 0.70 smooth6; then
  echo "[PREP] 6m radius too sparse; retrying 18m / 8m"
  if ! prepare_sequence 18 8 22 40 0.70 smooth8; then
    echo "[PREP] 8m radius too sparse; retrying 22m / 10m"
    if ! prepare_sequence 22 10 20 35 0.70 smooth10; then
      echo "[PREP] 10m radius too sparse; retrying 22m / 12m"
      if ! prepare_sequence 22 12 16 28 0.68 smooth12; then
        echo "[PREP] final density fallback: proven corridor / 15m radius"
        # This last setting remains much smoother than the old nearest-point
        # selector but prioritizes obtaining a valid temporal sequence.
        prepare_sequence 24 15 12 20 0.65 smooth15
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
print("[PROVEN-CORRIDOR/IRREGULAR-TURNS] route audit")
failed = []
for name, s in d["route_stats"].items():
    ratio = float(s["selected_ratio"])
    mean = float(s["actual_step_mean_m"])
    p90 = float(s["actual_step_p90_m"])
    center_p90 = float(s.get("centerline_cross_p90_m", 0.0))
    wobble_p90 = float(s.get("same_leg_lateral_delta_p90_m", 0.0))

    wp = json.loads((root / "routes" / name / "waypoints.json").read_text(encoding="utf-8"))["waypoints"]
    pts = [(float(x["pixel_x"]), float(x["pixel_y"])) for x in wp]
    headings, leg_m = [], []
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
    if len(turns) < 5:
        failed.append(f"{name}: expected multiple turns, got {len(turns)}")
    if turns and min(turns) < 25.0:
        failed.append(f"{name}: micro planned turn remained ({min(turns):.1f} deg)")
    if ratio < 0.65:
        failed.append(f"{name}: selected_ratio={ratio:.3f}")
if failed:
    raise SystemExit("[PROVEN-CORRIDOR/IRREGULAR-TURNS] audit failed: " + "; ".join(failed))
print("[PROVEN-CORRIDOR/IRREGULAR-TURNS] route audit: PASS")
PY

# Exact canonical v39 model/inference settings remain unchanged.
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

# Green = planned/reference straight-leg route.
# Cyan dots = actual sampled Bearing GT (not connected).
# Red = raw exact-v39 final prediction.
# Orange in diagnostic = Kalman state before final MeanShift.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing exact-v39 proven-corridor experiment finished"
echo "Route policy: multiple irregular turns on previously verified dense Bearing corridors"
echo "Each waypoint-to-waypoint section is a straight leg; no micro-bend waypoint inside a leg"
echo "Sample-selection mode: ${USED_MODE}; radius used: ${USED_SAMPLE_DISTANCE} m"
echo "Green: planned/reference route | Cyan dots: true sampled GT | Red: final prediction"
echo "Diagnostic orange: Kalman before final MeanShift"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Route preview: ${PREPARED_ROOT}/route_plan_full_satellite.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Diag 1 : ${OUTPUT_DIR}/test_01_kalman_vs_final_diagnostic.jpg"
echo "Diag 2 : ${OUTPUT_DIR}/test_02_kalman_vs_final_diagnostic.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
