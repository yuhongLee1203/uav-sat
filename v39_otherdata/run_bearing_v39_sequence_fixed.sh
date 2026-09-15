#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"
USED_SAMPLE_DISTANCE="?"
USED_MODE="?"

cd "${REPO_ROOT}"

prepare_sequence() {
  local max_cross="$1"
  local max_lat_jump="$2"
  local min_ratio="$3"
  local mode="$4"

  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] v12 dense->physical-prune: mode=${mode} cross<=${max_cross}m lateral_jump<=${max_lat_jump}m"

  if python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" \
      "${max_cross}" "${max_lat_jump}" "${min_ratio}" "${mode}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# These are the proven Bearing corridors: multiple irregular major turns with a
# straight waypoint-to-waypoint leg between turns.  Do not invent unsupported
# corridors merely to make the plot look straight.
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
seq.SELECTION_VERSION = "soft_sequence_v12_dense_then_physical_leg_prune"

args = argparse.Namespace(
    dataset_root=sys.argv[1],
    city=sys.argv[2],
    output_root=sys.argv[3],

    # Keep the successful v9 4 m target grid for dense matching.  We do NOT
    # demand one physical frame every 4 m; bad frames are removed afterwards.
    step_m=4.0,
    preferred_step_m=4.0,

    # Dense-stage settings deliberately match the previously successful regime.
    safety_max_step_m=22.0,
    max_sample_distance_m=10.0,
    candidate_limit=256,
    beam_width=512,
    skip_penalty=40.0,
    continuity_weight=2.5,
    large_step_weight=0.85,
    cross_weight=2.0,
    backward_weight=12.0,
    point_cross_weight=20.0,
    lateral_smooth_weight=35.0,
    big_turn_threshold_deg=25.0,

    # Second-stage physical straight-leg prune.
    max_cross_track_m=float(sys.argv[4]),
    max_same_leg_lateral_jump_m=float(sys.argv[5]),
    max_same_leg_backward_m=0.0,
    min_selected_ratio=float(sys.argv[6]),
)
seq.prepare(args)
PY
  then
    USED_SAMPLE_DISTANCE="10"
    USED_MODE="${mode}"
    return 0
  fi
  return 1
}

# The dense first stage is intentionally easy to satisfy because it reproduces
# the old 98-100% route matching.  Only the POST-selection physical prune is
# relaxed if a corridor is unusually sparse.
if ! prepare_sequence 5.5 2.75 0.30 strict; then
  echo "[PREP] strict post-prune kept too few frames; retrying balanced prune"
  if ! prepare_sequence 6.5 3.5 0.28 balanced; then
    echo "[PREP] balanced post-prune kept too few frames; retrying relaxed prune"
    prepare_sequence 8.0 4.5 0.25 relaxed
  fi
fi

python3 - "${PREPARED_ROOT}/experiment.json" "${PREPARED_ROOT}" <<'PY'
import json, math, sys
from pathlib import Path

p = Path(sys.argv[1])
root = Path(sys.argv[2])
d = json.loads(p.read_text(encoding="utf-8"))
print("[V12-DENSE-THEN-PRUNE] route audit")
failed = []

for name, s in d["route_stats"].items():
    ratio = float(s["selected_ratio"])
    dense_ratio = float(s.get("dense_selected_ratio", 0.0))
    keep_ratio = float(s.get("prune_keep_ratio", 0.0))
    center_p90 = float(s.get("centerline_cross_p90_m", 0.0))
    wobble_p90 = float(s.get("same_leg_lateral_delta_p90_m", 0.0))
    wobble_max = float(s.get("same_leg_lateral_delta_max_m", 0.0))
    backward = float(s.get("backward_step_pct", 0.0))
    same_back = float(s.get("same_leg_backward_step_pct", 0.0))
    step_mean = float(s.get("actual_step_mean_m", 0.0))
    step_p90 = float(s.get("actual_step_p90_m", 0.0))

    wp = json.loads(
        (root / "routes" / name / "waypoints.json").read_text(encoding="utf-8")
    )["waypoints"]
    pts = [(float(x["pixel_x"]), float(x["pixel_y"])) for x in wp]
    headings = []
    for a, b in zip(pts, pts[1:]):
        headings.append(math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))
    turns = [abs((b-a+180.0)%360.0-180.0) for a,b in zip(headings, headings[1:])]

    print(
        f"  {name}: dense={dense_ratio*100:.1f}% -> physical={ratio*100:.1f}% "
        f"(kept {keep_ratio*100:.1f}% of dense) frames={s['frames']} "
        f"step_mean={step_mean:.2f}m step_p90={step_p90:.2f}m "
        f"centerline_p90={center_p90:.2f}m "
        f"same_leg_wobble_p90={wobble_p90:.2f}m max={wobble_max:.2f}m "
        f"backward={backward:.2f}% same_leg_backward={same_back:.2f}% "
        f"turns={[round(v,1) for v in turns]}"
    )

    if int(s["frames"]) < 30:
        failed.append(f"{name}: only {s['frames']} frames")
    if same_back > 0.01:
        failed.append(f"{name}: same-leg backward remained {same_back:.3f}%")
    rule = float(s.get("max_same_leg_lateral_jump_rule_m", 999.0))
    if wobble_max > rule + 1e-5:
        failed.append(f"{name}: wobble {wobble_max:.3f} > rule {rule:.3f}")

if failed:
    raise SystemExit("[V12-DENSE-THEN-PRUNE] audit failed: " + "; ".join(failed))
print("[V12-DENSE-THEN-PRUNE] route audit: PASS")
PY

# Exact canonical v39 estimator.  Step=4 must match experiment.json so the
# runner reuses the v12 prepared data rather than rebuilding it with base prep.
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
echo "Bearing exact-v39 v12 experiment finished"
echo "Adapter: dense real-observation match -> per-straight-leg longest physical subsequence prune"
echo "Mode: ${USED_MODE}"
echo "No GT projection/relabeling; retained Bearing coordinates remain real"
echo "Green: planned/reference route | Cyan dots: true sampled GT | Red: final prediction"
echo "Diagnostic orange: Kalman before final MeanShift"
echo "Model unchanged: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Diag 1 : ${OUTPUT_DIR}/test_01_kalman_vs_final_diagnostic.jpg"
echo "Diag 2 : ${OUTPUT_DIR}/test_02_kalman_vs_final_diagnostic.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
