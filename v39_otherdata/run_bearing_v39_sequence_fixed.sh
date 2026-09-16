#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_bearing_adapted"

cd "${REPO_ROOT}"

echo "[CODE-AUDIT] compiling every file used by the Bearing-v39 run"
python3 -m py_compile \
  v39_otherdata/bearing_prepare.py \
  v39_otherdata/bearing_prepare_sequence_v3.py \
  v39_otherdata/data.py \
  v39_otherdata/bearing_runner.py \
  v39_otherdata/bearing_runner_exact_v39.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_DirectFinalMS/patch_direct_finalms.py
echo "[CODE-AUDIT] PASS"

rm -rf "${PREPARED_ROOT}"

echo "[PREP] multiple irregular turns + full-route coverage-safe pseudo-flight"
python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# Proven cityb corridors: multiple irregular turns.  Each waypoint-to-waypoint
# segment is a straight planned leg; no artificial micro-turn waypoint is added.
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
    step_m=4.0,
    preferred_step_m=4.0,
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
    # Cleanup may be used only when the implementation preserves full leg span;
    # bearing_prepare_sequence_v3.py now falls back to dense samples otherwise.
    max_cross_track_m=8.5,
    max_same_leg_lateral_jump_m=4.0,
    max_same_leg_backward_m=0.25,
    min_selected_ratio=0.25,
)
seq.prepare(args)
PY

# ---------------------------------------------------------------------------
# FULL-ROUTE COVERAGE AUDIT
# ---------------------------------------------------------------------------
python3 - "${PREPARED_ROOT}" <<'PY'
import csv, json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
exp = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
if exp.get("sequence_selection_version") != "soft_sequence_v12_dense_then_physical_leg_prune":
    raise SystemExit("[COVERAGE-AUDIT] unexpected selection version")

print("[COVERAGE-AUDIT] checking every waypoint against retained observations")
for route in ["train_01", "train_02", "train_03", "test_01", "test_02"]:
    with (root / "routes" / route / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 30:
        raise SystemExit(f"[COVERAGE-AUDIT] {route}: only {len(rows)} frames")
    pts = [(float(r["x_m"]), float(r["y_m"])) for r in rows]
    payload = json.loads((root / "routes" / route / "waypoints.json").read_text(encoding="utf-8"))
    wps = [
        (float(w["longitude"]), float(w["latitude"]))
        for w in sorted(payload["waypoints"], key=lambda w: int(w["waypoint_order"]))
    ]
    nearest = [min(math.hypot(x-wx, y-wy) for x, y in pts) for wx, wy in wps]
    start = math.hypot(pts[0][0]-wps[0][0], pts[0][1]-wps[0][1])
    end = math.hypot(pts[-1][0]-wps[-1][0], pts[-1][1]-wps[-1][1])
    if max(nearest) > 18.0 or start > 18.0 or end > 18.0:
        raise SystemExit(
            f"[COVERAGE-AUDIT] {route}: incomplete route | start={start:.2f}m "
            f"end={end:.2f}m worst_waypoint={max(nearest):.2f}m"
        )
    s = exp["route_stats"][route]
    print(
        f"  {route}: PASS frames={len(rows)} start={start:.2f}m end={end:.2f}m "
        f"worst_waypoint={max(nearest):.2f}m step_mean={float(s['actual_step_mean_m']):.2f}m "
        f"step_p90={float(s['actual_step_p90_m']):.2f}m"
    )
print("[COVERAGE-AUDIT] FULL ROUTE COVERAGE: PASS")

# Explicitly print the only route allowed to tune cadence.
t = exp["route_stats"]["train_01"]
print(
    "[TRAIN-CADENCE] train_01 only | mean=%.3fm p90=%.3fm p95=%.3fm"
    % (float(t["actual_step_mean_m"]), float(t["actual_step_p90_m"]), float(t["actual_step_p95_m"]))
)
PY

# ---------------------------------------------------------------------------
# SELECTED v39 METHOD + EXTERNAL-DATA PHYSICAL ADAPTER
# ---------------------------------------------------------------------------
# The legacy CLI name --epochs-per-route is retained by bearing_runner.py, but
# this wrapper requires 60 and bearing_runner_exact_v39.py performs ONE A-only
# 60-epoch run (it does NOT train three 20-epoch episodes).
python3 v39_otherdata/bearing_runner_exact_v39.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --gpu "${GPU}" \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 60 \
  --patience 10 \
  --jitter-m 8 \
  --step-m 4 \
  --max-sample-distance-m 10 \
  --heading-weight-px-per-deg 0

# ---------------------------------------------------------------------------
# FINAL COMPLETION / METRIC / ANTI-WIGGLE AUDIT
# ---------------------------------------------------------------------------
# Do not draw another misleading result.  The final image is created only if the
# route completes, metrics match the CSV, and the external adapter no longer
# exhibits the huge final-MS rescue jumps seen in the broken run.
python3 - "${PREPARED_ROOT}" "${OUTPUT_DIR}" <<'PY'
import csv, json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
summaries = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
with (root / "routes" / "train_01" / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
    a = list(csv.DictReader(f))
origin_x, origin_y = float(a[0]["x_m"]), float(a[0]["y_m"])

def quantile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    x = (len(values)-1) * float(q)
    lo, hi = int(math.floor(x)), int(math.ceil(x))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi-x) + values[hi] * (x-lo)

print("[FINAL-QUALITY-AUDIT]")
for route in ["test_01", "test_02"]:
    s = summaries[route]
    wp = json.loads((root / "routes" / route / "waypoints.json").read_text(encoding="utf-8"))["waypoints"]
    wp = sorted(wp, key=lambda w: int(w["waypoint_order"]))
    last_leg = len(wp) - 2
    pred_leg = int(s["FinalPredictedWaypointLeg"])
    gt_leg = int(s["FinalGTWaypointLeg"])
    if pred_leg != last_leg or gt_leg != last_leg:
        raise SystemExit(
            f"[FINAL-QUALITY-AUDIT] {route}: route incomplete pred={pred_leg} gt={gt_leg} required={last_leg}"
        )

    csv_path = Path(s["CSV"])
    if not csv_path.exists():
        csv_path = out / csv_path.name
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (root / "routes" / route / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if len(rows) != len(manifest):
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: CSV/manifest length mismatch")

    errors = [math.hypot(float(r["final_x"])-float(r["gt_x"]), float(r["final_y"])-float(r["gt_y"])) for r in rows]
    mle = sum(errors) / len(errors)
    if abs(mle - float(s["MLE_m"])) > 1e-5:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: summary MLE mismatch")

    last = rows[-1]
    final_abs = (float(last["final_x"])+origin_x, float(last["final_y"])+origin_y)
    end_wp = (float(wp[-1]["longitude"]), float(wp[-1]["latitude"]))
    end_distance = math.hypot(final_abs[0]-end_wp[0], final_abs[1]-end_wp[1])
    if end_distance > 35.0:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: final endpoint distance={end_distance:.2f}m")

    ms_shift = float(s["MS_MeanShiftFromKalman_m"])
    step_limited = float(s["KalmanStepLimited_pct"])
    jump = float(s["JumpRate_pct"])
    cross = [abs(float(r["final_cross_e"])) for r in rows]
    cross_delta = [abs(float(rows[i]["final_cross_e"])-float(rows[i-1]["final_cross_e"])) for i in range(1, len(rows))]
    cross_p90 = quantile(cross, 0.90)
    cross_delta_p90 = quantile(cross_delta, 0.90)

    # Broken uploaded run: MS mean=8.80/9.55 m and K-limit=85.5/81.8%.
    # These gates prevent that failure mode from being emitted as a final plot.
    if ms_shift > 7.0:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: final-MS rescue still too large: {ms_shift:.2f}m")
    if step_limited > 80.0:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: Kalman still cadence-limited: {step_limited:.2f}%")
    if jump > 5.0:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: jump rate too high: {jump:.2f}%")
    if cross_p90 > 7.5:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: route cross-track P90 too high: {cross_p90:.2f}m")
    if cross_delta_p90 > 7.0:
        raise SystemExit(f"[FINAL-QUALITY-AUDIT] {route}: lateral frame wobble P90 too high: {cross_delta_p90:.2f}m")

    print(
        f"  {route}: PASS | MLE={mle:.3f}m end={end_distance:.2f}m "
        f"KLimit={step_limited:.1f}% MSshift={ms_shift:.2f}m "
        f"crossP90={cross_p90:.2f}m lateralDeltaP90={cross_delta_p90:.2f}m jump={jump:.2f}%"
    )
print("[FINAL-QUALITY-AUDIT] BOTH TEST ROUTES: PASS")
PY

# ONLY final prediction-vs-reference images.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================"
echo "Bearing v39 physically-adapted experiment finished"
echo "Architecture: Weighted Centroid -> 3-frame GRU -> CV -> fixed Kalman -> final 6x6 MS (BW7)"
echo "Training: ONE Route A (train_01), 60 epochs; B/C = test_01/test_02"
echo "Scale: v39 physical SAT spacing/FOV preserved across 0.14 -> 0.25 m/px"
echo "Cadence: longitudinal limits derived from TRAIN train_01 only; test stats never tune parameters"
echo "Final MS: route-centerline spatial reference; true Bearing sample coordinates remain metric GT"
echo "Output: ONLY final prediction vs reference route"
echo "  ${OUTPUT_DIR}/test_01_final_result.jpg"
echo "  ${OUTPUT_DIR}/test_02_final_result.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Audit  : ${OUTPUT_DIR}/v39_bearing_training_audit.json"
echo "================================================================================"
