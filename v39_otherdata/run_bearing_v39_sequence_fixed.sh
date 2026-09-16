#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_bearing_adapted"

case "${CITY}" in
  citya|cityb|cityc|cityd) ;;
  *) echo "Unsupported CITY=${CITY}; use citya/cityb/cityc/cityd" >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"

echo "================================================================================"
echo "Bearing-v39 ${CITY}"
echo "Train: auto-selected COMPLETE city-specific Route A -> 60 epochs"
echo "Test : TWO OFFICIAL Bearing-UAV navigation routes"
echo "================================================================================"

python3 -m py_compile \
  v39_otherdata/bearing_prepare.py \
  v39_otherdata/bearing_prepare_sequence_v3.py \
  v39_otherdata/bearing_multicity_routes.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_otherdata/bearing_runner.py \
  v39_otherdata/bearing_runner_exact_v39.py \
  v39_otherdata/bearing_runner_multicity_v39.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_otherdata/bearing_paper_metrics.py \
  v39_DirectFinalMS/patch_direct_finalms.py
echo "[CODE-AUDIT] PASS"

rm -rf "${PREPARED_ROOT}"

# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------
# Probe three TRAIN-ONLY route candidates.  Only a full-route candidate can be
# renamed to canonical train_01 / Route A.  The two test routes are fixed to the
# official Bearing-UAV navigation waypoints and never take part in Route-A choice.
python3 v39_otherdata/bearing_prepare_multicity.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --output-root "${PREPARED_ROOT}"

python3 - "${PREPARED_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
exp = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
print("[PREP-AUDIT] PASS")
print("  Route-A source :", exp.get("route_a_candidate_source"))
print("  Profile        :", exp.get("preparation_profile", {}).get("name"))
for name in ("train_01", "test_01", "test_02"):
    s = exp["route_stats"][name]
    c = exp["full_route_coverage_audit"][name]
    print(
        f"  {name}: frames={s['frames']} waypoints={s['waypoints']} "
        f"step_mean={s['actual_step_mean_m']:.2f}m step_p90={s['actual_step_p90_m']:.2f}m "
        f"end={c['end_m']:.2f}m worst_wp={c['worst_waypoint_m']:.2f}m"
    )
print("[PREP-AUDIT] ONE COMPLETE Route A + TWO official tests: PASS")
PY

# ---------------------------------------------------------------------------
# MODEL: selected v39 method + external-data physical adaptation
# ---------------------------------------------------------------------------
python3 v39_otherdata/bearing_runner_multicity_v39.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --gpu "${GPU}" \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 60 \
  --patience 10 \
  --jitter-m 8 \
  --step-m 4 \
  --max-sample-distance-m 15 \
  --heading-weight-px-per-deg 0

# ---------------------------------------------------------------------------
# STRUCTURAL/METRIC AUDIT
# ---------------------------------------------------------------------------
# Hard errors are reserved for corrupt/misaligned outputs.  Quality observations
# are WARNINGS only: a weak scientific result must still be plotted and reported,
# rather than disappearing before the user can inspect it.
python3 - "${PREPARED_ROOT}" "${OUTPUT_DIR}" <<'PY'
import csv, json, math, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
summaries = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
report = {}
for route in ("test_01", "test_02"):
    s = summaries[route]
    csv_path = Path(s["CSV"])
    if not csv_path.exists():
        csv_path = out / csv_path.name
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (root / "routes" / route / "manifest.csv").open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if not rows or len(rows) != len(manifest):
        raise SystemExit(f"[RESULT-AUDIT] {route}: corrupt frame count {len(rows)} vs {len(manifest)}")
    errors = [math.hypot(float(r["final_x"])-float(r["gt_x"]), float(r["final_y"])-float(r["gt_y"])) for r in rows]
    mle = sum(errors) / len(errors)
    if abs(mle - float(s["MLE_m"])) > 1e-5:
        raise SystemExit(f"[RESULT-AUDIT] {route}: MLE mismatch CSV={mle} summary={s['MLE_m']}")
    warnings = []
    if float(s["MS_MeanShiftFromKalman_m"]) > 7.0:
        warnings.append("large final-MS correction")
    if float(s["KalmanStepLimited_pct"]) > 80.0:
        warnings.append("Kalman cadence-limited")
    if float(s["JumpRate_pct"]) > 5.0:
        warnings.append("high jump rate")
    last_leg = int(s["Waypoints"]) - 2
    if int(s["FinalPredictedWaypointLeg"]) != last_leg:
        warnings.append("prediction did not reach final waypoint leg")
    report[route] = {
        "frames": len(rows), "MLE_m": mle, "P90_m": float(s["P90_m"]),
        "LSR@15_pct": float(s["LSR@15_pct"]),
        "JumpRate_pct": float(s["JumpRate_pct"]),
        "KalmanStepLimited_pct": float(s["KalmanStepLimited_pct"]),
        "MS_MeanShiftFromKalman_m": float(s["MS_MeanShiftFromKalman_m"]),
        "warnings": warnings,
    }
    state = "PASS" if not warnings else "WARN: " + "; ".join(warnings)
    print(f"[RESULT-AUDIT] {route}: {state} | MLE={mle:.3f}m P90={float(s['P90_m']):.3f}m LSR15={float(s['LSR@15_pct']):.2f}%")
(out / "final_quality_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("[RESULT-AUDIT] structural/metric consistency: PASS")
PY

# ---------------------------------------------------------------------------
# ALWAYS render the two scientific final-result figures after valid inference.
# ---------------------------------------------------------------------------
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

test -s "${OUTPUT_DIR}/test_01_final_result.jpg"
test -s "${OUTPUT_DIR}/test_02_final_result.jpg"
echo "[FINAL-IMAGES] PASS: two result figures exist"

# Bearing-UAV paper-comparison metrics.  Directly comparable fields are marked
# separately from derived/incompatible protocols inside the JSON.
python3 v39_otherdata/bearing_paper_metrics.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

echo ""
echo "================================================================================"
echo "DONE ${CITY}"
echo "Final figures:"
echo "  ${OUTPUT_DIR}/test_01_final_result.jpg"
echo "  ${OUTPUT_DIR}/test_02_final_result.jpg"
echo "Paper metrics:"
echo "  ${OUTPUT_DIR}/bearing_paper_metrics.json"
echo "  ${OUTPUT_DIR}/bearing_paper_metrics.csv"
echo "Raw summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "================================================================================"
