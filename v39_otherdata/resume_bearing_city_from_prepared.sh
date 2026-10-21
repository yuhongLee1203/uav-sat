#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CITY="${CITY:-citya}"
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_bearing_adapted"

case "${CITY}" in
  citya|cityb|cityc|cityd) ;;
  *) echo "ERROR: unsupported CITY=${CITY}" >&2; exit 2 ;;
esac

# Resume contract: preparation must already be complete. Never rebuild/delete it.
for f in \
  v39_otherdata/data.py \
  "${PREPARED_ROOT}/experiment.json" \
  "${PREPARED_ROOT}/bearing_satellite.json" \
  "${PREPARED_ROOT}/routes/train_01/manifest.csv" \
  "${PREPARED_ROOT}/routes/train_01/waypoints.json" \
  "${PREPARED_ROOT}/routes/test_01/manifest.csv" \
  "${PREPARED_ROOT}/routes/test_01/waypoints.json" \
  "${PREPARED_ROOT}/routes/test_02/manifest.csv" \
  "${PREPARED_ROOT}/routes/test_02/waypoints.json"; do
  [[ -s "${f}" ]] || { echo "ERROR: required prepared file missing: ${f}" >&2; exit 3; }
done

python3 -m py_compile \
  v39_otherdata/data.py \
  v39_otherdata/bearing_runner.py \
  v39_otherdata/bearing_runner_exact_v39.py \
  v39_otherdata/bearing_runner_multicity_v39.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_otherdata/bearing_paper_metrics.py \
  v39_DirectFinalMS/patch_direct_finalms.py

echo "================================================================================"
echo "RESUME ${CITY} FROM EXISTING PREPARED ROUTES"
echo "No route preparation will be rerun or deleted."
echo "================================================================================"

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

python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

python3 v39_otherdata/bearing_paper_metrics.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/bearing_v39_summary.json"
test -s "${OUTPUT_DIR}/bearing_paper_metrics.json"
test -s "${OUTPUT_DIR}/paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg"
test -s "${OUTPUT_DIR}/paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg"

python3 - "${PREPARED_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
out = root / "v39_output_bearing_adapted"
summary = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
print("[RESUME-DONE]")
for route in ("test_01", "test_02"):
    s = summary[route]
    print(f"  {route}: MLE={s['MLE_m']:.3f}m MedLE={s['MedLE_m']:.3f}m P90={s['P90_m']:.3f}m LSR@15={s['LSR@15_pct']:.2f}%")
PY
