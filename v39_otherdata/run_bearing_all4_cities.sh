#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU="${GPU:-0}"
CITIES=(citya cityb cityc cityd)

export DATASET_ROOT GPU

echo "================================================================================"
echo "Bearing-v39 PAPER MAIN EXPERIMENT: ALL FOUR CITIES"
echo "Cities : ${CITIES[*]}"
echo "GPU    : ${GPU}"
echo "Dataset: ${DATASET_ROOT}"
echo "Method : identical existing DirectFinalMS Bearing workflow for every city"
echo "Output : two official test routes + waypoint-GT paper figures per city"
echo "================================================================================"

for CITY in "${CITIES[@]}"; do
  echo ""
  echo "################################################################################"
  echo "START ${CITY}"
  echo "################################################################################"
  CITY="${CITY}" \
  GPU="${GPU}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  bash v39_otherdata/run_bearing_v39_directfinalms_official_routes.sh

  OUT="v39_otherdata/generated/${CITY}/v39_output_bearing_adapted"
  test -s "${OUT}/bearing_v39_summary.json"
  test -s "${OUT}/bearing_paper_metrics.json"
  test -s "${OUT}/test_01_final_result.jpg"
  test -s "${OUT}/test_02_final_result.jpg"
  test -s "${OUT}/paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg"
  test -s "${OUT}/paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg"

  echo "[ALL4] ${CITY}: PASS"
done

python3 - <<'PY'
import csv
import json
import math
from pathlib import Path

import numpy as np

root = Path("v39_otherdata/generated")
cities = ["citya", "cityb", "cityc", "cityd"]
rows_out = []
all_errors = []

for city in cities:
    out = root / city / "v39_output_bearing_adapted"
    summary = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
    for route in ("test_01", "test_02"):
        s = summary[route]
        csv_path = Path(str(s["CSV"]))
        if not csv_path.exists():
            csv_path = out / csv_path.name
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            frame_rows = list(csv.DictReader(f))
        errors = [float(r["error_final_m"]) for r in frame_rows]
        all_errors.extend(errors)
        row = {
            "city": city,
            "route": route,
            "frames": len(errors),
            "MLE_m": float(np.mean(errors)),
            "MedLE_m": float(np.median(errors)),
            "P90_m": float(np.percentile(errors, 90)),
            "P95_m": float(np.percentile(errors, 95)),
            "P99_m": float(np.percentile(errors, 99)),
            "LSR@5_pct": 100.0 * float(np.mean(np.asarray(errors) <= 5.0)),
            "LSR@10_pct": 100.0 * float(np.mean(np.asarray(errors) <= 10.0)),
            "LSR@15_pct": 100.0 * float(np.mean(np.asarray(errors) <= 15.0)),
            "LSR@20_pct": 100.0 * float(np.mean(np.asarray(errors) <= 20.0)),
        }
        rows_out.append(row)

arr = np.asarray(all_errors, dtype=np.float64)
overall = {
    "city": "ALL",
    "route": "ALL_8_ROUTES",
    "frames": int(arr.size),
    "MLE_m": float(np.mean(arr)),
    "MedLE_m": float(np.median(arr)),
    "P90_m": float(np.percentile(arr, 90)),
    "P95_m": float(np.percentile(arr, 95)),
    "P99_m": float(np.percentile(arr, 99)),
    "LSR@5_pct": 100.0 * float(np.mean(arr <= 5.0)),
    "LSR@10_pct": 100.0 * float(np.mean(arr <= 10.0)),
    "LSR@15_pct": 100.0 * float(np.mean(arr <= 15.0)),
    "LSR@20_pct": 100.0 * float(np.mean(arr <= 20.0)),
}

payload = {
    "method": "v39 DirectFinalMS Bearing official-route workflow",
    "cities": cities,
    "routes": rows_out,
    "overall_8_routes": overall,
}

json_path = root / "paper_all4_summary.json"
csv_path = root / "paper_all4_summary.csv"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

fields = [
    "city", "route", "frames", "MLE_m", "MedLE_m", "P90_m", "P95_m", "P99_m",
    "LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)
    w.writerow(overall)

print("================================================================================")
print("ALL FOUR CITIES COMPLETE")
for r in rows_out:
    print(
        f"{r['city']} {r['route']}: frames={r['frames']} "
        f"MLE={r['MLE_m']:.3f}m MedLE={r['MedLE_m']:.3f}m "
        f"P90={r['P90_m']:.3f}m LSR@15={r['LSR@15_pct']:.2f}%"
    )
print("--------------------------------------------------------------------------------")
print(
    f"ALL 8 ROUTES: frames={overall['frames']} MLE={overall['MLE_m']:.3f}m "
    f"MedLE={overall['MedLE_m']:.3f}m P90={overall['P90_m']:.3f}m "
    f"LSR@15={overall['LSR@15_pct']:.2f}%"
)
print("Summary JSON:", json_path)
print("Summary CSV :", csv_path)
print("================================================================================")
PY
