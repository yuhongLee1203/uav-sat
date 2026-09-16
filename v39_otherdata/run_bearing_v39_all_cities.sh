#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU="${GPU:-0}"
CITIES="${CITIES:-citya cityb cityc cityd}"

cd "${REPO_ROOT}"

echo "================================================================================"
echo "Bearing-v39 MULTI-CITY evaluation"
echo "Cities : ${CITIES}"
echo "GPU    : ${GPU} (cities run sequentially)"
echo "Per city: Route A train 60 epochs -> TWO official held-out navigation routes"
echo "================================================================================"

for city in ${CITIES}; do
  case "${city}" in
    citya|cityb|cityc|cityd) ;;
    *) echo "Unsupported city in CITIES: ${city}" >&2; exit 2 ;;
  esac

  echo ""
  echo "################################################################################"
  echo "# START ${city}"
  echo "################################################################################"
  DATASET_ROOT="${DATASET_ROOT}" \
  CITY="${city}" \
  GPU="${GPU}" \
  bash v39_otherdata/run_bearing_v39_sequence_fixed.sh
  echo "################################################################################"
  echo "# DONE ${city}"
  echo "################################################################################"
done

# Aggregate all city/test summaries into one machine-readable JSON + CSV.
python3 - "${REPO_ROOT}/v39_otherdata/generated" ${CITIES} <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
cities = sys.argv[2:]
rows = []
city_payload = {}

for city in cities:
    summary_path = root / city / "v39_output_bearing_adapted" / "bearing_v39_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"[MULTICITY-AUDIT] missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary) != {"test_01", "test_02"}:
        raise SystemExit(f"[MULTICITY-AUDIT] {city}: expected exactly test_01/test_02")

    city_payload[city] = summary
    for route in ("test_01", "test_02"):
        s = summary[route]
        rows.append({
            "city": city,
            "route": route,
            "MLE_m": float(s["MLE_m"]),
            "MedLE_m": float(s["MedLE_m"]),
            "P90_m": float(s["P90_m"]),
            "P95_m": float(s["P95_m"]),
            "P99_m": float(s["P99_m"]),
            "LSR@5_pct": float(s["LSR@5_pct"]),
            "LSR@10_pct": float(s["LSR@10_pct"]),
            "LSR@15_pct": float(s["LSR@15_pct"]),
            "LSR@20_pct": float(s["LSR@20_pct"]),
            "JumpRate_pct": float(s["JumpRate_pct"]),
            "KalmanStepLimited_pct": float(s["KalmanStepLimited_pct"]),
            "MS_MeanShiftFromKalman_m": float(s["MS_MeanShiftFromKalman_m"]),
            "FinalPredictedWaypointLeg": int(s["FinalPredictedWaypointLeg"]),
            "FinalGTWaypointLeg": int(s["FinalGTWaypointLeg"]),
            "Waypoints": int(s["Waypoints"]),
        })

if len(rows) != 2 * len(cities):
    raise SystemExit("[MULTICITY-AUDIT] unexpected route count")

aggregate = {
    "cities": cities,
    "test_routes_per_city": 2,
    "total_test_routes": len(rows),
    "mean_MLE_m": sum(r["MLE_m"] for r in rows) / len(rows),
    "mean_P90_m": sum(r["P90_m"] for r in rows) / len(rows),
    "mean_LSR@5_pct": sum(r["LSR@5_pct"] for r in rows) / len(rows),
    "mean_LSR@10_pct": sum(r["LSR@10_pct"] for r in rows) / len(rows),
    "max_JumpRate_pct": max(r["JumpRate_pct"] for r in rows),
    "routes": rows,
    "raw_city_summaries": city_payload,
}

json_path = root / "bearing_multicity_summary.json"
csv_path = root / "bearing_multicity_summary.csv"
json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("[MULTICITY-AUDIT] PASS")
print(f"  cities={len(cities)} test_routes={len(rows)}")
print(f"  mean MLE={aggregate['mean_MLE_m']:.3f}m")
print(f"  mean P90={aggregate['mean_P90_m']:.3f}m")
print(f"  mean LSR@10={aggregate['mean_LSR@10_pct']:.2f}%")
print(f"[MULTICITY] JSON: {json_path}")
print(f"[MULTICITY] CSV : {csv_path}")
PY

echo ""
echo "================================================================================"
echo "ALL REQUESTED BEARING CITIES FINISHED"
echo "Each city produced TWO final test-route images:"
for city in ${CITIES}; do
  echo "  v39_otherdata/generated/${city}/v39_output_bearing_adapted/test_01_final_result.jpg"
  echo "  v39_otherdata/generated/${city}/v39_output_bearing_adapted/test_02_final_result.jpg"
done
echo "Aggregate JSON: v39_otherdata/generated/bearing_multicity_summary.json"
echo "Aggregate CSV : v39_otherdata/generated/bearing_multicity_summary.csv"
echo "================================================================================"
