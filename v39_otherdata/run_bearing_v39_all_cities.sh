#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU="${GPU:-0}"
CITIES="${CITIES:-citya cityb cityc cityd}"
GEN_ROOT="${REPO_ROOT}/v39_otherdata/generated"
FINAL_DIR="${GEN_ROOT}/final_results"
LOG_DIR="${GEN_ROOT}/logs"

cd "${REPO_ROOT}"
mkdir -p "${GEN_ROOT}"
rm -rf "${FINAL_DIR}" "${LOG_DIR}"
mkdir -p "${FINAL_DIR}" "${LOG_DIR}"

echo "================================================================================"
echo "Bearing-v39 MULTI-CITY evaluation"
echo "Cities : ${CITIES}"
echo "GPU    : ${GPU} (sequential)"
echo "Rule   : a failed city is recorded, but DOES NOT prevent later cities from running"
echo "Per city: one auto-selected COMPLETE Route A -> 60 epochs -> 2 official tests"
echo "================================================================================"

SUCCESS_CITIES=""
FAILED_CITIES=""

for city in ${CITIES}; do
  case "${city}" in
    citya|cityb|cityc|cityd) ;;
    *) echo "Unsupported city: ${city}" >&2; FAILED_CITIES="${FAILED_CITIES} ${city}"; continue ;;
  esac

  echo ""
  echo "################################################################################"
  echo "# START ${city}"
  echo "################################################################################"

  if DATASET_ROOT="${DATASET_ROOT}" CITY="${city}" GPU="${GPU}" \
       bash v39_otherdata/run_bearing_v39_sequence_fixed.sh \
       2>&1 | tee "${LOG_DIR}/${city}.log"; then
    out="${GEN_ROOT}/${city}/v39_output_bearing_adapted"
    if [[ -s "${out}/test_01_final_result.jpg" && -s "${out}/test_02_final_result.jpg" && -s "${out}/bearing_paper_metrics.json" ]]; then
      cp "${out}/test_01_final_result.jpg" "${FINAL_DIR}/${city}_test_01_final_result.jpg"
      cp "${out}/test_02_final_result.jpg" "${FINAL_DIR}/${city}_test_02_final_result.jpg"
      cp "${out}/bearing_paper_metrics.json" "${FINAL_DIR}/${city}_paper_metrics.json"
      SUCCESS_CITIES="${SUCCESS_CITIES} ${city}"
      echo "# DONE ${city}: SUCCESS + 2 FINAL IMAGES"
    else
      FAILED_CITIES="${FAILED_CITIES} ${city}"
      echo "# DONE ${city}: FAILED output existence audit" >&2
    fi
  else
    rc=$?
    FAILED_CITIES="${FAILED_CITIES} ${city}"
    echo "# DONE ${city}: FAILED rc=${rc}; continuing to next city" >&2
  fi
  echo "################################################################################"
done

export SUCCESS_CITIES FAILED_CITIES
python3 - "${GEN_ROOT}" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
success = os.environ.get("SUCCESS_CITIES", "").split()
failed = os.environ.get("FAILED_CITIES", "").split()
status = {
    "requested_cities": ["citya", "cityb", "cityc", "cityd"],
    "successful_cities": success,
    "failed_cities": failed,
    "successful_city_count": len(success),
    "successful_test_route_count": 2 * len(success),
}
(root / "bearing_multicity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

summary_rows = []
paper_rows = []
raw_city_summaries = {}
raw_paper = {}
for city in success:
    out = root / city / "v39_output_bearing_adapted"
    summary = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
    paper = json.loads((out / "bearing_paper_metrics.json").read_text(encoding="utf-8"))
    raw_city_summaries[city] = summary
    raw_paper[city] = paper
    for route in ("test_01", "test_02"):
        s = summary[route]
        p = paper["routes"][route]
        summary_rows.append({
            "city": city, "route": route,
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
            "Waypoints": int(s["Waypoints"]),
        })
        paper_rows.append({
            "city": city, "route": route, "frames": int(p["frames"]),
            "Recall@1_derived_same_quadrant_pct": float(p["Recall@1_derived_same_quadrant_pct"]),
            "MLE_m": float(p["MLE_m"]), "MedLE_m": float(p["MedLE_m"]),
            "LSR@15_pct": float(p["LSR@15_pct"]),
            "HSR@15_pct": "N/A",
            "MHE_deg": "N/A", "MedHE_deg": "N/A",
            "SR@20_pct": "N/A", "SPL_pct": "N/A", "NE_m": "N/A",
            "comparison_note": "MLE/MedLE/LSR@15 direct; Recall@1 derived same-quadrant; heading/navigation N/A (different protocol)",
        })

if summary_rows:
    total_frames = sum(r["frames"] for r in paper_rows)
    weighted = lambda key: sum(float(r[key]) * r["frames"] for r in paper_rows) / total_frames
    aggregate = {
        "successful_cities": success,
        "failed_cities": failed,
        "total_test_routes": len(paper_rows),
        "total_frames": total_frames,
        "weighted_Recall@1_derived_same_quadrant_pct": weighted("Recall@1_derived_same_quadrant_pct"),
        "weighted_MLE_m": weighted("MLE_m"),
        "weighted_MedLE_route_average_m": sum(float(r["MedLE_m"]) for r in paper_rows) / len(paper_rows),
        "weighted_LSR@15_pct": weighted("LSR@15_pct"),
        "directly_comparable_to_Bearing_UAV": ["MLE_m", "MedLE_m", "LSR@15_pct"],
        "derived_same_criterion": ["Recall@1_derived_same_quadrant_pct"],
        "not_directly_comparable_current_protocol": ["HSR@15", "MHE", "MedHE", "SR@20", "SPL", "NE"],
        "routes": paper_rows,
        "raw_city_paper_metrics": raw_paper,
    }
    (root / "bearing_multicity_summary.json").write_text(
        json.dumps({"status": status, "routes": summary_rows, "raw_city_summaries": raw_city_summaries}, indent=2),
        encoding="utf-8",
    )
    with (root / "bearing_multicity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    (root / "bearing_paper_comparison_multicity.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (root / "bearing_paper_comparison_multicity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paper_rows[0].keys()))
        w.writeheader(); w.writerows(paper_rows)

print("[MULTICITY-STATUS]", json.dumps(status, indent=2))
if paper_rows:
    print(f"[MULTICITY] paper-comparison rows={len(paper_rows)}")
    print(f"[MULTICITY] flat final images={2 * len(success)} under {root / 'final_results'}")
PY

echo ""
echo "================================================================================"
echo "MULTI-CITY RUN FINISHED"
echo "Success:${SUCCESS_CITIES:- none}"
echo "Failed :${FAILED_CITIES:- none}"
echo "Flat result images: ${FINAL_DIR}/<city>_test_0X_final_result.jpg"
echo "Status JSON       : ${GEN_ROOT}/bearing_multicity_status.json"
echo "Raw summary CSV   : ${GEN_ROOT}/bearing_multicity_summary.csv"
echo "Paper compare CSV : ${GEN_ROOT}/bearing_paper_comparison_multicity.csv"
echo "Paper compare JSON: ${GEN_ROOT}/bearing_paper_comparison_multicity.json"
echo "Per-city logs     : ${LOG_DIR}/"
echo "================================================================================"

if [[ -n "${FAILED_CITIES// }" ]]; then
  echo "Some cities failed, but all requested cities were attempted. Check logs above." >&2
  exit 1
fi

echo "[MULTICITY-AUDIT] PASS: all four cities, eight official test routes, eight final images"
