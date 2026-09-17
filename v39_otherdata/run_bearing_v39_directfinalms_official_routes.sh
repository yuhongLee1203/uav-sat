#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"

case "${CITY}" in
  citya|cityb|cityc|cityd) ;;
  *) echo "Unsupported CITY=${CITY}" >&2; exit 2 ;;
esac

export CITY GPU DATASET_ROOT

echo "================================================================================"
echo "v39_DirectFinalMS -> Bearing-UAV official navigation routes"
echo "Method : Weighted Centroid -> 3-frame GRU -> fixed Kalman -> one final 6x6 MS"
echo "Protocol: original selected v39 controlled_gt_jitter / predefined frame reference"
echo "Plot   : official waypoint GT + raw final_x/final_y prediction"
echo "City   : ${CITY}"
echo "================================================================================"

bash v39_otherdata/run_bearing_v39_sequence_fixed.sh

python3 - "v39_otherdata/generated/${CITY}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
exp = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
out = root / "v39_output_bearing_adapted"
summary = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
print("[DIRECTFINALMS-AUDIT] official test routes:")
for route in ("test_01", "test_02"):
    print(f"  {route}: {exp['test_route_source'][route]}")
    s = summary[route]
    print(f"    MLE={s['MLE_m']:.3f}m MedLE={s['MedLE_m']:.3f}m P90={s['P90_m']:.3f}m LSR@15={s['LSR@15_pct']:.2f}%")
print("[DIRECTFINALMS-AUDIT] PASS")
PY
