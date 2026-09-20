#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
SUITE="${1:-}"
if [[ -z "${SUITE}" ]]; then
  SUITE="$(find "${ROOT}/v39_otherdata" -maxdepth 1 -type d -name 'formal_bearing_v5_smooth_*' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
[[ -n "${SUITE}" && -d "${SUITE}" ]] || { echo "ERROR: smooth suite not found" >&2; exit 2; }
SUITE="$(readlink -f "${SUITE}")"
OUT="${SUITE}/paper_benchmark"
mkdir -p "${OUT}"
echo "[PAPER BENCHMARK] suite=${SUITE}"
python3 -u v39_otherdata/bearing_paper_metrics.py --suite-root "${SUITE}" --output-dir "${OUT}"
for f in bearing_paper_metrics.json table_route_metrics.csv table_city_metrics.csv table_literature_comparison.csv PAPER_TABLES.md; do
  [[ -s "${OUT}/${f}" ]] || { echo "ERROR missing ${OUT}/${f}" >&2; exit 3; }
done
cat > "${OUT}/README_PROTOCOL.txt" <<'EOF'
Bearing-UAV paper-aligned outputs

Directly computed from raw predictions:
  MLE, MedLE, P90/P95/P99, LSR@5/10/15/20,
  MHE, MedHE, HSR@15,
  latency/FPS, JumpRate, MaxFinalStep.

Important fairness constraints:
  1. Bearing-UAV Recall@1 is a four-adjacent-RST retrieval decision.
     Forward-18 navigation top-1 is NOT substituted; ours stays N/A.
  2. SR@20/SPL/NE in this package are route-replay diagnostics.
     They are NOT claimed as Bearing-Naver closed-loop navigation results.
  3. Use table_literature_comparison.csv for the literature table, preserving N/A fields.
EOF
echo "============================================================"
echo "DONE"
echo "Paper tables: ${OUT}/PAPER_TABLES.md"
echo "Comparison : ${OUT}/table_literature_comparison.csv"
echo "Per-city   : ${OUT}/table_city_metrics.csv"
echo "Per-route  : ${OUT}/table_route_metrics.csv"
echo "JSON       : ${OUT}/bearing_paper_metrics.json"
echo "============================================================"
