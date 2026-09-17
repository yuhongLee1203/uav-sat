#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CPU_THREADS="${CPU_THREADS:-2}"
GEN="$ROOT/v39_otherdata/generated"
OUT="$GEN/official_routes_only"
LOG="$GEN/official_routes_only_logs"
GPU_IDS=(0 5 6)

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

for g in "${GPU_IDS[@]}"; do
  nvidia-smi -i "$g" >/dev/null 2>&1 || { echo "GPU $g unavailable" >&2; exit 2; }
done

# ---------------------------------------------------------------------------
# CLEANUP: remove only obsolete experiment/runtime artifacts from earlier
# baseline/custom comparison attempts. Source code and Bearing_UAV_90K dataset
# are never touched.
# ---------------------------------------------------------------------------
echo "[CLEANUP] removing obsolete baseline/custom-route experiment artifacts"
rm -rf \
  "$ROOT/v39_otherdata/external" \
  "$GEN/full_benchmark_cache" \
  "$GEN/full_benchmark_baselines" \
  "$GEN/corrected_full_comparison" \
  "$GEN/corrected_full_logs" \
  "$GEN/native_m2t_cache" \
  "$GEN/native_m2t_baselines" \
  "$GEN/native_four_rst_cache" \
  "$GEN/native_four_rst_baselines" \
  "$GEN/native_four_rst_comparison" \
  "$GEN/native_four_rst_logs" \
  "$GEN/native_comparison" \
  "$GEN/native_comparison_logs" \
  "$GEN/official_bearinguav_same_routes" \
  "$GEN/official_bearinguav_corrected_routes" \
  "$GEN/route_baseline_cache" \
  "$GEN/paper_bundle" \
  "$GEN/final_results" \
  "$GEN/logs" \
  "$GEN/bearing_multicity_status.json" \
  "$GEN/bearing_multicity_summary.json" \
  "$GEN/bearing_multicity_summary.csv" \
  "$GEN/bearing_paper_comparison_multicity.json" \
  "$GEN/bearing_paper_comparison_multicity.csv" \
  "$GEN/official_routes_only" \
  "$GEN/official_routes_only_logs" \
  "$GEN/citya" "$GEN/cityb" "$GEN/cityc" "$GEN/cityd"

mkdir -p "$OUT" "$LOG"

# ---------------------------------------------------------------------------
# HARD AUDIT: tests must be the eight official Bearing-UAV waypoint files.
# Training Route A remains a separate training-only route because v39 requires
# training; it is NOT an evaluation route and is never reported as a test path.
# ---------------------------------------------------------------------------
python3 - <<'PY'
from v39_otherdata.bearing_multicity_routes import OFFICIAL_TEST_ROUTE_SOURCE, OFFICIAL_TEST_ROUTES
expected = {
    'citya': ('wps34bc_50.json','wps34bc_51.json'),
    'cityb': ('wps36bc_50.json','wps36bc_51.json'),
    'cityc': ('wps37bc_50.json','wps37bc_51.json'),
    'cityd': ('wps38bc_50.json','wps38bc_51.json'),
}
for city,(a,b) in expected.items():
    src = OFFICIAL_TEST_ROUTE_SOURCE[city]
    assert a in src['test_01'], (city, src)
    assert b in src['test_02'], (city, src)
    assert len(OFFICIAL_TEST_ROUTES[city]['test_01']) >= 10
    assert len(OFFICIAL_TEST_ROUTES[city]['test_02']) >= 10
print('[OFFICIAL-ROUTE-AUDIT] PASS: 4 cities x 2 official Bearing-UAV routes')
PY

run_city() {
  local city="$1" gpu="$2"
  echo "[OFFICIAL-RUN] START $city on GPU$gpu"
  DATASET_ROOT="$DATASET_ROOT" CITY="$city" GPU="$gpu" \
    bash v39_otherdata/run_bearing_v39_sequence_fixed.sh \
    > >(tee "$LOG/${city}.log") 2>&1
  echo "[OFFICIAL-RUN] DONE  $city on GPU$gpu"
}

# Maximize the three requested GPUs without sharing one GPU between jobs.
run_city citya 0 & p0=$!
run_city cityb 5 & p5=$!
run_city cityc 6 & p6=$!

fail=0
wait "$p0" || fail=1
wait "$p5" || fail=1
wait "$p6" || fail=1
[[ "$fail" -eq 0 ]] || { echo "One of citya/cityb/cityc failed; inspect $LOG" >&2; exit 10; }

# Fourth city uses GPU0 after the first wave completes.
run_city cityd 0

# ---------------------------------------------------------------------------
# Collect exactly the eight official-route outputs and one compact summary.
# ---------------------------------------------------------------------------
python3 - "$GEN" "$OUT" <<'PY'
import csv, json, shutil, sys
from pathlib import Path

gen = Path(sys.argv[1]); out = Path(sys.argv[2])
rows=[]
for city in ('citya','cityb','cityc','cityd'):
    root=gen/city/'v39_output_bearing_adapted'
    summary=json.loads((root/'bearing_v39_summary.json').read_text())
    metrics=json.loads((root/'bearing_paper_metrics.json').read_text())
    exp=json.loads((gen/city/'experiment.json').read_text())
    for route in ('test_01','test_02'):
        jpg=root/f'{route}_final_result.jpg'
        if not jpg.exists(): raise SystemExit(f'missing final image: {jpg}')
        dst=out/f'{city}_{route}_final_result.jpg'; shutil.copy2(jpg,dst)
        s=summary[route]
        rows.append({
            'city':city,
            'route':route,
            'official_route_source':exp.get('test_route_source',{}).get(route,''),
            'frames':int(metrics['routes'][route]['frames']),
            'MLE_m':float(s['MLE_m']),
            'MedLE_m':float(s['MedLE_m']),
            'P90_m':float(s['P90_m']),
            'P95_m':float(s['P95_m']),
            'P99_m':float(s['P99_m']),
            'LSR@5_pct':float(s['LSR@5_pct']),
            'LSR@10_pct':float(s['LSR@10_pct']),
            'LSR@15_pct':float(s['LSR@15_pct']),
            'LSR@20_pct':float(s['LSR@20_pct']),
            'JumpRate_pct':float(s['JumpRate_pct']),
        })

with (out/'official_routes_results.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

total=sum(r['frames'] for r in rows)
weighted=lambda k: sum(r[k]*r['frames'] for r in rows)/total
payload={
    'protocol':'Ours v39 evaluated only on the eight official Bearing-UAV navigation waypoint routes',
    'cities':4,'test_routes':8,'frames':total,
    'weighted_MLE_m':weighted('MLE_m'),
    'weighted_LSR@15_pct':weighted('LSR@15_pct'),
    'routes':rows,
    'note':'Route A is training-only. Only official Bearing-UAV test_01/test_02 routes are reported.',
}
(out/'official_routes_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

imgs=list(out.glob('*_final_result.jpg'))
if len(imgs)!=8: raise SystemExit(f'final image audit failed: {len(imgs)}/8')
print(f"[OFFICIAL-RESULT-AUDIT] PASS: 8/8 figures, weighted MLE={payload['weighted_MLE_m']:.3f}m, LSR15={payload['weighted_LSR@15_pct']:.2f}%")
PY

echo "================================================================================"
echo "OFFICIAL BEARING-UAV ROUTES ONLY: PASS"
echo "Results : $OUT/official_routes_results.csv"
echo "Summary : $OUT/official_routes_results.json"
echo "Figures : $OUT/*_final_result.jpg   (8 images)"
echo "Logs    : $LOG/"
echo "================================================================================"
