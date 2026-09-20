#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SUITE="${1:-}"
if [[ -z "${SUITE}" ]]; then
  SUITE="$(find "${ROOT}/v39_otherdata" -maxdepth 1 -type d -name 'formal_bearing_v5_smooth_*' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
[[ -n "${SUITE}" && -d "${SUITE}" ]] || { echo "ERROR: no completed smooth suite found" >&2; exit 2; }
SUITE="$(readlink -f "${SUITE}")"

echo "[UPLOAD-ONLY] suite=${SUITE}"

python3 - "${SUITE}" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
cities = ['citya','cityb','cityc','cityd']
rows=[]
for city in cities:
    full=root/city/'variants'/'full'
    summary=full/'bearing_v39_summary.json'
    if not summary.is_file() or summary.stat().st_size == 0:
        raise SystemExit(f'ERROR missing summary: {summary}')
    data=json.loads(summary.read_text())
    if len(data) != 2:
        raise SystemExit(f'ERROR {city} summary expected 2 held-out routes, got {list(data)}')
    rows.extend((city,k,v) for k,v in data.items())
    for name in ['nav50_result.jpg','nav51_result.jpg','plot_source_audit.json']:
        p=full/'formal_figures'/name
        if not p.is_file() or p.stat().st_size == 0:
            raise SystemExit(f'ERROR missing figure/audit: {p}')
agg=root/'formal_allcities_results.json'
if not agg.is_file():
    out={'run_type':'formal_smooth_v1','cities':{},'macro_average_over_8_held_out_routes':{},'held_out_route_count':len(rows),'figure_count':8}
    for city in cities:
        p=root/city/'variants'/'full'/'bearing_v39_summary.json'
        out['cities'][city]=json.loads(p.read_text())
    for key in ['MLE_m','P90_m','LSR@3_pct','LSR@5_pct','LSR@10_pct','LSR@15_pct']:
        vals=[float(m[key]) for _,_,m in rows if key in m]
        if vals: out['macro_average_over_8_held_out_routes'][key]=sum(vals)/len(vals)
    agg.write_text(json.dumps(out,indent=2))
print(f'[VERIFY] routes={len(rows)} figures=8 PASS')
print(agg.read_text())
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="paper_results/formal_bearing_v5_smooth_${STAMP}"
TMP="$(mktemp -d "${ROOT%/*}/uav-sat-smooth-upload-XXXXXX")"
BR="smooth-upload-${STAMP}-$$"
cleanup(){
  git -C "${ROOT}" worktree remove --force "${TMP}" >/dev/null 2>&1 || true
  git -C "${ROOT}" branch -D "${BR}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git fetch origin bearing-v5-formal-smooth-v1
git worktree add -b "${BR}" "${TMP}" origin/bearing-v5-formal-smooth-v1
mkdir -p "${TMP}/${DEST}"
cp "${SUITE}/formal_allcities_results.json" "${TMP}/${DEST}/"
for city in citya cityb cityc cityd; do
  mkdir -p "${TMP}/${DEST}/${city}"
  cp "${SUITE}/${city}/variants/full/bearing_v39_summary.json" "${TMP}/${DEST}/${city}/"
  cp "${SUITE}/${city}/variants/full/"*_frames.csv "${TMP}/${DEST}/${city}/" 2>/dev/null || true
  cp -r "${SUITE}/${city}/variants/full/formal_figures" "${TMP}/${DEST}/${city}/"
  cp "${SUITE}/${city}/train_frames3/kalman_calibration.json" "${TMP}/${DEST}/${city}/" 2>/dev/null || true
done

git -C "${TMP}" add "${DEST}"
git -C "${TMP}" -c user.name='OpenAI Results Uploader' -c user.email='results@local' commit -m "Upload completed Bearing V5 smooth results ${STAMP}"
git -C "${TMP}" push origin "HEAD:bearing-v5-formal-smooth-v1"

echo "[UPLOAD-ONLY DONE] ${DEST}"
