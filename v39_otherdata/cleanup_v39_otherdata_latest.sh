#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V39_ROOT="${REPO_ROOT}/v39_otherdata"
MODE="${1:---dry-run}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${REPO_ROOT%/*}/$(basename "${REPO_ROOT}")_v39_legacy_backup_${TS}"
PAPER_CITIES=(citya cityb cityc cityd)

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--apply" ]]; then
  echo "usage: bash v39_otherdata/cleanup_v39_otherdata_latest.sh [--dry-run|--apply]" >&2
  exit 2
fi

cd "${REPO_ROOT}"

keep_top_file() {
  case "$1" in
    README.md|\
    data.py|\
    bearing_prepare.py|\
    bearing_prepare_sequence_v3.py|\
    bearing_multicity_routes.py|\
    bearing_prepare_multicity.py|\
    bearing_runner.py|\
    bearing_runner_exact_v39.py|\
    bearing_runner_multicity_v39.py|\
    bearing_plot_final_vs_gt.py|\
    bearing_paper_metrics.py|\
    run_bearing_v39_sequence_fixed.sh|\
    run_bearing_v39_directfinalms_official_routes.sh|\
    run_bearing_all4_cities.sh|\
    resume_bearing_city_from_prepared.sh|\
    rerender_bearing_waypoint_gt.sh|\
    cleanup_v39_otherdata_latest.sh) return 0 ;;
    *) return 1 ;;
  esac
}

keep_generated_entry() {
  case "$1" in
    citya|cityb|cityc|cityd|paper_all4_summary.json|paper_all4_summary.csv) return 0 ;;
    *) return 1 ;;
  esac
}

keep_city_entry() {
  case "$1" in
    bearing_satellite.json|\
    experiment.json|\
    route_plan_full_satellite.jpg|\
    routes|\
    training_route_selection.json|\
    turn_diversity_audit.json|\
    v39_output_bearing_adapted) return 0 ;;
    *) return 1 ;;
  esac
}

keep_output_entry() {
  case "$1" in
    bearing_paper_metrics.csv|\
    bearing_paper_metrics.json|\
    bearing_v39_summary.json|\
    final_quality_audit.json|\
    v39_bearing_training_audit.json|\
    test_01_final_result.jpg|\
    test_02_final_result.jpg|\
    test_01_*_frames.csv|\
    test_02_*_frames.csv|\
    paper_figures_waypoint_gt) return 0 ;;
    *) return 1 ;;
  esac
}

show_or_move() {
  local src="$1"
  local rel="$2"
  [[ -e "${src}" || -L "${src}" ]] || return 0
  if [[ "${MODE}" == "--dry-run" ]]; then
    echo "[WOULD-ARCHIVE] ${rel}"
    return 0
  fi
  local dst="${BACKUP_ROOT}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  mv "${src}" "${dst}"
  echo "[ARCHIVED] ${rel}"
}

if [[ "${MODE}" == "--apply" ]]; then
  mkdir -p "${BACKUP_ROOT}"
  echo "[BACKUP] ${BACKUP_ROOT}"
fi

echo "================================================================================"
echo "Cleaning v39_otherdata to the four-city paper workflow"
echo "Mode : ${MODE}"
echo "Keep : canonical DirectFinalMS workflow + citya/b/c/d paper results"
echo "================================================================================"

# 1) Top-level: keep only the canonical workflow files + generated.
while IFS= read -r -d '' p; do
  name="$(basename "${p}")"
  if [[ -d "${p}" && ! -L "${p}" ]]; then
    [[ "${name}" == "generated" ]] && continue
    show_or_move "${p}" "v39_otherdata/${name}"
  else
    keep_top_file "${name}" && continue
    show_or_move "${p}" "v39_otherdata/${name}"
  fi
done < <(find "${V39_ROOT}" -mindepth 1 -maxdepth 1 -print0 | sort -z)

# 2) generated/: preserve all four paper cities and all-four aggregate summaries.
GEN="${V39_ROOT}/generated"
if [[ -d "${GEN}" ]]; then
  while IFS= read -r -d '' p; do
    name="$(basename "${p}")"
    keep_generated_entry "${name}" && continue
    show_or_move "${p}" "v39_otherdata/generated/${name}"
  done < <(find "${GEN}" -mindepth 1 -maxdepth 1 -print0 | sort -z)
fi

# 3) For every existing paper city, keep only rerun/rerender/audit data and the canonical output.
for CITY in "${PAPER_CITIES[@]}"; do
  CITY_ROOT="${GEN}/${CITY}"
  [[ -d "${CITY_ROOT}" ]] || continue

  OUT="${CITY_ROOT}/v39_output_bearing_adapted"
  PAPER_DIR="${OUT}/paper_figures_waypoint_gt"

  if [[ -d "${OUT}" ]]; then
    for f in \
      "${OUT}/bearing_v39_summary.json" \
      "${OUT}/bearing_paper_metrics.json" \
      "${OUT}/test_01_final_result.jpg" \
      "${OUT}/test_02_final_result.jpg"; do
      [[ -s "${f}" ]] || { echo "ERROR: ${CITY} canonical result incomplete: ${f}" >&2; exit 3; }
    done

    if [[ ! -s "${PAPER_DIR}/test_01_waypoint_gt_green.jpg" || ! -s "${PAPER_DIR}/test_02_waypoint_gt_green.jpg" ]]; then
      if [[ "${MODE}" == "--dry-run" ]]; then
        echo "[INFO] ${CITY}: waypoint-GT figures missing; --apply will render them first."
      else
        CITY="${CITY}" bash v39_otherdata/rerender_bearing_waypoint_gt.sh
      fi
    fi
  fi

  while IFS= read -r -d '' p; do
    name="$(basename "${p}")"
    keep_city_entry "${name}" && continue
    show_or_move "${p}" "v39_otherdata/generated/${CITY}/${name}"
  done < <(find "${CITY_ROOT}" -mindepth 1 -maxdepth 1 -print0 | sort -z)

  if [[ -d "${OUT}" ]]; then
    while IFS= read -r -d '' p; do
      name="$(basename "${p}")"
      keep_output_entry "${name}" && continue
      show_or_move "${p}" "v39_otherdata/generated/${CITY}/v39_output_bearing_adapted/${name}"
    done < <(find "${OUT}" -mindepth 1 -maxdepth 1 -print0 | sort -z)
  fi
done

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "================================================================================"
  echo "DRY RUN ONLY: nothing was moved."
  echo "All existing citya/b/c/d result packages are preserved."
  echo "================================================================================"
  exit 0
fi

python3 -m py_compile \
  v39_otherdata/data.py \
  v39_otherdata/bearing_prepare.py \
  v39_otherdata/bearing_prepare_sequence_v3.py \
  v39_otherdata/bearing_multicity_routes.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_otherdata/bearing_runner.py \
  v39_otherdata/bearing_runner_exact_v39.py \
  v39_otherdata/bearing_runner_multicity_v39.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_otherdata/bearing_paper_metrics.py

echo "================================================================================"
echo "CLEANUP COMPLETE"
echo "Backup of archived legacy files: ${BACKUP_ROOT}"
echo "Preserved paper cities: citya cityb cityc cityd (when present)"
echo "Git changes to review:"
git status --short -- v39_otherdata || true
echo "================================================================================"
