#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

CORE=(config.py robust_tracker.py visual_model.py visual_localizer.py data.py)

find_source_root() {
  python3 - "$ROOT" "${FORNX_DIR:-}" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
explicit = sys.argv[2].strip()
required = ["config.py", "robust_tracker.py", "visual_model.py", "visual_localizer.py", "data.py"]

search_roots = []
if explicit:
    p = Path(explicit).expanduser().resolve()
    if p.exists():
        search_roots.append(p)
else:
    for p in [repo / "forNX", repo / "fornx", repo / "ForNX", repo.parent / "forNX", Path("/yh/study/forNX"), Path("/yh/study/fornx"), repo]:
        if p.exists():
            search_roots.append(p.resolve())
    study = Path("/yh/study")
    if study.exists():
        search_roots.append(study)

candidates = {}
for base in search_roots:
    probes = []
    if base.is_dir():
        probes.append(base)
        try:
            probes.extend(p.parent for p in base.rglob("robust_tracker.py"))
        except (PermissionError, OSError):
            pass
    for d in probes:
        try:
            d = d.resolve()
        except OSError:
            continue
        if d in candidates:
            continue
        if not all((d / f).is_file() for f in required):
            continue
        try:
            tracker = (d / "robust_tracker.py").read_text(encoding="utf-8", errors="ignore")
            model = (d / "visual_model.py").read_text(encoding="utf-8", errors="ignore")
            cfg = ""
            for n in ("config.py", "config_base.py"):
                q = d / n
                if q.is_file():
                    cfg += q.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = 0
        low = str(d).lower()
        if "fornx" in low:
            score += 100
        if "ThreeFrameRouteStateGRU" in tracker or "ThreeFrameRouteStateGRU" in model:
            score += 20
        for h in ("correction_head", "variance_head", "motion_head", "heading_head"):
            if h in model:
                score += 5
        if "UAVSAT_EXPERIMENT_KALMAN" in cfg:
            score += 15
        if "teacher_meanshift_feedback" in tracker:
            score -= 1000
        candidates[d] = score

if not candidates:
    raise SystemExit(0)
ordered = sorted(candidates.items(), key=lambda x: (-x[1], str(x[0])))
best, score = ordered[0]
if score < 20:
    raise SystemExit(0)
print(best)
PY
}

SOURCE_ROOT="$(find_source_root)"
if [[ -z "$SOURCE_ROOT" || ! -d "$SOURCE_ROOT" ]]; then
  echo "ERROR: cannot find the real forNX/V36 project root." >&2
  echo "Need only these five files in the same code directory:" >&2
  printf '  - %s\n' "${CORE[@]}" >&2
  echo "Searches repo + /yh/study automatically; run_robust_tracker.sh is NOT required." >&2
  exit 2
fi

for f in "${CORE[@]}"; do
  [[ -f "$SOURCE_ROOT/$f" ]] || { echo "ERROR: source missing $SOURCE_ROOT/$f" >&2; exit 3; }
done
if grep -q 'teacher_meanshift_feedback' "$SOURCE_ROOT/robust_tracker.py"; then
  echo "ERROR: selected source is byTeacher, not original forNX V36: $SOURCE_ROOT" >&2
  exit 4
fi

DST="$ROOT/v36_GvsK"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Only archive after a valid source is found.
if [[ -e "$DST" ]]; then
  mv "$DST" "$ROOT/v36_GvsK_previous_${STAMP}"
fi
if [[ -e "$ROOT/v36-GvsK" ]]; then
  mv "$ROOT/v36-GvsK" "$ROOT/v36-GvsK_previous_${STAMP}"
fi

ORIG="$DST/original_forNX"
GONLY="$DST/G_only"
OUT="$DST/output"
mkdir -p "$ORIG" "$GONLY" "$OUT/original_full" "$OUT/G_only"

# Copy EVERYTHING in the actual code directory. No hand-picked source list.
cp -a "$SOURCE_ROOT/." "$ORIG/"
cp -a "$SOURCE_ROOT/." "$GONLY/"

# Original V36 commonly expects route_waypoints beside/in project root. If the
# uploaded forNX copy omitted them, use the repository's canonical waypoint files.
for tree in "$ORIG" "$GONLY"; do
  if [[ ! -d "$tree/route_waypoints" ]]; then
    [[ -d "$ROOT/route_waypoints" ]] || { echo "ERROR: route_waypoints missing in both forNX and repo" >&2; exit 5; }
    cp -a "$ROOT/route_waypoints" "$tree/route_waypoints"
  fi
  for f in "${CORE[@]}"; do
    [[ -f "$tree/$f" ]] || { echo "ERROR: copied tree missing $tree/$f" >&2; exit 6; }
  done
  for r in route_A route_B route_C; do
    [[ -f "$tree/route_waypoints/${r}_waypoints.json" ]] || { echo "ERROR: missing ${r}_waypoints.json" >&2; exit 7; }
  done
done

# G-only must use the original V36 built-in Kalman ablation switch. It can be
# declared in config.py or config_base.py. Do not rewrite model/tracker code.
if ! grep -R -q 'UAVSAT_EXPERIMENT_KALMAN' "$GONLY/config.py" "$GONLY/config_base.py" 2>/dev/null; then
  echo "ERROR: this forNX source has no UAVSAT_EXPERIMENT_KALMAN switch; refusing to invent a new architecture." >&2
  exit 8
fi

# Verify copied core source is identical before runtime override.
for f in "${CORE[@]}"; do
  cmp -s "$ORIG/$f" "$GONLY/$f" || { echo "ERROR: copies differ unexpectedly: $f" >&2; exit 9; }
done
[[ ! -f "$ORIG/config_base.py" || ! -f "$GONLY/config_base.py" ]] || cmp -s "$ORIG/config_base.py" "$GONLY/config_base.py" || { echo "ERROR: config_base copies differ" >&2; exit 10; }

cat > "$OUT/source_audit.txt" <<EOF
source_root=$SOURCE_ROOT
original_copy=$ORIG
G_only_copy=$GONLY
copy_method=cp -a entire source directory
core_files=${CORE[*]}
run_robust_tracker_sh_required=no
route_waypoints=source copy or repository fallback
teacher_meanshift_feedback=absent
G_only_core_code_difference=none
G_only_runtime_change=UAVSAT_EXPERIMENT_KALMAN=none
EOF

cat > "$DST/.prepared" <<EOF
source=$SOURCE_ROOT
prepared=$(date -Iseconds)
EOF

cat "$OUT/source_audit.txt"
echo "[OK] prepared original : $ORIG"
echo "[OK] prepared G-only   : $GONLY"
echo "[OK] output root       : $OUT"
