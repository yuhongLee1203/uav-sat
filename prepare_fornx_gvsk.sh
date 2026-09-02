#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

REQ=(config.py robust_tracker.py visual_model.py visual_localizer.py data.py run_robust_tracker.sh)

# Find the executable project root first. Nothing is moved until this succeeds.
SOURCE_ROOT="$({
python3 - "$ROOT" "${FORNX_DIR:-}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
explicit = sys.argv[2].strip()
required = {"config.py", "robust_tracker.py", "visual_model.py", "visual_localizer.py", "data.py", "run_robust_tracker.sh"}

bases = []
if explicit:
    p = Path(explicit).expanduser().resolve()
    if p.exists():
        bases.append(p)
else:
    for p in root.rglob("*"):
        if p.is_dir() and "fornx" in p.name.lower():
            bases.append(p.resolve())

candidates, seen = [], set()
for base in bases:
    probes = [base]
    try:
        probes.extend(p.parent for p in base.rglob("config.py"))
    except PermissionError:
        pass
    for d in probes:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        if not all((d / name).is_file() for name in required):
            continue
        try:
            config = (d / "config.py").read_text(encoding="utf-8", errors="ignore")
            tracker = (d / "robust_tracker.py").read_text(encoding="utf-8", errors="ignore")
            model = (d / "visual_model.py").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = 0
        score += 8 if "V34ProtocolCompactGRUSoftMSModeVarianceForward3x6PolynomialKalman_v36" in config else 0
        score += 5 if "ThreeFrameRouteStateGRU" in tracker else 0
        score += 4 if "UAVSAT_EXPERIMENT_KALMAN" in config else 0
        for h in ("correction_head", "variance_head", "motion_head", "heading_head"):
            score += 1 if h in model else 0
        teacher = False
        for py in d.glob("*.py"):
            try:
                if "teacher_meanshift_feedback" in py.read_text(encoding="utf-8", errors="ignore"):
                    teacher = True
                    break
            except OSError:
                pass
        if teacher:
            score -= 100
        candidates.append((score, d))

if not candidates:
    sys.exit(0)
candidates.sort(key=lambda x: (-x[0], str(x[1])))
best_score = candidates[0][0]
best = [d for s, d in candidates if s == best_score]
if len(best) != 1 or best_score < 15:
    sys.exit(0)
print(best[0])
PY
} 2>/dev/null)"

if [[ -z "$SOURCE_ROOT" || ! -d "$SOURCE_ROOT" ]]; then
  echo "ERROR: could not locate a complete forNX V36 project root." >&2
  echo "Required in the SAME project-root directory:" >&2
  printf '  - %s\n' "${REQ[@]}" >&2
  echo "No existing GvsK folder was moved or deleted." >&2
  echo "If needed, specify it explicitly:" >&2
  echo "FORNX_DIR=/absolute/path/to/forNX bash prepare_fornx_gvsk.sh" >&2
  exit 2
fi

for f in "${REQ[@]}"; do
  [[ -f "$SOURCE_ROOT/$f" ]] || { echo "ERROR: source root missing $f" >&2; exit 3; }
done
for r in route_A route_B route_C; do
  [[ -f "$SOURCE_ROOT/route_waypoints/${r}_waypoints.json" ]] || {
    echo "ERROR: source root missing route_waypoints/${r}_waypoints.json" >&2
    echo "Resolved root: $SOURCE_ROOT" >&2
    echo "No existing GvsK folder was moved or deleted." >&2
    exit 4
  }
done
if grep -R -q 'teacher_meanshift_feedback' "$SOURCE_ROOT" --include='*.py'; then
  echo "ERROR: resolved forNX contains teacher_meanshift_feedback; this is the wrong byTeacher architecture." >&2
  exit 5
fi
if ! grep -q 'ThreeFrameRouteStateGRU' "$SOURCE_ROOT/robust_tracker.py"; then
  echo "ERROR: resolved forNX is not the expected ThreeFrameRouteStateGRU V36." >&2
  exit 6
fi
if ! grep -q 'UAVSAT_EXPERIMENT_KALMAN' "$SOURCE_ROOT/config.py"; then
  echo "ERROR: resolved forNX has no original V36 Kalman ablation switch." >&2
  exit 7
fi
for h in correction_head variance_head motion_head heading_head; do
  grep -q "$h" "$SOURCE_ROOT/visual_model.py" || { echo "ERROR: visual_model.py missing $h" >&2; exit 8; }
done

DST="$ROOT/v36_GvsK"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -f "$DST/.prepared_from_forNX_v2" && -d "$DST/original_forNX" && -d "$DST/G_only" ]]; then
  echo "[GvsK] already prepared correctly from: $SOURCE_ROOT"
  exit 0
fi

# Archive failed/old experiments only now, after the correct source is proven valid.
[[ ! -e "$DST" ]] || mv "$DST" "$ROOT/v36_GvsK_previous_wrong_${STAMP}"
[[ ! -e "$ROOT/v36-GvsK" ]] || mv "$ROOT/v36-GvsK" "$ROOT/v36-GvsK_previous_wrong_${STAMP}"

ORIG="$DST/original_forNX"
GONLY="$DST/G_only"
OUT="$DST/output"
mkdir -p "$ORIG" "$GONLY" "$OUT/original_full" "$OUT/G_only"

# Entire project root, including every support file/checkpoint/output already in forNX.
cp -a "$SOURCE_ROOT/." "$ORIG/"
cp -a "$SOURCE_ROOT/." "$GONLY/"

# Exact-copy verification before any G-only change.
if ! diff -qr "$SOURCE_ROOT" "$ORIG" > "$OUT/forNX_copy_diff.txt"; then
  echo "ERROR: original_forNX is not an exact copy of $SOURCE_ROOT" >&2
  cat "$OUT/forNX_copy_diff.txt" >&2
  exit 9
fi

# Verify the runners' actual working directories, not just the source directory.
for tree in "$ORIG" "$GONLY"; do
  for f in "${REQ[@]}"; do
    [[ -f "$tree/$f" ]] || { echo "ERROR: copied tree missing $tree/$f" >&2; exit 10; }
  done
  for r in route_A route_B route_C; do
    [[ -f "$tree/route_waypoints/${r}_waypoints.json" ]] || { echo "ERROR: copied tree missing $r waypoint file" >&2; exit 11; }
  done
done

# G-only uses the original V36's built-in no-Kalman branch. This is the ONLY code edit.
python3 - "$GONLY/config.py" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
pat = re.compile(r'(EXPERIMENT_KALMAN\s*=\s*os\.environ\.get\(\s*["\']UAVSAT_EXPERIMENT_KALMAN["\']\s*,\s*)["\']learned["\'](\s*\))', re.S)
s2, n = pat.subn(r'\1"none"\2', s, count=1)
if n != 1:
    raise SystemExit("ERROR: expected exactly one EXPERIMENT_KALMAN default='learned'")
p.write_text(s2, encoding="utf-8")
PY

# Prove that source code differs in config.py only. Generated/cache directories are ignored.
python3 - "$ORIG" "$GONLY" "$OUT/original_vs_Gonly_code_diff.txt" <<'PY'
from pathlib import Path
import hashlib, sys
A, B, out = map(Path, sys.argv[1:])
ignore = {"outputs", "output", "__pycache__", ".cache", ".git"}
def scan(root):
    result = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in ignore for part in rel.parts):
            continue
        result[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result
a, b = scan(A), scan(B)
changed = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
out.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")
if changed != ["config.py"]:
    raise SystemExit("ERROR: G_only code differs from original in: " + repr(changed))
PY

cat > "$OUT/source_audit.txt" <<EOF
resolved_forNX_root=$SOURCE_ROOT
original_copy=$ORIG
G_only_copy=$GONLY
exact_full_directory_copy_before_patch=yes
core_files=${REQ[*]}
route_waypoints=A,B,C present
teacher_meanshift_feedback=absent
ThreeFrameRouteStateGRU=present
G_only_code_diff=config.py only
G_only_change=EXPERIMENT_KALMAN learned -> none
EOF
cat > "$DST/.prepared_from_forNX_v2" <<EOF
source=$SOURCE_ROOT
prepared_at=$(date -Iseconds)
EOF

cat "$OUT/source_audit.txt"
echo "[GvsK] original : $ORIG"
echo "[GvsK] G-only   : $GONLY"
echo "[GvsK] outputs  : $OUT"
