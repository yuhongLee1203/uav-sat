#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# -----------------------------------------------------------------------------
# 1) Locate the REAL forNX project root BEFORE touching any existing GvsK folder.
#    A valid project root must contain the complete executable V36 source set.
# -----------------------------------------------------------------------------
REQ=(config.py robust_tracker.py visual_model.py visual_localizer.py data.py run_robust_tracker.sh)

SOURCE_ROOT="$({
python3 - "$ROOT" "${FORNX_DIR:-}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
explicit = sys.argv[2].strip()
required = {
    "config.py",
    "robust_tracker.py",
    "visual_model.py",
    "visual_localizer.py",
    "data.py",
    "run_robust_tracker.sh",
}

bases = []
if explicit:
    p = Path(explicit).expanduser().resolve()
    if p.exists():
        bases.append(p)
else:
    # Search every directory whose name is forNX/fornx anywhere near the repo.
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "fornx":
            bases.append(p.resolve())

candidates = []
seen = set()
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
        # This is the architecture-changing byTeacher path that must NOT exist.
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
    print("", end="")
    sys.exit(0)

candidates.sort(key=lambda x: (-x[0], str(x[1])))
best_score = candidates[0][0]
best = [d for s, d in candidates if s == best_score]
if len(best) != 1 or best_score < 15:
    print("", end="")
    sys.exit(0)
print(best[0])
PY
} 2>/dev/null)"

if [[ -z "$SOURCE_ROOT" || ! -d "$SOURCE_ROOT" ]]; then
  echo "ERROR: could not locate a complete forNX V36 project root." >&2
  echo "A valid root must contain ALL of:" >&2
  printf '  - %s\n' "${REQ[@]}" >&2
  echo >&2
  echo "If your uploaded folder is nested, run with:" >&2
  echo "  FORNX_DIR=/yh/study/uav-sat/<path-to-forNX> bash prepare_fornx_gvsk.sh" >&2
  echo >&2
  echo "Nothing was moved or deleted." >&2
  exit 2
fi

for f in "${REQ[@]}"; do
  [[ -f "$SOURCE_ROOT/$f" ]] || { echo "ERROR: source root missing $f" >&2; exit 3; }
done

# Route geometry is required by this V36 config. Fail BEFORE copying if absent.
for r in route_A route_B route_C; do
  if [[ ! -f "$SOURCE_ROOT/route_waypoints/${r}_waypoints.json" ]]; then
    echo "ERROR: correct forNX root is missing route_waypoints/${r}_waypoints.json" >&2
    echo "Resolved source root: $SOURCE_ROOT" >&2
    echo "Nothing was moved or deleted." >&2
    exit 4
  fi
done

if grep -R -q 'teacher_meanshift_feedback' "$SOURCE_ROOT" --include='*.py'; then
  echo "ERROR: resolved forNX contains teacher_meanshift_feedback; refusing to use the wrong byTeacher architecture." >&2
  echo "Resolved source root: $SOURCE_ROOT" >&2
  exit 5
fi
if ! grep -q 'ThreeFrameRouteStateGRU' "$SOURCE_ROOT/robust_tracker.py"; then
  echo "ERROR: resolved forNX is not the expected ThreeFrameRouteStateGRU V36." >&2
  exit 6
fi
if ! grep -q 'UAVSAT_EXPERIMENT_KALMAN' "$SOURCE_ROOT/config.py"; then
  echo "ERROR: resolved forNX does not expose the original V36 Kalman ablation switch." >&2
  exit 7
fi
for h in correction_head variance_head motion_head heading_head; do
  if ! grep -q "$h" "$SOURCE_ROOT/visual_model.py"; then
    echo "ERROR: resolved forNX visual_model.py is missing $h" >&2
    exit 8
  fi
done

# -----------------------------------------------------------------------------
# 2) Only AFTER source validation, archive old failed GvsK folders.
# -----------------------------------------------------------------------------
DST="$ROOT/v36_GvsK"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Idempotent successful preparation: leave it intact.
if [[ -f "$DST/.prepared_from_forNX_v2" && -d "$DST/original_forNX" && -d "$DST/G_only" ]]; then
  echo "[GvsK] already prepared correctly from: $SOURCE_ROOT"
  exit 0
fi

if [[ -e "$DST" ]]; then
  mv "$DST" "$ROOT/v36_GvsK_previous_wrong_${STAMP}"
fi
if [[ -e "$ROOT/v36-GvsK" ]]; then
  mv "$ROOT/v36-GvsK" "$ROOT/v36-GvsK_previous_wrong_${STAMP}"
fi

ORIG="$DST/original_forNX"
GONLY="$DST/G_only"
OUT="$DST/output"
mkdir -p "$ORIG" "$GONLY" "$OUT/original_full" "$OUT/G_only"

# Copy the COMPLETE resolved forNX project root, not selected files.
cp -a "$SOURCE_ROOT/." "$ORIG/"
cp -a "$SOURCE_ROOT/." "$GONLY/"

# Byte-for-byte directory audit immediately after copy.
if ! diff -qr "$SOURCE_ROOT" "$ORIG" > "$OUT/forNX_copy_diff.txt"; then
  echo "ERROR: original_forNX is not an exact directory copy of $SOURCE_ROOT" >&2
  cat "$OUT/forNX_copy_diff.txt" >&2
  exit 9
fi

# Re-check the files INSIDE the copied tree, which is exactly where the runners cd.
for tree in "$ORIG" "$GONLY"; do
  for f in "${REQ[@]}"; do
    [[ -f "$tree/$f" ]] || { echo "ERROR: copied tree missing $tree/$f" >&2; exit 10; }
  done
  for r in route_A route_B route_C; do
    [[ -f "$tree/route_waypoints/${r}_waypoints.json" ]] || {
      echo "ERROR: copied tree missing route_waypoints/${r}_waypoints.json" >&2; exit 11;
    }
  done
done

# -----------------------------------------------------------------------------
# 3) G-only: modify ONE existing V36 default only: learned Kalman -> none.
#    Do NOT rewrite robust_tracker.py / visual_model.py / visual_localizer.py.
# -----------------------------------------------------------------------------
python3 - "$GONLY/config.py" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
pat = re.compile(r'(EXPERIMENT_KALMAN\s*=\s*os\.environ\.get\(\s*["\']UAVSAT_EXPERIMENT_KALMAN["\']\s*,\s*)["\']learned["\'](\s*\))', re.S)
s2, n = pat.subn(r'\1"none"\2', s, count=1)
if n != 1:
    raise SystemExit("ERROR: expected exactly one EXPERIMENT_KALMAN default='learned' in G_only/config.py")
p.write_text(s2, encoding="utf-8")
PY

# Ignore existing generated outputs/checkpoints when proving SOURCE-code equality;
# code files must differ in config.py only.
python3 - "$ORIG" "$GONLY" "$OUT/original_vs_Gonly_code_diff.txt" <<'PY'
from pathlib import Path
import hashlib, sys
A, B, out = map(Path, sys.argv[1:])
ignore_dirs = {"outputs", "output", "__pycache__", ".cache", ".git"}

def files(root):
    m = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in ignore_dirs for part in rel.parts):
            continue
        m[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return m

a, b = files(A), files(B)
keys = sorted(set(a) | set(b))
changed = [k for k in keys if a.get(k) != b.get(k)]
out.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")
if changed != ["config.py"]:
    raise SystemExit("ERROR: G_only differs from original_forNX in files other than config.py: " + repr(changed))
PY

cat > "$OUT/source_audit.txt" <<EOF
resolved_forNX_root=$SOURCE_ROOT
original_copy=$ORIG
G_only_copy=$GONLY
exact_copy_before_Gonly_patch=yes
required_core_files=${REQ[*]}
route_waypoints=route_A,route_B,route_C present
teacher_meanshift_feedback=absent
ThreeFrameRouteStateGRU=present
G_only_only_code_difference=config.py
G_only_change=EXPERIMENT_KALMAN learned -> none
EOF

cat > "$DST/.prepared_from_forNX_v2" <<EOF
source=$SOURCE_ROOT
prepared_at=$(date -Iseconds)
EOF

cat "$OUT/source_audit.txt"
echo "[GvsK] preparation complete."
echo "[GvsK] original: $ORIG"
echo "[GvsK] G-only  : $GONLY"
echo "[GvsK] output  : $OUT"
