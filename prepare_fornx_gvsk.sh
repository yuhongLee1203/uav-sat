#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SRC="${FORNX_DIR:-$ROOT/forNX}"
if [[ ! -d "$SRC" ]]; then
  found="$(find "$ROOT" -maxdepth 1 -mindepth 1 -type d -iname 'fornx' -print -quit || true)"
  if [[ -n "$found" ]]; then SRC="$found"; fi
fi
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: cannot find the correct local forNX directory." >&2
  echo "Expected: $ROOT/forNX  (or set FORNX_DIR=/absolute/path/to/forNX)" >&2
  exit 2
fi

DST="$ROOT/v36-GvsK"
BACKUP="$ROOT/v36-GvsK_wrong_byTeacher_backup"
ORIG="$DST/original_forNX"
GONLY="$DST/G_only"
OUT="$DST/output"
MARK="$DST/.prepared_from_forNX"

# Idempotent: do not overwrite a preparation that was already created from the
# user's local forNX source.  The runners may call this script independently.
if [[ -f "$MARK" && -d "$ORIG" && -d "$GONLY" ]]; then
  echo "[GvsK] already prepared from local forNX: $SRC"
  exit 0
fi

# Preserve the previous wrong byTeacher experiment before replacing v36-GvsK.
if [[ -d "$DST" && ! -d "$BACKUP" ]]; then
  echo "[GvsK] archiving previous wrong byTeacher folder -> $BACKUP"
  mv "$DST" "$BACKUP"
elif [[ -d "$DST" ]]; then
  echo "[GvsK] previous backup already exists: $BACKUP"
  rm -rf "$DST"
fi

mkdir -p "$ORIG" "$GONLY" "$OUT/original_full" "$OUT/G_only"
cp -a "$SRC/." "$ORIG/"
cp -a "$SRC/." "$GONLY/"

# Verify that original_forNX is byte-for-byte identical to the local source
# immediately after the copy.  This is the key safeguard against guessing an old
# Git commit again.
if ! diff -qr "$SRC" "$ORIG" > "$OUT/forNX_copy_diff.txt"; then
  echo "ERROR: original_forNX is not an exact copy of $SRC" >&2
  cat "$OUT/forNX_copy_diff.txt" >&2
  exit 3
fi

# Audit the key architecture properties and the reason the previous experiment
# was invalid.  These checks do not rewrite the original copy.
AUDIT="$OUT/source_audit.txt"
{
  echo "forNX source              : $SRC"
  echo "exact source copy         : PASS (diff -qr is empty)"
  echo "original copy             : $ORIG"
  echo "G-only copy               : $GONLY"
  echo "wrong byTeacher backup    : $BACKUP"
  echo
  echo "=== correct forNX key evidence ==="
  grep -n -m1 'ARCHITECTURE_NAME' "$ORIG/config.py" 2>/dev/null || true
  grep -n -m1 'ThreeFrameRouteStateGRU' "$ORIG/robust_tracker.py" 2>/dev/null || true
  for h in correction_head variance_head motion_head heading_head; do
    grep -n -m1 "$h" "$ORIG/visual_model.py" 2>/dev/null || true
  done
  grep -n -m1 'UAVSAT_EXPERIMENT_KALMAN' "$ORIG/config.py" 2>/dev/null || true
  echo
  echo "=== teacher-feedback check ==="
  if grep -R -n -m1 'teacher_meanshift_feedback' "$ORIG" --include='*.py' 2>/dev/null; then
    echo "WARNING: teacher_meanshift_feedback exists in forNX; inspect before using."
  else
    echo "correct forNX: no teacher_meanshift_feedback() state overwrite found"
  fi
  if [[ -d "$BACKUP" ]] && grep -R -n -m1 'teacher_meanshift_feedback' "$BACKUP" --include='*.py' 2>/dev/null; then
    echo "wrong backup: teacher_meanshift_feedback() FOUND (this is the architecture-changing path)"
  fi
} > "$AUDIT"

# Hard safeguards for the architecture the user identified as the correct V36.
for required in config.py robust_tracker.py visual_model.py visual_localizer.py data.py; do
  if [[ ! -f "$ORIG/$required" ]]; then
    echo "ERROR: forNX is missing required file: $required" >&2
    exit 4
  fi
done
if grep -R -q 'teacher_meanshift_feedback' "$ORIG" --include='*.py'; then
  echo "ERROR: local forNX contains teacher_meanshift_feedback; it is not the expected pre-byTeacher V36." >&2
  exit 5
fi
if ! grep -q 'ThreeFrameRouteStateGRU' "$ORIG/robust_tracker.py"; then
  echo "ERROR: local forNX does not use ThreeFrameRouteStateGRU." >&2
  exit 6
fi
for h in correction_head variance_head motion_head heading_head; do
  if ! grep -q "$h" "$ORIG/visual_model.py"; then
    echo "ERROR: local forNX visual_model.py is missing $h." >&2
    exit 7
  fi
done
if ! grep -q 'UAVSAT_OUTPUT_DIR' "$ORIG/config.py"; then
  echo "ERROR: forNX config.py does not support UAVSAT_OUTPUT_DIR; refusing to silently redirect outputs." >&2
  exit 8
fi
if ! grep -q 'UAVSAT_EXPERIMENT_KALMAN' "$ORIG/config.py"; then
  echo "ERROR: forNX config.py does not expose the original V36 Kalman ablation switch." >&2
  exit 9
fi

# Modify only the G-only copy's *default* Kalman mode.  The original V36 already
# contains the no-Kalman branch in RouteKalman.update(), where the GRU+MS
# measurement becomes the state/output.  We do not invent a new estimator and do
# not touch the original_forNX copy.
python3 - "$GONLY/config.py" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
pat = re.compile(r'(EXPERIMENT_KALMAN\s*=\s*os\.environ\.get\(\s*["\']UAVSAT_EXPERIMENT_KALMAN["\']\s*,\s*)["\']learned["\'](\s*\))', re.S)
s2, n = pat.subn(r'\1"none"\2', s, count=1)
if n != 1:
    raise SystemExit("Could not patch exactly one EXPERIMENT_KALMAN default in G_only/config.py")
p.write_text(s2, encoding="utf-8")
PY

# The two source trees must now differ only in config.py (plus generated files
# created after the runs).  Save an explicit audit for later thesis/repro checks.
diff -qr "$ORIG" "$GONLY" > "$OUT/original_vs_Gonly_source_diff.txt" || true
DIFF_COUNT="$(wc -l < "$OUT/original_vs_Gonly_source_diff.txt" | tr -d ' ')"
if [[ "$DIFF_COUNT" != "1" ]] || ! grep -q 'config.py' "$OUT/original_vs_Gonly_source_diff.txt"; then
  echo "ERROR: G_only source differs from original_forNX in more than config.py." >&2
  cat "$OUT/original_vs_Gonly_source_diff.txt" >&2
  exit 10
fi

cat > "$MARK" <<EOF
source=$SRC
prepared_at=$(date -Iseconds)
original_copy_exact=yes
G_only_change=only config.py default EXPERIMENT_KALMAN learned -> none
EOF

cat "$AUDIT"
echo
cat "$OUT/original_vs_Gonly_source_diff.txt"
echo "[GvsK] preparation complete."
