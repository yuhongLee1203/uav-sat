#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BRANCH="bearing-v5-formal-smooth-v1"
PATCHED="v39_otherdata/.run_bearing_core_v5_restore_v3_patched.sh"

git fetch origin "${BRANCH}" >/dev/null
git show "origin/${BRANCH}:v39_otherdata/run_bearing_core_v5_restore_v2.sh" > "${PATCHED}"

python3 - <<'PY'
from pathlib import Path
p = Path('v39_otherdata/.run_bearing_core_v5_restore_v3_patched.sh')
s = p.read_text(encoding='utf-8')
old_train = '  local city="$1" gpu="$2" train_root="${SUITE}/${city}/train_core_v5_restore"\n'
new_train = (
    '  local city="$1"\n'
    '  local gpu="$2"\n'
    '  local train_root="${SUITE}/${city}/train_core_v5_restore"\n'
)
if s.count(old_train) != 1:
    raise SystemExit(f'[CORE-V5 V3 PATCH] train_city match count={s.count(old_train)}; expected 1')
s = s.replace(old_train, new_train, 1)

old_run = '  local city="$1" gpu="$2" variant out\n'
new_run = (
    '  local city="$1"\n'
    '  local gpu="$2"\n'
    '  local variant\n'
    '  local out\n'
)
if s.count(old_run) != 1:
    raise SystemExit(f'[CORE-V5 V3 PATCH] run_city match count={s.count(old_run)}; expected 1')
s = s.replace(old_run, new_run, 1)

# Refuse to execute if the unsafe declaration survived.
if 'local city="$1" gpu="$2" train_root="${SUITE}/${city}/train_core_v5_restore"' in s:
    raise SystemExit('[CORE-V5 V3 PATCH] unsafe train_city declaration still present')

p.write_text(s, encoding='utf-8')
print('[CORE-V5 V3 PATCH] train_city local declaration: PASS')
print('[CORE-V5 V3 PATCH] run_city local declaration: PASS')
PY

chmod +x "${PATCHED}"
bash -n "${PATCHED}"

echo "[CORE-V5 V3 PRECHECK] bash syntax: PASS"
if grep -Fq 'local city="$1" gpu="$2" train_root="${SUITE}/${city}/train_core_v5_restore"' "${PATCHED}"; then
  echo "ERROR: unsafe Bash local declaration remains" >&2
  exit 91
fi
echo "[CORE-V5 V3 PRECHECK] set -u local expansion hazard removed: PASS"

exec bash "${PATCHED}" "$@"
