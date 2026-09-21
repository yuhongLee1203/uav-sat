#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Resource-safe defaults for the formal 4-city run.
# GPU 0/5/6 still run concurrently, but CPU math pools are capped and the
# 47,961-patch SAT backbone cache is built by only one city at a time.
CPU_THREADS_PER_CITY="${CPU_THREADS_PER_CITY:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
CPU_NICE="${CPU_NICE:-5}"
SAT_CACHE_LOCK="${SAT_CACHE_LOCK:-/tmp/uavsat_formal_v5_sat_cache.lock}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS_PER_CITY}"
export BLIS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TOKENIZERS_PARALLELISM=false

# Smaller cache batches reduce host RAM / image decode spikes. This changes only
# cache batching, not model architecture, labels, validation, or test protocol.
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${UAVSAT_VISUAL_CACHE_BATCH_SIZE:-${CACHE_BATCH_SIZE}}"

# All city processes share one advisory lock around the expensive SAT backbone
# gallery build. After a city's cache is ready, its GPU training continues while
# the next city gets the cache lock, so the three GPUs are still pipelined.
export UAVSAT_SERIALIZE_SAT_CACHE="${UAVSAT_SERIALIZE_SAT_CACHE:-1}"
export UAVSAT_SAT_CACHE_LOCK="${UAVSAT_SAT_CACHE_LOCK:-${SAT_CACHE_LOCK}}"

# Install the lock wrapper into the local base source idempotently. The formal
# runtime copies this source, so every spawned city inherits the same lock.
python3 - <<'PY'
from pathlib import Path

p = Path('v39_DirectFinalMS/base_src/visual_localizer.py')
s = p.read_text(encoding='utf-8')
marker = '[SAT-CACHE LOCK] waiting'

if marker not in s:
    if 'import math\nimport random\n' not in s:
        raise SystemExit('resource patch failed: import insertion point not found')
    s = s.replace(
        'import math\nimport random\n',
        'import fcntl\nimport math\nimport os\nimport random\n',
        1,
    )

    old = '''@torch.no_grad()
def _build_satellite_backbone_gallery(
    model,
    origin_lat,
    origin_lon,
    device,
):
'''
    new = '''@torch.no_grad()
def _build_satellite_backbone_gallery(
    model,
    origin_lat,
    origin_lon,
    device,
):
    if os.environ.get("UAVSAT_SERIALIZE_SAT_CACHE", "1") != "1":
        return _build_satellite_backbone_gallery_unlocked(
            model, origin_lat, origin_lon, device
        )

    lock_path = Path(
        os.environ.get(
            "UAVSAT_SAT_CACHE_LOCK",
            "/tmp/uavsat_formal_v5_sat_cache.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        print(
            f"[SAT-CACHE LOCK] waiting pid={os.getpid()} lock={lock_path}",
            flush=True,
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(
            f"[SAT-CACHE LOCK] acquired pid={os.getpid()}",
            flush=True,
        )
        try:
            return _build_satellite_backbone_gallery_unlocked(
                model, origin_lat, origin_lon, device
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(
                f"[SAT-CACHE LOCK] released pid={os.getpid()}",
                flush=True,
            )


@torch.no_grad()
def _build_satellite_backbone_gallery_unlocked(
    model,
    origin_lat,
    origin_lon,
    device,
):
'''
    if s.count(old) != 1:
        raise SystemExit(
            'resource patch failed: SAT gallery function match count=' + str(s.count(old))
        )
    s = s.replace(old, new, 1)

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[RESOURCE PATCH] serialized SAT cache build: PASS')
PY

python3 -m py_compile v39_DirectFinalMS/base_src/visual_localizer.py

printf '%s\n' \
  "================================================================================" \
  "FORMAL V5 RESOURCE-SAFE RUN" \
  "GPU jobs             : 0 / 5 / 6 (pipelined concurrently)" \
  "CPU threads per city : ${CPU_THREADS_PER_CITY}" \
  "SAT cache batch      : ${UAVSAT_VISUAL_CACHE_BATCH_SIZE}" \
  "SAT cache concurrency: 1 city at a time" \
  "CPU nice             : ${CPU_NICE}" \
  "SAT cache lock       : ${UAVSAT_SAT_CACHE_LOCK}" \
  "================================================================================"

# nice lowers CPU scheduling priority so the host stays responsive. All child
# Python jobs inherit the thread caps and cache lock settings above.
exec nice -n "${CPU_NICE}" bash v39_otherdata/run_bearing_iclr_ablation_fixed.sh
