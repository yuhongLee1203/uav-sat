#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REQUIRED = {
    "TEMPORAL_DIRECT_ACCEL_FORWARD_M": 'TEMPORAL_DIRECT_ACCEL_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M", "1.25"))',
    "TEMPORAL_DIRECT_ACCEL_CROSS_M": 'TEMPORAL_DIRECT_ACCEL_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M", "0.75"))',
    "TEMPORAL_DIRECT_STEP_FORWARD_M": 'TEMPORAL_DIRECT_STEP_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M", "2.00"))',
    "TEMPORAL_DIRECT_STEP_CROSS_M": 'TEMPORAL_DIRECT_STEP_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M", "1.00"))',
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patch_script")
    args = ap.parse_args()

    path = Path(args.patch_script).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    # The four definitions must live in the c += ''' ... ''' runtime-config append
    # block.  Do not use a whole-file substring test because the same symbol names
    # also appear in visual_model code strings.
    anchor = 'TEMPORAL_DELTA2_SCALE = float(os.environ.get("UAVSAT_TEMPORAL_DELTA2_SCALE", "1.00"))\n'
    if text.count(anchor) != 1:
        raise RuntimeError(f"expected exactly one TEMPORAL_DELTA2_SCALE config anchor, found {text.count(anchor)}")

    config_block_start = text.find("c += '''")
    if config_block_start < 0:
        raise RuntimeError("could not locate runtime config append block")
    config_block_end = text.find("'''", config_block_start + len("c += '''"))
    if config_block_end < 0:
        raise RuntimeError("unterminated runtime config append block")

    config_block = text[config_block_start:config_block_end]
    missing = [name for name, definition in REQUIRED.items() if definition not in config_block]
    if missing:
        insertion = "".join(REQUIRED[name] + "\n" for name in missing)
        text = text.replace(anchor, anchor + insertion, 1)
        path.write_text(text, encoding="utf-8")

    # Re-read and audit the config block specifically, not the whole source file.
    text = path.read_text(encoding="utf-8")
    config_block_start = text.find("c += '''")
    config_block_end = text.find("'''", config_block_start + len("c += '''"))
    config_block = text[config_block_start:config_block_end]
    absent = [name for name, definition in REQUIRED.items() if definition not in config_block]
    if absent:
        raise RuntimeError(f"direct-delta2 config definitions still missing: {absent}")

    compile(text, str(path), "exec")
    subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
    for name in REQUIRED:
        print(f"[DIRECT-CONFIG AUDIT] {name}: PASS")


if __name__ == "__main__":
    main()
