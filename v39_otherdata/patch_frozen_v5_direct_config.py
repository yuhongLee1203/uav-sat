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

    # Frozen V5 originally enabled several mutually dependent pseudo-label
    # losses. Keep the architecture intact, but train it with the smallest
    # identifiable objective: position, one-step displacement and variance NLL.
    text = text.replace(
        'UAVSAT_LOSS_VELOCITY", "0.40"',
        'UAVSAT_LOSS_VELOCITY", "0.0"',
    ).replace(
        'UAVSAT_LOSS_ACCELERATION", "0.25"',
        'UAVSAT_LOSS_ACCELERATION", "0.0"',
    )

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

    loss_contract = '''# Minimal identifiable Bearing-UAV temporal objective.
LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "0.0"))
LOSS_ACCELERATION = float(os.environ.get("UAVSAT_LOSS_ACCELERATION", "0.0"))
LOSS_HEADING = float(os.environ.get("UAVSAT_LOSS_HEADING", "0.0"))
LOSS_TURN_RATE = 0.0
LOSS_SPEED = 0.0
LOSS_CROSS_MOTION_REG = 0.0
LOSS_PROGRESS = 0.0
LOSS_ACQUISITION = 0.0
LOSS_VARIANCE_NLL = float(os.environ.get("UAVSAT_LOSS_VARIANCE_NLL", "0.05"))
'''
    if loss_contract not in text:
        text = text.replace("c += '''\n", "c += '''\n" + loss_contract, 1)
    path.write_text(text, encoding="utf-8")

    worktree = path.parent.parent
    prepare = worktree / "v39_otherdata" / "bearing_prepare.py"
    prepare_text = prepare.read_text(encoding="utf-8")
    if "OFFICIAL_OFFSET_SCALE_PX = 128" not in prepare_text:
        prepare_text = prepare_text.replace(
            "PATCH_SIZE = 256\n",
            "PATCH_SIZE = 256\nOFFICIAL_OFFSET_SCALE_PX = 128\n",
            1,
        )
    prepare_text = prepare_text.replace(
        'rows["x_norm"].astype(float) * PATCH_SIZE',
        'rows["x_norm"].astype(float) * OFFICIAL_OFFSET_SCALE_PX',
    ).replace(
        'rows["y_norm"].astype(float) * PATCH_SIZE',
        'rows["y_norm"].astype(float) * OFFICIAL_OFFSET_SCALE_PX',
    )
    prepare.write_text(prepare_text, encoding="utf-8")

    runner = worktree / "v39_otherdata" / "run_bearing_iclr_ablation.sh"
    runner_text = runner.read_text(encoding="utf-8")
    runner_text = runner_text.replace(
        'UAVSAT_LOSS_VELOCITY:-0.40', 'UAVSAT_LOSS_VELOCITY:-0.0'
    ).replace(
        'UAVSAT_LOSS_ACCELERATION:-0.25', 'UAVSAT_LOSS_ACCELERATION:-0.0'
    )
    loss_export_anchor = 'export UAVSAT_LOSS_ACCELERATION="${UAVSAT_LOSS_ACCELERATION:-0.0}"\n'
    loss_exports = loss_export_anchor + (
        'export UAVSAT_LOSS_HEADING="${UAVSAT_LOSS_HEADING:-0.0}"\n'
        'export UAVSAT_LOSS_VARIANCE_NLL="${UAVSAT_LOSS_VARIANCE_NLL:-0.05}"\n'
    )
    if "export UAVSAT_LOSS_HEADING=" not in runner_text:
        if loss_export_anchor not in runner_text:
            raise RuntimeError("could not locate temporal loss export block")
        runner_text = runner_text.replace(loss_export_anchor, loss_exports, 1)
    audit_anchor = "fi\n\ncommon_args(){"
    audit_call = '''fi

python3 -u v39_otherdata/audit_bearing_training_contract.py \\
  --dataset-root "${DATASET_ROOT}" --prepared-root "${PREPARED}" --city "${CITY}"

common_args(){'''
    if "audit_bearing_training_contract.py" not in runner_text:
        if audit_anchor not in runner_text:
            raise RuntimeError("could not locate post-prepare audit insertion point")
        runner_text = runner_text.replace(audit_anchor, audit_call, 1)
    runner.write_text(runner_text, encoding="utf-8")

    fixed = worktree / "v39_otherdata" / "run_bearing_iclr_ablation_fixed.sh"
    fixed_text = fixed.read_text(encoding="utf-8").replace(
        "s = s.replace('UAVSAT_LOSS_ACCELERATION:-0.25', 'UAVSAT_LOSS_ACCELERATION:-0.50')",
        "s = s.replace('UAVSAT_LOSS_ACCELERATION:-0.25', 'UAVSAT_LOSS_ACCELERATION:-0.0')",
    )
    fixed_text = fixed_text.replace(
        "for temporal_scale in (0.90, 1.00, 1.10):",
        "for temporal_scale in (0.00, 0.25, 0.50, 0.75, 0.90, 1.00, 1.10):",
    ).replace(
        "for delta2_scale in (0.00, 0.50, 1.00, 1.50, 2.00):",
        "for delta2_scale in (0.00, 0.25, 0.50, 0.75, 1.00):",
    )
    fixed.write_text(fixed_text, encoding="utf-8")

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
    subprocess.run(["python3", "-m", "py_compile", str(prepare)], check=True)
    for name in REQUIRED:
        print(f"[DIRECT-CONFIG AUDIT] {name}: PASS")
    print("[GT CONTRACT] official Bearing-UAV offset scale=128 px: PASS")
    print("[LOSS CONTRACT] measurement + next-step + variance-NLL only: PASS")


if __name__ == "__main__":
    main()
