#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def replace_line_range(lines: list[str], start_text: str, end_text: str, replacement: list[str], label: str) -> list[str]:
    starts = [i for i, line in enumerate(lines) if line.rstrip("\n") == start_text]
    if len(starts) != 1:
        raise RuntimeError(f"{label}: expected exactly one start marker, found {len(starts)}")
    start = starts[0]
    ends = [i for i in range(start, len(lines)) if lines[i].rstrip("\n") == end_text]
    if len(ends) != 1:
        raise RuntimeError(f"{label}: expected exactly one end marker after start, found {len(ends)}")
    end = ends[0]
    return lines[:start] + [x + "\n" for x in replacement] + lines[end + 1 :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    args = ap.parse_args()

    path = Path(args.target).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    lines = replace_line_range(
        lines,
        '  ( run_train 2 5 ) & p2=$!',
        '  [[ "${status}" == "0" ]] || { echo "ERROR: temporal training failed" >&2; exit 20; }',
        [
            '  echo "[STABLE MODE] frame-2 and frame-3 training sequential on GPU0"',
            '  run_train 2 0',
            '  run_train 3 0',
        ],
        "temporal training block",
    )

    lines = replace_line_range(
        lines,
        '( run_eval_group 0 no_gru grid4 grid7 ) & p0=$!',
        '[[ "${status}" == "0" ]] || { echo "ERROR: evaluation failed; inspect logs" >&2; exit 21; }',
        [
            'echo "[STABLE MODE] ablation evaluation sequential on GPU0"',
            'run_eval_group 0 no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8',
        ],
        "ablation evaluation block",
    )

    text = "".join(lines)
    required = [
        '[STABLE MODE] frame-2 and frame-3 training sequential on GPU0',
        'run_train 2 0',
        'run_train 3 0',
        '[STABLE MODE] ablation evaluation sequential on GPU0',
        'run_eval_group 0 no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8',
    ]
    forbidden = [
        '( run_train 2 5 ) & p2=$!',
        '( run_train 3 6 ) & p3=$!',
        '( run_eval_group 0 no_gru grid4 grid7 ) & p0=$!',
        '( run_eval_group 5 no_kalman frames1 grid5 ) & p5=$!',
        '( run_eval_group 6 no_ms frames2 grid8 ) & p6=$!',
    ]
    missing = [x for x in required if x not in text]
    stale = [x for x in forbidden if x in text]
    if missing or stale:
        raise RuntimeError(f"patch audit failed: missing={missing}, stale={stale}")

    path.write_text(text, encoding="utf-8")
    subprocess.run(["bash", "-n", str(path)], check=True)

    print("[SEQUENTIAL PATCH] temporal training: PASS")
    print("[SEQUENTIAL PATCH] ablation evaluation: PASS")
    print("[SEQUENTIAL PATCH] bash -n: PASS")


if __name__ == "__main__":
    main()
