#!/usr/bin/env python3
"""Check the archived temporal MLE/LSR@3/LSR@5 against a fresh replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED = {
    "frames1": (3.882988872803729, 39.8110661268556, 72.46963562753037),
    "frames2": (3.921011143065327, 39.13630229419703, 71.7948717948718),
    "full": (3.7591276968535423, 41.90, 73.75),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.suite_root
    rows = []
    for variant, expected in EXPECTED.items():
        errors = []
        for city in ("citya", "cityb", "cityc", "cityd"):
            folder = root / city / "variants" / variant
            summaries = json.loads((folder / "bearing_v39_summary.json").read_text())
            for route in ("test_01", "test_02"):
                csv_path = folder / Path(summaries[route]["CSV"]).name
                with csv_path.open(newline="") as handle:
                    errors.extend(float(r["error_final_m"]) for r in csv.DictReader(handle))
        n = len(errors)
        actual = (sum(errors) / n,
                  100 * sum(x <= 3.0 for x in errors) / n,
                  100 * sum(x <= 5.0 for x in errors) / n)
        if n != 1482:
            raise SystemExit(f"{variant}: expected 1482 frames, got {n}")
        if any(abs(a - b) > tol for a, b, tol in zip(actual, expected, (0.05, 0.15, 0.15))):
            raise SystemExit(f"{variant}: replay MLE/LSR@3/LSR@5 {actual} differs from archived {expected}")
        rows.append((variant, *actual))
        print(f"[3759 REPRODUCED] {variant}: MLE={actual[0]:.3f} m, "
              f"LSR@3={actual[1]:.2f}%, LSR@5={actual[2]:.2f}%", flush=True)
    lines = ["# Original v13 4 m temporal result, independently replayed", "",
             "| Input | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ |",
             "|---|---:|---:|---:|"]
    for variant, mle, lsr3, lsr5 in rows:
        label = {"frames1": "1 frame", "frames2": "2 frames", "full": "3 frames"}[variant]
        lines.append(f"| {label} | {mle:.3f} | {lsr3:.2f}% | {lsr5:.2f}% |")
    (root / "reproduced_temporal_3_5.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
