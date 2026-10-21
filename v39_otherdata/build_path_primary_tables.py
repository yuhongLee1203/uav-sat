#!/usr/bin/env python3
"""Build the MLE-primary ablation from measured ABCD shared-model outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_iclr_ablation_tables import _clean, _paired_bootstrap, _read_variant

COMPONENTS = ("path_no_gru", "path_no_ms", "path_full")
TEMPORAL = ("path_frames1", "path_frames2", "path_full")
LABELS = {
    "path_no_gru": "w/o GRU",
    "path_no_ms": "w/o MeanShift",
    "path_frames1": "1 frame",
    "path_frames2": "2 frames",
    "path_full": "Full (3-frame GRU + MeanShift)",
    "full": "+ optional Kalman stability extension",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    args = p.parse_args()
    root = Path(args.suite_root).resolve()
    keys = tuple(dict.fromkeys((*COMPONENTS, *TEMPORAL, "full")))
    rows = {key: _read_variant(root, args.cities, key) for key in keys}
    main = rows["path_full"]
    component_best = all(main["MLE_m"] <= rows[k]["MLE_m"] for k in COMPONENTS[:-1])
    temporal_best = all(main["MLE_m"] <= rows[k]["MLE_m"] for k in TEMPORAL[:-1])
    audit = {
        "path_full_component_mle_best": component_best,
        "three_frame_mle_best": temporal_best,
        "PATH_PRIMARY_TREND_CHECK": "PASS" if component_best and temporal_best else "FAIL",
        "paired_bootstrap": {
            key: _paired_bootstrap(main["errors"], rows[key]["errors"])
            for key in (*COMPONENTS[:-1], *TEMPORAL[:-1])
        },
        "selection": "Temporal/Kalman scales selected on each city's train_01 validation only.",
        "integrity": "All values are aggregated from raw held-out frame CSVs without editing or clipping.",
    }
    lines = [
        "# Bearing-UAV ABCD path-primary ablation (measured)", "",
        "Primary contribution: held-out route localization error (MLE).", "",
        "## Component ablation", "",
        "| Variant | MLE (m) | P90 (m) | LSR@5 | Jump rate |", "|---|---:|---:|---:|---:|",
    ]
    for key in COMPONENTS:
        r = rows[key]
        lines.append(f"| {LABELS[key]} | {r['MLE_m']:.3f} | {r['P90_m']:.3f} | {r['LSR@5_pct']:.2f}% | {r['JumpRate_pct']:.3f}% |")
    lines += ["", "## Temporal input", "", "| Input | MLE (m) | P90 (m) | LSR@5 |", "|---|---:|---:|---:|"]
    for key in TEMPORAL:
        r = rows[key]
        lines.append(f"| {LABELS[key]} | {r['MLE_m']:.3f} | {r['P90_m']:.3f} | {r['LSR@5_pct']:.2f}% |")
    k = rows["full"]
    lines += [
        "", "## Optional stability extension", "",
        "| Configuration | MLE (m) | P90 (m) | Jump rate |", "|---|---:|---:|---:|",
        f"| {LABELS['path_full']} | {main['MLE_m']:.3f} | {main['P90_m']:.3f} | {main['JumpRate_pct']:.3f}% |",
        f"| {LABELS['full']} | {k['MLE_m']:.3f} | {k['P90_m']:.3f} | {k['JumpRate_pct']:.3f}% |",
        "", f"`PATH_PRIMARY_TREND_CHECK={audit['PATH_PRIMARY_TREND_CHECK']}`", "",
    ]
    payload = {"cities": args.cities, "rows": {k: _clean(v) for k, v in rows.items()}, "audit": audit}
    (root / "paper_path_primary_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = list(_clean(next(iter(rows.values()))).keys())
    with (root / "paper_path_primary_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows(_clean(rows[k]) for k in keys)
    text = "\n".join(lines)
    (root / "paper_path_primary_tables.md").write_text(text, encoding="utf-8")
    (root / "paper_path_primary_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
