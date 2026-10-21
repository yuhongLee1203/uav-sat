#!/usr/bin/env python3
"""Publish only localization metrics whose predictions match their targets."""
import argparse
import json
from pathlib import Path

GROUPS = (
    ("Main architecture", ("full",)),
    ("Component removal", ("no_gru", "no_kalman", "no_ms", "full")),
    ("Temporal input", ("frames1", "frames2", "full")),
    ("Final MeanShift window", ("grid4", "grid5", "full", "grid7", "grid8")),
)
LABELS = {"full": "Full (6x6)", "no_gru": "w/o GRU", "no_kalman": "w/o Kalman",
          "no_ms": "w/o final Mean Shift", "frames1": "1 frame",
          "frames2": "2 frames", "grid4": "4x4", "grid5": "5x5",
          "grid7": "7x7", "grid8": "8x8"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True, type=Path)
    root = p.parse_args().suite_root.resolve()
    path = root / "original_3759_all_metrics.json"
    raw = json.loads(path.read_text())
    rows = {
        variant: {key: float(values[key]) for key in
                  ("MLE_m", "LSR@3_pct", "LSR@5_pct", "LSR@15_pct")}
        for variant, values in raw["rows"].items()
    }
    clean = {
        "protocol": "current-frame GT-assisted localization on synthetic routes",
        "cities": list(raw["cities"]), "test_frames": raw["test_frames"],
        "reported_metrics": ["LSR@3_pct", "LSR@5_pct", "LSR@15_pct", "MLE_m"],
        "excluded_metrics": {
            "HSR@15/MHE": "The heading head predicts synthetic route-motion direction; its training target is not UAV image yaw_deg.",
            "Recall@1": "The inference did not produce an official four-tile ranking.",
            "SR@20/SPL/NE": "No closed-loop navigation rollout was recorded.",
        },
        "historical_paper_value_check": raw["historical_paper_value_check"],
        "rows": rows,
    }
    legacy = root / "paper_ablation_tables.md"
    if legacy.exists():
        legacy.rename(root / "route_motion_heading_diagnostic.md")
    legacy_json = root / "paper_ablation_results.json"
    if legacy_json.exists():
        legacy_json.rename(root / "route_motion_heading_diagnostic.json")
    path.write_text(json.dumps(clean, indent=2) + "\n")
    gate = clean["historical_paper_value_check"]
    lines = [
        "# Original 3.759 m replay: valid GT-assisted localization metrics", "",
        f"Historical proximity gate: **{gate['status']}**; exact printed-value match: "
        f"**{'YES' if gate['exact_printed_match'] else 'NO'}**.", "",
        "City A/B/C/D, eight synthetic routes, 1,482 held-out images. Current-frame GT "
        "centers search and selects the acquisition hypothesis.", "",
    ]
    for title, variants in GROUPS:
        lines += [f"## {title}", "",
                  "| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |",
                  "|---|---:|---:|---:|---:|"]
        for variant in variants:
            r = rows[variant]
            lines.append(f"| {LABELS[variant]} | {r['LSR@3_pct']:.2f}% | "
                         f"{r['LSR@5_pct']:.2f}% | {r['LSR@15_pct']:.2f}% | "
                         f"{r['MLE_m']:.3f} |")
        lines.append("")
    lines += [
        "## Excluded metrics", "",
        "HSR@15/MHE are not reported as Bearing-UAV heading metrics: the saved heading "
        "head was trained against direction between neighboring points on the synthetic "
        "route, not the UAV image's yaw_deg. Recall@1 is excluded because no official "
        "four-tile ranking was produced. SR@20/SPL/NE are excluded because there is "
        "no closed-loop navigation rollout.", "",
        "The legacy heading table, if present, is retained as "
        "route_motion_heading_diagnostic.md for debugging only.", "",
    ]
    if gate["exact_mismatches"]:
        lines += ["## Historical printed-value mismatches", ""]
        lines += ["- " + x for x in gate["exact_mismatches"]]
        lines.append("")
    (root / "original_3759_all_metrics.md").write_text("\n".join(lines))
    print("[VALID LOCALIZATION REPORT]", root / "original_3759_all_metrics.md")


if __name__ == "__main__":
    main()
