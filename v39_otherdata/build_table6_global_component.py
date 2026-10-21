#!/usr/bin/env python3
"""Build one pooled ABCD component table in Bearing-UAV Table-6 style."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_iclr_ablation_tables import _clean, _read_variant

VARIANTS = ("no_gru", "no_kalman", "no_ms", "full")
LABELS = {
    "no_gru": "w/o GRU",
    "no_kalman": "w/o Kalman",
    "no_ms": "w/o MeanShift",
    "full": "Full",
}


def better(full, other):
    return {
        "R@1*": full["R@1*_pct"] > other["R@1*_pct"],
        "LSR@15": full["LSR@15_pct"] > other["LSR@15_pct"],
        "HSR@15": full["HSR@15_pct"] > other["HSR@15_pct"],
        "MLE": full["MLE_m"] < other["MLE_m"],
        "MHE": full["MHE_deg"] < other["MHE_deg"],
    }


def row(label, r):
    return (
        f"| {label} | {r['R@1*_pct']:.2f}% | {r['LSR@15_pct']:.2f}% | "
        f"{r['HSR@15_pct']:.2f}% | {r['MLE_m']:.3f} | {r['MHE_deg']:.2f} |"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    a = p.parse_args()
    root = Path(a.suite_root).resolve()
    official_audits = {}
    for city in a.cities:
        audit_path = root / city / "prepared" / "official96_split_audit.json"
        if audit_path.is_file():
            official_audits[city] = json.loads(audit_path.read_text(encoding="utf-8"))
    official96 = len(official_audits) == len(a.cities)
    schedule_path = root / "shared" / "frames3" / "episode_schedule.json"
    schedule = (
        json.loads(schedule_path.read_text(encoding="utf-8"))
        if schedule_path.is_file() else None
    )
    rows = {v: _read_variant(root, a.cities, v) for v in VARIANTS}
    full = rows["full"]
    checks = {v: better(full, rows[v]) for v in VARIANTS if v != "full"}
    payload = {
        "protocol": (
            "official rsi_id=96 seed-42 85/5/10 split adapted to independent "
            "per-city temporal episodes; one shared ABCD model; pooled official-test episodes"
            if official96 else
            "one shared ABCD model + one global Route-A-selected inference profile + pooled held-out B/C frames"
        ),
        "official96_split_adapter": bool(official96),
        "official96_split_audits": official_audits,
        "episode_training_schedule": schedule,
        "cities": a.cities,
        "metric_set": ["R@1*_pct", "LSR@15_pct", "HSR@15_pct", "MLE_m", "MHE_deg"],
        "rows": {v: _clean(rows[v]) for v in VARIANTS},
        "full_strictly_better": checks,
        "full_best_all_component_metrics": bool(all(all(x.values()) for x in checks.values())),
    }
    lines = [
        "# v39 temporal ABCD route component ablation", "",
        "Independent navigation routes from four cities are pooled. This is not "
        "Bearing-UAV's official rsi_id=96 static Table 6 protocol.", "",
        "| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if official96:
        lines[3:3] = [
            "Data: official rsi_id=96 merged metadata, seed-42 85/5/10 split; "
            "independent rows are ordered into per-city temporal episodes.", "",
        ]
    if schedule is not None:
        lines[3:3] = [
            f"Training: {schedule['rounds']} epochs; city episode order is "
            "shuffled from the recorded seed. GRU/Kalman state resets for each "
            "route; weights and optimizer persist.", "",
        ]
    for v in VARIANTS:
        lines.append(row(LABELS[v], rows[v]))
    lines += ["", f"Full strictly best on all five metrics vs every component removal: {payload['full_best_all_component_metrics']}"]
    md = "\n".join(lines) + "\n"
    (root / "table6_global_component.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "table6_global_component.md").write_text(md, encoding="utf-8")
    fields = ["variant", "label", "frames", "R@1*_pct", "LSR@15_pct", "HSR@15_pct", "MLE_m", "MHE_deg"]
    with (root / "table6_global_component.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in VARIANTS:
            r = rows[v]
            w.writerow({k: r[k] for k in fields})
    print(md, end="")
    print("[TABLE6]", root / "table6_global_component.md")

if __name__ == "__main__":
    main()
