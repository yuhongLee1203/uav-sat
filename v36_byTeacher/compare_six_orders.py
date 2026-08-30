"""Collect six-order evaluation summaries into CSV + Markdown rankings."""

import csv
import json
from pathlib import Path

import config

ORDERS = ("MKG", "MGK", "GMK", "GKM", "KGM", "KMG")
ROOT = Path(config.BACKBONE_OUTPUT_DIR) / "six_order_ablation"


def main():
    rows = []
    for order in ORDERS:
        path = ROOT / order / "summary.json"
        if not path.exists():
            print(f"skip {order}: {path} not found")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for route_name, s in payload.get("results", {}).items():
            rows.append(
                {
                    "Order": order,
                    "Route": route_name,
                    "MLE_m": float(s["MLE_m"]),
                    "MedLE_m": float(s["MedLE_m"]),
                    "P90_m": float(s["P90_m"]),
                    "P95_m": float(s["P95_m"]),
                    "CVaR90_m": float(s["CVaR90_m"]),
                    "LSR@5_pct": float(s["LSR@5_pct"]),
                    "LSR@10_pct": float(s["LSR@10_pct"]),
                    "LSR@15_pct": float(s["LSR@15_pct"]),
                    "LSR@20_pct": float(s["LSR@20_pct"]),
                    "FPS": float(s.get("Timing", {}).get("fps", float("nan"))),
                }
            )
    if not rows:
        raise SystemExit("No six-order summary.json files found")
    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "six_order_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Six-order M/G/K comparison",
        "",
        "Primary ranking is lower MLE, then lower P90. Report B and C separately before drawing a conclusion.",
        "",
        "| Route | Rank | Order | MLE (m) | P90 (m) | LSR@15 (%) | FPS |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for route in sorted({r["Route"] for r in rows}):
        group = sorted(
            [r for r in rows if r["Route"] == route],
            key=lambda r: (r["MLE_m"], r["P90_m"]),
        )
        for rank, r in enumerate(group, 1):
            fps = "-" if r["FPS"] != r["FPS"] else f"{r['FPS']:.2f}"
            md.append(
                f"| {route} | {rank} | {r['Order']} | {r['MLE_m']:.3f} | "
                f"{r['P90_m']:.3f} | {r['LSR@15_pct']:.2f} | {fps} |"
            )
    md_path = ROOT / "six_order_comparison.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
