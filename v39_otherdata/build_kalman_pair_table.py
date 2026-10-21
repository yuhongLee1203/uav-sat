#!/usr/bin/env python3
"""Build an isolated Full vs w/o-Kalman report from measured held-out outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_iclr_ablation_tables import _clean, _paired_bootstrap, _read_variant, _row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    args = p.parse_args()
    root = Path(args.suite_root).resolve()

    full = _read_variant(root, args.cities, "full")
    no_k = _read_variant(root, args.cities, "no_kalman")
    bootstrap = _paired_bootstrap(full["_errors"], no_k["_errors"])
    payload = {
        "cities": args.cities,
        "comparison": "same shared checkpoint / same seed / same prepared data / same final-MS profile; Kalman fixed vs none",
        "full": _clean(full),
        "no_kalman": _clean(no_k),
        "full_minus_no_kalman": {
            "R@1*_pct": float(full["R@1*_pct"] - no_k["R@1*_pct"]),
            "LSR@15_pct": float(full["LSR@15_pct"] - no_k["LSR@15_pct"]),
            "HSR@15_pct": float(full["HSR@15_pct"] - no_k["HSR@15_pct"]),
            "MLE_m": float(full["MLE_m"] - no_k["MLE_m"]),
            "MHE_deg": float(full["MHE_deg"] - no_k["MHE_deg"]),
        },
        "paired_bootstrap": bootstrap,
    }
    out_json = root / "kalman_pair_results.json"
    out_md = root / "kalman_pair_table.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Full vs w/o Kalman (held-out, measured)", "",
        "| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |",
        "|---|---:|---:|---:|---:|---:|",
        _row("w/o Kalman", no_k),
        _row("Full", full),
        "",
        f"Full − w/o Kalman MLE: {payload['full_minus_no_kalman']['MLE_m']:+.4f} m",
        f"Paired MLE 95% CI: [{bootstrap['MLE_95CI'][0]:+.4f}, {bootstrap['MLE_95CI'][1]:+.4f}] m",
        "",
        "R@1* is the Bearing-UAV same-quadrant/sign criterion derived from continuous final XY.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("[PAIR-JSON]", out_json, flush=True)
    print("[PAIR-MD]", out_md, flush=True)


if __name__ == "__main__":
    main()
