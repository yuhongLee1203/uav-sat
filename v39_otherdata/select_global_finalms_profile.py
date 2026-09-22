#!/usr/bin/env python3
"""Select one shared final-MeanShift profile from ABCD Route-A validation only.

The held-out test_01/test_02 outputs are never read.  All four cities contribute
to one shared decoder profile so the subsequent Full / w/o-Kalman comparison
uses exactly the same final decoder settings everywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    args = p.parse_args()

    root = Path(args.suite_root).resolve()
    by_name: dict[str, list[dict]] = {}
    source_files = []

    for city in args.cities:
        path = root / city / "train_frames3" / "finalms_trainval_calibration.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("held_out_navigation_read") is not False:
            raise RuntimeError(f"{path}: held_out_navigation_read must be false")
        if payload.get("selection_source") != "route_A_validation_only":
            raise RuntimeError(f"{path}: unexpected selection source")
        if str(payload.get("city")) != city:
            raise RuntimeError(f"{path}: city mismatch")
        source_files.append(str(path))
        for row in payload["profiles"]:
            by_name.setdefault(str(row["name"]), []).append({"city": city, **row})

    expected = set(args.cities)
    aggregates = []
    hyper_keys = (
        "ms_kf_prior_weight",
        "ms_reference_prior_weight",
        "ms_kf_sigma_m",
        "ms_reference_sigma_m",
    )
    for name, rows in sorted(by_name.items()):
        seen = {r["city"] for r in rows}
        if seen != expected:
            raise RuntimeError(f"profile {name}: cities={sorted(seen)} expected={sorted(expected)}")
        reference = {k: float(rows[0][k]) for k in hyper_keys}
        for row in rows[1:]:
            for key in hyper_keys:
                if abs(float(row[key]) - reference[key]) > 1e-12:
                    raise RuntimeError(f"profile {name}: inconsistent {key}")

        weights = [int(r["val_frames"]) for r in rows]
        total = sum(weights)
        if total <= 0:
            raise RuntimeError(f"profile {name}: no validation frames")

        def weighted(key: str) -> float:
            return sum(w * float(r[key]) for w, r in zip(weights, rows)) / total

        aggregates.append({
            "name": name,
            **reference,
            "val_frames": int(total),
            "val_mle_m": weighted("val_mle_m"),
            "val_p90_m_mean_across_city_frames": weighted("val_p90_m"),
            "val_lsr15_pct": weighted("val_lsr15_pct"),
            "val_hsr15_pct": weighted("val_hsr15_pct"),
            "val_mhe_deg": weighted("val_mhe_deg"),
            "objective": weighted("objective"),
            "per_city": rows,
        })

    if not aggregates:
        raise RuntimeError("no calibration profiles found")

    best = min(
        aggregates,
        key=lambda r: (
            float(r["objective"]),
            float(r["val_mle_m"]),
            float(r["val_p90_m_mean_across_city_frames"]),
            str(r["name"]),
        ),
    )
    payload = {
        "selection_source": "ABCD_route_A_validation_only",
        "held_out_navigation_read": False,
        "cities": args.cities,
        "selection_rule": "validation-frame-weighted mean of per-city (MLE + 0.05*P90)",
        "purpose": "choose one shared final MeanShift prior profile before held-out Full vs w/o-Kalman evaluation",
        "best": best,
        "profiles": aggregates,
        "source_files": source_files,
    }
    out = root / "finalms_global_calibration.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[GLOBAL-FINALMS-BEST]", json.dumps(best, sort_keys=True), flush=True)
    print("[GLOBAL-FINALMS-JSON]", out, flush=True)


if __name__ == "__main__":
    main()
