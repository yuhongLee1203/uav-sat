#!/usr/bin/env python3
"""Select one shared Kalman profile from A-only validation across four cities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("R@1*_pct", "LSR@15_pct", "HSR@15_pct", "MLE_m", "MHE_deg")


def aggregate(rows, key):
    total = sum(int(r[key]["frames"]) for r in rows)
    if total <= 0:
        raise RuntimeError("zero validation frames")
    out = {"frames": total}
    for metric in METRICS:
        out[metric] = sum(float(r[key][metric]) * int(r[key]["frames"]) for r in rows) / total
    return out


def margins(full, no_k):
    return {
        "R@1*_pct": full["R@1*_pct"] - no_k["R@1*_pct"],
        "LSR@15_pct": full["LSR@15_pct"] - no_k["LSR@15_pct"],
        "HSR@15_pct": full["HSR@15_pct"] - no_k["HSR@15_pct"],
        "MLE_m": no_k["MLE_m"] - full["MLE_m"],
        "MHE_deg": no_k["MHE_deg"] - full["MHE_deg"],
    }


def score(m):
    # Validation-only ranking among measured profiles.  Positive is better.
    return (
        0.02 * m["R@1*_pct"]
        + 0.02 * m["LSR@15_pct"]
        + 0.02 * m["HSR@15_pct"]
        + 1.00 * m["MLE_m"]
        + 0.05 * m["MHE_deg"]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", required=True)
    p.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    a = p.parse_args()
    root = Path(a.suite_root).resolve()

    city_payloads = {}
    for city in a.cities:
        path = root / city / "train_frames3" / "final_output_kalman_trainval.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("held_out_navigation_read") is not False:
            raise RuntimeError(f"{city}: calibration is not train-validation-only")
        city_payloads[city] = payload

    profile_names = [x["profile"]["name"] for x in city_payloads[a.cities[0]]["profiles"]]
    evaluated = []
    for name in profile_names:
        rows = []
        profile = None
        per_city = {}
        for city in a.cities:
            matches = [x for x in city_payloads[city]["profiles"] if x["profile"]["name"] == name]
            if len(matches) != 1:
                raise RuntimeError(f"{city}: expected one profile {name}")
            row = matches[0]
            profile = row["profile"]
            rows.append(row)
            per_city[city] = {
                "full": row["full"],
                "no_kalman": row["no_kalman"],
                "margins_full_better": row["margins_full_better"],
                "strict_all_five": row["strict_all_five"],
            }
        full = aggregate(rows, "full")
        no_k = aggregate(rows, "no_kalman")
        m = margins(full, no_k)
        evaluated.append({
            "profile": profile,
            "full": full,
            "no_kalman": no_k,
            "margins_full_better": m,
            "strict_all_five": bool(all(v > 0.0 for v in m.values())),
            "positive_metric_count": int(sum(v > 0.0 for v in m.values())),
            "validation_score": float(score(m)),
            "per_city": per_city,
        })

    strict = [r for r in evaluated if r["strict_all_five"]]
    if strict:
        best = max(strict, key=lambda r: (r["validation_score"], r["margins_full_better"]["MLE_m"]))
    else:
        best = max(evaluated, key=lambda r: (
            r["positive_metric_count"], r["validation_score"], r["margins_full_better"]["MLE_m"]
        ))

    payload = {
        "selection_source": "combined Route-A validation across citya/cityb/cityc/cityd only",
        "held_out_navigation_read": False,
        "architecture_changed": False,
        "metric_schema_changed": False,
        "strict_all_five_profile_found": bool(strict),
        "best": best,
        "profiles": evaluated,
    }
    out = root / "final_output_kalman_global_profile.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[GLOBAL VALIDATION PROFILE]", json.dumps(best, indent=2), flush=True)
    print("[STRICT ALL FIVE]", bool(strict), flush=True)
    print("[PROFILE JSON]", out, flush=True)


if __name__ == "__main__":
    main()
