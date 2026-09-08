#!/usr/bin/env python3
"""Compare the Kalman posterior against the post-Kalman final MeanShift output."""

import csv
from pathlib import Path

import numpy as np

import config


def metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {"MLE_m": float("nan"), "P90_m": float("nan"), "LSR15_pct": float("nan")}
    return {
        "MLE_m": float(np.mean(errors)),
        "P90_m": float(np.quantile(errors, 0.90)),
        "LSR15_pct": float(np.mean(errors <= 15.0) * 100.0),
    }


def load_errors(path):
    kalman_errors = []
    final_errors = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"kalman_error_m", "error_final_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            kalman_errors.append(float(row["kalman_error_m"]))
            final_errors.append(float(row["error_final_m"]))
    return np.asarray(kalman_errors), np.asarray(final_errors)


def main():
    output_dir = Path(config.OUTPUT_DIR)
    summary_rows = []

    print("Kalman vs post-Kalman MeanShift")
    print("=" * 88)
    for route in ("route_C", "route_B"):
        path = output_dir / f"{route}_autonomous_ms1_kf_gru_ms2_frames.csv"
        if not path.exists():
            print(f"{route}: missing {path}")
            continue

        kalman_errors, final_errors = load_errors(path)
        kf = metrics(kalman_errors)
        ms = metrics(final_errors)
        delta_mle = ms["MLE_m"] - kf["MLE_m"]
        delta_p90 = ms["P90_m"] - kf["P90_m"]
        delta_lsr15 = ms["LSR15_pct"] - kf["LSR15_pct"]
        verdict = "IMPROVED" if delta_mle < 0.0 else ("UNCHANGED" if abs(delta_mle) < 1e-12 else "WORSE")

        print(
            f"{route}: KF MLE={kf['MLE_m']:.3f}m P90={kf['P90_m']:.3f}m LSR@15={kf['LSR15_pct']:.2f}% | "
            f"FinalMS MLE={ms['MLE_m']:.3f}m P90={ms['P90_m']:.3f}m LSR@15={ms['LSR15_pct']:.2f}% | "
            f"dMLE={delta_mle:+.3f}m dP90={delta_p90:+.3f}m dLSR15={delta_lsr15:+.2f}pp => {verdict}"
        )

        summary_rows.append(
            {
                "route": route,
                "frames": int(len(kalman_errors)),
                "kalman_mle_m": kf["MLE_m"],
                "kalman_p90_m": kf["P90_m"],
                "kalman_lsr15_pct": kf["LSR15_pct"],
                "final_ms_mle_m": ms["MLE_m"],
                "final_ms_p90_m": ms["P90_m"],
                "final_ms_lsr15_pct": ms["LSR15_pct"],
                "delta_mle_m_final_minus_kf": delta_mle,
                "delta_p90_m_final_minus_kf": delta_p90,
                "delta_lsr15_pp_final_minus_kf": delta_lsr15,
                "mle_verdict": verdict,
            }
        )

    if not summary_rows:
        raise SystemExit(
            "No evaluation CSVs found. Run: bash run_train_eval.sh eval full"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "kf_vs_final_meanshift_comparison.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("=" * 88)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
