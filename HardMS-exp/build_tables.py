#!/usr/bin/env python3
"""Build a traceable result bundle for archived HardMS and new temporal runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
UAV_SAT = ROOT.parent
OUT = ROOT / "tables"
SRC = ROOT / "source_artifacts"
CSV_BACKUP = ROOT / "csv_backup"
LEGACY = STUDY / "UAV_GPS_allmap_imgonly4" / "outputs"
COMPARE = (
    STUDY
    / "Go_aaai"
    / "comapred_paper"
    / "results"
    / "newdata_local36_20260806_222100"
)
BEARING = STUDY / "Go_aaai" / "bearinguav_results"


def pct(value: float) -> float:
    return float(value) * 100.0


def metric_row(pred: np.ndarray, gt: np.ndarray, route: np.ndarray) -> Dict[str, float]:
    """Metrics used by the temporal code; routes are never linked across boundaries."""
    error = np.linalg.norm(pred - gt, axis=1)
    result = {
        "Frames": len(error),
        "MLE_m": float(error.mean()),
        "MedLE_m": float(np.median(error)),
        "P90_m": float(np.percentile(error, 90)),
        "P95_m": float(np.percentile(error, 95)),
        "P99_m": float(np.percentile(error, 99)),
        "CVaR90_m": float(error[error >= np.percentile(error, 90)].mean()),
        "MaxLE_m": float(error.max()),
        "LSR@5_pct": pct((error <= 5).mean()),
        "LSR@10_pct": pct((error <= 10).mean()),
        "LSR@15_pct": pct((error <= 15).mean()),
        "LSR@20_pct": pct((error <= 20).mean()),
    }
    jump_flags: List[np.ndarray] = []
    rpe: List[np.ndarray] = []
    thresholds: List[float] = []
    for name in np.unique(route):
        idx = np.flatnonzero(route == name)
        if len(idx) < 2:
            continue
        pred_step = np.linalg.norm(np.diff(pred[idx], axis=0), axis=1)
        gt_step = np.linalg.norm(np.diff(gt[idx], axis=0), axis=1)
        threshold = float(np.percentile(gt_step, 99) + 3.0)
        jump_flags.append(pred_step > threshold)
        rpe.append(np.linalg.norm(np.diff(pred[idx], axis=0) - np.diff(gt[idx], axis=0), axis=1))
        thresholds.append(threshold)
    result["RPE_m"] = float(np.concatenate(rpe).mean()) if rpe else float("nan")
    result["JumpRate_pct"] = pct(np.concatenate(jump_flags).mean()) if jump_flags else float("nan")
    result["JumpThreshold_m"] = float(np.mean(thresholds)) if thresholds else float("nan")
    return result


def read_temporal(run: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = UAV_SAT / "outputs" / run
    summary = json.loads((base / "robust_tracker_summary.json").read_text())
    rows: List[Dict[str, object]] = []
    for route in ("route_B", "route_C"):
        data = summary["routes"][route]
        # TemporalPathExpectation is an internal decoder state. Formal tables
        # report only the final RTL-CRF output.
        for method in ("RawTop1", "FixedHardMS", "RTL_CRF"):
            row = {"Run": run, "Route": route, "Method": method}
            row.update(data[method])
            rows.append(row)

    frames = []
    for route in ("route_B", "route_C"):
        frame = pd.read_csv(base / f"{route}_robust_frames.csv")
        frame["Route"] = route
        frames.append(frame)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def aggregate_temporal(run: str, frames: pd.DataFrame) -> pd.DataFrame:
    outputs = {
        "RawTop1": ("raw_top1_x", "raw_top1_y"),
        "FixedHardMS": ("hardms_x", "hardms_y"),
        "RTL_CRF": ("temporal_x", "temporal_y"),
    }
    frames = frames.sort_values(["Route", "frame_id"])
    gt = frames[["gt_x", "gt_y"]].to_numpy(float)
    route = frames["Route"].to_numpy()
    rows = []
    for method, cols in outputs.items():
        metric = metric_row(frames[list(cols)].to_numpy(float), gt, route)
        rows.append({"Run": run, "Method": method, **metric})
    return pd.DataFrame(rows)


def read_external(
    name: str, common_ids: Dict[str, set] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(COMPARE / name / "test_predictions.csv")
    if common_ids is not None:
        route_name = frame.route.str.lower().map({
            "route_b": "route_B",
            "route_c": "route_C",
        })
        keep = [
            int(frame_id) in common_ids[str(route)]
            for frame_id, route in zip(frame.frame_id, route_name)
        ]
        frame = frame.loc[keep].copy()
    pred = []
    for _, row in frame.iterrows():
        centers = np.asarray(json.loads(row.candidate_xy_json), dtype=float)
        pred.append(centers[int(row.top1_idx)])
    frame["pred_x"] = np.asarray(pred)[:, 0]
    frame["pred_y"] = np.asarray(pred)[:, 1]
    frame = frame.sort_values(["route", "frame_id"])
    return (
        frame[["pred_x", "pred_y"]].to_numpy(float),
        frame[["gt_x_m", "gt_y_m"]].to_numpy(float),
        frame["route"].to_numpy(),
    )


def read_bearing(trim_first: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = []
    for route, path in (
        ("route_B", BEARING / "newdata_eval_B_test" / "predictions.csv"),
        ("route_C", BEARING / "newdata_eval_C_short" / "predictions.csv"),
    ):
        frame = pd.read_csv(path).sort_values("frame_id")
        if trim_first:
            frame = frame.iloc[trim_first:].copy()
        # Infer the orthomosaic metres-per-pixel from the archived error field.
        pixel_error = np.linalg.norm(
            frame[["pred_pixel_x", "pred_pixel_y"]].to_numpy(float)
            - frame[["gt_pixel_x", "gt_pixel_y"]].to_numpy(float),
            axis=1,
        )
        scale = float(np.median(frame.error_m.to_numpy(float) / pixel_error))
        frame["pred_x"] = frame.pred_pixel_x * scale
        frame["pred_y"] = frame.pred_pixel_y * scale
        frame["gt_x"] = frame.gt_pixel_x * scale
        frame["gt_y"] = frame.gt_pixel_y * scale
        frame["route_norm"] = route
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    return (
        result[["pred_x", "pred_y"]].to_numpy(float),
        result[["gt_x", "gt_y"]].to_numpy(float),
        result.route_norm.to_numpy(),
    )


def markdown_table(
    frame: pd.DataFrame,
    columns: List[str],
    digits: int = 2,
    bold_min: Iterable[str] = (),
    bold_max: Iterable[str] = (),
) -> str:
    table = frame.loc[:, columns].copy()
    best_min = {
        column: float(pd.to_numeric(table[column], errors="coerce").min())
        for column in bold_min
        if column in table
    }
    best_max = {
        column: float(pd.to_numeric(table[column], errors="coerce").max())
        for column in bold_max
        if column in table
    }
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            def render(value: float) -> str:
                text = f"{value:.{digits}f}"
                if column in best_min and np.isclose(value, best_min[column]):
                    return f"**{text}**"
                if column in best_max and np.isclose(value, best_max[column]):
                    return f"**{text}**"
                return text
            table[column] = table[column].map(render)
    header = "| " + " | ".join(table.columns) + " |"
    rule = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def fixed_hardms_metrics(run_name: str) -> pd.DataFrame:
    matches = list((LEGACY / run_name).glob("controlled_decoder_controls_*/metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"No control table for {run_name}")
    frame = pd.read_csv(matches[0])
    return frame[frame.Method.eq("Fixed HardMS (snapped anchor)")].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    CSV_BACKUP.mkdir(parents=True, exist_ok=True)

    legacy_controls = LEGACY / "basinrank_B_fixed_hardms" / "controlled_decoder_controls_basinrank_B_fixed_hardms"
    legacy_metrics = pd.read_csv(legacy_controls / "metrics.csv")
    # The continuous coordinate is a diagnostic intermediate: it may land
    # between real anchors. Fixed HardMS is formally the nearest-anchor
    # (snapped) selection, so only that readout and the raw Top-1 baseline
    # belong in the presentation table.
    legacy_primary = legacy_metrics[legacy_metrics.Method.isin([
        "Top-1 patch center",
        "Fixed HardMS (snapped anchor)",
    ])].copy()
    legacy_primary.to_csv(OUT / "01_archived_hardms_decoder_controls.csv", index=False)

    sensitivity = pd.read_csv(LEGACY / "paper_evidence" / "hardms_sensitivity.csv")
    sensitivity.to_csv(OUT / "02_archived_hardms_sensitivity.csv", index=False)
    transitions = pd.read_json(LEGACY / "paper_evidence" / "hardms_transition_analysis.json")
    transitions.to_csv(OUT / "03_archived_hardms_transition_analysis.csv", index=False)
    latency = json.loads((LEGACY / "paper_evidence" / "hardms_decoder_latency.json").read_text())
    pd.DataFrame([latency]).to_csv(OUT / "04_archived_hardms_decoder_latency.csv", index=False)

    seed_rows = []
    for seed in (2027, 2028, 2029):
        frame = fixed_hardms_metrics(f"basinrank_seed{seed}_fixed_hardms")
        frame.insert(0, "Seed", seed)
        seed_rows.append(frame)
    pd.concat(seed_rows, ignore_index=True).to_csv(OUT / "05_archived_hardms_seed_stability.csv", index=False)

    grid_rows = []
    for grid in (4, 8, 10):
        frame = fixed_hardms_metrics(f"paper_grid{grid}_hardms")
        frame.insert(0, "GridSize", grid)
        grid_rows.append(frame)
    pd.concat(grid_rows, ignore_index=True).to_csv(OUT / "06_archived_hardms_grid_ablation.csv", index=False)

    scale_rows = []
    for scale in (25, 50, 75):
        frame = fixed_hardms_metrics(f"paper_scale{scale}_hardms")
        frame.insert(0, "TrainDataPercent", scale)
        scale_rows.append(frame)
    pd.concat(scale_rows, ignore_index=True).to_csv(OUT / "07_archived_hardms_train_scale.csv", index=False)

    # Keep exact input artifacts that generated the archived tables.
    for source in (
        legacy_controls / "metrics.csv",
        legacy_controls / "per_frame.csv",
        LEGACY / "paper_evidence" / "hardms_sensitivity.csv",
        LEGACY / "paper_evidence" / "hardms_transition_analysis.json",
        LEGACY / "paper_evidence" / "hardms_decoder_latency.json",
    ):
        target = SRC / source.name
        target.write_bytes(source.read_bytes())

    runs = [
        "strict_train_A_test_BC_no_position_scale",
        "strict_train_A_test_BC_no_position_scale_w3",
        "strict_train_A_test_BC_no_position_scale_w4",
        "strict_train_A_test_BC_t2only_w3",
        "strict_train_A_test_BC_t2only_w4",
        "strict_train_A_test_BC_t2only_w5",
    ]
    temporal_route_tables = []
    temporal_combined_tables = []
    for run in runs:
        base = UAV_SAT / "outputs" / run
        if not (base / "robust_tracker_summary.json").exists():
            continue
        route_table, frames = read_temporal(run)
        temporal_route_tables.append(route_table)
        temporal_combined_tables.append(aggregate_temporal(run, frames))
    temporal_route = pd.concat(temporal_route_tables, ignore_index=True)
    temporal_combined = pd.concat(temporal_combined_tables, ignore_index=True)
    temporal_route.to_csv(OUT / "10_temporal_routewise_metrics.csv", index=False)
    temporal_combined.to_csv(OUT / "11_temporal_combined_metrics.csv", index=False)

    # Main visual-temporal comparison: default no-position-scale run, the
    # cleanest completed version. Path expectation and RTL-CRF are separated.
    main_run = "strict_train_A_test_BC_no_position_scale"
    main_rows = temporal_combined[temporal_combined.Run.eq(main_run)].copy()
    main_rows.to_csv(OUT / "12_new_architecture_main_comparison.csv", index=False)
    window_rows = temporal_combined[
        temporal_combined.Method.eq("RTL_CRF")
        & temporal_combined.Run.str.contains("t2only_w")
    ].copy()
    window_rows.to_csv(OUT / "13_t2only_window_ablation.csv", index=False)

    # Compact T2-only comparison: all three runs already have saved per-frame
    # predictions and RTX 3090 latency benchmarks, so no retraining is needed.
    latency_rows = []
    for window in (3, 4, 5):
        run = f"strict_train_A_test_BC_t2only_w{window}"
        metrics = window_rows[window_rows.Run.eq(run)].iloc[0].to_dict()
        timing = json.loads(
            (UAV_SAT / "outputs" / run / "inference_time_benchmark.json").read_text()
        )
        latency_rows.append({
            "Window frames": window,
            "Frames": int(metrics["Frames"]),
            "MLE (m)": metrics["MLE_m"],
            "P90 (m)": metrics["P90_m"],
            "LSR@5 (%)": metrics["LSR@5_pct"],
            "LSR@10 (%)": metrics["LSR@10_pct"],
            "LSR@15 (%)": metrics["LSR@15_pct"],
            "LSR@20 (%)": metrics["LSR@20_pct"],
            "RPE (m)": metrics["RPE_m"],
            "JumpRate (%)": metrics["JumpRate_pct"],
            "Temporal head (ms)": timing["t2_only_rtl_crf_forward_only"]["mean_ms"],
            "End-to-end (ms)": timing["steady_state_one_new_gps"]["preprocessed_uav_to_gps"]["mean_ms"],
        })
    pd.DataFrame(latency_rows).to_csv(OUT / "14_t2only_window_accuracy_latency.csv", index=False)

    # All methods below use their raw Top-1 prediction sequence. Limit every
    # method to exact 4-frame temporal evaluation IDs (3,526 frames), then
    # recompute JumpRate under one definition rather than copying unrelated
    # paper metrics.
    temporal_frames = []
    for route in ("route_B", "route_C"):
        frame = pd.read_csv(
            UAV_SAT / "outputs" / main_run / f"{route}_robust_frames.csv"
        )
        frame["Route"] = route
        temporal_frames.append(frame)
    temporal_frames = pd.concat(temporal_frames, ignore_index=True)
    common_ids = {
        route: set(frame.frame_id.astype(int))
        for route, frame in temporal_frames.groupby("Route")
    }
    comparison = []
    for name, display in (
        ("sample4geo", "Sample4Geo-style (adapted)"),
        ("denseuav", "DenseUAV-style (adapted)"),
        ("game4loc", "Game4Loc-style (adapted)"),
    ):
        pred, gt, route = read_external(name, common_ids)
        comparison.append({"Method": display, "Decoder": "Raw Top-1", **metric_row(pred, gt, route)})

    pred, gt, route = read_bearing(trim_first=4)
    comparison.append({
        "Method": "Bearing-UAV (archived; trimmed)",
        "Decoder": "Coordinate regression",
        **metric_row(pred, gt, route),
    })

    visual = main_rows[main_rows.Method.isin(["RawTop1", "FixedHardMS"])]
    for _, row in visual.iterrows():
        comparison.append({
            "Method": "Our visual branch",
            "Decoder": row.Method,
            **{key: row[key] for key in row.index if key not in {"Run", "Method"}},
        })
    comparison = sorted(
        comparison,
        key=lambda row: {
            "RawTop1": 0,
            "FixedHardMS": 1,
            "Raw Top-1": 2,
            "Coordinate regression": 5,
        }.get(row["Decoder"], 9),
    )
    pd.DataFrame(comparison).to_csv(OUT / "20_common_frame_jump_comparison.csv", index=False)

    # Complete 3,534-frame localization table with normalized column names.
    # Bearing-UAV has no official JumpRate, so this table leaves it blank.
    full_rows = []
    for name, display in (
        ("sample4geo", "Sample4Geo-style (adapted)"),
        ("denseuav", "DenseUAV-style (adapted)"),
        ("game4loc", "Game4Loc-style (adapted)"),
        ("fieldanchor_fixed_hardms", "Archived Fixed HardMS"),
        ("bearinguav_newdata_existing", "Bearing-UAV (archived)"),
    ):
        source = json.loads((COMPARE / name / "test_summary.json").read_text())
        full_rows.append({
            "Method": display,
            "Frames": source["frames"],
            "MLE_m": source["mle_m"],
            "MedLE_m": source["medle_m"],
            "P90_m": source["p90le_m"],
            "CVaR90_m": source["cvar90_m"],
            "LSR@5_pct": pct(source["lsr_at_5"]),
            "LSR@10_pct": pct(source["lsr_at_10"]),
            "LSR@15_pct": pct(source["lsr_at_15"]),
            "LSR@20_pct": pct(source["lsr_at_20"]),
            "R@1_pct": None if source["recall_at_1"] is None else pct(source["recall_at_1"]),
            "MRR": source["mrr"],
            "Online_ms": source["total_online_ms_per_frame"],
        })
    full = pd.DataFrame(full_rows)
    full["JumpRate_pct"] = np.nan
    full.to_csv(OUT / "21_external_full_localization_comparison.csv", index=False)

    archived_short = legacy_primary[legacy_primary.Split.eq("B+C")].copy()
    report = [
        "# HardMS and Temporal Result Summary",
        "",
        "All values are archived experiment outputs or recomputed directly from their per-frame predictions.",
        "",
        "## 1. Archived single-frame HardMS controls (B+C, 3,534 frames)",
        markdown_table(
            archived_short,
            ["Method", "N", "MLE", "MedLE", "P90", "CVaR90", "LSR@10", "LSR@15", "LSR@20", "MaxLE"],
            bold_min=["MLE", "MedLE", "P90", "CVaR90", "MaxLE"],
            bold_max=["LSR@10", "LSR@15", "LSR@20"],
        ),
        "",
        "## 2. New temporal architecture final output (B+C, 3,526 frames)",
        markdown_table(
            main_rows,
            ["Method", "Frames", "MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct", "RPE_m", "JumpRate_pct"],
            bold_min=["MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "RPE_m", "JumpRate_pct"],
            bold_max=["LSR@10_pct", "LSR@15_pct", "LSR@20_pct"],
        ),
        "",
        "## 3. Common-frame jump comparison (3,526 frames)",
        markdown_table(
            pd.DataFrame(comparison),
            ["Method", "Decoder", "MLE_m", "P90_m", "CVaR90_m", "LSR@15_pct", "LSR@20_pct", "RPE_m", "JumpRate_pct"],
            bold_min=["MLE_m", "P90_m", "CVaR90_m", "RPE_m", "JumpRate_pct"],
            bold_max=["LSR@15_pct", "LSR@20_pct"],
        ),
        "",
        "`JumpRate`: a predicted adjacent-frame displacement greater than the route-specific 99th-percentile GT displacement plus 3 m. External rows are local-36 adaptations; Bearing-UAV's value is recomputed here from archived predictions, not copied from its paper.",
        "",
        "## 4. Final RTL-CRF temporal-window ablation",
        markdown_table(window_rows, ["Run", "Method", "Frames", "MLE_m", "P90_m", "LSR@15_pct", "RPE_m", "JumpRate_pct"]),
        "",
        "## 5. Archived decoder overhead",
        markdown_table(pd.DataFrame([latency]), list(latency)),
        "",
        "## 6. Archived Fixed HardMS seed stability",
        markdown_table(pd.concat(seed_rows, ignore_index=True), ["Seed", "Split", "MLE", "P90", "CVaR90", "LSR@15", "LSR@20"]),
        "",
        "## 7. Archived grid-size and training-scale ablations",
        markdown_table(pd.concat(grid_rows, ignore_index=True), ["GridSize", "Split", "MLE", "P90", "CVaR90", "LSR@15", "LSR@20"]),
        "",
        markdown_table(pd.concat(scale_rows, ignore_index=True), ["TrainDataPercent", "Split", "MLE", "P90", "CVaR90", "LSR@15", "LSR@20"]),
        "",
        "The temporal model uses a controlled GT-jitter local prior and contiguous frame windows. It is therefore reported separately from independent-frame retrieval methods rather than claimed as a directly interchangeable baseline.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    def save_comparison_plot(frame: pd.DataFrame, labels: List[str], filename: str, title: str) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
        colors = ["#6b7280"] * len(frame)
        colors[-1] = "#007c91"
        axes[0].bar(labels, frame["MLE_m"], color=colors)
        axes[0].set_title("Mean localization error")
        axes[0].set_ylabel("Error (m)")
        axes[1].bar(labels, frame["JumpRate_pct"], color=colors)
        axes[1].set_title("Jump rate")
        axes[1].set_ylabel("Frames (%)")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=30, labelsize=8)
        figure.suptitle(title, fontsize=13)
        for suffix in ("png", "pdf"):
            figure.savefig(ROOT / "figures" / f"{filename}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(figure)

    visual_plot = pd.DataFrame(comparison)
    save_comparison_plot(
        visual_plot,
        ["Sample4Geo", "DenseUAV", "Game4Loc", "Bearing-UAV", "Ours Raw", "Ours HardMS"],
        "01_common_frame_visual_comparison",
        "Independent visual localization on the common 3,526-frame subset",
    )
    temporal_plot = main_rows.copy()
    save_comparison_plot(
        temporal_plot,
        ["Raw Top-1", "Fixed HardMS", "RTL-CRF"],
        "02_temporal_architecture_comparison",
        "New temporal architecture on the 3,526-frame subset",
    )

    table_titles = {
        "01_archived_hardms_decoder_controls": "Archived Fixed HardMS Decoder Controls",
        "02_archived_hardms_sensitivity": "Archived Fixed HardMS Sensitivity",
        "03_archived_hardms_transition_analysis": "Archived Fixed HardMS Transition Analysis",
        "04_archived_hardms_decoder_latency": "Archived Fixed HardMS Decoder Latency",
        "05_archived_hardms_seed_stability": "Archived Fixed HardMS Seed Stability",
        "06_archived_hardms_grid_ablation": "Archived Fixed HardMS Grid Ablation",
        "07_archived_hardms_train_scale": "Archived Fixed HardMS Training-Scale Ablation",
        "10_temporal_routewise_metrics": "New Temporal Architecture: Route-wise Metrics",
        "11_temporal_combined_metrics": "New Temporal Architecture: Combined Metrics",
        "12_new_architecture_main_comparison": "New Temporal Architecture: Main Comparison",
        "13_t2only_window_ablation": "New Temporal Architecture: Final RTL-CRF Window Ablation",
        "14_t2only_window_accuracy_latency": "T2-only Window: Accuracy and Latency",
        "20_common_frame_jump_comparison": "Common-frame Jump Comparison",
        "21_external_full_localization_comparison": "Full External Localization Comparison",
    }
    highlight_rules = {
        "12_new_architecture_main_comparison": (
            ["MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "RPE_m", "JumpRate_pct"],
            ["LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct"],
        ),
        "13_t2only_window_ablation": (
            ["MLE_m", "P90_m", "RPE_m", "JumpRate_pct"],
            ["LSR@15_pct"],
        ),
        "14_t2only_window_accuracy_latency": (
            ["MLE (m)", "P90 (m)", "RPE (m)", "JumpRate (%)", "Temporal head (ms)", "End-to-end (ms)"],
            ["LSR@5 (%)", "LSR@10 (%)", "LSR@15 (%)", "LSR@20 (%)"],
        ),
        "20_common_frame_jump_comparison": (
            ["MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "RPE_m", "JumpRate_pct"],
            ["LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct"],
        ),
        "21_external_full_localization_comparison": (
            ["MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "Online_ms"],
            ["LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct", "R@1_pct", "MRR"],
        ),
    }
    # Tables are presented only as Markdown. CSV is kept as an audit backup.
    for csv_path in sorted(OUT.glob("*.csv")):
        backup_path = CSV_BACKUP / csv_path.name
        csv_path.replace(backup_path)
        frame = pd.read_csv(backup_path)
        title = table_titles.get(csv_path.stem, csv_path.stem.replace("_", " ").title())
        lower, higher = highlight_rules.get(csv_path.stem, ([], []))
        markdown = [
            f"# {title}",
            "",
            markdown_table(frame, list(frame.columns), bold_min=lower, bold_max=higher),
            "",
        ]
        (OUT / f"{csv_path.stem}.md").write_text("\n".join(markdown), encoding="utf-8")

    best_temporal = temporal_combined[
        temporal_combined.Method.eq("RTL_CRF")
    ].sort_values("MLE_m").iloc[[0]]
    overview = [
        "# 先看這裡：HardMS 與新架構總覽",
        "",
        "**這份正式總覽只保留 Route A 訓練、Route B+C 測試的結果。資料來源已包含 `uav-sat/outputs/` 內所有符合此協議且已完成的 runs。**",
        "",
        "## 舊 HardMS：單幀視覺定位（完整 B+C，3,534 frames）",
        markdown_table(
            archived_short,
            ["Method", "MLE", "P90", "CVaR90", "LSR@10", "LSR@15", "LSR@20", "MaxLE"],
            bold_min=["MLE", "P90", "CVaR90", "MaxLE"],
            bold_max=["LSR@10", "LSR@15", "LSR@20"],
        ),
        "",
        "## 新架構：最終 RTL-CRF 輸出（B+C，3,526 frames）",
        markdown_table(
            main_rows,
            ["Method", "MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct", "RPE_m", "JumpRate_pct"],
            bold_min=["MLE_m", "MedLE_m", "P90_m", "CVaR90_m", "RPE_m", "JumpRate_pct"],
            bold_max=["LSR@10_pct", "LSR@15_pct", "LSR@20_pct"],
        ),
        "",
        "## 所有已完成時序 runs 中，MLE 最佳設定",
        markdown_table(
            best_temporal,
            ["Run", "Method", "Frames", "MLE_m", "P90_m", "CVaR90_m", "LSR@15_pct", "LSR@20_pct", "RPE_m", "JumpRate_pct"],
            bold_min=["MLE_m", "P90_m", "CVaR90_m", "RPE_m", "JumpRate_pct"],
            bold_max=["LSR@15_pct", "LSR@20_pct"],
        ),
        "",
        "## 注意",
        "- 新時序架構使用連續 frame 與 controlled GT-jitter local prior；它必須和單幀 HardMS 分開解讀。",
        "- 正式表格只保留最終 `RTL_CRF`；中間 path expectation 不列為獨立方法。",
        "- 更完整的 route-wise 與 final-window 消融在 `10–13` 表格。",
        "",
    ]
    (OUT / "00_READ_FIRST_NEW_VS_HARDMS.md").write_text("\n".join(overview), encoding="utf-8")

    # One reader-facing comparison. The temporal row remains in the same table
    # but explicitly declares its four-frame context, so it is visible without
    # pretending to be a like-for-like single-frame baseline.
    common_frame = pd.DataFrame(comparison).copy()
    protocol_notes = {
        "MobileCLIP basic (adapted)": "Local-36 reimplementation",
        "Sample4Geo-style (adapted)": "Local-36 adaptation",
        "DenseUAV-style (adapted)": "Local-36 adaptation",
        "Game4Loc-style (adapted)": "Local-36 adaptation",
        "Bearing-UAV (archived; trimmed)": "Archived coordinate regressor; not Local-36",
        "Our visual branch": "Our shared visual branch",
    }
    common_frame["Protocol note"] = common_frame["Method"].map(protocol_notes)
    common_frame.loc[common_frame.Decoder.eq("RawTop1"), "Method"] = "Baseline: MobileCLIP + dual MLP (Top-1)"
    common_frame.loc[common_frame.Decoder.eq("FixedHardMS"), "Method"] = "Ours: Fixed HardMS"
    common_frame["Method"] = common_frame["Method"].replace({
        "Sample4Geo-style (adapted)": "Sample4Geo",
        "DenseUAV-style (adapted)": "DenseUAV",
        "Game4Loc-style (adapted)": "Game4Loc",
        "Bearing-UAV (archived; trimmed)": "Bearing-UAV",
    })
    common_frame["_presentation_order"] = pd.Categorical(
        common_frame["Method"],
        categories=[
            "Baseline: MobileCLIP + dual MLP (Top-1)",
            "Ours: Fixed HardMS",
            "Sample4Geo",
            "DenseUAV",
            "Game4Loc",
            "Bearing-UAV",
        ],
        ordered=True,
    )
    common_frame = common_frame.sort_values("_presentation_order").drop(columns="_presentation_order")
    visual_columns = [
        "Method", "Decoder", "Frames", "MLE_m", "P90_m", "CVaR90_m",
        "LSR@15_pct", "LSR@20_pct", "RPE_m", "JumpRate_pct", "Protocol note",
    ]

    common_frame["Input context"] = "Single frame"
    final_temporal = main_rows[main_rows.Method.eq("RTL_CRF")].copy()
    final_temporal["Method"] = "Ours: RTL-CRF (4-frame)"
    combined = pd.concat([common_frame, final_temporal], ignore_index=True, sort=False)
    combined_columns = [
        "Method", "MLE_m", "P90_m", "LSR@5_pct", "LSR@10_pct",
        "LSR@15_pct", "LSR@20_pct", "RPE_m", "JumpRate_pct",
    ]
    presentation = combined.loc[:, combined_columns].rename(columns={
        "MLE_m": "MLE (m)",
        "P90_m": "P90 (m)",
        "LSR@5_pct": "LSR@5 (%)",
        "LSR@10_pct": "LSR@10 (%)",
        "LSR@15_pct": "LSR@15 (%)",
        "LSR@20_pct": "LSR@20 (%)",
        "RPE_m": "RPE (m)",
        "JumpRate_pct": "JumpRate (%)",
    })
    big_table = [
        "# Large Comparison: Our Methods and Prior Architectures",
        "",
        markdown_table(
            presentation, list(presentation.columns),
            bold_min=["MLE (m)", "P90 (m)", "RPE (m)", "JumpRate (%)"],
            bold_max=["LSR@5 (%)", "LSR@10 (%)", "LSR@15 (%)", "LSR@20 (%)"],
        ),
        "",
    ]
    (OUT / "22_large_method_comparison.md").write_text("\n".join(big_table), encoding="utf-8")

    note = """# Result bundle\n\n- `01`--`04`: archived P320/S32 Fixed HardMS control, sensitivity, transition, and latency results.\n- `10`--`13`: new temporal RTL-CRF outputs. The main run is `strict_train_A_test_BC_no_position_scale`.\n- `20`: common-frame jump comparison. Every row uses the same 3,526 evaluation frames. External rows are local-36 adaptations, not official author benchmark reproductions.\n- `21`: complete 3,534-frame external localization table. Bearing-UAV is archived coordinate regression and has no author-reported JumpRate.\n\n**JumpRate definition.** For each route independently, a predicted step is a jump when its length exceeds the route's 99th-percentile GT step length plus 3 m. No trajectory links cross route boundaries.\n\n**Protocol warning.** All local-grid runs use a GT-jitter controlled local candidate prior. Results measure controlled local localization, not unconstrained global localization. The temporal rows additionally use contiguous frames, therefore they must not be ranked as a fair replacement for independent-frame external retrieval methods.\n"""
    (ROOT / "README.md").write_text(note, encoding="utf-8")
    print(f"Wrote tables to {OUT}")


if __name__ == "__main__":
    main()
