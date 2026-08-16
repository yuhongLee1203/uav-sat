#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "v36-exp"
INTERNAL = EXP / "outputs/internal/corrected_v2"
PAPERS = EXP / "outputs/papers"
NEED = EXP / "need.md"
RESULTS = EXP / "results.md"
BEGIN = "<!-- V36_EXP_RESULTS_BEGIN -->"
END = "<!-- V36_EXP_RESULTS_END -->"


def f(value, digits=3):
    return "PENDING" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def internal(name):
    root = INTERNAL / name
    summary_path = root / "robust_tracker_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = []
    for route in ("route_B", "route_C"):
        paths = list(root.glob(f"{route}_*_frames.csv"))
        if not paths:
            return None
        with paths[0].open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    errors = np.asarray([float(r["error_final_m"]) for r in rows])
    latency = np.asarray([float(r["end_to_end_latency_ms"]) for r in rows])
    latency = latency[np.isfinite(latency)]
    mode_counts = (
        np.asarray([int(float(r["softms_mode_count"])) for r in rows])
        if rows and "softms_mode_count" in rows[0] else np.asarray([], dtype=int)
    )
    return {
        "n": len(rows), "mle": errors.mean(), "median": np.median(errors),
        "p90": np.quantile(errors, .90), "p95": np.quantile(errors, .95),
        "lsr3": (errors <= 3).mean() * 100,
        "lsr5": (errors <= 5).mean() * 100, "lsr10": (errors <= 10).mean() * 100,
        "lsr15": (errors <= 15).mean() * 100, "lsr20": (errors <= 20).mean() * 100,
        "progress": np.mean([float(r["progress_error_m"]) for r in rows]),
        "speed": np.mean([float(r["speed_error_m_per_frame"]) for r in rows]),
        "jump": np.mean([int(float(r["abnormal_jump"])) for r in rows]) * 100,
        "maxstep": max(float(r["final_step_m"]) for r in rows),
        "ms": float(latency.mean()) if len(latency) else None,
        "fps": float(1000 / latency.mean()) if len(latency) else None,
        "mode_count": float(mode_counts.mean()) if len(mode_counts) else None,
        "mode_counts": mode_counts.tolist(),
        "summary": summary,
    }


def paper(name):
    path = PAPERS / name / "summary.json"
    if not path.exists(): return None
    return json.loads(path.read_text())["combined"]


def cols(method, value, keys):
    return "| " + method + " | " + " | ".join(f(value.get(k) if value else None) for k in keys) + " |"


def cols_with_last_precision(method, value, keys, last_digits=6):
    cells = []
    for index, key in enumerate(keys):
        cells.append(f(value.get(key) if value else None, last_digits if index == len(keys) - 1 else 3))
    return "| " + method + " | " + " | ".join(cells) + " |"


def bold_column_winners(markdown):
    """Bold the best numeric value in every result-table column.

    Error/time/step columns are minimised; LSR and FPS are maximised.  Ties
    remain ties, which is important for methods that produce identical runs.
    """
    lines = markdown.splitlines()
    result = []
    index = 0
    section = ""
    while index < len(lines):
        if lines[index].startswith("## "):
            section = lines[index]
        if (
            index + 2 < len(lines)
            and lines[index].startswith("|")
            and lines[index + 1].startswith("|---")
        ):
            header = [x.strip() for x in lines[index].strip("|").split("|")]
            rows = []
            end = index + 2
            while end < len(lines) and lines[end].startswith("|"):
                rows.append([x.strip() for x in lines[end].strip("|").split("|")])
                end += 1
            winners = {}
            for column in range(1, len(header)):
                if "表 7" in section or "表 8" in section:
                    continue
                if header[column] in {"候選數", "聚合座標數", "Frames", "frames"}:
                    continue
                values = []
                for row_index, row in enumerate(rows):
                    try:
                        values.append((row_index, float(row[column].replace("**", "").replace("%", ""))))
                    except (ValueError, IndexError):
                        pass
                # Do not crown a winner while the comparison row is PENDING.
                if len(values) < 2:
                    continue
                higher_is_better = "LSR" in header[column] or "FPS" in header[column]
                best = (max if higher_is_better else min)(value for _, value in values)
                winners[column] = {row for row, value in values if abs(value - best) < 1e-12}
            result.extend(lines[index:index + 2])
            for row_index, row in enumerate(rows):
                for column, winner_rows in winners.items():
                    if row_index in winner_rows:
                        row[column] = "**" + row[column] + "**"
                result.append("| " + " | ".join(row) + " |")
            index = end
        else:
            result.append(lines[index])
            index += 1
    return "\n".join(result)


def build():
    v = {name: internal(name) for name in [
        "full_v36", "weighted_centroid", "full_6x6", "forward_3x6_aligned", "frame1", "frame2",
        "softms_only", "softms_gru", "softms_gru_poly",
        "motion_kalman_cv", "motion_velocity",
    ]}
    weighted_aggregation_ms = None
    aggregation_path = EXP / "outputs/aggregation_benchmark.json"
    if aggregation_path.exists():
        aggregation = json.loads(aggregation_path.read_text())
        weighted_aggregation_ms = aggregation.get("weighted_centroid_ms_per_frame")
        if v["weighted_centroid"]:
            v["weighted_centroid"]["aggregation_ms"] = aggregation.get("weighted_centroid_ms_per_frame")
        if v["full_v36"]:
            timing = aggregation.get("softms_ms_per_frame_by_mode_count", {})
            count = v["full_v36"].get("mode_count")
            counts = v["full_v36"].get("mode_counts", [])
            if timing and count is not None and counts:
                per_frame = [timing[str(max(1, min(18, k)))] for k in counts]
                v["full_v36"]["aggregation_ms"] = float(np.mean(per_frame))
    lines = ["# V36 實驗自動彙整", "", "尚未完成的工作會顯示 `PENDING`。", ""]
    lines += ["## 表 1：SoftMS vs Weighted Centroid", "", "| 定位方式 | 聚合座標數 | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 | LSR@15 | 純座標聚合時間 (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
              "| Weighted Centroid | 18 | " + " | ".join([f(v["weighted_centroid"].get(k) if v["weighted_centroid"] else None) for k in ["mle","p90","lsr3","lsr5","lsr10","lsr15"]] + [f(weighted_aggregation_ms, 9)]) + " |",
              "| SoftMS 收斂 modes 加權 | " + f(v["full_v36"].get("mode_count") if v["full_v36"] else None, 2) + " | " + " | ".join([f(v["full_v36"].get(k) if v["full_v36"] else None) for k in ["mle","p90","lsr3","lsr5","lsr10","lsr15"]] + [f(v["full_v36"].get("aggregation_ms") if v["full_v36"] else None, 9)]) + " |", "",
              "聚合座標數是逐幀平均。時間只計算候選座標加權求和，不含圖片、backbone、matching、MeanShift、GRU 或 Kalman；不計 FPS。", ""]
    lines += ["## 表 2：V36 主要架構消融", "", "| 方法 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | Progress MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|",
              cols("SoftMS only", v["softms_only"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]),
              cols("SoftMS + 3-frame GRU", v["softms_gru"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]),
              cols("SoftMS + GRU + 慣性多項式", v["softms_gru_poly"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]),
              cols("完整 V36（含 learned-variance Kalman）", v["full_v36"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]), ""]
    lines += ["## 表 3：Forward 3×6 vs 6×6", "", "| 搜尋方式 | 候選數 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | 端到端時間 (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
              "| 完整 6×6 | 36 | " + " | ".join(f(v["full_6x6"].get(k) if v["full_6x6"] else None) for k in ["mle","p90","lsr3","lsr5","lsr10","lsr15","ms"]) + " |",
              "| Forward 3×6（causal-origin 修正版） | 18 | " + " | ".join(f(v["forward_3x6_aligned"].get(k) if v["forward_3x6_aligned"] else None) for k in ["mle","p90","lsr3","lsr5","lsr10","lsr15","ms"]) + " |", "",
              "3×6 的 origin backshift 固定為一個 gallery cell（4.75 m）；它由 Route-A/grid geometry 決定，沒有用 Route B/C 挑參數。", ""]
    lines += ["## 表 4：為什麼要三幀 GRU", "", "| 輸入影像數量 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | Progress MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|",
              cols("1 幀", v["frame1"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]),
              cols("2 幀", v["frame2"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]),
              cols("3 幀（V36）", v["full_v36"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress"]), ""]
    lines += ["## 表 5：慣性多項式實驗", "", "| 運動預測方式 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | Progress MAE | Speed MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
              cols("Kalman CV（不使用 learned polynomial）", v["motion_kalman_cv"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress","speed"]),
              cols("只使用 GRU 速度", v["motion_velocity"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress","speed"]),
              cols("GRU 速度 + 加速度二階多項式（V36）", v["full_v36"], ["mle","p90","lsr3","lsr5","lsr10","lsr15","progress","speed"]), "",
              "MAE = Mean Absolute Error（平均絕對誤差）；Progress MAE 是沿路徑進度誤差，Speed MAE 是每幀速度誤差。", "",
              "舊版「不使用運動預測」曾把每幀位移強制設為 0，但又保留 Kalman 每幀最多修正 3 m 的限制，因而人為累積出數百公尺落後；該數據無效。修正版是不使用 learned polynomial，但保留外部 Kalman 自身的 constant-velocity prediction。", ""]
    lines += ["## 表 6：Kalman / Measurement Variance", "", "| 最後輸出方式 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 |", "|---|---:|---:|---:|---:|---:|---:|",
              cols("不使用 Kalman", v["softms_gru_poly"], ["mle","p90","lsr3","lsr5","lsr10","lsr15"]),
              cols("完整 V36 learned-variance Kalman", v["full_v36"], ["mle","p90","lsr3","lsr5","lsr10","lsr15"]), ""]
    lines += ["## 表 7：最終 Route B / Route C", "", "| 路徑 | MLE | Median | P90 | P95 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | LSR@20 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    full = v["full_v36"]
    for route in ("route_B", "route_C"):
        s = full["summary"][route] if full else None
        vals = ["MLE_m","MedLE_m","P90_m","P95_m","LSR@3_pct","LSR@5_pct","LSR@10_pct","LSR@15_pct","LSR@20_pct"]
        if s and "LSR@3_pct" not in s:
            route_rows = []
            for path in (INTERNAL / "full_v36").glob(f"{route}_*_frames.csv"):
                with path.open(newline="") as handle: route_rows.extend(csv.DictReader(handle))
            s["LSR@3_pct"] = np.mean([float(row["error_final_m"]) <= 3 for row in route_rows]) * 100
        lines.append("| " + route.replace("route_", "Route ") + " | " + " | ".join(f(s.get(k) if s else None) for k in vals) + " |")
    lines.append(cols("平均（逐幀合併）", full, ["mle","median","p90","p95","lsr3","lsr5","lsr10","lsr15","lsr20"])); lines.append("")
    lines += ["## 表 8：其他論文原生協定比較", "", "| 方法 | 原生定位協定 | MLE | Median | P90 | P95 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | FPS |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("DenseUAV","Sample4Geo","Game4Loc","InfoGeo","Bearing-UAV"):
        protocol = "Global retrieval" if name != "Bearing-UAV" else "Neighbor-map position/heading regression"
        lines.append("| " + name + " | " + protocol + " | " + " | ".join(["PENDING"] * 9) + " |")
    lines.append("| V36（Ours） | GT+jitter Forward-3×6 local tracking | " + " | ".join(f(full.get(k) if full else None) for k in ["mle","median","p90","p95","lsr3","lsr5","lsr10","lsr15","fps"]) + " |")
    lines += ["", "舊版約 12 m 的 local-18 adapter 結果已判定無效，不再列入正式比較；表 8 必須由各官方模型的原生 retrieval/regression 流程重新產生。", ""]
    return bold_column_winners("\n".join(lines))


def main():
    text = build()
    RESULTS.write_text(text, encoding="utf-8")
    managed = BEGIN + "\n\n" + text + "\n" + END
    # need.md used to retain the hand-written legacy tables above this managed
    # block.  That left removed metrics (JumpRate/fixed-R/Max Step) visible next
    # to the corrected tables.  It is now a single generated source of truth.
    NEED.write_text(managed + "\n", encoding="utf-8")
    print(RESULTS)
    print(NEED)


if __name__ == "__main__": main()
