#!/usr/bin/env python3
"""Print the main failure diagnosis from a tracker summary JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "outputs/straight_line_hardms_v2/straight_line_tracker_v2_summary.json"
    )
    if not path.exists():
        raise SystemExit(f"找不到結果檔：{path}")

    results = json.loads(path.read_text(encoding="utf-8"))
    for result in results:
        metric = result["StraightLineTemporalHardMSV2"]
        ccr = float(result["CandidateCaptureRate_pct"])
        jump = float(metric["JumpRate_pct"])
        recovery = int(result["SearchLevelUsage"].get("recovery", 0))
        total = max(1, int(result["sampled_frames"]) - 5)
        recovery_rate = recovery / total * 100.0
        reliable = float(result["ReliableRate_pct"])

        print("=" * 72)
        print(result["route"])
        print(
            f"MLE={metric['MLE_m']:.2f} m | P90={metric['P90_m']:.2f} m | "
            f"LSR@15={metric['LSR@15_pct']:.2f}%"
        )
        print(
            f"CCR={ccr:.2f}% | Jump={jump:.2f}% | "
            f"Recovery usage={recovery_rate:.2f}% | Reliable={reliable:.2f}%"
        )

        problems = []
        if ccr < 90.0:
            problems.append("候選捕獲率過低：主要仍是 motion/search failure，不是 HardMS 本身。")
        if recovery_rate > 50.0:
            problems.append("超過一半影格處於 recovery：信心校準或速度更新仍有問題。")
        if reliable < 30.0:
            problems.append("可靠量測比例過低：檢查 mode_local_mass、spatial_std 與 checkpoint。")
        if jump > 5.0:
            problems.append("仍有明顯跳點：降低 correction cap 或提高 temporal mode penalty。")
        if metric["PathLengthRatio"] < 0.70:
            problems.append("路徑長度過短：速度估計偏小或視覺更新過弱。")
        if metric["PathLengthRatio"] > 1.30:
            problems.append("路徑長度過長：速度估計偏大或 recovery 修正過強。")
        if not problems:
            problems.append("主要追蹤診斷均通過，接著比較 ATE/P90 與 baseline。")

        for problem in problems:
            print(f"- {problem}")


if __name__ == "__main__":
    main()