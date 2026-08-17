#!/usr/bin/env python3
"""Generate oral-defense figures from the original controlled V36 outputs.

This script intentionally does not retrain a network.  It reads the completed
V36 runs, and recomputes only the geometric candidate-coverage sweep using the
same SAT gallery, stride, capture radius, per-frame prior and search heading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from data import RouteDataset, SatPatchGallery  # noqa: E402

OUT = HERE / "figures" / "local_prior_v36"
V36 = HERE / "outputs" / "internal" / "corrected_v2"
FULL = V36 / "full_v36"
FULL6 = V36 / "full_6x6"
# Register this TTC explicitly: the container's Matplotlib cache does not pick
# up system CJK fonts automatically.
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({"font.family": FONT_NAME, "axes.unicode_minus": False, "font.size": 13})

STRIDE_M = 32 * 0.14793025090452439
PATCH_M = 320 * 0.14793025090452439
RADIUS_M = 7.5


def summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_geometry_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-39, 45)
    ax.set_ylim(-28, 28)

    # The three boxes establish the causal story before the search grid.
    boxes = [
        (-37, 11, 20, 11, "真實位置（GT）\n僅供 controlled 實驗建立先驗", "#eef2ff"),
        (-37, -12, 20, 11, "粗略局部先驗\nGT + 平滑抖動（≤ 8 m）", "#fff7ed"),
        (-9, -1, 20, 13, "UAV–SAT 局部比對\n以視覺量測修正位置", "#ecfdf5"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=color, ec="#334155", lw=1.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold")
    ax.add_patch(FancyArrowPatch((-27, 11), (-27, -1), arrowstyle="->", mutation_scale=18, lw=2, color="#475569"))
    ax.add_patch(FancyArrowPatch((-17, -6), (-9, 4), arrowstyle="->", mutation_scale=18, lw=2, color="#475569"))
    ax.text(-27, 4.5, "加入受控誤差", ha="center", color="#9a3412", fontsize=11)

    # Candidate centers: 3 forward rows x 6 cross-track columns.
    ox, oy = 22, -10
    dx, dy = STRIDE_M, STRIDE_M
    for row in range(3):
        for col in range(6):
            x, y = ox + col * dx, oy + row * dy
            ax.add_patch(Rectangle((x - PATCH_M / 2, y - PATCH_M / 2), PATCH_M, PATCH_M,
                                   fc="#dbeafe", ec="#60a5fa", alpha=0.18, lw=0.9))
            ax.plot(x, y, "o", color="#1d4ed8", ms=5)
    ax.plot(ox + 2.5 * dx, oy - 5.5, marker="*", ms=16, color="#f97316", label="粗略先驗中心")
    ax.add_patch(Circle((ox + 2.5 * dx, oy - 5.5), 8, fill=False, ls="--", lw=2, ec="#f97316"))
    ax.add_patch(FancyArrowPatch((ox + 2.5 * dx, oy - 2), (ox + 2.5 * dx, oy + 19),
                                 arrowstyle="->", mutation_scale=18, lw=2.2, color="#047857"))
    ax.text(ox + 2.5 * dx + 1.8, oy + 20, "前向", color="#047857", weight="bold")
    ax.text(ox + 2.5 * dx, oy + 25,
            "V36：3 × 6 = 18 個前向候選中心\n中心間距 4.73 m；橫向中心跨度 23.67 m；前向跨度 9.47 m\n每張 SAT patch：47.34 m × 47.34 m（大量重疊）",
            ha="center", va="top", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#94a3b8"))
    ax.text(0.5, -25.7,
            "口試表述：系統假設已有「含誤差的粗略局部位置」，並以 UAV–SAT 視覺比對修正；不是模型自己知道 UAV 的位置。",
            ha="center", va="center", fontsize=13, weight="bold", color="#0f172a")
    fig.tight_layout()
    fig.savefig(OUT / "01_v36_local_prior_and_3x6_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def original_capture() -> dict:
    full = summary(FULL / "robust_tracker_summary.json")
    six = summary(FULL6 / "robust_tracker_summary.json")
    return {
        "Route A (train)": 92.39,  # repeated training-epoch log of the original V36 run
        "Route B (test)": full["route_B"]["SelectedCandidateCapture_pct"],
        "Route C (test)": full["route_C"]["SelectedCandidateCapture_pct"],
        "Route B 6×6": six["route_B"]["SelectedCandidateCapture_pct"],
        "Route C 6×6": six["route_C"]["SelectedCandidateCapture_pct"],
    }


def make_capture_figure(capture: dict) -> None:
    names = ["Route A\n(train)", "Route B\n(test)", "Route C\n(test)"]
    vals = [capture["Route A (train)"], capture["Route B (test)"], capture["Route C (test)"]]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bars = ax.bar(names, vals, color=["#2563eb", "#0ea5e9", "#14b8a6"], width=0.62)
    ax.axhline(95, ls="--", color="#dc2626", lw=1.7, label="95% 參考線")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Candidate Capture Rate (%)")
    ax.set_title("原始 V36：3×6 前向候選範圍的 Capture Rate")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center", weight="bold")
    ax.text(0.5, -0.19,
            "定義：min(真實 GT 與任一候選中心的距離) ≤ 7.5 m 的 frame 比例。\n"
            "結論：原始 3×6 在此受控先驗設定約為 90%，不能宣稱「超過 97%」。",
            transform=ax.transAxes, ha="center", va="top", fontsize=11.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "02_v36_3x6_candidate_capture_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def geometric_sweep(route_index: int, route_name: str) -> dict[int, float]:
    """Coverage-only sweep: exact V36 gallery/prior/heading; no neural rerun."""
    csv = FULL / f"{route_name}_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv"
    frame = pd.read_csv(csv)
    dataset = RouteDataset(config.ROUTE_ROOTS[route_index], train=False)
    gallery = SatPatchGallery(origin_lat=dataset.origin_lat, origin_lon=dataset.origin_lon)
    xy = np.asarray([[s["x_meter"], s["y_meter"]] for s in gallery.samples])
    pixels = np.asarray([[s["pixel_x"], s["pixel_y"]] for s in gallery.samples], dtype=int)
    _, nearest = cKDTree(xy).query(frame[["prior_center_x", "prior_center_y"]].to_numpy())
    pixel_to_idx = {tuple(pixel): idx for idx, pixel in enumerate(pixels)}
    base = np.asarray([
        [pixel_to_idx[(pixels[idx, 0] + dx * 32, pixels[idx, 1] + dy * 32)]
         for dy in range(-3, 3) for dx in range(-3, 3)]
        for idx in nearest
    ])
    centers = xy[base]
    origin = frame[["prior_center_x", "prior_center_y"]].to_numpy()
    heading = np.deg2rad(frame["search_heading_deg"].to_numpy())
    forward = ((centers - origin[:, None, :]) * np.stack([np.cos(heading), np.sin(heading)], axis=1)[:, None, :]).sum(axis=2)
    gt = frame[["gt_x", "gt_y"]].to_numpy()
    result = {}
    for rows in (2, 3, 4, 6):
        keep = rows * 6
        selected = np.argpartition(forward, -keep, axis=1)[:, -keep:]
        selected_centers = np.take_along_axis(centers, selected[:, :, None], axis=1)
        result[rows] = float((np.linalg.norm(selected_centers - gt[:, None, :], axis=2).min(axis=1) <= RADIUS_M).mean() * 100)
    return result


def make_sweep_figure(capture: dict) -> None:
    sweep_b = geometric_sweep(1, "route_B")
    sweep_c = geometric_sweep(2, "route_C")
    # Use direct logged values for the two configurations that were actually run.
    sweep_b[3], sweep_c[3] = capture["Route B (test)"], capture["Route C (test)"]
    sweep_b[6], sweep_c[6] = capture["Route B 6×6"], capture["Route C 6×6"]
    rows = [2, 3, 4, 6]
    table = pd.DataFrame({
        "search": [f"{r}×6" for r in rows], "candidates": [r * 6 for r in rows],
        "relative_visual_scoring": [r / 3 for r in rows],
        "route_B_capture_pct": [sweep_b[r] for r in rows],
        "route_C_capture_pct": [sweep_c[r] for r in rows],
    })
    table.to_csv(OUT / "v36_capture_geometry_sweep.csv", index=False)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(rows)); w = 0.34
    b = ax0.bar(x - w / 2, [sweep_b[r] for r in rows], w, label="Route B", color="#2563eb")
    c = ax0.bar(x + w / 2, [sweep_c[r] for r in rows], w, label="Route C", color="#14b8a6")
    ax0.set_xticks(x, [f"{r}×6\n({r*6} candidates)" for r in rows])
    ax0.set_ylim(0, 105); ax0.set_ylabel("Candidate Capture Rate (%)")
    ax0.set_title("V36 candidate-size sweep：coverage")
    ax0.legend()
    for bars in (b, c):
        for bar in bars:
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}", ha="center", fontsize=10)
    ax1.bar(["2×6", "3×6", "4×6", "6×6"], [2/3, 1, 4/3, 2], color=["#93c5fd", "#2563eb", "#93c5fd", "#60a5fa"])
    ax1.set_ylim(0, 2.2); ax1.set_ylabel("relative visual scoring cost\n(3×6 = 1.0)")
    ax1.set_title("候選數量 / 視覺比對成本")
    for i, (n, rel) in enumerate(zip([12, 18, 24, 36], [2/3, 1, 4/3, 2])):
        ax1.text(i, rel + .06, f"{n} candidates\n{rel:.2f}×", ha="center", fontsize=10)
    fig.text(0.5, -0.035,
             "註：2×6、4×6 為用相同 V36 frame、SAT gallery、粗略先驗與 heading 重算的「幾何 coverage」；未重訓，故不報 MLE。"
             "3×6、6×6 為完成的 V36 實際 run。",
             ha="center", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "03_v36_candidate_size_capture_cost_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def four_by_six_evidence(route_index: int, route_name: str) -> pd.DataFrame:
    """Return per-frame 4x6 coverage evidence under the original V36 geometry."""
    csv = FULL / f"{route_name}_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv"
    frame = pd.read_csv(csv)
    dataset = RouteDataset(config.ROUTE_ROOTS[route_index], train=False)
    gallery = SatPatchGallery(origin_lat=dataset.origin_lat, origin_lon=dataset.origin_lon)
    xy = np.asarray([[s["x_meter"], s["y_meter"]] for s in gallery.samples])
    pixels = np.asarray([[s["pixel_x"], s["pixel_y"]] for s in gallery.samples], dtype=int)
    _, nearest = cKDTree(xy).query(frame[["prior_center_x", "prior_center_y"]].to_numpy())
    pixel_to_idx = {tuple(pixel): idx for idx, pixel in enumerate(pixels)}
    base = np.asarray([
        [pixel_to_idx[(pixels[idx, 0] + dx * 32, pixels[idx, 1] + dy * 32)]
         for dy in range(-3, 3) for dx in range(-3, 3)]
        for idx in nearest
    ])
    centers = xy[base]
    origin = frame[["prior_center_x", "prior_center_y"]].to_numpy()
    heading = np.deg2rad(frame["search_heading_deg"].to_numpy())
    forward = ((centers - origin[:, None, :]) * np.stack([np.cos(heading), np.sin(heading)], axis=1)[:, None, :]).sum(axis=2)
    # 4 x 6 means retaining the 24 most-forward centers from V36's original 6x6 bank.
    selected = np.argpartition(forward, -24, axis=1)[:, -24:]
    selected_centers = np.take_along_axis(centers, selected[:, :, None], axis=1)
    gt = frame[["gt_x", "gt_y"]].to_numpy()
    distance = np.linalg.norm(selected_centers - gt[:, None, :], axis=2).min(axis=1)
    return pd.DataFrame({
        "route": route_name,
        "frame_id": frame["frame_id"],
        "min_4x6_candidate_center_distance_m": distance,
        "capture_threshold_m": RADIUS_M,
        "captured": distance <= RADIUS_M,
    })


def make_4x6_evidence_figure() -> None:
    data = pd.concat([four_by_six_evidence(1, "route_B"), four_by_six_evidence(2, "route_C")], ignore_index=True)
    data.to_csv(OUT / "v36_4x6_capture_per_frame.csv", index=False)
    stats = data.groupby("route").agg(
        total_frames=("captured", "size"), captured_frames=("captured", "sum"),
        min_distance_m=("min_4x6_candidate_center_distance_m", "min"),
        mean_distance_m=("min_4x6_candidate_center_distance_m", "mean"),
        max_distance_m=("min_4x6_candidate_center_distance_m", "max"),
    )
    stats["missed_frames"] = stats["total_frames"] - stats["captured_frames"]
    stats["capture_rate_pct"] = 100 * stats["captured_frames"] / stats["total_frames"]
    stats.to_csv(OUT / "v36_4x6_capture_summary.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6))
    routes = list(stats.index)
    captured = stats["captured_frames"].to_numpy()
    missed = stats["missed_frames"].to_numpy()
    total = stats["total_frames"].to_numpy()
    ax0.bar(routes, captured, color="#14b8a6", label="Captured (≤ 7.5 m)")
    ax0.bar(routes, missed, bottom=captured, color="#ef4444", label="Missed (> 7.5 m)")
    ax0.set_ylabel("frame count")
    ax0.set_title("V36 4×6：逐幀 coverage 統計")
    for i, route in enumerate(routes):
        ax0.text(i, total[i] * .50, f"{captured[i]} / {total[i]}\n{stats.iloc[i]['capture_rate_pct']:.2f}%", ha="center", va="center", color="white", fontsize=15, weight="bold")
        if missed[i]:
            ax0.text(i, total[i] - missed[i] / 2, f"miss {missed[i]}", ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax0.legend(loc="upper right")

    colors = {"route_B": "#2563eb", "route_C": "#14b8a6"}
    for route in routes:
        vals = np.sort(data.loc[data.route == route, "min_4x6_candidate_center_distance_m"].to_numpy())
        ax1.plot(vals, np.arange(1, len(vals) + 1) / len(vals) * 100, lw=2.5, color=colors[route], label=route)
    ax1.axvline(RADIUS_M, color="#dc2626", ls="--", lw=2, label="capture threshold = 7.5 m")
    ax1.set_xlabel("最近 4×6 候選中心至 GT 距離 (m)")
    ax1.set_ylabel("累積 frame 比例 (%)")
    ax1.set_xlim(left=0); ax1.set_ylim(0, 101)
    ax1.set_title("coverage 的距離證據（ECDF）")
    ax1.legend(loc="lower right", fontsize=10)
    fig.text(0.5, -0.03,
             "Coverage 定義：每一 frame 的 GT 與 24 個 4×6 候選中心中，最近距離 ≤ 7.5 m。\n"
             "因此本圖直接回答「4×6 是否涵蓋 GT」；不代表定位 MLE，MLE 需用 4×6 重新訓練／推論後另行量測。",
             ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "04_v36_4x6_capture_evidence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    capture = original_capture()
    make_geometry_figure()
    make_capture_figure(capture)
    make_sweep_figure(capture)
    make_4x6_evidence_figure()
    (OUT / "README.txt").write_text(
        "Figures are based on original controlled V36 outputs.\n"
        "V36 uses current-frame GT + deterministic smooth jitter as its local prior; it is not autonomous prior prediction.\n"
        "The 3x6 B/C capture rates are 89.63% / 90.46%.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
