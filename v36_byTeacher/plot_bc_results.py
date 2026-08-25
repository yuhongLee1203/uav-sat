#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def find_pair(df, candidates):
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_csv(root: Path, route: str, method: str, frame_count: int, search: str):
    csvs = list(root.rglob("*.csv"))
    csvs = [p for p in csvs if route in p.name or route in str(p)]

    if method == "full":
        csvs = [p for p in csvs if "meanshift" not in p.name.lower()]
        csvs = [p for p in csvs if f"{frame_count}frame" in str(p).lower() or f"frame{frame_count}" in str(p).lower() or "experiment_summary" not in p.name.lower()]
    else:
        csvs = [p for p in csvs if "meanshift" in p.name.lower()]

    if search:
        csvs2 = [p for p in csvs if search.lower() in str(p).lower()]
        if csvs2:
            csvs = csvs2

    # 排除 summary 類
    csvs = [p for p in csvs if "summary" not in p.name.lower() and "experiment_summary" not in p.name.lower()]

    # 優先選 frames 類
    preferred = []
    for p in csvs:
        s = str(p).lower()
        score = 0
        if "frame" in s:
            score += 10
        if route.lower() in s:
            score += 5
        if search.lower() in s:
            score += 3
        if method == "full" and f"{frame_count}frame" in s:
            score += 2
        preferred.append((score, p))

    if not preferred:
        return None
    preferred.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return preferred[0][1]


def plot_one(csv_path: Path, out_dir: Path, title_prefix: str):
    df = pd.read_csv(csv_path)

    ref_x, ref_y = find_pair(df, [
        ("reference_x", "reference_y"),
        ("route_x", "route_y"),
        ("planned_x", "planned_y"),
        ("gt_x", "gt_y"),
        ("ref_x", "ref_y"),
        ("reference_s", "reference_e"),
        ("route_s", "route_e"),
        ("gt_s", "gt_e"),
    ])

    final_x, final_y = find_pair(df, [
        ("final_x", "final_y"),
        ("pred_x", "pred_y"),
        ("posterior_x", "posterior_y"),
        ("kf_x", "kf_y"),
        ("final_s", "final_e"),
        ("pred_s", "pred_e"),
        ("posterior_s", "posterior_e"),
        ("kf_s", "kf_e"),
    ])

    ms_x, ms_y = find_pair(df, [
        ("current_softms_x", "current_softms_y"),
        ("softms_x", "softms_y"),
        ("ms_x", "ms_y"),
        ("current_ms_x", "current_ms_y"),
        ("softms_s", "softms_e"),
        ("ms_s", "ms_e"),
        ("current_softms_s", "current_softms_e"),
    ])

    err_final = find_col(df, [
        "error_final_m", "final_error_m", "kf_error_m", "pred_error_m", "error_m"
    ])
    err_ms = find_col(df, [
        "error_softms_m", "softms_error_m", "ms_error_m"
    ])

    frame_col = find_col(df, ["frame_idx", "frame", "index"])
    if frame_col is None:
        df["frame_idx_auto"] = range(len(df))
        frame_col = "frame_idx_auto"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 路徑圖
    plt.figure(figsize=(8, 6))
    if ref_x and ref_y:
        plt.plot(df[ref_x], df[ref_y], label="Reference route", linewidth=2)
    if ms_x and ms_y:
        plt.plot(df[ms_x], df[ms_y], label="MeanShift", linewidth=1.5)
    if final_x and final_y:
        plt.plot(df[final_x], df[final_y], label="Final localization", linewidth=1.5)

    if ref_x and ref_y and len(df) > 0:
        plt.scatter(df[ref_x].iloc[0], df[ref_y].iloc[0], s=50, marker="o", label="Start")
        plt.scatter(df[ref_x].iloc[-1], df[ref_y].iloc[-1], s=60, marker="x", label="End")

    plt.title(f"{title_prefix} trajectory")
    plt.xlabel("X / s")
    plt.ylabel("Y / e")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_trajectory.png", dpi=200)
    plt.close()

    # 誤差圖
    if err_final or err_ms:
        plt.figure(figsize=(9, 4.8))
        if err_ms:
            plt.plot(df[frame_col], df[err_ms], label="MeanShift error")
        if err_final:
            plt.plot(df[frame_col], df[err_final], label="Final localization error")
        plt.title(f"{title_prefix} localization error")
        plt.xlabel("Frame")
        plt.ylabel("Error (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{title_prefix}_error.png", dpi=200)
        plt.close()

    # 終端機印出欄位檢查
    print(f"[OK] {title_prefix}")
    print(f"  CSV: {csv_path}")
    print(f"  ref pair   : {ref_x}, {ref_y}")
    print(f"  final pair : {final_x}, {final_y}")
    print(f"  ms pair    : {ms_x}, {ms_y}")
    print(f"  err_final  : {err_final}")
    print(f"  err_ms     : {err_ms}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/yh/study/uav-sat/v36_byTeacher/output/mobilenet_v3_small")
    parser.add_argument("--method", type=str, default="full", choices=["full", "meanshift"])
    parser.add_argument("--frame-count", type=int, default=2)
    parser.add_argument("--search", type=str, default="3x6")
    args = parser.parse_args()

    root = Path(args.root)
    plot_dir = root / "plots"

    for route in ["route_B", "route_C"]:
        csv_path = choose_csv(root, route, args.method, args.frame_count, args.search)
        if csv_path is None:
            print(f"[WARN] 找不到 {route} 的 CSV (method={args.method}, frame={args.frame_count}, search={args.search})")
            continue

        prefix = f"{route}_{args.method}_{args.frame_count}frame_{args.search}" if args.method == "full" else f"{route}_meanshift_{args.search}"
        plot_one(csv_path, plot_dir, prefix)

    print(f"plots saved in: {plot_dir}")


if __name__ == "__main__":
    main()
