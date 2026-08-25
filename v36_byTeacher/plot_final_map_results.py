#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def find_pair(df, candidates):
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None


def choose_csv(root: Path, route: str, frame_count: int, search: str):
    csvs = list(root.rglob("*.csv"))
    keep = []
    for p in csvs:
        s = str(p).lower()
        if "experiment_summary" in s or "meanshift" in s or "summary" in s:
            continue
        if route.lower() not in s:
            continue
        if search.lower() not in s:
            continue
        if f"{frame_count}frame" not in s and f"frame{frame_count}" not in s:
            continue
        keep.append(p)

    if not keep:
        return None

    scored = []
    for p in keep:
        s = str(p).lower()
        score = 0
        if "frame" in s:
            score += 5
        if route.lower() in s:
            score += 5
        if search.lower() in s:
            score += 5
        if "full" in s or "teacher" in s or "v36_byteacher" in s:
            score += 2
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return scored[0][1]


def auto_find_map(repo_root: Path, route: str):
    route_key = route.lower()
    patterns = [
        f"*{route_key}*map*.png",
        f"*{route_key}*map*.jpg",
        f"*{route_key}*sat*.png",
        f"*{route_key}*sat*.jpg",
        f"*{route_key}*background*.png",
        f"*{route_key}*background*.jpg",
        f"*{route_key}*overview*.png",
        f"*{route_key}*overview*.jpg",
    ]

    all_files = []
    for pat in patterns:
        all_files.extend(repo_root.rglob(pat))

    # 優先 route_waypoints / figures / output 內的圖
    all_files = sorted(set(all_files), key=lambda p: (
        0 if "route_waypoints" in str(p) else
        1 if "figures" in str(p) else
        2 if "output" in str(p) else
        3,
        len(str(p))
    ))
    return all_files[0] if all_files else None


def metric_to_pixel(xs, ys, w, h, all_x, all_y):
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    if xmax - xmin < 1e-9:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    px = [(x - xmin) / (xmax - xmin) * (w - 1) for x in xs]
    py = [(1.0 - (y - ymin) / (ymax - ymin)) * (h - 1) for y in ys]
    return px, py


def draw_one(csv_path: Path, bg_path: Path, out_dir: Path, title: str):
    df = pd.read_csv(csv_path)

    ref_x, ref_y = find_pair(df, [
        ("reference_x", "reference_y"),
        ("ref_x", "ref_y"),
        ("route_x", "route_y"),
        ("gt_x", "gt_y"),
        ("reference_s", "reference_e"),
        ("ref_s", "ref_e"),
        ("route_s", "route_e"),
        ("gt_s", "gt_e"),
    ])

    final_x, final_y = find_pair(df, [
        ("final_x", "final_y"),
        ("posterior_x", "posterior_y"),
        ("pred_x", "pred_y"),
        ("kf_x", "kf_y"),
        ("final_s", "final_e"),
        ("posterior_s", "posterior_e"),
        ("pred_s", "pred_e"),
        ("kf_s", "kf_e"),
    ])

    if ref_x is None or final_x is None:
        raise RuntimeError(f"找不到 reference/final 欄位: {csv_path}")

    img = Image.open(bg_path).convert("RGB")
    w, h = img.size

    # 如果 CSV 本來就有 pixel 欄位就直接用，沒有就把 metric 座標縮放到地圖上
    ref_px, ref_py = find_pair(df, [
        ("reference_px", "reference_py"),
        ("ref_px", "ref_py"),
        ("route_px", "route_py"),
        ("gt_px", "gt_py"),
    ])
    final_px, final_py = find_pair(df, [
        ("final_px", "final_py"),
        ("posterior_px", "posterior_py"),
        ("pred_px", "pred_py"),
        ("kf_px", "kf_py"),
    ])

    if ref_px is not None and final_px is not None:
        rx, ry = df[ref_px].tolist(), df[ref_py].tolist()
        fx, fy = df[final_px].tolist(), df[final_py].tolist()
    else:
        all_x = df[ref_x].tolist() + df[final_x].tolist()
        all_y = df[ref_y].tolist() + df[final_y].tolist()
        rx, ry = metric_to_pixel(df[ref_x].tolist(), df[ref_y].tolist(), w, h, all_x, all_y)
        fx, fy = metric_to_pixel(df[final_x].tolist(), df[final_y].tolist(), w, h, all_x, all_y)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.plot(rx, ry, linewidth=2.5, label="Reference Route")
    plt.plot(fx, fy, linewidth=2.0, label="Final Localization")

    if len(rx) > 0:
        plt.scatter(rx[0], ry[0], s=60, marker='o', label="Start")
        plt.scatter(rx[-1], ry[-1], s=80, marker='x', label="End")

    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    save_path = out_dir / f"{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"[OK] saved: {save_path}")
    print(f"  csv = {csv_path}")
    print(f"  bg  = {bg_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/yh/study/uav-sat/v36_byTeacher/output/mobilenet_v3_small")
    parser.add_argument("--repo-root", type=str, default="/yh/study/uav-sat")
    parser.add_argument("--frame-count", type=int, default=2)
    parser.add_argument("--search", type=str, default="3x6")
    parser.add_argument("--bg-b", type=str, default="")
    parser.add_argument("--bg-c", type=str, default="")
    args = parser.parse_args()

    root = Path(args.root)
    repo_root = Path(args.repo_root)
    out_dir = root / "plots_final_map"

    csv_b = choose_csv(root, "route_B", args.frame_count, args.search)
    csv_c = choose_csv(root, "route_C", args.frame_count, args.search)

    if csv_b is None:
        raise RuntimeError("找不到 route_B 的 full CSV")
    if csv_c is None:
        raise RuntimeError("找不到 route_C 的 full CSV")

    bg_b = Path(args.bg_b) if args.bg_b else auto_find_map(repo_root, "route_B")
    bg_c = Path(args.bg_c) if args.bg_c else auto_find_map(repo_root, "route_C")

    if bg_b is None or not bg_b.exists():
        raise RuntimeError("找不到 Route B 的背景地圖，請手動用 --bg-b 指定")
    if bg_c is None or not bg_c.exists():
        raise RuntimeError("找不到 Route C 的背景地圖，請手動用 --bg-c 指定")

    draw_one(csv_b, bg_b, out_dir, f"Route B Final 2frame {args.search}")
    draw_one(csv_c, bg_c, out_dir, f"Route C Final 2frame {args.search}")

    print(f"\nplots saved in: {out_dir}")


if __name__ == "__main__":
    main()
