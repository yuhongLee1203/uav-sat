#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def jump_information(df, tolerance_m):
    gt = df[["gt_x", "gt_y"]].to_numpy(dtype=float)
    pred = df[["final_x", "final_y"]].to_numpy(dtype=float)

    if len(df) < 2:
        return 0.0, np.zeros(0, dtype=int), np.zeros(0)

    gt_step = np.linalg.norm(
        np.diff(gt, axis=0),
        axis=1,
    )
    pred_step = np.linalg.norm(
        np.diff(pred, axis=0),
        axis=1,
    )

    threshold = (
        float(np.percentile(gt_step, 99))
        + float(tolerance_m)
    )

    jump_rows = (
        np.where(pred_step > threshold)[0]
        + 1
    )

    return threshold, jump_rows, pred_step


def render_route(
    csv_path,
    output_dir,
    tolerance_m,
    max_video_frames,
    fps,
):
    df = pd.read_csv(csv_path)

    required = {
        "frame_id",
        "gt_x",
        "gt_y",
        "final_x",
        "final_y",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{csv_path}: missing columns {sorted(missing)}"
        )

    route = csv_path.name.split(
        "_route_rnn_filterpy_frames.csv"
    )[0]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt = df[
        ["gt_x", "gt_y"]
    ].to_numpy(dtype=float)

    final = df[
        ["final_x", "final_y"]
    ].to_numpy(dtype=float)

    error = np.linalg.norm(
        final - gt,
        axis=1,
    )

    threshold, jump_rows, pred_step = (
        jump_information(
            df,
            tolerance_m,
        )
    )

    # --------------------------------------------------------------
    # Static trajectory: GT vs FINAL ONLY.
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.plot(
        gt[:, 0],
        gt[:, 1],
        linewidth=2.0,
        label="GT trajectory",
    )

    ax.plot(
        final[:, 0],
        final[:, 1],
        linewidth=1.6,
        label="Final Kalman trajectory",
    )

    if len(jump_rows):
        ax.scatter(
            final[jump_rows, 0],
            final[jump_rows, 1],
            s=18,
            label=(
                f"Detected jumps "
                f"({len(jump_rows)})"
            ),
        )

    ax.scatter(
        [gt[0, 0]],
        [gt[0, 1]],
        s=80,
        marker="o",
        label="Start",
    )

    ax.set_title(
        f"{route}: GT vs Final Kalman\n"
        f"jump threshold={threshold:.2f} m, "
        f"detected jumps={len(jump_rows)}"
    )
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    trajectory_png = (
        output_dir
        / f"{route}_trajectory_final_vs_gt.png"
    )
    fig.savefig(
        trajectory_png,
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------------
    # Error over time.
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        df["frame_id"].to_numpy(),
        error,
        label="Final localization error",
    )

    ax.set_title(
        f"{route}: Final Kalman error over time"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    error_png = (
        output_dir
        / f"{route}_error_over_time.png"
    )
    fig.savefig(
        error_png,
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------------
    # Per-frame displacement: directly shows trajectory smoothness.
    # --------------------------------------------------------------
    if len(df) > 1:
        gt_step = np.linalg.norm(
            np.diff(gt, axis=0),
            axis=1,
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        frames = df[
            "frame_id"
        ].to_numpy()[1:]

        ax.plot(
            frames,
            gt_step,
            label="GT displacement / frame",
        )

        ax.plot(
            frames,
            pred_step,
            label="Final displacement / frame",
        )

        ax.axhline(
            threshold,
            linestyle="--",
            label=(
                f"jump threshold "
                f"{threshold:.2f} m"
            ),
        )

        ax.set_title(
            f"{route}: frame-to-frame displacement"
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Displacement (m)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        step_png = (
            output_dir
            / f"{route}_frame_displacement.png"
        )
        fig.savefig(
            step_png,
            dpi=200,
        )
        plt.close(fig)
    else:
        step_png = None

    # --------------------------------------------------------------
    # Animation: GT + FINAL only.
    # --------------------------------------------------------------
    all_xy = np.vstack(
        [gt, final]
    )

    x_span = max(
        float(
            all_xy[:, 0].max()
            - all_xy[:, 0].min()
        ),
        1.0,
    )

    y_span = max(
        float(
            all_xy[:, 1].max()
            - all_xy[:, 1].min()
        ),
        1.0,
    )

    x_margin = max(
        20.0,
        0.05 * x_span,
    )

    y_margin = max(
        20.0,
        0.05 * y_span,
    )

    stride = max(
        1,
        int(
            np.ceil(
                len(df)
                / max(
                    int(max_video_frames),
                    1,
                )
            )
        ),
    )

    animation_indices = list(
        range(
            0,
            len(df),
            stride,
        )
    )

    if (
        animation_indices
        and animation_indices[-1]
        != len(df) - 1
    ):
        animation_indices.append(
            len(df) - 1
        )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.set_xlim(
        all_xy[:, 0].min() - x_margin,
        all_xy[:, 0].max() + x_margin,
    )

    ax.set_ylim(
        all_xy[:, 1].min() - y_margin,
        all_xy[:, 1].max() + y_margin,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.grid(True, alpha=0.25)

    gt_line, = ax.plot(
        [],
        [],
        linewidth=2.0,
        label="GT",
    )

    final_line, = ax.plot(
        [],
        [],
        linewidth=1.8,
        label="Final Kalman",
    )

    gt_dot, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=7,
        label="Current GT",
    )

    final_dot, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=7,
        label="Current Final",
    )

    title = ax.set_title("")
    ax.legend()

    def update(animation_step):
        index = animation_indices[
            animation_step
        ]

        gt_line.set_data(
            gt[: index + 1, 0],
            gt[: index + 1, 1],
        )

        final_line.set_data(
            final[: index + 1, 0],
            final[: index + 1, 1],
        )

        gt_dot.set_data(
            [gt[index, 0]],
            [gt[index, 1]],
        )

        final_dot.set_data(
            [final[index, 0]],
            [final[index, 1]],
        )

        title.set_text(
            f"{route} | "
            f"frame={int(df['frame_id'].iloc[index])} | "
            f"error={error[index]:.1f} m"
        )

        return (
            gt_line,
            final_line,
            gt_dot,
            final_dot,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(
            animation_indices
        ),
        interval=1000.0
        / max(
            int(fps),
            1,
        ),
        blit=False,
    )

    video_path = (
        output_dir
        / f"{route}_trajectory_final_vs_gt.mp4"
    )

    writer = FFMpegWriter(
        fps=int(fps),
        bitrate=2200,
    )

    animation.save(
        video_path,
        writer=writer,
        dpi=130,
    )

    plt.close(fig)

    summary = {
        "route": route,
        "frames": int(len(df)),
        "mean_error_m": float(
            error.mean()
        ),
        "median_error_m": float(
            np.median(error)
        ),
        "p90_error_m": float(
            np.percentile(error, 90)
        ),
        "jump_threshold_m": float(
            threshold
        ),
        "jump_count": int(
            len(jump_rows)
        ),
        "jump_rate_pct": float(
            len(jump_rows)
            / max(
                len(df) - 1,
                1,
            )
            * 100.0
        ),
        "max_frame_displacement_m": (
            float(pred_step.max())
            if len(pred_step)
            else 0.0
        ),
    }

    summary_path = (
        output_dir
        / f"{route}_visual_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            f"{key}: {value}"
            for key, value
            in summary.items()
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(route)
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        f"trajectory: {trajectory_png}"
    )
    print(
        f"error:      {error_png}"
    )

    if step_png is not None:
        print(
            f"steps:      {step_png}"
        )

    print(
        f"video:      {video_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "outputs/"
            "route_rnn_filterpy_full_retrain"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--route",
        choices=(
            "route_B",
            "route_C",
            "both",
        ),
        default="both",
    )

    parser.add_argument(
        "--jump-tolerance-m",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = (
            args.input_dir
            / "visualizations"
        )
    else:
        output_dir = (
            args.output_dir
        )

    routes = (
        ["route_B", "route_C"]
        if args.route == "both"
        else [args.route]
    )

    for route in routes:
        csv_path = (
            args.input_dir
            / (
                f"{route}_"
                "route_rnn_filterpy_frames.csv"
            )
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                csv_path
            )

        render_route(
            csv_path=csv_path,
            output_dir=output_dir,
            tolerance_m=float(
                args.jump_tolerance_m
            ),
            max_video_frames=int(
                args.max_video_frames
            ),
            fps=int(args.fps),
        )


if __name__ == "__main__":
    main()
