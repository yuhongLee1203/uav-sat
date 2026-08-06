

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import config
from robust_tracker import (
    RouteCache,
    build_route_cache,
    load_checkpoint,
    predict_split,
    set_seed,
    train_model,
)
from visual_localizer import FrozenVisualLocalizer
from visual_model import TemporalLatticeCRF


ARCHITECTURE_NAME = "ResidualSecondOrderTemporalLatticeCRF"


def route_catalog() -> Dict[str, Tuple[int, Path]]:
    """Return route name -> (global route index, dataset root).

    The global index is important because deterministic_jitter() seeds each route
    with config.SEED + 1009 * route_index.  We must preserve A=0, B=1, C=2 even
    when only A+C are selected.
    """
    if len(config.ROUTE_NAMES) != len(config.ROUTE_ROOTS):
        raise RuntimeError(
            "config.ROUTE_NAMES and config.ROUTE_ROOTS have different lengths"
        )
    return {
        name: (index, Path(root))
        for index, (name, root) in enumerate(
            zip(config.ROUTE_NAMES, config.ROUTE_ROOTS)
        )
    }


def normalize_route_names(
    names: Optional[Iterable[str]],
    default: Sequence[str],
    argument_name: str,
) -> List[str]:
    selected = list(default if not names else names)
    if not selected:
        raise ValueError(f"{argument_name} must contain at least one route")

    catalog = route_catalog()
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError(
            f"unknown routes in {argument_name}: {unknown}; "
            f"valid routes are {list(config.ROUTE_NAMES)}"
        )

    # Keep user order while removing duplicates.
    deduplicated: List[str] = []
    seen = set()
    for name in selected:
        if name not in seen:
            deduplicated.append(name)
            seen.add(name)
    return deduplicated


def safe_experiment_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("experiment name becomes empty after sanitization")
    return value


def default_experiment_name(
    train_routes: Sequence[str],
    eval_routes: Sequence[str],
) -> str:
    train_code = "-".join(name.replace("route_", "") for name in train_routes)
    eval_code = "-".join(name.replace("route_", "") for name in eval_routes)
    return f"train_{train_code}__test_{eval_code}"


def configure_experiment_paths(
    output_root: Path,
    experiment_name: str,
) -> Tuple[Path, Path]:
    """Redirect config paths only for this Python process."""
    output_dir = Path(output_root) / safe_experiment_name(experiment_name)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "rtl_crf_cross_route.pt"

    config.OUTPUT_DIR = output_dir
    config.CHECKPOINT_DIR = checkpoint_dir
    config.TEMPORAL_CHECKPOINT = checkpoint_path
    return output_dir, checkpoint_path


def selected_route_records(names: Sequence[str]) -> List[Tuple[int, Path, str]]:
    catalog = route_catalog()
    return [
        (catalog[name][0], catalog[name][1], name)
        for name in names
    ]


def build_caches(
    records: Sequence[Tuple[int, Path, str]],
    visual: FrozenVisualLocalizer,
    device: torch.device,
    jitter_m: float,
) -> Dict[str, RouteCache]:
    caches: Dict[str, RouteCache] = {}
    for global_index, root, name in records:
        if name in caches:
            continue
        print(
            f"\n[cache] {name}: global_route_index={global_index}, root={root}",
            flush=True,
        )
        caches[name] = build_route_cache(
            root=root,
            name=name,
            route_index=global_index,
            visual=visual,
            device=device,
            jitter_m=float(jitter_m),
        )
    return caches


def ordered_caches(
    cache_map: Dict[str, RouteCache],
    names: Sequence[str],
) -> List[RouteCache]:
    return [cache_map[name] for name in names]


def check_capture_rates(
    train_caches: Sequence[RouteCache],
    eval_caches: Sequence[RouteCache],
) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for cache in list(train_caches) + list(eval_caches):
        rates[cache.name] = float(cache.capture.float().mean().item())

    minimum_train = min(rates[cache.name] for cache in train_caches)
    if minimum_train < float(config.MIN_TRAIN_CAPTURE_RATE):
        raise RuntimeError(
            "At least one training route has candidate capture below "
            f"{100.0 * float(config.MIN_TRAIN_CAPTURE_RATE):.1f}%. "
            "Training windows would be heavily filtered. Run with --jitter-m 0 "
            "first to verify gallery/map geometry."
        )

    for cache in train_caches:
        rate = rates[cache.name]
        if rate < 0.98:
            print(
                f"warning: training route {cache.name} capture={rate * 100:.2f}% "
                "is below 98%; training uses only fully captured windows",
                flush=True,
            )

    for cache in eval_caches:
        rate = rates[cache.name]
        if rate < 0.98:
            print(
                f"warning: unseen evaluation route {cache.name} "
                f"capture={rate * 100:.2f}% is below 98%; evaluation still "
                "includes every frame",
                flush=True,
            )
    return rates


def checkpoint_metadata(
    train_routes: Sequence[str],
    eval_routes: Sequence[str],
    eval_split: str,
    jitter_m: float,
    experiment_name: str,
) -> Dict[str, object]:
    return {
        "cross_route_experiment": True,
        "experiment_name": experiment_name,
        "train_routes": list(train_routes),
        "validation_routes": list(train_routes),
        "eval_routes": list(eval_routes),
        "eval_split": eval_split,
        "jitter_m": float(jitter_m),
        "architecture": ARCHITECTURE_NAME,
    }


def validate_resume_checkpoint(
    checkpoint_path: Path,
    expected: Dict[str, object],
) -> None:
    if not checkpoint_path.exists():
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            f"checkpoint architecture mismatch: {checkpoint.get('architecture')}"
        )

    saved_train = checkpoint.get("train_routes")
    if saved_train is not None and list(saved_train) != list(expected["train_routes"]):
        raise RuntimeError(
            f"resume checkpoint was trained on {saved_train}, but current "
            f"--train-routes are {expected['train_routes']}"
        )

    saved_jitter = checkpoint.get("jitter_m")
    if saved_jitter is not None and abs(
        float(saved_jitter) - float(expected["jitter_m"])
    ) > 1e-9:
        raise RuntimeError(
            f"resume checkpoint jitter={saved_jitter}, but current jitter="
            f"{expected['jitter_m']}"
        )


def attach_checkpoint_metadata(
    checkpoint_path: Path,
    metadata: Dict[str, object],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found after training: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.update(metadata)
    torch.save(checkpoint, checkpoint_path)


def save_rows_for_eval_routes(
    model: TemporalLatticeCRF,
    eval_caches: Sequence[RouteCache],
    eval_split: str,
    device: torch.device,
):
    return predict_split(
        model=model,
        caches=eval_caches,
        split_name=eval_split,
        device=device,
        save_rows=True,
    )


def load_reference_summary(
    reference_path: Path,
) -> Optional[Dict[str, object]]:
    if not reference_path.exists():
        return None
    try:
        with reference_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(
            f"warning: cannot read reference summary {reference_path}: {error}",
            flush=True,
        )
        return None


def build_comparison(
    cross_summary: Dict[str, object],
    reference_summary: Optional[Dict[str, object]],
) -> Dict[str, object]:
    comparison: Dict[str, object] = {
        "note": (
            "Cross-route metrics use a model trained without the evaluation route. "
            "Reference metrics, when available, come from the existing in-route "
            "A+B+C experiment."
        ),
        "routes": {},
    }

    if reference_summary is None:
        comparison["reference_available"] = False
        return comparison

    comparison["reference_available"] = True
    reference_routes = reference_summary.get("routes", {})
    cross_routes = cross_summary.get("routes", {})

    metric_names = (
        "MLE_m",
        "P90_m",
        "LSR@5_pct",
        "LSR@10_pct",
        "LSR@15_pct",
        "RPE_m",
        "JumpRate_pct",
        "StationaryDriftP90_m",
        "PathLengthRatio",
    )

    route_rows = {}
    for route_name, cross_route_summary in cross_routes.items():
        reference_route_summary = reference_routes.get(route_name)
        if not isinstance(reference_route_summary, dict):
            route_rows[route_name] = {
                "cross_route": cross_route_summary.get("RTL_CRF"),
                "reference": None,
            }
            continue

        cross_metric = cross_route_summary.get("RTL_CRF", {})
        reference_metric = reference_route_summary.get("RTL_CRF", {})
        deltas = {}
        for metric_name in metric_names:
            if metric_name in cross_metric and metric_name in reference_metric:
                deltas[metric_name] = (
                    float(cross_metric[metric_name])
                    - float(reference_metric[metric_name])
                )

        route_rows[route_name] = {
            "cross_route": cross_metric,
            "reference_in_route_ABC": reference_metric,
            "cross_minus_reference": deltas,
        }

    comparison["routes"] = route_rows
    return comparison


def print_key_metrics(summary: Dict[str, object]) -> None:
    print("\n================ CROSS-ROUTE RESULTS ================", flush=True)
    routes = summary.get("routes", {})
    for route_name, route_summary in routes.items():
        hardms = route_summary["FixedHardMS"]
        rtl = route_summary["RTL_CRF"]
        print(f"\n{route_name} ({route_summary['split']} split)", flush=True)
        print(
            "  FixedHardMS: "
            f"MLE={hardms['MLE_m']:.3f} m, "
            f"P90={hardms['P90_m']:.3f} m, "
            f"LSR@10={hardms['LSR@10_pct']:.2f}%, "
            f"Jump={hardms['JumpRate_pct']:.2f}%",
            flush=True,
        )
        print(
            "  RTL-CRF:     "
            f"MLE={rtl['MLE_m']:.3f} m, "
            f"P90={rtl['P90_m']:.3f} m, "
            f"LSR@10={rtl['LSR@10_pct']:.2f}%, "
            f"RPE={rtl['RPE_m']:.3f} m, "
            f"Jump={rtl['JumpRate_pct']:.2f}%",
            flush=True,
        )
        improvement = (
            (hardms["MLE_m"] - rtl["MLE_m"])
            / max(hardms["MLE_m"], 1e-9)
            * 100.0
        )
        print(
            f"  MLE improvement over HardMS: {improvement:.2f}%",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RTL-CRF on selected routes and evaluate on disjoint unseen routes."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_eval"),
        default="train_eval",
    )
    parser.add_argument(
        "--train-routes",
        nargs="+",
        default=["route_A", "route_C"],
        help="routes used for training and validation",
    )
    parser.add_argument(
        "--eval-routes",
        nargs="+",
        default=["route_B"],
        help="completely unseen routes used only for evaluation",
    )
    parser.add_argument(
        "--eval-split",
        choices=("test", "all"),
        default="test",
        help=(
            "'test' compares directly with the existing last-15%% result; "
            "'all' evaluates the entire unseen route"
        ),
    )
    parser.add_argument("--epochs", type=int, default=int(config.EPOCHS))
    parser.add_argument(
        "--jitter-m",
        type=float,
        default=float(config.LOCAL_PRIOR_JITTER_M),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="output folder name; generated automatically when omitted",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path(config.PROJECT_ROOT)
            / "outputs"
            / "temporal_prior_hardms_cross_route"
        ),
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=(
            Path(config.PROJECT_ROOT)
            / "outputs"
            / "temporal_prior_hardms"
            / "robust_tracker_summary.json"
        ),
        help="existing A+B+C summary used only for automatic comparison",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="render inference videos for evaluation routes after CSV output",
    )
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--video-width", type=int, default=1600)
    parser.add_argument("--video-height", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_routes = normalize_route_names(
        args.train_routes,
        default=("route_A", "route_C"),
        argument_name="--train-routes",
    )
    eval_routes = normalize_route_names(
        args.eval_routes,
        default=("route_B",),
        argument_name="--eval-routes",
    )

    overlap = sorted(set(train_routes) & set(eval_routes))
    if overlap:
        raise ValueError(
            "Cross-route evaluation requires disjoint route sets. "
            f"These routes appear in both training and evaluation: {overlap}"
        )

    experiment_name = (
        safe_experiment_name(args.experiment_name)
        if args.experiment_name
        else default_experiment_name(train_routes, eval_routes)
    )
    output_dir, checkpoint_path = configure_experiment_paths(
        output_root=args.output_root,
        experiment_name=experiment_name,
    )
    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)

    metadata = checkpoint_metadata(
        train_routes=train_routes,
        eval_routes=eval_routes,
        eval_split=args.eval_split,
        jitter_m=float(args.jitter_m),
        experiment_name=experiment_name,
    )

    print("=====================================================", flush=True)
    print("RTL-CRF CROSS-ROUTE EXPERIMENT", flush=True)
    print(f"experiment:       {experiment_name}", flush=True)
    print(f"train routes:     {train_routes}", flush=True)
    print(f"validation routes:{train_routes}", flush=True)
    print(f"evaluation routes:{eval_routes}", flush=True)
    print(f"evaluation split: {args.eval_split}", flush=True)
    print(f"jitter:           {args.jitter_m:.3f} m", flush=True)
    print(f"output directory: {output_dir}", flush=True)
    print(f"checkpoint:       {checkpoint_path}", flush=True)
    print("=====================================================", flush=True)

    if args.resume:
        validate_resume_checkpoint(checkpoint_path, metadata)

    set_seed(int(config.SEED))
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)

    visual = FrozenVisualLocalizer(device)

    # Build the union once, preserving each route's original global index.
    all_names = list(train_routes)
    for name in eval_routes:
        if name not in all_names:
            all_names.append(name)
    records = selected_route_records(all_names)
    cache_map = build_caches(
        records=records,
        visual=visual,
        device=device,
        jitter_m=float(args.jitter_m),
    )

    train_caches = ordered_caches(cache_map, train_routes)
    eval_caches = ordered_caches(cache_map, eval_routes)
    capture_rates = check_capture_rates(train_caches, eval_caches)

    model = TemporalLatticeCRF().to(device)

    if args.mode in ("train", "train_eval"):
        train_model(
            model=model,
            caches=train_caches,
            device=device,
            epochs=int(args.epochs),
            resume=bool(args.resume),
        )
        attach_checkpoint_metadata(checkpoint_path, metadata)

    if args.mode in ("eval", "train_eval"):
        # train_model() leaves the best model loaded.  Pure eval mode must load it.
        if args.mode == "eval":
            checkpoint = load_checkpoint(model, device)
            saved_train = checkpoint.get("train_routes")
            if saved_train is not None and list(saved_train) != list(train_routes):
                raise RuntimeError(
                    f"checkpoint train_routes={saved_train}, but command requests "
                    f"{train_routes}"
                )

        outputs = save_rows_for_eval_routes(
            model=model,
            eval_caches=eval_caches,
            eval_split=args.eval_split,
            device=device,
        )

        summary: Dict[str, object] = {
            "method": ARCHITECTURE_NAME,
            "protocol": (
                "GT+jitter local prior; train/validation routes are disjoint "
                "from evaluation routes; no model-output candidate propagation"
            ),
            "experiment_name": experiment_name,
            "train_routes": list(train_routes),
            "validation_routes": list(train_routes),
            "eval_routes": list(eval_routes),
            "eval_split": args.eval_split,
            "jitter_m": float(args.jitter_m),
            "candidate_capture_rate_pct": {
                name: rate * 100.0 for name, rate in capture_rates.items()
            },
            "checkpoint": str(checkpoint_path),
            "routes": {
                name: route_summary
                for name, route_summary, _ in outputs
            },
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "cross_route_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        reference_summary = load_reference_summary(args.reference_summary)
        comparison = build_comparison(summary, reference_summary)
        comparison_path = output_dir / "comparison_with_in_route_ABC.json"
        with comparison_path.open("w", encoding="utf-8") as file:
            json.dump(comparison, file, ensure_ascii=False, indent=2)

        print_key_metrics(summary)
        print(f"\nsummary:    {summary_path}", flush=True)
        print(f"comparison: {comparison_path}", flush=True)
        print(
            "per-frame CSV files are in the same experiment directory",
            flush=True,
        )

        if args.render_video:
            from render_results_video import render_routes

            eval_pairs = [
                (route_catalog()[name][1], name)
                for name in eval_routes
            ]
            video_paths = render_routes(
                eval_pairs,
                fps=float(args.video_fps),
                video_width=int(args.video_width),
                video_height=int(args.video_height),
            )
            for video_path in video_paths:
                print(f"inference video: {video_path}", flush=True)


if __name__ == "__main__":
    main()
