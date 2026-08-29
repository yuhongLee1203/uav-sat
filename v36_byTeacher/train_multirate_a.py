"""Train/evaluate v36_byTeacher using only native-speed Route A.

Training:
  * Route-A at the recorded/original temporal rate only.
Validation:
  * Route-C for checkpoint selection / early stopping.
Test:
  * Route-B only after checkpoint selection.
"""

import argparse

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument(
        "--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument(
        "--patience", type=int, default=int(config.EARLY_STOP_PATIENCE)
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M)
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--train-visual-if-missing",
        action="store_true",
        help="train the Route-A-only visual retrieval model if missing",
    )
    parser.add_argument("--measure-latency", action="store_true")
    return parser.parse_args()


def ensure_visual(device, args):
    if not config.VISUAL_CHECKPOINT.exists():
        if not args.train_visual_if_missing:
            raise FileNotFoundError(
                f"visual checkpoint not found: {config.VISUAL_CHECKPOINT}; "
                "pass --train-visual-if-missing to train it"
            )
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(config.VISUAL_EPOCHS),
            jitter_m=float(args.jitter_m),
            resume=False,
        )
    return FrozenVisualLocalizer(device)


def train_native_a(visual, device, args):
    route_a_cache = rt.build_route_cache(
        "route_A", config.ROUTE_ROOTS[0], visual, device
    )
    c_index = config.ROUTE_NAMES.index("route_C")
    route_c_cache = rt.build_route_cache(
        "route_C", config.ROUTE_ROOTS[c_index], visual, device
    )
    return rt.train_temporal_route_a(
        visual=visual,
        route_a_cache=route_a_cache,
        route_c_cache=route_c_cache,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume),
    )


def evaluate_c_and_b(visual, device, args):
    model = rt.load_temporal_model(device)
    for route_name in ("route_C", "route_B"):
        route_index = config.ROUTE_NAMES.index(route_name)
        cache = rt.build_route_cache(
            route_name, config.ROUTE_ROOTS[route_index], visual, device
        )
        summary = rt.run_route_inference(
            route_name,
            visual,
            model,
            cache,
            device,
            save_csv=True,
            measure_latency=bool(args.measure_latency),
        )
        role = "validation" if route_name == "route_C" else "test"
        print(
            f"{route_name} ({role}): "
            f"prior={summary['Prior_MLE_m']:.3f}m "
            f"MS1={summary['MS1_MLE_m']:.3f}m "
            f"KF={summary['Kalman_MLE_m']:.3f}m "
            f"MS2/final={summary['MLE_m']:.3f}m "
            f"P90={summary['P90_m']:.3f}m "
            f"LSR@15={summary['LSR@15_pct']:.2f}%",
            flush=True,
        )


def main():
    args = parse_args()
    rt.set_seed(config.SEED)
    device = rt.resolve_device()
    visual = ensure_visual(device, args)

    print("=" * 96, flush=True)
    print("Temporal training: Route-A ORIGINAL SPEED ONLY", flush=True)
    print("Route-C validation; Route-B final untouched test", flush=True)
    print("=" * 96, flush=True)

    if args.mode in ("train", "all"):
        _, best = train_native_a(visual, device, args)
        print(f"best Route-C validation MLE={best:.3f}m", flush=True)

    if args.mode in ("eval", "all"):
        evaluate_c_and_b(visual, device, args)


if __name__ == "__main__":
    main()
