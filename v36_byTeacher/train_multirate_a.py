"""Train the autonomous temporal model on Route-A native + stride-N Route-A.

Default stride=2 gives the requested approximately 2x per-frame motion sequence.
Route-C is used for validation/checkpoint selection. Route-B is reserved for
final testing and is not used during training or checkpoint selection.
"""

import argparse

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument(
        "--patience", type=int, default=int(config.EARLY_STOP_PATIENCE)
    )
    parser.add_argument(
        "--extra-stride", type=int, default=int(config.TEMPORAL_EXTRA_A_STRIDE)
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M)
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--train-visual-if-missing",
        action="store_true",
        help="train the Route-A-only visual retrieval model if its checkpoint is absent",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if int(args.extra_stride) < 2:
        raise ValueError("--extra-stride must be >= 2")

    rt.set_seed(config.SEED)
    device = rt.resolve_device()

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

    visual = FrozenVisualLocalizer(device)
    route_a_cache = rt.build_route_cache(
        "route_A", config.ROUTE_ROOTS[0], visual, device
    )
    c_index = config.ROUTE_NAMES.index("route_C")
    route_c_cache = rt.build_route_cache(
        "route_C", config.ROUTE_ROOTS[c_index], visual, device
    )

    _, best = rt.train_temporal_multirate(
        visual=visual,
        route_a_cache=route_a_cache,
        route_c_cache=route_c_cache,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume),
        extra_stride=int(args.extra_stride),
    )
    print(f"best Route-C validation MLE={best:.3f}m", flush=True)


if __name__ == "__main__":
    main()
