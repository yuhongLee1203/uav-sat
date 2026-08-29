"""Compatibility entry point for the causal v36_byTeacher v8 trainer.

The previous multirate trainer depended on the old reference-centered teacher
pipeline and GT-dependent Kalman caps.  That path is intentionally removed.
This file now delegates to the same causal Route-A training / Route-B validation
pipeline as robust_tracker.py so it cannot accidentally re-enable the legacy
cheating path.
"""

import argparse

import config
import robust_tracker as rt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS))
    parser.add_argument("--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS))
    parser.add_argument("--patience", type=int, default=int(config.EARLY_STOP_PATIENCE))
    parser.add_argument("--reuse-visual", action="store_true")
    parser.add_argument("--resume-visual", action="store_true")
    parser.add_argument("--resume-temporal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rt.set_seed(config.SEED)
    device = rt.resolve_device()
    print(
        "train_multirate_a.py now uses the causal v8 trainer; legacy GT-dependent "
        "multirate code is disabled.",
        flush=True,
    )
    rt.train_pipeline(args, device)


if __name__ == "__main__":
    main()
