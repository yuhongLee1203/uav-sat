# Legacy GT-centered 90K pilot (2026-09-24)

This page describes an exploratory protocol that **does not meet** the newer
requirement that GT appear only in losses. It selected local visual-training
patches around the GT and initialized temporal weights from GT-prior training.
Its numbers must not be reported as strict causal/GT-loss-only results.
The old runner now exits with an explanation. See
`v40/README_STRICT_GTLOSS.md` for the replacement.

The visual task heads are fine-tuned on official Bearing-UAV-90K metadata.
All City A/B `test_01` and `test_02` UAV image paths are excluded before
sampling. Temporal training uses only the prepared City A/B `train_01` route
episodes, with recurrent/Kalman state reset at each episode. Frame GT is
permitted for the two training losses; the held-out route inference asserts
`NO_GT_INFERENCE` and causal `ROUTE_REFERENCE_ONLY`.

This is **not** the official Bearing-UAV Table 6 split. The prepared route
episodes are constructed from independent 90K observations, not recorded
continuous flights. In City A/B, only about 1% of adjacent prepared frames
have consecutive original metadata indices. Do not present the route MLE as
official Table 6 reproduction.

The tested 2-epoch visual run used 2,000 metadata rows per city. Its random
metadata validation SoftMS MLE was 6.06 m (City A) and 6.12 m (City B), but
that is **GT-centered local candidate validation**, not end-to-end route MLE.

| Temporal setting | A test_01 | A test_02 | B test_01 | B test_02 |
|---|---:|---:|---:|---:|
| causal, 2 epochs | 88.421 | 90.088 | 88.685 | 107.746 |
| mixed teacher, 2 epochs | **10.980** | **20.867** | 25.161 | 44.832 |
| teacher, 2 epochs | 21.001 | 43.621 | **12.765** | **21.427** |
| mixed teacher, 6 full-route epochs | 122.116 | 134.710 | 130.131 | 173.497 |
| teacher, 6 full-route epochs | 160.115 | 250.616 | 140.055 | 259.203 |
| hybrid, 6 full-route epochs | 86.787 | 84.556 | 90.164 | 113.116 |

Values are held-out GT-free route MLE in metres. The two-epoch mixed setting
is the strongest *single shared checkpoint* from these experiments; no tested
setting achieved <10 m on all four routes. More epochs caused large drift and
should not be represented as an improvement.

The legacy command is deliberately disabled:

```bash
bash v40/run_gt_90k_ab_pilot.sh
```

The visual backbone is frozen, and reusable metadata features are under
`v40/cache/visual90k`. Outputs, temporal checkpoint, reports and run logs are
under the timestamped `v40/output/gt_90k_ab_*` directory. GPU 0 handles visual
fine-tuning, GPU 5 the temporal stage by default; both use two CPU threads.
Changing the visual sample size, training horizon or route policy changes the
experiment and must be re-evaluated without test-GT tuning.
