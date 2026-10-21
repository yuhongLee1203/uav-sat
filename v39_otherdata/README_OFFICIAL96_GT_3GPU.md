# Reassembled c4m Official96 GT ablation on GPUs 0, 5, 6

Run from `/yh/study/uav-sat`:

```bash
bash v39_otherdata/run_official96_gt_ablation_3gpu.sh
```

The command copies the previously prepared `official_rsi96_seed42_temporal_v1`
City A/B/C/D episodes into a separate output suite, checks the 76,500/4,500/9,000
split and GT conversion, then trains one shared ABCD model per frame count.
Three-frame training uses GPU 0, two-frame GPU 5, and one-frame GPU 6. Each
model visits the four city train episodes sequentially and resets route state
at episode boundaries. The three groups train simultaneously with distinct
feature caches. Evaluation distributes every city and variant over the same
three GPUs. Reusing the same command resumes completed work.

The ten measured rows are `full`, `no_gru`, `no_kalman`, `no_ms`, `frames1`,
`frames2`, `grid4`, `grid5`, `grid7`, and `grid8`. All use v39's
`controlled_gt_jitter` protocol: current-frame GT determines a bounded
reference point for patch search. This is a GT-assisted result.

The pooled outputs are under
`v39_otherdata/output/official96_gt_ablation_3gpu_20260925/`:

- `paper_ablation_tables.md`: component, frame-count, and grid-window tables.
- `official96_gt_all_metrics.md`: pooled localization and navigation-style
  metrics for all ten rows; matching JSON and CSV are also written.
- `logs/`: separate training logs per GPU and evaluation logs per job.

`R@1*` is the existing same-quadrant score derived from final XY. It is not a
four-tile retrieval head's official Recall@1. `SR@20`, `SPL`, and `NE` in the
combined table are offline synthetic-route proxies computed from predicted
frame positions and the predeclared waypoint path. The underlying 90k metadata
has independent geotagged images, no recorded timestamped flight or closed-loop
navigation log, so these values are not official Bearing-Naver navigation scores.
