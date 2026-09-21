# Bearing-UAV Formal V5 Runbook

## Frozen checkpoint

- Branch: `bearing-v5-citya-pass-20260920`
- Purpose: immutable reference point for the CityA V5 ablation run where `FULL_TREND_CHECK=PASS`, `component_full_best=true`, and `three_frame_full_best=true`.
- Do not continue development on this branch.

## Formal all-city branch

- Branch: `bearing-v5-formal-allcities`
- Purpose: official Full-only City A/B/C/D training and held-out evaluation.
- No ablation variants are run here.

## Formal pipeline

For each city independently:

1. Fresh prepare the city data.
2. Train the Full 3-frame V5 model.
3. Select temporal/Kalman profile using only the current city's training validation split.
4. Evaluate held-out `nav50` and `nav51`.
5. Save numeric summary and per-frame CSV files.
6. Render `nav50` and `nav51` trajectory figures from raw `final_x/final_y` predictions.

The visualization uses the official waypoint trajectory as the green GT polyline and raw model predictions as the red polyline. No prediction smoothing is applied by the plotting script.

Internal route aliases used by the code:

- `test_01` = `nav50`
- `test_02` = `nav51`

`bearing_plot_final_vs_gt.py` accepts either naming scheme and resolves the corresponding summary entry automatically.

## GPU scheduling

- GPU 0: first city slot
- GPU 5: second city slot
- GPU 6: third city slot
- City D is launched immediately on whichever GPU finishes first.

## Resource-safe launcher

Use `v39_otherdata/run_bearing_formal_v5_safe.sh` for fresh formal runs.

Default host limits:

- CPU math threads per city process: 2
- SAT backbone cache batch: 128
- CPU nice level: 5
- SAT backbone gallery cache concurrency: exactly 1 city at a time
- GPU 0/5/6 training remains pipelined concurrently after each cache is ready

The serialized cache stage prevents three separate 47,961-patch SAT cache builders from saturating the CPU at the same time. These controls change resource scheduling/batching only; they do not change the V5 architecture, labels, city validation calibration, or held-out evaluation definition.

Optional overrides:

```bash
CPU_THREADS_PER_CITY=2 CACHE_BATCH_SIZE=128 CPU_NICE=5 \
  bash v39_otherdata/run_bearing_formal_v5_safe.sh
```

## Resume an interrupted formal run

Use `v39_otherdata/resume_bearing_formal_v5_safe.sh` with the original `FORMAL_SUITE_ROOT`.

The resume runner inspects each city independently:

- completed summary + completed figures: skip everything;
- completed summary + missing figures: plot only;
- completed checkpoint + missing summary: evaluate only, then plot;
- incomplete checkpoint: resume training without `--force-train`, then evaluate and plot.

This prevents a plotting failure from forcing already-measured cities to be retrained.

Important: when resuming an interrupted run, do not overwrite the locally generated `bearing_iclr_ablation.py` with the branch-base copy before the resume command. The interrupted formal launcher already generated the V5-aligned runner locally.

## Formal output tree

```text
formal_bearing_v5_allcities_<timestamp>/
├── formal_allcities_results.json
├── logs/
├── citya/
│   ├── prepared/
│   ├── train_frames3/
│   │   ├── checkpoints/
│   │   ├── experiment_manifest.json
│   │   └── kalman_calibration.json
│   └── variants/full/
│       ├── bearing_v39_summary.json
│       ├── experiment_manifest.json
│       ├── *_frames.csv
│       └── formal_figures/
│           ├── nav50_result.jpg
│           ├── nav51_result.jpg
│           └── plot_source_audit.json
├── cityb/
├── cityc/
└── cityd/
```

There are 8 final trajectory images total: 2 per city.

## Primary formal files

- Four-city aggregate metrics: `formal_allcities_results.json`
- Per-city metrics: `<city>/variants/full/bearing_v39_summary.json`
- Per-frame predictions: `<city>/variants/full/*_frames.csv`
- nav50 figure: `<city>/variants/full/formal_figures/nav50_result.jpg`
- nav51 figure: `<city>/variants/full/formal_figures/nav51_result.jpg`
- Figure provenance audit: `<city>/variants/full/formal_figures/plot_source_audit.json`

## Protocol note

The current formal V5 run uses the controlled local-prior protocol. These measurements must not be described as GT-free autonomous deployment results.
