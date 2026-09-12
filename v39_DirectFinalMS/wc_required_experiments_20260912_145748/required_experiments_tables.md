# V39 Required Remaining Experiments — Corrected Results

Selected main chain: **Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> one final 5x5 MeanShift**.

Evaluation protocol for these ablations: **controlled_gt_jitter**, matching the protocol used to train the available v39 temporal checkpoint.

## 1. GRU necessity in the completed chain

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| WC -> Kalman -> MS (no GRU) | 2.035 | 1.665 | 1.904 | 3.842 | 96.43% |
| WC -> GRU -> Kalman -> MS | 2.190 | 1.789 | 2.047 | 4.112 | 95.53% |

## 2. Selected 5x5 main-method online runtime

- Route B: 110.019 ms / 9.1 FPS
- Route C: 110.362 ms / 9.1 FPS
- Pooled-frame weighted B+C: **110.141 ms / 9.1 FPS**

## 3. Correct combined-distribution statistics

B+C percentiles are computed by concatenating all Route B and Route C per-frame `error_final_m` values first, then taking the percentile. Route-level P90 values are never averaged.

- Full B+C MedLE: 1.730 m
- Full B+C P90: 4.112 m
- Full B+C P95: 4.885 m
- Full B+C P99: 5.950 m
- Full B+C LSR@5/10/15/20: 95.53% / 100.00% / 100.00% / 100.00%

## Protocol note

`route_reference` closed-loop is intentionally **not** included in this ablation suite. The available checkpoint was trained with one controlled local window, while `route_reference` changes inference to a 13-window ±60 m acquisition bank. A valid autonomous comparison requires separate training under that same acquisition protocol; directly evaluating the controlled checkpoint under it is an out-of-distribution protocol change, not a module ablation.

## Audit

- checkpoint architecture compatibility: PASS
- no-GRU ablation: PASS
- 5x5 full-chain E2E timing: PASS
- exactly one final MeanShift in the full chain: PASS
- pooled B+C percentile calculation: PASS
- mismatched route_reference evaluation excluded: PASS
