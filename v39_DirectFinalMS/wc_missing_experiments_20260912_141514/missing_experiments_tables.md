# V39 Missing Experiments — Corrected Results

Selected main chain: **Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> one final 5x5 MeanShift**.

B+C P90 is computed from the concatenated Route B+C per-frame errors, not from a weighted average of route-level P90 values.

## 1. GRU necessity in the completed chain

| Setting | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|
| WC -> Kalman -> MS (no GRU) | 1.904 | 3.842 | 96.43% |
| WC -> GRU -> Kalman -> MS | 2.047 | 4.112 | 95.53% |

## 2. Selected 5x5 main-method online runtime

- Route B: 108.943 ms / 9.2 FPS
- Route C: 106.628 ms / 9.4 FPS
- Pooled-frame weighted B+C: **108.119 ms / 9.2 FPS**

## 3. True no-GT route-reference evaluation

Current-frame GT center use is disabled; final MeanShift reference is audited to equal the causal local-search reference on every frame.

| Protocol | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| route_reference (no current-frame GT) | 205.611 | 187.032 | 198.997 | 491.148 | 2.80% |

## Audit

- checkpoint architecture compatibility: PASS
- no-GRU ablation: PASS
- 5x5 full-chain E2E timing: PASS
- no-GT route_reference: PASS
- final-MS GT-leak check for no-GT run: PASS
- pooled B+C percentile calculation: PASS
