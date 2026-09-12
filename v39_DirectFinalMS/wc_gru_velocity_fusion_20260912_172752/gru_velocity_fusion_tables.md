# V39 GRU Velocity-Fusion Results

Main: **Weighted Centroid -> context-aware 3-frame GRU residual/velocity -> fixed-R Kalman -> one final 5x5 MeanShift**.

The no-GRU ablation is not intentionally weakened; without a learned velocity source, the same external Kalman falls back to its internal constant-velocity state.

## GRU necessity

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| without GRU | 2.384 | 2.195 | 2.317 | 4.231 | 95.61% |
| full context-GRU | 2.191 | 1.779 | 2.045 | 4.110 | 95.73% |

- Full-vs-no-GRU B+C MLE improvement: **11.75%**

## Full 5x5 pooled distribution

- MedLE: 1.717 m
- P90/P95/P99: 4.110 / 4.823 / 5.911 m
- LSR@5/10/15/20: 95.73% / 100.00% / 100.00% / 100.00%

## Same-GPU paired E2E runtime

- 5x5: **36.667 ms / 27.3 FPS**
- 6x6: **39.554 ms / 25.3 FPS**
- Runtime sanity: **PASS**

## Audit

- fresh Route-A-only context-GRU checkpoint: PASS
- patience = 5: PASS
- B/C evaluation only: PASS
- GRU receives posterior-weighted SAT context: PASS
- full model uses learned GRU velocity: PASS
- no-GRU uses Kalman CV fallback, not an artificial zero-motion penalty: PASS
- fixed-R Kalman: PASS
- exactly one final MeanShift: PASS
- GPU-resident grid indexing: PASS
- full model better than no-GRU: **PASS**
