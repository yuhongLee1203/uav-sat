# V39 GRU-Corrected Final Results

Main chain: **Weighted Centroid -> context-aware 3-frame GRU residual refinement -> fixed-R Kalman (internal constant velocity) -> one final 5x5 MeanShift**.

## GRU necessity

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| without GRU | 2.384 | 2.195 | 2.317 | 4.231 | 95.61% |
| full context-GRU | 2.470 | 2.233 | 2.385 | 4.367 | 94.48% |

- Full-vs-no-GRU B+C MLE improvement: **-2.96%**

## Full 5x5 pooled distribution

- MedLE: 2.238 m
- P90/P95/P99: 4.367 / 5.109 / 6.201 m
- LSR@5/10/15/20: 94.48% / 100.00% / 100.00% / 100.00%

## Same-GPU paired E2E runtime

- 5x5: **35.621 ms / 28.1 FPS**
- 6x6: **40.148 ms / 24.9 FPS**
- Runtime sanity (5x5 <= 1.15 x 6x6): **PASS**

## Method audit

- fresh Route-A-only context-GRU checkpoint: PASS
- B/C used for evaluation, not checkpoint reuse: PASS
- GRU receives posterior-weighted satellite context: PASS
- Kalman owns constant-velocity state propagation: PASS
- fixed-R Kalman: PASS
- exactly one online final MeanShift: PASS
- GPU-resident gallery indexing (no full-gallery per-frame GPU->CPU copy): PASS
- full model better than no-GRU: **NOT YET**
