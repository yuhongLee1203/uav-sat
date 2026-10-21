# Final coordinate aggregation latency

GPU: NVIDIA GeForce RTX 3090 (cuda:0). All 1482 test frames had exactly one final Mean Shift mode. Eight real mid-route frames (one per route) supply the timed inputs.

| Decoder | Inputs | Single-call median (ms) | Single-call P90 (ms) | GPU-only median (ms) | GPU-only P90 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weighted Average (36 patches) | 36 | 0.041984 | 0.045056 | 0.003920 | 0.004528 |
| Mean Shift final aggregation (1 mode) | 1 | 0.042480 | 0.045056 | 0.002640 | 0.003040 |

Timing starts after the 36 candidate coordinates and regularized logits are available. For Mean Shift it starts after all iterations and basin merging, using the already converged one-mode coordinate and weight. Both rows compute only `(weights[:, None] * coordinates).sum(0)`. The single-call GPU event includes the gap between Python kernel launches; the CUDA graph replay amortizes host launch overhead over 64 reductions and reports device execution time per reduction. Neither number is full decoder or end-to-end inference latency.
