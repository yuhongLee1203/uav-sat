| Final decoder | LSR@3 ↑ | LSR@15 ↑ | MLE (m) ↓ | GPU-only aggregation (ms) ↓ | Single-call aggregation (ms) ↓ |
| --- | --- | --- | --- | --- | --- |
| Weighted Average (36 patches) | 42.51% | 100.00% | 3.749 | 0.003920 | 0.041984 |
| Mean Shift (1 mode) | 41.90% | 100.00% | 3.759 | 0.002640 | 0.042480 |

Both rows start from the same 36 final 6×6 candidate centers and regularized logits after identical front Mean Shift, GRU, and Kalman outputs. Weighted Average reduces all 36 softmax(logit/τ)-weighted centers. Mean Shift first iterates and merges its 36 seeds into active modes; all 1,482 frames have exactly one active mode, which its final aggregation reduces.
Probe-vs-original Full maximum XY difference: 0.000000000 m across 1482 frames. The logits include Kalman and current-frame GT reference priors.
Latency uses real probe inputs on an RTX 3090. GPU-only is CUDA-graph device time per final reduction; single-call is a synchronized CUDA-event measurement. Feature extraction, scoring, Mean Shift iterations and mode merging are excluded. The complete final Mean Shift stage takes about 7.24 ms/frame in the original replay.
Mean Shift minus Weighted Average paired MLE: +0.011 m (route bootstrap 95% interval [+0.006, +0.015] m; Mean Shift better on 0/8 routes). The observed result favors Weighted Average slightly; this experiment does not show a unique Mean Shift advantage.
