# Paper-Ready 6x6 Experimental Results

All localization metrics are pooled over Route B+C at frame level. The final MeanShift grid is fixed to 6x6 for every experiment. The visual readout is fixed inside the localization front-end and is not shown as a standalone ablation component.

## Table 1. Overall performance of the proposed framework

| Method | MLE (m) ↓ | MedLE (m) ↓ | P90 (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | Jump Rate ↓ | E2E (ms) ↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed Method (6x6) | 2.045 | 1.718 | 4.111 | 77.56% | 95.73% | 100.00% | 0.000% | 40.406 | 24.7 |

## Table 2. Ablation study of the main components

| Variant | Forward 3x6 | Temporal GRU | Kalman Fusion | Final MS | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | Jump Rate ↓ | E2E (ms) ↓ | FPS ↑ |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| w/o Forward 3x6 Restriction | ✗ | ✓ | ✓ | ✓ | 2.181 | 76.66% | 94.20% | 100.00% | 0.000% | 62.587 | 16.0 |
| w/o Temporal GRU | ✓ | ✗ | ✓ | ✓ | 2.316 | 66.69% | 95.61% | 100.00% | 0.000% | 35.875 | 27.9 |
| w/o External Kalman Fusion | ✓ | ✓ | ✗ | ✓ | 2.616 | 62.90% | 92.53% | 99.63% | 1.274% | 34.155 | 29.3 |
| w/o Final MeanShift | ✓ | ✓ | ✓ | ✗ | 4.550 | 36.45% | 60.38% | 94.77% | 0.057% | 27.296 | 36.6 |
| Full Model | ✓ | ✓ | ✓ | ✓ | 2.045 | 77.56% | 95.73% | 100.00% | 0.000% | 40.406 | 24.7 |

## Table 3. Effect of temporal context length

| Temporal Input | First Difference | Second Difference | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ |
|---|:---:|:---:|---:|---:|---:|---:|
| 1 frame | ✗ | ✗ | 2.061 | 77.45% | 95.42% | 100.00% |
| 2 frames | ✓ | ✗ | 2.053 | 77.56% | 95.50% | 100.00% |
| 3 frames | ✓ | ✓ | 2.045 | 77.56% | 95.73% | 100.00% |

## Table 4. External Kalman fusion design

| Kalman Configuration | Measurement Variance | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | Jump Rate ↓ |
|---|---|---:|---:|---:|---:|---:|
| No Kalman Fusion | - | 2.616 | 62.90% | 92.53% | 99.63% | 1.274% |
| Learned-R Kalman | Learned | 2.106 | 75.92% | 94.93% | 100.00% | 0.000% |
| Fixed-R Kalman | Fixed | 2.045 | 77.56% | 95.73% | 100.00% | 0.000% |

## Table 5. Effect of the forward candidate restriction

| Candidate Search | Scored Candidates | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | E2E (ms) ↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full 6x6 search | 36 | 2.181 | 76.66% | 94.20% | 100.00% | 62.587 | 16.0 |
| Forward 3x6 restriction | 18 | 2.045 | 77.56% | 95.73% | 100.00% | 40.406 | 24.7 |

## Table 6. MeanShift bandwidth sensitivity at fixed 6x6

| Bandwidth | MLE (m) ↓ | P90 (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ |
|---:|---:|---:|---:|---:|---:|
| 1 m | 19.148 | 52.091 | 22.89% | 36.39% | 56.28% |
| 3 m | 2.225 | 4.384 | 73.66% | 94.60% | 100.00% |
| 5 m | 2.078 | 4.164 | 76.51% | 95.56% | 100.00% |
| 7 m | 2.045 | 4.111 | 77.56% | 95.73% | 100.00% |
| 9 m | 2.034 | 4.096 | 77.96% | 95.73% | 100.00% |
| 11 m | 2.029 | 4.099 | 78.10% | 95.73% | 100.00% |

## Table 7. Runtime efficiency of the proposed architecture

| Configuration | Local Geometry | Visually Scored Candidates | Final MS Grid | Latency (ms) ↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|
| Proposed Method | 6x6 | 18 | 6x6 | 40.406 | 24.7 |

## Audit

- final MeanShift grid fixed to 6x6 for all experiments: PASS
- Route-A-only temporal training; B+C evaluation only: PASS
- no-forward variant separately retrained on Route A: PASS
- 1-frame and 2-frame variants separately retrained on Route A: PASS
- no-GRU uses Kalman CV fallback, not forced zero motion: PASS
- no-Kalman bypasses external Kalman fusion: PASS
- no-final-MS returns the pre-MS estimator output directly: PASS
- pooled LSR@3/5/10 recomputed from B+C frame-level errors: PASS
- main ablation E2E rows measured sequentially on physical GPU0: PASS
- training patience = 5: PASS
