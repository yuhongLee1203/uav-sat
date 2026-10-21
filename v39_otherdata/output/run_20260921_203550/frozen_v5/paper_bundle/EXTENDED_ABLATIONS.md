# Extended frozen V5 ablations

## Final MeanShift grid
| Grid | MLE | P90 | LSR@5 | Final-MS latency ms |
|---|---:|---:|---:|---:|
| 4x4 | 4.1019 | 7.4844 | 70.081 | 3.1340 |
| 5x5 | 3.6213 | 6.4839 | 77.883 | 3.7892 |
| 6x6 | 3.6211 | 6.4841 | 77.883 | 5.0761 |
| 7x7 | 3.5923 | 6.4264 | 78.562 | 6.7525 |
| 8x8 | 3.5928 | 6.4272 | 78.562 | 7.8517 |

## Search geometry
| Search | Candidates | MLE | P90 | Capture | End-to-end latency ms |
|---|---:|---:|---:|---:|---:|
| Full 6x6 | 36 | 3.3757 | 6.1731 | 100.000 | 36.9150 |
| Forward 3x6 | 18 | 3.6211 | 6.4841 | 99.118 | 28.2738 |

## Visual decoder (accuracy; aggregation-only timing is in decoder_microbenchmark/)
| Decoder | MLE | P90 | LSR@5 |
|---|---:|---:|---:|
| Weighted | 3.6305 | 6.6088 | 74.423 |
| MeanShift | 3.3757 | 6.1731 | 78.290 |
