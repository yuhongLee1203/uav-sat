# Extended frozen V5 ablations

## Final MeanShift grid
| Grid | MLE | P90 | LSR@5 | Final-MS latency ms |
|---|---:|---:|---:|---:|
| 4x4 | 4.3726 | 7.8936 | 63.900 | 2.8990 |
| 5x5 | 3.8891 | 6.8054 | 71.862 | 3.8153 |
| 6x6 | 3.8863 | 6.8036 | 71.930 | 6.0410 |
| 7x7 | 3.8656 | 6.7307 | 72.402 | 6.7188 |
| 8x8 | 3.8662 | 6.7312 | 72.402 | 8.5988 |

## Search geometry
| Search | Candidates | MLE | P90 | Capture | End-to-end latency ms |
|---|---:|---:|---:|---:|---:|
| Full 6x6 | 36 | 3.6241 | 6.6239 | 100.000 | 42.4132 |
| Forward 3x6 | 18 | 3.8863 | 6.8036 | 98.650 | 32.4378 |

## Visual decoder (accuracy; aggregation-only timing is in decoder_microbenchmark/)
| Decoder | MLE | P90 | LSR@5 |
|---|---:|---:|---:|
| Weighted | 3.8077 | 6.8804 | 72.132 |
| MeanShift | 3.6241 | 6.6239 | 74.764 |
