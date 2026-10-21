# Run 20260921_203550: measured-result summary

## Completion

- Four cities completed: CityA, CityB, CityC, CityD.
- Eight custom full-method figures completed (`test_01`, `test_02` per city), plus the four-city contact sheet.
- Grid, search geometry, decoder, official Sat./UAV localization tables and raw official figures completed.
- Official closed-loop navigation was intentionally not run (`RUN_NAVIGATION=0`).

## Frozen V5 custom local-prior tracker (all four cities, 1,474 frames)

| Variant | MLE m | P90 m | LSR@5 |
|---|---:|---:|---:|
| Full | 3.6211 | 6.4841 | 77.883% |
| w/o GRU | 4.1021 | 7.0114 | 69.878% |
| w/o Kalman | 3.5870 | 6.4329 | 78.290% |
| w/o MeanShift | 6.5587 | 11.4751 | 37.449% |
| 1 frame | 3.5972 | 6.4306 | 78.155% |
| 2 frames | 3.6040 | 6.4233 | 77.680% |

Verdict: Full strongly supports the GRU and final MeanShift components. It does **not** support a claim that the Kalman component or 3-frame input is universally accuracy-best: `w/o Kalman` and 1-frame are marginally better on pooled MLE/LSR@5. These are measured results and must not be edited or selectively hidden.

## Search / decoder / grid trade-offs

| Setting | MLE m | P90 m | LSR@5 | latency ms |
|---|---:|---:|---:|---:|
| Full 6x6 search (36 candidates) | 3.3757 | 6.1731 | 78.290% | 36.915 end-to-end |
| Forward 3x6 search (18 candidates) | 3.6211 | 6.4841 | 77.883% | 28.274 end-to-end |
| Weighted decoder, 36 candidates | 3.6305 | 6.6088 | 74.423% | 32.726 end-to-end |
| MeanShift decoder, 36 candidates | 3.3757 | 6.1731 | 78.290% | 36.915 end-to-end |

The 18-candidate forward search is 23.4% faster than full 36 candidates while losing 0.2454 m MLE. MeanShift beats Weighted on all reported accuracy metrics, with additional end-to-end cost. The final MeanShift 6x6 window is a balance point: 7x7/8x8 achieve a small MLE gain (3.5923/3.5928 m) but increase final-MS latency (6.7525/7.8517 ms versus 5.0761 ms).

## Official Bearing-UAV evaluation

| View | Recall@1 | LSR@15 | HSR@15 | MLE m | MHE deg |
|---|---:|---:|---:|---:|---:|
| Sat. (all cities) | 90.833% | 98.700% | 97.944% | 5.286 | 4.161 |
| UAV (all cities) | 83.633% | 89.067% | 77.967% | 8.588 | 12.631 |

These official-model metrics are separate from the frozen V5 custom tracker and should not be mixed into its ablation claims.

## Publication/upload decision

Do not upload this run as a claim that Full, Kalman and 3-frame are all best. The run is complete and valid, but `FULL_TREND_CHECK=FAIL` for every city and for the four-city pooled audit. The source tables, figures and audit remain preserved in this run directory for revision.
