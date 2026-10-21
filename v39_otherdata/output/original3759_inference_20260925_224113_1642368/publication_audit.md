# Publication audit: original 3.759 m replay

**Verdict:** Report these results only as a GT-assisted localization ablation with the exact test-time reference protocol disclosed. They do not establish no-GT localization or closed-loop navigation performance.

Historical-value proximity check: PASS; exact printed-value match: NO. Test set: 1,482 frames, eight synthetic routes, City A/B/C/D.
Exact mismatch: no_kalman: MLE 3.763/3.763; LSR@5 73.82%/73.68%

## Test-time GT use

Current-frame GT centers the local search prior on 1,482/1,482 frames. The selected acquisition hypothesis matches the GT target hypothesis on 1,482/1,482 frames. The GT-centered prior alone has 5.372 m MLE and 21.46% LSR@5.
The inference path also passes current GT progress and step bounds to the motion estimator. A per-frame final-cap event flag of zero does not mean GT was absent.

## Paired MLE differences

Full minus comparison, in metres; negative favors Full. The 95% interval resamples the eight whole routes, preserving route frame counts.

| Comparison | Difference | Route bootstrap 95% interval | Routes Full better |
|---|---:|---:|---:|
| no_gru | -0.5159 | [-0.6150, -0.4179] | 8/8 |
| no_kalman | -0.0036 | [-0.0116, +0.0048] | 4/8 |
| no_ms | -3.0264 | [-3.3568, -2.6322] | 8/8 |
| frames1 | -0.1239 | [-0.1544, -0.0983] | 8/8 |
| frames2 | -0.1619 | [-0.1870, -0.1342] | 8/8 |
| grid4 | -0.4091 | [-0.5278, -0.2699] | 8/8 |
| grid5 | -0.0034 | [-0.0102, +0.0003] | 4/8 |
| grid7 | +0.0212 | [+0.0102, +0.0313] | 0/8 |
| grid8 | +0.0208 | [+0.0099, +0.0308] | 0/8 |

## Paper-facing conclusions

- Three frames has lower MLE than one and two frames on every route.
- Full and w/o Kalman have comparable MLE; the paired interval crosses zero.
- The original 6x6 Full window is slower and less accurate than 7x7 and 8x8 in this test. Both larger windows have lower MLE on all eight routes.
- LSR@15 saturates at 100% for several rows. Official heading, Recall@1, SR@20, SPL and NE are unavailable in this evaluation.

A changed Full configuration chosen after viewing these test results is exploratory. A new best-model claim needs selection on training/validation data and evaluation on a fresh unseen test split.
