# Core V5-Restore paper tables

## Table 1. Component ablation

| Label | MLE_m | MedLE_m | P90_m | LSR@5_pct | LSR@15_pct | JumpRate_pct | MaxFinalStep_m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w/o GRU | 4.137 | 3.909 | 7.009 | 68.792 | 99.864 | 1.018 | 24.670 |
| w/o Kalman | 3.594 | 3.303 | 6.441 | 76.798 | 100.000 | 0.678 | 24.781 |
| w/o Final MeanShift | 6.540 | 6.274 | 11.493 | 38.128 | 97.897 | 7.191 | 27.935 |
| Full | 3.626 | 3.337 | 6.501 | 76.526 | 100.000 | 0.814 | 24.532 |

Protocol: all rows use the same restored 3-frame Full checkpoint; only the named component is disabled.

## Table 2. Temporal-context truncation

| Label | MLE_m | P90_m | LSR@5_pct | LSR@15_pct | JumpRate_pct |
| --- | --- | --- | --- | --- | --- |
| 1 frame context | 3.599 | 6.479 | 77.069 | 100.000 | 0.814 |
| 2 frame context | 3.606 | 6.480 | 77.001 | 100.000 | 0.814 |
| Full | 3.626 | 6.501 | 76.526 | 100.000 | 0.814 |

Protocol: 1f/2f/3f reuse the same restored 3-frame Full checkpoint.

## Audit

- Full-best core check: {'MLE': False, 'P90': False, 'LSR5': False, 'LSR15': True, 'JumpRate': False}
- Full-best temporal check: {'MLE': False, 'P90': False, 'LSR5': False, 'LSR15': True, 'JumpRate': True}
- Restored estimator dynamics are fixed before held-out evaluation.
- Held-out nav50/nav51 are not used for automatic parameter selection.
