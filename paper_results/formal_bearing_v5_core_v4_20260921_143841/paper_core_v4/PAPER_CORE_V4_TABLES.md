# Core V4 paper tables

## Table 1. Component ablation

| Label | MLE_m | MedLE_m | P90_m | LSR@5_pct | LSR@15_pct | JumpRate_pct | MaxFinalStep_m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w/o GRU | 4.158 | 3.922 | 7.053 | 68.792 | 99.932 | 0.882 | 24.670 |
| w/o Kalman | 3.658 | 3.350 | 6.535 | 76.526 | 100.000 | 0.746 | 24.926 |
| w/o Final MeanShift | 6.658 | 6.475 | 11.497 | 34.600 | 98.168 | 7.598 | 27.168 |
| Full | 3.658 | 3.366 | 6.502 | 76.866 | 100.000 | 0.746 | 24.657 |

Protocol: every row uses the same trained 3-frame Full checkpoint; only the named component is disabled.

## Table 2. Temporal-context truncation

| Label | MLE_m | P90_m | LSR@5_pct | LSR@15_pct | JumpRate_pct |
| --- | --- | --- | --- | --- | --- |
| 1 frame context | 3.644 | 6.484 | 76.934 | 100.000 | 0.746 |
| 2 frame context | 3.644 | 6.496 | 77.069 | 100.000 | 0.746 |
| Full | 3.658 | 6.502 | 76.866 | 100.000 | 0.746 |

Protocol: 1f/2f/3f use the same trained 3-frame Full checkpoint. No 1f/2f retraining is used in this table.

## Audit

- Full-best core check: {'MLE': False, 'P90': True, 'LSR5': True, 'LSR15': True, 'JumpRate': True}
- Full-best temporal check: {'MLE': False, 'P90': False, 'LSR5': False, 'LSR15': True, 'JumpRate': True}
- Kalman/runtime profile is selected only from train_01 validation calibration.
- Held-out nav50/nav51 metrics are never used for profile selection or automatic retuning.
