# Main Comparison Table

| Method | LSR@15 (%) ↑ | MLE (m) ↓ |
| --- | ---: | ---: |
| University-1652 | 15.11 | 33.15 |
| SUES-200 | 15.76 | 30.83 |
| DenseUAV | 16.54 | 28.79 |
| GTA-UAV | 27.96 | 28.43 |
| Bearing-UAV | 89.36 | 8.61 |
| Bearing-UAV + WA | 92.88 | 7.48 |
| FieldAnchor | 100.00 | 3.76 |

## Source and protocol

- University-1652 through Bearing-UAV + WA: published UAV-view values from [Bearing-UAV Table 2](https://arxiv.org/abs/2603.22153); these rows were not rerun here.
- FieldAnchor: [original v13 inference metrics](../v39_otherdata/output/original3759_inference_20260925_220434_1615534/original_3759_all_metrics.json), 1,482 held-out frames, MLE 3.7591277 m rounded to 3.76 m.
- FieldAnchor uses a GT-assisted local/path reference during acquisition. The published rows and FieldAnchor do not share an identical input protocol; this table is a reported-values comparison, not a same-protocol benchmark.
