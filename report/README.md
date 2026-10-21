# FieldAnchor report figures and tables

This folder contains 16 figures (both PDF and PNG), 6 numeric tables, paired frame-level data, and a reproducible plotting script. The figures use the **original v13 City A/B/C/D GT-assisted replay**: 1,482 held-out observations on eight synthetic routes. The local reference center and acquisition hypothesis at every frame use current-frame GT. They must **not** be presented as the v40 no-GT-at-inference result.

## Where to find each requested item

| Requested item | Figure files | Numeric source |
| --- | --- | --- |
| Main method comparison: MLE, LSR@15 | [MLE](figures/main/main_mle.pdf), [LSR@15](figures/main/main_lsr15.pdf) | [main_comparison.csv](tables/main_comparison.csv) |
| Component ablation | [component_ablation.pdf](figures/ablation/component_ablation.pdf) | [component_ablation.csv](tables/component_ablation.csv) |
| Temporal input ablation | [temporal_window.pdf](figures/ablation/temporal_window.pdf) | [temporal_window.csv](tables/temporal_window.csv) |
| Final neighborhood accuracy and speed | [accuracy](figures/ablation/final_neighborhood_accuracy.pdf), [runtime](figures/ablation/final_neighborhood_runtime.pdf) | [final_neighborhood.csv](tables/final_neighborhood.csv) |
| Per-city MLE, LSR@5, P90 | [MLE](figures/city/per_city_mle.pdf), [LSR@5](figures/city/per_city_lsr5.pdf), [P90](figures/city/per_city_p90.pdf) | [per_city.csv](tables/per_city.csv) |
| Per-city reference-to-model improvement | [MLE](figures/baseline/per_city_baseline_mle.pdf), [LSR@5](figures/baseline/per_city_baseline_lsr5.pdf) | [per_city.csv](tables/per_city.csv), [per_route_reference_improvement.csv](tables/per_route_reference_improvement.csv) |
| Error CDF | [error_cdf.pdf](figures/baseline/error_cdf.pdf) | [error_cdf.csv](data/error_cdf.csv), [frame_errors_and_trajectory.csv](data/frame_errors_and_trajectory.csv) |
| Qualitative routes with GT/reference/full | [City A](figures/trajectory/citya_routes.pdf), [City B](figures/trajectory/cityb_routes.pdf), [City C](figures/trajectory/cityc_routes.pdf), [City D](figures/trajectory/cityd_routes.pdf) | [frame_errors_and_trajectory.csv](data/frame_errors_and_trajectory.csv) |

PNG versions are alongside every PDF. [REPORT.md](REPORT.md) contains the tables and interpretation; [PAPER_CAPTIONS.md](PAPER_CAPTIONS.md) has draft English captions.

## Protocol and sources

- **Published methods:** UAV-view columns in Table 2 of [Liu et al., *Beyond Matching to Tiles*, CVPR 2026](https://arxiv.org/abs/2603.22153), including Bearing-UAV with weather augmentation (WA). These are the authors' reported values; no published baseline model was run here.
- **Our Method:** original v13 [replay audit](../v39_otherdata/output/original3759_inference_20260925_220434_1615534/RESULTS_FOR_PAPER.md) and its paired frame CSVs. The 3.759 m MLE and 100% LSR@15 were recomputed from 1,482 frame errors in [recomputation_checks.json](data/recomputation_checks.json).
- **Reference baseline:** `prior_center_x/y` in the same replay. Its pooled MLE is 5.372 m. This is a **GT-centered jittered reference**, not an independent navigation or no-GT baseline.
- **Important:** the published rows and Our Method have different local-reference and evaluation protocols. The main charts are a visual context comparison, **not** a controlled ranking or an apples-to-apples claim. The v40 no-GT validation results belong to a separate experiment and do not appear in these figures.
- **Final MS latency:** use the original replay's `paper_ablation_results.csv` timing summary to match the reported 4.04/5.00/7.52/9.50/10.27 ms values. The [window table](tables/final_neighborhood.csv) also includes the recomputed frame-weighted mean to make this choice auditable.

To regenerate the figures and tables from the saved original replay:

```bash
cd /yh/study/uav-sat && python3 report/scripts/generate.py
```
