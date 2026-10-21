# FieldAnchor supplementary thesis experiments

## Scope and provenance

- Source: `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368`. This run analyzes the existing v39 Full replay and paired ablation/probe CSVs; **no checkpoint was retrained or modified**.
- Protocol: **GT-assisted current-frame reference and hypothesis selection**. It is not a no-GT localization evaluation. Routes are synthetic sequences assembled from individually geotagged Bearing-UAV-90K images, rather than continuous UAV video.
- The four city evaluations share one temporal checkpoint (identical SHA-256) but use four distinct city visual checkpoints. Thus “one complete checkpoint for all cities” would be inaccurate.
- Every test route is independent. The source `run_route_inference` creates a new Kalman state and sets `hidden=None` before processing each route. Stored CSVs cannot independently prove execution-state behavior beyond this source audit.
- Numerical tables below are recomputed from the original per-frame predictions, not copied from thesis prose. MLE/median are in meters; LSR is percentage of frames with Euclidean error at most the threshold.

## 1. Route-wise localization — Chapter 4, stability across routes

| City | Route | Samples | MLE (m) | Median Error (m) | LSR@3 (%) | LSR@15 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | test_01 | 161 | 3.66 | 3.40 | 44.72 | 100.00 |
| A | test_02 | 219 | 3.65 | 3.37 | 42.01 | 100.00 |
| B | test_01 | 146 | 3.85 | 3.71 | 37.67 | 100.00 |
| B | test_02 | 171 | 3.59 | 3.32 | 42.69 | 100.00 |
| C | test_01 | 148 | 3.57 | 3.25 | 45.27 | 100.00 |
| C | test_02 | 226 | 4.19 | 3.76 | 38.94 | 100.00 |
| D | test_01 | 150 | 4.08 | 3.48 | 40.67 | 100.00 |
| D | test_02 | 261 | 3.52 | 3.33 | 43.30 | 100.00 |

Pooled sample-level aggregate is saved separately in [overall_results.json](overall_results.json): MLE 3.759 m, LSR@3 41.90%, LSR@15 100.00%. No Overall row appears in the route table. [CSV](tables/route_wise_results.csv) · [JSON](tables/route_wise_results.json)

## 2. Final decoder — Chapter 4, what final Mean Shift itself contributes

| Final Decoder | MLE (m) | Median Error (m) | LSR@3 (%) | LSR@15 (%) |
| --- | --- | --- | --- | --- |
| Argmax | 4.14 | 3.76 | 36.10 | 100.00 |
| Weighted Average | 3.75 | 3.44 | 42.51 | 100.00 |
| Mean Shift | 3.76 | 3.44 | 41.90 | 100.00 |

All three decoders use the same 36 final candidate positions and **regularized logits** from the exact same UAV/front Mean Shift/GRU/Kalman output. The logits include visual similarity plus Kalman/current-GT-derived reference priors; raw visual-only similarities were not logged and are not mislabeled as such. Argmax selects the highest logit; Weighted Average uses softmax(logit/0.30); Mean Shift is the original recorded decoder. Mean Shift output matches Full frame by frame. The comparison should be reported as measured even if another decoder wins. [CSV](tables/final_decoder_comparison.csv) · [raw NPZ](raw/final_decoder_predictions.npz)

## 3. Stage-wise localization — Chapter 4, stage contributions

| Estimation Stage | MLE (m) | Median Error (m) | LSR@3 (%) | LSR@15 (%) |
| --- | --- | --- | --- | --- |
| Supplied Local Reference | 5.37 | 5.59 | 0.00 | 100.00 |
| Front Mean Shift | 9.06 | 8.88 | 7.42 | 91.43 |
| Corrected Visual Measurement | 7.88 | 7.41 | 8.91 | 94.67 |
| Kalman Posterior | 6.79 | 6.58 | 15.45 | 98.04 |
| Final Mean Shift | 3.76 | 3.44 | 41.90 | 100.00 |

Corrected Visual Measurement x/y is reconstructed from saved measurement_s/e and the unmodified WaypointRoute.xy_from_se mapping (36 m post-turn route-frame smoothing). All listed stages are map-meter positions. Front Mean Shift uses saved softms_x/y. Supplied Reference uses prior_center_x/y, not frame_reference_x/y (the latter is a separate GT-progress centerline prior used inside final scoring).

These stages need not improve monotonically: the Kalman output can be worse than the supplied reference while the final GT-conditioned refinement recovers accuracy. [CSV](tables/stage_wise_results.csv) · [raw NPZ](raw/stage_wise_predictions.npz)

## 4. Computational cost — Chapter 4, efficiency

| Metric | Value |
| --- | --- |
| Trainable Parameters | 2848021 |
| Total Parameters | 3775029 |
| Mean End-to-End Latency (ms) | 39.88 |
| Median End-to-End Latency (ms) | 37.56 |
| Latency Standard Deviation (ms) | 6.32 |
| FPS | 25.08 |
| GPU | NVIDIA GeForce RTX 3090 (original replay distributed across GPU 0/5/6) |
| Precision | FP32 compute; some precomputed feature caches stored as FP16 |
| Batch Size | 1 |
| UAV Input Resolution | 256x256 |
| Timing Observations | 1082 |
| Warm-up Observations per Route | 50 |

The original per-frame timer synchronizes CUDA before and after the complete image-tensor-to-coordinate inference, including fresh UAV backbone extraction. Analysis excludes the first 50 observations of each route, leaving 1,082 timed frames. Timing excludes disk I/O, preprocessing, model loading, and satellite gallery construction; concurrent GPU load in the replay was uncontrolled. Hardware/software details and parameter breakdown are in [environment.txt](environment.txt). [CSV](tables/runtime_results.csv) · [raw ms](raw/runtime_raw_ms.csv)

## 5–6. Localization trajectories — Chapter 4 result figure and appendix

[(Four-city PNG)](figures/representative_trajectories.png) · [(PDF)](figures/representative_trajectories.pdf) · [(SVG)](figures/representative_trajectories.svg) · [(Eight-route PNG)](figures/all_eight_routes_appendix.png) · [(PDF)](figures/all_eight_routes_appendix.pdf). Both use each city’s **complete satellite map** as background (original 4096 × 4096 pixels, 1024 × 1024 m; rendered at 2048 pixels for manageable paper files). The green Ground Truth Route joins the saved start/end and only waypoints with an absolute direction change ≥30° using straight segments (a display-only major-turn simplification); the red FieldAnchor line joins **raw, unsmoothed per-frame predictions**. Per-frame GT coordinates remain the source of all numeric localization metrics and are intentionally not drawn as a jagged line or scattered points. The four-city figure uses test_01 for every city; the eight-route figure reports each route MLE. The four official *_50 files have identical XY waypoint arrays, so test_01 route shape is intentionally repeated; each panel uses a different city satellite, UAV observations, predictions, and per-frame error trace. [Original map audit](raw/trajectory_map_audit.json) · [duplicate-route and result-source audit](raw/trajectory_duplicate_audit.json).

## 7. Real final responses — Chapter 4 decoder interpretation

[(PNG)](figures/final_response_visualization.png) · [(PDF)](figures/final_response_visualization.pdf) · [four raw samples](raw/final_response_visualization_samples.npz). Panels are algorithmically selected from unedited 6×6 response logits. “Multiple local lattice peaks” means separate local maxima on the discrete 6×6 grid; it does **not** mean multiple converged Mean Shift modes. All 1,482 recorded final outputs had one active Mean Shift mode. Every panel uses the same 36 × 36 m map-coordinate window about its Kalman center.

## 8. Error along route — Chapter 4, sequential behavior

[(Raw route PNG)](figures/localization_error_over_sequence.png) · [(PDF)](figures/localization_error_over_sequence.pdf) · [(Normalized progress PNG)](figures/error_vs_route_progress.png) · [(PDF)](figures/error_vs_route_progress.pdf) · [101-bin data](raw/error_vs_route_progress.csv). Each route is independently interpolated to 0–100% progress only for the aggregate progress plot; original predictions themselves are never interpolated or smoothed. Shading is route-level IQR.

## 9. Improvement over supplied reference — Chapter 4, reference-conditioned refinement

| City | Route | Reference MLE (m) | FieldAnchor MLE (m) | Improvement (m) | Improvement (%) |
| --- | --- | --- | --- | --- | --- |
| A | test_01 | 5.60 | 3.66 | 1.94 | 34.69 |
| A | test_02 | 5.24 | 3.65 | 1.59 | 30.29 |
| B | test_01 | 5.65 | 3.85 | 1.80 | 31.87 |
| B | test_02 | 5.55 | 3.59 | 1.96 | 35.34 |
| C | test_01 | 5.64 | 3.57 | 2.08 | 36.79 |
| C | test_02 | 5.18 | 4.19 | 0.99 | 19.16 |
| D | test_01 | 5.64 | 4.08 | 1.55 | 27.53 |
| D | test_02 | 4.93 | 3.52 | 1.41 | 28.59 |

[(CSV)](tables/route_reference_improvement.csv) · [(PNG)](figures/route_reference_improvement.png) · [(PDF)](figures/route_reference_improvement.pdf). The reference is the saved `prior_center_x/y`, matching the publication audit reference MLE 5.372 m.

## 10. Localization error distribution — Chapter 4, robustness

| Method | Mean (m) | Median (m) | P75 (m) | P90 (m) | P95 (m) | Max (m) |
| --- | --- | --- | --- | --- | --- | --- |
| Supplied Reference | 5.37 | 5.59 | 5.89 | 5.98 | 6.00 | 6.00 |
| FieldAnchor | 3.76 | 3.44 | 5.12 | 6.84 | 8.15 | 12.90 |
| w/o GRU | 4.28 | 4.06 | 5.67 | 7.42 | 8.62 | 17.38 |
| w/o Final Refinement | 6.79 | 6.58 | 9.21 | 11.51 | 12.95 | 18.38 |

[(CSV)](tables/localization_error_distribution.csv) · [(CDF PNG)](figures/localization_error_cdf.png) · [(PDF)](figures/localization_error_cdf.pdf). CDF x-axis includes all observed errors; w/o Final Refinement is the **whole final refinement stage** removal.

## 11. Dataset sequence statistics — Experimental setup

| City | Train Sequence | Train Samples | Validation Samples | Test_01 Samples | Test_02 Samples | Train Route Total | Unused Guard/Tail Samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | train_01 | 122 | 1 | 161 | 219 | 198 | 75 |
| B | train_01 | 121 | 1 | 146 | 171 | 196 | 74 |
| C | train_01 | 121 | 1 | 148 | 226 | 196 | 74 |
| D | train_01 | 120 | 1 | 150 | 261 | 195 | 74 |

Train/validation counts above use the **temporal trainer** `split_ranges` on each saved `train_01` route: 70%/15% chronological cut points, 16-frame guards, with a one-frame minimum temporal validation interval. The remaining train-route observations are guards/tail, not a new split. The visual checkpoints are reused per city; their original visual train/validation observation counts cannot be reconstructed reliably from these test manifests and are not claimed here. [CSV](tables/dataset_split_statistics.csv)

## 12. Existing ablations exported as machine-readable data

### Components

| Variant | GRU | Kalman | Final Refinement | LSR@3 (%) | LSR@15 (%) | MLE (m) |
| --- | --- | --- | --- | --- | --- | --- |
| w/o GRU | No | Yes | Yes | 32.66 | 99.87 | 4.28 |
| w/o Kalman | Yes | No | Yes | 41.30 | 100.00 | 3.76 |
| w/o Final Refinement | Yes | Yes | No | 15.45 | 98.04 | 6.79 |
| Full | Yes | Yes | Yes | 41.90 | 100.00 | 3.76 |

`w/o Final Refinement` removes final-neighborhood construction, candidate rescoring, and Mean Shift decoding together. It is not a single-function Mean Shift ablation. [CSV](tables/component_ablation.csv)

### Temporal context

| Input Length (frames) | LSR@3 (%) | LSR@15 (%) | MLE (m) |
| --- | --- | --- | --- |
| 1 | 39.81 | 100.00 | 3.88 |
| 2 | 39.14 | 100.00 | 3.92 |
| 3 | 41.90 | 100.00 | 3.76 |

[CSV](tables/temporal_context_ablation.csv)

### Final neighborhood

| Window | Candidates | LSR@3 (%) | LSR@15 (%) | MLE (m) | Measured Mean Shift Time (ms) |
| --- | --- | --- | --- | --- | --- |
| 4x4 | 16 | 37.65 | 99.93 | 4.17 | 3.83 |
| 5x5 | 25 | 41.90 | 100.00 | 3.76 | 5.61 |
| 6x6 | 36 | 41.90 | 100.00 | 3.76 | 7.21 |
| 7x7 | 49 | 42.04 | 100.00 | 3.74 | 8.59 |
| 8x8 | 64 | 42.04 | 100.00 | 3.74 | 10.96 |

Latency here is the per-frame final Mean Shift latency recorded in this v39 replay, recomputed from CSV; it may differ from a separately timed manuscript table. [CSV](tables/final_neighborhood_ablation.csv)

### Explicit discrepancy audit against the numbers supplied with this request

| Row / Metric | Supplied manuscript value | Recomputed original v39 replay | Difference | Likely explanation |
| --- | ---: | ---: | ---: | --- |
| 5×5 MLE (m) | 3.77 | 3.76 (exact 3.762507) | −0.01 displayed | The saved replay rounds to 3.76; the manuscript value may come from another run or a different rounding source. The original manuscript run cannot be reconstructed from this CSV alone. |
| 4×4 Mean Shift time (ms) | 4.04 | 3.83 | −0.21 | Timing run/environment differs; these are actual saved per-frame replay timings. |
| 5×5 Mean Shift time (ms) | 5.00 | 5.61 | +0.61 | Same timing provenance difference. |
| 6×6 Mean Shift time (ms) | 7.52 | 7.21 | −0.31 | Same timing provenance difference. |
| 7×7 Mean Shift time (ms) | 9.50 | 8.59 | −0.91 | Same timing provenance difference. |
| 8×8 Mean Shift time (ms) | 10.27 | 10.96 | +0.69 | Same timing provenance difference. |

The ablation MLE/LSR values otherwise agree with the supplied table at two decimals. Do not silently substitute one timing series for the other in the thesis. Source: per-route frame CSVs under the original v39 replay; evaluator: this script `ablation()`. Exact hardware load of the manuscript timing run is not available.

## Consistency and limits

- Eight routes total 1,482 frames. Full and all ablations align on frame ID, UAV image path, and GT x/y; decoder probe Mean Shift equals original Full for every frame.
- The 6×6 row equals Full by construction from the same original Full predictions.
- The saved CSVs and source code show separate route invocations with fresh recurrent and Kalman state. No model was retrained. Figures use raw saved predictions and no clipped errors.
- The 3.759 m result depends on current-frame GT-informed reference/hypothesis and cannot support a no-GT claim. Direct comparison against published full retrieval methods requires protocol qualification.
- Original paper numbers sometimes used different rounding/timing runs; compare original decimal and recomputed columns before replacing thesis tables. [Machine audit](audit.json).

Recreate all outputs with `python3 report/add_more/generate_thesis_experiments.py` from `/yh/study/uav-sat`.
