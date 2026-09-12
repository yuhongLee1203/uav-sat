# V39 Final Experiment Data Summary

Final validated architecture:

**Weighted Centroid -> context-aware 3-frame GRU residual/velocity refinement -> fixed-R external Kalman -> one final 5x5 MeanShift**

Latest validated suite:

`v39_DirectFinalMS/wc_gru_velocity_fusion_20260912_172752`

Training/evaluation split:

- Train: Route A only
- Evaluation: Route B and Route C only
- Backbone: `torchvision:mobilenet_v3_small`
- Temporal input: 3 frames
- Early-stopping patience: 5
- Kalman mode: fixed-R
- Full-model motion input: learned GRU velocity
- No-GRU fallback: external Kalman internal constant-velocity state
- Final MeanShift: exactly one online MS
- Selected final MS grid: 5x5
- Runtime GPU: RTX 3090

> Important protocol note: these final numbers are from the controlled local-refinement protocol (`controlled_gt_jitter`). Internally, the experiment uses the current-frame controlled local prior and waypoint coordinates. These results must not be described as a fully autonomous no-prior closed-loop localization result.

---

## 1. Final GRU ablation — primary result

| Setting | B MLE (m) | C MLE (m) | B+C MLE (m) | B+C MedLE (m) | B+C P90 (m) | B+C P95 (m) | B+C P99 (m) | LSR@5 | LSR@10 | LSR@15 | LSR@20 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No GRU | 2.384 | 2.195 | 2.317 | 2.209 | 4.231 | 4.842 | 6.120 | 95.61% | 100.00% | 100.00% | 100.00% | 3534 |
| **Full context-GRU** | **2.191** | **1.779** | **2.045** | **1.717** | **4.110** | **4.823** | **5.911** | **95.73%** | **100.00%** | **100.00%** | **100.00%** | 3534 |

Main conclusion:

- Full GRU improves pooled B+C MLE from **2.317 m to 2.045 m**.
- Relative MLE improvement: **11.75%**.
- Route B improvement: **8.09%**.
- Route C improvement: **18.93%**.
- Full model is better on B, C, pooled MLE, pooled MedLE, pooled P90 and pooled P99.
- Final audit status: **PASS**.

---

## 2. Route B — complete final-model metrics

| Metric | No GRU | Full GRU |
|---|---:|---:|
| MLE (m) | 2.384 | **2.191** |
| MedLE (m) | 2.366 | **1.981** |
| P90 (m) | **4.079** | 4.152 |
| P95 (m) | **4.545** | 4.702 |
| P99 (m) | 5.728 | **5.712** |
| LSR@5 | **97.10%** | 96.62% |
| LSR@10 | 100.00% | 100.00% |
| LSR@15 | 100.00% | 100.00% |
| LSR@20 | 100.00% | 100.00% |
| Selected-candidate capture | 89.59% | 89.63% |
| Bank-candidate capture | 89.59% | 89.63% |
| Acquisition accuracy | 100.00% | 100.00% |
| Mean acquisition confidence | 0.478 | 0.473 |
| Heading MAE (deg) | 14.416 | **12.293** |
| Jump rate | 0.00% | 0.00% |
| Mean final step (m) | 3.178 | 3.173 |
| Max final step (m) | 9.105 | **8.830** |
| Mean excess step over reference step (m) | 0.337 | **0.333** |
| Kalman step-limited rate | 39.50% | 73.20% |
| Mean speed error (m/frame) | 3.135 | **0.566** |
| Mean progress error (m) | 14.400 | **5.822** |
| Motion-prediction MAE (m) | 5.585 | **4.740** |
| Motion-prediction P90 (m) | 9.826 | **9.029** |
| Visual-measurement MAE (m) | 9.654 | **8.187** |
| Visual-measurement P90 (m) | 15.620 | **13.647** |
| Kalman MAE (m) | 5.511 | **4.808** |
| Final MS mean shift from Kalman (m) | 3.464 | **2.949** |
| Final MS max shift from Kalman (m) | **8.247** | 10.089 |
| Final predicted waypoint leg | 14 | 14 |
| Final reference waypoint leg | 14 | 14 |

GRU contribution on Route B:

- Speed MAE reduction: **81.96%**.
- Motion-prediction MAE reduction: **15.13%**.
- Visual-measurement MAE reduction: **15.20%**.
- Kalman MAE reduction: **12.74%**.
- Progress MAE reduction: **59.57%**.

---

## 3. Route C — complete final-model metrics

| Metric | No GRU | Full GRU |
|---|---:|---:|
| MLE (m) | 2.195 | **1.779** |
| MedLE (m) | 1.760 | **1.341** |
| P90 (m) | 4.517 | **3.944** |
| P95 (m) | 5.328 | **5.219** |
| P99 (m) | 6.815 | **6.060** |
| LSR@5 | 92.93% | **94.12%** |
| LSR@10 | 100.00% | 100.00% |
| LSR@15 | 100.00% | 100.00% |
| LSR@20 | 100.00% | 100.00% |
| Selected-candidate capture | **91.81%** | 91.73% |
| Bank-candidate capture | **91.81%** | 91.73% |
| Acquisition accuracy | 100.00% | 100.00% |
| Mean acquisition confidence | 0.442 | 0.441 |
| Heading MAE (deg) | 35.337 | **35.280** |
| Jump rate | 0.00% | 0.00% |
| Mean final step (m) | 2.187 | 2.186 |
| Max final step (m) | 8.845 | 8.844 |
| Mean excess step over reference step (m) | **0.238** | 0.242 |
| Kalman step-limited rate | 45.71% | 66.53% |
| Mean speed error (m/frame) | 2.159 | **1.153** |
| Mean progress error (m) | **7.354** | 10.568 |
| Motion-prediction MAE (m) | 5.348 | **4.024** |
| Motion-prediction P90 (m) | 10.277 | **8.079** |
| Visual-measurement MAE (m) | 8.412 | **7.421** |
| Visual-measurement P90 (m) | 14.227 | **12.403** |
| Kalman MAE (m) | 5.267 | **4.083** |
| Final MS mean shift from Kalman (m) | 3.393 | **2.599** |
| Final MS max shift from Kalman (m) | 11.450 | **9.642** |
| Final predicted waypoint leg | 10 | 10 |
| Final reference waypoint leg | 10 | 10 |

GRU contribution on Route C:

- Speed MAE reduction: **46.61%**.
- Motion-prediction MAE reduction: **24.76%**.
- Visual-measurement MAE reduction: **11.78%**.
- Kalman MAE reduction: **22.48%**.
- Final MLE improves strongly even though the route-progress MAE itself increases; therefore Route C progress error should not be used alone as the localization-accuracy conclusion.

---

## 4. Stage-wise diagnostic: where the GRU helps

| Route | Setting | Motion prediction MAE | Visual measurement MAE | Kalman MAE | Final MLE |
|---|---|---:|---:|---:|---:|
| B | No GRU | 5.585 | 9.654 | 5.511 | 2.384 |
| B | **Full GRU** | **4.740** | **8.187** | **4.808** | **2.191** |
| C | No GRU | 5.348 | 8.412 | 5.267 | 2.195 |
| C | **Full GRU** | **4.024** | **7.421** | **4.083** | **1.779** |

Interpretation:

- The current GRU is no longer only improving an intermediate visual feature.
- It improves the learned temporal velocity, motion prediction, visual measurement and the downstream Kalman estimate.
- This improvement is preserved through the final MeanShift stage and produces a lower final localization error.

---

## 5. Current final runtime — same RTX 3090, same revision

End-to-end timing definition:

`prepared UAV tensor -> backbone -> visual retrieval/GRU -> external RouteKalman -> final XY`

Excluded from timer:

- image disk I/O
- image preprocessing
- model/checkpoint loading
- satellite-gallery construction

### 5x5 final MeanShift

| Route | Mean latency | Median latency | P90 latency | P95 latency | FPS |
|---|---:|---:|---:|---:|---:|
| B | 36.797 ms | 35.993 ms | 44.789 ms | 47.840 ms | 27.18 |
| C | 36.430 ms | 35.903 ms | 43.792 ms | 47.421 ms | 27.45 |
| **B+C pooled** | **36.667 ms** | - | - | - | **27.27** |

### 6x6 runtime comparison

| Route | Mean latency | Median latency | P90 latency | P95 latency | FPS |
|---|---:|---:|---:|---:|---:|
| B | 39.363 ms | 38.885 ms | 46.947 ms | 50.914 ms | 25.40 |
| C | 39.903 ms | 39.070 ms | 47.480 ms | 50.683 ms | 25.06 |
| **B+C pooled** | **39.554 ms** | - | - | - | **25.28** |

Runtime conclusion:

- 5x5 is faster than 6x6.
- Pooled 5x5 FPS: **27.27 FPS**.
- Pooled 6x6 FPS: **25.28 FPS**.
- 5x5 provides about **7.9% higher FPS** than 6x6 in the latest paired same-GPU run.
- Runtime sanity audit: **PASS**.

---

## 6. Final audit checklist

| Check | Result |
|---|---|
| Fresh Route-A-only context-GRU checkpoint | PASS |
| Early-stopping patience = 5 | PASS |
| B/C evaluation only | PASS |
| GRU receives posterior-weighted SAT context | PASS |
| Full model uses learned GRU velocity | PASS |
| No-GRU uses Kalman internal CV fallback | PASS |
| No artificial zero-motion penalty in no-GRU | PASS |
| Fixed-R external Kalman | PASS |
| Exactly one online final MeanShift | PASS |
| GPU-resident gallery indexing | PASS |
| 5x5 vs 6x6 same-GPU runtime sanity | PASS |
| Full model beats no-GRU | **PASS** |

Final audit status: **PASS**.

---

# Earlier / superseded v39 experiments

The following tables are preserved for experiment history. They were generated before the current context-GRU + learned-velocity Kalman fusion redesign. They must not be mixed directly with the latest primary ablation table as if they were produced by exactly the same final architecture.

## 7. Earlier progressive architecture ablation

Earlier chain:

`Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> final MeanShift`

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| Weighted Centroid | 6.070 | 5.122 | 5.732 | 45.36% | 4.527% / 3.580% |
| + GRU | 6.352 | 5.543 | 6.064 | 42.02% | 6.066% / 4.375% |
| + Kalman | 4.805 | 4.124 | 4.563 | 60.27% | 0.044% / 0.159% |
| + final MS | 2.190 | 1.789 | 2.047 | 95.53% | 0.000% / 0.000% |

This table is useful for showing the historical development of the pipeline, but the old `+GRU` behavior is superseded by the latest context-GRU + velocity-fusion model.

---

## 8. Earlier temporal frame-count ablation

Each configuration was trained separately on Route A under the earlier v39 temporal setup.

| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |
|---:|---|---:|---:|---:|---:|
| 1 | current frame only | 2.212 | 1.822 | 2.073 | 4.133 |
| 2 | current + previous; first difference | 2.227 | 1.822 | 2.083 | 4.143 |
| **3** | current + previous two; first + second difference | **2.190** | **1.789** | **2.047** | **4.077** |

Historical conclusion:

- 3 frames gave the best B+C MLE and P90 among 1/2/3-frame variants.
- This is the reason the final architecture keeps a 3-frame temporal input.

---

## 9. Earlier final MeanShift grid sensitivity

Pure-MS latency starts after candidate centers and regularized logits are already prepared. It measures only the single `soft_mean_shift` decoder through metric XY output.

| Grid | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.216 | 4.014 ms | 249.1 |
| **5x5** | 25 | **2.047** | **5.044 ms** | **198.3** |
| 6x6 | 36 | 2.047 | 7.024 ms | 142.4 |
| 7x7 | 49 | 2.046 | 10.125 ms | 98.8 |
| 8x8 | 64 | 2.046 | 10.368 ms | 96.4 |

Selection rule:

- If accuracy is within **0.5% of the best MLE**, choose the lower-latency configuration.
- 5x5 therefore remains the selected main grid because it reaches effectively the same accuracy as larger grids while being faster.

---

## 10. Earlier runtime records — do not use as latest runtime

Older E2E records include multiple implementation revisions and therefore should not replace the latest same-GPU velocity-fusion measurement.

Historical values encountered during development:

- Older optimized run: about **34.045 ms / 29.4 FPS** pooled B+C.
- A temporary implementation with repeated full-gallery GPU-to-CPU copies measured about **110.141 ms / 9.1 FPS**.
- After the GPU-resident indexing fix, runtime returned to the expected real-time range.
- The **current authoritative runtime** is the latest paired measurement: **36.667 ms / 27.27 FPS for 5x5**.

---

## 11. Recommended numbers for the paper / presentation

Use these as the current final headline values:

- Final B MLE: **2.191 m**
- Final C MLE: **1.779 m**
- Final B+C MLE: **2.045 m**
- Final B+C MedLE: **1.717 m**
- Final B+C P90: **4.110 m**
- Final B+C P95: **4.823 m**
- Final B+C P99: **5.911 m**
- Final B+C LSR@5: **95.73%**
- Final B+C LSR@10/15/20: **100% / 100% / 100%**
- Jump rate: **0% on Route B and 0% on Route C**
- No-GRU B+C MLE: **2.317 m**
- GRU improvement over no-GRU: **11.75%**
- Final 5x5 E2E: **36.667 ms / 27.27 FPS on RTX 3090**
- 6x6 E2E: **39.554 ms / 25.28 FPS on RTX 3090**

Suggested GRU ablation sentence:

> Removing the temporal GRU increases the pooled Route-B/Route-C MLE from 2.045 m to 2.317 m, corresponding to an 11.75% degradation. The full temporal model also substantially reduces speed, motion-prediction, visual-measurement and Kalman-stage errors, showing that the GRU provides useful temporal motion information rather than acting only as an auxiliary feature block.

Suggested protocol wording for external writing:

> The experiments evaluate controlled local refinement along the planned route, using predefined reference points to open the local satellite search region. Route A is used for training, while Routes B and C are reserved for evaluation.

Do not describe the 2.045 m result as a fully autonomous global-search or no-prior localization result.
