# V39 Experimental Results and Ablation Study

Final validated architecture:

**Weighted Centroid -> context-aware 3-frame GRU residual/velocity refinement -> fixed-R external Kalman -> one final 5x5 MeanShift -> final position**

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
- MeanShift bandwidth: 7 m
- Runtime GPU: NVIDIA GeForce RTX 3090
- PyTorch: 1.13.1+cu117
- CUDA runtime: 11.7

> **Paper-facing protocol note.** These results evaluate **controlled local refinement along the planned route**. Predefined reference points are used to open the local satellite search region. Route A is used for training, while Routes B and C are reserved for evaluation. The reported local-refinement numbers must not be described as fully autonomous unrestricted global-search or no-prior closed-loop localization results.

---

# 1. Evaluation Metrics

| Metric | Definition |
|---|---|
| MLE | Mean localization error in meters. |
| MedLE | Median localization error in meters. |
| P90 / P95 / P99 | 90th / 95th / 99th percentile localization error. |
| LSR@5/10/15/20 | Percentage of frames whose localization error is at most 5/10/15/20 m. |
| Jump Rate | Percentage of frames whose final step exceeds the corresponding reference step by more than the 5 m tolerance. |
| Speed MAE | Mean absolute forward-speed error in meters per frame. |
| Progress MAE | Mean absolute route-progress error in meters. |
| Heading MAE | Mean absolute heading error in degrees. |
| Motion Prediction MAE | Error of the temporal motion prior before the visual update. |
| Visual Measurement MAE | Error of the visual measurement before external Kalman fusion. |
| Kalman MAE | Error after external Kalman fusion and before the final MeanShift. |
| E2E latency | Prepared UAV tensor -> backbone -> visual retrieval/GRU -> external Kalman -> final MeanShift -> metric XY. |

The pooled B+C distribution statistics are computed from the concatenated per-frame errors, not by averaging Route-B and Route-C percentiles. The final pooled evaluation contains **3,534 frames**.

---

# 2. Final Localization Results

## 2.1 Primary GRU Ablation

**Table 1. Final no-GRU vs. full context-GRU localization performance.**

| Setting | B MLE (m) | C MLE (m) | B+C MLE (m) | B+C MedLE (m) | B+C P90 (m) | B+C P95 (m) | B+C P99 (m) | LSR@5 | LSR@10 | LSR@15 | LSR@20 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No GRU | 2.384 | 2.195 | 2.317 | 2.209 | 4.231 | 4.842 | 6.120 | 95.61% | 100.00% | 100.00% | 100.00% | 3534 |
| **Full context-GRU** | **2.191** | **1.779** | **2.045** | **1.717** | **4.110** | **4.823** | **5.911** | **95.73%** | **100.00%** | **100.00%** | **100.00%** | 3534 |

Main result:

- Full GRU improves pooled B+C MLE from **2.317 m to 2.045 m**.
- Improvement normalized by the no-GRU baseline: **11.75%**.
- Conversely, removing the GRU increases MLE from 2.045 m to 2.317 m, which is a **13.31% degradation relative to the full model**.
- Route B MLE improvement: **8.09%**.
- Route C MLE improvement: **18.93%**.
- Pooled MedLE improvement: **22.30%**.
- Pooled P90 improvement: **2.85%**.
- Pooled P99 improvement: **3.41%**.
- Pooled LSR@5 increases by approximately **0.11 percentage points**.
- Final audit status: **PASS**.

## 2.2 Final Full-Model Accuracy by Route

**Table 2. Selected final-model route-wise and pooled localization distribution.**

| Evaluation set | MLE | MedLE | P90 | P95 | P99 | LSR@5 | LSR@10 | LSR@15 | LSR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Route B | **2.191** | 1.981 | 4.152 | 4.702 | 5.712 | 96.62% | 100.00% | 100.00% | 100.00% |
| Route C | **1.779** | 1.341 | 3.944 | 5.219 | 6.060 | 94.12% | 100.00% | 100.00% | 100.00% |
| **B+C pooled** | **2.045** | **1.717** | **4.110** | **4.823** | **5.911** | **95.73%** | **100.00%** | **100.00%** | **100.00%** |

---

# 3. Route B — Complete Final-Model Metrics

**Table 3. Route-B no-GRU and full-GRU diagnostics.**

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
| Final predicted route leg | 14 | 14 |
| Final reference route leg | 14 | 14 |

GRU contribution on Route B:

- Speed MAE reduction: **81.96%**.
- Motion-prediction MAE reduction: **15.13%**.
- Visual-measurement MAE reduction: **15.20%**.
- Kalman MAE reduction: **12.74%**.
- Progress MAE reduction: **59.57%**.
- Heading MAE reduction: **14.73%**.

Route B has a slightly worse P90/P95 and LSR@5 after adding the GRU, but the mean, median, P99, speed, motion, visual-measurement, Kalman-stage and heading metrics improve. Therefore, the GRU conclusion should be based on the complete metric set rather than one Route-B tail statistic alone.

---

# 4. Route C — Complete Final-Model Metrics

**Table 4. Route-C no-GRU and full-GRU diagnostics.**

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
| Final predicted route leg | 10 | 10 |
| Final reference route leg | 10 | 10 |

GRU contribution on Route C:

- Speed MAE reduction: **46.61%**.
- Motion-prediction MAE reduction: **24.76%**.
- Visual-measurement MAE reduction: **11.78%**.
- Kalman MAE reduction: **22.48%**.
- Final MLE reduction: **18.93%**.
- Route-progress MAE increases from 7.354 m to 10.568 m even though final XY localization improves strongly; therefore route-progress error and final position error should be discussed separately.

---

# 5. Where the GRU Helps

## 5.1 Stage-Wise Route Comparison

**Table 5. Stage-wise error propagation.**

| Route | Setting | Motion prediction MAE | Visual measurement MAE | Kalman MAE | Final MLE |
|---|---|---:|---:|---:|---:|
| B | No GRU | 5.585 | 9.654 | 5.511 | 2.384 |
| B | **Full GRU** | **4.740** | **8.187** | **4.808** | **2.191** |
| C | No GRU | 5.348 | 8.412 | 5.267 | 2.195 |
| C | **Full GRU** | **4.024** | **7.421** | **4.083** | **1.779** |

The latest GRU improves the learned temporal velocity, motion prior, visual measurement and external Kalman estimate. The gain survives the final MeanShift and produces a lower final localization error.

## 5.2 Frame-Weighted B+C Diagnostics

**Table 6. Frame-weighted B+C intermediate metrics.**

| Diagnostic | No GRU | Full GRU | Relative improvement |
|---|---:|---:|---:|
| Speed MAE (m/frame) | 2.788 | **0.775** | **72.21%** |
| Progress MAE (m) | 11.892 | **7.512** | **36.83%** |
| Motion prediction MAE (m) | 5.501 | **4.485** | **18.46%** |
| Visual measurement MAE (m) | 9.212 | **7.914** | **14.09%** |
| Kalman MAE (m) | 5.424 | **4.550** | **16.11%** |
| Heading MAE (deg) | 21.863 | **20.475** | **6.35%** |
| Final MS mean shift from Kalman (m) | 3.438 | **2.825** | **17.85%** |

---

# 6. Stability and Acquisition Diagnostics

**Table 7. Selected full-model stability and acquisition metrics.**

| Metric | Route B | Route C |
|---|---:|---:|
| Selected-candidate capture | 89.63% | 91.73% |
| Bank-candidate capture | 89.63% | 91.73% |
| Acquisition accuracy | 100.00% | 100.00% |
| Mean acquisition confidence | 0.473 | 0.441 |
| Longest selected-candidate miss streak | 41 frames | 18 frames |
| Mean final step | 3.173 m | 2.186 m |
| Maximum final step | 8.830 m | 8.844 m |
| Mean excess over reference step | 0.333 m | 0.242 m |
| Kalman step-limited rate | 73.20% | 66.53% |
| Jump rate | **0.00%** | **0.00%** |
| Heading MAE | 12.293 deg | 35.280 deg |
| Final predicted route leg | 14 | 10 |
| Final reference route leg | 14 | 10 |

The final predicted route leg matches the reference route leg for both routes, and the jump rate is zero on both Route B and Route C.

---

# 7. Current Final Runtime

The latest runtime pair uses the same RTX 3090, the same velocity-fusion code revision, and GPU-resident satellite-gallery indexing.

Timing definition:

`prepared UAV tensor -> backbone -> visual retrieval/GRU -> external RouteKalman -> final MeanShift -> final metric XY`

Excluded from timer:

- image disk I/O
- image preprocessing
- model/checkpoint loading
- satellite-gallery construction

Warm-up: 30 frames.

## 7.1 Current 5x5 Runtime

**Table 8. Current 5x5 E2E runtime.**

| Route | Mean latency | Median latency | P90 latency | P95 latency | FPS |
|---|---:|---:|---:|---:|---:|
| B | 36.797 ms | 35.993 ms | 44.789 ms | 47.840 ms | 27.18 |
| C | 36.430 ms | 35.903 ms | 43.792 ms | 47.421 ms | 27.45 |
| **B+C pooled** | **36.667 ms** | - | - | - | **27.27** |

## 7.2 Current 6x6 Runtime

**Table 9. Current 6x6 E2E runtime.**

| Route | Mean latency | Median latency | P90 latency | P95 latency | FPS |
|---|---:|---:|---:|---:|---:|
| B | 39.363 ms | 38.885 ms | 46.947 ms | 50.914 ms | 25.40 |
| C | 39.903 ms | 39.070 ms | 47.480 ms | 50.683 ms | 25.06 |
| **B+C pooled** | **39.554 ms** | - | - | - | **25.28** |

## 7.3 Current 5x5 vs. 6x6 Accuracy/Efficiency

**Table 10. Latest same-revision 5x5 vs. 6x6 comparison.**

| Final MS window | Candidates | B MLE | C MLE | B+C MLE | Pooled E2E | Pooled FPS |
|---|---:|---:|---:|---:|---:|---:|
| **5x5** | 25 | **2.1912** | **1.7794** | **2.0446** | **36.667 ms** | **27.27** |
| 6x6 | 36 | 2.1913 | 1.7797 | 2.0448 | 39.554 ms | 25.28 |

The accuracy difference is negligible. The 5x5 setting reduces pooled latency by approximately **7.30%** and increases FPS by approximately **7.87%**, so **5x5 is the current selected operating point**.

---

# 8. Final Audit Checklist

**Table 11. Final experiment audit.**

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
| 5x5 vs. 6x6 same-GPU runtime sanity | PASS |
| Full model beats no-GRU | **PASS** |

Final audit status: **PASS**.

---

# 9. Paper-Ready Main Results Discussion

The selected context-aware temporal model improves both localization accuracy and temporal estimation quality. On the pooled Route B+C evaluation, the complete method achieves an MLE of **2.045 m**, MedLE of **1.717 m**, P90 of **4.110 m**, and LSR@5 of **95.73%**. The no-GRU baseline produces a pooled MLE of **2.317 m**, so the full model provides an **11.75% improvement when normalized by the no-GRU baseline**. Equivalently, removing the GRU from the full architecture causes a **13.31% MLE degradation relative to the full model**.

The GRU contribution is visible before the final spatial decoder. Frame-weighted speed MAE decreases from approximately **2.788 to 0.775 m/frame**, motion-prediction MAE decreases from **5.501 to 4.485 m**, visual-measurement MAE decreases from **9.212 to 7.914 m**, and Kalman-stage MAE decreases from **5.424 to 4.550 m**. These results show that the GRU contributes useful temporal motion information instead of acting only as an auxiliary feature block.

The final 5x5 MeanShift setting preserves the accuracy of 6x6 while reducing same-GPU latency from **39.554 to 36.667 ms**. The selected model therefore runs at approximately **27.27 FPS** on an RTX 3090. Both Route B and Route C maintain a **0% jump rate**, showing that the improved localization accuracy does not introduce temporal instability.

---

# 10. Supporting / Appendix Experiments

The following tables preserve earlier valid design-stage experiments. They were generated before the current context-GRU + learned-velocity Kalman fusion redesign. They are useful for explaining architectural choices, but should **not** be numerically merged with the latest primary Table 1 as if every row were generated by an identical final checkpoint.

## 10.1 Earlier Progressive Architecture Ablation

Earlier chain:

`Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> final MeanShift`

**Table A1. Earlier progressive architecture ablation.**

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| Weighted Centroid | 6.070 | 5.122 | 5.732 | 45.36% | 4.527% / 3.580% |
| + GRU | 6.352 | 5.543 | 6.064 | 42.02% | 6.066% / 4.375% |
| + Kalman | 4.805 | 4.124 | 4.563 | 60.27% | 0.044% / 0.159% |
| + final MS | **2.190** | **1.789** | **2.047** | **95.53%** | **0.000% / 0.000%** |

This older table exposed the original GRU/Kalman integration problem. The old GRU-only stage did not improve positional accuracy. The final context-aware velocity-fusion redesign resolves this issue, as shown by the current Table 1.

## 10.2 Earlier Temporal Frame-Count Ablation

Each temporal configuration below was trained separately on Route A.

**Table A2. Temporal input-frame ablation.**

| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |
|---:|---|---:|---:|---:|---:|
| 1 | current frame only | 2.212 | 1.822 | 2.073 | 4.133 |
| 2 | current + previous; first difference | 2.227 | 1.822 | 2.083 | 4.143 |
| **3** | current + previous two; first + second difference | **2.190** | **1.789** | **2.047** | **4.077** |

Historical conclusion: the 3-frame model gives the best pooled MLE and P90 among the 1/2/3-frame configurations, supporting the final use of three consecutive UAV frames.

## 10.3 Earlier Motion-Prediction Design

All rows use the earlier 3-frame GRU.

**Table A3. Motion prediction model.**

| Motion model | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|---:|
| No learned motion | 2.476 | 2.264 | 2.401 | 4.415 | 94.40% | 0 / 0 |
| **Constant Velocity** | **2.162** | **1.747** | **2.014** | 3.979 | 95.67% | 0 / 0 |
| Velocity + Acceleration | 2.163 | 1.748 | 2.015 | **3.956** | **95.81%** | 0 / 0 |

Constant Velocity gives the best pooled MLE. Velocity + Acceleration is nearly identical, so the simpler constant-velocity interpretation is sufficient.

## 10.4 Earlier Kalman Design

**Table A4. Kalman measurement design.**

| Kalman setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|---:|
| No Kalman | 2.787 | 2.339 | 2.628 | 4.666 | 92.28% | 1.231% / 0.796% |
| Learned measurement variance | 2.218 | 1.752 | 2.052 | 4.116 | 95.16% | 0 / 0 |
| **Fixed measurement variance** | **2.162** | **1.747** | **2.014** | **3.979** | **95.67%** | **0 / 0** |

The fixed-R Kalman gives the best pooled MLE in this design sweep and removes the jump behavior observed without Kalman filtering.

## 10.5 Earlier Final MeanShift Grid Sensitivity — Pure Decoder Timing

Pure-MS latency starts only after the candidate centers and regularized logits are already available.

**Table A5. Pure MeanShift decoder latency.**

| Grid | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.216 | **4.014 ms** | **249.1** |
| **5x5** | 25 | 2.047 | **5.044 ms** | **198.3** |
| 6x6 | 36 | 2.047 | 7.024 ms | 142.4 |
| 7x7 | 49 | **2.046** | 10.125 ms | 98.8 |
| 8x8 | 64 | **2.046** | 10.368 ms | 96.4 |

Selection rule: among windows whose MLE is within **0.5% of the best value**, choose the lower-latency configuration. This supports the current 5x5 choice.

## 10.6 Earlier Broader Final-Stage Window Timing

A separate clean sweep used a broader timer that included candidate indexing/scoring plus the final MeanShift. Therefore, its absolute latency must not be compared directly with the pure decoder values in Table A5.

**Table A6. Earlier final-stage window sweep.**

| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 | Final-stage latency | Final-stage FPS |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.184 | 4.601 | 92.08% | 7.122 ms | 140.4 |
| 5x5 | 25 | 2.014 | 3.981 | 95.70% | 8.005 ms | 124.9 |
| 6x6 | 36 | 2.014 | 3.979 | 95.67% | 10.041 ms | 99.6 |
| 7x7 | 49 | **2.012** | 3.979 | **95.81%** | 11.019 ms | 90.8 |
| 8x8 | 64 | 2.013 | **3.978** | 95.78% | 12.951 ms | 77.2 |

## 10.7 Earlier MeanShift Bandwidth Sensitivity

The final-MS window is fixed at 6x6 for this earlier sensitivity sweep.

**Table A7. MeanShift bandwidth sensitivity from 1 to 14 m.**

| Bandwidth | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 m | 24.284 | 10.259 | 19.291 | 51.283 | 36.53% |
| 2 m | 2.961 | 2.233 | 2.701 | 5.077 | 89.59% |
| 3 m | 2.338 | 1.938 | 2.195 | 4.306 | 94.62% |
| 4 m | 2.239 | 1.831 | 2.094 | 4.123 | 95.30% |
| 5 m | 2.194 | 1.782 | 2.047 | 4.058 | 95.56% |
| 6 m | 2.173 | 1.759 | 2.026 | 4.008 | 95.67% |
| **7 m (selected operating point)** | **2.162** | **1.747** | **2.014** | 3.979 | 95.67% |
| 8 m | 2.155 | 1.741 | 2.008 | 3.971 | 95.64% |
| 9 m | 2.151 | 1.737 | 2.004 | 3.963 | 95.59% |
| 10 m | 2.149 | 1.734 | 2.001 | 3.956 | 95.56% |
| 11 m | 2.147 | 1.732 | 1.999 | 3.952 | 95.59% |
| 12 m | 2.145 | 1.730 | 1.998 | 3.951 | 95.61% |
| 13 m | 2.144 | 1.729 | 1.997 | 3.952 | 95.64% |
| 14 m | **2.143** | **1.729** | **1.996** | **3.950** | 95.67% |

The 1 m bandwidth is clearly too narrow. Accuracy stabilizes from approximately 5-7 m onward, and improvements above 7 m are marginal. The selected 7 m bandwidth remains a reasonable non-boundary operating point rather than choosing the edge of the sweep solely for a very small numerical gain.

## 10.8 Historical Runtime Records

Different earlier implementation revisions produced different absolute E2E timings. They are retained only for traceability.

**Table A8. Historical runtime records.**

| Revision / definition | Pooled B+C latency | Approx. FPS | Interpretation |
|---|---:|---:|---|
| Earlier clean optimized sweep | 34.013 ms | 29.4 | Older model/runtime revision. |
| Earlier final-ablation runner | 34.045 ms | 29.4 | Older pre-context-GRU revision. |
| Temporary GPU->CPU indexing bottleneck | 110.141 ms | 9.1 | Invalid as representative compute throughput; full-gallery copy caused synchronization. |
| **Current authoritative 5x5 velocity-fusion run** | **36.667 ms** | **27.27** | Use this value in the current paper. |

---

# 11. Recommended Main-Paper vs. Appendix Placement

**Table 12. Recommended paper organization.**

| Paper location | Table(s) to use | Purpose |
|---|---|---|
| Main Results | Table 2 | Definitive final-model route-wise accuracy. |
| Main GRU Ablation | Table 1 | Current no-GRU vs. full context-GRU result. |
| Temporal Analysis | Tables 5-6 | Shows where the GRU contributes before final decoding. |
| Stability Analysis | Table 7 | Jump, capture, progress and tracking stability. |
| Efficiency Analysis | Tables 8-10 | Current RTX 3090 E2E runtime and 5x5 selection. |
| Appendix / Supplement | Table A1 | Historical progressive architecture evolution. |
| Appendix / Supplement | Table A2 | 1/2/3-frame temporal study. |
| Appendix / Supplement | Table A3 | Motion-model sensitivity. |
| Appendix / Supplement | Table A4 | Kalman design sensitivity. |
| Appendix / Supplement | Tables A5-A6 | MeanShift grid and timing-definition studies. |
| Appendix / Supplement | Table A7 | Full bandwidth sensitivity. |
| Internal traceability only | Table A8 | Runtime evolution across code revisions. |

---

# 12. Final Selected Configuration

**Table 13. Final method configuration.**

| Component | Selected setting |
|---|---|
| UAV temporal input | **3 frames** |
| Visual front measurement | **Posterior Weighted Centroid** |
| Temporal model | **Context-aware GRU residual refinement + learned velocity** |
| Temporal UAV features | **Temporal mean + first difference + second difference** |
| Satellite context | **Posterior-weighted retained satellite embeddings** |
| External filter | **Fixed-R Kalman** |
| Full-model motion propagation | **GRU learned velocity** |
| No-GRU fallback | **Kalman internal constant-velocity state** |
| Original local geometry | **6x6 neighborhood** |
| Online causal support | **Forward 3x6 subset** |
| Final MeanShift window | **5x5 = 25 candidates** |
| MeanShift bandwidth | **7 m** |
| Online MeanShift count | **Exactly one final MeanShift** |
| Training route | **Route A only** |
| Evaluation routes | **Routes B and C** |
| Early-stopping patience | **5** |
| Runtime platform | **NVIDIA GeForce RTX 3090** |
| Current pooled MLE | **2.045 m** |
| Current pooled P90 | **4.110 m** |
| Current pooled LSR@5 | **95.73%** |
| Current pooled E2E | **36.667 ms / 27.27 FPS** |
| Jump rate | **0% on Route B and Route C** |

---

# 13. Recommended Headline Numbers for Paper / Presentation

Use the following as the current authoritative values:

- Final Route-B MLE: **2.191 m**
- Final Route-C MLE: **1.779 m**
- Final pooled B+C MLE: **2.045 m**
- Final pooled B+C MedLE: **1.717 m**
- Final pooled B+C P90: **4.110 m**
- Final pooled B+C P95: **4.823 m**
- Final pooled B+C P99: **5.911 m**
- Final pooled B+C LSR@5: **95.73%**
- Final pooled B+C LSR@10/15/20: **100% / 100% / 100%**
- Jump rate: **0% on Route B and 0% on Route C**
- No-GRU pooled B+C MLE: **2.317 m**
- Full-vs-no-GRU improvement: **11.75%**
- GRU-removal degradation relative to full model: **13.31%**
- Final 5x5 E2E: **36.667 ms / 27.27 FPS on RTX 3090**
- Current 6x6 E2E: **39.554 ms / 25.28 FPS on RTX 3090**

Suggested GRU ablation sentence:

> The full context-aware temporal model reduces the pooled Route-B/Route-C MLE from 2.317 m without the GRU to 2.045 m, corresponding to an 11.75% improvement relative to the no-GRU baseline. Removing the GRU from the complete model therefore produces a 13.31% degradation relative to the full system. The GRU also reduces speed, motion-prediction, visual-measurement and Kalman-stage errors, indicating that it contributes useful temporal motion information rather than acting only as an auxiliary feature block.

Suggested efficiency sentence:

> The 5x5 and 6x6 final MeanShift windows provide nearly identical localization accuracy, whereas the 5x5 configuration reduces pooled end-to-end latency from 39.554 ms to 36.667 ms and increases throughput from 25.28 FPS to 27.27 FPS on the same RTX 3090. Therefore, the 5x5 configuration is selected as the final accuracy-efficiency operating point.

Suggested external protocol wording:

> The experiments evaluate controlled local refinement along the planned route, using predefined reference points to open the local satellite search region. Route A is used for training, while Routes B and C are reserved for evaluation.

---

# 14. Reporting Rules

1. Use **2.045 m** as the current final pooled B+C MLE, not older 2.047 m or 2.014 m values from previous revisions.
2. Use **5x5** as the current final MeanShift window.
3. Use **36.667 ms / 27.27 FPS** as the current authoritative pooled E2E runtime.
4. Use **2.317 m without GRU vs. 2.045 m with full GRU** for the definitive current GRU ablation.
5. State **11.75%** when reporting improvement normalized by the no-GRU baseline.
6. State **13.31%** when reporting degradation caused by GRU removal relative to the full model.
7. Do not average Route-B and Route-C P90/P95/P99 values to obtain the pooled percentile. Use the concatenated per-frame error distribution.
8. Do not merge the older progressive/motion/Kalman/bandwidth results into the current final-model table as if every row used the same context-GRU checkpoint.
9. The external Kalman outputs position and velocity; heading is evaluated separately from the recurrent heading estimate.
10. The main results are controlled local-refinement results and must not be described as unrestricted autonomous global-search localization.

---

# 15. Result Provenance

Current main results:

`v39_DirectFinalMS/wc_gru_velocity_fusion_20260912_172752/`

Supporting progressive and temporal-frame ablations:

`v39_DirectFinalMS/wc_final_ablation_20260911_214923/`

Supporting motion/Kalman/window/bandwidth sweeps:

`v39_DirectFinalMS/wc_clean_experiments_20260911_190615/`

The latest experiment audit reports **PASS** for the full-vs-no-GRU accuracy condition, same-GPU runtime sanity, patience=5, fixed-R Kalman, exactly one final MeanShift, and GPU-resident gallery indexing.
