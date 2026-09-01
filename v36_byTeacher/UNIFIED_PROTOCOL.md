# Unified v36_byTeacher Experiment Protocol

This directory is intentionally reduced to the code needed by the final controlled experiment family.

## Formal search prior
- Route A: training.
- Routes B/C: evaluation only.
- Current-frame reference position is perturbed by **exactly 8.0 m**.
- The perturbation direction changes deterministically by frame and is identical across methods.
- Formal search: full centered **6x6 = 36** satellite candidates.
- Decoder: SoftMS.
- MeanShift bandwidth: **8 m**.
- Score temperature: **0.30**.
- Main model seed: **2026**.
- No 0 m oracle is included in formal tables.

Before training/evaluation, the code computes candidate capture from the actual satellite-gallery geometry. The formal 6x6/8m experiment is rejected if capture is below 95%.

## Eight main architectures
Same-frame:
1. MKG: M -> K -> G
2. MGK: M -> G -> K
3. GMK: M -> G -> M -> K
4. GKM: M -> G -> K -> M
5. KGM: M -> K -> G -> M
6. KMG: M -> K -> M -> G

One-frame delayed:
7. delayKG: current M -> K -> G(next provisional) -> next-frame M finalization
8. delayGK: current M -> G(next proposal) -> K -> next-frame M finalization

The delayed comparison uses independent one-step pairs. Pair state starts from the same fixed-8m perturbed coarse current/previous positions. The next-frame final M is centered at the architecture's provisional prediction, never at the next-frame reference position.

## Formal ablations
All ordinary ablations keep the same fixed 8m prior:
- search window: 4x4, 5x5, 6x6, 7x7, 8x8
- components: M, M+K, M+G, M+K+G
- GRU input branches
- MeanShift bandwidth
- score temperature
- Weighted Centroid vs SoftMS accuracy
- random-seed stability

Robustness is the only table that changes prior error. Each tested error magnitude is capture-checked first; a level below the 95% capture threshold is marked invalid and skipped.

Decoder timing is aggregation-only:
- Weighted: existing 36 patch coordinates + weights -> XY
- SoftMS: existing converged active modes + mode weights -> XY
No backbone, similarity, softmax, MeanShift convergence, Kalman, GRU, or FPS is included in this timing.
