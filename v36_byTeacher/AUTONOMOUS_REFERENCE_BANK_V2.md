# Autonomous Reference-Bank V2

This experiment is the no-frame-aligned-reference version of the six M/G/K architecture ablation.

## What was wrong in the previous controlled six-architecture experiment

The previous experiment passed `cache.gt_xy[index]` into `forward_frame(...)` as `reference_prior_xy`. Therefore the current frame's corresponding reference coordinate directly opened the local satellite window. That experiment is useful as a controlled upper-bound/local-refinement comparison, but it is not the autonomous reference-selection test.

The previous implementation also used a posterior weighted centroid as the initial visual coordinate when M was not the first symbol. This conflicts with the clarified definition that a visual position is always obtained from satellite-patch scoring followed by MeanShift.

## Runtime reference selection in V2

The current frame index is never used to select a route reference point.

1. Load only the ordered predefined waypoint polyline for the route.
2. Densify the polyline into a static reference bank (default spacing 5 m).
3. The Kalman filter predicts the current location from its own previous posterior.
4. Search the ordered static bank from the last selected progress onward and choose the nearest reference point to the Kalman predicted location.
5. Use the selected reference point and its local route tangent to open the satellite search.
6. Current-frame labels are read only after the prediction exists: Route A for GRU supervision; Routes B/C for metrics.

This removes direct frame-to-reference correspondence from runtime localization.

## Visual position and uncertainty

Every architecture begins with the same base visual localization step:

`selected route reference -> 6x6 lattice geometry -> nearest center-adjacent forward 3x6 -> similarity -> MeanShift -> visual position + visual variance`

The base forward 3x6 is the nearest forward half of the 6x6 lattice. It includes the center plane and the next two rows/columns in the route direction (`0,+1,+2` or `0,-1,-2`), rather than selecting the farthest 18 candidates.

Visual variance is computed around the MeanShift output coordinate, not around a weighted-centroid coordinate.

## Meaning of M in the six architectures

All architectures have a base MeanShift because visual position must always be an MS result.

- If M is the first symbol, M is the base forward 3x6 MeanShift itself.
- If M appears after G and/or K, the incoming stage position becomes the center of a second full centered 6x6 search. All 36 satellite candidates are scored and MeanShift produces the correction position and new visual variance.

Therefore:

| Architecture | Corrected runtime flow |
|---|---|
| MKG | Base forward MS -> K -> G -> Final |
| MGK | Base forward MS -> G -> K -> Final |
| GMK | Base forward MS -> G -> centered 6x6 MS -> K -> Final |
| GKM | Base forward MS -> G -> K -> centered 6x6 MS -> Final |
| KGM | Base forward MS -> K -> G -> centered 6x6 MS -> Final |
| KMG | Base forward MS -> K -> centered 6x6 MS -> G -> Final |

GKM and KGM are the two permutations in which MeanShift is literally the final correction stage.

## Search direction rule

The base visual acquisition uses forward 3x6 because it is opening the local satellite evidence from a route reference and should remain causal in the planned route direction.

A later correction MeanShift uses a full centered 6x6 because the incoming G/K coordinate is already a current-frame estimate. At that stage the goal is local spatial correction, not forward acquisition. Applying forward-only support again would introduce a directional bias into the final correction.

## Output diagnostics

Each Route B/C CSV includes:

- Kalman predicted query used for autonomous reference selection
- selected static reference-bank index and XY
- selected-reference error (metrics only)
- base visual MeanShift XY/error
- GRU/Kalman/centered-MS intermediate coordinates when present
- final localization error

Each summary additionally reports `SearchReferenceMLE_m` and `BaseVisualMS_MLE_m`, allowing failures to be separated into reference-selection drift versus visual/localization-stage error.

## Run

```bash
git checkout six-mgk-autonomous-reference-bank-v2
git pull origin six-mgk-autonomous-reference-bank-v2
cd v36_byTeacher
EPOCHS=60 REF_SPACING_M=5.0 bash run_six_architectures_autoref_gpu056.sh
```
