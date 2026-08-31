# Autonomous Reference-Bank V2 — Full Centered 6x6 MeanShift

This experiment removes frame-aligned reference lookup and now uses **full centered 6x6 satellite search for every MeanShift stage**.

## Runtime reference selection

The current frame index is never used to select a route reference point.

1. Load only the ordered predefined waypoint polyline for the route.
2. Densify the polyline into a static reference bank (default spacing 5 m).
3. The Kalman filter predicts the current location from its own previous posterior.
4. Search the ordered static bank from the last selected progress onward and choose the nearest reference point to the Kalman predicted location.
5. Use that selected reference point only as the center for the base visual search.
6. Current-frame labels are read only after prediction: Route A for GRU supervision; Routes B/C for metrics.

## Visual position and visual variance

Every architecture begins with the same base visual localization:

`selected reference center -> full centered 6x6 = 36 SAT patches -> similarity -> MeanShift -> visual position + visual variance`

There is **no forward 3x6 selection**. All 36 patches are scored and all 36 participate in MeanShift.

Visual position is the MeanShift output coordinate.

Visual variance is computed from the candidate posterior around the MeanShift coordinate:

- `variance_x = sum p_j (x_j - x_MS)^2`
- `variance_y = sum p_j (y_j - y_MS)^2`

The code checks that every MeanShift receives exactly 36 candidates and raises an error otherwise.

## Meaning of M in the six architectures

All architectures have the common base visual MeanShift because visual position is always an MS result.

- If M is the first symbol, M is the common base centered 6x6 MeanShift.
- If M appears later, the incoming G/K coordinate becomes the center of another full centered 6x6 search; all 36 patches are scored and MeanShift performs the correction.

| Architecture | Runtime flow |
|---|---|
| MKG | Base centered 6x6 MS -> K -> G -> Final |
| MGK | Base centered 6x6 MS -> G -> K -> Final |
| GMK | Base centered 6x6 MS -> G -> centered 6x6 MS -> K -> Final |
| GKM | Base centered 6x6 MS -> G -> K -> centered 6x6 MS -> Final |
| KGM | Base centered 6x6 MS -> K -> G -> centered 6x6 MS -> Final |
| KMG | Base centered 6x6 MS -> K -> centered 6x6 MS -> G -> Final |

GKM and KGM are the two orderings whose final output is directly produced by the final MeanShift correction.

## GRU definition

The six-architecture GRU remains a direct current-frame position refiner. It has no external heading head, speed head, acceleration head, motion head, or polynomial motion output.

GRU inputs are current incoming XY, visual variance, temporal UAV visual features, and the recurrent hidden state. GRU outputs a learned XY correction and corrected current-frame XY.

## Separate outputs

To avoid mixing these results with the previous forward-3x6 experiment:

- checkpoints: `six_autoref_center6x6_*`
- results: `output/<backbone>/six_architecture_autonomous_reference_center6x6/`
- logs: `logs/six_architecture_autonomous_reference_center6x6/`

## Run all six on GPU 0 / 5 / 6

```bash
git fetch origin
git checkout six-mgk-autonomous-reference-bank-v2
git pull origin six-mgk-autonomous-reference-bank-v2
cd /yh/study/uav-sat/v36_byTeacher
python3 -m py_compile six_architecture_model.py six_architecture_autoref_experiment.py
EPOCHS=60 REF_SPACING_M=5.0 bash run_six_architectures_autoref_gpu056.sh
```

GPU allocation:

- GPU 0: MKG -> MGK
- GPU 5: GMK -> GKM
- GPU 6: KGM -> KMG
