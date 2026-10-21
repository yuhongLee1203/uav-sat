# V36 GvsK

This directory contains two separate V36 paths.

- `original_forNX/` is an unmodified source copy for code inspection.
- `meanshift_gru/` keeps the same V36 data flow, visual retrieval, causal forward
  3x6 search, Soft Mean-Shift, three-frame GRU and training losses.  Its only
  estimator change is `UAVSAT_EXPERIMENT_KALMAN=none`: final localization is
  the GRU-corrected SoftMS visual measurement, rather than a Kalman posterior.
  The GvsK final output is also not subsequently clipped by the controlled
  GT-progress display cap.

`run_original_fornx_benchmark.sh` uses the original packaged checkpoints and
Route B+C data for reproducibility evaluation.  The package contains no Route A,
so it cannot retrain the original model.

`run_meanshift_gru_train_eval.sh` trains on Route A and evaluates Route B+C.
Its default `v36_training_data/` links to the original V36 route assignment:
Route A = `new_data_2/model_dataset_new_1_flight`, Route B =
`new_data_2/model_dataset_new_2_flight`, Route C = `new_data/model_dataset_flight`.
Set `UAVSAT_DATA_ROOT` only when replacing this complete A/B/C dataset root.
