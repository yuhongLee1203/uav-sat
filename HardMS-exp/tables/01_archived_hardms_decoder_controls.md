# Archived Fixed HardMS Decoder Controls

| Split | Method | N | MLE | MedLE | P90 | P95 | P99 | CVaR90 | LSR@5 | LSR@10 | LSR@15 | LSR@20 | MaxLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B+C | Top-1 patch center | 3534 | 14.75 | 14.34 | 24.86 | 26.72 | 30.42 | 27.44 | 10.47 | 29.82 | 53.11 | 72.58 | 36.26 |
| B+C | Gaussian local average + Top-1 | 3534 | 15.40 | 14.90 | 25.55 | 27.75 | 31.61 | 28.23 | 9.05 | 27.50 | 50.40 | 69.75 | 36.26 |
| B+C | 3x3 local average + Top-1 | 3534 | 15.63 | 15.37 | 25.76 | 28.12 | 31.97 | 28.64 | 9.03 | 26.68 | 48.59 | 68.62 | 36.26 |
| B+C | Fixed HardMS (continuous mode; diagnostic) | 3534 | 11.29 | 11.22 | 17.86 | 20.01 | 23.49 | 20.49 | 11.52 | 42.08 | 76.49 | 94.96 | 29.92 |
| B+C | Fixed HardMS (snapped anchor) | 3534 | 11.46 | 11.20 | 18.23 | 20.62 | 23.60 | 20.89 | 11.04 | 41.14 | 73.46 | 93.80 | 29.91 |
| B+C | Oracle nearest anchor | 3534 | 1.78 | 1.83 | 2.62 | 2.83 | 3.15 | 2.87 | 100.00 | 100.00 | 100.00 | 100.00 | 3.32 |
| model_dataset_new_2_flight | Top-1 patch center | 2276 | 14.85 | 14.58 | 24.89 | 26.71 | 30.54 | 27.47 | 10.33 | 29.53 | 52.02 | 72.01 | 34.46 |
| model_dataset_new_2_flight | Gaussian local average + Top-1 | 2276 | 15.40 | 15.03 | 25.60 | 27.63 | 31.75 | 28.26 | 9.27 | 27.64 | 49.91 | 69.82 | 34.96 |
| model_dataset_new_2_flight | 3x3 local average + Top-1 | 2276 | 15.68 | 15.44 | 25.73 | 27.98 | 31.98 | 28.61 | 8.79 | 25.97 | 48.02 | 68.80 | 34.96 |
| model_dataset_new_2_flight | Fixed HardMS (continuous mode; diagnostic) | 2276 | 11.47 | 11.46 | 18.38 | 20.44 | 23.72 | 20.95 | 11.34 | 41.04 | 75.44 | 93.89 | 29.92 |
| model_dataset_new_2_flight | Fixed HardMS (snapped anchor) | 2276 | 11.63 | 11.41 | 18.62 | 20.86 | 24.34 | 21.31 | 10.98 | 40.29 | 72.14 | 93.06 | 29.91 |
| model_dataset_new_2_flight | Oracle nearest anchor | 2276 | 1.79 | 1.83 | 2.61 | 2.86 | 3.19 | 2.89 | 100.00 | 100.00 | 100.00 | 100.00 | 3.32 |
| model_dataset_flight | Top-1 patch center | 1258 | 14.57 | 13.95 | 24.78 | 26.74 | 30.20 | 27.36 | 10.73 | 30.37 | 55.09 | 73.61 | 36.26 |
| model_dataset_flight | Gaussian local average + Top-1 | 1258 | 15.40 | 14.69 | 25.44 | 27.79 | 31.32 | 28.18 | 8.66 | 27.27 | 51.27 | 69.63 | 36.26 |
| model_dataset_flight | 3x3 local average + Top-1 | 1258 | 15.54 | 15.15 | 25.77 | 28.14 | 31.95 | 28.69 | 9.46 | 27.98 | 49.60 | 68.28 | 36.26 |
| model_dataset_flight | Fixed HardMS (continuous mode; diagnostic) | 1258 | 10.97 | 10.88 | 17.36 | 19.13 | 22.20 | 19.53 | 11.84 | 43.96 | 78.38 | 96.90 | 25.64 |
| model_dataset_flight | Fixed HardMS (snapped anchor) | 1258 | 11.16 | 10.89 | 17.57 | 19.78 | 22.33 | 20.02 | 11.13 | 42.69 | 75.83 | 95.15 | 26.35 |
| model_dataset_flight | Oracle nearest anchor | 1258 | 1.76 | 1.81 | 2.64 | 2.77 | 3.08 | 2.84 | 100.00 | 100.00 | 100.00 | 100.00 | 3.19 |
