# Bearing paper output status

## Already present and staged

- CityA/CityB/CityC/CityD: two final-result figures per city plus an all-city contact sheet are in `existing_verified/city_results/`.
- Frozen PASS CityA component/temporal/grid ablation is in `existing_verified/citya_pass_grid/`.
- Decoder aggregation-only GPU timing is in `existing_verified/decoder_microbenchmark/`.

Existing CityA final-MeanShift grid measurements (367 held-out frames):

| Grid | MLE m | P90 m | LSR@5 % | Final-MS latency ms |
|---|---:|---:|---:|---:|
| 4x4 | 4.4297 | 8.1207 | 64.033 | 10.3237 |
| 5x5 | 3.9364 | 6.8039 | 71.935 | 11.4788 |
| 6x6 | 3.9368 | 6.8042 | 71.935 | 9.7027 |
| 7x7 | 3.8986 | 6.7665 | 72.752 | 16.2849 |
| 8x8 | 3.8993 | 6.7674 | 72.752 | 22.0888 |

These are measured values, not edited to force a monotonic trend. The 6x6 row is a balance choice; 7x7/8x8 have slightly better localization and clearly higher MeanShift latency.

## Missing before this repair

- Exact frozen-PASS four-city rerun in one output tree.
- Full 6x6 (36) versus forward 3x6 (18) search table.
- Weighted versus MeanShift accuracy table on the same 36 candidates.
- Official Bearing-UAV Sat./UAV per-city Recall@1, LSR@15, HSR@15, MLE, MedLE, MHE and MedHE table.
- Official Bearing-Naver SR@20, SPL and NE table.

The complete runner now generates all of these under a timestamped `run_*` directory here. It uses GPUs 0, 5 and 6 and defaults to two CPU threads/workers per task.

```bash
cd /yh/study/uav-sat
CPU_THREADS_PER_TASK=2 BEARING_NUM_WORKERS=2 RUN_NAVIGATION=0 \
  bash v39_otherdata/run_complete_bearing_paper_suite.sh
```

Official Bearing-Naver route simulation is unrelated to the custom architecture and is disabled by default. Set `RUN_NAVIGATION=1` only when SR@20/SPL/NE from the separate official navigation benchmark are explicitly needed.
