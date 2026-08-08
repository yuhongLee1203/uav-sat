# Result bundle

- `01`--`04`: archived P320/S32 Fixed HardMS control, sensitivity, transition, and latency results.
- `10`--`13`: new temporal RTL-CRF outputs. The main run is `strict_train_A_test_BC_no_position_scale`.
- `20`: common-frame jump comparison. Every row uses the same 3,526 evaluation frames. External rows are local-36 adaptations, not official author benchmark reproductions.
- `21`: complete 3,534-frame external localization table. Bearing-UAV is archived coordinate regression and has no author-reported JumpRate.

**JumpRate definition.** For each route independently, a predicted step is a jump when its length exceeds the route's 99th-percentile GT step length plus 3 m. No trajectory links cross route boundaries.

**Protocol warning.** All local-grid runs use a GT-jitter controlled local candidate prior. Results measure controlled local localization, not unconstrained global localization. The temporal rows additionally use contiguous frames, therefore they must not be ranked as a fair replacement for independent-frame external retrieval methods.
