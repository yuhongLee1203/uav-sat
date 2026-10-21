# Invalid temporal protocol — do not report these results

This suite chained independent samples from Bearing-UAV's random official-96
split into synthetic routes.  Its test cadence (about 19–22 m/frame) does not
match training (about 9 m/frame), and test-frame cross-track P95 is about
30–44 m.  It therefore fails `audit_bearing_temporal_routes.py` and is not a
valid GRU/Kalman/MeanShift navigation ablation.

Bearing-UAV Table 6 is a static geo-localization ablation.  The v39 temporal
component ablation must instead use the aligned City A/B/C/D navigation
episodes and report pooled metrics with recurrent state reset at route/city
boundaries.
