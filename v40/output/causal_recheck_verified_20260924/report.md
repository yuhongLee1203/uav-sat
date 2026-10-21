# Causal City A held-out evaluation

Checkpoint: `/yh/study/uav-sat/v39_otherdata/output/abcd_shared_20260922_121010/citya/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt`

| Route | Model MLE (m) | Model LSR@15 | Route-duration-only MLE (m)* |
|---|---:|---:|---:|
| route_B | 11.995 | 75.16% | 12.069 |
| route_C | 30.749 | 25.11% | 22.162 |

*Diagnostic only: route duration is not an image input and can nearly solve these synthetic routes by itself.
