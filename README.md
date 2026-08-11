# Reversible Topology Recovery LSTM v10

- LOCAL: conceptual 6x6, only forward 3x6.
- RECOVERY: current UAV image searches complete previous/current/next leg corridors; it is not centered on Pred and is not forward-limited.
- Route state is reversible PREVIOUS/CURRENT/NEXT; a wrong advance can rollback.
- No endpoint-distance gate, predicted progress, observed-motion vector, polynomial extrapolation, or ECC heading.
- Route-A temporal training is zero-teacher from epoch 1. B/C GT and waypoint frame_index are not inference inputs.
- Hard HOLD/LOCAL/RECOVERY visual XY is followed by a position-only Kalman [x,y].
