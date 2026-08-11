# Stable Visual-Inertial RNN v14

## Design decision

This version intentionally returns to FULL 6x6 local visual search.

The previous forward-half models could permanently lose the correct location
when heading or predicted position was wrong. v14 never deletes the rear/side
candidates. The previous RNN motion is supplied as a learned soft directional
prior to the 36-candidate scorer.

## Architecture

```text
known start W0
   |
previous FINAL visual/Kalman XY
   |
FULL 6x6 SAT search = 36 patches
   |
current UAV embedding + 36 SAT embeddings/similarities
+ previous image-derived RNN hidden/motion
   |
plain nn.RNNCell
   |
   +-- refined 36 visual candidate scores
   +-- next-frame image-derived motion state (0..10 m/frame)
   +-- stop probability
   +-- small sub-anchor XY residual
   +-- measurement uncertainty
   |
hard current-image SAT anchor + bounded residual
   |
current visual XY
   |
position-only Kalman [x,y], no vx/vy
   |
FINAL XY
```

The RNN motion cannot directly move current XY. It affects temporal state and
soft candidate ranking only. Current localization therefore remains anchored by
the current UAV/SAT visual match.

## Training

Route A only.

GT is used as supervised labels and, during the early scheduled-sampling
curriculum, as a training-only candidate search center. GT is never passed into
`StableVisualInertialRNN.forward_step()`.

Teacher-center schedule:

- epochs 1-5: 1.00
- epochs 6-24: linearly decays
- epoch 25 onward: 0.00, fully closed-loop

Validation is always fully closed-loop and best-validation checkpoint selection
is used.

There is no large squared ahead-loss.

## Speed / stopping

The RNN motion vector is bounded by vector magnitude:

`0 <= |delta_xy| <= 10 m/frame`

Zero is explicitly supported through the stop head. Route-A motion targets are
not clipped at 3 m/frame.

The final localization step is also radially capped at 10 m/frame as requested.

## Heading

Heading is not a separate arbitrary classifier. For a non-stop frame it is

`atan2(rnn_motion_dy, rnn_motion_dx)`

in ENU coordinates.

At a stop, heading is marked invalid instead of inventing an angle.

The renderer creates an arrow endpoint in world ENU coordinates and sends it
through the satellite-map geotransform. This avoids the old 90-degree
screen-coordinate drawing error.

## Video

Only two map colors are used:

- green = GT
- magenta = FINAL prediction

No waypoint line, no waypoint markers, and no SAT-candidate triangle/diamond.
