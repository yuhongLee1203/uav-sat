# Current V39 architecture

Current requested inference architecture:

`Forward 3x6 SAT candidates -> Soft MeanShift -> 3-frame GRU -> Constant Velocity -> Fixed-R Kalman -> Final 6x6 Soft MeanShift (BW=7m) -> Final Position`

The previous Weighted-Centroid experiments and `run.sh` ablation history are kept for reproducibility. The current no-retraining evaluation entry point is:

```bash
bash v39_DirectFinalMS/run_softms_eval.sh
```

`run_softms_eval.sh` builds a fresh temporary runtime, applies `patch_direct_finalms.py`, then `patch_front_softms.py`, and reuses the existing checkpoints without training.
