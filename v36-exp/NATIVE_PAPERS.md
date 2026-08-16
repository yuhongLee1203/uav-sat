# Native-paper execution

`run_native_papers_gpu56.sh` prepares only the class-paired image structure
required by the official repositories. It does not construct a per-frame local
candidate list and does not call `run_paper_baseline.py`.

Protocol:

- Route A is the training UAV/satellite pair set.
- Route B and Route C are test UAV queries.
- One fixed satellite gallery, shared by every test query, contains all unique
  satellite locations represented in Route A/B/C.
- No GT+jitter, forward 3x6, waypoint/reference point, GRU, polynomial or Kalman
  is passed to a paper model.
- DenseUAV uses its official `train.py`, ResNet-50 backbone and FSRA-CNN head.
- Sample4Geo uses its official `train_university.py`, shared ConvNeXt encoder and
  InfoNCE training/evaluation.
- Game4Loc uses its official `train_university.py`, descriptor model and InfoNCE
  training/evaluation.

InfoGeo is not launched because the cloned official repository is missing the
`cvgl_base.dataset.university` module imported by its native dense training
entrypoint. Bearing-UAV is not launched because its required cross-view model
weights and neighbor-map/heading annotations are absent. Neither is replaced by
a generic backbone adapter.

Run from the project root:

```bash
bash v36-exp/run_native_papers_gpu56.sh
```

Optional shorter smoke run:

```bash
DENSEUAV_EPOCHS=1 GAME4LOC_EPOCHS=1 SAMPLE4GEO_EPOCHS=1 \
NATIVE_PAPER_BATCH=8 \
bash v36-exp/run_native_papers_gpu56.sh
```

The default epoch counts follow the minimum/default native entries currently in
the repositories: DenseUAV 120, Game4Loc 5, and Sample4Geo 1.
