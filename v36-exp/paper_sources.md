# Official baseline sources and comparison protocol

The source trees under `others_paper/` retain their own Git history and official remotes:

- DenseUAV — <https://github.com/Dmmm1997/DenseUAV>
- Sample4Geo — <https://github.com/Skyy93/Sample4Geo>
- Game4Loc / GTA-UAV — <https://github.com/Yux1angJi/GTA-UAV>
- InfoGeo — <https://github.com/HRT00/Official_InfoGeo>
- Bearing-UAV — <https://github.com/liukejia121/bearinguav>

Their published checkpoints were trained on different datasets and cannot honestly be
reported as metre error on this project's Route B/C without a protocol-specific dataset
adapter and retraining.  In particular, DenseUAV, Sample4Geo, Game4Loc and InfoGeo are
global-retrieval methods; Bearing-UAV is a neighbouring-map position/heading regression
method.  They must not receive V36's GT+jitter reference point or Forward-3x6 candidate
window.

`run_paper_baseline.py` and its old outputs are retained only for auditability.  They used
the papers' backbone families inside V36's local-search protocol, so those approximately
12 m results are invalid for Table 8.  `run_all_v36_experiments.sh` no longer calls that
adapter.  Table 8 stays `PENDING` until each official repository has a native Route-A
training adapter and its own native Route-B/C inference protocol.
