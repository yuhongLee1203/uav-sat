CORRECTED PROTOCOL
==================
1. The 8 test routes are exactly the official Bearing-UAV navigation waypoint routes.
   Official lengths: 524..1119 m; navigation step=25 m; waypoint arrival threshold=20 m.
2. University-1652 / SUES-200 / DenseUAV / GTA-UAV route evaluation uses ONLY
   the four adjacent p1/p2/p3/p4 RSTs for each UAV frame. Prediction is the
   retrieved tile centre. They receive no waypoint, route, previous-frame,
   temporal, Kalman, or v39 local prior.
3. Bearing-UAV official uses the authors' released VGG-16 checkpoint and native
   four-RST pose regression.
4. Ours retains its own temporal controlled-local-prior protocol.
5. Published Table values are kept separately. Route-selected results are not
   expected to exactly equal the full static localization benchmark.
6. bearinguav_official_paper_protocol.json verifies the official checkpoint on
   the released code's full-metadata 85/5/10 seed-42 test protocol.

KEY FILES
=========
same_routes_pooled.csv
same_routes_route_level.csv
bearinguav_published_uav_reference.csv
bearinguav_official_paper_protocol.json
route_protocol_audit.json
figures/
