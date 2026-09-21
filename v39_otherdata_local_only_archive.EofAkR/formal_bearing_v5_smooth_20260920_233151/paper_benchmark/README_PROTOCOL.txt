Bearing-UAV paper-aligned outputs

Directly computed from raw predictions:
  MLE, MedLE, P90/P95/P99, LSR@5/10/15/20,
  MHE, MedHE, HSR@15,
  latency/FPS, JumpRate, MaxFinalStep.

Important fairness constraints:
  1. Bearing-UAV Recall@1 is a four-adjacent-RST retrieval decision.
     Forward-18 navigation top-1 is NOT substituted; ours stays N/A.
  2. SR@20/SPL/NE in this package are route-replay diagnostics.
     They are NOT claimed as Bearing-Naver closed-loop navigation results.
  3. Use table_literature_comparison.csv for the literature table, preserving N/A fields.
