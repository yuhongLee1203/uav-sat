# Supervised-training / GT-free-test sweep

All variants use one shared A+B temporal model. Test GT is metrics-only.

| Trial | City | Test route | MLE (m) | P90 (m) | LSR@15 |
|---|---|---|---:|---:|---:|
| causal | citya | test_01 | 17.906 | 30.466 | 40.37% |
| causal | citya | test_02 | 23.331 | 46.784 | 34.70% |
| causal | cityb | test_01 | 27.408 | 43.205 | 23.29% |
| causal | cityb | test_02 | 25.889 | 45.071 | 21.64% |
| mixed | citya | test_01 | 13.193 | 25.597 | 65.22% |
| mixed | citya | test_02 | 25.720 | 50.412 | 31.51% |
| mixed | cityb | test_01 | 14.687 | 24.814 | 58.22% |
| mixed | cityb | test_02 | 32.795 | 45.312 | 2.92% |
| teacher | citya | test_01 | 21.812 | 40.140 | 33.54% |
| teacher | citya | test_02 | 35.549 | 73.210 | 22.83% |
| teacher | cityb | test_01 | 15.446 | 29.005 | 54.11% |
| teacher | cityb | test_02 | 17.614 | 40.604 | 56.14% |
