| Item | Value |
| --- | --- |
| GPU | NVIDIA GPU 0/5/6 across parallel city evaluation |
| End-to-end mean latency | 39.72 ms/frame |
| Throughput equivalent | 25.18 FPS |
| Timing samples | 1242 |
| Final 6×6 Mean Shift latency | 7.24 ms/frame |

End-to-end timing starts at a prepared UAV tensor and excludes disk I/O, preprocessing, model loading, and satellite gallery construction. GPU models and concurrent load are in the replay environment; this is a measured replay, not a deployment benchmark.
