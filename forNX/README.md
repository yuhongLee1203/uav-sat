# forNX — V39 Forward3x6 SoftMS

這個資料夾現在只保留 V39 的 NX inference / latency 測試版本。

目前 pipeline：

`MobileNetV3-Small -> Forward 3x6 SAT candidates -> Soft MeanShift -> 3-frame GRU -> Constant Velocity -> Fixed-R Kalman -> Final MeanShift (6x6, BW=7m) -> Final XY`

前端不再使用 Weighted Centroid。`run_nx.sh` 會先套用 DirectFinalMS cleanup patch，再套用 `patch_front_softms.py`，把前端 3x6 decoder 恢復成 Soft MeanShift。

NX 上只要執行：

```bash
cd forNX
bash run_nx.sh
```

程式會 warm-up 後跑完整 V39 inference，最後直接印出：

- Route B MLE + mean / median / P90 latency / FPS
- Route C MLE + mean / median / P90 latency / FPS
- `V39_FULL_PIPELINE_MEAN_MS`
- `V39_FULL_PIPELINE_FPS`

這是 eval-only benchmark，不會重新訓練 checkpoint。

需要一起帶到 NX 的本機資料（Git 不上傳大型檔）：

- `data/`
- `pretrained_cache/`
- `weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt`
- `weights/v39_directfinalms/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt`

計時範圍：prepared UAV tensor -> backbone -> Forward3x6 visual scoring / SoftMS -> GRU -> Kalman -> final MeanShift -> final XY。模型載入、磁碟 I/O、影像前處理與一次性的 satellite gallery/cache 建立不計入單幀 latency。
