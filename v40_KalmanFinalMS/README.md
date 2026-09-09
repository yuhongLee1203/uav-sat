# v40_KalmanFinalMS

## 最終流程

MS1 → GRU → Kalman Predict/Update → MS2 → Final Position

## MS2 做法

最後階段不使用第二次 Kalman Update，也不使用參考點作為 MS2 的位置限制。

1. 取得單一 Kalman Filter 更新後的位置。
2. 找到最接近 Kalman 位置的永久衛星格點。
3. 以該格點為中心建立完整 6×6（36 個）衛星候選。
4. 對每個候選計算 UAV–SAT 視覺相似度。
5. 另外依候選與 Kalman 輸出位置的距離給予空間權重，避免 MeanShift 被遠方重複農田紋理吸走。
6. 使用加權後的候選分數執行 Soft MeanShift。
7. MeanShift 輸出的 XY 就是 Final Position。

因此架構圖可以直接畫：

Kalman Filter → MS2 → Final Position

Kalman 的位置權重不是額外模組，只是 MS2 內部用來限制搜尋模式不要離 Kalman 結果太遠的分數項。

## 執行

```bash
CUDA_VISIBLE_DEVICES=5 UAVSAT_DEVICE=cuda:0 JITTER_M=8 bash v40_KalmanFinalMS/run.sh
```

輸出：`v40_KalmanFinalMS/output/robust_tracker_summary.json`

## 注意

這裡移除的是「最後 MS2 的參考點限制」。整體實驗仍沿用原本 controlled local-search protocol，因此前端局部搜尋的參考點設定並沒有被移除。
