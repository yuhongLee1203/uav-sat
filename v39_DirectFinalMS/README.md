# v39_DirectFinalMS

這個版本把 v38 的第二次 Kalman Update 完全移除，最終流程簡化為：

`MS1 -> GRU -> Kalman Predict/Update -> MS2 -> Final`

## 為什麼不能只把「純 MeanShift」直接接在 Kalman 後面？

因為農田場景有很多外觀很像的衛星區塊。如果 MS2 只看 UAV 與 SAT 的影像相似度，MeanShift 可能被附近另一群相似農田吸走，反而把原本穩定的 Kalman 結果拉壞。

所以這個版本不是增加第二個 Kalman，而是直接修改 MS2 的分數。

MS2 對完整 6x6 = 36 個候選同時考慮：

1. UAV 與 SAT 的影像相似度。
2. 候選點距離單一 Kalman 輸出有多遠。
3. 候選點距離目前 predefined reference point 有多遠。

因此 Kalman 後面真的直接接 MS2，但 MS2 不會只因為某個遠方農田「長得很像」就整個跳過去。

## 直接流程

1. 前方 3x6 候選執行 MS1。
2. GRU 估計短期移動狀態。
3. 單一 Kalman 完成 Predict / Update，得到穩定位置。
4. 以 Kalman 位置對應的永久 SAT lattice 點為中心，開完整 6x6。
5. MS2 將 visual likelihood、Kalman spatial prior、reference spatial prior 合併。
6. Soft MeanShift #2 輸出的 XY 直接作為 Final Position。

沒有第二次 Kalman Update。

## 預設 MS2 約束

- Kalman prior sigma: 4.0 m
- Reference prior sigma: 4.0 m
- Kalman prior weight: 1.5
- Reference prior weight: 2.5
- MeanShift bandwidth: 5.0 m

這些參數可以透過環境變數調整。

## 執行

```bash
cd /yh/study/uav-sat

CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

結果：

`v39_DirectFinalMS/output/robust_tracker_summary.json`

## 注意

這仍然是 controlled predefined-reference-point protocol。MS2 的空間分數會使用目前 frame 的 predefined reference point，因此論文中必須透明描述為 reference-guided local refinement，不能描述成完全不知道目前位置的 autonomous localization。
