# 1016 使用權重固定參考

本次確認並保存的版本：原始 v39_otherdata、GT-assisted、MLE 3.759 m、四城市8條route／1,482 observations。後續提到「1016原權重」以此檔及decoder_latency/weights_manifest.json為準。

來源suite：`/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368`。

每個城市的Full模型讀取該城市 `train_frames3/checkpoints/` 下的：

- `visual_retrieval_A_only.pt`
- `controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt`（正式推論權重，不是latest）
- Kalman設定：`train_frames3/kalman_calibration.json`。

四城市 temporal checkpoint 的SHA-256相同：`34b42624eb359fc2f952a655874ba3a1cb2b9f0bb0737106ec6c74b8589b55f6`。Visual checkpoint各城市檔案雜湊不同；保留各城市原檔，不互換。正式checkpoint的 `backbone_source` 為 `torchvision:mobilenet_v3_small`，實際backbone是MobileNet-V3-Small；`bearing_iclr_ablation.py` 的CLI預設也為此值，會覆蓋runtime檔案的MobileCLIP2-S2預設。先前只重算保存候選／分數的decoder，未載入backbone，所以該預設值註記不影響已保存的decoder數據。2026-10-17效率測試已依checkpoint核對並修正此註記。

完整絕對路徑、檔案大小、SHA-256：

| City | 檔案 | SHA-256 |
| --- | --- | --- |
| citya | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/citya/train_frames3/checkpoints/visual_retrieval_A_only.pt` | `16af088d55cb93eee7f3d3d7126a89627e9545de27047fd6e363eeec85be9090` |
| citya | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/citya/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt` | `34b42624eb359fc2f952a655874ba3a1cb2b9f0bb0737106ec6c74b8589b55f6` |
| citya | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/citya/train_frames3/kalman_calibration.json` | `e12bdb1a323af8a4d54f1c82ea120055d7e5009becb41f62cfc4d9a225c54998` |
| cityb | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityb/train_frames3/checkpoints/visual_retrieval_A_only.pt` | `ba51192db5b2a44b860ea5b765e8a96a8604955be380451c5379f97a79841ba3` |
| cityb | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityb/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt` | `34b42624eb359fc2f952a655874ba3a1cb2b9f0bb0737106ec6c74b8589b55f6` |
| cityb | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityb/train_frames3/kalman_calibration.json` | `b3fa7e14cc0f0665729d9ebd0ccd73c9a76e3abd34946a1545a8a13e554fdbc2` |
| cityc | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityc/train_frames3/checkpoints/visual_retrieval_A_only.pt` | `2fbb3f588553614d569bd06b9233ed76086e7bcf1a27ba52fb57fb756f1f90d1` |
| cityc | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityc/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt` | `34b42624eb359fc2f952a655874ba3a1cb2b9f0bb0737106ec6c74b8589b55f6` |
| cityc | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityc/train_frames3/kalman_calibration.json` | `dca54c73e46a9a2325d00a2f3f8b4783186cd090e52b4ae44b33b3b96ea5e327` |
| cityd | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityd/train_frames3/checkpoints/visual_retrieval_A_only.pt` | `cfc7ba5f2d76ae7c7226b21f5ebecf7554a6f8572263d593f66f00911efd950c` |
| cityd | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityd/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt` | `34b42624eb359fc2f952a655874ba3a1cb2b9f0bb0737106ec6c74b8589b55f6` |
| cityd | `/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368/cityd/train_frames3/kalman_calibration.json` | `383e7784928ec4d3718afcfface7142fca63d36f0d7ab85dafd82237e83e879f` |

Final decoder參數：6×6／36候選，τ=0.30、bandwidth=7m、iterations=3、β=12。新解碼結果不回灌後續tracker。

GPU0計時結果：[decoder_latency/comparison.md](decoder_latency/comparison.md)。計時排除MS收斂及合併；GPU純運算與單次呼叫分開呈現。

2026-10-17公平比較重跑，沿用本權重及候選：[decoder_latency_fair/comparison.md](decoder_latency_fair/comparison.md)。MS主表改為36個收斂點的等價XY加總，舊1點數字只作歷史參考。

同日原碼流程重新計時：[decoder_latency_native/comparison.md](decoder_latency_native/comparison.md)，沿用同一權重、候選及分數，只改明確計時範圍。
