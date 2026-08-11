# RTL-CRF for UAV–Satellite Visual Localization

      GNSS-denied UAV 視覺定位，使用 UAV 影像與具有地理參考的衛星影像進行局部 retrieval，並利用 Residual Second-Order Temporal Lattice CRF（RTL-CRF）提升時間一致性與定位穩定性。

## 正式實驗設定

#
CUDA_VISIBLE_DEVICES=0 python3 new.py 

- Visual retrieval 訓練：Route A
- Visual retrieval 驗證：Route A
- RTL-CRF 訓練：Route A
- RTL-CRF 驗證：Route A
- 最終測試：Route B、Route C

Route B 與 Route C 不參與任何模型訓練、驗證或 checkpoint 選擇。

Visual backbone 使用公開預訓練 MobileCLIP2-S2，backbone 維持 frozen。

UAV/SAT retrieval heads 與 RTL-CRF task-specific parameters 均從新的 task-specific 初始化開始，只使用 Route A 訓練，不載入舊 UAV/SAT task checkpoint。

## Controlled Local-Prior Protocol

 controlled noisy local prior：

GT + deterministic jitter

 noisy prior 為中心建立固定 6×6 = 36 個 SAT candidates。

trolled local prior 下，模型是否能從 Route A 泛化至未見的 Route B / Route C，並改善 UAV–Satellite retrieval 定位與時間一致性。

>'EOF'    closed-loop navigation 實驗。

## Retrieval

 UAV 影像與 36 個 SAT candidates 分別經過 MobileCLIP backbone 與 UAV/SAT heads。

 36 個 cosine similarity scores。

Retrieval heads 僅使用 Route A 的 train split 訓練，Route A validation split 選擇最佳 checkpoint。

## RTL-CRF

RTL-CRF 使用最近 5 幀。

 36 個候選，每個候選保留真實 2D 地圖座標。

CUDA_VISIBLE_DEVICES=0 python3 new.py 

1. Emission network 重新評估每個候選的單點可信度。
2. First-order transition 評估相鄰兩幀候選之間的二維移動。
3. Second-order transition 使用連續三幀候選建模速度、加速度與方向一致性。
4. CRF 使用 log-sum-exp dynamic programming 整合所有可能的候選路徑。
5. 得到最後第 t 幀 36 個候選的 posterior probability。
6. 使用 posterior-weighted expectation 得到連續二維位置。
7. 與 Fixed HardMS anchor 經 learned residual correction 得到最終 RTL-CRF 位置。

 5-frame window 只正式輸出最後第 t 幀的位置。

' sliding window：

frames 1–5 -> output frame 5  
frames 2–6 -> output frame 6  
frames 3–7 -> output frame 7

## 主要檔案

`config.py`  
#
CUDA_VISIBLE_DEVICES=0 python3 new.py  Train A / Test B+C protocol。

`data.py`  
UAV dataset、satellite map 與座標處理。

`visual_model.py`  
MobileCLIP retrieval heads 與 TemporalLatticeCRF。

`visual_localizer.py`  
Route-A-only retrieval training、satellite gallery、candidate retrieval 與 Fixed HardMS。

`robust_tracker.py`  
'EOF'    strict experiment：
A-only retrieval training → A-only RTL-CRF training → B/C evaluation。

`run_robust_tracker.sh`  
#


`visualize_rtl_crf_stages.py`  
RTL-CRF 中間過程視覺化。

`render_results_video.py`  
 Route B / Route C 定位影片。

## 從零重新訓練

```bash
CUDA_VISIBLE_DEVICES=0 \
bash run_robust_tracker.sh \
  --mode train_eval \
  --visual-epochs 30 \
  --epochs 40 \
  --jitter-m 12 \
  --eval-split all





test\
