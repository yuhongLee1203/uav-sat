# R@1 與 Navigation 指標：本版本能算什麼

| 指標 | 成功條件／用途 | 3.759 m 版本現在可否報告 | 必須使用的名稱或限制 |
| --- | --- | --- | --- |
| InfoGeo retrieval R@1 | Top-1 gallery 影像是 query 的標註正樣本 | 現有 replay 不能直接得出 | 需要固定全 gallery、正樣本 ID 與影像排名；36 個 GT-conditioned 局部 patch 不是同一 protocol |
| Bearing-style 四 tile R@1 | 預測與 GT 對應四鄰接 tile 中相同的一張 | 可以，已計算 93.66% | Four-tile coordinate-derived R@1，GT-assisted route replay |
| LSR@3/@5/@15 | 每幀位置誤差 ≤ 3/5/15 m | 可以，已計算 | 依 1,482 筆原始 GT／預測計算 |
| MLE | 每幀定位距離誤差的平均 | 可以，3.759 m | GT-assisted localization |
| SR@20 | 完整導航 episode 在規定步數／終止條件下到達目標 20 m 內 | 目前 N/A | 需要動作影響下一個真實位置與觀測的 closed-loop rollout |
| SPL | 成功率再依導航實際行走／飛行長度給分 | 目前 N/A | 需要 controller 走出的實際軌跡；估計座標連線長度不可當實際飛行距離 |
| NE | episode 結束時真實 UAV 位置到目標的距離 | 目前 N/A | 不是最後一幀 predicted XY 到 GT XY 的誤差，也不是全程 MLE |
| MHE／HSR | camera heading 的角度誤差／成功率 | 目前 N/A | 此版本保存的 heading 目標是合成路線運動方向，不能代替資料的 camera yaw |

## 為什麼 regression 模型也能有四 tile R@1

四張對稱 tile 的最近中心分界是 RSB 中心的水平／垂直軸。把 final XY 轉回絕對 map 座標，再計算它相對於該筆 RSB 中心的 x/y 正負號，即可判斷是哪一張 tile。GT 同樣轉換後比較，無需新增 classification head。本次逐筆核對 manifest 的 UAV 影像名稱、metadata source_index、GT 座標與象限，沒有 GT 落在分界線上的平手樣本。

這個值不表示模型真的輸出了全 gallery 檢索排名。小於 15 m 的定位誤差也可能跨過 tile 邊界，因此 LSR@15=100% 而四 tile R@1=93.66% 並不矛盾。

先前「沒有 retrieval head 所以任何 R@1 都不能算」的說法過於籠統：沒有 paired-image ranking 時不能宣稱 InfoGeo 式 R@1，但可以按 Bearing-UAV 的座標衍生規則另算四 tile 診斷。名稱相同並不保證兩種 R@1 可直接比較。

## SR、SPL、NE 的計算方式

對第 i 個導航 episode，令 g_i 為導航目標、x_i^end 為終止時的真實位置、p_i 為實際行進長度、l_i 為事先指定之可行最短／參考路徑長度（有必經 waypoint 時須遵循該 benchmark 定義）：

- NE_i = d(x_i^end, g_i)，NE = mean(NE_i)。
- S_i = 1[NE_i ≤ 20 m 且符合該 benchmark 的完成／終止條件]，SR@20 = 100 × mean(S_i)。
- SPL = mean(S_i × l_i / max(l_i, p_i))；通常介於 0 與 1，若乘 100 要明標百分比。

這裡列的是通用定義；Bearing 論文的公開文字給出 25 m 步長、20 m waypoint 半徑，但沒有在該段完整寫出 SPL 路徑長度／終止細節。正式比較前仍要鎖定 simulator 與 evaluator 的實作，不可只套相同縮寫。

## 目前為何沒有硬算 navigation 數值

3.759 m 資料是固定影像序列，不是模型執行動作後取得的影像。就算估計位置偏離，下一幀依然是原先選好的影像。此外這個版本以當前 GT 提供參考點與選 hypothesis。把八條 replay 的最後一幀定位誤差當 NE、把它們都讀到最後一張當 SR，會混淆定位與導航，因此本次主表保留 N/A，沒有填虛構的導航成績。

架構可以成為導航系統的定位模組；要產生有效導航結果，還需接上：可依實際位置與 heading 產生觀測的環境、利用預測位置決定下一步的 controller、明確的 heading 來源、GT-free 的定位輸入，以及與 benchmark 一致的 reset／step size／終止／失敗規則。Simulator 可用真實狀態生成觀測與評分，但不能把真實狀態當作模型搜尋參考。若使用外部已知 heading 或 oracle reference，須另列為對應條件的導航實驗。

## 核對來源

- [Bearing-UAV 論文：Appendix A.2、Section 4.3.6](https://arxiv.org/html/2603.22153v1)
- [InfoGeo：Section 5.1](https://arxiv.org/pdf/2605.07099)
- 本機 Bearing 座標指標實作：`/yh/study/bearinguav_lastver/cvphr/test/cvphr_test.py`，`RECALL_AT_K_PHR`。
- 本機導航流程：`/yh/study/bearinguav_lastver/naver/runners/nav.py`，`phreg`、heading 更新與 next-point 更新。
- 原始版本限制：`v39_otherdata/README_ORIGINAL_3759_INFERENCE.md`、原 suite 的 `publication_audit.md`。

- [SPL 原始定義：On Evaluation of Embodied Navigation Agents](https://arxiv.org/pdf/1807.06757)，Section 4。
