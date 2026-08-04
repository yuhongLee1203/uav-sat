前 5 個 GT nodes
→ 初始化位置與穩健速度

最近 5 個最終位置
→ least-squares 直線擬合
→ 更新平滑速度與方向
→ 向前預測下一個位置

預測位置
→ 根據 tracking confidence 選擇 6×6 / 10×10 / 14×14 / 18×18

UAV × SAT candidates
→ cosine logits
→ 沿行進方向較寬、左右方向較窄的 motion prior
→ fused logits
→ Fixed HardMS
→ fused visual position

直線預測 + fused visual position
→ 限制沿線與橫向最大修正
→ continuous correction
→ final temporal position