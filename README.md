> v40 已於 2026-10-21 精簡，只保留最佳歷史驗證、固定 6×6 最佳與最新四城市版本及必要依賴；[訓練程式與結果](v40/README.md)。

# UAV–Satellite 定位實驗：程式與版本結果

這是 `/yh/study/uav-sat` 的精簡上傳快照（2026-10-21）。包含各版本程式、實驗設定、可取得的結果數據及報告。

- **[各版本結果索引](RESULTS_INDEX.md)**：先從這裡找版本對應的數據。
- [完整結果清單](VERSION_RESULTS.csv)：保留原版本與實驗目錄，不混合不同測試條件。
- [v40 程式與設定](v40/)／[v40 最佳／最新與訓練來源](v40/README.md)。
- [v39_otherdata](v39_otherdata/)／[report/1016](report/1016/)。
- [快照統計](upload_metadata/snapshot_summary.json)、[納入檔案與 SHA256](upload_metadata/included_files.csv)、[排除清單](upload_metadata/excluded_files.csv)。

不包含模型權重、模型快取、原始資料集影像、影片、大型二進位檔及重複的本地備份。一般單檔上限 5 MiB；結果圖片上限 2 MiB。大於 1 MiB 的 CSV 以 `.csv.gz` 無損壓縮，完整逐幀數字仍在，可用 `gzip -dk path/to/result.csv.gz` 還原。

GitHub 原本已有、而目前本地分支沒有的檔案原則上保留；只移除不符本次上傳政策的產物及本地已刪除的對應檔案。沒有改寫舊 commit 歷史，因此歷史版本不受這次精簡刪檔影響。

第三方方法保留 git submodule 參照與來源網址；本地追蹤檔修改保存在 `upload_metadata/submodules/*/local_changes.patch`，新增的少量程式也一併保存，沒有替第三方 repository 推送。先 `git submodule update --init`，再於相應 submodule 目錄套用 patch。詳細參照 [submodules.json](upload_metadata/submodules.json)。

這次範圍僅為 `/yh/study/uav-sat`；未將其他目錄的實驗冒充為此快照內容。快照不是重新執行實驗，也不保證外部資料／權重下載後可不調整絕對路徑即重現；請依各版本 README 的環境與資料條件操作。
