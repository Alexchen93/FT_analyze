# Impact Analysis / ImpactCapture（繁體中文）

本專案提供以 Python 開發的即時與離線工具，用於接收、可視化、與分析由 ESP32 韌體（`ImpactCapture.ino`）傳來的事件資料（EVT0/FRM0）。

- 即時監看（GUI）：`impact_analysis/impact_monitor.py`
- 命令列接收（CLI）：`impact_analysis/receive_events.py`
- 單檔離線視覺化：`impact_analysis/visualize_event.py`
- ESP32 韌體：`ImpactCapture/ImpactCapture.ino`
- 事件輸出：`events/YYYYMMDD_HHMMSS/evt_*.wav`、`features.csv`

## 特色
- 即時擷取與繪圖：波形、頻譜、頻譜圖（spectrogram，dB 色階）
- 儲存事件音檔（WAV）與特徵摘要（CSV）
- 支援離線載入單一 WAV 做可視化
- 簡潔 UI 操作（Start/Stop、檔案分析、F1/F11/Ctrl+Q 等）

## 目錄結構
- `impact_analysis/impact_monitor.py`：Tkinter GUI，即時擷取/繪圖/檔案分析
- `impact_analysis/receive_events.py`：命令列接收與（可選）即時繪圖
- `impact_analysis/visualize_event.py`：單一 WAV 的離線視覺化
- `impact_analysis/requirements.txt`：Python 依賴（NumPy、Matplotlib、PySerial 等）
- `ImpactCapture/ImpactCapture.ino`：ESP32 韌體（輸出 EVT0/FRM0）
- `events/`：工作階段輸出（例：`YYYYMMDD_HHMMSS/evt_*.wav`、`features.csv`）

## 環境安裝
- 建議 Python 3.10 以上
- Windows PowerShell 範例：

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r impact_analysis/requirements.txt
```

macOS/Linux 可改用 `source .venv/bin/activate` 啟用虛擬環境。

## 使用方式
### 1) 啟動 GUI 監看
```bash
python impact_analysis/impact_monitor.py
```

### 2) 命令列（無頭）接收與儲存
以 Windows 的 `COM5` 為例，並顯示即時繪圖、限制最高頻率 8 kHz：
```bash
python impact_analysis/receive_events.py -p COM5 -o events --live --fmax 8000
```
- Windows 連接埠：`COMx`
- macOS：`/dev/tty.usbserial*` 或 `/dev/tty.SLAB_USBtoUART`
- Linux：`/dev/ttyUSB*` 或 `/dev/ttyACM*`

### 3) 離線視覺化已儲存的 WAV
```bash
python impact_analysis/visualize_event.py events/20250101_120000/evt_*.wav --fmax 8000
```

### 4) 燒錄與設定 ESP32 韌體
- 使用 Arduino IDE 開啟 `ImpactCapture/ImpactCapture.ino`
- 選擇正確的 ESP32 板子與序列埠，編譯並上傳

## 輸出資料說明
- 事件目錄：`events/YYYYMMDD_HHMMSS/`
  - 事件音檔：`evt_*.wav`
  - 特徵摘要：`features.csv`
- 建議定期清理舊的 `events/` 以節省磁碟空間

## 開發與貢獻指南
- 程式風格：PEP 8、四空白縮排、適度型別註記；函式/模組 `snake_case`，常數 `UPPER_SNAKE_CASE`
- UI：次要操作放入選單（Help/Exit），簡潔標籤，快捷鍵一致（F1/F11/Ctrl+Q）
- 繪圖：軸標題與標籤齊全；spectrogram 採 dB 色階並隨資料更新 colorbar
- 測試：
  - 無正式測試套件；新增純函式（如特徵）時，建議以 `pytest` 撰寫小型單元測試
  - 大型樣本請勿加入版本控制
- Commit：簡短祈使句標題（≤72 字元），必要時補充動機與關聯 issue
- PR：聚焦單一議題，清楚描述、重現步驟、UI 變更附圖，標註所用序列埠與平台

## Git 與檔案忽略
- 已建議忽略：
  - `events/`（避免提交大量輸出）
  - `AGENT.md`／`AGENTS.md`（若存在）
- 若檔案已被追蹤：使用 `git rm --cached <path>` 後再提交
- 若需「從歷史抹除」敏感檔案，可使用 `git filter-repo` 或 BFG，並強制推送

## 安全與設定建議
- 確認正確序列埠（例：Windows `COM5`，macOS `/dev/tty.usbserial*`）
- 以 session 目錄（`events/YYYYMMDD_HHMMSS/`）管理輸出，定期清理
- 燒錄前於 Arduino IDE 選擇正確的 ESP32 板型與埠口

## 常見問題（FAQ）
- 啟動 GUI 後無資料？
  - 檢查序列埠是否正確，ESP32 是否已上傳韌體並供電
- 無法開啟序列埠／權限不足？
  - macOS/Linux 需將使用者加入對應的 dialout/tty 群組或以管理員權限執行
- 繪圖視窗不更新或閃爍？
  - 確認已安裝 `matplotlib` 與相容的後端；降低刷新率或資料率亦可改善

## 授權（License）
請於此處填入專案授權條款（例如 MIT、Apache-2.0 等）。

---
若需要簡體中文或英文版 README，我可以為你同步產生並保持內容一致。
