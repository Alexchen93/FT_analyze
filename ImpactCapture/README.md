# ESP32 Impact Capture (MAX9814)

以 ESP32 + MAX9814 即時擷取敲擊聲，偵測事件並即時顯示波形/FFT/頻譜圖，同時把事件切片存成 WAV 並輸出特徵。

- 韌體：Arduino 草稿 `ImpactCapture/ImpactCapture.ino`（I2S-ADC；支援 FRM0 串流 + EVT0 事件）
- 工具：Python（GUI `impact_analysis/impact_monitor.py`、CLI `impact_analysis/receive_events.py`、離線 `impact_analysis/visualize_event.py`）

## 接線
- MAX9814 `VCC -> 3V3`、`GND -> GND`
- MAX9814 `OUT -> ESP32 GPIO34`（ADC1_CH6；避免使用 ADC2 腳位）

## 序列協定（921600 bps）
- FRM0（事件期間即時短幀，用於即時顯示）：
  `"FRM0"(4) + sample_rate<uint32_le>(4) + sample_count<uint32_le>(4) + int16_le[count]`
- EVT0（事件結束後完整切片，用於存檔與分析）：
  `"EVT0"(4) + sample_rate<uint32_le>(4) + sample_count<uint32_le>(4) + int16_le[count]`

## 燒錄（Arduino IDE）
1. 安裝板卡核心：偏好設定加入 `https://dl.espressif.com/dl/package_esp32_index.json`，Boards Manager 安裝「ESP32 by Espressif Systems」（建議 2.0.x）。
2. 開啟 `ImpactCapture/ImpactCapture.ino`。
3. Tools → Board 選 `ESP32 Dev Module`（若為 S3/C3/S2 請調整板型與 ADC 腳位）。
4. Upload 速度 `921600`（若失敗改 `460800`），按 Upload。

上傳後請不要用 Serial Monitor 觀察資料（為二進位封包），改用下方 Python 工具。

## Python 工具
安裝依賴：`pip install -r impact_analysis/requirements.txt`

- GUI（建議）：`python impact_analysis/impact_monitor.py`
  - 視窗可選 Port、設定輸出資料夾與 fmax。
  - 顯示：
    - 波形（最近 2 秒）
    - 即時 FFT（最新窗口）
    - 即時頻譜圖（最近幾秒）
  - 偵測到事件時，會自動建立「以開啟時間命名」的子資料夾，存放當次所有 WAV 與 `features.csv`。

- CLI：`python impact_analysis/receive_events.py -p COMx -b 921600 --live --fmax 8000 -o events`
  - 功能同上（事件切片存檔、特徵計算、可選即時波形/FFT）。
  - 亦會為每次執行自動建立時間戳子資料夾。

- 離線視覺化：`python impact_analysis/visualize_event.py events/<session>/evt_xxx.wav --fmax 8000`
  - 顯示單一 WAV 的波形與頻譜圖。

## 偵測/擷取參數（在 ImpactCapture.ino）
- 門檻：`MIN_RMS_START`、`START_RATIO`、`CONT_RATIO`（依環境噪聲調整）
- 擷取窗：`PRE_MS`、`POST_MS`、`MAX_EVENT_MS`
- 取樣率：`SAMPLE_RATE`（預設 16 kHz；可調高）

## 注意
- 串口輸出為二進位封包，Serial Monitor 看到亂碼屬正常。
- 非經典 ESP32（如 S3/C3/S2）需調整板子設定與可用 ADC 腳位，並確認 I2S-ADC 支援。

