# Impact Analysis (Python)

Python 工具用於接收 ESP32 事件（EVT0 封包）、存 WAV、計算特徵與視覺化。

## 安裝

- 建議使用 Python 3.9+。
- 安裝套件：
  - `pip install -r impact_analysis/requirements.txt`

## 接收事件並計算特徵

- 關閉 Arduino Serial Monitor（避免占用 COM 埠）。
- 執行：

```
python impact_analysis/receive_events.py -p COM5 -b 921600 -o events
```

- 產出：
  - `events/<開啟時間>/*.wav`：每次執行自動建立子資料夾（例：`events/20250101_120301/`）
  - `events/<開啟時間>/features.csv`：本次執行的特徵檔

## 視覺化

- 針對某個 WAV 檔畫波形與頻譜圖：

```
python impact_analysis/visualize_event.py events/evt_YYYYMMDD_HHMMSS_NNNN.wav --fmax 8000
```

## 一鍵圖形介面（建議）

執行單一程式，先選擇 Port 後即時顯示並自動儲存事件：

```
python impact_analysis/impact_monitor.py
```

- 視窗中選擇 `Port`、確認 `Baud=921600`、輸出資料夾 `Out Dir`，按 `Start`。
- 三個視圖：
  - 波形（最近 2 秒）
  - 即時 FFT（最新窗口，Hann）
  - 即時頻譜圖（Spectrogram；最近幾秒，dB）
- 事件進行時持續更新；事件結束後自動存 WAV 與寫入 `features.csv`。
- 每次按下 `Start`，會於 `Out Dir` 底下建立當次時間的子資料夾並存放本次所有檔案。

預設開啟即為全螢幕模式：
- 按 F11 可切換全螢幕/視窗模式；按 Esc 可離開全螢幕。

### 檔案分析模式
- 按 GUI 上的 `Analysis` 按鈕，開啟「檔案讀取分析」視窗。
- 視窗左側：瀏覽 `Out Dir` 下各次執行（時間戳資料夾）與其中的 WAV 檔。
- 點選任一 WAV：右側顯示該檔案的波形、FFT 與頻譜圖。
- 可用上方 `Browse`/`Refresh` 換資料夾或重整列表。

## 常見問題

- 讀不到資料：確認 `-p COMx` 正確、鮑率 `-b 921600`，且 ESP32 韌體正在輸出 EVT0 封包（勿開 Serial Monitor）。
- 亂碼：正常，韌體輸出二進位；請使用本工具接收。
- 特徵不足：可擴充 `receive_events.py` 的 `extract_features()`（例如增加譜平坦度、短時能量包絡等）。
