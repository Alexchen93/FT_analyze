# Python Impact Analysis Tools

Python 工具集用於 ESP32 即時撞擊聲音分類系統。

## 工具概覽

### `real_time_classifier.py` （階段1：即時監控）
20Hz 判斷頻率的即時分類監控 GUI。

**功能：**
- 解析 RSLT 封包（判斷結果，每 50ms）
- 解析 EVT0 封包（完整音訊，每 250ms）
- 即時顯示：判斷結果 (0/1)、RMS 數值
- 可視化：波形、FFT、頻譜圖
- 自動儲存：`results.csv` (所有判斷) + `features.csv` (完整特徵) + WAV 檔案

**使用方式：**
```bash
python impact_analysis/real_time_classifier.py
```

**輸出結構：**
```
events/YYYYMMDD_HHMMSS/
├── results.csv         # 20Hz 判斷結果 (timestamp, result, rms, peak)
├── features.csv        # 4Hz 完整特徵 (9 DSP features + wav filename)
└── chunk_*.wav         # 50ms 音訊檔案 (800 samples @ 16kHz)
```

---

### `train_classifier.py` （階段2：訓練分類器）
從收集的數據訓練機器學習模型。

**功能：**
- 從多個 session 資料夾載入標註數據
- 訓練 Random Forest 和 SVM 分類器
- 生成混淆矩陣、ROC 曲線、特徵重要性圖表
- 尋找最佳閾值（單特徵分類器）

**使用方式：**
```bash
# 收集數據
mkdir -p events/positive_samples events/negative_samples
python impact_analysis/real_time_classifier.py
# 執行目標撞擊動作 → 輸出至 positive_samples
# 執行其他不相關聲音 → 輸出至 negative_samples

# 訓練模型
python impact_analysis/train_classifier.py \
  --positive events/positive_samples/*/ \
  --negative events/negative_samples/*/ \
  --model both \
  --output models
```

**參數說明：**
- `--positive`: 正樣本資料夾路徑（可多個）
- `--negative`: 負樣本資料夾路徑（可多個）
- `--model`: 模型類型 (`rf`, `svm`, `both`)
- `--output`: 模型輸出目錄（預設：`models/`）
- `--test-size`: 測試集比例（預設：0.2）

**輸出：**
```
models/
├── random_forest.joblib       # Random Forest 模型
├── svm.joblib                 # SVM 模型
├── rf_confusion_matrix.png    # 混淆矩陣
├── rf_roc_curve.png          # ROC 曲線
├── svm_confusion_matrix.png
└── svm_roc_curve.png
```

**終端輸出：**
- 訓練/測試準確率
- 分類報告（precision, recall, F1-score）
- 特徵重要性排名
- 最佳 RMS 閾值（如適用）

---

### `visualize_event.py` （離線視覺化）
離線分析已儲存的 WAV 檔案。

**功能：**
- 繪製波形圖
- 繪製頻譜圖（Spectrogram）

**使用方式：**
```bash
python impact_analysis/visualize_event.py events/20250107_*/chunk_*.wav --fmax 8000
```

**參數說明：**
- `wav_files`: WAV 檔案路徑（可多個，支援萬用字元）
- `--fmax`: 頻譜圖最高頻率（Hz，預設：8000）

---

### `impact_monitor.py` / `receive_events.py` （舊版工具）
事件驅動的擷取工具（非固定週期），保留用於向下相容。

**使用方式：**
```bash
# GUI 版本
python impact_analysis/impact_monitor.py

# CLI 版本
python impact_analysis/receive_events.py -p COM5 -b 921600 -o events --live --fmax 8000
```

**注意：** 這些工具使用舊的事件驅動協定（FRM0/EVT0），不支援 RSLT 封包。建議使用 `real_time_classifier.py` 進行新開發。

---

## 安裝依賴

```bash
pip install -r impact_analysis/requirements.txt
```

**依賴套件：**
- `numpy`: 數值計算
- `matplotlib`: 繪圖
- `pyserial`: 串口通訊
- `scikit-learn`: 機器學習（訓練工具需要）
- `joblib`: 模型儲存/載入

---

## 資料格式說明

### `results.csv` 格式（RSLT 封包）
```csv
timestamp_ms,result,rms,peak
0,0,45.23,0.12
50,0,48.91,0.15
100,1,385.67,0.78
150,1,392.11,0.81
```

**欄位說明：**
- `timestamp_ms`: ESP32 啟動後經過的毫秒數
- `result`: 分類結果（0 或 1）
- `rms`: 均方根值（RMS）
- `peak`: 峰值

### `features.csv` 格式（EVT0 封包）
```csv
timestamp_ms,wav,fs,nsamples,result,peak,rms,crest,zcr,duration_ms,centroid_hz,rolloff85_hz,bandwidth_hz,dominant_hz
20250107_153042_123,chunk_20250107_153042_123_r1.wav,16000,800,1,0.78,385.67,2.03,0.15,42.5,3245.12,5632.89,1234.56,3100.0
```

**欄位說明：**
- `timestamp_ms`: 時間戳（YYYYMMDD_HHMMSS_mmm 格式）
- `wav`: 音訊檔案名稱
- `fs`: 取樣率（Hz）
- `nsamples`: 樣本數
- `result`: 對應的分類結果
- 後續 9 個欄位為 DSP 特徵（時域 + 頻域）

---

## 常見問題

### Q: `real_time_classifier.py` 無法開啟序列埠？
A:
1. 確認 ESP32 已連接並燒錄韌體
2. 關閉 Arduino Serial Monitor（會佔用埠口）
3. Windows: 檢查裝置管理員中的 COM 埠號
4. Linux/macOS: 使用 `ls /dev/tty*` 查看可用埠口，可能需要 sudo 權限

### Q: 訓練時顯示 "沒有載入任何數據"？
A:
1. 確認資料夾路徑正確（需包含 `features.csv`）
2. 檢查 `--positive` 和 `--negative` 參數是否正確
3. 確保資料夾名稱或父目錄名稱可被映射到標籤（例如：`positive_samples`, `negative_samples`）

### Q: 模型準確率很低？
A:
1. 收集更多樣本（建議每類至少 100 個）
2. 確保正負樣本數量平衡
3. 檢視特徵重要性，選擇最相關的特徵
4. 調整模型超參數（修改 `train_classifier.py` 中的參數）

### Q: 如何調整 RMS 閾值？
A:
1. 執行階段1收集數據
2. 打開 `results.csv`，觀察正負樣本的 RMS 範圍
3. 設定閾值在兩者之間（例如：正樣本 RMS > 350，負樣本 RMS < 200，設定閾值 = 275）
4. 或使用 `train_classifier.py` 自動尋找最佳閾值

---

## 進階使用

### 動態調整閾值
在 GUI 執行時，透過 Serial 命令調整閾值：
```python
import serial
import struct

ser = serial.Serial('COM5', 921600)
new_threshold = 350.0
ser.write(b'T' + struct.pack('<f', new_threshold))
ser.close()
```

### 自定義特徵
修改 `extract_features()` 函式以加入新的 DSP 特徵：
```python
def extract_features(x: np.ndarray, fs: int):
    # 現有特徵...

    # 新增：頻譜平坦度
    mag = np.abs(np.fft.rfft(x))
    spectral_flatness = np.exp(np.mean(np.log(mag + 1e-12))) / (np.mean(mag) + 1e-12)

    return peak, rms, crest, zcr, duration_ms, centroid, rolloff85, bandwidth, dominant, spectral_flatness
```

### 多類別分類
1. 修改數據收集：建立多個標籤資料夾（`class_A/`, `class_B/`, `class_C/`）
2. 修改 `train_classifier.py` 的 `label_map` 以支援 0, 1, 2, ... 多個類別
3. 修改 ESP32 韌體以支援多類別輸出

---

## 參考資料

- 完整系統說明：[`../README_zh-TW.md`](../README_zh-TW.md)
- 階段3部署指南：[`../STAGE3_DEPLOYMENT.md`](../STAGE3_DEPLOYMENT.md)
- 開發指南：[`../AGENTS.md`](../AGENTS.md)
