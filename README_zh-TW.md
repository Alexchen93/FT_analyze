# ESP32 即時撞擊聲音分類系統

**20Hz 判斷頻率 | 本地邊緣推理 | 三階段進化架構**

本專案提供完整的撞擊聲音偵測與分類解決方案，從數據收集、模型訓練到嵌入式部署。

## 系統架構

```
[ESP32 + MAX9814] ──(921600 baud)──> [Python 工具]
       │                                      │
       ├─ 20Hz 本地判斷 (RMS/多特徵)         ├─ 即時可視化
       ├─ RSLT 封包 (17 bytes, 0.15ms)      ├─ 數據收集
       └─ EVT0 封包 (音訊 + 特徵)            └─ 模型訓練
```

## 三階段進化路徑

### 階段1：簡單閾值分類器（立即可用）
- ESP32 每 50ms 計算 RMS，與閾值比較
- 輸出判斷結果 (0/1) + 音訊數據
- Python 即時監控與可視化
- **適合：快速原型驗證、單一特徵足夠的場景**

### 階段2：數據驅動分類器（收集樣本後）
- 使用階段1收集正負樣本
- Python 訓練 Random Forest / SVM 模型
- 找出最佳特徵組合與閾值
- **適合：需要多特徵組合、有標註數據**

### 階段3：嵌入式機器學習（TinyML）
- 將模型部署到 ESP32 本地推理
- 支援決策樹、神經網路、Edge Impulse
- 推理延遲 < 5ms，無需 Python 即可運行
- **適合：複雜決策邊界、獨立運行需求**

## 快速開始

### 1. 硬體接線
```
MAX9814 → ESP32
  VCC   → 3V3
  GND   → GND
  OUT   → GPIO34 (ADC1_CH6)
```

### 2. 燒錄韌體
```bash
# Arduino IDE
1. 開啟 ImpactCapture/ImpactCapture.ino
2. 選擇板子：ESP32 Dev Module
3. 上傳速度：921600
4. 上傳
```

### 3. 安裝 Python 環境
```bash
python -m venv .venv
.\.venv\Scripts\activate           # Windows
source .venv/bin/activate          # macOS/Linux
pip install -r impact_analysis/requirements.txt
```

### 4. 啟動即時監控（階段1）
```bash
python impact_analysis/real_time_classifier.py
```

介面功能：
- 選擇序列埠、設定輸出資料夾
- 即時顯示判斷結果 (0/1)、RMS 數值
- 波形、FFT、頻譜圖可視化
- 自動儲存 `results.csv`（所有判斷）和 `features.csv`（完整特徵）

### 5. 收集訓練數據（階段2 準備）
```bash
# 分別收集正樣本和負樣本
# 正樣本：執行你想要偵測的撞擊動作
mkdir -p events/positive_samples
python impact_analysis/real_time_classifier.py
# 輸出到 events/positive_samples/20250107_xxx/

# 負樣本：執行其他不相關的聲音
mkdir -p events/negative_samples
python impact_analysis/real_time_classifier.py
# 輸出到 events/negative_samples/20250107_xxx/
```

### 6. 訓練分類器（階段2）
```bash
python impact_analysis/train_classifier.py \
  --positive events/positive_samples/20250107_*/ \
  --negative events/negative_samples/20250107_*/ \
  --model both \
  --output models
```

輸出：
- `models/random_forest.joblib`：隨機森林模型
- `models/svm.joblib`：SVM 模型
- `models/*_confusion_matrix.png`：混淆矩陣
- `models/*_roc_curve.png`：ROC 曲線
- 終端顯示：特徵重要性、最佳閾值

### 7. 部署到 ESP32（階段3）

#### 方案 A：更新閾值（最簡單）
```cpp
// ImpactCapture.ino
float THRESHOLD_RMS = 350.0f;  // 使用階段2找出的最佳值
```

#### 方案 B：多特徵規則
```cpp
// ImpactCapture.ino 啟用進階分類器
uint8_t classify_advanced(float rms, float peak, float dominant_freq) {
  if (rms > 350.0 && peak > 0.6 &&
      dominant_freq > 2000 && dominant_freq < 5000) {
    return 1;
  }
  return 0;
}
```

#### 方案 C：Edge Impulse（完整 TinyML）
參考 [`STAGE3_DEPLOYMENT.md`](STAGE3_DEPLOYMENT.md) 詳細步驟。

## 目錄結構

```
.
├── ImpactCapture/
│   └── ImpactCapture.ino          # ESP32 韌體 (20Hz 分類器)
├── impact_analysis/
│   ├── real_time_classifier.py    # 階段1: 即時監控 GUI
│   ├── train_classifier.py        # 階段2: 訓練分類器
│   ├── visualize_event.py         # 離線視覺化工具
│   └── requirements.txt           # Python 依賴
├── events/                         # 數據收集輸出
│   ├── positive_samples/
│   │   └── 20250107_HHMMSS/
│   │       ├── results.csv        # 所有判斷結果 (20Hz)
│   │       ├── features.csv       # 完整特徵 (4Hz)
│   │       └── chunk_*.wav        # 音訊檔案
│   └── negative_samples/
│       └── ...
├── models/                         # 訓練好的模型
│   ├── random_forest.joblib
│   ├── svm.joblib
│   └── *.png                      # 評估圖表
├── STAGE3_DEPLOYMENT.md           # 階段3 詳細指南
└── README_zh-TW.md                # 本文件
```

## 序列協定說明

### RSLT 封包（判斷結果，每 50ms）
```
"RSLT" (4 bytes) +
timestamp_ms (uint32_le, 4 bytes) +
result (uint8, 1 byte) +          ← 0 或 1
rms (float32_le, 4 bytes) +
peak (float32_le, 4 bytes)
= 17 bytes 總計
```

### EVT0 封包（完整音訊，每 250ms 或偵測到時）
```
"EVT0" (4 bytes) +
sample_rate (uint32_le, 4 bytes) +
sample_count (uint32_le, 4 bytes) +
result (uint8, 1 byte) +          ← 對應的判斷結果
int16_le[sample_count]            ← 音訊數據
```

## 性能指標

| 指標 | 數值 |
|------|------|
| 判斷頻率 | 20Hz (每 50ms) |
| 判斷延遲 | < 2ms (ESP32 本地) |
| RSLT 傳輸時間 | 0.15ms |
| EVT0 傳輸時間 | 14ms (800 samples @ 921600 baud) |
| 音訊取樣率 | 16 kHz |
| 最高頻率分析 | 8 kHz (Nyquist) |

## 調整參數

### ESP32 韌體 (`ImpactCapture.ino`)
```cpp
#define CHUNK_MS 50                // 判斷週期 (50ms = 20Hz)
#define SAMPLE_RATE 16000          // 取樣率
float THRESHOLD_RMS = 300.0f;      // RMS 閾值
#define SEND_AUDIO_EVERY_N 5       // 每 N 次送一次音訊 (5 = 4Hz)
```

### Python 動態調整閾值（進階）
```python
import serial
import struct

ser = serial.Serial('COM5', 921600)
new_threshold = 350.0
ser.write(b'T' + struct.pack('<f', new_threshold))
```

## 常見問題

### Q: 判斷結果都是 0 或都是 1？
A: 調整 `THRESHOLD_RMS`。執行階段1 收集數據後，檢視 `results.csv` 的 RMS 欄位，設定閾值在正負樣本之間。

### Q: 如何提高判斷頻率到 50Hz (20ms)?
A: 修改韌體 `#define CHUNK_MS 20`，但需注意串口傳輸時間可能不足，建議只送 RSLT 封包（註解掉 EVT0 傳送部分）。

### Q: 可以使用其他 ADC 腳位嗎？
A: 可以，但必須是 ADC1 通道（GPIO32-39）。修改韌體：
```cpp
#define ADC_CHANNEL ADC1_CHANNEL_4  // GPIO32
```

### Q: Edge Impulse 訓練失敗？
A: 確保每個類別至少有 20 個樣本，且音訊長度一致（50ms = 800 samples）。

### Q: 模型準確率不高怎麼辦？
A:
1. 收集更多樣本（每類至少 100 個）
2. 確保正負樣本平衡
3. 嘗試不同特徵組合（檢視特徵重要性）
4. 使用更複雜的模型（階段3）

## 進階功能

### 多類別分類（3+ 類別）
修改韌體輸出 `uint8_t` 範圍 0-N，Python 訓練多類別分類器。

### 連續動作追蹤
在韌體中加入狀態機，追蹤連續判斷結果：
```cpp
if (result == 1 && prev_result == 0) {
  // 上升沿：開始動作
}
```

### 資料庫儲存
修改 Python 工具，將結果寫入 SQLite/PostgreSQL 而非 CSV。

## 貢獻指南

- 程式風格：PEP 8、四空白縮排
- Commit：簡短祈使句（≤50 字元）
- PR：聚焦單一功能，附上測試結果

## 授權

MIT License

## 參考資源

- [Edge Impulse 文檔](https://docs.edgeimpulse.com/)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32 I2S-ADC 範例](https://github.com/espressif/esp-idf/tree/master/examples/peripherals/adc)
