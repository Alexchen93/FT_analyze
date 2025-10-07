# ESP32 Impact Classifier Firmware

ESP32 韌體：20Hz 即時撞擊聲音分類器

## 功能特色

- **20Hz 固定週期判斷**：每 50ms 分析一次音訊（800 samples @ 16kHz）
- **本地即時分類**：ESP32 端計算 RMS/peak，判斷延遲 < 2ms
- **雙協定序列輸出**：
  - RSLT 封包：輕量判斷結果（17 bytes，每 50ms）
  - EVT0 封包：完整音訊資料（可配置頻率，預設每 250ms）
- **三階段進化架構**：
  - 階段1：簡單 RMS 閾值分類器（當前實作）
  - 階段2：數據驅動多特徵分類器（預留介面）
  - 階段3：嵌入式機器學習模型（TinyML）

## 硬體需求

### 元件
- ESP32 開發板（ESP32 Dev Module）
- MAX9814 電容式麥克風放大器模組
- USB 連接線

### 接線
```
MAX9814 → ESP32
  VCC   → 3.3V
  GND   → GND
  OUT   → GPIO34 (ADC1_CH6)
```

**重要：** 必須使用 ADC1 通道（GPIO32-39），ADC2 通道與 WiFi 衝突。

## 燒錄韌體

### 使用 Arduino IDE

1. **安裝 ESP32 核心：**
   - 檔案 → 偏好設定 → 額外板子管理員網址
   - 加入：`https://dl.espressif.com/dl/package_esp32_index.json`
   - 工具 → 板子 → 板子管理員
   - 搜尋並安裝 "ESP32 by Espressif Systems" (建議 2.0.x 版本)

2. **開啟韌體：**
   ```
   開啟 ImpactCapture/ImpactCapture.ino
   ```

3. **設定板子：**
   - 工具 → 板子：`ESP32 Dev Module`
   - 工具 → Upload Speed：`921600` (若失敗改為 `460800`)
   - 工具 → 連接埠：選擇對應的 COM 埠（Windows: COMx, macOS/Linux: /dev/ttyUSBx）

4. **上傳：**
   - 點擊「上傳」按鈕
   - 若上傳失敗，按住 ESP32 的 BOOT 按鈕再重試

5. **重要提醒：**
   - **不要開啟 Arduino Serial Monitor**！輸出為二進位封包，需使用 Python 工具解析。

## 序列協定

### 鮑率
```cpp
921600 baud (若不穩定改為 460800)
```

### RSLT 封包（判斷結果）
```
"RSLT" (4 bytes) +
timestamp_ms (uint32_t, 4 bytes, little-endian) +  // ESP32 啟動後毫秒數
result (uint8_t, 1 byte) +                         // 0 或 1
rms (float, 4 bytes, little-endian) +              // 均方根值
peak (float, 4 bytes, little-endian)               // 峰值
= 17 bytes 總計
```

**傳輸時間：** ~0.15ms @ 921600 baud

**頻率：** 每 50ms (20Hz)

### EVT0 封包（完整音訊）
```
"EVT0" (4 bytes) +
sample_rate (uint32_t, 4 bytes) +                  // 取樣率（16000）
sample_count (uint32_t, 4 bytes) +                 // 樣本數（800）
result (uint8_t, 1 byte) +                         // 對應的判斷結果
audio_data (int16_t[], sample_count*2 bytes)       // 音訊數據
```

**傳輸時間：** ~14ms for 800 samples @ 921600 baud

**頻率：** 可配置（預設每 5 次 = 4Hz）

## 參數調整

### 判斷週期
```cpp
#define CHUNK_MS 50                 // 50ms = 20Hz
```
- 修改為 `20` → 50Hz (20ms 週期)
- 修改為 `100` → 10Hz (100ms 週期)

### 取樣率
```cpp
#define SAMPLE_RATE 16000           // 16 kHz
```
- 支援範圍：8000 - 48000 Hz
- 提高取樣率會增加串口傳輸負擔

### 分類閾值（階段1）
```cpp
float THRESHOLD_RMS = 300.0f;
```
- 根據實際環境調整
- 使用階段2訓練工具找出最佳值

### 音訊傳輸頻率
```cpp
#define SEND_AUDIO_EVERY_N 5        // 每 5 次送一次 (4Hz)
```
- 設為 `1` → 每次都送 (20Hz，串口負擔重)
- 設為 `10` → 每 10 次送一次 (2Hz)

**或啟用「只在偵測時送音訊」模式：**
```cpp
#define SEND_AUDIO_ON_DETECT_ONLY
```
取消註解此行 → 只在 result=1 時送出 EVT0 封包

### ADC 通道
```cpp
#define ADC_CHANNEL ADC1_CHANNEL_6  // GPIO34
```
可用的 ADC1 通道：
- `ADC1_CHANNEL_0` = GPIO36
- `ADC1_CHANNEL_3` = GPIO39
- `ADC1_CHANNEL_4` = GPIO32
- `ADC1_CHANNEL_5` = GPIO33
- `ADC1_CHANNEL_6` = GPIO34（預設）
- `ADC1_CHANNEL_7` = GPIO35

## 分類器架構

### 階段1：簡單閾值分類器（當前實作）
```cpp
uint8_t classify_simple(float rms, float peak) {
  if (rms > THRESHOLD_RMS) {
    return 1;  // 偵測到目標撞擊
  }
  return 0;    // 未偵測
}
```

### 階段2：多特徵規則（預留介面）
```cpp
// 取消註解以啟用進階分類器
/*
uint8_t classify_advanced(float rms, float peak, float dominant_freq) {
  if (rms > THRESHOLD_RMS &&
      peak > THRESHOLD_PEAK &&
      dominant_freq > THRESHOLD_FREQ_MIN &&
      dominant_freq < THRESHOLD_FREQ_MAX) {
    return 1;
  }
  return 0;
}
*/
```

### 階段3：嵌入式機器學習
參考 [`../STAGE3_DEPLOYMENT.md`](../STAGE3_DEPLOYMENT.md)：
- 決策樹手動轉換
- Edge Impulse 平台部署
- TensorFlow Lite Micro 整合

## 動態參數調整

韌體支援透過序列命令動態調整閾值：

**協定：** `'T' (1 byte) + threshold (float, 4 bytes, little-endian)`

**Python 範例：**
```python
import serial
import struct

ser = serial.Serial('COM5', 921600)
new_threshold = 350.0
ser.write(b'T' + struct.pack('<f', new_threshold))
ser.close()
```

## 技術細節

### I2S-ADC 設定
- I2S 模式：`I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN`
- 位元深度：12-bit ADC (對齊到 16-bit)
- DMA 緩衝：6 個 × 256 樣本

### 訊號處理
1. **DC 校正：** 啟動時取 20 個 frame 平均值
2. **高通濾波：** 1階 IIR，截止頻率 ~20Hz（去除低頻漂移）
   ```cpp
   y = x - x_prev + 0.995*y_prev
   ```
3. **噪聲底線追蹤：** 指數移動平均（EMA）
   ```cpp
   noise_rms = 0.995*noise_rms + 0.005*current_rms
   ```

### 記憶體使用
- `chunk_buf`: 800 samples × 2 bytes = 1.6 KB
- `i2s_buf`: 256 samples × 2 bytes = 512 bytes
- 總 SRAM 使用：約 3 KB（ESP32 有 520 KB 可用）

### 處理時序
```
loop() 每 16ms (256 samples @ 16kHz):
├─ I2S 讀取：~1ms
├─ DC 校正 + 高通濾波：~0.5ms
└─ 累積到 chunk_buf

每 50ms (chunk_buf 滿):
├─ 計算 RMS/peak：~0.5ms
├─ 分類判斷：< 0.1ms
├─ 送出 RSLT：0.15ms
└─ [可選] 送出 EVT0：14ms
```

## 常見問題

### Q: 上傳失敗 "Connecting..."？
A:
1. 檢查 USB 連接
2. 按住 ESP32 的 BOOT 按鈕，點擊上傳，等待 "Connecting..." 出現後放開
3. 嘗試降低上傳速度（460800 或 115200）

### Q: 如何確認韌體是否正常運作？
A:
1. 上傳成功後，ESP32 會自動重啟
2. 使用 Python 工具 `real_time_classifier.py` 監控
3. 應該看到持續的 20Hz RSLT 封包

### Q: 可以同時使用 WiFi 嗎？
A:
可以，但需注意：
1. 必須使用 ADC1 通道（ADC2 與 WiFi 衝突）
2. WiFi 會增加功耗和處理負擔
3. 確保 WiFi 任務不會阻塞主迴圈

### Q: 如何提高判斷頻率到 50Hz？
A:
```cpp
#define CHUNK_MS 20  // 50Hz
```
但需注意：
- 每個 chunk 只有 320 samples（20ms @ 16kHz）
- 頻譜特徵解析度降低
- 考慮只送 RSLT 封包（註解掉 EVT0 傳送）

## 效能指標

| 指標 | 數值 |
|------|------|
| 判斷頻率 | 20Hz (50ms 週期) |
| 判斷延遲 | < 2ms |
| 取樣率 | 16 kHz |
| 音訊窗口 | 50ms (800 samples) |
| 最高頻率 | 8 kHz (Nyquist) |
| RSLT 傳輸 | 0.15ms |
| EVT0 傳輸 | 14ms |
| SRAM 使用 | ~3 KB |
| Flash 使用 | ~200 KB |

## 參考資料

- 完整系統文檔：[`../README_zh-TW.md`](../README_zh-TW.md)
- Python 工具說明：[`../impact_analysis/README.md`](../impact_analysis/README.md)
- 階段3部署指南：[`../STAGE3_DEPLOYMENT.md`](../STAGE3_DEPLOYMENT.md)
- ESP32 I2S-ADC 文檔：https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html
