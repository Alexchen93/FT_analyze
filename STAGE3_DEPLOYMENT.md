# 階段3：模型部署到 ESP32 (TinyML)

本文檔說明如何將訓練好的分類器部署到 ESP32 進行邊緣推理。

## 部署方案比較

| 方案 | 複雜度 | 模型大小 | 推理速度 | 適用場景 |
|------|--------|----------|----------|----------|
| 手動閾值 | ⭐ 極低 | < 1KB | < 0.1ms | 單特徵/多特徵規則 |
| 決策樹 | ⭐⭐ 低 | < 10KB | < 1ms | 多特徵分類 |
| 隨機森林 | ⭐⭐⭐ 中 | 10-50KB | 1-5ms | 複雜特徵組合 |
| SVM | ⭐⭐⭐ 中 | 5-20KB | 1-3ms | 小數據集高維分類 |
| 神經網路 | ⭐⭐⭐⭐ 高 | 10-100KB | 5-20ms | 需要非線性決策邊界 |

## 方案1：手動閾值/規則（最簡單）

### 步驟
1. 在階段2找出最佳閾值
2. 直接修改韌體中的 `THRESHOLD_RMS` 或 `classify_simple()` 函式

### 範例
```cpp
// ImpactCapture.ino

// 單特徵閾值
float THRESHOLD_RMS = 350.0f;  // 從訓練數據找出

uint8_t classify_simple(float rms, float peak) {
  return (rms > THRESHOLD_RMS) ? 1 : 0;
}
```

```cpp
// 多特徵規則
float THRESHOLD_RMS = 350.0f;
float THRESHOLD_PEAK = 0.6f;
float THRESHOLD_FREQ_MIN = 2000.0f;
float THRESHOLD_FREQ_MAX = 5000.0f;

uint8_t classify_advanced(float rms, float peak, float dominant_freq) {
  if (rms > THRESHOLD_RMS &&
      peak > THRESHOLD_PEAK &&
      dominant_freq > THRESHOLD_FREQ_MIN &&
      dominant_freq < THRESHOLD_FREQ_MAX) {
    return 1;
  }
  return 0;
}
```

## 方案2：決策樹（推薦入門 TinyML）

### 工具
使用 `sklearn` 訓練後手動轉換為 C 代碼。

### 步驟
1. 訓練決策樹
```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
```

2. 匯出為 C 代碼
```python
from sklearn.tree import export_text

tree_rules = export_text(clf, feature_names=feature_names)
print(tree_rules)
```

3. 手動轉換為 ESP32 函式
```cpp
uint8_t classify_decision_tree(float rms, float peak, float centroid, float crest) {
  if (rms <= 300.0) {
    return 0;
  } else {
    if (centroid <= 3000.0) {
      if (peak <= 0.5) {
        return 0;
      } else {
        return 1;
      }
    } else {
      return 1;
    }
  }
}
```

## 方案3：Edge Impulse（推薦完整工作流程）

[Edge Impulse](https://www.edgeimpulse.com/) 是專為嵌入式設備設計的機器學習平台，支援 ESP32。

### 優點
- 自動處理數據預處理
- 支援多種模型（神經網路、SVM、隨機森林等）
- 一鍵部署到 ESP32
- 內建特徵提取（MFCC、頻譜等）

### 步驟

#### 1. 準備數據
將收集的 WAV 檔案分類到兩個資料夾：
```
training_data/
├── positive/
│   ├── chunk_*.wav
└── negative/
    ├── chunk_*.wav
```

#### 2. 上傳到 Edge Impulse
- 註冊 [Edge Impulse](https://studio.edgeimpulse.com/)
- 建立新專案
- 上傳 WAV 檔案並標記標籤

#### 3. 設計 Impulse
```
[時域音訊] → [MFCC/頻譜] → [神經網路/分類器] → [輸出 0/1]
```

#### 4. 訓練模型
- 調整超參數
- 訓練並驗證準確率

#### 5. 部署到 ESP32
- 下載 Arduino 函式庫（.zip）
- 在 Arduino IDE 安裝函式庫
- 修改 `ImpactCapture.ino` 整合推理

#### 6. 整合範例
```cpp
#include <your_project_inferencing.h>

// 在 process_chunk() 中
void process_chunk() {
  // ... 計算特徵 ...

  // 準備推理輸入
  signal_t signal;
  signal.total_length = CHUNK_SAMPLES;
  signal.get_data = &get_signal_data;

  ei_impulse_result_t result;
  run_classifier(&signal, &result, false);

  // 取得分類結果
  uint8_t classification = (result.classification[1].value > 0.5) ? 1 : 0;

  // 輸出結果
  serial_send_result(millis(), classification, rms, peak);
}
```

## 方案4：TensorFlow Lite Micro（進階）

適合需要自定義神經網路架構的場景。

### 步驟

#### 1. 訓練 Keras 模型
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

#### 2. 轉換為 TFLite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 3. 轉換為 C 陣列
```bash
xxd -i model.tflite > model_data.cc
```

#### 4. 整合到 ESP32
```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h"

// 初始化 TFLite
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
interpreter.AllocateTensors();

// 推理
TfLiteTensor* input = interpreter.input(0);
// 填入特徵...
interpreter.Invoke();
TfLiteTensor* output = interpreter.output(0);
uint8_t result = (output->data.f[0] > 0.5) ? 1 : 0;
```

## 性能評估

### 記憶體需求（ESP32 限制）
- 可用 SRAM: ~520KB
- 可用 Flash: ~4MB

### 推理時間（50ms 週期內完成）
- 簡單閾值: < 0.1ms ✅
- 決策樹: < 1ms ✅
- 隨機森林 (10棵樹): ~2ms ✅
- 神經網路 (2層, 16+8神經元): ~5ms ✅
- 複雜 CNN: 10-50ms ⚠️

## 建議流程

1. **階段2結束後**：檢視模型準確率和特徵重要性
2. **選擇方案**：
   - 準確率 > 95% 且單特徵足夠 → 方案1（手動閾值）
   - 需要 2-3 個特徵組合 → 方案2（決策樹）
   - 複雜非線性邊界 → 方案3（Edge Impulse）
3. **測試**：在 ESP32 上實測推理時間和準確率
4. **優化**：調整模型大小或特徵數量

## 除錯技巧

### 驗證推理結果
在 ESP32 同時輸出特徵值和判斷結果：
```cpp
Serial.printf("RMS=%.2f, Peak=%.2f, Result=%d\n", rms, peak, result);
```

### 對比 Python 和 ESP32 結果
1. 儲存同一筆音訊的特徵
2. 在 Python 端用訓練好的模型推理
3. 比對兩者結果是否一致

### 性能分析
```cpp
uint32_t start = micros();
uint8_t result = classify(...);
uint32_t elapsed = micros() - start;
Serial.printf("Inference time: %u us\n", elapsed);
```

## 參考資源

- [Edge Impulse 文檔](https://docs.edgeimpulse.com/)
- [TensorFlow Lite Micro for ESP32](https://github.com/espressif/esp-tflite-micro)
- [Arduino Library for Edge Impulse](https://docs.edgeimpulse.com/docs/deployment/running-your-impulse-arduino)
