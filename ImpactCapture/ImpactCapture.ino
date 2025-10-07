// ESP32 + MAX9814 Real-time Impact Classifier (20Hz)
// 三階段架構：
// 階段1: 固定週期 RMS 判斷 + 雙協定輸出
// 階段2: Python 端收集數據訓練模型
// 階段3: 嵌入式模型部署（TinyML）
//
// 接線：MAX9814 OUT -> GPIO34 (ADC1_CH6), VCC->3V3, GND->GND
// 序列協定：921600 baud
//   - RSLT: 輕量判斷結果封包（每 50ms，17 bytes）
//   - EVT0: 完整音訊封包（可配置頻率）

#include <Arduino.h>
#include <driver/i2s.h>
#include <driver/adc.h>
#include <math.h>
#include <stdint.h>

// ============ 基本設定 ============
#define I2S_PORT         I2S_NUM_0
#define SAMPLE_RATE      16000          // 取樣率 (Hz)
#define ADC_CHANNEL      ADC1_CHANNEL_6 // GPIO34
#define SERIAL_BAUD      921600

// ============ 判斷週期設定 ============
#define CHUNK_MS         50             // 50ms = 20Hz 判斷頻率
#define CHUNK_SAMPLES    (SAMPLE_RATE * CHUNK_MS / 1000)  // 800 samples

// ============ 分類器參數 ============
// 階段1: 簡單閾值分類器
float THRESHOLD_RMS = 300.0f;           // 可透過 Serial 動態調整

// 階段3: 預留多特徵分類器空間
// float THRESHOLD_PEAK = 0.5f;
// float THRESHOLD_FREQ_MIN = 2000.0f;
// float THRESHOLD_FREQ_MAX = 5000.0f;

// ============ 輸出控制 ============
#define SEND_AUDIO_EVERY_N  5           // 每 N 次送一次完整音訊 (5次 = 4Hz)
// #define SEND_AUDIO_ON_DETECT_ONLY    // 取消註解：只在 result=1 時送音訊

// ============ I2S 設定 ============
#define FRAME_SAMPLES    256            // 每次 I2S 讀取大小
static int16_t i2s_buf[FRAME_SAMPLES];
static int16_t chunk_buf[CHUNK_SAMPLES];
static int chunk_idx = 0;

// ============ DC 校正與高通濾波 ============
static float dc_offset = 2048.0f;
static float hp_prev_x = 0.0f, hp_prev_y = 0.0f;
const  float HP_R = 0.995f;             // 一階高通 ~20Hz @ 16kHz

// ============ 噪聲底線追蹤 ============
static float noise_rms = 20.0f;
const  float NF_ALPHA = 0.995f;

// ============ 計數器 ============
static uint32_t chunk_counter = 0;
static uint32_t millis_start = 0;

// ============ 函式宣告 ============
inline float dc_block(float x) {
  float y = x - hp_prev_x + HP_R * hp_prev_y;
  hp_prev_x = x;
  hp_prev_y = y;
  return y;
}

void i2s_adc_init() {
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);

  i2s_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB);
  cfg.intr_alloc_flags = 0;
  cfg.dma_buf_count = 6;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  i2s_driver_install(I2S_PORT, &cfg, 0, nullptr);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);
  i2s_adc_enable(I2S_PORT);
  i2s_set_sample_rates(I2S_PORT, SAMPLE_RATE);
}

void calibrate_dc(int frames = 20) {
  size_t br = 0;
  int64_t sum = 0;
  int total = 0;
  for (int k = 0; k < frames; ++k) {
    i2s_read(I2S_PORT, (void*)i2s_buf, sizeof(i2s_buf), &br, portMAX_DELAY);
    int n = br / 2;
    for (int i = 0; i < n; ++i) {
      uint16_t v = (uint16_t)i2s_buf[i] & 0x0FFF;
      sum += v;
    }
    total += n;
  }
  if (total > 0) dc_offset = (float)sum / (float)total;
}

// ============ 序列協定 ============
// RSLT 封包: "RSLT" + timestamp_ms(4) + result(1) + rms(4) + peak(4) = 17 bytes
void serial_send_result(uint32_t timestamp_ms, uint8_t result, float rms, float peak) {
  const char hdr[4] = {'R', 'S', 'L', 'T'};
  Serial.write((const uint8_t*)hdr, 4);
  Serial.write((const uint8_t*)&timestamp_ms, 4);
  Serial.write((const uint8_t*)&result, 1);
  Serial.write((const uint8_t*)&rms, 4);
  Serial.write((const uint8_t*)&peak, 4);
}

// EVT0 封包: "EVT0" + sample_rate(4) + sample_count(4) + result(1) + int16[]
void serial_send_audio(const int16_t* data, uint32_t count, uint8_t result) {
  const char hdr[4] = {'E', 'V', 'T', '0'};
  uint32_t sr = SAMPLE_RATE;
  Serial.write((const uint8_t*)hdr, 4);
  Serial.write((const uint8_t*)&sr, 4);
  Serial.write((const uint8_t*)&count, 4);
  Serial.write((const uint8_t*)&result, 1);
  Serial.write((const uint8_t*)data, count * sizeof(int16_t));
}

// ============ 階段1: 簡單 RMS 分類器 ============
uint8_t classify_simple(float rms, float peak) {
  // 簡單閾值判斷
  if (rms > THRESHOLD_RMS) {
    return 1;
  }
  return 0;
}

// ============ 階段3: 多特徵分類器（預留） ============
// uint8_t classify_advanced(float rms, float peak, float dominant_freq) {
//   if (rms > THRESHOLD_RMS &&
//       peak > THRESHOLD_PEAK &&
//       dominant_freq > THRESHOLD_FREQ_MIN &&
//       dominant_freq < THRESHOLD_FREQ_MAX) {
//     return 1;
//   }
//   return 0;
// }

// ============ 處理完整 chunk ============
void process_chunk() {
  // 計算特徵
  double sumsq = 0.0;
  float peak = 0.0f;

  for (int i = 0; i < CHUNK_SAMPLES; ++i) {
    float val = (float)chunk_buf[i] / 32768.0f;
    sumsq += (double)val * (double)val;
    float absval = fabsf(val);
    if (absval > peak) peak = absval;
  }

  float rms = sqrtf((float)(sumsq / (double)CHUNK_SAMPLES));

  // 更新噪聲底線
  noise_rms = NF_ALPHA * noise_rms + (1.0f - NF_ALPHA) * rms;

  // 分類判斷
  uint8_t result = classify_simple(rms, peak);

  // 取得時間戳（從啟動後經過的毫秒數）
  uint32_t timestamp_ms = millis() - millis_start;

  // 送出 RSLT 封包（每次都送，0.15ms）
  serial_send_result(timestamp_ms, result, rms, peak);

  // 決定是否送出完整音訊
  bool send_audio = false;

  #ifdef SEND_AUDIO_ON_DETECT_ONLY
    if (result == 1) send_audio = true;
  #else
    if (chunk_counter % SEND_AUDIO_EVERY_N == 0) send_audio = true;
  #endif

  if (send_audio) {
    serial_send_audio(chunk_buf, CHUNK_SAMPLES, result);
  }

  chunk_counter++;
}

// ============ 設定與主迴圈 ============
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  i2s_adc_init();
  calibrate_dc();

  millis_start = millis();
  chunk_idx = 0;
  chunk_counter = 0;
}

void loop() {
  size_t br = 0;
  i2s_read(I2S_PORT, (void*)i2s_buf, sizeof(i2s_buf), &br, portMAX_DELAY);
  int n = br / 2;

  // 處理並累積到 chunk_buf
  for (int i = 0; i < n; ++i) {
    uint16_t v12u = (uint16_t)i2s_buf[i] & 0x0FFF;
    float x = (float)v12u - dc_offset;
    float y = dc_block(x);
    float s = y * 16.0f;  // 12-bit -> 16-bit 對齊

    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;

    chunk_buf[chunk_idx++] = (int16_t)s;

    // 當累積滿一個 chunk (50ms = 800 samples)
    if (chunk_idx >= CHUNK_SAMPLES) {
      process_chunk();
      chunk_idx = 0;
    }
  }

  // 檢查是否有 Serial 命令（階段2: 動態調整閾值）
  if (Serial.available() > 0) {
    // 簡單協定: "T" + float (設定新的 RMS 閾值)
    uint8_t cmd = Serial.read();
    if (cmd == 'T' && Serial.available() >= 4) {
      Serial.readBytes((char*)&THRESHOLD_RMS, 4);
    }
  }
}

// 若需改用其它 ADC1 腳位，請對照：
// ADC1_CHANNEL_0=GPIO36, 1=GPIO37, 2=GPIO38, 3=GPIO39,
// 4=GPIO32, 5=GPIO33, 6=GPIO34, 7=GPIO35
