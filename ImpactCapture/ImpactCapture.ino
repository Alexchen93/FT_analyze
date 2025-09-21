// ESP32 + MAX9814 Impact Event Capture (Arduino IDE)
// 串口輸出為二進位封包："EVT0" + sample_rate(uint32) + sample_count(uint32) + int16_le[count]
// 接線：MAX9814 OUT -> GPIO34 (ADC1_CHANNEL_6)，VCC->3V3，GND->GND

#include <Arduino.h>
#include <driver/i2s.h>
#include <driver/adc.h>
#include <math.h>
#include <stdint.h>

#define I2S_PORT         I2S_NUM_0
#define SAMPLE_RATE      16000
#define FRAME_SAMPLES    256
#define ADC_CHANNEL      ADC1_CHANNEL_6   // GPIO34
#define SERIAL_BAUD      921600

// 事件擷取窗設定
#define PRE_MS           100
#define POST_MS          300
#define MAX_EVENT_MS     600

static int16_t i2s_buf[FRAME_SAMPLES];
static int16_t s16_frame[FRAME_SAMPLES];

// 推導大小
const int PRE_SAMPLES        = SAMPLE_RATE * PRE_MS / 1000;
const int POST_SAMPLES       = SAMPLE_RATE * POST_MS / 1000;
const int MAX_EVENT_SAMPLES  = SAMPLE_RATE * MAX_EVENT_MS / 1000;

// Ring buffer（2 的次方長度）
#define RING_SIZE 16384
static int16_t ring_buf[RING_SIZE];
static uint32_t ring_w = 0;

// 事件暫存
#define EVENT_BUF_SIZE (PRE_SAMPLES + MAX_EVENT_SAMPLES + POST_SAMPLES + FRAME_SAMPLES)
static int16_t event_buf[EVENT_BUF_SIZE];
static int event_idx = 0;

// DC 校正與一階高通濾波（去低頻/緩慢漂移）
static float dc_offset = 2048.0f;   // 啟動後校正
static float hp_prev_x = 0.0f, hp_prev_y = 0.0f;
const  float HP_R = 0.995f; // 約 20 Hz @ 16 kHz

// 噪聲底線（RMS）追蹤
static float noise_rms = 20.0f;
const  float NF_ALPHA  = 0.995f;

// 偵測門檻（依現場可調）
const float MIN_RMS_START   = 300.0f; // 太低不觸發
const float START_RATIO     = 3.5f;   // 啟動門檻：rms > noise_rms * ratio
const float CONT_RATIO      = 1.2f;   // 持續門檻：rms > noise_rms * ratio
const int   MIN_EVENT_MS    = 50;

static bool capturing = false;
static int  silent_frames = 0;

inline float dc_block(float x) {
  float y = x - hp_prev_x + HP_R * hp_prev_y;
  hp_prev_x = x;
  hp_prev_y = y;
  return y;
}

void i2s_adc_init() {
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);

  i2s_config_t cfg; memset(&cfg, 0, sizeof(cfg));
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
  i2s_set_sample_rates(I2S_PORT, SAMPLE_RATE); // 再次確保取樣率
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

inline void ring_push(int16_t s) {
  ring_buf[ring_w & (RING_SIZE - 1)] = s;
  ring_w++;
}

inline void copy_pre_from_ring(int count) {
  if (count > RING_SIZE) count = RING_SIZE;
  uint32_t start = (ring_w - (uint32_t)count) & (RING_SIZE - 1);
  for (int i = 0; i < count; ++i) {
    event_buf[event_idx++] = ring_buf[(start + i) & (RING_SIZE - 1)];
  }
}

inline void serial_send_event(int16_t* data, uint32_t count) {
  const char hdr[4] = {'E','V','T','0'};
  uint32_t sr = SAMPLE_RATE;
  Serial.write((const uint8_t*)hdr, 4);
  Serial.write((const uint8_t*)&sr, 4);
  Serial.write((const uint8_t*)&count, 4);
  Serial.write((const uint8_t*)data, count * sizeof(int16_t));
}

inline void serial_send_frame(const int16_t* data, uint32_t count) {
  const char hdr[4] = {'F','R','M','0'};
  uint32_t sr = SAMPLE_RATE;
  Serial.write((const uint8_t*)hdr, 4);
  Serial.write((const uint8_t*)&sr, 4);
  Serial.write((const uint8_t*)&count, 4);
  Serial.write((const uint8_t*)data, count * sizeof(int16_t));
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  i2s_adc_init();
  calibrate_dc();
  // 注意：不要使用 Serial.println()，以免破壞二進位封包
}

void loop() {
  size_t br = 0;
  i2s_read(I2S_PORT, (void*)i2s_buf, sizeof(i2s_buf), &br, portMAX_DELAY);
  int n = br / 2;

  double sumsq = 0.0;

  for (int i = 0; i < n; ++i) {
    uint16_t v12u = (uint16_t)i2s_buf[i] & 0x0FFF;
    float x = (float)v12u - dc_offset;    // 去 DC 偏置
    float y = dc_block(x);                // 一階高通

    float s = y * 16.0f;                  // 12-bit -> 16-bit 對齊
    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    int16_t s16 = (int16_t)s;
    s16_frame[i] = s16;
    ring_push(s16);
    if (capturing) {
      if (event_idx < EVENT_BUF_SIZE) {
        event_buf[event_idx++] = s16;
      }
    }
    sumsq += (double)y * (double)y;
  }

  // 串流當前 frame（僅在事件期間）
  if (capturing && n > 0) {
    serial_send_frame(s16_frame, (uint32_t)n);
  }

  float rms = sqrtf((float)(sumsq / (double)n));
  noise_rms = NF_ALPHA * noise_rms + (1.0f - NF_ALPHA) * rms;
  float ratio = (noise_rms > 1.0f) ? (rms / noise_rms) : 9999.0f;

  const float frame_ms = 1000.0f * (float)FRAME_SAMPLES / (float)SAMPLE_RATE;
  const int   post_frames_target = (int)(POST_MS / frame_ms + 0.5f);
  const int   min_event_frames   = (int)(MIN_EVENT_MS / frame_ms + 0.5f);

  if (!capturing) {
    if (rms > MIN_RMS_START && ratio > START_RATIO) {
      capturing = true;
      event_idx = 0;
      silent_frames = 0;
      copy_pre_from_ring(PRE_SAMPLES);
    }
  } else {
    if (ratio > CONT_RATIO) {
      silent_frames = 0;
    } else {
      silent_frames++;
    }

    bool enough_length = (event_idx >= (PRE_SAMPLES + min_event_frames * FRAME_SAMPLES));
    bool long_silence  = (silent_frames >= post_frames_target);
    bool too_long      = (event_idx >= (PRE_SAMPLES + MAX_EVENT_SAMPLES));

    if ((enough_length && long_silence) || too_long) {
      int count = event_idx;
      if (count > 0) {
        serial_send_event(event_buf, (uint32_t)count);
      }
      capturing = false;
      event_idx = 0;
      silent_frames = 0;
    }
  }
}

// 若需改用其它 ADC1 腳位，請對照：
// ADC1_CHANNEL_0=GPIO36, 1=GPIO37, 2=GPIO38, 3=GPIO39,
// 4=GPIO32, 5=GPIO33, 6=GPIO34, 7=GPIO35
