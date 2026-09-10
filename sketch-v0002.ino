//========================================================================
//  XIAO ESP32-S3 Sense - On-Device Vision Trainer v2.0
//  ---------------------------------------------------------------
//  v2 adds:
//   - SSD1306 72x40 OLED (U8g2lib, HW I2C: SDA=GPIO5/D4, SCL=GPIO6/D5)
//       * scrollable menu with highlighted active item (5 visible lines)
//       * live black/white camera view while capturing / inferring
//       * training progress page (epoch/loss/acc/ETA + progress bar)
//       * inference OLED refresh throttled to every 10th frame
//   - Capacitive touch navigation (touchRead):
//       D0 (GPIO1/T1) = ACTIVATE / ENTER item
//       D1 (GPIO2/T2) = SCROLL UP
//       D2 (GPIO3/T3) = SCROLL DOWN
//       D3 (GPIO4/T4) = EXIT / BACK
//     NOTE: pins D6 (GPIO43 UART TX) and D7 (GPIO44 UART RX) have NO
//     touch channel; D8 (GPIO7) is the SD card SCK and is in use.
//     That is why navigation lives on D0-D3.
//   - Capture menu entries auto-shoot 10 training images per class:
//       CAP0 x10 / CAP1 x10 / CAP2 x10 -> /vision/train/class_{0,1,2}
//
//  Model: 12288 (64x64x3) -> 64 ReLU -> 3 softmax, float32 SGD.
//  Serial menu @115200 still fully functional in parallel.
//
//  Build FQBN (PSRAM is REQUIRED):
//      esp32:esp32:XIAO_ESP32S3:PSRAM=opi
//========================================================================

#include "esp_camera.h"
#include <SD.h>
#include <FS.h>
#include <SPI.h>
#include <Wire.h>
#include <U8g2lib.h>
#include <math.h>

// ---------------- XIAO ESP32-S3 Sense camera pins (B2B connector) ------
#define PWDN_GPIO_NUM  (-1)
#define RESET_GPIO_NUM (-1)
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

#define SD_CS_PIN       21          // Sense SD slot CS (shared with user LED, active LOW)

// ---------------- OLED + touch -----------------------------------------
// 72x40 SSD1306 on the default Wire pins (SDA=GPIO5/D4, SCL=GPIO6/D5).
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

#define TOUCH_ENTER   1   // D0 = GPIO1, TOUCH1  : activate / enter
#define TOUCH_UP      2   // D1 = GPIO2, TOUCH2  : scroll up
#define TOUCH_DOWN    3   // D2 = GPIO3, TOUCH3  : scroll down
#define TOUCH_BACK    4   // D3 = GPIO4, TOUCH4  : exit / back
#define TOUCH_PINS    4
static const int g_touchPin[TOUCH_PINS] = { TOUCH_ENTER, TOUCH_UP, TOUCH_DOWN, TOUCH_BACK };
static int  g_touchThr[TOUCH_PINS];
static bool g_touchPrev[TOUCH_PINS] = {false, false, false, false};
static uint32_t g_touchLockUntil = 0;

// ---------------- model / dataset shape ---------------------------------
#define IMG_W      64
#define IMG_H      64
#define IMG_CH     3
#define INPUT_N    (IMG_W * IMG_H * IMG_CH)   // 12288
#define HIDDEN_N   64
#define OUT_N      3
#define MAX_PER_CL 48
#define MAX_DATASET (MAX_PER_CL * OUT_N)      // 144 images x 12288 B

// ---------------- globals -------------------------------------------------
static float *W1 = NULL;
static float *b1 = NULL;
static float *W2 = NULL;
static float *b2 = NULL;
static bool   g_weightInit = false;
static uint32_t g_totalEpochs = 0;
static float  g_lastLoss = 0.0f;

static uint16_t g_epochs = 30;
static float    g_lr = 0.05f;

static uint8_t *g_trainData = NULL;
static uint8_t  g_trainLabels[MAX_DATASET];
static int      g_trainCount = 0;

static char g_classNames[OUT_N][24];
static bool g_sdOk = false;
static bool g_camOk = false;
static bool g_psramOk = false;

static float  g_xnorm[INPUT_N];
static float  g_hval[HIDDEN_N];
static float  g_y[OUT_N];
static float  g_dh[HIDDEN_N];
static uint8_t g_curImg[INPUT_N];       // latest 64x64x3 frame

// training progress shared with the OLED page
static volatile int   g_prEpoch = 0, g_prEpochs = 0;
static volatile float g_prLoss = 0.0f, g_prAcc = 0.0f;
static volatile uint32_t g_prEtaS = 0;
static volatile bool  g_prActive = false;

static uint32_t s_rng = 0x9E3779B9u;

// ---------------- OLED state -------------------------------------------------
enum OledMode { OLM_MENU, OLM_CAMERA, OLM_TRAIN, OLM_INFO };
typedef struct {
  const char *label;      // <=13 chars, fits 14-col 5x8 font at 72px minus margin
  void (*action)(void);
} MenuItem;

static OledMode g_oledMode = OLM_MENU;
static bool    g_oledDirty = true;
static int     g_menuCur = 0;
static int     g_menuTop = 0;
static int     g_menuCount = 0;
static const MenuItem *g_menuItems = NULL;

// static info page lines (4 visible lines on 40px @5x8)
static char g_infoLines[5][14];

// line(s) shown above the live B/W image during inference
static char  g_camInfo[14];
static int   g_camFps = 0;

// camera-sub-mode: what the OLED is currently showing while in OLM_CAMERA
enum CamShow { CSH_NONE, CSH_PREVIEW, CSH_INFER, CSH_CAPTURE };
static CamShow g_camShow = CSH_NONE;
static int    g_capProg = 0, g_capTotal = 0, g_capCls = -1;

// ---------------- helpers -----------------------------------------------------
static void putU16(uint8_t *b, uint16_t v) { b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8); }
static void putU32(uint8_t *b, uint32_t v) { b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8); b[2] = (uint8_t)(v >> 16); b[3] = (uint8_t)(v >> 24); }

static bool endsWithBmp(const char *n) {
  size_t l = strlen(n);
  return l >= 4 && n[l - 4] == '.' &&
         (n[l - 3] == 'b' || n[l - 3] == 'B') &&
         (n[l - 2] == 'm' || n[l - 2] == 'M') &&
         (n[l - 1] == 'p' || n[l - 1] == 'P');
}

static void trainDirPath(char *buf, size_t len, int cls) { snprintf(buf, len, "/vision/train/class_%d", cls); }
static void testDirPath(char *buf, size_t len, int cls)  { snprintf(buf, len, "/vision/test/class_%d", cls); }

static float frand(float lo, float hi) {
  s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
  return lo + ((float)(s_rng & 0xFFFF) / 65535.0f) * (hi - lo);
}

static void setLed(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }

static bool nnReady() { return W1 && b1 && W2 && b2; }

// mark OLED for the next poll tick
static inline void oledTouchRequest() { g_oledDirty = true; }

// ---------------- touch ----------------------------------------------------
static void touchCalibrate() {
  for (int i = 0; i < TOUCH_PINS; i++) {
    int best = touchRead(g_touchPin[i]);
    uint32_t t0 = millis();
    while (millis() - t0 < 250) {
      int v = touchRead(g_touchPin[i]);
      if (v < best) best = v;
      delay(2);
    }
    g_touchThr[i] = (int)(best * 0.62f) + 5;   // finger press drops raw value well below
    g_touchPrev[i] = false;
    Serial.printf("  touch D%d (GPIO%d): baseline=%d thr=%d\n",
                  i == 0 ? 0 : (i == 1 ? 1 : (i == 2 ? 2 : 3)),
                  g_touchPin[i], best, g_touchThr[i]);
  }
}

// Returns a mask of freshly-pressed keys (debounced, non-blocking, rate-limited).
static uint8_t pollTouchKeys() {
  if ((int32_t)(millis() - g_touchLockUntil) < 0) return 0;
  uint8_t mask = 0;
  for (int i = 0; i < TOUCH_PINS; i++) {
    bool pressed = touchRead(g_touchPin[i]) < g_touchThr[i];
    if (pressed && !g_touchPrev[i]) {
      mask |= (uint8_t)(1u << i);
      g_touchLockUntil = millis() + 220;   // debounce: ignore retriggers for 220ms
    }
    g_touchPrev[i] = pressed;
  }
  if (mask) oledTouchRequest();
  return mask;
}

static inline bool keyPressed(uint8_t mask, int idx) { return (mask & (1u << idx)) != 0; }

// ---------------- camera ------------------------------------------------
static bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("  camera init failed, err=0x%x\n", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_framesize(s, FRAMESIZE_QVGA);
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
  }
  return true;
}

static float pixGray(const uint8_t *img, size_t k) {
  return (3.0f * img[k] + 6.0f * img[k + 1] + 1.0f * img[k + 2]) / 10.0f;
}

// Area-average downsample of the camera frame to 64x64 RGB888 (stored in g_curImg).
static bool captureRGB888() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return false;
  const int sx = (int)fb->width, sy = (int)fb->height;
  const uint16_t *src = (const uint16_t *)fb->buf;
  uint8_t *out = g_curImg;
  for (int oy = 0; oy < IMG_H; oy++) {
    int y0 = (oy * sy) / IMG_H, y1 = ((oy + 1) * sy) / IMG_H;
    if (y1 <= y0) y1 = y0 + 1;
    for (int ox = 0; ox < IMG_W; ox++) {
      int x0 = (ox * sx) / IMG_W, x1 = ((ox + 1) * sx) / IMG_W;
      if (x1 <= x0) x1 = x0 + 1;
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int yy = y0; yy < y1; yy++) {
        const uint16_t *row = src + (size_t)yy * sx;
        for (int xx = x0; xx < x1; xx++) {
          uint16_t p = row[xx];
          r += (p >> 11) & 0x1F;
          g += (p >> 5) & 0x3F;
          b += p & 0x1F;
          n++;
        }
      }
      if (n == 0) n = 1;
      size_t k = ((size_t)oy * IMG_W + ox) * 3;
      out[k + 0] = (uint8_t)((r * 255) / (31 * n));
      out[k + 1] = (uint8_t)((g * 255) / (63 * n));
      out[k + 2] = (uint8_t)((b * 255) / (31 * n));
    }
  }
  esp_camera_fb_return(fb);
  return true;
}

static void previewASCII(const uint8_t *img) {
  const char ramp[] = " .:-=+*#%@";
  for (int oy = 0; oy < 16; oy++) {
    for (int ox = 0; ox < 16; ox++) {
      uint32_t g = 0; int n = 0;
      for (int dy = 0; dy < 4; dy++)
        for (int dx = 0; dx < 4; dx++) {
          size_t k = ((size_t)(oy * 4 + dy) * IMG_W + (ox * 4 + dx)) * 3;
          g += (uint32_t)pixGray(img, k);
          n++;
        }
      int idx = (int)(((uint32_t)(g / n) * 9) / 255);
      Serial.print(ramp[idx]);
    }
    Serial.println();
  }
}

// ---------------- BMP 64x64x24 IO -----------------------------------------
static bool writeBmp64(const char *path, const uint8_t *rgb) {
  uint8_t hdr[54] = {0};
  const uint32_t raw = IMG_W * IMG_H * 3;
  const uint32_t fsz = 54 + raw;
  hdr[0] = 'B'; hdr[1] = 'M';
  putU32(hdr + 2, fsz); putU32(hdr + 10, 54);
  putU32(hdr + 14, 40);
  putU32(hdr + 18, IMG_W); putU32(hdr + 22, IMG_H);
  putU16(hdr + 26, 1); putU16(hdr + 28, 24);
  putU32(hdr + 34, raw);
  File f = SD.open(path, FILE_WRITE);
  if (!f) return false;
  setLed(false);                      // release SD CS line we share with the LED
  f.write(hdr, 54);
  uint8_t line[IMG_W * 3];
  for (int y = IMG_H - 1; y >= 0; y--) {
    const uint8_t *row = rgb + (size_t)y * IMG_W * 3;
    for (int x = 0; x < IMG_W; x++) {
      line[x * 3 + 0] = row[x * 3 + 2];
      line[x * 3 + 1] = row[x * 3 + 1];
      line[x * 3 + 2] = row[x * 3 + 0];
    }
    f.write(line, sizeof(line));
  }
  f.close();
  return true;
}

static bool readBmp64(const char *path, uint8_t *rgb) {
  File f = SD.open(path, FILE_READ);
  if (!f) return false;
  if (f.size() < (int64_t)(54 + (uint32_t)IMG_W * IMG_H * 3)) { f.close(); return false; }
  uint8_t hdr[54];
  if (f.read(hdr, 54) != 54 || hdr[0] != 'B' || hdr[1] != 'M') { f.close(); return false; }
  uint32_t w = hdr[18] | (hdr[19] << 8) | ((uint32_t)hdr[20] << 16) | ((uint32_t)hdr[21] << 24);
  uint32_t h = hdr[22] | (hdr[23] << 8) | ((uint32_t)hdr[24] << 16) | ((uint32_t)hdr[25] << 24);
  if (w != IMG_W || h != IMG_H) { f.close(); return false; }
  uint8_t line[IMG_W * 3];
  for (int y = IMG_H - 1; y >= 0; y--) {
    if (f.read(line, sizeof(line)) != (int)sizeof(line)) { f.close(); return false; }
    uint8_t *row = rgb + (size_t)y * IMG_W * 3;
    for (int x = 0; x < IMG_W; x++) {
      row[x * 3 + 0] = line[x * 3 + 2];
      row[x * 3 + 1] = line[x * 3 + 1];
      row[x * 3 + 2] = line[x * 3 + 0];
    }
  }
  f.close();
  return true;
}

// ---------------- SD layout ----------------------------------------------
static void ensureTree() {
  SD.mkdir("/vision");
  SD.mkdir("/vision/train");
  SD.mkdir("/vision/test");
  SD.mkdir("/vision/models");
  for (int c = 0; c < OUT_N; c++) {
    char p[64];
    trainDirPath(p, sizeof(p), c); SD.mkdir(p);
    testDirPath(p, sizeof(p), c);  SD.mkdir(p);
  }
  for (int c = 0; c < OUT_N; c++) {
    char from[64], to[64];
    snprintf(from, sizeof(from), "/vision/class%d", c);
    trainDirPath(to, sizeof(to), c);
    if (SD.exists(from) && !SD.exists(to)) { SD.rename(from, to); }
  }
}

static int nextImgIndex(const char *folder) {
  for (int idx = 1; idx < 100000; idx++) {
    char p[96];
    snprintf(p, sizeof(p), "%s/img_%04d.bmp", folder, idx);
    if (!SD.exists(p)) return idx;
  }
  return -1;
}

static int countBmp(const char *folder) {
  int n = 0;
  File d = SD.open(folder);
  if (d) {
    File f = d.openNextFile();
    while (f) {
      if (!f.isDirectory() && endsWithBmp(f.name())) n++;
      f.close();
      f = d.openNextFile();
    }
    d.close();
  }
  return n;
}

static void rmRecursive(const char *path) {
  File r = SD.open(path);
  if (!r) return;
  File f = r.openNextFile();
  while (f) {
    if (f.isDirectory()) {
      char sub[96];
      snprintf(sub, sizeof(sub), "%s/%s", path, f.name());
      rmRecursive(sub);
    } else {
      char full[96];
      snprintf(full, sizeof(full), "%s/%s", path, f.name());
      SD.remove(full);
    }
    f.close();
    f = r.openNextFile();
  }
  r.close();
  SD.rmdir(path);
}

static void listTree(const char *path, int depth) {
  File d = SD.open(path);
  if (!d) { Serial.printf("  cannot open %s\n", path); return; }
  File f = d.openNextFile();
  while (f) {
    for (int i = 0; i < depth; i++) Serial.print("  ");
    if (f.isDirectory()) {
      Serial.printf("%s/\n", f.name());
      char sub[96];
      snprintf(sub, sizeof(sub), "%s/%s", path, f.name());
      listTree(sub, depth + 1);
    } else {
      Serial.printf("%s (%u bytes)\n", f.name(), (unsigned)f.size());
    }
    f.close();
    f = d.openNextFile();
  }
  d.close();
}

// ---------------- classes.txt --------------------------------------------
static void loadClasses() {
  for (int c = 0; c < OUT_N; c++) snprintf(g_classNames[c], sizeof(g_classNames[c]), "Class %d", c);
  File f = SD.open("/vision/classes.txt", FILE_READ);
  if (f) {
    for (int c = 0; c < OUT_N; c++) {
      String s = f.readStringUntil('\n');
      s.trim();
      if (s.length() > 0) {
        memset(g_classNames[c], 0, sizeof(g_classNames[c]));
        memcpy(g_classNames[c], s.c_str(), min((size_t)s.length(), sizeof(g_classNames[c]) - 1));
      }
    }
    f.close();
  }
}

static void saveClasses() {
  File f = SD.open("/vision/classes.txt", FILE_WRITE);
  if (f) {
    for (int c = 0; c < OUT_N; c++) { f.print(g_classNames[c]); f.print('\n'); }
    f.close();
    Serial.println("  classes.txt written");
  } else {
    Serial.println("  cannot write classes.txt");
  }
}

static int countBmp(const char *folder);
static int countBmpEx(int cls, bool train);
static int nModelCount();

// ---------------- dataset -------------------------------------------------
static bool allocDataset() {
  if (g_trainData) return true;
  if (!psramFound()) return false;
  g_trainData = (uint8_t *)ps_malloc((size_t)MAX_DATASET * INPUT_N);
  return g_trainData != NULL;
}

static int loadDataset() {
  g_trainCount = 0;
  for (int c = 0; c < OUT_N; c++) {
    char dir[64];
    trainDirPath(dir, sizeof(dir), c);
    File d = SD.open(dir);
    if (!d) continue;
    File f = d.openNextFile();
    while (f && g_trainCount < MAX_DATASET) {
      if (!f.isDirectory() && endsWithBmp(f.name())) {
        char full[96];
        snprintf(full, sizeof(full), "%s/%s", dir, f.name());
        if (readBmp64(full, g_trainData + (size_t)g_trainCount * INPUT_N)) {
          g_trainLabels[g_trainCount] = (uint8_t)c;
          g_trainCount++;
        }
      }
      f.close();
      f = d.openNextFile();
    }
    f.close();
    d.close();
  }
  return g_trainCount;
}

// ---------------- neural net ----------------------------------------------
static bool nnAlloc() {
  if (W1 && b1 && W2 && b2) return true;
  if (W1) free(W1); if (b1) free(b1); if (W2) free(W2); if (b2) free(b2);
  W1 = b1 = W2 = b2 = NULL;
  if (!psramFound()) return false;
  W1 = (float *)ps_malloc((size_t)INPUT_N * HIDDEN_N * sizeof(float));
  b1 = (float *)ps_malloc((size_t)HIDDEN_N * sizeof(float));
  W2 = (float *)ps_malloc((size_t)OUT_N * HIDDEN_N * sizeof(float));
  b2 = (float *)ps_malloc((size_t)OUT_N * sizeof(float));
  return W1 && b1 && W2 && b2;
}

static void nnInitWeights() {
  s_rng = esp_random();
  float k1 = sqrtf(6.0f / (float)(INPUT_N + HIDDEN_N));
  float k2 = sqrtf(6.0f / (float)(HIDDEN_N + OUT_N));
  for (int h = 0; h < HIDDEN_N; h++) {
    b1[h] = 0.0f;
    float *w = &W1[h * INPUT_N];
    for (int i = 0; i < INPUT_N; i++) w[i] = frand(-k1, k1);
  }
  for (int o = 0; o < OUT_N; o++) {
    b2[o] = 0.0f;
    float *w = &W2[o * HIDDEN_N];
    for (int h = 0; h < HIDDEN_N; h++) w[h] = frand(-k2, k2);
  }
  g_weightInit = true;
  g_totalEpochs = 0;
}

static inline void normImage(const uint8_t *img, float *x) {
  for (int i = 0; i < INPUT_N; i++) x[i] = img[i] * (1.0f / 255.0f);
}

static void forwardPass() {
  for (int h = 0; h < HIDDEN_N; h++) {
    float acc = b1[h];
    const float *w = &W1[h * INPUT_N];
    for (int i = 0; i < INPUT_N; i++) acc += w[i] * g_xnorm[i];
    g_hval[h] = acc > 0.0f ? acc : 0.0f;
  }
  float mx = -1e30f;
  for (int o = 0; o < OUT_N; o++) {
    float acc = b2[o];
    const float *w = &W2[o * HIDDEN_N];
    for (int h = 0; h < HIDDEN_N; h++) acc += w[h] * g_hval[h];
    g_y[o] = acc;
    if (acc > mx) mx = acc;
  }
  float s = 0.0f;
  for (int o = 0; o < OUT_N; o++) {
    g_y[o] = expf(g_y[o] - mx);
    s += g_y[o];
  }
  for (int o = 0; o < OUT_N; o++) g_y[o] /= s;
}

static int bestClass(float *probs) {
  forwardPass();
  int best = 0;
  for (int o = 1; o < OUT_N; o++) if (g_y[o] > g_y[best]) best = o;
  if (probs) memcpy(probs, g_y, sizeof(float) * OUT_N);
  return best;
}

static void trainStep(int label) {
  float deltaO[OUT_N];
  for (int o = 0; o < OUT_N; o++) deltaO[o] = g_y[o] - (o == label ? 1.0f : 0.0f);
  for (int h = 0; h < HIDDEN_N; h++) {
    float dh = 0.0f;
    for (int o = 0; o < OUT_N; o++) dh += W2[o * HIDDEN_N + h] * deltaO[o];
    g_dh[h] = dh * (g_hval[h] > 0.0f ? 1.0f : 0.0f);
  }
  for (int o = 0; o < OUT_N; o++) {
    b2[o] -= g_lr * deltaO[o];
    float *w = &W2[o * HIDDEN_N];
    for (int h = 0; h < HIDDEN_N; h++) w[h] -= g_lr * deltaO[o] * g_hval[h];
  }
  for (int h = 0; h < HIDDEN_N; h++) {
    b1[h] -= g_lr * g_dh[h];
    const float g = g_lr * g_dh[h];
    float *w = &W1[h * INPUT_N];
    for (int i = 0; i < INPUT_N; i++) w[i] -= g * g_xnorm[i];
  }
}

static void trainModelInPlace() {
  static uint16_t order[MAX_DATASET];
  const int n = g_trainCount;
  for (int i = 0; i < n; i++) order[i] = (uint16_t)i;

  uint32_t tStart = millis();
  double totalLoss = 0.0;
  for (int e = 0; e < (int)g_epochs; e++) {
    for (int i = n - 1; i > 0; i--) {
      int j = (int)(esp_random() % (uint32_t)(i + 1));
      uint16_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    float lossSum = 0.0f;
    int ok = 0;
    uint32_t e0 = millis();
    for (int s = 0; s < n; s++) {
      int idx = order[s];
      normImage(g_trainData + (size_t)idx * INPUT_N, g_xnorm);
      forwardPass();
      lossSum -= logf(g_y[g_trainLabels[idx]] + 1e-7f);
      int p = 0;
      for (int o = 1; o < OUT_N; o++) if (g_y[o] > g_y[p]) p = o;
      if (p == g_trainLabels[idx]) ok++;
      trainStep(g_trainLabels[idx]);
    }
    uint32_t eMs = millis() - e0;
    float avgLoss = lossSum / (float)n;
    totalLoss = avgLoss;
    float acc = 100.0f * ok / (float)n;
    uint32_t remaining = (uint32_t)(((double)eMs * (g_epochs - 1 - e)) / 1000.0);
    // publish state for the OLED training page
    g_prEpoch = e + 1; g_prEpochs = (int)g_epochs;
    g_prLoss = avgLoss; g_prAcc = acc; g_prEtaS = remaining;
    Serial.printf("  epoch %d/%d  loss=%.4f  acc=%.1f%%  %ums/epoch  ETA~%lus\n",
                  e + 1, (int)g_epochs, avgLoss, acc, (unsigned)eMs, (unsigned)remaining);
    oledTouchRequest();
    delay(10);
  }
  g_totalEpochs += g_epochs;
  g_lastLoss = (float)totalLoss;
}

// ---------------- model save / load --------------------------------------
static void putF32(uint8_t *b, float v) { memcpy(b, &v, 4); }

static bool saveModel() {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); return false; }
  if (!g_weightInit) { Serial.println("  no weights yet - train or load first"); return false; }
  int idx = 1;
  char path[96];
  while (idx < 10000) {
    snprintf(path, sizeof(path), "/vision/models/model_%04d.bin", idx);
    if (!SD.exists(path)) break;
    idx++;
  }
  if (idx >= 10000) { Serial.println("  too many models"); return false; }

  File f = SD.open(path, FILE_WRITE);
  if (!f) { Serial.println("  cannot create model file"); return false; }
  setLed(false);

  uint8_t hdr[32] = {0};
  memcpy(hdr, "XIAOMLP1", 8);
  putU32(hdr + 8, 1);
  putU16(hdr + 12, INPUT_N);
  putU16(hdr + 14, HIDDEN_N);
  putU16(hdr + 16, OUT_N);
  putU16(hdr + 18, 2);
  putU32(hdr + 20, g_totalEpochs);
  putF32(hdr + 24, g_lastLoss);
  f.write(hdr, 32);
  f.write((const uint8_t *)W1, (size_t)INPUT_N * HIDDEN_N * 4);
  f.write((const uint8_t *)b1, (size_t)HIDDEN_N * 4);
  f.write((const uint8_t *)W2, (size_t)OUT_N * HIDDEN_N * 4);
  f.write((const uint8_t *)b2, (size_t)OUT_N * 4);
  f.close();

  char info[96];
  snprintf(info, sizeof(info), "/vision/models/model_%04d.txt", idx);
  File t = SD.open(info, FILE_WRITE);
  if (t) {
    t.printf("arch: %d-%d-%d\nepochs: %u\nloss: %.4f\nclasses: %s | %s | %s\n",
             INPUT_N, HIDDEN_N, OUT_N, (unsigned)g_totalEpochs, g_lastLoss,
             g_classNames[0], g_classNames[1], g_classNames[2]);
    t.close();
  }
  double mb = (32.0 + ((double)INPUT_N * HIDDEN_N + HIDDEN_N + OUT_N * HIDDEN_N + OUT_N) * 4.0) / 1048576.0;
  Serial.printf("  model saved -> %s (%.2f MB)\n", path, mb);
  return true;
}

static int listModels() {
  int idx = 1, shown = 0;
  char path[96];
  while (idx < 10000) {
    snprintf(path, sizeof(path), "/vision/models/model_%04d.bin", idx);
    if (SD.exists(path)) {
      File f = SD.open(path);
      uint32_t sz = f ? (uint32_t)f.size() : 0;
      if (f) f.close();
      Serial.printf("  [%d] %s (%u bytes)\n", shown + 1, path, sz);
      shown++;
    }
    idx++;
  }
  if (shown == 0) Serial.println("  no models on SD yet");
  return shown;
}

static bool loadModel(int which) {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); return false; }
  char path[96];
  snprintf(path, sizeof(path), "/vision/models/model_%04d.bin", which);
  File f = SD.open(path, FILE_READ);
  if (!f) { Serial.println("  cannot open model file"); return false; }
  setLed(false);
  uint8_t hdr[32];
  if (f.read(hdr, 32) != 32 || memcmp(hdr, "XIAOMLP1", 8) != 0) {
    Serial.println("  bad model header");
    f.close();
    return false;
  }
  uint32_t ver = hdr[8] | ((uint32_t)hdr[9] << 8) | ((uint32_t)hdr[10] << 16) | ((uint32_t)hdr[11] << 24);
  uint16_t in = hdr[12] | ((uint16_t)hdr[13] << 8);
  uint16_t hid = hdr[14] | ((uint16_t)hdr[15] << 8);
  uint16_t out = hdr[16] | ((uint16_t)hdr[17] << 8);
  (void)ver;
  if (in != INPUT_N || hid != HIDDEN_N || out != OUT_N) {
    Serial.printf("  shape mismatch %d-%d-%d (firmware %d-%d-%d)\n", in, hid, out, INPUT_N, HIDDEN_N, OUT_N);
    f.close();
    return false;
  }
  if (f.read((uint8_t *)W1, (size_t)INPUT_N * HIDDEN_N * 4) != (int)(INPUT_N * HIDDEN_N * 4)) { f.close(); return false; }
  if (f.read((uint8_t *)b1, (size_t)HIDDEN_N * 4) != (int)(HIDDEN_N * 4)) { f.close(); return false; }
  if (f.read((uint8_t *)W2, (size_t)OUT_N * HIDDEN_N * 4) != (int)(OUT_N * HIDDEN_N * 4)) { f.close(); return false; }
  if (f.read((uint8_t *)b2, (size_t)OUT_N * 4) != (int)(OUT_N * 4)) { f.close(); return false; }
  f.close();
  g_totalEpochs = hdr[20] | ((uint32_t)hdr[21] << 8) | ((uint32_t)hdr[22] << 16) | ((uint32_t)hdr[23] << 24);
  memcpy(&g_lastLoss, hdr + 24, 4);
  g_weightInit = true;
  Serial.printf("  loaded %s  (~%u epochs, loss=%.4f)\n", path, (unsigned)g_totalEpochs, g_lastLoss);
  return true;
}

// =========================================================================
//  OLED rendering
// =========================================================================
// Screen: 72x40. Font: 5x8 -> 14 chars x 5 rows.
// g_curImg is the current 64x64 RGB888 frame.

static void oledRenderImage(int srcY0, int rows, int dstX, int dstY) {
  // draw dark pixels as lit (threshold ~110): outlines/scenes read clearly
  for (int y = 0; y < rows; y++) {
    int sy = srcY0 + y;
    size_t row = (size_t)sy * IMG_W;
    for (int x = 0; x < IMG_W; x++) {
      size_t k = (row + x) * 3;
      if (pixGray(g_curImg, k) < 110.0f) u8g2.drawPixel(dstX + x, dstY + y);
    }
  }
}

static void oledRenderMenu() {
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x8_tf);
    for (int r = 0; r < 5; r++) {
      int idx = g_menuTop + r;
      if (idx >= g_menuCount) break;
      bool act = (idx == g_menuCur);
      if (act) {
        u8g2.setDrawColor(1);
        u8g2.drawBox(0, r * 8, 72, 8);
        u8g2.setDrawColor(0);
        u8g2.drawStr(1, r * 8 + 7, g_menuItems[idx].label);
        u8g2.setDrawColor(1);
      } else {
        u8g2.drawStr(1, r * 8 + 7, g_menuItems[idx].label);
      }
    }
  } while (u8g2.nextPage());
}

static void oledRenderInfo() {
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x8_tf);
    for (int r = 0; r < 5; r++) {
      if (g_infoLines[r][0]) u8g2.drawStr(1, r * 8 + 7, g_infoLines[r]);
    }
  } while (u8g2.nextPage());
}

static void oledRenderTrain() {
  char a[14], b[14], c[14], d[14];
  snprintf(a, sizeof(a), "TRAIN %d/%d", g_prEpoch, g_prEpochs);
  snprintf(b, sizeof(b), "loss %.3f", g_prLoss);
  snprintf(c, sizeof(c), "acc %.1f%%", g_prAcc);
  snprintf(d, sizeof(d), "ETA ~%lus", (unsigned)g_prEtaS);
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x8_tf);
    u8g2.drawStr(1, 7, a);
    u8g2.drawStr(1, 15, b);
    u8g2.drawStr(1, 23, c);
    u8g2.drawStr(1, 31, d);
    // progress bar, bottom 8px
    int pct = (g_prEpoch <= 0) ? 0 : (int)(70L * g_prEpoch / g_prEpochs);
    u8g2.drawFrame(0, 33, 72, 6);
    if (pct > 1) u8g2.drawBox(1, 34, pct, 4);
  } while (u8g2.nextPage());
}

static void oledRenderCamera() {
  char fpline[14];
  snprintf(fpline, sizeof(fpline), "%s %d fps", g_camInfo, g_camFps);
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x8_tf);
    if (g_camShow == CSH_INFER) {
      // 2 text rows + 24-row image view (center of frame)
      u8g2.drawStr(1, 7, g_camInfo);
      u8g2.drawStr(1, 15, fpline);
      oledRenderImage(20, 24, (72 - IMG_W) / 2, 16);
    } else if (g_camShow == CSH_CAPTURE) {
      // shot counter row + 32-row image = 40px total
      char shot[14];
      snprintf(shot, sizeof(shot), "SHOT %d/%d C%d", g_capProg, g_capTotal, g_capCls);
      u8g2.drawStr(1, 7, shot);
      oledRenderImage(16, 32, (72 - IMG_W) / 2, 8);
    } else {
      // full-screen B/W preview: 64x40 center strip
      oledRenderImage(12, 40, (72 - IMG_W) / 2, 0);
    }
  } while (u8g2.nextPage());
}

static void oledUpdate() {
  if (!g_oledDirty) return;
  switch (g_oledMode) {
    case OLM_MENU:   oledRenderMenu(); break;
    case OLM_TRAIN:  oledRenderTrain(); break;
    case OLM_CAMERA: oledRenderCamera(); break;
    case OLM_INFO:   oledRenderInfo(); break;
  }
  g_oledDirty = false;
}

static void oledEnterMenu() {
  g_oledMode = OLM_MENU;
  g_camShow = CSH_NONE;
  oledTouchRequest();
}

static void oledSetInfo(const char *l0, const char *l1, const char *l2, const char *l3, const char *l4) {
  memset(g_infoLines, 0, sizeof(g_infoLines));
  const char *ls[5] = { l0, l1, l2, l3, l4 };
  for (int i = 0; i < 5; i++)
    if (ls[i]) { strncpy(g_infoLines[i], ls[i], 13); g_infoLines[i][13] = 0; }
  g_oledMode = OLM_INFO;
  oledTouchRequest();
}

// =========================================================================
//  Serial + input helpers
// =========================================================================
static String readLine(uint32_t timeoutMs) {
  String s = "";
  uint32_t t0 = millis();
  while (millis() - t0 < timeoutMs) {
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n') return s;
      if (c != '\r') s += c;
    }
    if (pollTouchKeys()) return s;      // any touch aborts a long serial prompt
    delay(2);
  }
  return s;
}

static long promptInt(const char *prompt, long lo, long hi) {
  while (true) {
    Serial.print(prompt);
    String s = readLine(10000);
    long v = s.toInt();
    if (s.length() > 0 && v >= lo && v <= hi) return v;
    Serial.printf("  please enter %d..%d\n", lo, hi);
  }
}

static bool confirmAction(const char *what) {
  Serial.printf("  type YES to %s: ", what);
  String s = readLine(10000);
  s.toUpperCase();
  s.trim();
  return s == "YES";
}

// =========================================================================
//  ML / capture action flows (shared by OLED touch menu + serial menu)
// =========================================================================
static bool enteredFromTouch = false;

static void captureRGB888Auto(int cls, bool testSet, int count) {
  if (!g_sdOk) { Serial.println("  no SD card"); return; }
  if (!g_camOk) { Serial.println("  camera unavailable"); return; }
  char dir[64];
  if (testSet) testDirPath(dir, sizeof(dir), cls); else trainDirPath(dir, sizeof(dir), cls);
  g_oledMode = OLM_CAMERA;
  g_camShow = CSH_CAPTURE;
  g_capCls = cls; g_capTotal = count;
  for (int k = 0; k < count; k++) {
    if (!captureRGB888()) { Serial.println("  capture failed"); break; }
    int idx = nextImgIndex(dir);
    if (idx < 0) { Serial.println("  folder full"); break; }
    char path[96];
    snprintf(path, sizeof(path), "%s/img_%04d.bmp", dir, idx);
    if (!writeBmp64(path, g_curImg)) { Serial.println("  SD write failed"); break; }
    g_capProg = k + 1;
    oledTouchRequest();
    Serial.printf("  saved %s\n", path);
    if (count > 1 && k < count - 1) {
      // 3s gap between shots so the user can change the object pose
      uint32_t t0 = millis();
      while (millis() - t0 < 3000) {
        uint8_t m = pollTouchKeys();
        if (keyPressed(m, 3)) {          // D3 = skip remaining shots
          Serial.println("  sequence aborted by D3");
          oledEnterMenu();
          return;
        }
        oledUpdate();
        delay(2);
      }
    }
  }
  Serial.printf("  --- %s class %d [%s]: %d shots ---\n",
                testSet ? "test" : "train", cls, g_classNames[cls], g_capProg);
  previewASCII(g_curImg);
  oledEnterMenu();
}

static void streamCamera(bool withInference) {
  if (!g_camOk) { Serial.println("  camera unavailable"); oledEnterMenu(); return; }
  if (withInference && !nnReady()) { Serial.println("  MLP weights unavailable (PSRAM?)"); oledEnterMenu(); return; }
  if (withInference && !g_weightInit) { Serial.println("  no model - train (5) or load (9) first"); oledEnterMenu(); return; }

  g_oledMode = OLM_CAMERA;
  g_camShow = withInference ? CSH_INFER : CSH_PREVIEW;
  uint32_t frameCount = 0;
  uint32_t fpsWindow = millis();
  uint32_t lastSerial = 0;
  g_oledDirty = true;

  while (true) {
    if (!captureRGB888()) { delay(5); continue; }
    frameCount++;

    if (withInference) {
      normImage(g_curImg, g_xnorm);
      float probs[OUT_N];
      int best = bestClass(probs);
      float fps = 0.0f;
      if (millis() - fpsWindow >= 1000) {
        fps = 1000.0f * frameCount / (float)(millis() - fpsWindow);
        fpsWindow = millis();
        frameCount = 0;
      }
      if (millis() - lastSerial >= 500) {
        lastSerial = millis();
        Serial.printf("  [live] %s %.1f%%   fps=%.1f  (D3 to exit)\n",
                      g_classNames[best], 100.0f * probs[best], fps);
      }
      // throttle OLED: draw every 10th inference frame
      if (frameCount % 10 == 1) {
        snprintf(g_camInfo, sizeof(g_camInfo), "%s %.0f%%",
                 g_classNames[best], 100.0f * probs[best]);
        g_camInfo[12] = 0;                       // force 13-char max
        g_camFps = (int)fps;
        g_oledDirty = true;
        oledUpdate();
      }
    } else {
      if (millis() - fpsWindow >= 200) {   // ~5 fps preview refresh
        fpsWindow = millis();
        g_oledDirty = true;
        oledUpdate();
      }
    }

    uint8_t m = pollTouchKeys();
    if (keyPressed(m, 3)) {                 // D3 = exit streaming
      Serial.println("  stream stopped");
      break;
    }
    delay(1);
  }
  oledEnterMenu();
}

static void streamPreview()  { streamCamera(false); }
static void streamInference(){ streamCamera(true); }

static void trainFlow() {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); oledEnterMenu(); return; }
  if (!g_sdOk) { Serial.println("  no SD card - cannot read dataset"); oledEnterMenu(); return; }
  if (!g_weightInit) {
    Serial.println("  fresh random weights (Glorot init)");
    nnInitWeights();
  }
  if (!allocDataset()) { Serial.println("  not enough PSRAM for dataset"); oledEnterMenu(); return; }
  int n = loadDataset();
  if (n == 0) {
    Serial.println("  no training images - capture some first (menu 1)");
    oledEnterMenu();
    return;
  }
  int perCl[OUT_N] = {0};
  for (int i = 0; i < n; i++) perCl[g_trainLabels[i]]++;
  Serial.printf("  dataset: %d images  (", n);
  for (int c = 0; c < OUT_N; c++) Serial.printf("%s:%d ", g_classNames[c], perCl[c]);
  Serial.printf(")\n  epochs=%d lr=%.3f  (change via serial menu B)\n", (int)g_epochs, g_lr);
  Serial.println("  training...");

  g_prActive = true;
  g_oledMode = OLM_TRAIN;
  g_oledDirty = true;
  uint32_t t0 = millis();
  trainModelInPlace();
  uint32_t secs = (millis() - t0) / 1000;
  g_prActive = false;
  Serial.printf("  --- training done in %lus, final loss=%.4f, total epochs=%u ---\n",
                (unsigned)secs, g_lastLoss, (unsigned)g_totalEpochs);

  char l[5][14];
  snprintf(l[0], 14, "DONE in %lus", (unsigned)secs);
  snprintf(l[1], 14, "loss %.4f", g_lastLoss);
  snprintf(l[2], 14, "epochs %u", (unsigned)g_totalEpochs);
  snprintf(l[3], 14, "D0 save / D3 end");
  snprintf(l[4], 14, "");
  oledSetInfo(l[0], l[1], l[2], l[3], l[4]);
  g_oledDirty = true;

  // give the user a moment on OLED: D0 = save now, D3 = skip
  uint32_t t1 = millis();
  while (millis() - t1 < 20000) {
    uint8_t m = pollTouchKeys();
    if (keyPressed(m, 0)) { saveModel(); oledEnterMenu(); break; }
    if (keyPressed(m, 3)) { Serial.println("  skip save"); oledEnterMenu(); break; }
    oledUpdate();
    delay(2);
  }
  if (g_oledMode == OLM_INFO) {
    if (enteredFromTouch) {
      oledEnterMenu();                       // timeout on OLED path: back to menu
    } else {
      if (confirmAction("save the model to SD now")) saveModel();
      oledEnterMenu();
    }
  }
}

static void evalTestSet() {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); oledEnterMenu(); return; }
  if (!g_weightInit) { Serial.println("  no model - train (5) or load (9) first"); oledEnterMenu(); return; }
  int cm[OUT_N][OUT_N] = {{0}};
  int perClassTotal[OUT_N] = {0};
  int ok = 0, total = 0;
  for (int c = 0; c < OUT_N; c++) {
    char dir[64];
    testDirPath(dir, sizeof(dir), c);
    File d = SD.open(dir);
    if (!d) continue;
    File f = d.openNextFile();
    while (f) {
      if (!f.isDirectory() && endsWithBmp(f.name())) {
        char full[96];
        snprintf(full, sizeof(full), "%s/%s", dir, f.name());
        if (readBmp64(full, g_curImg)) {
          normImage(g_curImg, g_xnorm);
          int p = bestClass(NULL);
          cm[c][p]++;
          perClassTotal[c]++;
          total++;
          if (p == c) ok++;
        }
      }
      f.close();
      f = d.openNextFile();
    }
    d.close();
  }
  if (total == 0) { Serial.println("  no test images under /vision/test"); oledEnterMenu(); return; }
  Serial.println("  confusion matrix (rows=true class, cols=predicted):");
  for (int c = 0; c < OUT_N; c++) {
    Serial.printf("  %s: ", g_classNames[c]);
    for (int p = 0; p < OUT_N; p++) Serial.printf("%d ", cm[c][p]);
    Serial.printf("(n=%d)\n", perClassTotal[c]);
  }
  Serial.printf("  overall accuracy: %.1f%% (%d/%d)\n", 100.0f * ok / total, ok, total);
  char l[5][14];
  snprintf(l[0], 14, "ACC %.1f%%", 100.0f * ok / total);
  snprintf(l[1], 14, "%d/%d right", ok, total);
  snprintf(l[2], 14, "");
  snprintf(l[3], 14, "");
  snprintf(l[4], 14, "D3 to continue");
  oledSetInfo(l[0], l[1], l[2], l[3], l[4]);
  uint32_t t1 = millis();
  while (millis() - t1 < 10000) {
    uint8_t m = pollTouchKeys();
    if (keyPressed(m, 3)) break;
    oledUpdate();
    delay(2);
  }
  oledEnterMenu();
}

static void loadNewestModel() {
  if (!g_sdOk) { Serial.println("  no SD card"); oledEnterMenu(); return; }
  int idx = 1, found = 0;
  while (idx < 10000) {
    char p[96];
    snprintf(p, sizeof(p), "/vision/models/model_%04d.bin", idx);
    if (SD.exists(p)) found = idx;
    idx++;
  }
  if (found == 0) {
    Serial.println("  no models on SD - train & save first");
    oledEnterMenu();
    return;
  }
  loadModel(found);
  oledEnterMenu();
}

static void datasetSummary() {
  Serial.println("  dataset:");
  for (int c = 0; c < OUT_N; c++) {
    char dir[64];
    trainDirPath(dir, sizeof(dir), c);
    Serial.printf("    train/class_%d [%s]: %d images\n", c, g_classNames[c], countBmp(dir));
  }
  for (int c = 0; c < OUT_N; c++) {
    char dir[64];
    testDirPath(dir, sizeof(dir), c);
    Serial.printf("    test/class_%d [%s]: %d images\n", c, g_classNames[c], countBmp(dir));
  }
  Serial.println("  models on SD: ");
  listModels();
  if (nnReady() && g_weightInit)
    Serial.printf("  ram model: initialized (%u epochs trained, last loss=%.4f)\n", (unsigned)g_totalEpochs, g_lastLoss);
  else
    Serial.println("  ram model: NOT initialized");
  char l[5][14];
  snprintf(l[0], sizeof(l[0]), "T0 %d", countBmpEx(0, true));
  snprintf(l[1], sizeof(l[1]), "T1 %d", countBmpEx(1, true));
  snprintf(l[2], sizeof(l[2]), "T2 %d", countBmpEx(2, true));
  snprintf(l[3], sizeof(l[3]), "test sum %d", countBmpEx(0, false) + countBmpEx(1, false) + countBmpEx(2, false));
  snprintf(l[4], sizeof(l[4]), "D3 to continue");
  oledSetInfo(l[0], l[1], l[2], l[3], l[4]);
  uint32_t t1 = millis();
  while (millis() - t1 < 10000) {
    uint8_t m = pollTouchKeys();
    if (keyPressed(m, 3)) break;
    oledUpdate();
    delay(2);
  }
  oledEnterMenu();
}

// =========================================================================
//  OLED menu definition
// =========================================================================
static void actCap0() { captureRGB888Auto(0, false, 10); }
static void actCap1() { captureRGB888Auto(1, false, 10); }
static void actCap2() { captureRGB888Auto(2, false, 10); }
static void actPreview() { streamPreview(); }
static void actTrain() { trainFlow(); }
static void actInfer() { streamInference(); }
static void actEval() { evalTestSet(); }
static void actSave() { saveModel(); oledEnterMenu(); }
static void actLoad() { loadNewestModel(); }
static void actInfo() {
  char l[5][14];
  snprintf(l[0], 14, "T0 %d/%d/%d", countBmpEx(0, true), countBmpEx(1, true), countBmpEx(2, true));
  snprintf(l[1], 14, "T1 %d/%d/%d", countBmpEx(0, false), countBmpEx(1, false), countBmpEx(2, false));
  snprintf(l[2], sizeof(l[2]), "%s", nModelCount() ? "MDL yes" : "MDL none");
  snprintf(l[3], 14, "RAM %s", (nnReady() && g_weightInit) ? "trained" : "none");
  snprintf(l[4], 14, "D3 to continue");
  oledSetInfo(l[0], l[1], l[2], l[3], l[4]);
  uint32_t t1 = millis();
  while (millis() - t1 < 10000) {
    uint8_t m = pollTouchKeys();
    if (keyPressed(m, 3)) break;
    oledUpdate();
    delay(2);
  }
  oledEnterMenu();
}
static void actAbout() {
  char l[5][14];
  snprintf(l[0], 14, "mx+b 64x64x3");
  snprintf(l[1], 14, "act D0 up D1");
  snprintf(l[2], 14, "dn D2 bak D3");
  snprintf(l[3], 14, "OLED 72x40");
  snprintf(l[4], 14, "D3 to continue");
  oledSetInfo(l[0], l[1], l[2], l[3], l[4]);
  uint32_t t1 = millis();
  while (millis() - t1 < 8000) {
    uint8_t m = pollTouchKeys();
    if (keyPressed(m, 3)) break;
    oledUpdate();
    delay(2);
  }
  oledEnterMenu();
}

// simplified info needed by actInfo above
static int countBmpEx(int cls, bool train) {
  char dir[64];
  if (train) trainDirPath(dir, sizeof(dir), cls); else testDirPath(dir, sizeof(dir), cls);
  return countBmp(dir);
}
static int nModelCount() {
  int idx = 1, n = 0;
  char p[96];
  while (idx < 10000) {
    snprintf(p, sizeof(p), "/vision/models/model_%04d.bin", idx);
    if (SD.exists(p)) n++;
    idx++;
  }
  return n;
}

static const MenuItem g_oledMenu[] = {
  { "CAP0 x10", actCap0 },
  { "CAP1 x10", actCap1 },
  { "CAP2 x10", actCap2 },
  { "PREVIEW",  actPreview },
  { "TRAIN",    actTrain },
  { "INFER LIVE", actInfer },
  { "TEST EVAL",  actEval },
  { "SAVE MODEL", actSave },
  { "LOAD NEW",   actLoad },
  { "DATA INFO",  actInfo },
  { "ABOUT",      actAbout },
};

static void oledSetupMenu() {
  g_menuItems = g_oledMenu;
  g_menuCount = (int)(sizeof(g_oledMenu) / sizeof(g_oledMenu[0]));
  g_menuCur = 0;
  g_menuTop = 0;
  oledEnterMenu();
  oledUpdate();
}

// touch-driven menu navigation
static void oledMenuNavigate(uint8_t keys) {
  if (g_oledMode != OLM_MENU) {
    // RUNNING an item: only D3 (back) handled inside the loops; nothing here
    return;
  }
  if (keyPressed(keys, 1) && g_menuCur > 0) {           // D1 = up
    g_menuCur--;
    if (g_menuCur < g_menuTop) g_menuTop = g_menuCur;
    oledTouchRequest();
  } else if (keyPressed(keys, 2) && g_menuCur < g_menuCount - 1) {  // D2 = down
    g_menuCur++;
    if (g_menuCur >= g_menuTop + 5) g_menuTop = g_menuCur - 4;
    oledTouchRequest();
  } else if (keyPressed(keys, 0)) {                     // D0 = activate / enter
    Serial.printf("  [OLED-menu] %s\n", g_menuItems[g_menuCur].label);
    enteredFromTouch = true;
    oledUpdate();
    g_menuItems[g_menuCur].action();
    enteredFromTouch = false;
    oledEnterMenu();
  } else if (keyPressed(keys, 3)) {                     // D3 = back (idle = nothing)
    // nothing to back out of at top level
  }
}

// =========================================================================
//  Serial menu
// =========================================================================
static void printMenu() {
  Serial.println();
  Serial.println("===== XIAO ESP32-S3 Sense - On-device Vision v2.0 =====");
  Serial.println(" touch: D0=enter  D1=up  D2=down  D3=back  |  OLED menu mirrors this");
  Serial.println(" 1  capture 10 TRAIN imgs (pick class 0-2)");
  Serial.println(" 2  capture 1 TEST img (pick class 0-2)");
  Serial.println(" 3  camera ASCII preview");
  Serial.println(" 4  dataset summary");
  Serial.println(" 5  TRAIN model on device");
  Serial.println(" 6  live inference (camera, D3 to stop)");
  Serial.println(" 7  evaluate on test set");
  Serial.println(" 8  save model to SD  (/vision/models/model_XXXX.bin)");
  Serial.println(" 9  load model from SD");
  Serial.println(" A  list files on SD (/vision tree)");
  Serial.println(" B  hyperparameters (epochs / learning rate)");
  Serial.println(" C  class names");
  Serial.println(" D  delete data (train / test / models)");
  Serial.println(" E  touch raw value demo");
  Serial.println(" 0  print this menu");
  Serial.println("========================================================");
  Serial.print("> ");
}

static void captureTrainFlow() {
  int cls = (int)promptInt("  which class (0..2): ", 0, OUT_N - 1);
  int cnt = (int)promptInt("  how many images (1..20): ", 1, 20);
  Serial.printf("  show class %d [%s] to the camera...\n", cls, g_classNames[cls]);
  captureRGB888Auto(cls, false, cnt);
}

static void captureTestFlow() {
  int cls = (int)promptInt("  which class (0..2): ", 0, OUT_N - 1);
  captureRGB888Auto(cls, true, 1);
}

static void hyperparamFlow() {
  Serial.println("  [B] hyperparameters");
  Serial.println("    1  epochs (1..500)     2  learning rate (0.001..1.0)     3 back");
  long ch = promptInt("    > ", 1, 3);
  if (ch == 1) {
    g_epochs = (uint16_t)promptInt("    epochs (1..500): ", 1, 500);
  } else if (ch == 2) {
    for (int attempt = 0; attempt < 10; attempt++) {
      Serial.print("    learning rate (0.001..1.0): ");
      String s = readLine(10000);
      if (s.length() > 0) {
        float v = s.toFloat();
        if (v >= 0.001f && v <= 1.0f) { g_lr = v; break; }
      }
      Serial.println("    invalid - try again");
    }
    delay(10);
  }
  Serial.printf("  -> epochs=%d lr=%.3f\n", (int)g_epochs, g_lr);
}

static void classNamesFlow() {
  for (int c = 0; c < OUT_N; c++) Serial.printf("  [%d] %s\n", c, g_classNames[c]);
  int c = (int)promptInt("  edit which class (0..2): ", 0, OUT_N - 1);
  Serial.printf("  new name for class %d (max 23 chars): ", c);
  String s = readLine(10000);
  s.trim();
  if (s.length() > 0 && s.length() <= 23) {
    memset(g_classNames[c], 0, sizeof(g_classNames[c]));
    memcpy(g_classNames[c], s.c_str(), s.length());
    saveClasses();
  } else {
    Serial.println("  name unchanged");
  }
}

static void deleteFlow() {
  Serial.println("  [D] delete");
  Serial.println("    1 train dataset   2 test dataset   3 all models   4 train+test+models   5 back");
  long ch = promptInt("    > ", 1, 5);
  if (ch == 5) return;
  const char *target = NULL;
  if (ch == 1) target = "the TRAIN dataset (/vision/train)";
  else if (ch == 2) target = "the TEST dataset (/vision/test)";
  else if (ch == 3) target = "ALL models (/vision/models)";
  else target = "ALL data (train + test + models)";
  if (!confirmAction(target)) { Serial.println("  cancelled"); return; }
  if (ch == 1 || ch == 4) { rmRecursive("/vision/train"); Serial.println("  /vision/train removed"); }
  if (ch == 2 || ch == 4) { rmRecursive("/vision/test"); Serial.println("  /vision/test removed"); }
  if (ch == 3 || ch == 4) { rmRecursive("/vision/models"); Serial.println("  /vision/models removed"); }
  ensureTree();
  Serial.println("  folders re-created");
}

static void touchDemo() {
  Serial.println("  touch raw values for 10s (touch pads with finger):");
  uint32_t t0 = millis();
  while (millis() - t0 < 10000) {
    Serial.printf("  D0=%d D1=%d D2=%d D3=%d\n",
                  touchRead(TOUCH_ENTER), touchRead(TOUCH_UP),
                  touchRead(TOUCH_DOWN), touchRead(TOUCH_BACK));
    delay(400);
  }
}

static void dispatchMenu(String line) {
  line.trim();
  if (line.length() == 0) return;
  char c = line[0];
  if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
  switch (c) {
    case '1': captureTrainFlow(); break;
    case '2': captureTestFlow(); break;
    case '3':
      if (g_camOk && captureRGB888()) previewASCII(g_curImg);
      else Serial.println("  camera unavailable");
      break;
    case '4': datasetSummary(); break;
    case '5': trainFlow(); break;
    case '6': streamInference(); break;
    case '7': evalTestSet(); break;
    case '8': saveModel(); break;
    case '9': {
      int shown = listModels();
      if (shown > 0) {
        long w = promptInt("  load which model #: ", 1, shown);
        loadModel((int)w);
      }
      break;
    }
    case 'a': listTree("/vision", 0); break;
    case 'b': hyperparamFlow(); break;
    case 'c': classNamesFlow(); break;
    case 'd': deleteFlow(); break;
    case 'e': touchDemo(); break;
    case '0': printMenu(); break;
    default: Serial.printf("  unknown command '%c' - menu 0\n", c);
  }
  Serial.println();
  Serial.print("> ");
}

// =========================================================================
//  setup / loop
// =========================================================================
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  setLed(true);
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  g_psramOk = psramFound();

  Serial.println();
  Serial.println("XIAO ESP32-S3 Sense - On-Device Vision Trainer v2.0");
  Serial.println("build: esp32:esp32:XIAO_ESP32S3:PSRAM=opi (PSRAM required)");
  Serial.printf("PSRAM: %s\n", g_psramOk ? "OK" : "MISSING - camera/model will fail");

  if (g_psramOk && !nnAlloc()) Serial.println("FATAL: could not allocate MLP weights in PSRAM");
  if (!allocDataset()) Serial.println("warning: dataset buffer unavailable");

  g_camOk = initCamera();
  Serial.printf("camera: %s\n", g_camOk ? "OK" : "FAILED");

  // OLED on Wire (SDA=GPIO5/D4, SCL=GPIO6/D5)
  u8g2.begin();
  u8g2.setFont(u8g2_font_5x8_tf);
  Serial.printf("OLED: 72x40 SSD1306 %s\n", u8g2.getBufferPtr() ? "OK" : "FAILED");

  if (SD.begin(SD_CS_PIN)) {
    g_sdOk = true;
    Serial.printf("SD: OK - card type %d, size %llu MB\n",
                  (int)SD.cardType(), (unsigned long long)(SD.cardSize() / (1024ULL * 1024ULL)));
    ensureTree();
    loadClasses();
  } else {
    Serial.println("SD: FAILED - insert FAT32 microSD and reset");
  }
  (void)SPI; // SPI not used directly; SD() binds internally

  touchCalibrate();
  oledSetupMenu();
  setLed(false);
  printMenu();
}

void loop() {
  uint8_t keys = pollTouchKeys();
  if (keys) oledMenuNavigate(keys);

  oledUpdate();

  if (Serial.available()) {
    String line = readLine(10000);
    if (line.length() > 0) {
      oledEnterMenu();          // any serial command bounces OLED back to menu
      dispatchMenu(line);
    }
  }
  delay(2);
}
// deerflow_build_id=20260910T041811Z-9ce6a778f8ab-01584f311f99
