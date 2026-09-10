//========================================================================
//  XIAO ESP32-S3 Sense - On-Device Vision Trainer
//  ---------------------------------------------------------------
//  - Captures 64x64 RGB images from the onboard camera
//  - Stores BMP images in sensibly named SD folders under /vision
//  - Trains a fully-connected neural net (MLP) ON DEVICE:
//      12,288 (64x64x3) -> 64 ReLU -> 3 (softmax)
//    float32 SGD + cross-entropy, no cloud / no PC training
//  - Model binary saved to SD: /vision/models/model_XXXX.bin
//  - Serial menu system @ 115200 baud
//
//  Build FQBN (PSRAM is REQUIRED):
//      esp32:esp32:XIAO_ESP32S3:PSRAM=opi
//========================================================================

#include "esp_camera.h"
#include <SD.h>
#include <FS.h>
#include <SPI.h>
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

#define SD_CS_PIN       21          // Sense SD slot chip select (same pin as user LED)
#define LED_BUILTIN_NEG 21          // active LOW

// ---------------- Model shape ----------------------------------------
#define IMG_W      64
#define IMG_H      64
#define IMG_CH     3
#define INPUT_N    (IMG_W * IMG_H * IMG_CH)   // 12288
#define HIDDEN_N   64
#define OUT_N      3
#define MAX_PER_CL 48
#define MAX_DATASET (MAX_PER_CL * OUT_N)      // 144 images x 12288 bytes

// ---------------- Globals ----------------------------------------------
static float *W1 = NULL;   // [HIDDEN][INPUT]  in PSRAM (~3.0 MB float32)
static float *b1 = NULL;   // [HIDDEN]
static float *W2 = NULL;   // [OUT][HIDDEN]
static float *b2 = NULL;   // [OUT]
static bool   g_weightInit = false;
static uint32_t g_totalEpochs = 0;
static float  g_lastLoss = 0.0f;

static uint16_t g_epochs = 30;
static float    g_lr = 0.05f;

static uint8_t *g_trainData = NULL;   // raw 64x64x3 pixels, PSRAM
static uint8_t  g_trainLabels[MAX_DATASET];
static int      g_trainCount = 0;

static char g_classNames[OUT_N][24];
static bool g_sdOk = false;
static bool g_camOk = false;
static bool g_psramOk = false;

// Scratch (internal SRAM is fine at these sizes)
static float  g_xnorm[INPUT_N];       // 49 KB
static float  g_hval[HIDDEN_N];
static float  g_y[OUT_N];
static float  g_dh[HIDDEN_N];
static uint8_t g_curImg[INPUT_N];     // 12 KB latest captured image

static uint32_t s_rng = 0x9E3779B9u;

// ---------------- helpers ----------------------------------------------
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

static void setLed(bool on) { digitalWrite(LED_BUILTIN_NEG, on ? LOW : HIGH); }

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
  config.frame_size = FRAMESIZE_QVGA;      // 320x240 RGB565, then box->64x64
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;                // mandatory field, unused for RGB565
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

// Area-average downsample of the camera frame to 64x64 RGB888.
static bool captureRGB888(uint8_t *out) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return false;
  const int sx = (int)fb->width, sy = (int)fb->height;
  const uint16_t *src = (const uint16_t *)fb->buf;
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
          g += (3 * img[k] + 6 * img[k + 1] + img[k + 2]) / 10;
          n++;
        }
      int idx = (int)((uint32_t)(g / n) * 9 / 255);
      Serial.print(ramp[idx]);
    }
    Serial.println();
  }
}

// ---------------- BMP 64x64x24 IO ---------------------------------------
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
  f.write(hdr, 54);
  uint8_t line[IMG_W * 3];
  for (int y = IMG_H - 1; y >= 0; y--) {            // BMP rows are bottom-up
    const uint8_t *row = rgb + (size_t)y * IMG_W * 3;
    for (int x = 0; x < IMG_W; x++) {
      line[x * 3 + 0] = row[x * 3 + 2];             // B
      line[x * 3 + 1] = row[x * 3 + 1];             // G
      line[x * 3 + 2] = row[x * 3 + 0];             // R
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

// ---------------- SD layout ---------------------------------------------
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
  // legacy migration: pre-v2 folder name -> v2 name
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

// ---------------- classes.txt labels ------------------------------------
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
    int perClass = 0;
    while (f && g_trainCount < MAX_DATASET) {
      if (!f.isDirectory() && endsWithBmp(f.name())) {
        char full[96];
        snprintf(full, sizeof(full), "%s/%s", dir, f.name());
        if (readBmp64(full, g_trainData + (size_t)g_trainCount * INPUT_N)) {
          g_trainLabels[g_trainCount] = (uint8_t)c;
          g_trainCount++;
          perClass++;
        }
      }
      f.close();
      f = d.openNextFile();
    }
    f.close();
    d.close();
    (void)perClass;
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

// SGD step: assumes forwardPass() already ran for the same sample.
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
    uint32_t elapsedMin = (millis() - tStart) / 60000;
    uint32_t remaining = (uint32_t)(((double)eMs * (g_epochs - 1 - e)) / 1000.0);
    Serial.printf("  epoch %d/%d  loss=%.4f  acc=%.1f%%  %ums/epoch  elapsed=%lum%lus  ETA~%lus\n",
                  e + 1, (int)g_epochs, avgLoss, acc, (unsigned)eMs,
                  (unsigned)(elapsedMin), (unsigned)((millis() - tStart) / 1000 % 60), (unsigned)remaining);
    delay(10);   // let the serial monitor breathe
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

  uint8_t hdr[32] = {0};
  memcpy(hdr, "XIAOMLP1", 8);
  putU32(hdr + 8, 1);          // format version
  putU16(hdr + 12, INPUT_N);
  putU16(hdr + 14, HIDDEN_N);
  putU16(hdr + 16, OUT_N);
  putU16(hdr + 18, 2);         // flags: 2 = trained
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

// ---------------- menu actions ---------------------------------------------
static String readLine(uint32_t timeoutMs) {
  String s = "";
  uint32_t t0 = millis();
  while (millis() - t0 < timeoutMs) {
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n') return s;
      if (c != '\r') s += c;
    }
    delay(2);
  }
  return s;
}

static long promptInt(const char *prompt, long lo, long hi) {
  while (true) {
    Serial.print(prompt);
    String s = readLine(20000);
    long v = s.toInt();
    if (s.length() > 0 && v >= lo && v <= hi) return v;
    Serial.printf("  please enter %d..%d\n", lo, hi);
  }
}

static bool confirmAction(const char *what) {
  Serial.printf("  type YES to %s: ", what);
  String s = readLine(20000);
  s.toUpperCase();
  s.trim();
  return s == "YES";
}

static void captureToClass(int cls, bool testSet, int count) {
  if (!g_sdOk) { Serial.println("  no SD card"); return; }
  if (!g_camOk) { Serial.println("  camera unavailable"); return; }
  char dir[64];
  if (testSet) testDirPath(dir, sizeof(dir), cls); else trainDirPath(dir, sizeof(dir), cls);
  for (int k = 0; k < count; k++) {
    int idx = nextImgIndex(dir);
    if (idx < 0) { Serial.println("  folder full"); return; }
    char path[96];
    snprintf(path, sizeof(path), "%s/img_%04d.bmp", dir, idx);
    if (!captureRGB888(g_curImg)) { Serial.println("  capture failed"); return; }
    if (!writeBmp64(path, g_curImg)) { Serial.println("  SD write failed"); return; }
    Serial.printf("  saved %s\n", path);
    if (k < count - 1) delay(600);
  }
  Serial.printf("  --- %s (%s) ---\n", testSet ? "test" : "train", g_classNames[cls]);
  previewASCII(g_curImg);
}

static void liveInference() {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); return; }
  if (!g_camOk) { Serial.println("  camera unavailable"); return; }
  if (!g_weightInit) { Serial.println("  no model - train (5) or load (9) first"); return; }
  if (!captureRGB888(g_curImg)) { Serial.println("  capture failed"); return; }
  previewASCII(g_curImg);
  normImage(g_curImg, g_xnorm);
  float probs[OUT_N];
  int best = bestClass(probs);
  Serial.printf("  -> class %d [%s]: %.2f%%\n", best, g_classNames[best], 100.0f * probs[best]);
  // runner-ups
  int order[OUT_N];
  for (int i = 0; i < OUT_N; i++) order[i] = i;
  for (int i = 0; i < OUT_N - 1; i++)
    for (int j = i + 1; j < OUT_N; j++)
      if (probs[order[j]] > probs[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
  Serial.print("  top-3: ");
  for (int i = 0; i < OUT_N; i++)
    Serial.printf("%s=%.0f%%  ", g_classNames[order[i]], 100.0f * probs[order[i]]);
  Serial.println();
}

static void evalTestSet() {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); return; }
  if (!g_weightInit) { Serial.println("  no model - train (5) or load (9) first"); return; }
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
  if (total == 0) { Serial.println("  no test images under /vision/test"); return; }
  Serial.println("  confusion matrix (rows=true class, cols=predicted):");
  for (int c = 0; c < OUT_N; c++) {
    Serial.printf("  %s: ", g_classNames[c]);
    for (int p = 0; p < OUT_N; p++) Serial.printf("%d ", cm[c][p]);
    Serial.printf("(n=%d)\n", perClassTotal[c]);
  }
  Serial.printf("  overall accuracy: %.1f%% (%d/%d)\n", 100.0f * ok / total, ok, total);
}

static bool nnReady() { return W1 && b1 && W2 && b2; }

static void printMenu() {
  Serial.println();
  Serial.println("================ XIAO ESP32-S3 Sense - On-device Vision ================");
  Serial.println(" 1  capture -> TRAIN (pick class 0-2, then count)");
  Serial.println(" 2  capture -> TEST  (pick class 0-2)");
  Serial.println(" 3  camera ASCII preview");
  Serial.println(" 4  dataset summary");
  Serial.println(" 5  TRAIN model on device");
  Serial.println(" 6  live inference (camera)");
  Serial.println(" 7  evaluate on test set");
  Serial.println(" 8  save model to SD  (/vision/models/model_XXXX.bin)");
  Serial.println(" 9  load model from SD");
  Serial.println(" A  list files on SD (/vision tree)");
  Serial.println(" B  hyperparameters (epochs / learning rate)");
  Serial.println(" C  class names");
  Serial.println(" D  delete data (train / test / models)");
  Serial.println(" 0  print this menu");
  Serial.println("========================================================================");
  Serial.print("> ");
}

static void captureTrainFlow() {
  int cls = (int)promptInt("  which class (0..2): ", 0, OUT_N - 1);
  int cnt = (int)promptInt("  how many images (1..20): ", 1, 20);
  Serial.printf("  show class %d [%s] to the camera...\n", cls, g_classNames[cls]);
  captureToClass(cls, false, cnt);
}

static void captureTestFlow() {
  int cls = (int)promptInt("  which class (0..2): ", 0, OUT_N - 1);
  captureToClass(cls, true, 1);
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
}

static void trainFlow() {
  if (!nnReady()) { Serial.println("  MLP weights unavailable (PSRAM missing?)"); return; }
  if (!g_sdOk) { Serial.println("  no SD card - cannot read dataset"); return; }
  if (!g_weightInit) {
    Serial.println("  fresh random weights (Glorot init)");
    nnInitWeights();
  }
  if (!allocDataset()) { Serial.println("  not enough PSRAM for dataset"); return; }
  int n = loadDataset();
  if (n == 0) { Serial.println("  no training images - capture some first (menu 1)"); return; }
  int perCl[OUT_N] = {0};
  for (int i = 0; i < n; i++) perCl[g_trainLabels[i]]++;
  Serial.printf("  dataset: %d images  (", n);
  for (int c = 0; c < OUT_N; c++) Serial.printf("%s:%d ", g_classNames[c], perCl[c]);
  Serial.printf(")\n  epochs=%d lr=%.3f  (change via menu B)\n", (int)g_epochs, g_lr);
  Serial.println("  training...");
  uint32_t t0 = millis();
  trainModelInPlace();
  uint32_t secs = (millis() - t0) / 1000;
  Serial.printf("  --- training done in %lus, final loss=%.4f, total epochs=%u ---\n",
                (unsigned)secs, g_lastLoss, (unsigned)g_totalEpochs);
  if (confirmAction("save the model to SD now")) saveModel();
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
      String s = readLine(20000);
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
  String s = readLine(20000);
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

static void dispatchMenu(String line) {
  line.trim();
  if (line.length() == 0) return;
  char c = line[0];
  if (c >= 'A' && c <= 'Z') c = (char)(c + 32);   // manual lowercase, no ctype.h
  switch (c) {
    case '1': captureTrainFlow(); break;
    case '2': captureTestFlow(); break;
    case '3':
      if (g_camOk && captureRGB888(g_curImg)) previewASCII(g_curImg);
      else Serial.println("  camera unavailable");
      break;
    case '4': datasetSummary(); break;
    case '5': trainFlow(); break;
    case '6': liveInference(); break;
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
    case '0': printMenu(); break;
    default: Serial.printf("  unknown command '%c' - menu 0\n", c);
  }
  Serial.println();
  Serial.print("> ");
}

// ---------------- setup / loop ----------------------------------------------
void setup() {
  pinMode(LED_BUILTIN_NEG, OUTPUT);
  setLed(true);
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);   // don't deadlock on USB CDC

  g_psramOk = psramFound();

  Serial.println();
  Serial.println("XIAO ESP32-S3 Sense - On-Device Vision Trainer v1.0");
  Serial.println("build: esp32:esp32:XIAO_ESP32S3:PSRAM=opi  (PSRAM required)");
  Serial.printf("PSRAM: %s\n", g_psramOk ? "OK" : "MISSING - camera/model will fail");

  if (g_psramOk && !nnAlloc()) {
    Serial.println("FATAL: could not allocate MLP weights in PSRAM");
  }
  if (!allocDataset()) {
    Serial.println("warning: dataset buffer unavailable");
  }

  g_camOk = initCamera();
  Serial.printf("camera: %s\n", g_camOk ? "OK" : "FAILED");

  SPI.begin(7, 8, 9, SD_CS_PIN);   // Sense SD slots: SCK=GPIO7, MISO=GPIO8, MOSI=GPIO9
  if (SD.begin(SD_CS_PIN)) {
    g_sdOk = true;
    Serial.printf("SD: OK - card type %d, size %llu MB\n",
                  (int)SD.cardType(), (unsigned long long)(SD.cardSize() / (1024ULL * 1024ULL)));
    ensureTree();
    loadClasses();
  } else {
    Serial.println("SD: FAILED - insert FAT32 microSD and reset");
  }

  setLed(false);
  printMenu();
}

void loop() {
  static uint32_t lastBlink = 0;
  static bool ledState = false;
  if (millis() - lastBlink >= 1000) {
    lastBlink = millis();
    ledState = !ledState;
    setLed(ledState);
  }
  if (Serial.available()) {
    String line = readLine(10000);
    dispatchMenu(line);
  }
}
// deerflow_build_id=20260909T045352Z-d441fb747625-1cd53b6ad0ba
