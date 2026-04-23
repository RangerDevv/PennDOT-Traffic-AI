/*
 * PennDOT Lane Congestion Detector  –  Live Dashboard
 * ESP32-S3-EYE  –  Arduino Sketch
 *
 * Classifies two road states captured by the on-board OV2640 camera:
 *   0 – Left Lane Congestion
 *   1 – Right Lane Congestion
 *
 * Features:
 *   - Live MJPEG camera stream at /stream
 *   - JSON inference API at /inference
 *   - Web dashboard with virtual traffic lights at /
 *
 * Board: "ESP32S3 Dev Module" or "ESP32-S3-EYE"
 *   - Flash: QIO 80 MHz, 8 MB
 *   - PSRAM: OPI PSRAM (required)
 *   - Partition scheme: Huge APP (3 MB)
 */

#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "img_converters.h"

// TFLite Micro
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Generated model data (produced by train_model.py)
#include "model_data.h"

// ── WiFi credentials ──────────────────────────────────────────────────────────
// CHANGE THESE to your network
const char* WIFI_SSID = "Fios-7WLn5";
const char* WIFI_PASS = "vial35epic63hue";

// ── ESP32-S3-EYE camera pin map ───────────────────────────────────────────────
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD     4
#define CAM_PIN_SIOC     5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2       8
#define CAM_PIN_D1       9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC    6
#define CAM_PIN_HREF     7
#define CAM_PIN_PCLK    13

// ── Inference parameters (from model_data.h) ─────────────────────────────────
constexpr size_t   kTensorArenaSize     = 128 * 1024;
static uint8_t*    tensor_arena         = nullptr;
constexpr float    kConfidenceThreshold = 0.50f;
constexpr uint32_t kInferenceIntervalMs = 0;   // run as fast as possible

// ── Globals ───────────────────────────────────────────────────────────────────
static const tflite::Model*      tfl_model    = nullptr;
static tflite::MicroInterpreter* interpreter  = nullptr;
static TfLiteTensor*             input_tensor = nullptr;
static TfLiteTensor*             output_tensor= nullptr;

// Reusable RGB buffer (allocated once in PSRAM)
static uint8_t* g_rgb_buf = nullptr;
static size_t   g_rgb_buf_size = 0;

// Latest inference result (shared with web server)
static volatile int   g_best_class = -1;
static volatile float g_best_score = 0.0f;
static volatile float g_scores[2]  = {0, 0};
static volatile unsigned long g_last_inference_ms = 0;

// HTTP server handle
static httpd_handle_t http_server = NULL;

// ── Camera initialisation ─────────────────────────────────────────────────────
static bool init_camera() {
    camera_config_t cfg = {};
    cfg.ledc_channel  = LEDC_CHANNEL_0;
    cfg.ledc_timer    = LEDC_TIMER_0;
    cfg.pin_d0        = CAM_PIN_D0;
    cfg.pin_d1        = CAM_PIN_D1;
    cfg.pin_d2        = CAM_PIN_D2;
    cfg.pin_d3        = CAM_PIN_D3;
    cfg.pin_d4        = CAM_PIN_D4;
    cfg.pin_d5        = CAM_PIN_D5;
    cfg.pin_d6        = CAM_PIN_D6;
    cfg.pin_d7        = CAM_PIN_D7;
    cfg.pin_xclk      = CAM_PIN_XCLK;
    cfg.pin_pclk      = CAM_PIN_PCLK;
    cfg.pin_vsync     = CAM_PIN_VSYNC;
    cfg.pin_href      = CAM_PIN_HREF;
    cfg.pin_sccb_sda  = CAM_PIN_SIOD;
    cfg.pin_sccb_scl  = CAM_PIN_SIOC;
    cfg.pin_pwdn      = CAM_PIN_PWDN;
    cfg.pin_reset     = CAM_PIN_RESET;
    cfg.xclk_freq_hz  = 20000000;
    cfg.pixel_format  = PIXFORMAT_JPEG;
    cfg.frame_size    = FRAMESIZE_QVGA;     // 320x240
    cfg.jpeg_quality  = 12;                  // 1-63, lower = better quality
    if (heap_caps_get_free_size(MALLOC_CAP_SPIRAM) > 0) {
        cfg.fb_location = CAMERA_FB_IN_PSRAM;
        cfg.fb_count    = 3;   // 3 buffers: avoids stream starving inference
    } else {
        cfg.fb_location = CAMERA_FB_IN_DRAM;
        cfg.fb_count    = 1;
    }
    cfg.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        return false;
    }

    // ESP32-S3-EYE has the sensor mounted upside-down
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_vflip(s, 1);
        s->set_hmirror(s, 1);
    }

    Serial.println("[CAM] Init OK");
    return true;
}

// ── TFLite initialisation ─────────────────────────────────────────────────────
static bool init_tflite() {
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        Serial.println("[TFL] PSRAM alloc failed");
        return false;
    }

    tfl_model = tflite::GetModel(g_model_data);
    if (!tfl_model || tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFL] Model load/version error");
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        tfl_model, resolver, tensor_arena, kTensorArenaSize,
        &micro_error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFL] AllocateTensors failed");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("[TFL] Arena used: %u / %u bytes\n",
                  interpreter->arena_used_bytes(), kTensorArenaSize);
    return true;
}

// ── WiFi initialisation ───────────────────────────────────────────────────────
static void init_wifi() {
    WiFi.disconnect(true);        // clear any stale connection state
    // Enable both Station and Access Point mode
    WiFi.mode(WIFI_AP_STA);
    // Start AP with custom SSID and password
    const char* AP_SSID = "PennDOT-Traffic-AI";
    const char* AP_PASS = "congestion";
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.printf("[WiFi] AP started!  SSID: %s  PASS: %s  IP: %s\n", AP_SSID, AP_PASS, WiFi.softAPIP().toString().c_str());

    // Connect to existing WiFi as STA
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("[WiFi] Connecting");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      WiFi.setAutoReconnect(true);
      Serial.printf("\n[WiFi] Connected!  IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
      Serial.println("\n[WiFi] Connection failed – check SSID/password and restart.");
    }
}

// ── Preprocessing: JPEG → RGB888 → model input ──────────────────────────────
// Camera stays in JPEG mode. We decode one frame to RGB888 for inference.
static bool run_inference_cycle() {
    unsigned long t0 = millis();

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return false;
    unsigned long t1 = millis();

    // Allocate / reuse RGB buffer in PSRAM
    const size_t rgb_size = fb->width * fb->height * 3;
    if (!g_rgb_buf || g_rgb_buf_size < rgb_size) {
        if (g_rgb_buf) heap_caps_free(g_rgb_buf);
        g_rgb_buf = (uint8_t*)heap_caps_malloc(rgb_size, MALLOC_CAP_SPIRAM);
        g_rgb_buf_size = g_rgb_buf ? rgb_size : 0;
    }
    if (!g_rgb_buf) {
        esp_camera_fb_return(fb);
        Serial.println("[INF] RGB alloc failed");
        return false;
    }

    if (!fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, g_rgb_buf)) {
        esp_camera_fb_return(fb);
        Serial.println("[INF] JPEG decode failed");
        return false;
    }
    unsigned long t2 = millis();

    const int src_w = fb->width;
    const int src_h = fb->height;
    esp_camera_fb_return(fb);

    // Preprocess: crop bottom half, resize to 96x96, quantise to INT8
    const int dst_w = kImageWidth;
    const int dst_h = kImageHeight;
    const int crop_y_start = (int)(src_h * (1.0f - kCropFraction));
    const int crop_h = src_h - crop_y_start;

    const int   zero_point = input_tensor->params.zero_point;
    const float scale      = input_tensor->params.scale;
    // Pre-compute integer LUT: pixel_value → quantised int8
    // Avoids per-pixel float division (maps 0-255 → int8)
    static int8_t quant_lut[256];
    static float  lut_scale = 0;
    if (lut_scale != scale) {
        lut_scale = scale;
        for (int v = 0; v < 256; v++) {
            float q = (v / 255.0f / scale) + zero_point;
            q = q < -128.0f ? -128.0f : (q > 127.0f ? 127.0f : q);
            quant_lut[v] = (int8_t)q;
        }
    }
    int8_t* dst = input_tensor->data.int8;

    for (int dy = 0; dy < dst_h; dy++) {
        const int sy = crop_y_start + (dy * crop_h / dst_h);
        const uint8_t* src_row = g_rgb_buf + (size_t)sy * src_w * 3;
        for (int dx = 0; dx < dst_w; dx++) {
            const int sx = dx * src_w / dst_w;
            const uint8_t* px = src_row + sx * 3;
            const int idx = (dy * dst_w + dx) * 3;
            dst[idx + 0] = quant_lut[px[0]];
            dst[idx + 1] = quant_lut[px[1]];
            dst[idx + 2] = quant_lut[px[2]];
        }
    }

    // Run inference
    unsigned long t3 = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[TFL] Invoke failed");
        return false;
    }
    unsigned long t4 = millis();

    // Dequantise output
    const float out_scale      = output_tensor->params.scale;
    const int   out_zero_point = output_tensor->params.zero_point;

    int   best_class = -1;
    float best_score = -1.0f;
    for (int i = 0; i < kNumClasses; i++) {
        float sc = (output_tensor->data.int8[i] - out_zero_point) * out_scale;
        g_scores[i] = sc;
        if (sc > best_score) { best_score = sc; best_class = i; }
    }
    g_best_class = best_class;
    g_best_score = best_score;
    g_last_inference_ms = millis();

    Serial.printf("[INF] %s (%.1f%%)  [grab:%lums decode:%lums invoke:%lums total:%lums]\n",
                  kClassNames[best_class], best_score * 100.0f,
                  t1-t0, t2-t1, t4-t3, t4-t0);
    return true;
}

// ── Web server: Dashboard HTML ────────────────────────────────────────────────
static const char DASHBOARD_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PennDOT Lane Congestion Detector</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#eee;min-height:100vh}
.header{background:#16213e;padding:16px 24px;text-align:center;border-bottom:3px solid #0f3460}
.header h1{font-size:1.5rem;color:#e94560}
.header p{font-size:0.85rem;color:#aaa;margin-top:4px}
.container{display:flex;flex-wrap:wrap;gap:20px;padding:20px;max-width:1200px;margin:0 auto;justify-content:center}
.video-panel{flex:1;min-width:320px;max-width:640px}
.video-panel img{width:100%;border-radius:8px;border:2px solid #0f3460}
.side-panel{display:flex;flex-direction:column;gap:16px;min-width:280px;max-width:360px}
.card{background:#16213e;border-radius:12px;padding:20px;border:1px solid #0f3460}
.card h2{font-size:1rem;color:#e94560;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
.traffic-lights{display:flex;gap:24px;justify-content:center;align-items:flex-start}
.lane-light{text-align:center}
.lane-light .label{font-size:0.8rem;margin-bottom:8px;color:#aaa}
.light-housing{background:#111;border-radius:12px;padding:10px 14px;display:inline-flex;flex-direction:column;gap:8px;border:2px solid #333}
.bulb{width:40px;height:40px;border-radius:50%;border:2px solid #333;transition:all 0.4s ease}
.bulb.red{background:#440000}.bulb.red.on{background:#ff1a1a;box-shadow:0 0 20px #ff1a1a,0 0 40px #ff1a1a80}
.bulb.yellow{background:#443300}.bulb.yellow.on{background:#ffcc00;box-shadow:0 0 20px #ffcc00,0 0 40px #ffcc0080}
.bulb.green{background:#003300}.bulb.green.on{background:#00ff55;box-shadow:0 0 20px #00ff55,0 0 40px #00ff5580}
.scores{list-style:none}
.scores li{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #0f346040}
.scores li:last-child{border:none}
.score-bar-bg{width:100px;height:10px;background:#0f3460;border-radius:5px;overflow:hidden;margin-left:10px}
.score-bar{height:100%;border-radius:5px;transition:width 0.5s ease;background:linear-gradient(90deg,#e94560,#0f3460)}
.result-box{text-align:center;padding:16px;border-radius:8px;font-size:1.2rem;font-weight:700;transition:all 0.4s}
.result-box.congested{background:#ff1a1a22;border:2px solid #ff1a1a;color:#ff6b6b}
.result-box.partial{background:#ffcc0022;border:2px solid #ffcc00;color:#ffcc00}
.result-box.clear{background:#00ff5522;border:2px solid #00ff55;color:#00ff55}
.result-box.unknown{background:#66666622;border:2px solid #666;color:#999}
.status{font-size:0.75rem;color:#666;text-align:center;margin-top:4px}
</style>
</head>
<body>
<div class="header">
  <h1>PennDOT Lane Congestion Detector</h1>
  <p>ESP32-S3-EYE &bull; MobileNetV2 INT8 &bull; Real-Time AI</p>
</div>
<div class="container">
  <div class="video-panel">
    <img id="stream" alt="Camera Stream">
    <script>document.getElementById('stream').src = location.protocol+'//'+location.hostname+':81/stream';</script>
  </div>
  <div class="side-panel">
    <div class="card">
      <h2>Traffic Status</h2>
      <div class="traffic-lights">
        <div class="lane-light">
          <div class="label">LEFT LANE</div>
          <div class="light-housing">
            <div class="bulb red" id="left-red"></div>
            <div class="bulb yellow" id="left-yellow"></div>
            <div class="bulb green" id="left-green"></div>
          </div>
        </div>
        <div class="lane-light">
          <div class="label">RIGHT LANE</div>
          <div class="light-housing">
            <div class="bulb red" id="right-red"></div>
            <div class="bulb yellow" id="right-yellow"></div>
            <div class="bulb green" id="right-green"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Detection Result</h2>
      <div class="result-box unknown" id="result">Initializing...</div>
      <div class="status" id="status">Waiting for first inference</div>
    </div>
    <div class="card">
      <h2>Class Probabilities</h2>
      <ul class="scores" id="scores"></ul>
    </div>
  </div>
</div>
<script>
function updateLights(cls, conf) {
  var ids = ['left-red','left-yellow','left-green','right-red','right-yellow','right-green'];
  ids.forEach(function(id){document.getElementById(id).classList.remove('on')});
  if (conf < 0.50) {
    document.getElementById('left-yellow').classList.add('on');
    document.getElementById('right-yellow').classList.add('on');
    return;
  }
  // 2-class logic: 0 = Left Lane Congestion, 1 = Right Lane Congestion
  if (cls === 0) {
    document.getElementById('left-green').classList.add('on');
    document.getElementById('right-red').classList.add('on');
  } else if (cls === 1) {
    document.getElementById('left-red').classList.add('on');
    document.getElementById('right-green').classList.add('on');
  }
}

function updateResult(cls, conf, names) {
  var el = document.getElementById('result');
  el.className = 'result-box';
  if (conf < 0.50) {
    el.textContent = 'UNCERTAIN (' + (conf*100).toFixed(1) + '%)';
    el.classList.add('unknown');
  } else {
    el.textContent = names[cls] + ' (' + (conf*100).toFixed(1) + '%)';
    if (cls === 0) el.classList.add('congested');
    else el.classList.add('partial');
  }
}

function fetchInference() {
  fetch('/inference').then(function(r){return r.json()}).then(function(d) {
    if (!d.ready) {
      document.getElementById('status').textContent = 'Model warming up...';
      return;
    }
    updateLights(d.class, d.confidence);
    updateResult(d.class, d.confidence, d.names);
    var ul = document.getElementById('scores');
    ul.innerHTML = '';
    for (var i = 0; i < d.names.length; i++) {
      var pct = (d.scores[i]*100).toFixed(1);
      var li = document.createElement('li');
      li.innerHTML = '<span>' + d.names[i] + '</span>' +
        '<span style="display:flex;align-items:center">' + pct + '%' +
        '<div class="score-bar-bg"><div class="score-bar" style="width:' +
        Math.max(0,Math.min(100,pct)) + '%"></div></div></span>';
      ul.appendChild(li);
    }
    document.getElementById('status').textContent =
      'Last update: ' + new Date().toLocaleTimeString();
  }).catch(function(){
    document.getElementById('status').textContent = 'Connection lost - retrying...';
  });
}

setInterval(fetchInference, 500);
fetchInference();
</script>
</body>
</html>
)rawliteral";

// ── Web server handlers (esp_http_server – async, non-blocking) ───────────────

// GET / – serve dashboard HTML
static esp_err_t handler_root(httpd_req_t* req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, DASHBOARD_HTML, strlen(DASHBOARD_HTML));
    return ESP_OK;
}

// GET /inference – JSON API
static esp_err_t handler_inference(httpd_req_t* req) {
    char json[512];
    int cls = (int)g_best_class;
    if (cls < 0 || cls >= kNumClasses) cls = -1;
    snprintf(json, sizeof(json),
      "{\"class\":%d,\"confidence\":%.4f,"
      "\"scores\":[%.4f,%.4f],"
      "\"names\":[\"%s\",\"%s\"],"
      "\"uptime\":%lu,\"ready\":%s}",
      cls, (float)g_best_score,
      (float)g_scores[0], (float)g_scores[1],
      kClassNames[0], kClassNames[1],
      millis() / 1000,
      cls >= 0 ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, json, strlen(json));
    return ESP_OK;
}

// GET /capture – single JPEG frame for data collection
static esp_err_t handler_capture(httpd_req_t* req) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_send(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return ESP_OK;
}

// GET /stream – MJPEG stream (runs on its own connection, non-blocking)
#define PART_BOUNDARY "frame123boundary"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t handler_stream(httpd_req_t* req) {
    esp_err_t res = ESP_OK;
    char part_buf[64];

    httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "15");

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[STREAM] No frame");
            res = ESP_FAIL;
            break;
        }

        size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);

        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);

        esp_camera_fb_return(fb);

        if (res != ESP_OK) break;

        // Target ~15 fps
        delay(66);
    }
    return res;
}

// ── Setup web server routes ───────────────────────────────────────────────────
static void init_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_uri_handlers = 8;
    config.stack_size = 8192;
    config.lru_purge_enable = true;   // recycle stale sockets on refresh
    config.max_open_sockets = 4;

    if (httpd_start(&http_server, &config) == ESP_OK) {
        httpd_uri_t uri_root = { .uri = "/", .method = HTTP_GET, .handler = handler_root, .user_ctx = NULL };
        httpd_uri_t uri_inf  = { .uri = "/inference", .method = HTTP_GET, .handler = handler_inference, .user_ctx = NULL };
        httpd_uri_t uri_cap  = { .uri = "/capture", .method = HTTP_GET, .handler = handler_capture, .user_ctx = NULL };
        httpd_register_uri_handler(http_server, &uri_root);
        httpd_register_uri_handler(http_server, &uri_inf);
        httpd_register_uri_handler(http_server, &uri_cap);
        Serial.println("[WEB] Dashboard server started on port 80");
    } else {
        Serial.println("[WEB] ERROR: Dashboard server failed to start!");
    }

    // Stream server on port 81 (separate so it doesn't block the main server)
    httpd_handle_t stream_httpd = NULL;
    httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
    stream_config.server_port = 81;
    stream_config.ctrl_port = 32769;
    stream_config.max_uri_handlers = 1;
    stream_config.stack_size = 8192;
    stream_config.lru_purge_enable = true;
    stream_config.max_open_sockets = 2;

    if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
        httpd_uri_t uri_stream = { .uri = "/stream", .method = HTTP_GET, .handler = handler_stream, .user_ctx = NULL };
        httpd_register_uri_handler(stream_httpd, &uri_stream);
        Serial.println("[WEB] Stream server started on port 81");
    } else {
        Serial.println("[WEB] ERROR: Stream server failed to start!");
    }
}

// ── Arduino entry points ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== PennDOT Lane Congestion Detector ===");
    Serial.printf("Model size : %u bytes\n", g_model_data_len);
    Serial.printf("PSRAM size : %u bytes free\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    Serial.printf("SRAM free  : %u bytes\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    if (!init_camera()) {
        Serial.println("Camera init failed - halting");
        while (true) { delay(1000); }
    }

    if (!init_tflite()) {
        Serial.println("TFLite init failed - halting");
        while (true) { delay(1000); }
    }

    init_wifi();
    init_webserver();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("Dashboard: http://%s\n", WiFi.localIP().toString().c_str());
        Serial.printf("Stream:    http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    }
    Serial.println("Ready!\n");
}

void loop() {
    // Run inference periodically (web server runs async in background)
    static unsigned long last_inf = 0;
    if (millis() - last_inf >= kInferenceIntervalMs) {
        last_inf = millis();
        run_inference_cycle();
    }
    delay(10);
}
