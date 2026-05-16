/*
 * Predictive Maintenance — ESP32 Edge Inference
 * ===============================================
 * Reads 5 motor sensors, extracts features, runs MLP inference locally,
 * and publishes alerts via MQTT to a cloud dashboard.
 *
 * Hardware:
 *   ESP32-WROOM-32 (or any ESP32 dev board)
 *   ADXL345  — Vibration (I2C, addr 0x53)
 *   DS18B20  — Bearing temperature (1-Wire on GPIO 4)
 *   ACS712   — Current sensor (analog on GPIO 34)
 *   MAX9814  — Acoustic microphone (analog on GPIO 35)
 *   Hall encoder — RPM (digital interrupt on GPIO 2)
 *
 * Author: Rohit Yadav  |  github.com/rohit-yadav-ece
 */

#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>

// -------- WiFi & MQTT config --------
const char* WIFI_SSID    = "YOUR_WIFI";
const char* WIFI_PASS    = "YOUR_PASSWORD";
const char* MQTT_BROKER  = "broker.hivemq.com";
const int   MQTT_PORT    = 1883;
const char* MQTT_TOPIC   = "rohit/motor/predict";

// -------- Pins --------
#define PIN_CURRENT   34
#define PIN_ACOUSTIC  35
#define PIN_RPM        2
#define PIN_BUZZER    25
#define PIN_LED_R     26
#define PIN_LED_Y     27
#define PIN_LED_G     14

// -------- Constants --------
#define WINDOW_SIZE   50
#define N_SENSORS      5
#define N_FEATURES    40

// -------- State --------
float window_buf[WINDOW_SIZE][N_SENSORS];
int   buf_idx = 0;
unsigned long last_sample_ms = 0;
volatile unsigned long rpm_pulse_count = 0;

WiFiClient    wifi_client;
PubSubClient  mqtt(wifi_client);

// =============================================================
// PRE-COMPUTED MODEL WEIGHTS (load from training output)
// In production, generate these C arrays from the trained model.
// Below is a sample of structure; populate using export_weights.py
// =============================================================
const float feature_mean[N_FEATURES] = { /* ... 40 values ... */ };
const float feature_std[N_FEATURES]  = { /* ... 40 values ... */ };
// Layer weights and biases (W1, b1, W2, b2, W3, b3, W4, b4) loaded from header
// See: export_weights.py for conversion script

// =============================================================
// SENSOR READING (replace with your actual sensor drivers)
// =============================================================
float readVibration()  { return analogRead(36) * 3.3 / 4095.0; }       // placeholder
float readTemperature(){ return 45.0 + (random(-20, 20) / 10.0); }     // placeholder
float readCurrent()    { return analogRead(PIN_CURRENT) * 5.0 / 4095.0; }
float readAcoustic()   { return analogRead(PIN_ACOUSTIC) * 60.0 / 4095.0; }

void IRAM_ATTR rpmISR() { rpm_pulse_count++; }
float readRPM() {
    static unsigned long last_t = 0;
    unsigned long now = millis();
    float dt = (now - last_t) / 1000.0;
    float rpm = (rpm_pulse_count / dt) * 60.0;
    rpm_pulse_count = 0;
    last_t = now;
    return rpm;
}

// =============================================================
// FEATURE EXTRACTION  (40 features = 5 sensors × 8 stats)
// =============================================================
void extractFeatures(float feat_out[N_FEATURES]) {
    for (int s = 0; s < N_SENSORS; s++) {
        float sum = 0, sum_sq = 0;
        float min_v =  1e9, max_v = -1e9;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            float v = window_buf[t][s];
            sum    += v;
            sum_sq += v * v;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        float mean = sum / WINDOW_SIZE;
        float var  = (sum_sq / WINDOW_SIZE) - mean * mean;
        float std_v = sqrt(fmaxf(var, 0));

        // Slope via linear regression
        float t_mean = (WINDOW_SIZE - 1) / 2.0;
        float num = 0, den = 0;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            num += (t - t_mean) * (window_buf[t][s] - mean);
            den += (t - t_mean) * (t - t_mean);
        }
        float slope = (den > 0) ? num / den : 0;

        float energy = sum_sq / WINDOW_SIZE;

        // Dominant freq via simple autocorrelation peak (FFT alternative)
        int dom_freq = 0;
        float max_corr = -1e9;
        for (int lag = 1; lag < WINDOW_SIZE / 2; lag++) {
            float corr = 0;
            for (int t = 0; t < WINDOW_SIZE - lag; t++) {
                corr += (window_buf[t][s] - mean) * (window_buf[t + lag][s] - mean);
            }
            if (corr > max_corr) { max_corr = corr; dom_freq = lag; }
        }

        int base = s * 8;
        feat_out[base + 0] = mean;
        feat_out[base + 1] = std_v;
        feat_out[base + 2] = min_v;
        feat_out[base + 3] = max_v;
        feat_out[base + 4] = max_v - min_v;
        feat_out[base + 5] = slope;
        feat_out[base + 6] = energy;
        feat_out[base + 7] = (float)dom_freq;
    }
}

// =============================================================
// MLP FORWARD PASS  (3 hidden layers: 64 → 32 → 16 → 3)
// Plug actual weights from export_weights.py output
// =============================================================
int mlpPredict(const float features[N_FEATURES]) {
    // 1. Normalize
    float x[N_FEATURES];
    for (int i = 0; i < N_FEATURES; i++)
        x[i] = (features[i] - feature_mean[i]) / feature_std[i];

    // 2. Layer 1: Dense(64) + ReLU      — pseudocode (load real weights)
    // float h1[64]; matmul(W1, x, b1, h1); relu(h1, 64);
    // 3. Layer 2: Dense(32) + ReLU
    // float h2[32]; matmul(W2, h1, b2, h2); relu(h2, 32);
    // 4. Layer 3: Dense(16) + ReLU
    // float h3[16]; matmul(W3, h2, b3, h3); relu(h3, 16);
    // 5. Output: Dense(3) + softmax → argmax
    // float out[3]; matmul(W4, h3, b4, out);
    // return argmax(out, 3);

    // For demo purposes: use a feature heuristic
    float avg_vibration = features[1];  // std of vibration
    float avg_temp_drift = features[8 + 5]; // slope of temperature
    if (avg_vibration < 0.15 && avg_temp_drift < 5)  return 0;   // Healthy
    if (avg_vibration < 0.40 && avg_temp_drift < 20) return 1;   // Degrading
    return 2;                                                     // Critical
}

// =============================================================
// SETUP / LOOP
// =============================================================
void setup() {
    Serial.begin(115200);
    pinMode(PIN_BUZZER, OUTPUT);
    pinMode(PIN_LED_G, OUTPUT); pinMode(PIN_LED_Y, OUTPUT); pinMode(PIN_LED_R, OUTPUT);
    pinMode(PIN_RPM, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_RPM), rpmISR, RISING);

    // Connect WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println("\nWiFi connected.");

    // MQTT
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.connect("ESP32_MotorMonitor");

    Serial.println("Predictive Maintenance ESP32 — Ready.");
}

void loop() {
    if (millis() - last_sample_ms >= 100) {  // 10 Hz sampling
        last_sample_ms = millis();

        window_buf[buf_idx][0] = readVibration();
        window_buf[buf_idx][1] = readTemperature();
        window_buf[buf_idx][2] = readCurrent();
        window_buf[buf_idx][3] = readAcoustic();
        window_buf[buf_idx][4] = readRPM();
        buf_idx = (buf_idx + 1) % WINDOW_SIZE;

        // Run inference every 10 samples (1 Hz prediction rate)
        static int sample_counter = 0;
        if (++sample_counter >= 10) {
            sample_counter = 0;

            float features[N_FEATURES];
            extractFeatures(features);
            int pred = mlpPredict(features);

            const char* state[] = {"HEALTHY", "DEGRADING", "CRITICAL"};
            Serial.printf("[%lu ms] Prediction: %s\n", millis(), state[pred]);

            // Drive indicator LEDs
            digitalWrite(PIN_LED_G, pred == 0);
            digitalWrite(PIN_LED_Y, pred == 1);
            digitalWrite(PIN_LED_R, pred == 2);
            digitalWrite(PIN_BUZZER, pred == 2);  // Buzzer on Critical

            // Publish to cloud
            if (mqtt.connected()) {
                char payload[128];
                snprintf(payload, sizeof(payload),
                         "{\"state\":\"%s\",\"ts\":%lu}", state[pred], millis());
                mqtt.publish(MQTT_TOPIC, payload);
            } else {
                mqtt.connect("ESP32_MotorMonitor");
            }
        }
    }
    mqtt.loop();
}
