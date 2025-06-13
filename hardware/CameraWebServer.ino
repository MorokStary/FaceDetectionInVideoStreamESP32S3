#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_websocket_client.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include <WiFi.h>
#include "app_httpd.h"
#include "esp_websocket_client.h"
#include "esp_event.h"
#include "esp_log.h"

esp_websocket_client_handle_t ws_client;
static const char *TAG = "ws_client";

#define CAM_LED_PIN 4
#define RELAY_PIN 15
#define BUTTON_PIN 14

const char* ssid = "MiWify";
const char* password = "Qwer123";
const char* websocket_uri = "ws://192.168.1.120:5000/ws";

esp_websocket_client_handle_t client;

void onWebSocketEvent(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
  switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
      ESP_LOGI(TAG, "WebSocket Connected");
      break;
    case WEBSOCKET_EVENT_DISCONNECTED:
      ESP_LOGW(TAG, "WebSocket Disconnected");
      break;
    case WEBSOCKET_EVENT_DATA: {
      esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
      std::string msg((char *)data->data_ptr, data->data_len);
      ESP_LOGI(TAG, "Received: %s", msg.c_str());

      if (msg == "unlock") {
        digitalWrite(RELAY_PIN, HIGH);
        delay(3000);
        digitalWrite(RELAY_PIN, LOW);
      }

      break;
    }
    case WEBSOCKET_EVENT_ERROR:
      ESP_LOGE(TAG, "WebSocket Error");
      break;
  }
}


void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    ESP.restart();
  }
}

void onWebSocketEvent(void* handler_args, esp_event_base_t base, int32_t event_id, void* event_data) {
  switch (event_id) {
    case WEBSOCKET_EVENT_DATA: {
      esp_websocket_event_data_t* data = (esp_websocket_event_data_t*)event_data;
      String msg = String((char*)data->data_ptr).substring(0, data->data_len);
      if (msg == "unlock") {
        digitalWrite(RELAY_PIN, HIGH);
        delay(3000);
        digitalWrite(RELAY_PIN, LOW);
      }
      break;
    }
    default:
      break;
  }
}

void setupWebSocket() {
  esp_websocket_client_config_t websocket_cfg = {};
  websocket_cfg.uri = websocket_uri;

  client = esp_websocket_client_init(&websocket_cfg);
  esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY, onWebSocketEvent, NULL);
  esp_websocket_client_start(client);
}

void setup() {
  Serial.begin(115200);
  gpio_set_direction(RELAY_PIN, GPIO_MODE_OUTPUT);
  gpio_set_direction(BUTTON_PIN, GPIO_MODE_INPUT);
  gpio_set_level(RELAY_PIN, 0);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");

  initCamera();
  startCameraServer();
  setupWebSocket();
    esp_websocket_client_config_t websocket_cfg = {
  .uri = "ws://192.168.1.120:5000/ws"
  };
  ws_client = esp_websocket_client_init(&websocket_cfg);
    esp_websocket_register_events(ws_client, WEBSOCKET_EVENT_ANY, onWebSocketEvent, NULL);
    esp_websocket_client_start(ws_client);
}

void loop() {
  if (digitalRead(BUTTON_PIN) == HIGH) {
    Serial.println("Button pressed");
    delay(300);
  }
  if (esp_websocket_client_is_connected(ws_client)) {
  // const char *msg = "heartbeat";
  // esp_websocket_client_send_text(ws_client, msg, strlen(msg), portMAX_DELAY);
}
}
