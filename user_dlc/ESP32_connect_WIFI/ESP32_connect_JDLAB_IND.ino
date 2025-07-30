#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "JDLAB_IND";        // Wi-Fi 名稱
const char* password = "40427200";  // Wi-Fi 密碼

// 固定 IP 設定
IPAddress local_IP(192, 168, 12, 144);  // ESP32 保持的固定 IP
IPAddress gateway(192, 168, 12, 1);     // 路由器的 IP 位址
IPAddress subnet(255, 255, 255, 0);    // 子網路遮罩
IPAddress primaryDNS(8, 8, 8, 8);      // 首選 DNS
IPAddress secondaryDNS(8, 8, 4, 4);    // 備用 DNS


WebServer server(80);  // 設定 HTTP 伺服器在端口 80

String received_otp = "";

// 當收到 /send-otp 請求時的處理
void handleSendOTP() {
  if (server.hasArg("otp")) {
    received_otp = server.arg("otp");
    Serial.println("收到 OTP：");
    Serial.println(received_otp);
    Serial2.println(received_otp);  // 傳送 OTP 至 Arduino
    server.send(200, "text/plain", "OTP 接收成功");
  } else {
    Serial.println("未收到有效 OTP 參數");
    server.send(400, "text/plain", "缺少 OTP 參數");
  }
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17);  // 與 Arduino 通信
  
  // 設定固定 IP
  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) 
  {
    Serial.println("STA 設定失敗！");
  }
  WiFi.begin(ssid, password);
  Serial.println("連線中...");
  Serial.println("ESP32 測試 - 串口初始化成功");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("正在嘗試連線...");
  }

  Serial.println("Wi-Fi 連線成功！");
  Serial.print("IP 位址：");
  Serial.println(WiFi.localIP());

  // 註冊處理程序
  server.on("/send-otp", HTTP_POST, handleSendOTP);

  // 啟動伺服器
  server.begin();
  Serial.println("HTTP 伺服器已啟動");
}

void loop() {
  Serial.println("ESP32 正常運行...");
  delay(1000);
  server.handleClient();  // 處理客戶端連線
  delay(100);  // 防止卡死
}