#include <Keypad.h>

// LED 腳位定義
const int successLED = 2;
const int failureLED = 3;

const byte ROWS = 4;
const byte COLS = 4;

// 定義鍵盤按鍵排列
char keys[ROWS][COLS] = {
  {'D','C','B','A'},
  {'#','9','6','3'},
  {'0','8','5','2'},
  {'*','7','4','1'}
};

byte rowPins[ROWS] = {4, 5, 6, 7};
byte colPins[COLS] = {8, 9, 10, 11};

// 初始化鍵盤物件
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

const int maxInputLength = 6;
const int maxOTPCount = 10;      // 陣列最多儲存 10 組 OTP
String otpList[maxOTPCount];     // 儲存 OTP 的陣列
int otpCount = 0;                // 當前 OTP 數量
String enteredOTP = "";

void setup() {
  Serial.begin(115200);      // PC 監控
  Serial1.begin(115200);     // 與 ESP32 通信
  pinMode(successLED, OUTPUT);
  pinMode(failureLED, OUTPUT);
  Serial.println("系統已啟動，等待按鍵輸入...");
}

void loop() {
  // 檢查是否有新的 OTP 從 ESP32 傳入
  if (Serial1.available()) {
    String newOTP = Serial1.readStringUntil('\n');
    newOTP.trim();
    Serial.print("收到來自 ESP32 的 OTP：");
    Serial.println(newOTP);

    // 加入新 OTP
    addNewOTP(newOTP);
  }

  char key = keypad.getKey();
  if (key) {
    feedbackBlink();
    Serial.print("按鍵：");
    Serial.println(key);

    if (key == '#') {
      Serial.print("已輸入 OTP：");
      Serial.println(enteredOTP);
      if (verifyOTP(enteredOTP)) {
        Serial.println("OTP 正確！");
        activateSuccess();
      } else {
        Serial.println("OTP 錯誤！");
        activateFailure();
      }
      enteredOTP = "";  // 重置輸入
    } else if (key == '*') {
      enteredOTP = "";
      Serial.println("輸入已重置");
      feedbackBlink();
    } else {
      if (enteredOTP.length() < maxInputLength) {
        enteredOTP += key;
        Serial.print("當前輸入：");
        Serial.println(enteredOTP);
      } else {
        Serial.println("輸入長度已達上限");
        feedbackBlink();
      }
    }
  }
}

// 加入新 OTP 並移除最舊的一組
void addNewOTP(String newOTP) {
  if (otpCount < maxOTPCount) {
    otpList[otpCount] = newOTP;
    otpCount++;
  } else {
    // 移除最舊 OTP
    for (int i = 0; i < maxOTPCount - 1; i++) {
      otpList[i] = otpList[i + 1];
    }
    otpList[maxOTPCount - 1] = newOTP;
  }
  Serial.println("當前 OTP 列表：");
  for (int i = 0; i < otpCount; i++) {
    Serial.println(otpList[i]);
  }
}

// 驗證輸入的 OTP 是否存在
bool verifyOTP(String inputOTP) {
  for (int i = 0; i < otpCount; i++) {
    if (otpList[i] == inputOTP) {
      // 刪除已驗證的 OTP
      for (int j = i; j < otpCount - 1; j++) {
        otpList[j] = otpList[j + 1];
      }
      otpCount--;
      return true;
    }
  }
  return false;
}

// 成功提示：LED 輪流亮滅
void activateSuccess() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(successLED, HIGH);
    digitalWrite(failureLED, LOW);
    delay(300);
    digitalWrite(successLED, LOW);
    digitalWrite(failureLED, HIGH);
    delay(300);
  }
  digitalWrite(successLED, LOW);
  digitalWrite(failureLED, LOW);
}

// 失敗提示：紅燈慢閃
void activateFailure() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(failureLED, HIGH);
    delay(400);
    digitalWrite(failureLED, LOW);
    delay(400);
  }
}

// 按鍵回饋：短暫亮燈
void feedbackBlink() {
  digitalWrite(successLED, HIGH);
  delay(100);
  digitalWrite(successLED, LOW);
}
