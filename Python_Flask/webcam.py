import cv2
import time
from ultralytics import YOLO
import numpy as np
import torch
import os
from datetime import datetime

# 初始化設備與模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備：{device}")

model_path = r"C:\Users\james\Desktop\runs\train3allonly\weights\best.pt"
model = YOLO(model_path)
class_names = ['X', 'O']

# 影像來源
stream_url = 'http://192.168.12.113:8080/?action=stream'
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ 串流無法打開")
    exit()

# 儲存設定
save_folder = "12"
os.makedirs(save_folder, exist_ok=True)
last_saved_time = 0
save_interval = 3

# 模式控制變數
mode = 'high'  # 預設高準度模式
frame_id = 0

# FPS 計算
prev_time = time.time()
frame_count = 0
total_fps = 0

def get_color(class_name):
    color_dict = {'X': (0, 255, 0), 'O': (0, 0, 255)}
    return color_dict.get(class_name, (0, 255, 255))

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 無法讀取影像")
        break

    frame_id += 1
    original_frame = frame.copy()

    # 模式調整
    if mode == 'fast':
        frame = cv2.resize(frame, (320, 240))
        detect_this_frame = (frame_id % 5 == 0)
    else:
        detect_this_frame = True  # 每幀都辨識

    detected_X = False
    if detect_this_frame:
        results = model(frame, conf=0.3, iou=0.3, stream=True, device=device, verbose=False)
        for result in results:
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            confs = result.boxes.conf.cpu().tolist()

            for cls, bbox, conf in zip(classes, bboxes, confs):
                (x, y, x2, y2) = bbox
                class_name = class_names[cls]
                color = get_color(class_name)
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if class_name == 'X':
                    detected_X = True

    current_time = time.time()
    if detected_X and (current_time - last_saved_time > save_interval):
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        filepath = os.path.join(save_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 儲存影像至 {filepath}")
        last_saved_time = current_time

    # FPS 顯示
    actual_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    frame_count += 1
    total_fps += actual_fps

    cv2.putText(frame, f"FPS: {actual_fps:.2f} | Mode: {mode.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 - 切換模式：F(流暢)/H(高準度)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('f'):
        mode = 'fast'
        print("流暢模式")
    elif key == ord('h'):
        mode = 'high'
        print("高準度模式")

if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"結束，平均 FPS：{avg_fps:.2f}")
else:
    print("沒有處理任何畫面")

cap.release()
cv2.destroyAllWindows()