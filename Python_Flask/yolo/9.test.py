import cv2
import time
from ultralytics import YOLO
import numpy as np
import torch

# 自動選擇設備（CUDA 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備：{device}")

# 影像來源：攝像頭或影片檔案 0 表示使用攝像頭
#video_source = r"D:\紅樓垃圾\FOOT2\4.mp4"
#video_source = r"D:\紅樓垃圾\影二\767690873.518051.mp4"
#video_source = r"C:\Users\james\OneDrive\桌面\腳\766993860.828273.mp4"
video_source = r"D:\圖\foot\FOOT3\99.mp4"
cap = cv2.VideoCapture(video_source)

# 檢查是否成功打開影片
if not cap.isOpened():
    print("❌ 無法打開影片或攝像頭")
    exit()

# 取得影片FPS
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"影片原始FPS：{fps_video}")

# 計算每一幀的理論間隔時間
frame_interval = 1 / fps_video

# 模型路徑及載入
model_path = r"Z:\專題\datatest\train7\weights\best.pt"

class_names = ['X', 'O']
model = YOLO(model_path)

# 初始化計數
prev_time = time.time()
frame_count = 0
total_fps = 0  # 用於計算平均FPS

def get_color(class_name):
    """根據類別名稱返回不同的顏色"""
    color_dict = {'user': (0, 255, 0), 'nouser': (0, 0, 255)}
    return color_dict.get(class_name, (0, 255, 255))

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ 無法讀取影像或影片結束")

        break

    # YOLOv8 進行偵測
    results = model(frame, iou=0.3, conf=0.5, device=device)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confs = results[0].boxes.conf.cpu().tolist()

    # 繪製邊框和標註
    for cls, bbox, conf in zip(classes, bboxes, confs):
        (x, y, x2, y2) = bbox
        class_name = class_names[cls]
        color = get_color(class_name)

        # 畫框和標註
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 計算實際 FPS
    current_time = time.time()
    actual_fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 更新幀數和累積FPS
    frame_count += 1
    total_fps += actual_fps

    # 顯示 FPS
    cv2.putText(frame, f"FPS: {actual_fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示影像
    cv2.imshow("YOLOv11 Detection", frame)

    # 計算處理時間
    processing_time = time.time() - start_time

    # 根據原始 FPS 和處理時間來同步播放速度
    delay = max(int((frame_interval - processing_time) * 1000), 1)
    key = cv2.waitKey(delay)
    if key == 27:  # 按 ESC 鍵退出
        break

# 計算平均FPS（如果有至少一個幀）
if frame_count > 0:
    average_fps = total_fps / frame_count
    print(f"影片原始FPS：{fps_video}")
    print(f"使用設備：{device}")
    print(f"\n📊 結束程式，總幀數：{frame_count}，平均 FPS：{average_fps:.2f}")
else:
    print("⚠️ 無法計算平均FPS，因為沒有處理任何幀數")

# 釋放資源
cap.release()
cv2.destroyAllWindows()
