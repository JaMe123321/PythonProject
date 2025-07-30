import os
import cv2
import time
from ultralytics import YOLO
import numpy as np
import torch

# 自動選擇設備
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備：{device}")

# 指定影片資料夾
video_folder = r"D:\圖\garbage\影三"
# 載入 YOLO 模型
#model_path = r"Z:\專題\datatest\train7\weights\best.pt"
model_path = r"Z:\專題\紅樓垃圾\train12\weights\best.pt"
#class_names = ['foot', 'shoe']
class_names = ['Bottle', 'Tissue','Plastic']
model = YOLO(model_path)
# 支援的影片格式
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

# 自動列出資料夾內所有影片
video_sources = [
    os.path.join(video_folder, f) for f in os.listdir(video_folder)
    if os.path.splitext(f)[1].lower() in video_extensions
]

# 確認找到影片
if not video_sources:
    print("❌ 資料夾裡沒有找到任何影片")
    exit()

print(f"找到 {len(video_sources)} 部影片")
for idx, video in enumerate(video_sources):
    print(f"{idx+1}. {video}")

video_index = 0  # 目前播放到第幾個影片



# 初始化計數
frame_count = 0
total_fps = 0
prev_time = time.time()

# 顏色設定
def get_color(class_name):
    color_dict = {'user': (0, 255, 0), 'nouser': (0, 0, 255)}
    return color_dict.get(class_name, (0, 255, 255))

# 打開第一個影片
cap = cv2.VideoCapture(video_sources[video_index])
if not cap.isOpened():
    print(f"❌ 無法打開影片 {video_sources[video_index]}")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1 / fps_video
print(f"\n▶️ 播放影片：{video_sources[video_index]}")
print(f"影片原始FPS：{fps_video}")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        # 這部影片播放完，換下一部
        video_index += 1
        if video_index >= len(video_sources):
            print("🎬 影片全部播放完畢")
            break
        cap.release()
        cap = cv2.VideoCapture(video_sources[video_index])
        if not cap.isOpened():
            print(f"❌ 無法打開影片 {video_sources[video_index]}")
            break
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1 / fps_video
        print(f"\n▶️ 播放影片：{video_sources[video_index]}")
        continue

    # YOLO 偵測
    results = model(frame, iou=0.3, conf=0.5, device=device)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confs = result.boxes.conf.cpu().tolist()

    for cls, bbox, conf in zip(classes, bboxes, confs):
        (x, y, x2, y2) = bbox
        class_name = class_names[cls]
        color = get_color(class_name)

        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # FPS計算
    current_time = time.time()
    actual_fps = 1 / (current_time - prev_time)
    prev_time = current_time

    frame_count += 1
    total_fps += actual_fps

    cv2.putText(frame, f"FPS: {actual_fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 Detection", frame)

    processing_time = time.time() - start_time
    delay = max(int((frame_interval - processing_time) * 1000), 1)
    key = cv2.waitKey(delay)
    if key == 27:  # ESC
        break

# 統計
if frame_count > 0:
    average_fps = total_fps / frame_count
    print(f"\n📊 總幀數：{frame_count}，平均 FPS：{average_fps:.2f}")
else:
    print("⚠️ 沒有讀取到任何幀")

# 收尾
cap.release()
cv2.destroyAllWindows()
