import cv2
import threading
import queue

# MJPEG 串流網址
stream_url = 'http://192.168.12.113:8080/?action=stream'

# 影像緩衝區
frame_queue = queue.Queue(maxsize=1)

def grab_frames():
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ 串流開啟失敗")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 串流中斷")
            break

        # 降解析度加速（可選）
        frame = cv2.resize(frame, (320, 240))

        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

# 背景執行抓取影像
thread = threading.Thread(target=grab_frames, daemon=True)
thread.start()

# 顯示影像
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        cv2.imshow("IP CAM 串流", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 離開
        break

cv2.destroyAllWindows()