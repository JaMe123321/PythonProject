# app_ws.py
import os
# —— 一定要在任何需要 OpenMP 的库 import 之前就设置！ ——
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import time
import base64
import cv2
import eventlet
eventlet.monkey_patch()
from flask import Flask
from flask_socketio import SocketIO
from ultralytics import YOLO

# ─── 初始化 Flask + SocketIO ───────────────────────────
app = Flask(__name__, static_folder='static', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许所有跨域

# ─── 全局：摄像头、模型、统计容器 ───────────────────────

cap = cv2.VideoCapture(r"D:\圖\garbage\影二\1.mp4")
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头：请检查索引或流地址")
model = YOLO(r"D:\圖\garbage\all3\train18\weights\best.pt")  # 指定你训练好的权重

latest_frame = {"data": None, "ts": 0.0}
latest_stats = {"Tissue": 0, "Bottle": 0, "Plastic": 0, "Total": 0}
lock = threading.Lock()

recognition_active = True
category_counts = {"Tissue": 0, "Bottle": 0, "Plastic": 0, "Total": 0}

# ─── 后台线程：不断读取、检测、更新缓存 ─────────────────
def camera_thread():
    label_map = {0: "Bottle", 1: "Tissue", 2: "Plastic"}

    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        disp = frame.copy()
        now = time.time()

        # YOLO 跟踪，persist=True 保留 ID
        results = model.track(frame, iou=0.3, conf=0.5, persist=True, device="cuda")

        # 每帧重置一下本次的「穿线」操作
        threshold_y = disp.shape[0] - 350
        last_seen = {}
        crossed_ids = set()
        prev_y2 = {}

        # 遍历检测到的 box
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = label_map.get(cls, "Unknown")

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            tid = int(box.id.item()) if box.id is not None else None

            # 计数穿线
            if tid is not None:
                prev = prev_y2.get(tid, 0)
                if prev < threshold_y <= y2 and tid not in crossed_ids:
                    crossed_ids.add(tid)
                    category_counts[label] += 1
                    category_counts["Total"] += 1
                prev_y2[tid] = y2

            # 在画面上画框
            disp = cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
            disp = cv2.putText(
                disp, label, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
            )

        # 画红线
        cv2.line(disp, (0, threshold_y), (disp.shape[1], threshold_y), (0, 0, 255), 2)

        # 把画面编码成 Base64
        ok, buf = cv2.imencode('.jpg', disp)
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode('utf8')

        with lock:
            latest_frame["data"] = b64
            latest_frame["ts"] = now
            # 拷贝一份，不要直接推 category_counts 本身
            latest_stats.clear()
            latest_stats.update(category_counts)

        # 控制采集频率
        time.sleep(0.1)

# ─── 后台线程：周期性通过 WebSocket 推送 ──────────────────
def broadcast_thread():
    while True:
        with lock:
            payload = {
                "ts": latest_frame["ts"],
                "image": latest_frame["data"],
                "stats":  latest_stats
            }
        socketio.emit('update', payload)
        # 每 100ms 推一次
        socketio.sleep(0.1)

# 启动两个后台线程
threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=broadcast_thread, daemon=True).start()

# ─── HTTP 路由：只提供静态页面 ─────────────────────────

def index():
    return app.send_static_file('socket.html')
@app.route('/ping')
def ping():
    return 'pong'

# ─── 启动 SocketIO 服务 ────────────────────────────────
if __name__ == '__main__':
    print(">>> 啟動 SocketIO 伺服器…")
    socketio.run(
        app,
        host='0.0.0.0',
        port=7777,
        allow_unsafe_werkzeug=True
    )

