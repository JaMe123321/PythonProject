import cv2, os

# 增加穩定性（TCP 串流 + 10 秒 timeout）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;10000000"
#rtsp_url = "rtsp://192.168.1.194:8554/live"
rtsp_url = "rtsp://192.168.52.138:8554/live"  # 一定要含 test
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ 無法連線 RTSP")
else:
    print("✅ 成功接收到 RTSP")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 無法讀取影像")
            break
        cv2.imshow("RTSP 測試", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
