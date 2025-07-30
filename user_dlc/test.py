import cv2
import time
import numpy as np
import threading

# RTSP URLs
RTSP_URL1 = "rtsp://192.168.1.37:8554/live"  # 車鏡頭
RTSP_URL2 = "rtsp://192.168.1.194:8554/live"  # 門口鏡頭


# Function to connect to RTSP and display in a window
def show_camera(rtsp_url, window_name):
    width, height = 640, 480
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def reconnect():
        print(f"🔄 嘗試重連 {window_name} ...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # 等待連線穩定
        time.sleep(0.5)
        return cap

    cap = reconnect()
    last_valid_frame = black_frame.copy()
    fail_count = 0
    MAX_FAIL = 30  # 連續幾幀失敗後判斷斷線

    while True:
        success, frame = cap.read()

        if not success or frame is None:
            print(f"⚠️ {window_name} 讀取失敗（第 {fail_count+1} 次）")
            fail_count += 1
            frame = black_frame.copy()
        else:
            fail_count = 0
            last_valid_frame = frame.copy()

        # 若失敗次數太多，就釋放並重連
        if fail_count >= MAX_FAIL:
            print(f"❌ {window_name} 鏡頭斷線，重新連接中...")
            cap.release()
            time.sleep(1)
            cap = reconnect()
            fail_count = 0
            continue

        # 顯示畫面（成功或黑畫面）
        frame = cv2.resize(frame, (width, height))
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    # Start two threads to display both cameras
    t1 = threading.Thread(target=show_camera, args=(RTSP_URL1, 'Camera 1'), daemon=True)
    t2 = threading.Thread(target=show_camera, args=(RTSP_URL2, 'Camera 2'), daemon=True)
    t1.start()
    t2.start()

    try:
        while True:
            # Main thread can perform other tasks or just sleep
            time.sleep(1)
    except KeyboardInterrupt:
        print("退出程式，關閉所有視窗...")
        cv2.destroyAllWindows()
