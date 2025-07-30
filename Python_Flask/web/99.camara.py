import cv2

# 初始化攝影機（索引為 0，可更改為 1、2、-1 嘗試）
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    # 讀取影像幀
    ret, frame = cap.read()
    if not ret:
        print("無法獲取影像幀，攝影機可能已被佔用或不存在")
        break

    # 顯示攝影機畫面
    cv2.imshow('Camera Test', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) == ord('q'):
        break

# 釋放攝影機和關閉視窗
cap.release()
cv2.destroyAllWindows()
