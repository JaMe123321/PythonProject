from ultralytics import YOLO
import cv2

# 載入你訓練好的 YOLOv11 模型
model = YOLO("Z:/專題/FOOT/train4/weights/best.pt")

# 讀取圖片
img = cv2.imread(r"C:\Users\james\PycharmProjects\PythonProject\Python_Flask\test.jpg.jpg")

# 推論（注意要轉成 RGB）
results = model.predict(source=img, save=False, conf=0.25)

# 取得第一張圖的預測結果
result = results[0]
boxes = result.boxes
names = model.names

# 繪製預測框
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{names[cls_id]} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 顯示結果
cv2.imshow("YOLOv11 Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
