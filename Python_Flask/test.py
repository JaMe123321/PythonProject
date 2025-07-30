from collections import defaultdict
import cv2
from ultralytics import YOLO
from Python_Flask.web.canny import classify_cucumber, convent_image, draw_frame


#cap = cv2.VideoCapture(r"Z:\PyCharm\Project\pythonProject\video\gherkin\13.mp4")
cap = cv2.VideoCapture(0)
model = YOLO(r"C:/202501/masterdegree/yoloV8&HTML&CSS&JS/Python_Flask/Python_Flask/real_gherkinv3.pt", 'track')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
track_history = defaultdict(lambda: [])
total_gherkin, level_S, level_A, level_B, level_C = 0, 0, 0, 0, 0

curvature_threshold = 0.05
area_difference_threshold = 8
length_optimal_max = 15
width_optimal_max = 2.8

level_result = ""


def box_label(image, box, label='', color=(255, 128, 128), txt_color=(0, 0, 0)):
    # 框框的左上角和右下角座標
    p1, p2 = (int(box[0]-15), int(box[1]-25)), (int(box[2]+15), int(box[3]+25))
    cv2.rectangle(image, p1, p2, color, thickness=4, lineType=cv2.LINE_AA)

    if label:
        # 計算框的上邊框中心點
        center_x = (p1[0] + p2[0]) // 2
        label_y = p1[1] - 10  # 在框上方稍微偏移，避免貼住框

        # 計算標籤文字的寬度和高度
        w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

        # 設置標籤的左上角座標
        label_x = center_x - w // 2

        # 繪製標籤背景
        label_p1 = (label_x - 5, label_y - h - 5)  # 標籤背景左上角
        label_p2 = (label_x + w + 5, label_y + 5)  # 標籤背景右下角
        cv2.rectangle(image, label_p1, label_p2, color, -1, cv2.LINE_AA)

        # 繪製標籤文字
        cv2.putText(image, label, (label_x, label_y), 0, 1, txt_color, 2, cv2.LINE_AA)
focus_value = 20  # 调整为适合的值，通常范围为 0 到 255
cap.set(cv2.CAP_PROP_FOCUS, focus_value)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, iou=0.1, conf=0.9, persist=True, device="cuda")
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

    for track_id, box in zip(track_ids, results[0].boxes.data):
        if box[-1] == 0:
            box_label(frame, box, 'NO.' + str(track_id), (0, 255, 0))
            x, y, x2, y2 = map(int, box[:4])
            x -= 10
            y -= 20
            x2 += 10
            y2 += 20

            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 1:
                cropped_img = frame[y:y2, x:x2]
                cv2.imshow('frame', cropped_img)
                if not cropped_img.size == 0:
                    _, h = track[-2]
                    frame = draw_frame(cropped_img, frame)
                    length_cm, width_cm, curvature, head_area, tail_area = convent_image(cropped_img)
                    grade = classify_cucumber(length_cm, width_cm, curvature, head_area, tail_area,
                                                length_optimal_max, width_optimal_max,
                                                curvature_threshold, area_difference_threshold)
                    level_result = grade
                    if h < size[1] - 250 <= y:
                        total_gherkin += 1
                        if level_result == "良":
                            level_A += 1
                        elif level_result == "優":
                            level_S += 1
                        elif level_result == "佳":
                            level_B += 1
                        elif level_result == "可":
                            level_C += 1


    cv2.line(frame, (200, size[1] - 250), (400, size[1] - 250), (0, 0, 255), 2, 4)
    cv2.imshow("result", frame)
    key = cv2.waitKey(1)
    if key == 27: #esc
        break

cap.release()
cv2.destroyAllWindows()