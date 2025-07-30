import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from scipy.spatial.distance import euclidean

class SlidingWindow:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = deque()

    def add_value(self, value):
        if len(self.data) == self.max_size:
            self.data.popleft()
        self.data.append(value)

    def get_average(self):
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)

def calculate_red_line_distance(contour_rotated):
    leftmost_point = tuple(contour_rotated[contour_rotated[:, :, 0].argmin()][0])
    rightmost_point = tuple(contour_rotated[contour_rotated[:, :, 0].argmax()][0])

    horizontal_distance = euclidean(leftmost_point, rightmost_point)

    mid_point = ((leftmost_point[0] + rightmost_point[0]) / 2,
                 (leftmost_point[1] + rightmost_point[1]) / 2)

    window_size = 1000
    furthest_distance = 0
    furthest_point = None
    for point in contour_rotated:
        point = point[0]
        distance = np.abs((rightmost_point[1] - leftmost_point[1]) * point[0] -
                          (rightmost_point[0] - leftmost_point[0]) * point[1] +
                          rightmost_point[0] * leftmost_point[1] -
                          rightmost_point[1] * leftmost_point[0]) / horizontal_distance
        if distance > furthest_distance:
            furthest_distance = distance
            furthest_point = point

    curvature_window = SlidingWindow(window_size)
    new_curvature = furthest_distance / 100

    curvature_window.add_value(new_curvature)

    curvature = round(curvature_window.get_average(), 2)

    return curvature

def calculate_area(cropped_region, w1):
    head_part = cropped_region[:, :w1 // 3]
    tail_part = cropped_region[:, 2 * w1 // 3:]

    window_size = 1000

    head_area_window = SlidingWindow(window_size)
    tail_area_window = SlidingWindow(window_size)

    new_head_area = np.count_nonzero(head_part) / 6 ** 2
    new_tail_area = np.count_nonzero(tail_part) / 5.5 ** 2

    head_area_window.add_value(new_head_area)
    tail_area_window.add_value(new_tail_area)

    head_area_avg = round(head_area_window.get_average())
    tail_area_avg = round(tail_area_window.get_average())

    return head_area_avg, tail_area_avg


def convent_image(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (75, 75), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 30, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return 0, 0, 0, 0, 0

        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        x, y, w, h = cv2.boundingRect(contour)

        major_axis_length, minor_axis_length = axes
        if major_axis_length < minor_axis_length:
            angle += 90

        height, width = edges.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((height * sin) + (width * cos))
        new_h = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated_image = cv2.warpAffine(edges, rotation_matrix, (new_w, new_h),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        #cv2.imshow('rotated image', rotated_image)

        x1, y1, w1, h1 = cv2.boundingRect(rotated_image)

        cropped_region = rotated_image[y1:y1 + h1, x1:x1 + w1]
        contours_cropped, _ = cv2.findContours(cropped_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours_cropped:
            contour_cropped = max(contours_cropped, key=cv2.contourArea)
            if len(contour_cropped) < 5:
                return 0, 0, 0, 0, 0

            # 計算長度
            leftmost_point = tuple(contour_cropped[contour_cropped[:, :, 0].argmin()][0])
            rightmost_point = tuple(contour_cropped[contour_cropped[:, :, 0].argmax()][0])
            length_px = np.linalg.norm(np.array(rightmost_point) - np.array(leftmost_point))
            length_cm = round(length_px / 22, 1)

            # 計算中心水平線的寬度
            M = cv2.moments(contour_cropped)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])  # 輪廓中心的 x 座標
                center_y = int(M["m01"] / M["m00"])  # 輪廓中心的 y 座標
            else:
                center_x, center_y = 0, 0

            # 找到靠近中心垂直線的輪廓點
            vertical_points = []
            for point in contour_cropped:
                point = point[0]
                if abs(point[0] - center_x) < 2:  # 在中心垂直線附近的點（容差設置為2像素）
                    vertical_points.append(point)

            if len(vertical_points) >= 2:
                vertical_points = np.array(vertical_points)
                topmost_point = vertical_points[vertical_points[:, 1].argmin()]  # 最上方的點
                bottommost_point = vertical_points[vertical_points[:, 1].argmax()]  # 最下方的點
                width_px = np.linalg.norm(bottommost_point - topmost_point)  # 計算垂直距離
                width_cm = round(width_px / 28, 1)  # 根據比例轉換為厘米
            else:
                width_cm = 0

            curvature = calculate_red_line_distance(contour_cropped)
            curvature = curvature/length_cm
            head_area_avg, tail_area_avg = calculate_area(cropped_region, w1)

            return length_cm, width_cm, curvature, head_area_avg, tail_area_avg

    return 0, 0, 0, 0, 0

def classify_cucumber(length_cm, width_cm, curvature, head_area_avg, tail_area_avg,
                      length_threshold, width_threshold, curvature_threshold, area_difference_threshold, width_max, width_min):
    grade = None
    # 分級
    if length_cm > length_threshold:
        grade = "上"
    elif width_cm > width_max:
        grade = "上"
    elif length_cm < length_threshold and length_cm > width_threshold:
        grade = "中"
    elif width_cm < width_max and width_cm > width_min:
        grade = "中"
    elif length_cm < width_threshold:
        grade = "下"
    elif width_cm < width_min:
        grade = "下"
    return grade



def draw_frame(cropped_img, frame):
    length_cm, width_cm, curvature, head_area_avg, tail_area_avg= convent_image(cropped_img)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font_path = r"C:\Users\luk80\PycharmProjects\yolov8_code\flask_v7\font\kaiu.ttf"
    font = ImageFont.truetype(font_path, 30)
    text_color = (0, 0, 0)

    draw.text((10, 10), f"果長: {length_cm:.2f} cm", font=font, fill=text_color)
    draw.text((10, 40), f"果寬: {width_cm:.2f} cm", font=font, fill=text_color)
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    return frame
