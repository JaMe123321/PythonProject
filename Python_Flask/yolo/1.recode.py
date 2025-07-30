import cv2
import os
import datetime

is_recording = False
video_writer = None

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') or key == ord('C'):
        is_recording = True
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f'output_{timestamp}.mp4'
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame.shape[1], frame.shape[0]))

    if is_recording:
        video_writer.write(frame)
        cv2.putText(frame, "Recording", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if key == ord('s') or key == ord('S'):
        is_recording = False
        if video_writer:
            video_writer.release()
            save_path = 'D:\lwj\yolov8_code\video\oo'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            os.rename(video_filename, os.path.join(save_path, video_filename))
            print(f"Video saved to: {os.path.join(save_path, video_filename)}")

    if key == ord('q') or key == ord('Q'):
        break

camera.release()
cv2.destroyAllWindows()
