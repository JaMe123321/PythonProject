import cv2
import os

# 來源影片資料夾
video_folder = r"D:\圖\garbage\影五"# <- 修改成你影片所在的資料夾
# 輸出圖片資料夾（所有圖片存在一起）
output_dir = r"C:\Users\james\Desktop\2" # <- 所有圖片會儲存在這裡

# 建立圖片儲存資料夾（若不存在）
os.makedirs(output_dir, exist_ok=True)

# 找出所有 .mp4 影片
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

total_frames = 0  # 用來記錄總共幾張圖片

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    video_name = os.path.splitext(video_file)[0]  # 去掉副檔名

    print(f"🎞️ 開始處理：{video_file}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{video_file}")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = f"{video_name}_frame{frame_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)

        if cv2.imwrite(filepath, frame):
            print(f"✅ 寫入 {filename}")
            total_frames += 1
        else:
            print(f"❌ 寫入失敗：{filename}")
        frame_count += 1

    cap.release()
    print(f"✅ 完成 {video_file}：共 {frame_count} 張圖片\n")

print(f"🎉 所有影片處理完成，共產生 {total_frames} 張圖片！")
