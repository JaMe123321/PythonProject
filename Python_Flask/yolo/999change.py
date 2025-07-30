import os
import shutil

label_folder = r"D:\紅樓垃圾\影\1"  # 修改為你的標註檔路徑
backup_folder = os.path.join(label_folder, "backup_labels")

# 建立備份資料夾
os.makedirs(backup_folder, exist_ok=True)

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_folder, filename)
        backup_path = os.path.join(backup_folder, filename)

        # 備份原始檔案
        #shutil.copyfile(file_path, backup_path)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].isdigit():
                parts[0] = "2"  # 修改 class index 為 2
                new_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

print("✅ 已完成備份與修改所有標註檔（class index 改為 2）")
