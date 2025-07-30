import os

# 指定資料夾路徑
folder_path = r"D:\圖\foot\FOOT1\1\labels"  # 替換為您的資料夾路徑

# 遍歷資料夾中的所有 .txt 檔案
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # 檢查並替換檔名中的數字部分
        if "766993860.905321" in filename:
            new_filename = filename.replace("766993860.905321", "5")

            # 舊檔案的完整路徑
            old_file_path = os.path.join(folder_path, filename)

            # 新檔案的完整路徑
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名檔案
            os.rename(old_file_path, new_file_path)
            print(f"檔案 {filename} 已更名為 {new_filename}")
