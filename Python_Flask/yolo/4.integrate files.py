import os
import shutil
from tqdm import tqdm  # 進度條

# 設定來源資料夾與目標資料夾
source_folders = [r"D:\紅樓垃圾\影二\all_frames"]  # 請換成你的實際資料夾名稱
target_folder = r"D:\紅樓垃圾\影\1"

# 建立目標資料夾（若不存在）
os.makedirs(target_folder, exist_ok=True)

# 收集所有成對存在的 .jpg 和 .txt 檔案
paired_files = []

for folder in source_folders:
    try:
        files = os.listdir(folder)
        base_names = set(os.path.splitext(f)[0] for f in files)

        for name in base_names:
            txt_path = os.path.join(folder, name + '.txt')
            jpg_path = os.path.join(folder, name + '.jpg')

            if os.path.isfile(txt_path) and os.path.isfile(jpg_path):
                paired_files.append((txt_path, jpg_path))
    except Exception as e:
        print(f"無法存取資料夾 {folder}：{e}")

# 重新排序並複製到新資料夾，顯示進度條
for idx, (txt_path, jpg_path) in enumerate(tqdm(paired_files, desc="整合檔案")):
    new_name = f"{idx:03d}"
    try:
        shutil.copy(txt_path, os.path.join(target_folder, new_name + '.txt'))
        shutil.copy(jpg_path, os.path.join(target_folder, new_name + '.jpg'))
    except Exception as e:
        print(f"無法複製檔案 {txt_path} 或 {jpg_path}：{e}")

print(f"\n共整合了 {len(paired_files)} 對檔案到資料夾：{target_folder}")
