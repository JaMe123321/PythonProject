import os
import random
import shutil
from tqdm import tqdm  # 進度條

# ====== 整合來源資料夾（成對的 .jpg/.txt） ======
source_folders = [r"D:\紅樓垃圾\影三\all_frames(3)"]
merged_folder = r"D:\紅樓垃圾\影三\1"
merged_images = os.path.join(merged_folder, "images")
merged_labels = os.path.join(merged_folder, "labels")

# 建立 images 和 labels 資料夾
os.makedirs(merged_images, exist_ok=True)
os.makedirs(merged_labels, exist_ok=True)

# 收集所有成對的檔案
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

# 依序複製到 merged/images 與 merged/labels
for idx, (txt_path, jpg_path) in enumerate(tqdm(paired_files, desc="整合檔案")):
    new_name = f"{idx:03d}"
    try:
        shutil.copy(txt_path, os.path.join(merged_labels, new_name + '.txt'))
        shutil.copy(jpg_path, os.path.join(merged_images, new_name + '.jpg'))
    except Exception as e:
        print(f"無法複製檔案 {txt_path} 或 {jpg_path}：{e}")

print(f"\n✅ 共整合了 {len(paired_files)} 對檔案到資料夾：{merged_folder}")

# ====== 切分 train/val/test 並重新命名 ======
output_dir = os.path.join(merged_folder, 'datatest')
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# 收集有標註的圖片（從整合後的資料夾）
image_files = [f for f in os.listdir(merged_images) if os.path.exists(os.path.join(merged_labels, f.replace('.jpg', '.txt')))]
random.seed(42)
random.shuffle(image_files)

num_total = len(image_files)
num_train = int(num_total * 0.8)
num_val = int(num_total * 0.1)
num_test = num_total - num_train - num_val

splits_data = {
    'train': image_files[:num_train],
    'val': image_files[num_train:num_train + num_val],
    'test': image_files[num_train + num_val:]
}

def move_files(file_list, split):
    for idx, file_name in enumerate(tqdm(file_list, desc=f"搬移 {split} 數據集")):
        new_name = f"{idx:03d}"
        src_img = os.path.join(merged_images, file_name)
        src_lbl = os.path.join(merged_labels, file_name.replace('.jpg', '.txt'))
        dst_img = os.path.join(output_dir, split, 'images', new_name + '.jpg')
        dst_lbl = os.path.join(output_dir, split, 'labels', new_name + '.txt')

        try:
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)
        except Exception as e:
            print(f"❌ 搬移失敗：{file_name} - {e}")

for split in splits:
    move_files(splits_data[split], split)

def count_files(folder):
    images_count = len(os.listdir(os.path.join(folder, 'images')))
    labels_count = len(os.listdir(os.path.join(folder, 'labels')))
    return images_count, labels_count

print("\n🚀 整合與數據集切分完成！")
train_count = count_files(os.path.join(output_dir, 'train'))
val_count = count_files(os.path.join(output_dir, 'val'))
test_count = count_files(os.path.join(output_dir, 'test'))

print(f"訓練集：圖片數 {train_count[0]}, 標籤數 {train_count[1]}")
print(f"驗證集：圖片數 {val_count[0]}, 標籤數 {val_count[1]}")
print(f"測試集：圖片數 {test_count[0]}, 標籤數 {test_count[1]}")

if train_count[0] == train_count[1] and val_count[0] == val_count[1] and test_count[0] == test_count[1]:
    print("\n✅ 切分成功：圖片與標籤數量一致！")
else:
    print("\n⚠️ 警告：部分數據集圖片與標籤數量不一致，請檢查！")
