import os
import shutil
import random
from tqdm import tqdm

# 設定來源資料夾
image_dirs = [r"D:\圖\1\images",r"D:\圖\2(需要把0跟2調換)\images",r"D:\圖\garbage\all\images"]
label_dirs = [r"D:\圖\1\labels", r"D:\圖\2(需要把0跟2調換)\labels",r"D:\圖\garbage\all\labels"]

# 輸出整合資料夾
base_output = r"D:\圖\garbage\all3"
output_images = os.path.join(base_output, "images")
output_labels = os.path.join(base_output, "labels")
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# 步驟一：整合資料
counter = 0
all_files = []

for img_dir, lbl_dir in zip(image_dirs, label_dirs):
    for file in sorted(os.listdir(img_dir)):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(file)[0]
        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(lbl_dir, base_name + '.txt')

        if os.path.exists(label_path):
            new_name = f"{counter:03d}"
            new_img_path = os.path.join(output_images, new_name + '.jpg')
            new_lbl_path = os.path.join(output_labels, new_name + '.txt')

            shutil.copy(img_path, new_img_path)
            shutil.copy(label_path, new_lbl_path)

            all_files.append(new_name)
            counter += 1

print(f"✅ 整合完成，共處理 {counter} 對圖片與標註。")

# 步驟二：切分資料集
random.seed(42)
random.shuffle(all_files)

num_total = len(all_files)
num_train = int(num_total * 0.8)
num_val = int(num_total * 0.1)
num_test = num_total - num_train - num_val

splits_data = {
    'train': all_files[:num_train],
    'val': all_files[num_train:num_train + num_val],
    'test': all_files[num_train + num_val:]
}

# 步驟三：建立子資料夾並搬移資料
for split in ['train', 'val', 'test']:
    split_img_dir = os.path.join(base_output, 'datatest', split, 'images')
    split_lbl_dir = os.path.join(base_output, 'datatest', split, 'labels')
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

    for name in tqdm(splits_data[split], desc=f"搬移 {split}"):
        shutil.copy(os.path.join(output_images, name + '.jpg'), os.path.join(split_img_dir, name + '.jpg'))
        shutil.copy(os.path.join(output_labels, name + '.txt'), os.path.join(split_lbl_dir, name + '.txt'))

print("\n🚀 數據集切分完成！")
