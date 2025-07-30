import os
import random
import shutil
from tqdm import tqdm
from collections import Counter

# === 第一步：從 base_dir 分類出 images_dir 與 labels_dir（只留成對資料） ===
base_dir = r"D:\圖\2(需要把0跟2調換)"
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

print("📦 正在分類成對圖片與標註檔...")
for file in tqdm(os.listdir(base_dir), desc="分類中"):
    base, ext = os.path.splitext(file)
    if ext.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    image_path = os.path.join(base_dir, file)
    label_path = os.path.join(base_dir, base + '.txt')
    if os.path.exists(label_path):
        shutil.copy(image_path, os.path.join(images_dir, base + '.jpg'))
        shutil.copy(label_path, os.path.join(labels_dir, base + '.txt'))

# === 第二步：資料集切分 ===
output_dir = os.path.join(base_dir, 'datatest')
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

random.seed(42)

image_files = []
for f in os.listdir(images_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        label_file = os.path.splitext(f)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            image_files.append(f)

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
        src_image = os.path.join(images_dir, file_name)
        dst_image = os.path.join(output_dir, split, 'images', new_name + '.jpg')
        shutil.copy(src_image, dst_image)

        label_file = file_name.replace(os.path.splitext(file_name)[1], '.txt')
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(output_dir, split, 'labels', new_name + '.txt')

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"⚠️ 標籤檔不存在：{label_file}")

move_files(splits_data['train'], 'train')
move_files(splits_data['val'], 'val')
move_files(splits_data['test'], 'test')

def count_files(folder):
    images_count = len(os.listdir(os.path.join(folder, 'images')))
    labels_count = len(os.listdir(os.path.join(folder, 'labels')))
    return images_count, labels_count

print("\n🚀 數據集切分完成！")
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

# === 第三步：統計每個 split 的類別分佈（0 / 1 / 2） ===
def count_classes(label_folder):
    class_counter = Counter()
    for f in os.listdir(label_folder):
        if f.endswith(".txt"):
            with open(os.path.join(label_folder, f), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
    return class_counter

for split in splits:
    label_path = os.path.join(output_dir, split, 'labels')
    class_counts = count_classes(label_path)
    print(f"\n📊 {split.upper()} 類別統計：")
    for cls in sorted(class_counts.keys()):
        print(f"  類別 {cls}：{class_counts[cls]} 個標註")
