import os
from tqdm import tqdm

# 標註資料夾路徑
label_dir = r"D:\圖\2(需要把0跟2調換)"

# 統計
modified_count = 0
line_change_count = 0

for file in tqdm(os.listdir(label_dir), desc="更新標註"):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(label_dir, file)
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    changed = False
    for line in lines:
        parts = line.strip().split()
        if not parts or not parts[0].isdigit():
            continue
        if parts[0] == "9":
            parts[0] = "2"  # ✅ 將類別 0 改為 2
            changed = True
            line_change_count += 1
        new_lines.append(" ".join(parts) + "\n")

    if changed:
        with open(path, "w") as f:
            f.writelines(new_lines)
        modified_count += 1

print(f"\n✅ 已修改 {modified_count} 個標註檔案")
print(f"🔢 共修改 {line_change_count} 行，將類別 0 改為 2")
