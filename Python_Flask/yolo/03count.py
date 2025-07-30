from collections import Counter
import os

label_dir = r"D:\圖\foot\FOOT1\2"
counter = Counter()

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, file), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].isdigit():
                counter[int(parts[0])] += 1

print("各類別統計：")
for k in sorted(counter):
    print(f"Class {k}: {counter[k]} 筆")
