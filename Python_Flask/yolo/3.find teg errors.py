import os
import shutil
from tqdm import tqdm  # 用於顯示進度條

# 標籤資料夾路徑
label_folder = r"D:\紅樓垃圾\datatest\datatest2\train\labels"  # 訓練集標籤資料夾
val_label_folder = r"D:\紅樓垃圾\datatest\datatest2\val\labels"  # 驗證集標籤資料夾
classes_file = r"Z:\專題\紅樓垃圾\picture1\classes.txt"  # 類別檔案路徑

# 建立備份資料夾
backup_folder = r"D:\紅樓垃圾\datatest\backup"
os.makedirs(backup_folder, exist_ok=True)

# 自動修正開關（True 表示自動改錯誤類別為 0）
auto_fix = True

# 讀取類別數，從類別檔案中讀取每一行，去除空白行
with open(classes_file, "r", encoding="utf-8") as f:
    class_list = [line.strip() for line in f if line.strip()]
max_class_index = len(class_list) - 1  # 類別索引範圍
print(f"✔️ 找到 {len(class_list)} 個類別，允許的 class index：0 ~ {max_class_index}\n")

# 錯誤與修正統計變數
error_count = 0  # 紀錄錯誤數量
fixed_count = 0  # 紀錄修正數量
error_files = []  # 儲存錯誤檔案資訊 (檔名, 行數, 錯誤類別)

def fix_labels(folder):
    """修正標籤檔案的類別索引錯誤及特定類別轉換"""
    global error_count, fixed_count  # 使用全域變數以進行統計

    # 遍歷資料夾中的所有標籤檔案
    for root, _, files in os.walk(folder):
        for file in tqdm(files, desc=f"處理資料夾: {os.path.basename(folder)}"):
            if file.endswith(".txt"):  # 僅處理 .txt 檔案
                file_path = os.path.join(root, file)  # 完整檔案路徑

                # 備份原始標籤檔案，確保數據安全
                backup_path = os.path.join(backup_folder, file)
                try:
                    shutil.copy(file_path, backup_path)  # 複製至備份資料夾
                except Exception as e:
                    print(f"❌ 無法備份檔案: {file_path} - {e}")
                    continue  # 若備份失敗，跳過該檔案

                try:
                    # 開啟檔案並讀取所有行
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    new_lines = []  # 儲存修正後的內容
                    modified = False  # 標記是否有修改
                    for i, line in enumerate(lines):
                        parts = line.strip().split()  # 以空格拆分每一行
                        # 檢查第一個元素是否為數字（類別索引）
                        if len(parts) >= 1 and parts[0].isdigit():
                            class_index = int(parts[0])  # 取得類別索引
                            # 檢查類別索引是否超過範圍
                            if class_index > max_class_index:
                                print(f"❌ 檔案：{file} | 第 {i+1} 行 | 錯誤類別 index：{class_index}")
                                error_count += 1
                                error_files.append((file, i + 1, class_index))  # 記錄錯誤資訊
                                if auto_fix:
                                    parts[0] = "0"  # 自動修正為類別 0
                                    fixed_count += 1
                                    modified = True
                            # 固定將類別 1 修改為類別 0
                            elif class_index == 1:
                                parts[0] = "0"  # 修正類別
                                fixed_count += 1
                                modified = True
                        # 合併修正後的行
                        new_lines.append(" ".join(parts) + "\n")

                    # 僅當檔案有修改時才寫回檔案
                    if modified:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)  # 將修改內容寫入
                        print(f"✅ 已修正: {file_path}")
                except Exception as e:
                    print(f"❌ 無法處理檔案: {file_path} - {e}")

# 修正訓練集與驗證集的標籤檔案
fix_labels(label_folder)
fix_labels(val_label_folder)

# 統計結果報告
print(f"\n🚀 檢查完成，共發現 {error_count} 處錯誤。")
if auto_fix:
    print(f"✅ 自動修正了 {fixed_count} 處錯誤。")

# 顯示所有錯誤檔案及其行數
if error_files:
    print("\n📂 錯誤檔案清單：")
    for file, line_num, class_index in error_files:
        print(f"檔案：{file} | 第 {line_num} 行 | 錯誤類別 index：{class_index}")
else:
    print("🎉 沒有發現任何錯誤標註！")

print("🚀 所有標籤檔案修正完成！")
