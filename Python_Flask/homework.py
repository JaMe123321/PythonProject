import torch
import pandas as pd

# 自動偵測是否有 CUDA 裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (a) 設定參數權重 (Weighting) 與偏移 (bias)
weights = torch.tensor([0.20, 0.20, 0.20, 0.40], dtype=torch.float32, device=device)  # 權重丟到GPU
bias = torch.tensor(5.0, dtype=torch.float32, device=device)  # 偏移量丟到GPU

# 擊發函數 σ(s)：使用 ReLU（正值輸出，負值歸零）
def activation_function(x):
    return torch.maximum(x, torch.tensor(0.0, device=device))

# (c) 20位考生4科成績資料
students = [f"考生{i+1}" for i in range(20)]
scores_list = [
    [86, 58, 74, 87],    [73, 68, 69, 79],    [90, 79, 72, 71],    [77, 62, 85, 91],
    [82, 76, 84, 78],    [89, 87, 82, 73],    [74, 67, 77, 80],    [91, 82, 86, 89],
    [68, 72, 65, 77],    [85, 80, 81, 82],    [76, 61, 70, 74],    [80, 66, 78, 84],
    [83, 79, 76, 79],    [75, 63, 71, 73],    [78, 68, 67, 76],    [66, 60, 74, 70],
    [81, 77, 82, 85],    [88, 85, 90, 93],    [70, 65, 66, 72],    [79, 74, 80, 86]
]

# 轉成 Tensor 並搬到 GPU
scores_tensor = torch.tensor(scores_list, dtype=torch.float32, device=device)

# (c) 計算總分
# 總分 = activation(成績 * 權重 + 偏移)
total_scores = activation_function(torch.matmul(scores_tensor, weights) + bias)

# 整理成表格（成績 + 總分）
original_df = pd.DataFrame(scores_list, columns=["國文", "英文", "數學", "專業科目"])
original_df["考生"] = students
original_df["總分"] = total_scores.cpu().numpy()  # 從GPU搬回CPU才能轉成numpy處理表格

# (d) 排序並標示錄取狀態
sorted_df = original_df.sort_values(by="總分", ascending=False).reset_index(drop=True)

admission_status = []
for idx in range(len(sorted_df)):
    if idx < 12:
        admission_status.append('正取')
    elif idx < 15:
        admission_status.append('備取')
    else:
        admission_status.append('不錄取')

sorted_df["錄取情況"] = admission_status

# 輸出成績與錄取情況
print("\n📚 20位考生完整成績與總分：")
print(original_df[["考生", "國文", "英文", "數學", "專業科目", "總分"]])

print("\n🏆 排名結果與錄取情況：")
print(sorted_df[["考生", "國文", "英文", "數學", "專業科目", "總分", "錄取情況"]])
