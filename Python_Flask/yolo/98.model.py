from ultralytics import YOLO
import os
import torch
from thop import profile  # <-- 這是 FLOPs 計算器
# === 這邊是你指定的路徑 ===
weight_path = r"Z:\專題\datatest\train7\weights\best.pt"
output_folder = r"Z:\專題\datatest\train7\weights"


def export_model_info(weight_path, output_folder):
    print(f"🔍 載入模型：{weight_path}")
    model = YOLO(weight_path)

    # 收集所有資訊
    info_lines = []

    info_lines.append("📦 【模型基本資訊】")
    info_lines.append(f"模型檔案: {os.path.basename(weight_path)}")

    # 類別數量
    if hasattr(model.model, 'names'):
        model_nc = len(model.model.names)
        info_lines.append(f"類別數量 (nc): {model_nc}")
    else:
        model_nc = '未知'
        info_lines.append("❓ 類別數量: 無法讀取 (可能是舊版模型)")

    # 輸入大小
    imgsz = getattr(model.model, 'args', {}).get('imgsz', 640)
    if isinstance(imgsz, list):  # 有些imgsz是list
        imgsz = imgsz[0]
    info_lines.append(f"輸入大小 (imgsz): {imgsz}")

    # 模型參數數量
    total_params = sum(p.numel() for p in model.model.parameters())
    info_lines.append(f"模型參數總數 (parameters): {total_params:,}")

    # 層數
    total_layers = len(list(model.model.modules()))
    info_lines.append(f"模型結構深度 (layers): {total_layers}")

    # 計算 FLOPs
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to('cpu')  # 用cpu算比較穩
    try:
        flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
        info_lines.append(f"推論運算量 (GFLOPs): {flops / 1e9:.2f} GFLOPs")
    except Exception as e:
        info_lines.append(f"推論運算量 (GFLOPs): 無法計算，錯誤：{e}")

    info_lines.append("\n📋 【類別名稱列表】")
    if hasattr(model.model, 'names'):
        for idx, name in model.model.names.items():
            info_lines.append(f"{idx}: {name}")
    else:
        info_lines.append("❌ 無類別名稱資訊")

    info_lines.append("\n📑 【模型層級結構簡述】")
    for name, module in model.model.named_modules():
        if module.__class__.__name__ in ['Conv', 'C3', 'C3k2', 'SPPF', 'Detect', 'C2f', 'C2fAttn', 'C2fB', 'C2PSA']:
            info_lines.append(f"{name}: {module}")



    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    output_txt_path = os.path.join(output_folder, "model_info.txt")

    print(f"\n💾 儲存到: {output_txt_path}")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for line in info_lines:
            f.write(line + "\n")

    print("\n✅ 完成！已將模型資訊存成 TXT 檔案！")


# 執行
export_model_info(weight_path, output_folder)
