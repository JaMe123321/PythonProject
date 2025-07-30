from ultralytics import YOLO


def main():
    model = YOLO("yolo11m.pt")  # 載入預訓練模型

    model.train(
        data=r"C:\Users\james\PycharmProjects\PythonProject\Python_Flask\dataset_custom.yaml",  # 自訂 dataset
        imgsz=640,        # 輸入大小
        batch=8,          # 適度提升 batch size，若顯示 CUDA OOM 可改回 2 或 1
        epochs=100,
        device=0,         # 使用 CUDA:0
      #  cache=False,      # ✅ 不預載全部圖片 → 省記憶體
        workers=0,        # ✅ 禁用多進程 dataloader → 避免 Windows crash
        amp=True          # ✅ 自動混合精度訓練 → 降低 VRAM 需求、加快訓練速度
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # ✅ Windows 多進程安全防護
    main()