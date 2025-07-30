import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"顯示卡名稱: {torch.cuda.get_device_name(0)}")
