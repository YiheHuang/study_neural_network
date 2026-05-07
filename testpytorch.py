import torch

# 1. 检查 CUDA 是否可用 (最关键)
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 2. 查看可用的 GPU 数量
print(f"GPU 数量: {torch.cuda.device_count()}")

# 3. 查看当前显卡名称
if torch.cuda.is_available():
    print(f"当前显卡: {torch.cuda.get_device_name(0)}")

print(torch.__version__)