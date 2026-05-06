import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 与脚本同目录，数据与模型权重放在本目录下
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_BASE, "data")

# Fashion-MNIST 各类别名称（10 类）
CLASS_NAMES = (
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)

# 官方常用归一化（整集统计近似值；与 MNIST 的 0.13/0.31 不同）
_FASHION_MEAN = (0.2860,)
_FASHION_STD = (0.3530,)

# 1. 训练与测试预处理（与基线一致：仅 ToTensor + 标准化，不做几何增强）
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_FASHION_MEAN, _FASHION_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_FASHION_MEAN, _FASHION_STD),
])

# 2. 加载 Fashion-MNIST
train_dataset = datasets.FashionMNIST(
    root=_DATA_ROOT, train=True, download=True, transform=train_transform
)
test_dataset = datasets.FashionMNIST(
    root=_DATA_ROOT, train=False, download=True, transform=test_transform
)

# 3. DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 4. CNN（与 MNIST 版结构一致：28x28 单通道 -> 10 类）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# 5. 检查点路径（实验并行时可设 FASHION_CKPT_TAG 避免覆盖）
_CKPT_TAG = os.environ.get("FASHION_CKPT_TAG", "").strip()
_CKPT_SUFFIX = f"_{_CKPT_TAG}" if _CKPT_TAG else ""
_CKPT_BEST = os.path.join(_BASE, f"fashion_cnn_best{_CKPT_SUFFIX}.pth")
_CKPT_LAST = os.path.join(_BASE, f"fashion_cnn_last{_CKPT_SUFFIX}.pth")


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch : {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss : {loss.item():.6f}"
            )


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {avg_test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({100 * accuracy:.2f}%)"
    )
    return avg_test_loss, accuracy


def _set_seed_from_env():
    """供 experiment.py 注入 FASHION_SEED，保证多组实验可复现。"""
    s = os.environ.get("FASHION_SEED")
    if s is None:
        return
    s = int(s)
    import random

    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        import numpy as np

        np.random.seed(s)
    except ImportError:
        pass


if __name__ == "__main__":
    _set_seed_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 15
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        _, accuracy = test(model, device, test_loader, criterion)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), _CKPT_BEST)
            print(
                f"Best model updated at epoch {epoch}, accuracy: {best_accuracy * 100:.2f}%"
            )
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), _CKPT_LAST)
    print(
        f"\n最佳模型已保存到 {_CKPT_BEST}（epoch={best_epoch}, acc={best_accuracy * 100:.2f}%）"
    )
    print(f"最后一轮模型已保存到 {_CKPT_LAST}")
