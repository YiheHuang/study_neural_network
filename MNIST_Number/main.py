import torch  # 导入 PyTorch 主库，用于张量计算和自动求导。
import torch.nn as nn  # 导入神经网络模块，并简写为 nn。
import torch.optim as optim  # 导入优化器模块，并简写为 optim。
from torchvision import datasets, transforms  # 导入常用数据集接口和图像预处理工具。
from torchvision.transforms import InterpolationMode  # 导入插值模式枚举，供仿射变换使用。
from torchvision.transforms import functional as TF  # 导入函数式图像变换接口，用于 TTA。
from torch.utils.data import DataLoader  # 导入 DataLoader，用于按批次加载数据。

# 1. 定义训练与测试预处理（训练使用 RandomAffine 增强）
train_transform = transforms.Compose([  # 将训练阶段预处理步骤按顺序组合。
    transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),  # 随机小角度旋转和平移，增强泛化能力。
    transforms.ToTensor(),  # 把 PIL 图像转为张量，并把像素值缩放到 [0,1]。
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的均值和标准差做标准化。
])  # 结束训练预处理定义。

test_transform = transforms.Compose([  # 将测试阶段预处理步骤按顺序组合。
    transforms.ToTensor(),  # 把测试图像转为张量。
    transforms.Normalize((0.1307,), (0.3081,))  # 对测试图像做同分布标准化。
])  # 结束测试预处理定义。

# 2. 下载训练集合测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)  # 构建训练集对象。
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)  # 构建测试集对象。

# 3. DataLoader 批量加载数据，形成 Batch
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练集每批 64，打乱顺序。
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # 测试集每批 1000，不打乱顺序。

# 4. 定义 CNN 神经网络结构

class SimpleCNN(nn.Module):  # 定义一个简单卷积神经网络类，继承 nn.Module。
    def __init__(self):  # 初始化网络结构。
        super(SimpleCNN, self).__init__()  # 调用父类构造函数，完成基础初始化。

        self.features = nn.Sequential(  # 定义卷积特征提取部分。
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 第一层卷积：1 通道输入 -> 32 通道输出。
            nn.BatchNorm2d(32),  # 对第一层卷积输出做批归一化，稳定训练过程。
            nn.ReLU(),  # 第一层激活函数，引入非线性。
            nn.MaxPool2d(2),  # 第一次池化，空间尺寸减半：28x28 -> 14x14。
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 第二层卷积：32 通道 -> 64 通道。
            nn.BatchNorm2d(64),  # 对第二层卷积输出做批归一化，提升收敛稳定性。
            nn.ReLU(),  # 第二层激活函数。
            nn.MaxPool2d(2)  # 第二次池化，空间尺寸再减半：14x14 -> 7x7。
        )  # 结束特征提取模块。

        self.classifier = nn.Sequential(  # 定义分类头（全连接层部分）。
            nn.Linear(64 * 7 * 7, 128),  # 把展平后的特征映射到 128 维隐藏层。
            nn.ReLU(),  # 隐藏层激活函数。
            nn.Dropout(p=0.3),  # 随机失活 30% 神经元，减少过拟合。
            nn.Linear(128, 10)  # 输出 10 维 logits，对应数字 0~9。
        )  # 结束分类模块。

    # 定义前向传播
    def forward(self, x):  # 定义输入 x 经过网络得到输出的过程。
        x = self.features(x)  # 先经过卷积和池化提取特征。
        x = torch.flatten(x, 1)  # 从第 1 维开始展平，保留 batch 维度。
        logits = self.classifier(x)  # 展平特征输入分类头得到 logits。
        return logits  # 返回模型输出（未经过 softmax）。


# 5. 初始化模型、损失函数、优化器与训练配置
model = SimpleCNN()  # 实例化 CNN 模型。
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数，用于多分类任务。
num_epochs = 15  # 定义总训练轮数。
tta_times = 5  # 定义测试时 TTA 次数（包含原图一次预测）。
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 使用 AdamW 优化器并加入权重衰减。
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)  # 使用余弦退火调度学习率。

# 6. 训练模型
def train(model, device, train_loader, criterion, optimizer, epoch):  # 定义单个 epoch 的训练函数。
    model.train()  # 切换到训练模式（启用 Dropout 等训练行为）。
    for batch_idx, (data, target) in enumerate(train_loader):  # 按批次遍历训练数据。
        data, target = data.to(device), target.to(device)  # 把输入和标签移动到 CPU/GPU 设备上。
        optimizer.zero_grad()  # 清空上一轮反向传播累积的梯度。

        output = model(data)  # 前向传播，得到当前批次预测结果。

        loss = criterion(output, target)  # 计算当前批次损失。

        loss.backward()  # 反向传播，计算每个参数的梯度。

        optimizer.step()  # 使用优化器根据梯度更新参数。

        if batch_idx % 100 == 0:  # 每训练 100 个 batch 打印一次日志。
            print(f"Train Epoch : {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss : {loss.item():.6f}")  # 输出当前进度和损失值。


# 7. 测试模型（含 TTA）
def tta_predict(model, data, tta_times):  # 定义 TTA 预测函数，对同一批样本做多次增强并平均输出。
    logits_sum = model(data)  # 第一次使用原图做预测并作为累加起点。
    for _ in range(tta_times - 1):  # 其余次数使用随机仿射增强后预测。
        angle, translations, scale, shear = transforms.RandomAffine.get_params(  # 采样一次随机仿射参数。
            degrees=(-10, 10),  # 旋转角度范围与训练增强保持一致。
            translate=(0.08, 0.08),  # 平移比例范围与训练增强保持一致。
            scale_ranges=None,  # 不做随机缩放。
            shears=None,  # 不做随机错切。
            img_size=[28, 28],  # 指定 MNIST 图像尺寸。
        )  # 结束随机参数采样。
        augmented = TF.affine(  # 对整批数据应用同一组仿射参数。
            data,  # 输入待增强的 batch 张量。
            angle=angle,  # 使用采样得到的旋转角度。
            translate=translations,  # 使用采样得到的平移像素。
            scale=scale,  # 使用采样得到的缩放系数（这里为 1.0）。
            shear=shear,  # 使用采样得到的错切参数（这里为 0）。
            interpolation=InterpolationMode.BILINEAR,  # 使用双线性插值，减少变换锯齿。
        )  # 完成一次 TTA 仿射增强。
        logits_sum += model(augmented)  # 把增强图像的预测 logits 累加。
    return logits_sum / tta_times  # 返回多次预测平均后的 logits。


def test(model, device, test_loader, criterion, tta_times=1):  # 定义测试函数，在测试集上评估模型。
    model.eval()  # 切换到评估模式（关闭 Dropout 等训练特性）。
    test_loss = 0.0  # 初始化测试损失累计值。
    correct = 0  # 初始化预测正确样本数。
    with torch.no_grad():  # 关闭梯度计算，减少显存占用并加速推理。
        for data, target in test_loader:  # 按批次遍历测试数据。
            data, target = data.to(device), target.to(device)  # 把测试数据移动到同一设备。
            output = tta_predict(model, data, tta_times) if tta_times > 1 else model(data)  # 根据配置选择是否启用 TTA。
            test_loss += criterion(output, target).item()  # 累加当前 batch 的损失标量。
            pred = output.argmax(dim=1, keepdim=True)  # 取每个样本最大 logit 的类别作为预测值。
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计当前 batch 预测正确数量并累加。

        avg_test_loss = test_loss / len(test_loader)  # 计算整个测试集的平均 batch 损失。
        accuracy = correct / len(test_loader.dataset)  # 计算测试准确率小数形式。
        print(f"\nTest set: Average loss: {avg_test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100 * accuracy:.2f}%)")  # 打印测试损失与准确率。
        return avg_test_loss, accuracy  # 返回测试损失与准确率给主流程用于保存最佳模型。


if __name__ == "__main__":  # 仅当本文件被直接运行时才执行以下主流程。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU（可用时）或 CPU。
    model.to(device)  # 将模型参数迁移到目标计算设备。
    best_accuracy = 0.0  # 记录当前最优测试准确率。
    best_epoch = 0  # 记录最优准确率对应的 epoch。

    for epoch in range(1, num_epochs + 1):  # 按配置训练若干轮，epoch 从 1 开始计数。
        train(model, device, train_loader, criterion, optimizer, epoch)  # 执行一轮训练。
        avg_test_loss, accuracy = test(model, device, test_loader, criterion, tta_times=tta_times)  # 每轮训练后在测试集评估一次（启用 TTA）。
        if accuracy > best_accuracy:  # 若当前准确率更好，则更新最佳记录并保存模型。
            best_accuracy = accuracy  # 更新最佳准确率值。
            best_epoch = epoch  # 记录最佳轮次。
            torch.save(model.state_dict(), "mnist_cnn_best.pth")  # 保存当前最佳模型参数。
            print(f"Best model updated at epoch {epoch}, accuracy: {best_accuracy * 100:.2f}%")  # 打印最佳模型更新日志。
        scheduler.step()  # 每个 epoch 结束后更新学习率。
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")  # 打印当前学习率，便于观察调度效果。

    torch.save(model.state_dict(), "mnist_cnn_last.pth")  # 额外保存最后一轮模型参数，便于复现实验。
    print(f"\n最佳模型已保存到 mnist_cnn_best.pth（epoch={best_epoch}, acc={best_accuracy * 100:.2f}%）")  # 提示最佳模型保存结果。
    print("最后一轮模型已保存到 mnist_cnn_last.pth")  # 提示最后一轮模型保存完成。