import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import pandas as pd
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')  # 服务器端绘图必须用 Agg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================================

# ================= 配置区域 =================
# 🔴 请确保这里的路径指向 DTD 的 images 文件夹，该文件夹下应有 47 个子文件夹
DATA_PATH = "/root/TexRec/dtd/images"

IMG_SIZE = 384
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 47  # DTD 数据集有 47 个类别

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device} | 图像尺寸: {IMG_SIZE}x{IMG_SIZE} | 类别数: {NUM_CLASSES}")

# ================= 1. 数据增强与加载 =================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 包装器：为了给 train 和 val 不同的 transform
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dtd_dataloaders():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据集路径: {DATA_PATH}")

    # 使用 ImageFolder 读取所有图片
    full_dataset = datasets.ImageFolder(root=DATA_PATH)
    classes = full_dataset.classes  # 获取 47 个类别的名称

    # 随机划分 80% 训练, 20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))

    # 应用增强
    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset = DatasetWrapper(val_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ 数据加载完成: 训练集 {train_size} 张, 验证集 {val_size} 张.")
    return train_loader, val_loader, classes


# ================= 2. B-CNN 模型定义 =================
class BilinearResNet(nn.Module):
    def __init__(self, num_classes=47):
        super(BilinearResNet, self).__init__()
        # 使用 ResNet-34 作为特征提取器
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # 截取特征提取层 (去掉最后的 AdaptiveAvgPool2d 和 Linear 层)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 线性分类器: 输入维度 = 通道数 * 通道数 = 512 * 512
        self.classifier = nn.Linear(512 * 512, num_classes)

        # 初始化分类器权重
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 1. 提取特征图: shape = (Batch, 512, H, W)
        x = self.features(x)

        # 2. 调整形状以准备外积: (Batch, 512, H*W)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)

        # 3. 双线性池化 (Bilinear Pooling)
        x = torch.bmm(x, x.transpose(1, 2)) / (height * width)

        # 4. 展平: shape = (Batch, 512 * 512)
        x = x.view(batch_size, channels * channels)

        # 5. 有符号平方根归一化 (Signed Square Root Normalization)
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-5)

        # 6. L2 归一化 (L2 Normalization)
        x = F.normalize(x, p=2, dim=1)

        # 7. 分类输出
        x = self.classifier(x)
        return x


# ================= 3. 训练流程 =================
def train_bcnn(model, train_loader, val_loader, num_epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # B-CNN 优化策略：分类器学习率大一点，特征提取器学习率小一点
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_acc = 0.0

    print(f"🚀 开始 B-CNN (DTD 数据集) 训练...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- 验证 ---
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'bcnn_dtd_best.pth')
            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | 🔥 New Best!')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}')

    return best_acc


# ================= 4. 评估与混淆矩阵 =================
def evaluate_model_detailed(model, dataloader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    print("\n正在进行详细评估...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"整体准确率 (Accuracy): {acc:.4f}")

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\n" + "=" * 60)
    print("详细分类性能报告 (Classification Report)")
    print("=" * 60)
    print(report)

    # 画混淆矩阵 (47类太多，关闭数字显示 annot=False，调整图表大小)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(24, 20))  # 把图表拉大，防止标签重叠
    sns.heatmap(pd.DataFrame(cm, index=classes, columns=classes), annot=False, cmap='Blues', cbar=True)
    plt.title('B-CNN Confusion Matrix on DTD Dataset', fontsize=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)

    # 旋转 x 轴标签以防重叠
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig('./BCNN_DTD_confusion_matrix.png', dpi=300)
    print("✅ DTD 混淆矩阵已保存为: ./BCNN_DTD_confusion_matrix.png")


def main():
    try:
        train_loader, val_loader, classes = get_dtd_dataloaders()
    except FileNotFoundError as e:
        print(e)
        return

    # 实例化 B-CNN 模型
    model = BilinearResNet(num_classes=NUM_CLASSES)

    # 训练模型
    train_bcnn(model, train_loader, val_loader, NUM_EPOCHS)

    # 评估历史最佳模型
    print("\n训练结束，加载历史最佳模型进行最终评估...")
    model.load_state_dict(torch.load('bcnn_dtd_best.pth'))
    evaluate_model_detailed(model, val_loader, device, classes)


if __name__ == "__main__":
    main()