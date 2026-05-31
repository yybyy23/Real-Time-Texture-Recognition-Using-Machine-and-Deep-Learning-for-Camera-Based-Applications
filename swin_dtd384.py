import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os

# 🟢 [关键] 强制使用镜像站防止预训练权重下载失败
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from PIL import Image
import timm
import pandas as pd
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================================

# ================= 配置区域 (Swin 冲刺版) =================
# 🔴 请确保这里的路径指向 DTD 的 images 文件夹
DATA_PATH = "/root/TexRec/dtd/images"

IMG_SIZE = 384
BATCH_SIZE = 16  # 物理显存占用
ACCUMULATION_STEPS = 4  # 等效 BatchSize = 16 * 4 = 64

NUM_EPOCHS = 50
LEARNING_RATE = 3e-5  # Swin-Base 微调建议使用较低学习率
MIXUP_ALPHA = 0.2
NUM_CLASSES = 47  # DTD 数据集有 47 个类别

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 启动 Swin-Base-384 (DTD) | 设备: {device} | 图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
print(f"🔥 有效 Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS} | 类别数: {NUM_CLASSES}")

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
        raise FileNotFoundError(f"❌ 找不到数据集路径: {DATA_PATH}")

    full_dataset = datasets.ImageFolder(root=DATA_PATH)
    classes = full_dataset.classes

    # 随机划分 80% 训练, 20% 验证 (固定随机种子保证可复现)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))

    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset = DatasetWrapper(val_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ 数据加载完成: 训练集 {train_size} 张, 验证集 {val_size} 张.")
    return train_loader, val_loader, classes


# ================= 2. Mixup 策略 =================
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ================= 3. 训练流程 =================
def train_swin(model, train_loader, val_loader, num_epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 使用 Mixup
            inputs_m, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            inputs_m, targets_a, targets_b = map(torch.autograd.Variable, (inputs_m, targets_a, targets_b))

            outputs = model(inputs_m)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # 梯度累积
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS * inputs.size(0)

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
            torch.save(model.state_dict(), 'swin_dtd_384_best.pth')
            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | 🔥 New Best!')
        else:
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}')

    return best_acc


# ================= 4. 详细评估与混淆矩阵 =================
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

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\n" + "=" * 60)
    print("详细分类性能报告 (Classification Report)")
    print("=" * 60)
    print(report)

    # 绘制大型混淆矩阵 (适配 47 个类别)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(pd.DataFrame(cm, index=classes, columns=classes), annot=False, cmap='Blues')
    plt.title('Swin-384 Confusion Matrix on DTD Dataset', fontsize=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('./Swin_DTD_384_confusion_matrix.png', dpi=300)
    print("✅ 混淆矩阵已保存为: ./Swin_DTD_384_confusion_matrix.png")


def main():
    try:
        train_loader, val_loader, classes = get_dtd_dataloaders()
    except FileNotFoundError as e:
        print(e)
        return

    # 加载原生支持 384 的 Swin Base 版本，分类数设为 47
    print("加载模型 (Swin Transformer Base 384)...")
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=NUM_CLASSES)

    # 训练模型
    train_swin(model, train_loader, val_loader, NUM_EPOCHS)

    # 评估历史最佳模型
    print("\n训练结束，加载历史最佳模型进行最终评估...")
    model.load_state_dict(torch.load('swin_dtd_384_best.pth'))
    evaluate_model_detailed(model, val_loader, device, classes)


if __name__ == "__main__":
    main()