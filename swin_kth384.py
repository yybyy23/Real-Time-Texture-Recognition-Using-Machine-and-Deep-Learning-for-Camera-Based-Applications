import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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

# ================= 配置区域 (冲刺版) =================
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"

# 🟢 1. 分辨率提升至 384
IMG_SIZE = 384

# 🟢 2. 防显存溢出 & 梯度累积
BATCH_SIZE = 16  # 物理显存占用
ACCUMULATION_STEPS = 4  # 等效 BatchSize = 16 * 4 = 64

NUM_EPOCHS = 50

# 🟢 3. 学习率下调 (防止过拟合和剧烈震荡)
LEARNING_RATE = 3e-5
MIXUP_ALPHA = 0.2

CLASSES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cork', 'cotton',
    'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool'
]
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 启动 Swin-Base-384 训练 | 设备: {device} | 图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
print(f"🔥 有效 Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS} | 学习率: {LEARNING_RATE}")


# 1. 数据加载
class KTHTIPS2bDataset(Dataset):
    def __init__(self, root_dir, samples_to_use, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}

        for cls_name in CLASSES:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder): continue
            cls_idx = self.class_to_idx[cls_name]
            for sample_name in samples_to_use:
                sample_folder = os.path.join(cls_folder, sample_name)
                if not os.path.isdir(sample_folder): continue
                for img_name in os.listdir(sample_folder):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.image_paths.append(os.path.join(sample_folder, img_name))
                        self.labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label


# 2. Mixup 策略
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


# 3. 数据增强 (保留 AutoAugment)
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


# 4. 训练流程 (包含梯度累积)
def train_swin(model, train_loader, val_loader, num_epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # 初始化梯度

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            inputs_m, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            inputs_m, targets_a, targets_b = map(torch.autograd.Variable, (inputs_m, targets_a, targets_b))

            outputs = model(inputs_m)

            # 计算 Loss 并缩放
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss = loss / ACCUMULATION_STEPS

            loss.backward()

            # 当达到累积步数或数据耗尽时更新参数
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
            torch.save(model.state_dict(), 'swin_384_best.pth')
            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | 🔥 New Best!')
        else:
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}')

    return best_acc


# 5. 详细评估与混淆矩阵
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

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm, index=classes, columns=classes), annot=True, fmt='d', cmap='Blues')
    plt.title('Swin-384 Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./Swin_384_confusion_matrix.png')
    print("✅ Swin 混淆矩阵已保存为: ./Swin_384_confusion_matrix.png")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据集路径不存在: {DATA_PATH}")
        return

    train_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_a', 'sample_b', 'sample_c'], train_transform)
    val_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_d'], val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 🟢 核心修改：加载原生支持 384 的 Swin Base 版本
    print("加载模型 (Swin Transformer Base 384)...")
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=len(CLASSES))

    # 训练模型
    train_swin(model, train_loader, val_loader, NUM_EPOCHS)

    # 评估历史最佳模型
    print("\n训练结束，加载历史最佳模型进行最终评估...")
    model.load_state_dict(torch.load('swin_384_best.pth'))
    evaluate_model_detailed(model, val_loader, device, CLASSES)


if __name__ == "__main__":
    main()