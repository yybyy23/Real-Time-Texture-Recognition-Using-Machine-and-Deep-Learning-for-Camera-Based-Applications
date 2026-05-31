import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import timm  # 必须安装: pip install timm

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')  # 强制使用非交互模式
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================================

# ================= 配置区域 =================
# DTD 数据集路径 (请确认路径是否正确)
DATA_PATH = "/root/TexRec/dtd_fold1"

BATCH_SIZE = 32  # 如果显存爆了，改小到 16
NUM_EPOCHS = 50
# ViT 学习率建议比 CNN 小，5e-5 是经验值
LEARNING_RATE = 5e-5
MIXUP_ALPHA = 0.2
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据预处理
# ViT 需要较强的数据增强来防止过拟合
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # 随机裁剪缩放
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_datasets(data_path):
    if not os.path.exists(data_path):
        print(f"错误: 路径 {data_path} 不存在")
        return None, None, None, None

    # 使用 ImageFolder 加载标准结构的 DTD
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=val_transform)

    class_names = train_dataset.classes
    return train_dataset, val_dataset, test_dataset, class_names


# 2. Mixup 辅助函数
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


# 3. 创建 ViT 模型
def create_model(num_classes):
    print(f"正在加载 ViT-Lite (DeiT-Small) | 类别数: {num_classes}...")

    # 使用 timm 加载预训练的 DeiT-Small
    model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)

    return model


# 4. 训练函数
def train_model(model, train_loader, val_loader, num_epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # [关键] Transformer 必须用 AdamW，且给予一定的 weight_decay 防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses, val_accs = [], []
    best_val_acc = 0.0

    print("开始训练 (ViT + AdamW + Mixup)...")

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 简单估算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (lam * predicted.eq(targets_a.data).cpu().sum().float() +
                              (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)

        # --- 验证阶段 ---
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_accs.append(val_acc.item())

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'dtd_vit_best.pth')

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    return train_losses, val_accs


# 5. 绘图函数
def plot_history(train_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_title('Loss')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy')
    plt.savefig('dtd_vit_history.png')
    plt.close()


# 6. 详细评估函数
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    print("\n正在进行最终测试与评估...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            # [修复] 加上逗号，进行解包赋值
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 整体准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n整体准确率 (Accuracy): {acc:.4f}")

    # 详细报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 绘制混淆矩阵
    try:
        print("正在绘制混淆矩阵 (这可能需要一点时间)...")
        cm = confusion_matrix(all_labels, all_preds)

        # DTD 有 47 类，必须使用超大画布
        fig, ax = plt.subplots(figsize=(24, 24))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)

        plt.title(f"ViT-Lite (DTD) Confusion Matrix (Acc: {acc:.2%})", fontsize=20)
        plt.tight_layout()
        plt.savefig('dtd_vit_confusion.png', dpi=300)
        print("混淆矩阵已保存至: dtd_vit_confusion.png")
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    # 1. 加载数据
    train_dataset, val_dataset, test_dataset, class_names = load_datasets(DATA_PATH)
    if train_dataset is None: return

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")
    print(f"类别数: {len(class_names)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 创建模型
    model = create_model(len(class_names))

    # 3. 训练
    train_losses, val_accs = train_model(model, train_loader, val_loader, NUM_EPOCHS)
    plot_history(train_losses, val_accs)

    # 4. 评估
    # 加载最佳模型权重
    model.load_state_dict(torch.load('dtd_vit_best.pth'))
    evaluate_model(model, test_loader, class_names)


if __name__ == "__main__":
    main()
