import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from PIL import Image

# ================= 绘图与评估库 (新增) =================
import matplotlib

matplotlib.use('Agg')  # 强制使用非交互后端
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ====================================================

# ================= 配置区域 =================
# 路径请确保正确，下面要有 aluminium_foil 等 11 个文件夹
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"

# KTH 只有 11 类，ResNet18 足够了，跑起来也快
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MIXUP_ALPHA = 0.2  # Mixup 强度

CLASSES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cork', 'cotton',
    'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool'
]
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 自定义数据集加载器 (处理 sample_a/b/c 结构)
class KTHTIPS2bDataset(Dataset):
    def __init__(self, root_dir, samples_to_use, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}

        for cls_name in CLASSES:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            cls_idx = self.class_to_idx[cls_name]

            # 遍历指定的样本文件夹 (如 sample_a, sample_b)
            for sample_name in samples_to_use:
                sample_folder = os.path.join(cls_folder, sample_name)
                if not os.path.isdir(sample_folder):
                    continue
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
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. Mixup 函数
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


# 3. 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # KTH 图片通常是 200x200，Resize 到 224 适配 ResNet
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # 纹理无方向，垂直翻转有效
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # KTH 对光照敏感，增强光照鲁棒性
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 4. 创建模型
def create_model(num_classes):
    print("加载 ResNet18 (预训练)...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# 5. 训练主流程
def train_model(model, train_loader, val_loader, num_epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    print("开始训练 (ResNet18 + Mixup + KTH Leave-One-Out)...")

    for epoch in range(num_epochs):
        # --- 训练 ---
        model.train()
        running_loss = 0.0
        # 简单统计 Train Acc (仅供参考，Mixup 下 Train Acc 不直观)
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

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            # 简单估算准确率
            correct_train += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                              + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # --- 验证 (Sample D) ---
        model.eval()
        val_loss_run = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_run += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss_run / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'kth_resnet18_best.pth')

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Acc (Sample D): {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    return train_losses, val_losses, train_accs, val_accs


# ================= 6. 详细评估函数 (新增) =================
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    print("\n正在进行最终测试与评估 (Sample D)...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. 准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n整体准确率 (Accuracy): {acc:.4f}")

    # 2. 详细分类报告 (Precision, Recall, F1)
    print("\n详细分类报告 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # 3. 绘制混淆矩阵
    print("正在绘制混淆矩阵...")
    try:
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(12, 12))  # KTH 只有11类，12x12足够清晰
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)

        plt.title(f"KTH-TIPS2b (Sample D) Confusion Matrix (Acc: {acc:.2%})", fontsize=15)
        plt.tight_layout()
        plt.savefig('kth_confusion_matrix.png', dpi=300)
        print("混淆矩阵已保存为: kth_confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"错误: 路径 {DATA_PATH} 不存在")
        return

    # 训练集: Sample A, B, C
    print("加载训练集 (Sample A, B, C)...")
    train_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_a', 'sample_b', 'sample_c'], train_transform)

    # 验证/测试集: Sample D
    print("加载验证集 (Sample D)...")
    val_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_d'], val_transform)

    print(f"训练集数量: {len(train_dataset)}")
    print(f"验证集数量: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # val_loader 即作为测试集
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = create_model(len(CLASSES))

    t_loss, v_loss, t_acc, v_acc = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_loss, label='Train Loss (Mixup)')
    plt.plot(v_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(t_acc, label='Train Acc')
    plt.plot(v_acc, label='Val Acc (Sample D)')
    plt.legend()
    plt.title('Accuracy')

    plt.savefig('kth_result.png', dpi=300)
    print("训练曲线图已保存至 kth_result.png")
    plt.close()  # 释放内存

    # 加载最佳模型进行评估
    print("\n加载最佳模型进行详细评估...")
    model.load_state_dict(torch.load('kth_resnet18_best.pth'))
    evaluate_model(model, val_loader)

    print(f"\n训练结束! 结果已保存。")


if __name__ == "__main__":
    main()