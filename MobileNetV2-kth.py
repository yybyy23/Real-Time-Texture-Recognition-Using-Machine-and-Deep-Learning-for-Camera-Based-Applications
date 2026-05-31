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

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')  # 强制使用非交互后端
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================================

# ================= 配置区域 =================
# KTH 数据集路径
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"

BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.0005
MIXUP_ALPHA = 0.4  # Mixup 强度

# KTH-TIPS2-b 的 11 个类别
CLASSES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cork', 'cotton',
    'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool'
]
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 自定义数据集加载器 (适配 KTH 的 sample_a/b/c/d 结构)
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


# 3. 数据预处理
# KTH 的图片也是纹理，所以保留 RandomResizedCrop 和 RandomErasing 是非常有益的
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 随机擦除
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 4. 创建模型：MobileNetV2
def create_model(num_classes):
    print("正在加载 MobileNetV2 (预训练)...")
    model = models.mobilenet_v2(pretrained=True)

    # 冻结策略：冻结前 10 层 features
    for i, child in enumerate(model.features.children()):
        if i < 10:
            for param in child.parameters():
                param.requires_grad = False

    # 修改分类器
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# 5. 训练函数
def train_model(model, train_loader, val_loader, num_epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 只优化需要更新的参数
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LEARNING_RATE, weight_decay=5e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None

    print(f"开始训练 (MobileNetV2 + Mixup + KTH Leave-One-Out)...")

    for epoch in range(num_epochs):
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

            # 估算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                              + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证 (Sample D)
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # 验证集不用 Mixup

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())

        scheduler.step()

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_state = model.state_dict().copy()

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Acc (Sample D): {val_epoch_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Train Loss (Mixup)', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('kth_mobilenet_history.png', dpi=300)
    print("训练图表已保存至: kth_mobilenet_history.png")
    plt.close()


# ================= 6. 详细评估函数 =================
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
        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)

        plt.title(f"MobileNetV2 (KTH) Confusion Matrix (Acc: {acc:.2%})", fontsize=15)
        plt.tight_layout()
        plt.savefig('kth_mobilenet_confusion.png', dpi=300)
        print("混淆矩阵已保存为: kth_mobilenet_confusion.png")
        plt.close()
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")


def save_model(model):
    torch.save(model.state_dict(), 'kth_mobilenet_best.pth')
    print("最佳模型已保存: kth_mobilenet_best.pth")


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
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 创建 MobileNetV2
    model = create_model(len(CLASSES))

    # 训练
    model, t_loss, v_loss, t_acc, v_acc = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # 绘图
    plot_training_history(t_loss, v_loss, t_acc, v_acc)

    # 详细评估 (F1, 混淆矩阵)
    evaluate_model(model, val_loader)

    # 保存
    save_model(model)


if __name__ == "__main__":
    main()