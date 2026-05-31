import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from PIL import Image
import timm  # [核心依赖] 必须安装: pip install timm

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================================

# ================= 配置区域 =================
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"
BATCH_SIZE = 32  # 显存如果不够，改小到 16
NUM_EPOCHS = 50
# ViT 对学习率很敏感，通常比 CNN 要小。建议 1e-4 或 5e-5
LEARNING_RATE = 5e-5
MIXUP_ALPHA = 0.2

CLASSES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cork', 'cotton',
    'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool'
]
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 数据集加载器 (保持不变)
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


# 2. Mixup (保持不变)
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
# ViT 通常需要更强的增强，但我们先保持和 ResNet 一致以便对比
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # ImageNet 标准归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 4. [核心修改] 创建 ViT 模型
def create_model(num_classes):
    print("正在加载 ViT-Lite (DeiT-Small)...")

    # 使用 timm 创建模型
    # 选项 A: 'deit_small_patch16_224' (推荐，轻量且高效)
    # 选项 B: 'vit_tiny_patch16_224' (非常小，速度快)
    # 选项 C: 'swin_tiny_patch4_window7_224' (Swin Transformer)

    model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)

    # 打印参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_parameters / 1e6:.2f}M")

    return model


# 5. 训练主流程
def train_model(model, train_loader, val_loader, num_epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # [核心修改] Transformer 必须使用 AdamW 优化器，而不是 SGD
    # Weight decay (权重衰减) 对 ViT 很重要，通常设置 0.01 或 0.05
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    print("开始训练 (ViT + AdamW + Mixup)...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
            correct_train += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(
                targets_b.data).cpu().sum().float())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证
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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'kth_vit_best.pth')

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    return train_losses, val_accs


# 6. 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    print("\n正在评估...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"整体准确率: {acc:.4f}")
    print("\n详细报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # 绘制混淆矩阵
    try:
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)
        plt.title(f"ViT (DeiT-Small) Confusion Matrix (Acc: {acc:.2%})")
        plt.savefig('kth_vit_confusion.png', dpi=300, bbox_inches='tight')
        print("混淆矩阵已保存至 kth_vit_confusion.png")
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    if not os.path.exists(DATA_PATH):
        print("路径不存在")
        return
    train_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_a', 'sample_b', 'sample_c'], train_transform)
    val_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_d'], val_transform)  # Sample D 测试

    print(f"训练集: {len(train_dataset)}, 测试集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = create_model(len(CLASSES))
    train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # 加载最佳模型评估
    model.load_state_dict(torch.load('kth_vit_best.pth'))
    evaluate_model(model, val_loader)


if __name__ == "__main__":
    main()