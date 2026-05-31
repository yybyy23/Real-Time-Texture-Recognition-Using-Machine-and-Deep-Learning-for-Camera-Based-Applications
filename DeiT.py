import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from PIL import Image
import timm
import torch.nn.functional as F
import pandas as pd

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')  # 服务器端绘图必须用 Agg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================================

# ================= 配置区域 (关键修改) =================
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"

# 🟢 [修改 1] 分辨率提升至 384 (大幅提升纹理细节)
IMG_SIZE = 384

# 🟢 [修改 2] BatchSize 降低以适应高分辨率 (防止显存溢出)
BATCH_SIZE = 32

NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
MIXUP_ALPHA = 0.2

# 🟢 [修改 3] 蒸馏策略优化
DISTILLATION_TYPE = 'soft'
DISTILLATION_ALPHA = 0.5
DISTILLATION_TAU = 4.0  # 稍微调高温度，让分布更平滑
TEACHER_ARCH = 'resnet50'  # 🟢 [修改 4] 使用更强的 ResNet50 老师
TEACHER_PATH = './kth_resnet50_best.pth'  # 请确保此文件存在

CLASSES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cork', 'cotton',
    'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool'
]
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device} | 图像尺寸: {IMG_SIZE}x{IMG_SIZE}")


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


# 2. Mixup
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


# 3. 增强 (已应用 384 分辨率)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),  # 🟢 使用 384
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 🟢 使用 384
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 4. [核心] 获取老师模型 (更新为 ResNet50)
def get_teacher_model(num_classes, device, model_path):
    print(f"正在加载 Teacher 模型 ({TEACHER_ARCH}) 从: {model_path} ...")

    # 🟢 [修改] 初始化 ResNet50
    teacher = models.resnet50(weights=None)
    in_features = teacher.fc.in_features
    teacher.fc = nn.Linear(in_features, num_classes)

    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到文件 {model_path}")
        print("💡 提示: 请先运行 train_teacher.py 训练一个 ResNet50 老师模型")
        return None

    try:
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[7:]

            # 自动修复 fc.1 问题
            if "fc.1." in name:
                print(f"检测到结构差异，自动修正: {name} -> {name.replace('fc.1.', 'fc.')}")
                name = name.replace("fc.1.", "fc.")

            new_state_dict[name] = v

        msg = teacher.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Teacher 模型加载成功！")

    except Exception as e:
        print(f"❌ 严重错误: 加载 Teacher 权重失败: {e}")
        return None

    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


# 5. [核心] 蒸馏 Loss 函数
class DistillationLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, dist_type, alpha, tau):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.dist_type = dist_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.dist_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs / T, dim=1),
                F.softmax(teacher_outputs / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

        elif self.dist_type == 'hard':
            teacher_labels = teacher_outputs.argmax(dim=1)
            distillation_loss = F.cross_entropy(outputs, teacher_labels)

        else:
            distillation_loss = 0.0

        return distillation_loss


# 6. 训练流程
def train_model(student, teacher, train_loader, val_loader, num_epochs):
    student = student.to(device)

    base_criterion = nn.CrossEntropyLoss()
    distillation_loss_fn = DistillationLoss(
        base_criterion,
        teacher,
        DISTILLATION_TYPE,
        DISTILLATION_ALPHA,
        tau=DISTILLATION_TAU
    )

    optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0

    print(f"开始蒸馏训练 (Type: {DISTILLATION_TYPE} | T={DISTILLATION_TAU} | Alpha={DISTILLATION_ALPHA})...")

    for epoch in range(num_epochs):
        student.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs_m, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            inputs_m, targets_a, targets_b = map(torch.autograd.Variable, (inputs_m, targets_a, targets_b))

            optimizer.zero_grad()

            outputs = student(inputs_m)
            loss_base = mixup_criterion(base_criterion, outputs, targets_a, targets_b, lam)
            loss_distill = distillation_loss_fn(inputs_m, outputs, labels)

            loss = (1 - DISTILLATION_ALPHA) * loss_base + DISTILLATION_ALPHA * loss_distill

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- 验证 ---
        student.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), 'vit_distilled_best.pth')
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | 🔥 New Best!')
        else:
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    return best_acc


# 7. 详细评估函数
def evaluate_model_detailed(model, dataloader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    print("\n正在进行详细评估...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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
    print("=" * 60)
    return acc


# 8. 混淆矩阵绘制
def plot_confusion_matrix(model, dataloader, device, classes):
    print("正在计算并绘制混淆矩阵...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = './DeiT_confusion_matrix.png'
    plt.savefig(save_path)
    print(f"✅ 混淆矩阵已保存为: {save_path}")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据集路径不存在: {DATA_PATH}")
        return

    # KTH 设置
    train_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_a', 'sample_b', 'sample_c'], train_transform)
    val_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_d'], val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 🟢 [修改] 切换到支持 384 输入的学生模型
    print(f"加载学生 (ViT) [ImgSize: {IMG_SIZE}]...")
    try:
        # 尝试加载 384 专用模型 (如果在 timm 库中存在)
        student = timm.create_model('deit_small_patch16_384', pretrained=True, num_classes=len(CLASSES))
    except Exception:
        print("未找到 deit_small_patch16_384，回退到 224 模型并调整分辨率...")
        student = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=len(CLASSES),
                                    img_size=IMG_SIZE)

    # 加载老师
    teacher = get_teacher_model(len(CLASSES), device, model_path=TEACHER_PATH)

    if teacher is None:
        print("🔴 错误：Teacher 模型未成功加载，程序退出。请检查路径或权重文件。")
        return

    # 训练
    train_model(student, teacher, train_loader, val_loader, NUM_EPOCHS)

    # 训练结束后，加载【历史最佳】模型进行最终评估
    print("\n训练结束，加载历史最佳模型进行最终评估...")
    student.load_state_dict(torch.load('vit_distilled_best.pth'))

    # 运行详细评估
    evaluate_model_detailed(student, val_loader, device, CLASSES)
    # 绘制混淆矩阵
    plot_confusion_matrix(student, val_loader, device, CLASSES)


if __name__ == "__main__":
    main()