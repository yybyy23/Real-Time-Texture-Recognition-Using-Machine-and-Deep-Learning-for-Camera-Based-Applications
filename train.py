import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ================= 绘图与评估库 =================
import matplotlib

matplotlib.use('Agg')  # 强制使用非交互后端，防止服务器报错
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================================

# ================= 配置区域 =================
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
DATA_PATH = "/root/TexRec/dtd_fold1"
# ===========================================

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载数据集
def load_datasets(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"路径不存在: {data_path}")

    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=val_transform)

    return train_dataset, val_dataset, test_dataset


# 2. 模型结构
def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=40):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=3e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None

    print(f"正在训练... (只更新 Layer3, Layer4 和 FC 层)")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
            f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs


# 可视化训练过程
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results_resnet18_optimized.png', dpi=300)
    print(f"训练历史图表已保存")
    plt.close()


# ================= 核心修改：详细评估函数 =================
def evaluate_model(model, test_loader, class_names):
    """
    在测试集上进行详细评估：Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("\n正在进行最终测试与评估...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 收集预测值和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. 计算整体准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n整体准确率 (Accuracy): {acc:.4f}")

    # 2. 打印详细分类报告 (包含 Precision, Recall, F1-Score)
    print("\n详细分类报告 (Classification Report):")
    # digits=4 保留4位小数
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 3. 绘制并保存混淆矩阵
    print("正在绘制混淆矩阵...")
    try:
        cm = confusion_matrix(all_labels, all_preds)
        # 设置画布大小，因为类别较多(47类)，需要大一点
        fig, ax = plt.subplots(figsize=(20, 20))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # xticks_rotation='vertical' 防止x轴标签重叠
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)

        plt.title(f"ResNet18 Confusion Matrix (Acc: {acc:.2%})", fontsize=20)
        plt.tight_layout()
        plt.savefig('confusion_matrix_resnet18.png', dpi=300)
        print("混淆矩阵已保存为: confusion_matrix_resnet18.png")
        plt.close()
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")

    return acc


# 保存模型
def save_model(model, class_names, filepath='texture_classifier_resnet18.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names)
    }
    torch.save(checkpoint, filepath)
    print(f"最佳模型已保存到: {filepath}")


# 主函数
def main():
    print("开始配置训练...")
    try:
        train_dataset, val_dataset, test_dataset = load_datasets(DATA_PATH)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"类别数量: {num_classes}")
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("正在创建 ResNet-18 模型...")
    model = create_model(num_classes=num_classes)

    print(f"开始训练 (Epochs: {NUM_EPOCHS})...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=NUM_EPOCHS
    )

    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # 使用新的评估函数替代旧的 test_model
    evaluate_model(model, test_loader, class_names)

    save_model(model, class_names)
    print("所有任务完成!")


if __name__ == "__main__":
    main()