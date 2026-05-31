import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

# ================= 配置 =================
# 建议先把分辨率拉大，老师强，学生才强
IMG_SIZE = 384
BATCH_SIZE = 16  # 分辨率大了，Batch Size 调小点防止显存爆炸
LR = 1e-4
EPOCHS = 30
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"
SAVE_PATH = "./kth_resnet50_best.pth"
# =======================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_teacher():
    print(f"🚀 开始训练 ResNet50 老师模型 (ImgSize={IMG_SIZE})...")

    # 1. 数据增强 (加强版)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 简易加载数据 (假设文件夹结构是标准的)
    # 这里我们简单地把 sample_a/b/c 混在一起训练，d 做验证
    # 实际操作中，使用 ImageFolder 会自动读取所有子文件夹
    # 为了简单，我们直接用你之前的 Dataset 类逻辑，或者直接用 ImageFolder 只要路径对
    # 这里为了保证代码独立性，我简化逻辑：假设你有 KTHTIPS2bDataset 类，或者我们用通用逻辑
    # 鉴于你之前的代码，我们复用你的 Dataset 逻辑：
    from DeiT import KTHTIPS2bDataset, CLASSES  # 引用你主文件里的类

    train_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_a', 'sample_b', 'sample_c'], transform_train)
    val_dataset = KTHTIPS2bDataset(DATA_PATH, ['sample_d'], transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 加载 ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')  # 使用最新官方权重
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch + 1} | Loss: {running_loss / len(train_loader):.4f} | Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 保存最佳模型: {best_acc:.4f}")

    print(f"✅ 老师训练完成！最佳准确率: {best_acc:.4f}")
    print(f"权重已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    train_teacher()