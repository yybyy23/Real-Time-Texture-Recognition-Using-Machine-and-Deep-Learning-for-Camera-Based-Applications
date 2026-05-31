# ================= 强制配置绘图后端 (必须在最前面) =================
import matplotlib

matplotlib.use('Agg')  # 强制使用非交互模式，解决服务器无法保存图片的问题
import matplotlib.pyplot as plt
# ==============================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import traceback  # 用于打印详细错误

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay

# ================= 配置区域 =================
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cork', 'cotton',
    'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool'
]


# ===========================================

# 1. 自定义数据集加载器
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

            # 遍历指定的样本文件夹
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


def get_data_loaders(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_path):
        print(f"错误: 路径 {data_path} 不存在")
        return None, None

    # Sample A+B+C 用于训练，Sample D 用于测试
    print("正在加载数据集...")
    train_dataset = KTHTIPS2bDataset(data_path, ['sample_a', 'sample_b', 'sample_c'], transform)
    test_dataset = KTHTIPS2bDataset(data_path, ['sample_d'], transform)

    print(f"训练集 (Sample A+B+C): {len(train_dataset)} 张")
    print(f"测试集 (Sample D):     {len(test_dataset)} 张")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader


def extract_features(model, loader):
    model.eval()
    features_list = []
    labels_list = []

    print("正在提取特征...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            output = model(inputs)
            output = torch.flatten(output, 1)
            features_list.append(output.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


def main():
    print(f"使用设备: {DEVICE}")

    # 1. 准备模型
    print("加载 ResNet50 (作为特征提取器)...")
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules)
    model = model.to(DEVICE)

    # 2. 准备数据
    train_loader, test_loader = get_data_loaders(DATA_PATH)
    if train_loader is None: return

    # 3. 提取特征
    print("\n--- 提取训练集特征 ---")
    X_train, y_train = extract_features(model, train_loader)

    print("\n--- 提取测试集特征 ---")
    X_test, y_test = extract_features(model, test_loader)

    # 4. 标准化
    print("\n正在标准化特征...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. 训练 SVM
    print("\n开始训练 SVM (LinearSVC)...")
    clf = LinearSVC(C=0.01, max_iter=10000, dual=True)
    clf.fit(X_train, y_train)

    # 6. 预测
    print("\n正在进行预测 (Sample D)...")
    y_pred = clf.predict(X_test)

    # 7. 评估区域
    print("\n" + "=" * 30)
    print("       最终评估报告")
    print("=" * 30)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"总体准确率 (Accuracy):   {acc:.4f} ({acc * 100:.2f}%)")
    print(f"宏平均精确率 (Precision): {precision:.4f}")
    print(f"宏平均召回率 (Recall):    {recall:.4f}")
    print(f"宏平均 F1-Score:         {f1:.4f}")
    print("-" * 30)

    print("\n--- 详细分类报告 ---")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))

    # 8. 绘制混淆矩阵 (增强版)
    print("正在绘制混淆矩阵...")
    try:
        cm = confusion_matrix(y_test, y_pred)

        # 显式创建 Figure 和 Axes
        fig, ax = plt.subplots(figsize=(12, 12))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)

        plt.title(f"SVM (ResNet50 Features) on KTH\nAcc: {acc:.2%}", fontsize=14)
        plt.tight_layout()

        # 定义文件名并保存
        save_filename = 'kth_svm_confusion_metrics.png'
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')

        # 打印绝对路径，确保你能找到它
        abs_path = os.path.abspath(save_filename)
        print(f"\n[成功] 混淆矩阵已保存至: {abs_path}")

        plt.close(fig)  # 释放内存

    except Exception as e:
        print("\n[错误] 混淆矩阵绘制失败！")
        print(traceback.format_exc())  # 打印详细的错误栈


if __name__ == "__main__":
    main()