import os
import argparse
import pickle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="数据集根目录，内部应包含 train / val / test 三个子文件夹"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="特征提取时的 batch size"
    )
    parser.add_argument(
        "--save-bundle",
        type=str,
        default="cnn_svm_bundle.pkl",
        help="保存 scaler + svm + class_names 的文件名"
    )
    return parser.parse_args()


def get_data_loaders(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')

    if not os.path.exists(train_dir):
        print(f"错误：在 {data_path} 下没找到 train 文件夹。")
        return None, None, None, None

    if not os.path.exists(val_dir):
        print(f"错误：在 {data_path} 下没找到 val 文件夹。")
        return None, None, None, None

    if not os.path.exists(test_dir):
        print(f"错误：在 {data_path} 下没找到 test 文件夹。")
        return None, None, None, None

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names


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


def plot_confusion_matrix_custom(y_true, y_pred, class_names, save_path):
    print(f"正在生成混淆矩阵图片: {save_path} ...")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(24, 24))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d', colorbar=False)

    plt.title("SVM Confusion Matrix (ResNet50 Features)", fontsize=20)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"成功保存: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")
    finally:
        plt.close(fig)


def main():
    args = parse_args()
    print(f"使用设备: {DEVICE}")
    print(f"数据路径: {args.data_path}")

    print("加载 ResNet50 (特征提取器)...")
    try:
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
    except AttributeError:
        # 兼容老版本 torchvision
        resnet = models.resnet50(pretrained=True)

    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules).to(DEVICE)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(args.data_path, args.batch_size)
    if train_loader is None:
        return

    print("\n--- 提取特征 ---")
    X_train, y_train = extract_features(model, train_loader)
    X_val, y_val = extract_features(model, val_loader)
    X_test, y_test = extract_features(model, test_loader)

    print("\n正在标准化特征...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n开始训练 SVM (LinearSVC)...")
    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)

    clf = LinearSVC(C=0.01, max_iter=10000, dual=True)
    clf.fit(X_combined, y_combined)

    print("\n正在进行预测...")
    y_pred_test = clf.predict(X_test)

    print("\n" + "=" * 20 + " 最终测试集评估 " + "=" * 20)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_test, target_names=class_names, digits=4))

    plot_confusion_matrix_custom(y_test, y_pred_test, class_names, 'svm_confusion_matrix.png')

    bundle = {
        "scaler": scaler,
        "svm": clf,
        "class_names": class_names
    }
    with open(args.save_bundle, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n已保存: {args.save_bundle}")


if __name__ == "__main__":
    main()