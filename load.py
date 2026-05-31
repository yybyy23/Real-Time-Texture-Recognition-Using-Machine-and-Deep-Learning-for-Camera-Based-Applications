import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集 - 使用ImageFolder自动从目录结构获取标签
def load_datasets(data_path):
    """
    加载训练集、验证集和测试集
    """
    train_dataset = datasets.ImageFolder(
        root=f"{data_path}/train",
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f"{data_path}/val", 
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=f"{data_path}/test",
        transform=val_transform
    )
    
    return train_dataset, val_dataset, test_dataset

# 使用示例
data_path = "C:\\Users\\Admin\\Desktop\\YBY\\Pytorch\\model\\dtd_organized"  # 修改为您的实际路径
train_dataset, val_dataset, test_dataset = load_datasets(data_path)

# 获取类别信息
class_names = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

print(f"类别数量: {len(class_names)}")
print(f"类别名称: {class_names}")
print(f"类别到索引的映射: {class_to_idx}")
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

