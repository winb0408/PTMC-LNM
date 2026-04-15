import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, group_file, transform=None):
        """
        初始化数据集
        :param image_dir: 图像存放路径
        :param label_dir: 标签存放路径
        :param group_file: 包含图像名称的文件路径（训练集、验证集或测试集）
        :param transform: 数据预处理变换
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # 读取图像名称列表
        with open(group_file, 'r') as f:
            self.image_names = [line.strip() for line in f]

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本索引
        :return: 处理后的图像和对应的标签
        """
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name + '.jpg')  # 假设图像文件为.jpg格式
        label_path = os.path.join(self.label_dir, image_name + '.txt')  # 假设标签文件为.txt格式

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 加载标签
        with open(label_path, 'r') as f:
            label = int(f.read().strip())  # 假设标签为0或1

        # 应用数据预处理
        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders(image_dir, label_dir, groups_dir, batch_size, train_transform, val_transform):
    """
    创建数据加载器
    :param image_dir: 图像存放路径
    :param label_dir: 标签存放路径
    :param groups_dir: 包含图像名称的文件路径（train, val, test）
    :param batch_size: 批量大小
    :param train_transform: 训练集数据预处理变换
    :param val_transform: 验证集和测试集数据预处理变换
    :return: 训练集、验证集和测试集的数据加载器
    """
    # 读取训练集、验证集和测试集的图像名称
    train_dataset = CustomDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        group_file=os.path.join(groups_dir, 'train.txt'),
        transform=train_transform
    )

    val_dataset = CustomDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        group_file=os.path.join(groups_dir, 'val.txt'),
        transform=val_transform
    )

    test_dataset = CustomDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        group_file=os.path.join(groups_dir, 'test.txt'),
        transform=val_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader


# 示例使用
if __name__ == '__main__':
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 使用ImageNet的归一化参数
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 数据路径
    image_dir = 'data/imgs'
    label_dir = 'data/LNM/CentralLNM'
    groups_dir = 'data/groups'
    batch_size = 32

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        image_dir=image_dir,
        label_dir=label_dir,
        groups_dir=groups_dir,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform
    )

    # 测试数据加载器
    for images, labels in train_loader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Label: {labels[0]}")
        break