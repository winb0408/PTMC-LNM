import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from MedMamba import VSSM as medmamba
from LNM_DataLoader import create_data_loaders


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 数据加载
    data_root = '/data2/wjb/code/PTMC_LNM/data/'
    image_dir = os.path.join(data_root, "imgs")
    label_dir = os.path.join(data_root, "LNM/CentralLNM")
    groups_dir = os.path.join(data_root, "LNM_groups")
    batch_size = 32

    train_loader, val_loader, test_loader = create_data_loaders(
        image_dir=image_dir,
        label_dir=label_dir,
        groups_dir=groups_dir,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform
    )

    # 初始化模型
    num_classes = 2  # 当前任务为二分类
    model_name = "Medmamba"
    net = medmamba(num_classes=num_classes)
    net.to(device)

    # 加载预训练权重
    pretrained_path = '/data2/wjb/code/PTMC_LNM/code/MedMamba-main/checkpoints/MedMamba.pth'
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = net.state_dict()
        # 仅加载预训练模型中与当前模型结构匹配的部分
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print("Pretrained weights loaded successfully!")
    else:
        print("Pretrained weights file not found!")

    # 替换分类层
    net.head = nn.Linear(net.num_features, num_classes)
    net.head.to(device)

    # 定义损失函数和优化器
    # criterion = nn.BCEWithLogitsLoss()  # 适用于二分类任务
    criterion = nn.CrossEntropyLoss()
    # 选择微调整个模型，但降低学习率
    optimizer = optim.Adam(net.parameters(), lr=0.00001)  # 降低学习率

    # 训练参数
    epochs = 500
    best_acc = 0.0
    result_root = '/data2/wjb/code/PTMC_LNM/trained_models/MedMamba/'  # 确保result_root已存在或创建它
    os.makedirs(result_root, exist_ok=True)
    save_path = os.path.join(result_root, './{}Net.pth'.format(model_name))

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")

        # 训练阶段
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, (images, labels) in enumerate(train_bar):
            images = images.to(device)
            labels = labels.to(device)
            # labels = labels.to(device).float().unsqueeze(1)  # 调整标签形状

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_description("Train Loss: {:.4f}".format(loss.item()))

        avg_train_loss = running_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # 验证阶段
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                # labels = labels.to(device).float().unsqueeze(1)  # 调整标签形状

                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算预测概率
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                # 计算指标
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                precision += (preds * labels).sum().item()
                recall += (preds == 1).sum().item()
                f1_score += (2 * precision * recall) / (precision + recall + 1e-8)  # 避免除零

                val_bar.set_description(
                    "Val Loss: {:.4f} | Acc: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
                        loss.item(), correct / total, precision / total, recall / total, f1_score / total))

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_precision = precision / total
        val_recall = recall / total
        val_f1 = f1_score / total

        print(
            f"Average Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f"Saved best model with accuracy: {best_acc:.4f}")

    print('Finished Training')

    # 测试阶段
    net.load_state_dict(torch.load(save_path))
    net.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for images, labels in test_bar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # 调整标签形状

            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算预测概率
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # 计算指标
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            precision += (preds * labels).sum().item()
            recall += (preds == 1).sum().item()
            f1_score += (2 * precision * recall) / (precision + recall + 1e-8)  # 避免除零

            test_bar.set_description(
                "Test Loss: {:.4f} | Acc: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
                    loss.item(), correct / total, precision / total, recall / total, f1_score / total))

    avg_test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    test_precision = precision / total
    test_recall = recall / total
    test_f1 = f1_score / total

    print(
        f"Average Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")


if __name__ == '__main__':
    main()