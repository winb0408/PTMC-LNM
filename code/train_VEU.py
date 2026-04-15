import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as transforms

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from datasets.LNM_Central import LNM_Central
# from archs.modified_se_resnet import ModifiedSEResNet50
from archs.modified_se_resnet import ModifiedSEResNet50

# 设定环境
use_cuda = True
cuda_id = "1"
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
torch.cuda.set_device(1)

# 路径和超参数
data_root = "/data2/wjb/code/PTMC_LNM/data"
save_root = "/data2/wjb/code/PTMC_LNM/trained_models/Modified_SeResnet_VEU/"
epochs = 600
batch_size = 64
lr = 2e-5
# clinical_features_path = '/data2/wjb/code/pumc-thyroid/data//clinical_features.npy'  # 临床特征数据的路径
# clinical_flag_path = '/data2/wjb/code/pumc-thyroid/data//clinical_flag.txt'  # 临床特征顺序文件
clinical_features_path = "/data2/wjb/code/pumc-thyroid/data/lnm_clinical_features.npy"
clinical_flag_path = "/data2/wjb/code/pumc-thyroid/data/lnm_clinical_flag.txt"

if not os.path.exists(save_root):
    os.mkdir(save_root)


# 定义损失权重优化网络
class LossWeightNet(nn.Module):
    def __init__(self):
        super(LossWeightNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, f1, f2):
        f1_mean = f1.mean(dim=1, keepdim=True)
        f2_mean = f2.mean(dim=1, keepdim=True)
        f1_max = f1.max(dim=1)[0].unsqueeze(1)
        f2_max = f2.max(dim=1)[0].unsqueeze(1)
        features = torch.cat([f1_mean, f2_mean, f1_max, f2_max], dim=1)
        return self.fc(features)


# 定义模型
# class DualInputModel(nn.Module):
#     def __init__(self, num_classes=2, clinical_feature_dim=32):  # 注意 clinical_feature_dim 可变，测试时需调整
#         super(DualInputModel, self).__init__()
#         self.feature_extractor1 = ModifiedSEResNet50(num_classes=1000)
#         self.feature_extractor2 = ModifiedSEResNet50(num_classes=1000)
#
#         # 将图像特征和临床特征结合起来
#         combined_feature1 = 1000 + clinical_feature_dim  # 1000是仅超声图像特征，clinical_feature_dim是临床特征
#         self.classifier1 = nn.Sequential(
#             nn.Linear(combined_feature1, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
#         combined_feature2 = 2000 + clinical_feature_dim  # 2000是融合图像特征，clinical_feature_dim是临床特征
#         self.classifier2 = nn.Sequential(
#             nn.Linear(combined_feature2, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
#         self.loss_weights_net = LossWeightNet()
#
#     def forward(self, x1, x2, clinical_features):
#         f1 = self.feature_extractor1(x1)
#         f2 = self.feature_extractor2(x2)
#         combined_features = torch.cat((f1, f2), dim=1)
#
#         # 结合临床特征
#         combined_feature1_with_clinical = torch.cat((f1, clinical_features), dim=1)
#         combined_features_with_clinical = torch.cat((combined_features, clinical_features), dim=1)
#
#         # 进行分类
#         pred1 = self.classifier1(combined_feature1_with_clinical)
#         pred2 = self.classifier2(combined_features_with_clinical)
#         loss_weights = self.loss_weights_net(f1, f2)
#
#         return pred1, pred2, loss_weights, combined_feature1_with_clinical, combined_features_with_clinical

class ClinicalAttentionModule(nn.Module):
    def __init__(self, img_feature_dim, clinical_feature_dim):
        super(ClinicalAttentionModule, self).__init__()
        self.fc_img_to_clinical = nn.Linear(img_feature_dim, clinical_feature_dim)
        self.fc_clinical_to_img = nn.Linear(clinical_feature_dim, img_feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, clinical_features):
        # 将图像特征映射到临床特征空间
        img_features_transformed = self.fc_img_to_clinical(img_features)
        # 将临床特征映射到图像特征空间
        clinical_features_transformed = self.fc_clinical_to_img(clinical_features)

        # 计算注意力权重
        # 方法①: 映射后的图像维度与临床特征计算
        attention_weights_clinical = self.sigmoid(self.relu(img_features_transformed + clinical_features))
        # 方法②: 映射后的临床特征与图像特征计算
        attention_weights_img = self.sigmoid(self.relu(clinical_features_transformed + img_features))

        # 应用注意力权重
        enhanced_img_features = img_features * attention_weights_img  # 扩展维度以匹配图像特征的形状
        enhanced_clinical_features = clinical_features * attention_weights_clinical  # 扩展维度以匹配临床特征的形状

        return enhanced_img_features, enhanced_clinical_features

class DualInputModel(nn.Module):
    def __init__(self, num_classes=2, clinical_feature_dim=10):  # 注意 clinical_feature_dim 可变，测试时需调整
        super(DualInputModel, self).__init__()
        self.base_model = ModifiedSEResNet50(num_classes=2)
        self.fc_clinical = nn.Linear(clinical_feature_dim, 256)
        self.clinical_attention_module = ClinicalAttentionModule(1024, 256) #img_feature_dim, clinical_feature_dim

        # 将图像特征和临床特征结合起来
        combined_feature1 = 256 * 4 + 256  # 1024是仅超声图像特征，256是临床特征
        self.classifier1 = nn.Sequential(
            nn.Linear(combined_feature1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        combined_feature2 = 256 * 4 * 2 + 256  # 2048是融合图像特征，clinical_feature_dim是临床特征
        self.classifier2 = nn.Sequential(
            nn.Linear(combined_feature2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.loss_weights_net = LossWeightNet()

    def forward(self, x1, x2, clinical_features):
        # b-mode 图像特征提取
        c1 = self.base_model.base_model.layer1(
            self.base_model.base_model.maxpool(self.base_model.base_model.relu(
                self.base_model.base_model.bn1(self.base_model.base_model.conv1(x1)))
            )
        )
        c2 = self.base_model.base_model.layer2(c1)
        c3 = self.base_model.base_model.layer3(c2)
        c4 = self.base_model.base_model.layer4(c3)

        # FPN特征融合
        p1, p2, p3, p4 = self.base_model.fpn(c1, c2, c3, c4)
        p1 = self.base_model.cbam(p1)
        p2 = self.base_model.cbam(p2)
        p3 = self.base_model.cbam(p3)
        p4 = self.base_model.cbam(p4)

        # 池化并拼接特征
        p1 = F.adaptive_avg_pool2d(p1, 1).view(p1.size(0), -1)
        p2 = F.adaptive_avg_pool2d(p2, 1).view(p2.size(0), -1)
        p3 = F.adaptive_avg_pool2d(p3, 1).view(p3.size(0), -1)
        p4 = F.adaptive_avg_pool2d(p4, 1).view(p4.size(0), -1)

        bmode_features = torch.cat((p1, p2, p3, p4), dim=1)

        # VEU 图像特征提取
        c1_x2 = self.base_model.base_model.layer1(
            self.base_model.base_model.maxpool(self.base_model.base_model.relu(
                self.base_model.base_model.bn1(self.base_model.base_model.conv1(x2)))
            )
        )
        c2_x2 = self.base_model.base_model.layer2(c1_x2)
        c3_x2 = self.base_model.base_model.layer3(c2_x2)
        c4_x2 = self.base_model.base_model.layer4(c3_x2)

        # FPN特征融合
        p1_x2, p2_x2, p3_x2, p4_x2 = self.base_model.fpn(c1_x2, c2_x2, c3_x2, c4_x2)
        p1_x2 = self.base_model.cbam(p1_x2)
        p2_x2 = self.base_model.cbam(p2_x2)
        p3_x2 = self.base_model.cbam(p3_x2)
        p4_x2 = self.base_model.cbam(p4_x2)

        # 池化并拼接特征
        p1_x2 = F.adaptive_avg_pool2d(p1_x2, 1).view(p1.size(0), -1)
        p2_x2 = F.adaptive_avg_pool2d(p2_x2, 1).view(p2.size(0), -1)
        p3_x2 = F.adaptive_avg_pool2d(p3_x2, 1).view(p3.size(0), -1)
        p4_x2 = F.adaptive_avg_pool2d(p4_x2, 1).view(p4.size(0), -1)

        VEU_features = torch.cat((p1_x2, p2_x2, p3_x2, p4_x2), dim=1)

        # 临床特征处理
        clinical_features = self.fc_clinical(clinical_features)
        bmode_features_enhanced, clinical_features_enhanced = self.clinical_attention_module(bmode_features, clinical_features)
        VEU_features_enhanced, _ = self.clinical_attention_module(VEU_features, clinical_features)

        combined_features = torch.cat((bmode_features_enhanced, VEU_features_enhanced), dim=1)

        # 结合临床特征
        combined_feature1_with_clinical = torch.cat((bmode_features_enhanced, clinical_features_enhanced), dim=1)
        combined_features_with_clinical = torch.cat((combined_features, clinical_features_enhanced), dim=1)

        # 进行分类
        pred1 = self.classifier1(combined_feature1_with_clinical)
        pred2 = self.classifier2(combined_features_with_clinical)
        loss_weights = self.loss_weights_net(bmode_features, combined_features)

        return pred1, pred2, loss_weights

# 数据变换
trans = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5,), (1.0,))
])

# 数据加载
dataset = LNM_Central(
    data_dir=data_root,
    split="train",
    clinical_features_path=clinical_features_path,
    clinical_flag_path=clinical_flag_path
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 模型初始化
clinical_feature_dim = np.load(clinical_features_path).shape[1]  # 获取临床特征的维度
model = DualInputModel(num_classes=2, clinical_feature_dim=clinical_feature_dim)
model.to(device)

# 损失函数和优化器
# # 调整交叉熵损失权重
# class_weights = torch.tensor([2.0, 1.0]).to(device)  # 假设负类（类别0）较重要，给予更高的权重
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss_fn = nn.CrossEntropyLoss()
# loss_ls = LevelSetLoss(mu=-0.1, lambda1=1., lambda2=0.8, reduction=None)
optim = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=10, eps=1e-6)

# grad_cam = GradCam(model=model)
# # 分类阈值调整
# threshold = 0.6  # 将阈值从0.5提高到0.6

# 训练循环
steps, losses = [], []
log_file = os.path.join(save_root, "training_log.txt")
for epoch in range(epochs):
    avg_loss = 0
    total_l= 0.
    for i, (us_imgs, el_imgs, masks, _, task_labels_central_lnm, clinical_features) in enumerate(loader):
        us_imgs = us_imgs.to(device).permute((0, 3, 1, 2))
        el_imgs = el_imgs.to(device).permute((0, 3, 1, 2))
        masks = masks.to(device)
        labels = task_labels_central_lnm.to(device)
        clinical_features = clinical_features.to(device)

        us_imgs = trans(us_imgs)
        el_imgs = trans(el_imgs)

        pred1, pred2, loss_weights = model(us_imgs, el_imgs, clinical_features)

        loss_fn1 = loss_fn(pred1, labels)
        loss_fn2 = loss_fn(pred2, labels)
        weighted_loss = (loss_fn1 * loss_weights[:, 0] + loss_fn2 * loss_weights[:, 1]).mean()

        total_l += weighted_loss.item()

        optim.zero_grad()
        weighted_loss.backward()
        optim.step()

        avg_loss += weighted_loss.item()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f]" % (epoch, epochs, i, len(loader), weighted_loss.item()))

    steps.append(epoch + 1)
    losses.append(avg_loss / len(loader))

    # 更新学习率
    scheduler.step(avg_loss / len(loader))

    plt.switch_backend('agg')
    plt.figure()
    plt.plot(steps, losses)
    plt.savefig(os.path.join(save_root, "losses.jpg"))
    np.savetxt(os.path.join(save_root, "losses.txt"), np.array(losses))

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), os.path.join(save_root, f"se_resnet50_{epoch + 1}.pth"))
        # 保存当前的损失权重
        loss_weights_path = os.path.join(save_root, f"loss_weights_{epoch + 1}.txt")
        current_loss_weights = loss_weights.detach().cpu().numpy()
        np.savetxt(loss_weights_path, current_loss_weights)

    # 保存日志
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: total_loss={total_l:.6f}, lr={optim.param_groups[0]['lr']}\n")
print("Training completed.")