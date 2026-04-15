import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
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
from archs.modified_se_resnet import ModifiedSEResNet50

# 设定环境
use_cuda = True
cuda_id = "0"
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
torch.cuda.set_device(0)

# 路径和超参数
data_root = "/data2/wjb/code/PTMC_LNM/data"
save_root = "/data2/wjb/code/PTMC_LNM/trained_models/Modified_SeResnet/"
epochs = 600
batch_size = 64
lr = 2e-5
clinical_features_path = '/data2/wjb/code/PTMC_LNM/data/lnm_clinical_features.npy'  # 临床特征数据的路径
clinical_flag_path = '/data2/wjb/code/PTMC_LNM/data/lnm_clinical_flag.txt'  # 临床特征顺序文件

if not os.path.exists(save_root):
    os.mkdir(save_root)

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
        attention_weights_clinical = self.sigmoid(self.relu(img_features_transformed + clinical_features))
        attention_weights_img = self.sigmoid(self.relu(clinical_features_transformed + img_features))

        # 应用注意力权重
        enhanced_img_features = img_features * attention_weights_img
        enhanced_clinical_features = clinical_features * attention_weights_clinical

        return enhanced_img_features, enhanced_clinical_features

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=2, clinical_feature_dim=32, img_feature_dim=1024):
        super(MultiTaskModel, self).__init__()
        # self.feature_extractor = ModifiedSEResNet50(num_classes=1000)
        self.feature_extractor = ModifiedSEResNet50(num_classes=1024)
        self.clinical_attention_module = ClinicalAttentionModule(img_feature_dim, clinical_feature_dim)
        self.fc_clinical = nn.Linear(clinical_feature_dim, 256)
        # Classifiers for each task
        # self.classifier_lnm = nn.Sequential(
        #     nn.Linear(256 * 4 + 256, 512), #img_feature_dim + clinical_feature_dim
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )

        self.classifier_central_lnm = nn.Sequential(
            nn.Linear(256 * 4 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, clinical_features):
        image_features = self.feature_extractor(x)

        image_features_enhanced, clinical_features_enhanced = self.clinical_attention_module(image_features, clinical_features)
        clinical_features_enhanced = self.fc_clinical(clinical_features_enhanced)
        combined_features = torch.cat((image_features_enhanced, clinical_features_enhanced), dim=1)

        # pred_lnm = self.classifier_lnm(combined_features)
        pred_central_lnm = self.classifier_central_lnm(combined_features)

        return pred_central_lnm

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
clinical_feature_dim = np.load(clinical_features_path).shape[1]
model = MultiTaskModel(num_classes=2, clinical_feature_dim=clinical_feature_dim)
model.to(device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=10, eps=1e-6)

# 训练循环
steps, losses_central_lnm= [], []
log_file = os.path.join(save_root, "training_log.txt")

# # 初始化任务权重和关键性能指标(KPI)
# task_weights = {
#     "ki67": 0.5,
#     "lnm": 0.5,
#     "central_lnm": 0.5,
#     # "vegf": 0.2
# }
# kpi_avg_losses = {
#     "ki67": 0.5,
#     "lnm": 0.5,
#     "central_lnm": 0.5,
#     # "vegf": 0
# }
# constraint_weight = 0.5  # 用于控制约束损失的影响
# alpha = 0.9  # 平滑参数，用于动态调整权重

for epoch in range(epochs):
    # avg_loss_lnm = 0
    avg_loss_central_lnm = 0
    # avg_constraint_loss = 0
    # avg_total_loss = 0
    for i, (us_imgs, _, _, _, task_labels_central_lnm, clinical_features) in enumerate(loader):
        us_imgs = us_imgs.to(device).permute((0, 3, 1, 2))
        clinical_features = clinical_features.to(device)

        # task_labels_lnm = task_labels_lnm.to(device)
        task_labels_central_lnm = task_labels_central_lnm.to(device)

        us_imgs = trans(us_imgs)

        pred_central_lnm = model(us_imgs, clinical_features)

        # 计算每个任务的损失
        # loss_lnm = loss_fn(pred_lnm, task_labels_central_lnm)
        loss_central_lnm = loss_fn(pred_central_lnm, task_labels_central_lnm)
        # loss_vegf = loss_fn(pred_vegf, task_labels_vegf)

        # 计算LNM约束损失
        # pred_lateral_lnm_constraint = (pred_lnm > 0) & (~pred_central_lnm > 0)
        # pred_lateral_lnm_constraint.to(device)
        # # pred_lateral_lnm_constraint = torch.max(pred_central_lnm.softmax(dim=1)[:, 1], pred_lateral_lnm.softmax(dim=1)[:, 1])
        # true_lateral_lnm_constraint = task_labels_lateral_lnm.float()
        # constraint_loss = F.binary_cross_entropy(pred_lateral_lnm_constraint, true_lateral_lnm_constraint)

        # 动态计算总损失
        # total_loss = (
        #         task_weights["central_lnm"] * loss_lnm +
        #         task_weights["lateral_lnm"] * loss_central_lnm +
        #         # task_weights["vegf"] * loss_vegf +
        #         constraint_weight * constraint_loss
        # )

        optim.zero_grad()
        loss_central_lnm.backward()
        optim.step()

        avg_loss_central_lnm += loss_central_lnm.item()
        # avg_constraint_loss += constraint_loss.item()
        # avg_total_loss += total_loss.item()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Loss CentralLNM: %f]" % (
                epoch, epochs, i, len(loader), loss_central_lnm.item()))

    steps.append(epoch + 1)
    losses_central_lnm.append(avg_loss_central_lnm / len(loader))
    # losses_constraint.append(avg_constraint_loss / len(loader))
    # total_losses.append(avg_total_loss / len(loader))

    # 计算平均损失
    avg_loss_central_lnm /= len(loader)

    # 更新每个任务的KPI（使用指数移动平均）
    # kpi_avg_losses["lnm"] = alpha * kpi_avg_losses["lnm"] + (1 - alpha) * avg_loss_central_lnm
    # kpi_avg_losses["central_lnm"] = alpha * kpi_avg_losses["central_lnm"] + (1 - alpha) * avg_loss_lateral_lnm

    # 更新任务权重，使得较高的损失获得更高的权重
    # total_kpi_loss = sum(kpi_avg_losses.values())
    # for task in task_weights:
    #     task_weights[task] = kpi_avg_losses[task] / total_kpi_loss

    # 更新学习率
    scheduler.step(avg_loss_central_lnm / len(loader))
    print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {optim.param_groups[0]['lr']}")
    # print(f"Updated Task Weights: {task_weights}, Constraint Loss Weight: {constraint_weight}")

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), save_root + "central_lnm_se_resnet50_%d.pth" % (epoch + 1))

    # 保存日志
    with open(log_file, "a") as f:
        f.write(
            f"Epoch {epoch + 1}: Loss CentralLNM={avg_loss_central_lnm / len(loader):.6f} , lr={optim.param_groups[0]['lr']}\n")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(steps, losses_central_lnm, label='Loss Central_LNM')
# plt.plot(steps, losses_constraint, label='Loss Constraint')
# plt.plot(steps, total_losses, label='Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig(os.path.join(save_root, "losses.jpg"))
plt.show()
# plt.savefig(save_root + "losses.jpg")
np.savetxt(save_root + "losses_lateral_lnm.txt", losses_central_lnm)
# np.savetxt(save_root + "losses_constraint.txt", losses_constraint)
# np.savetxt(save_root + "losses_total.txt", total_losses)

print("Training completed.")