import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import cv2
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from archs.se_resnet import se_resnet50
from archs.modified_se_resnet import ModifiedSEResNet50
from datasets.LNM_Central import LNM_Central

# Configurations
batch_size = 1
use_cuda = True
cuda_id = "1"
epoch_number = 600
model_root = "/data2/wjb/code/PTMC_LNM/trained_models/Modified_SeResnet_VEU/"
result_root = "/data2/wjb/code/PTMC_LNM/results/Modified_SeResnet_VEU/finetune600/"
result_class_dir = os.path.join(result_root, "class")
gradcam_dir = os.path.join(result_root, "gradcam")

if not os.path.exists(result_root):
    os.makedirs(result_root)
if not os.path.exists(result_class_dir):
    os.makedirs(result_class_dir)
if not os.path.exists(gradcam_dir):
    os.makedirs(gradcam_dir)

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
    def __init__(self, num_classes=2, clinical_feature_dim=45):  # 注意 clinical_feature_dim 可变，测试时需调整
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

data_root = "/data2/wjb/code/PTMC_LNM/data"
# clinical_features_path = "/data2/wjb/code/pumc-thyroid/data/clinical_features.npy"
# clinical_flag_path = "/data2/wjb/code/pumc-thyroid/data/clinical_flag.txt"
clinical_features_path = "/data2/wjb/code/PTMC_LNM/data/lnm_clinical_features.npy"
clinical_flag_path = "/data2/wjb/code/PTMC_LNM/data/lnm_clinical_flag.txt"

# 使用包含临床特征的数据集类
dataset = LNM_Central(
    data_dir=data_root,
    split="test",
    transform=None, with_name=True,
    clinical_features_path=clinical_features_path,
    clinical_flag_path=clinical_flag_path
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5), (1.)),
])

model = DualInputModel()
model.load_state_dict(torch.load(model_root + f"se_resnet50_{epoch_number}.pth"))
model.eval()
weights_file = f"{model_root}loss_weights_{epoch_number}.txt"

if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    model.cuda()
    device = "cuda"
else:
    device = "cpu"

weights = np.loadtxt(weights_file)
average_weights = np.mean(weights, axis=0)

y_gt = []
y_pred = []
y_prob = []

def save_gradcam(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register hooks on the final convolutional layer of feature_extractor1
final_conv_layer = model.base_model.base_model.layer4[-1]
final_conv_layer.register_forward_hook(forward_hook)
final_conv_layer.register_full_backward_hook(backward_hook)

for i, (name, us_imgs, el_imgs, masks, _, task_labels_central_lnm, clinical_features) in enumerate(loader):
    us_imgs = us_imgs.to(device).permute((0, 3, 1, 2))
    el_imgs = el_imgs.to(device).permute((0, 3, 1, 2))
    labels = task_labels_central_lnm.to(device)
    clinical_features = clinical_features.to(device)

    us_imgs = trans(us_imgs)
    el_imgs = trans(el_imgs)

    us_imgs.requires_grad = True

    pred1, pred2, loss_weights = model(us_imgs, el_imgs, clinical_features)
    weighted_pred = average_weights[0] * pred1 + average_weights[1] * pred2
    final_pred = weighted_pred.round().long()

    weighted_pred.sum().backward(retain_graph=True)

    global gradients, activations
    gradients = gradients.cpu().detach().numpy()[0]
    activations = activations.cpu().detach().numpy()[0]
    weights = np.mean(gradients, axis=(1, 2))
    gradcam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        gradcam += w * activations[i, :, :]

    gradcam = np.maximum(gradcam, 0)
    gradcam = gradcam - np.min(gradcam)
    gradcam = gradcam / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (224, 224))

    img = us_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    img = img - np.min(img)
    img = img / np.max(img)

    save_gradcam(img, gradcam, os.path.join(gradcam_dir, f"{name[0]}.jpg"))

    p_cls = final_pred.argmax(1).cpu().numpy()
    p_prob = final_pred.cpu().numpy()[:, 1]

    y_gt.extend(labels.cpu().numpy())
    y_pred.extend(p_cls)
    y_prob.extend(p_prob)

    image_name = name[0]
    result_file_path = os.path.join(result_class_dir, f"{image_name}.txt")
    with open(result_file_path, "w") as f:
        f.write(f"Image Name: {image_name}\n")
        f.write(f"Ground Truth: {labels.cpu().numpy()[0]}\n")
        f.write(f"Prediction: {p_cls[0]}\n")
        f.write(f"Probability: {p_prob[0]}\n")

    sys.stdout.write(
        "\r[Batch %d/%d] [Name %s] [Label %d] [Prediction %d]"
        % (
            i,
            len(loader),
            name,
            labels.cpu().numpy()[0],
            p_cls[0],
        )
    )

print("\n")
print(y_gt)
print(y_pred)
print(y_prob)

fpr, tpr, thresholds = metrics.roc_curve(y_gt, y_prob)
auc = metrics.auc(fpr, tpr)
con_mat = metrics.confusion_matrix(y_gt, y_pred)

# Bootstrapping to compute 95% CI for AUC
n_bootstraps = 1000  # 设置Bootstrap采样的次数
rng_seed = 42  # 随机种子保证结果可重复
bootstrapped_scores = []

# Convert to numpy arrays for easier manipulation
y_gt_np = np.array(y_gt)
y_prob_np = np.array(y_prob)

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # Resample with replacement
    indices = rng.randint(0, len(y_prob_np), len(y_prob_np))
    if len(np.unique(y_gt_np[indices])) < 2:
        # We need at least one positive and one negative sample to compute ROC AUC
        continue

    score = metrics.roc_auc_score(y_gt_np[indices], y_prob_np[indices])
    bootstrapped_scores.append(score)

# 计算95%置信区间
sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# 95%置信区间
confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
# 输出结果
print(f"AUC: {auc:.3f}")
print(f"95% CI for AUC: [{confidence_lower:.3f}, {confidence_upper:.3f}]")

roc = np.c_[fpr, tpr]
np.savetxt(result_root + "roc.txt", roc, fmt="%6f")
np.savetxt(result_root + "auc.txt", [auc], fmt="%6f")
np.savetxt(result_root + "confusion_matrix.txt", con_mat, fmt="%d")
np.savetxt(os.path.join(result_root, "95%CI.txt"), [[confidence_lower, confidence_upper]], fmt="%6f")

print(con_mat)
plt.figure()
plt.plot(fpr, tpr)
plt.title(str(auc))
plt.savefig(os.path.join(result_root, "roc_curve.png"))
plt.show()