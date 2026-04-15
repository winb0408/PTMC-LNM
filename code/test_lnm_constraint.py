import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn import metrics
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from datasets.LNM_Central import withVEGF_LNM  # 使用包含LNM的数据集类
from archs.modified_se_resnet import ModifiedSEResNet50

# Configurations
batch_size = 1
use_cuda = True
cuda_id = "1"
epoch_number = 500
model_root = "/data2/wjb/code/pumc-thyroid/code/trained_models/data_new/Multitask_LNM_constraint_VEGF/"
result_root = "/data2/wjb/code/pumc-thyroid/code/results/data_new/Multitask_LNM_constraint_VEGF/finetune500/"
result_class_dir = os.path.join(result_root, "class")
gradcam_dir = os.path.join(result_root, "gradcam")

data_root = "/data2/wjb/code/pumc-thyroid/data"
clinical_features_path = "/data2/wjb/code/pumc-thyroid/data/lnm_clinical_features.npy"
clinical_flag_path = "/data2/wjb/code/pumc-thyroid/data/lnm_clinical_flag.txt"

if not os.path.exists(result_root):
    os.makedirs(result_root)
if not os.path.exists(result_class_dir):
    os.makedirs(result_class_dir)
if not os.path.exists(gradcam_dir):
    os.makedirs(gradcam_dir)

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
    def __init__(self, num_classes=2, clinical_feature_dim=32, img_feature_dim=1000):
        super(MultiTaskModel, self).__init__()
        self.feature_extractor = ModifiedSEResNet50(num_classes=1000)

        self.clinical_attention_module = ClinicalAttentionModule(img_feature_dim, clinical_feature_dim)

        # Classifiers for each task
        self.classifier_ki67 = nn.Sequential(
            nn.Linear(img_feature_dim + clinical_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.classifier_central_lnm = nn.Sequential(
            nn.Linear(img_feature_dim + clinical_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.classifier_lateral_lnm = nn.Sequential(
            nn.Linear(img_feature_dim + clinical_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.classifier_vegf = nn.Sequential(
            nn.Linear(img_feature_dim + clinical_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, clinical_features, task_labels_ki67, task_labels_central_lnm, task_labels_lateral_lnm):
        f = self.feature_extractor(x)
        f_enhanced, clinical_features_enhanced = self.clinical_attention_module(f, clinical_features)
        combined_features = torch.cat((f_enhanced, clinical_features_enhanced), dim=1)

        pred_ki67 = self.classifier_ki67(combined_features)
        pred_central_lnm = self.classifier_central_lnm(combined_features)
        pred_lateral_lnm = self.classifier_lateral_lnm(combined_features)
        pred_vegf = self.classifier_vegf(combined_features)

        return pred_ki67, pred_central_lnm, pred_lateral_lnm, pred_vegf

# 数据变换
trans = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5,), (1.0,))
])

# 数据加载
dataset = withVEGF_LNM(
    data_dir=data_root,
    split="test",
    with_name=True,
    clinical_features_path=clinical_features_path,
    clinical_flag_path=clinical_flag_path
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 模型初始化
clinical_feature_dim = np.load(clinical_features_path).shape[1]
model = MultiTaskModel(num_classes=2, clinical_feature_dim=clinical_feature_dim)
model.load_state_dict(torch.load(model_root + f"multi_task_lnm_se_resnet50_{epoch_number}.pth"))
model.eval()

if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    model.cuda()
    device = "cuda"
else:
    device = "cpu"

# 初始化存储预测结果的列表
y_gt_ki67 = []
y_pred_ki67 = []
y_prob_ki67 = []
y_gt_lnm = []
y_pred_lnm = []
y_prob_lnm = []
y_gt_central_lnm = []
y_pred_central_lnm = []
y_prob_central_lnm = []
y_gt_lateral_lnm = []
y_pred_lateral_lnm = []
y_prob_lateral_lnm = []
y_gt_vegf = []
y_pred_vegf = []
y_prob_vegf = []

def save_gradcam(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))

# Register hooks on the final convolutional layer of the base model
# 找到最终卷积层并注册钩子
# 对于FPN，我们可能需要进一步细化
# 如果希望只在特定输出特征图上应用Grad-CAM，可以选择p1到p4中的某一个.
#   如果你想观察全局的特征，选择更高层的特征图，如p4。这可以让你看到模型如何在更抽象的层面上做出决策。
#   如果你想观察局部的特征，选择分辨率较高的特征图，如p1。这更适合细粒度的对象定位
# 使用 hook 在 FPN 输出中捕获 p1 激活

global activations, gradients
activations = None
gradients = None
def forward_hook(module, input, output):
    global activations
    activations = output
def backward_hook(module, grad_input, grad_output):
    global gradients
    if grad_output is not None and len(grad_output) > 0:
        gradients = grad_output[0]
    else:
        print("Warning: grad_output is empty or None.")

# 访问 `fpn` 组件并注册钩子
final_conv_layer = model.feature_extractor.base_model.layer4[-1]

final_conv_layer.register_forward_hook(forward_hook)
final_conv_layer.register_full_backward_hook(backward_hook)

for i, (name, us_imgs, _, _, task_labels_ki67, task_labels_lnm, task_labels_central_lnm, task_labels_lateral_lnm, task_labels_vegf, clinical_features) in enumerate(loader):
    us_imgs = us_imgs.to(device).permute((0, 3, 1, 2))
    clinical_features = clinical_features.to(device)
    task_labels_ki67 = task_labels_ki67.to(device)
    task_labels_lnm = task_labels_lnm.to(device)
    task_labels_central_lnm = task_labels_central_lnm.to(device)
    task_labels_lateral_lnm = task_labels_lateral_lnm.to(device)
    task_labels_vegf = task_labels_vegf.to(device)

    us_imgs = trans(us_imgs)

    us_imgs.requires_grad = True

    pred_ki67_bmode, pred_central_lnm, pred_lateral_lnm, pred_vegf = model(us_imgs, clinical_features, task_labels_ki67,
                                                          task_labels_central_lnm,
                                                          task_labels_lateral_lnm)

    final_pred_ki67 = pred_ki67_bmode.round().long()
    final_pred_central_lnm = pred_central_lnm.round().long()
    final_pred_lateral_lnm = pred_lateral_lnm.round().long()
    final_pred_vegf = pred_vegf.round().long()
    pred_lnm_constraint = torch.max(pred_central_lnm.softmax(dim=1)[:, 1], pred_lateral_lnm.softmax(dim=1)[:, 1])
    final_pred_lnm = pred_lnm_constraint.round().long()

    pred_ki67_bmode.sum().backward(retain_graph=True)

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

    p_cls_ki67 = final_pred_ki67.detach().argmax(1).cpu().numpy()
    p_prob_ki67 = pred_ki67_bmode.detach().cpu().numpy()[:, 1]
    p_cls_lnm = final_pred_lnm
    p_cls_central_lnm = final_pred_central_lnm.detach().argmax(1).cpu().numpy()
    p_prob_central_lnm = pred_central_lnm.detach().cpu().numpy()[:, 1]
    p_cls_lateral_lnm = final_pred_lateral_lnm.detach().argmax(1).cpu().numpy()
    p_prob_lateral_lnm = pred_lateral_lnm.detach().cpu().numpy()[:, 1]
    p_cls_vegf = final_pred_vegf.detach().argmax(1).cpu().numpy()
    p_prob_vegf = pred_vegf.detach().cpu().numpy()[:, 1]


    # 存储预测结果
    y_gt_ki67.extend(task_labels_ki67.cpu().numpy())
    y_pred_ki67.extend(p_cls_ki67)
    y_prob_ki67.extend(p_prob_ki67)
    y_gt_lnm.extend(task_labels_lnm.cpu().numpy())
    y_pred_lnm.extend(p_cls_lnm)
    y_gt_central_lnm.extend(task_labels_central_lnm.cpu().numpy())
    y_pred_central_lnm.extend(p_cls_central_lnm)
    y_prob_central_lnm.extend(p_prob_central_lnm)
    y_gt_lateral_lnm.extend(task_labels_lateral_lnm.cpu().numpy())
    y_pred_lateral_lnm.extend(p_cls_lateral_lnm)
    y_prob_lateral_lnm.extend(p_prob_lateral_lnm)
    y_gt_vegf.extend(task_labels_vegf.cpu().numpy())
    y_pred_vegf.extend(p_cls_vegf)
    y_prob_vegf.extend(p_prob_vegf)


    image_name = name[0]
    result_file_path = os.path.join(result_class_dir, f"{image_name}.txt")
    with open(result_file_path, "w") as f:
        f.write(f"Image Name: {image_name}\n")
        f.write(f"Ground Truth Ki67: {task_labels_ki67.cpu().numpy()[0]}\n")
        f.write(f"Prediction Ki67: {p_cls_ki67[0]}\n")
        f.write(f"Probability Ki67: {p_prob_ki67[0]}\n")
        f.write(f"Ground Truth LNM: {task_labels_lnm.cpu().numpy()[0]}\n")
        f.write(f"Prediction LNM: {p_cls_lnm[0]}\n")
        f.write(f"Ground Truth Central LNM: {task_labels_central_lnm.cpu().numpy()[0]}\n")
        f.write(f"Prediction Central LNM: {p_cls_central_lnm[0]}\n")
        f.write(f"Probability Central LNM: {p_prob_central_lnm[0]}\n")
        f.write(f"Ground Truth Lateral LNM: {task_labels_lateral_lnm.cpu().numpy()[0]}\n")
        f.write(f"Prediction Lateral LNM: {p_cls_lateral_lnm[0]}\n")
        f.write(f"Probability Lateral LNM: {p_prob_lateral_lnm[0]}\n")
        f.write(f"Ground Truth VEGF: {task_labels_vegf.cpu().numpy()[0]}\n")
        f.write(f"Prediction VEGF: {p_cls_vegf[0]}\n")
        f.write(f"Probability VEGF: {p_prob_vegf[0]}\n")

    sys.stdout.write(
        "\r[Batch %d/%d] [Name %s] [Task_ki67 Label %d] [Task_ki67 Prediction %d] [Task_lnm Label %d] [Task_lnm Prediction %d] [Task_central_lnm Label %d] [Task_central_lnm Prediction %d] [Task_lateral_lnm Label %d] [Task_lateral_lnm Prediction %d]"
        % (
            i,
            len(loader),
            name,
            task_labels_ki67.cpu().numpy()[0],
            p_cls_ki67[0],
            task_labels_lnm.cpu().numpy()[0],
            p_cls_lnm[0],
            task_labels_central_lnm.cpu().numpy()[0],
            p_cls_central_lnm[0],
            task_labels_lateral_lnm.cpu().numpy()[0],
            p_cls_lateral_lnm[0],
        )
    )

    print("\nTesting completed.")


def compute_95ci(y_gt, y_prob, n_bootstraps=1000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []

    y_gt_np = np.array(y_gt)
    y_prob_np = np.array(y_prob)

    for i in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, len(y_prob_np), len(y_prob_np))
        if len(np.unique(y_gt_np[indices])) < 2:
            continue  # 需要至少一个正样本和一个负样本

        score = metrics.roc_auc_score(y_gt_np[indices], y_prob_np[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95%置信区间
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper


# 评估两个任务的性能
def evaluate_performance(y_gt, y_pred, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_gt, y_prob)
    auc = metrics.auc(fpr, tpr)
    con_mat = metrics.confusion_matrix(y_gt, y_pred)
    ci_lower, ci_upper = compute_95ci(y_gt, y_prob)
    return fpr, tpr, auc, con_mat, ci_lower, ci_upper

# 计算95%置信区间
def compute_95ci(y_gt, y_prob, n_bootstraps=1000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []

    y_gt_np = np.array(y_gt)
    y_prob_np = np.array(y_prob)

    for i in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, len(y_prob_np), len(y_prob_np))
        if len(np.unique(y_gt_np[indices])) < 2:
            continue  # 需要至少一个正样本和一个负样本

        score = metrics.roc_auc_score(y_gt_np[indices], y_prob_np[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95%置信区间
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return confidence_lower, confidence_upper

# 评估四个任务的性能
fpr_ki67, tpr_ki67, auc_ki67, con_mat_ki67, ci_lower_ki67, ci_upper_ki67 = evaluate_performance(y_gt_ki67, y_pred_ki67, y_prob_ki67)
fpr_central_lnm, tpr_central_lnm, auc_central_lnm, con_mat_central_lnm, ci_lower_central_lnm, ci_upper_central_lnm = evaluate_performance(y_gt_central_lnm, y_pred_central_lnm, y_prob_central_lnm)
fpr_lateral_lnm, tpr_lateral_lnm, auc_lateral_lnm, con_mat_lateral_lnm, ci_lower_lateral_lnm, ci_upper_lateral_lnm = evaluate_performance(y_gt_lateral_lnm, y_pred_lateral_lnm, y_prob_lateral_lnm)
fpr_vegf, tpr_vegf, auc_vegf, con_mat_vegf, ci_lower_vegf, ci_upper_vegf = evaluate_performance(y_gt_vegf, y_pred_vegf, y_prob_vegf)

# 打印评估结果
print("Task 1 (Ki67) Confusion Matrix:")
print(con_mat_ki67)
print(f"Task 1 (Ki67) AUC: {auc_ki67:.3f}")
print(f"Task 1 (Ki67) 95% CI: [{ci_lower_ki67:.3f}, {ci_upper_ki67:.3f}]")

print("Task 3 (CentralLNM) Confusion Matrix:")
print(con_mat_central_lnm)
print(f"Task 3 (CentralLNM) AUC: {auc_central_lnm:.3f}")
print(f"Task 3 (CentralLNM) 95% CI: [{ci_lower_central_lnm:.3f}, {ci_upper_central_lnm:.3f}]")

print("Task 4 (LateralLNM) Confusion Matrix:")
print(con_mat_lateral_lnm)
print(f"Task 4 (LateralLNM) AUC: {auc_lateral_lnm:.3f}")
print(f"Task 4 (LateralLNM) 95% CI: [{ci_lower_lateral_lnm:.3f}, {ci_upper_lateral_lnm:.3f}]")

print("Task 5 (VEGF) Confusion Matrix:")
print(con_mat_vegf)
print(f"Task 5 (VEGF) AUC: {auc_vegf:.3f}")
print(f"Task 5 (VEGF) 95% CI: [{ci_lower_vegf:.3f}, {ci_upper_vegf:.3f}]")

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_ki67, tpr_ki67, label=f'Ki67 (AUC = {auc_ki67:.3f}, 95% CI [{ci_lower_ki67:.3f}, {ci_upper_ki67:.3f}])')
plt.plot(fpr_central_lnm, tpr_central_lnm, label=f'CentralLNM (AUC = {auc_central_lnm:.3f}, 95% CI [{ci_lower_central_lnm:.3f}, {ci_upper_central_lnm:.3f}])')
plt.plot(fpr_lateral_lnm, tpr_lateral_lnm, label=f'LateralLNM (AUC = {auc_lateral_lnm:.3f}, 95% CI [{ci_lower_lateral_lnm:.3f}, {ci_upper_lateral_lnm:.3f}])')
plt.plot(fpr_vegf, tpr_vegf, label=f'VEGF (AUC = {auc_vegf:.3f}, 95% CI [{ci_lower_vegf:.3f}, {ci_upper_vegf:.3f}])')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig(os.path.join(result_root, "roc_curves.png"))
plt.show()

# 保存评估结果
np.savetxt(os.path.join(result_root, "roc_ki67.txt"), np.c_[fpr_ki67, tpr_ki67], fmt="%6f")
np.savetxt(os.path.join(result_root, "auc_ki67.txt"), [auc_ki67], fmt="%6f")
np.savetxt(os.path.join(result_root, "confusion_matrix_ki67.txt"), con_mat_ki67, fmt="%d")
np.savetxt(os.path.join(result_root, "ci_ki67.txt"), [[ci_lower_ki67, ci_upper_ki67]], fmt="%6f")

np.savetxt(os.path.join(result_root, "roc_central_lnm.txt"), np.c_[fpr_central_lnm, tpr_central_lnm], fmt="%6f")
np.savetxt(os.path.join(result_root, "auc_central_lnm.txt"), [auc_central_lnm], fmt="%6f")
np.savetxt(os.path.join(result_root, "confusion_matrix_central_lnm.txt"), con_mat_central_lnm, fmt="%d")
np.savetxt(os.path.join(result_root, "ci_central_lnm.txt"), [[ci_lower_central_lnm, ci_upper_central_lnm]], fmt="%6f")

np.savetxt(os.path.join(result_root, "roc_lateral_lnm.txt"), np.c_[fpr_lateral_lnm, tpr_lateral_lnm], fmt="%6f")
np.savetxt(os.path.join(result_root, "auc_lateral_lnm.txt"), [auc_lateral_lnm], fmt="%6f")
np.savetxt(os.path.join(result_root, "confusion_matrix_lateral_lnm.txt"), con_mat_lateral_lnm, fmt="%d")
np.savetxt(os.path.join(result_root, "ci_lateral_lnm.txt"), [[ci_lower_lateral_lnm, ci_upper_lateral_lnm]], fmt="%6f")

np.savetxt(os.path.join(result_root, "roc_vegf.txt"), np.c_[fpr_vegf, tpr_vegf], fmt="%6f")
np.savetxt(os.path.join(result_root, "auc_vegf.txt"), [auc_vegf], fmt="%6f")
np.savetxt(os.path.join(result_root, "confusion_matrix_vegf.txt"), con_mat_vegf, fmt="%d")
np.savetxt(os.path.join(result_root, "ci_vegf.txt"), [[ci_lower_vegf, ci_upper_vegf]], fmt="%6f")