import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as transforms

from archs.swin_transformer import swin_t
from datasets.LNM_Central import LNM_Central

# configurations
batch_size = 1
use_cuda = True
cuda_id = "0"
model_root = "/data2/wjb/code/PTMC_LNM/trained_models/SwinTransformer/"
epoch_number = 500
result_root = "/data2/wjb/code/PTMC_LNM/results/SwinTransformer/finetune500/"
result_class_dir = os.path.join(result_root, "class")
gradcam_dir = os.path.join(result_root, "gradcam")

if not os.path.exists(result_root):
    os.mkdir(result_root)
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

class MultiModel(nn.Module):
    def __init__(self, num_classes=2, clinical_feature_dim=45, img_feature_dim=768):
        super(MultiModel, self).__init__()
        # model = swin_t(num_classes=1000, pretrained=pretrained)
        # model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.base_model = swin_t(num_classes=2, pretrained=True)
        # 冻结所有层的参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 解冻后三层
        # 遍历stage4模块中的layers，并解冻最后三个SwinBlock层
        stage4_layers = self.base_model.stage4.layers
        # print("Length of stage4_layers:", len(stage4_layers))
        # for i, module_list in enumerate(stage4_layers):
        #     print(f"ModuleList {i} length:", len(module_list))
        #     for j, layer in enumerate(module_list):
        #         print(f"ModuleList {i}, Layer {j}: {type(layer)}")
        # 遍历每个 ModuleList，并解冻其中的后两层
        for i in range(len(stage4_layers)):
            module_list = stage4_layers[i]
            # 解冻 module_list 中的后两层
            for j in range(max(0, len(module_list) - 2), len(module_list)):
                for param in module_list[j].parameters():
                    param.requires_grad = True

        # 移除原始的全连接层
        self.base_model.mlp_head = nn.Identity()

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
            nn.Linear(img_feature_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, clinical_features):
        image_features = self.base_model(x)
        # print("Image features shape:", image_features.shape)
        # print("clinical features shape:", clinical_features.shape)
        image_features_enhanced, clinical_features_enhanced = self.clinical_attention_module(image_features, clinical_features)
        clinical_features_enhanced = self.fc_clinical(clinical_features_enhanced)
        combined_features = torch.cat((image_features_enhanced, clinical_features_enhanced), dim=1)

        # pred_lnm = self.classifier_lnm(combined_features)
        pred_central_lnm = self.classifier_central_lnm(combined_features)

        return pred_central_lnm

# load dataset
data_root = "/data2/wjb/code/PTMC_LNM/data"
clinical_features_path = '/data2/wjb/code/PTMC_LNM/data/lnm_clinical_features.npy'  # 临床特征数据的路径
clinical_flag_path = '/data2/wjb/code/PTMC_LNM/data/lnm_clinical_flag.txt'  # 临床特征顺序文件

# load model and pre-trained parameters
clinical_feature_dim = np.load(clinical_features_path).shape[1]
model = MultiModel(num_classes=2, clinical_feature_dim=clinical_feature_dim)
model.load_state_dict(torch.load(model_root + f"swin_transformer_{epoch_number}.pth"))
model.eval()

if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    model.cuda()
    device = "cuda"
else:
    device = "cpu"


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
final_stage_module = model.base_model.stage4
final_stage_module.register_forward_hook(forward_hook)
final_stage_module.register_full_backward_hook(backward_hook)

y_gt = []
y_pred = []
y_prob = []
for i, (name, us_imgs, _, _, _, task_labels_central_lnm, clinical_features) in enumerate(loader):
    imgs = us_imgs.to(device).permute((0, 3, 1, 2))
    imgs = trans(imgs)
    labels = task_labels_central_lnm.long()

    clinical_features = clinical_features.to(device)
    imgs.requires_grad = True

    preds = model(imgs, clinical_features)

    # 计算Grad-CAM
    preds.sum().backward(retain_graph=True)

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

    img = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    img = img - np.min(img)
    img = img / np.max(img)

    save_gradcam(img, gradcam, os.path.join(gradcam_dir, f"{name[0]}.jpg"))

    p_cls = preds.argmax(1).cpu().numpy()
    p_prob = preds.round().long().cpu().numpy()[:, 1]

    y_gt.extend(labels.numpy())
    y_pred.extend(p_cls)
    y_prob.extend(p_prob)

    image_name = name[0]
    result_file_path = os.path.join(result_class_dir, f"{image_name}.txt")
    with open(result_file_path, "w") as f:
        f.write(f"Image Name: {image_name}\n")
        f.write(f"Ground Truth Central LNM: {task_labels_central_lnm.cpu().numpy()[0]}\n")
        f.write(f"Prediction Central LNM: {p_cls[0]}\n")
        f.write(f"Probability Central LNM: {p_prob[0]}\n")

    sys.stdout.write(
        "\r[Batch %d/%d] [name %s] [label %d] [prediction %d]"
        % (
            i,
            len(loader),
            name,
            labels.numpy()[0],
            p_cls[0],
        )
    )

    if False:
        imgs = imgs.permute((0, 2, 3, 1)).cpu().numpy()[0]
        imgs += 0.5
        plt.imshow(imgs, cmap="gray")
        plt.show()

print("\n")
print(y_gt)
print(y_pred)
print(y_prob)

fpr, tpr, thresholds = metrics.roc_curve(y_gt, y_prob)
auc = metrics.auc(fpr, tpr)
con_mat = metrics.confusion_matrix(y_gt, y_pred)

roc = np.c_[fpr, tpr]
np.savetxt(result_root + "roc.txt", roc, fmt="%6f")
np.savetxt(result_root + "auc.txt", [auc], fmt="%6f")
np.savetxt(result_root + "confusion_matrix.txt", con_mat, fmt="%d")

print(con_mat)
plt.figure()
plt.plot(fpr, tpr)
plt.title(str(auc))
plt.show()