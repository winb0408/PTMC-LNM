'''
nohup python train_swin.py > nohup3.out &
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as transforms

from archs.swin_transformer import swin_t
from datasets.LNM_Central import LNM_Central

# configurations
epochs = 500
batch_size = 64
lr = 2e-5  # learning rate
use_cuda = True
cuda_id = "1"
pretrained = True
save_root = "/data2/wjb/code/PTMC_LNM/trained_models/SwinTransformer/"

#
if not os.path.exists(save_root):
    os.mkdir(save_root)

# load model
# if pretrained:
#     model = swin_t(num_classes=1000, pretrained=pretrained)
#     model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
# else:
#     model = swin_t(num_classes=2)

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
    def __init__(self, num_classes=2, clinical_feature_dim=32, img_feature_dim=768):
        super(MultiModel, self).__init__()
        # model = swin_t(num_classes=1000, pretrained=pretrained)
        # model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.base_model = swin_t(num_classes=2, pretrained=pretrained)
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
dataset = LNM_Central(
    data_dir=data_root,
    split="train",
    clinical_features_path=clinical_features_path,
    clinical_flag_path=clinical_flag_path
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 模型初始化
clinical_feature_dim = np.load(clinical_features_path).shape[1]
model = MultiModel(num_classes=2, clinical_feature_dim=clinical_feature_dim)

if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    model.cuda()
    device = "cuda"
else:
    device = "cpu"
model.to(device)

trans = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5), (1.)),
])

# loss function and optimizer
# weight=torch.tensor([0.4, 0.6])
# if use_cuda:
#     weight = weight.to(device)
loss_fn = nn.CrossEntropyLoss()

optim = Adam(model.parameters(), lr=lr)
# if pretrained:
#    optim = Adam(model.fc.parameters(), lr=lr)
# else:
#    optim = Adam(model.parameters(), lr=lr)

steps, losses = [], []
for epoch in range(epochs):
    avg_loss = 0
    for i, (us_imgs, _, _, _, task_labels_central_lnm, clinical_features) in enumerate(loader):
        imgs = us_imgs.to(device).permute((0, 3, 1, 2))
        imgs = trans(imgs)
        imgs += 0.01 * torch.randn((batch_size, 3, 224, 224), device=device)
        labels = task_labels_central_lnm.long().to(device)
        # print("clinical features shape0:", clinical_features.shape)
        clinical_features = clinical_features.to(device)

        preds = model(imgs, clinical_features)

        loss = loss_fn(preds, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        avg_loss += loss.item()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
            % (
                epoch,
                epochs,
                i,
                len(loader),
                loss.item(),
            )
        )
    steps.append(epoch + 1)
    losses.append(avg_loss / i)
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(steps, losses)
    plt.savefig(save_root + "losses.jpg")
    np.savetxt(save_root + "losses.txt", losses)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_root + "swin_transformer_%d.pth" % (epoch + 1))