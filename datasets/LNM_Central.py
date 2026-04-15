import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize
import torchvision.transforms as transforms

class LNM_Central(Dataset):
    def __init__(self, data_dir, split, transform=None, with_name=False, clinical_features_path=None,
                 clinical_flag_path=None):
        self.data_dir = data_dir
        self.split = split
        self.with_name = with_name
        self.transform = transform
        self.clinical_features_path = clinical_features_path
        self.clinical_flag_path = clinical_flag_path
        self.set_file()
        self.load_clinical_features()

    def set_file(self):
        # 读取图像文件名（来自分组txt文件）
        file_path = os.path.join(self.data_dir, 'groups', self.split + '.txt')
        self.files = np.loadtxt(file_path, dtype=str)
        self.files = [f.rstrip(".jpg") for f in self.files]

    def load_clinical_features(self):
        if self.clinical_features_path and self.clinical_flag_path:
            # 加载排序好的临床特征
            self.clinical_features = np.load(self.clinical_features_path)
            with open(self.clinical_flag_path, 'r') as f:
                clinical_flags = [line.strip().rstrip(".jpg") for line in f]

            # 确保 clinical_flags 的顺序与 self.files 对应
            self.clinical_feature_dict = {name: feature for name, feature in
                                          zip(clinical_flags, self.clinical_features)}

            # 对 self.files 进行重排序，以确保它与临床特征的顺序一致
            self.files = [file for file in self.files if file in self.clinical_feature_dict]
            self.clinical_features = [self.clinical_feature_dict[file] for file in self.files]
        else:
            self.clinical_features = None
            print("No clinical features found. Please check your paths.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        B_mode_path = os.path.join(self.data_dir, "imgs", self.files[i] + ".jpg")
        VEU_path = os.path.join(self.data_dir, "VEU", self.files[i] + ".jpg")
        mask_path = os.path.join(self.data_dir, "masks", self.files[i] + ".jpg")
        # class_path = os.path.join(self.data_dir, "classes", self.files[i] + ".txt")
        # VEGF_path = os.path.join(self.data_dir, "VEGF", self.files[i] + ".txt")  # VEGF标签路径
        LNM_file = os.path.join(self.data_dir, "LNM")  # LNM标签目录
        LNM_path = os.path.join(LNM_file, "LNM", self.files[i] + ".txt")
        CentralLNM_path = os.path.join(LNM_file, "CentralLNM", self.files[i] + ".txt")
        # LateralLNM_path = os.path.join(LNM_file, "LateralLNM", self.files[i] + ".txt")

        B_mode = imread(B_mode_path)
        mask = imread(mask_path)
        VEU = imread(VEU_path)

        # cls_str = np.loadtxt(class_path, dtype=str)
        # if cls_str == "Ki67>5%" or cls_str == "Ki67>=5%":
        #     cls_ki67 = 1
        # elif cls_str == "Ki67<5%" or cls_str == "Ki67<=5%":
        #     cls_ki67 = 0
        #
        # VEGF_str = np.loadtxt(VEGF_path, dtype=str)
        # if VEGF_str == "1":
        #     cls_vegf = 1
        # elif VEGF_str == "0":
        #     cls_vegf = 0

        LNM_str = np.loadtxt(LNM_path, dtype=str)
        if LNM_str == "1":
            cls_LNM = 1
        elif LNM_str == "0":
            cls_LNM = 0

        CentralLNM_path_str = np.loadtxt(CentralLNM_path, dtype=str)
        if CentralLNM_path_str == "1":
            cls_CentralLNM = 1
        elif CentralLNM_path_str == "0":
            cls_CentralLNM = 0

        # LateralLNM_path_str = np.loadtxt(LateralLNM_path, dtype=str)
        # if LateralLNM_path_str == "1":
        #     cls_LateralLNM = 1
        # elif LateralLNM_path_str == "0":
        #     cls_LateralLNM = 0

        masked_img, masked_mask, _ = mask_image(B_mode, mask, self.files[i])
        masked_img = resize(masked_img, [256, 256])
        masked_mask = resize(masked_mask, [256, 256])

        masked_VEU, _, _ = mask_image(VEU, mask, self.files[i])
        masked_VEU = resize(masked_VEU, [256, 256])

        if self.transform:
            masked_img = self.transform(masked_img)
            masked_VEU = self.transform(masked_VEU)
            masked_mask = transforms.ToTensor()(masked_mask)  # 将 mask 转换为 tensor

        # 获取对应的临床特征
        if self.clinical_features is not None:
            clinical_feature = torch.tensor(self.clinical_features[i], dtype=torch.float32)
        else:
            clinical_feature = torch.tensor([])  # 如果没有临床特征，返回一个空张量

        if self.with_name:
            return self.files[i], masked_img, masked_VEU, masked_mask, cls_LNM, cls_CentralLNM, clinical_feature
        else:
            return masked_img, masked_VEU, masked_mask, cls_LNM, cls_CentralLNM, clinical_feature

def mask_image(image, mask, image_name, ext=0.3):
    h, w = image.shape[:2]
    xs, ys = np.where(mask)

    # 如果掩码为空，打印警告并返回原始图像及掩码和图像名称
    if xs.size == 0 or ys.size == 0:
        print(f"Warning: Empty mask detected for image {image_name}. Returning original image and mask.")
        return image, mask, image_name

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    dx, dy = abs(x_max - x_min), abs(y_max - y_min)

    if dx > dy:
        x_start = max(0, x_min - int(dx * ext))
        x_end = min(h, x_max + int(dx * ext))
        yc = (y_min + y_max) / 2
        dh = abs(x_end - x_start)
        y_start = max(0, int(yc - dh // 2))
        y_end = min(w, int(yc + dh // 2))
    elif dx < dy:
        y_start = max(0, y_min - int(dy * ext))
        y_end = min(w, y_max + int(dy * ext))
        xc = (x_min + x_max) / 2
        dw = abs(y_end - y_start)
        x_start = max(0, int(xc - dw // 2))
        x_end = min(h, int(xc + dw // 2))
    else:
        x_start = max(0, x_min - int(dx * ext))
        x_end = min(h, x_max + int(dx * ext))
        y_start = max(0, y_min - int(dy * ext))
        y_end = min(w, int(y_max + dy * ext))

    x_start, x_end = round(x_start), round(x_end)
    y_start, y_end = round(y_start), round(y_end)

    masked_img = image[x_start:x_end, y_start:y_end]
    masked_mask = mask[x_start:x_end, y_start:y_end]

    # 检查剪裁后的图像或掩码是否为空
    if masked_img.size == 0 or masked_mask.size == 0:
        print(f"Warning: Cropped image or mask is empty for image {image_name}. Returning original image and mask.")
        return image, mask, image_name

    return masked_img, masked_mask, image_name