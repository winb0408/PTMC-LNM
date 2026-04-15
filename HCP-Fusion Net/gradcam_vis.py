import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from options import Options
from dataset import build_dataloader, MultiModalDataset
from networks import build_model


class MultiModalGradCAM:
    """
    针对当前 HCP-Fusion Net（late fusion + ResNet backbone）的多模态 Grad-CAM 工具。

    设计假设（与你现在的“满血”配置一致）：
    - fusion_type == 'late'
    - backbone 为 torchvision 的 ResNet 系列（有 layer4 这一层）
    """

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.handles = []

        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

        if model.fusion_type != 'late':
            raise ValueError('MultiModalGradCAM 目前仅支持 fusion_type=late 的配置。')

        # 为每个模态的 backbone.layer4 注册 hook
        for mod, backbone in model.backbones.items():
            if not hasattr(backbone, 'layer4'):
                raise ValueError(
                    f'当前 backbone 不包含 layer4（仅支持 ResNet 系列）。模态: {mod}'
                )

            def _forward_hook(modality):
                def hook(module, inp, out):
                    # out: [B, C, H, W]
                    self.feature_maps[modality] = out.detach()

                return hook

            def _backward_hook(modality):
                def hook(module, grad_in, grad_out):
                    # grad_out[0]: [B, C, H, W]
                    self.gradients[modality] = grad_out[0].detach()

                return hook

            h_f = backbone.layer4.register_forward_hook(_forward_hook(mod))
            h_b = backbone.layer4.register_full_backward_hook(_backward_hook(mod))
            self.handles.extend([h_f, h_b])

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _compute_cam(self, fmap: torch.Tensor, grad: torch.Tensor) -> np.ndarray:
        """
        标准 Grad-CAM 计算。
        fmap, grad: [C, H, W]
        返回归一化到 [0,1] 的 H×W numpy 数组。
        """
        # GAP 求通道权重
        alpha = grad.view(grad.shape[0], -1).mean(dim=1)  # [C]
        cam = (alpha.view(-1, 1, 1) * fmap).sum(dim=0)    # [H, W]
        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()

    def generate_for_batch(
        self,
        batch: Dict,
        task_idx: int,
        task_mode: str,
    ) -> Dict[str, np.ndarray]:
        """
        对单个 batch（通常 batch_size=1）生成各模态的 Grad-CAM 热力图。

        Args:
            batch: 与训练/测试阶段相同的 batch 字典（已在 CPU 上）
            task_idx: 0=LNM, 1=CentralLNM, 2=LateralLNM
            task_mode: 'single' / 'multi_parallel' / 'multi_cascaded'

        Returns:
            {modality_name: heatmap_numpy_HxW}
        """
        self.model.zero_grad()
        self.feature_maps.clear()
        self.gradients.clear()

        # 移动到设备（支持嵌套 dict，行为与 train.py 中 _move_batch_to_device 一致）
        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device, non_blocking=True)
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x

        batch_dev = {k: _to_device(v) for k, v in batch.items()}

        preds = self.model(batch_dev)

        # 选择要解释的任务 logit
        if task_mode == 'single':
            # 仅 lateral
            logit = preds['lateral']
        else:
            # multi_* 模式：按照 TASK_KEYS 顺序
            task_key = self.model.TASK_KEYS[task_idx]  # 'lnm' / 'central' / 'lateral'
            logit = preds[task_key]

        # 只对当前 batch 的第一个样本求导
        if logit.ndim > 1:
            logit = logit[:, 0]
        target_logit = logit[0]
        target_logit.backward(retain_graph=True)

        # 计算各模态 CAM
        cams: Dict[str, np.ndarray] = {}
        for mod in self.model.modalities:
            fmap = self.feature_maps.get(mod, None)    # [B, C, H, W]
            grad = self.gradients.get(mod, None)       # [B, C, H, W]
            if fmap is None or grad is None:
                continue
            fmap_b = fmap[0]
            grad_b = grad[0]
            cams[mod] = self._compute_cam(fmap_b, grad_b)

        return cams


def denormalize_image(t: torch.Tensor, mean, std) -> np.ndarray:
    """
    将归一化后的 Tensor 转回 0-1 的 numpy H×W×3，便于可视化。
    t: [3, H, W]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = t * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    return img_np


def overlay_heatmap_on_image(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    将 CAM 热力图叠加到原图上。
    img: H×W×3, [0,1]
    cam: H×W, [0,1]
    """
    import cv2

    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0  # BGR→RGB, [0,1]

    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_cases(opt, task_idx: int = 2, max_cases: int = 8):
    """
    对验证集或测试集中的若干病例，生成多模态 Grad-CAM 可视化图像。

    Args:
        opt: Options.parse() 返回的配置
        task_idx: 0=LNM, 1=CentralLNM, 2=LateralLNM（默认重点看 Lateral）
        max_cases: 可视化的病例数量（从 val 集按顺序取前 N 个）
    """
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    # 复用临床 loader，构建验证集 DataLoader（batch_size=1, 不打乱）
    val_loader, clinical_loader = build_dataloader(
        opt=opt,
        split='val',
        clinical_loader=None,
        use_weighted_sampler=False,
    )

    # 构建并加载模型
    clinical_input_dim = getattr(opt, 'clinical_input_dim', 0)
    model = build_model(opt, clinical_input_dim=clinical_input_dim)
    model = model.to(device)

    best_ckpt_path = os.path.join(opt.exp_dir, 'best_model.pth')
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f'找不到最佳模型权重: {best_ckpt_path}')
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    gradcam = MultiModalGradCAM(model, device)

    save_root = os.path.join(opt.exp_dir, 'gradcam_vis')
    os.makedirs(save_root, exist_ok=True)

    mean = opt.normalize_mean
    std = opt.normalize_std

    n_cases = 0
    for batch in val_loader:
        if n_cases >= max_cases:
            break

        # 生成 CAM
        cams = gradcam.generate_for_batch(
            batch=batch,
            task_idx=task_idx,
            task_mode=model.task_mode,
        )

        # 取当前病例名（batch_size=1）
        name = batch['name'][0] if isinstance(batch['name'], list) else str(batch['name'])

        # 画一张多模态对比图
        n_modalities = len(opt.modalities)
        fig, axes = plt.subplots(2, n_modalities, figsize=(4 * n_modalities, 6))

        for j, mod in enumerate(opt.modalities):
            # 原图 tensor: [B, 3, H, W]（late fusion）
            img_t = batch['images'][mod][0].cpu()
            img_np = denormalize_image(img_t, mean, std)

            ax_orig = axes[0, j]
            ax_cam = axes[1, j]

            ax_orig.imshow(img_np)
            ax_orig.set_title(f'{mod} - 原图')
            ax_orig.axis('off')

            cam = cams.get(mod, None)
            if cam is not None:
                # CAM 尺寸可能与图像尺寸不同，插值到图像大小
                cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
                cam_up = F.interpolate(
                    cam_t,
                    size=img_np.shape[:2],
                    mode='bilinear',
                    align_corners=False,
                )[0, 0].cpu().numpy()
                overlay = overlay_heatmap_on_image(img_np, cam_up, alpha=0.4)
                ax_cam.imshow(overlay)
                ax_cam.set_title(f'{mod} - Grad-CAM')
            else:
                ax_cam.imshow(img_np)
                ax_cam.set_title(f'{mod} - (无 CAM)')
            ax_cam.axis('off')

        task_name = ['LNM', 'CentralLNM', 'LateralLNM'][task_idx] if task_idx in (0, 1, 2) else str(task_idx)
        fig.suptitle(f'病例: {name} | 任务: {task_name}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(save_root, f'{name}_task-{task_name}.png')
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f'[GradCAM] 已保存病例 {name} 的多模态可视化至: {save_path}')
        n_cases += 1

    gradcam.remove_hooks()
    print(f'[GradCAM] 可视化完成，共处理 {n_cases} 个病例。')


if __name__ == '__main__':
    """
    使用示例（在 HCP-Fusion Net 目录下）：

    python gradcam_vis.py \
        --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
        --modalities B VUSE VCEUS \
        --fusion_type late \
        --task_mode multi_cascaded \
        --use_clinical true \
        --backbone resnet50 \
        --exp_name exp_fullmodal_cascaded_clinical_bs16_e150

    默认从对应实验目录下加载 best_model.pth，
    在 val 集上选取若干病例，针对 Lateral 任务生成多模态 Grad-CAM 图。
    """
    opt_manager = Options()
    opt = opt_manager.parse()
    # 这里默认 task_idx=2（LateralLNM），max_cases=8
    visualize_cases(opt, task_idx=2, max_cases=8)

#     . 如果你想换任务 / 多画几例
# 换任务（例如想看 LNM 的注意力）：
# 在文件最后一行改 task_idx=0 即可：
# visualize_cases(opt, task_idx=0, max_cases=8)  # 0=LNM, 1=Central, 2=Lateral
# 多画病例：把 max_cases 改成你想要的数，比如 20。

