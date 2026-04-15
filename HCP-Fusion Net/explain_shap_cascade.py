"""
explain_shap_cascade.py

用于单病例的概率推理可视化：
1. 基于 SHAP，对“全部临床特征”做局部解释（针对 Lateral 任务），输出瀑布图。
2. 展示级联概率流向（LNM → Central → Lateral），输出简单的概率流向图。

说明：
- 为了避免破坏训练/测试流程，本脚本独立于 train/test，只复用 Options、build_dataloader 和已训练好的 best_model.pth。
- 需要额外安装 shap 和 plotly（可选）：
    pip install shap plotly kaleido
"""

import os
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt

from options import Options
from dataset import build_dataloader
from networks import build_model


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """与 train.py 中逻辑一致的递归搬运到指定设备。"""
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            moved[k] = {
                kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            moved[k] = v
    return moved


def explain_one_case_shap_clinical(
    opt,
    model,
    device: torch.device,
    max_background: int = 64,
    task_idx: int = 2,
):
    """
    使用 SHAP 对单个病例的“全部临床特征”进行局部解释（针对指定任务，默认 Lateral）。

    设计要点：
    - 图像部分保持该病例本身不变，只对临床特征做“加减扰动”；
    - SHAP 模型 f(X) 的输入 X 是临床原始特征矩阵（未经过 ClinicalMLP），
      输出是指定任务的概率 P(task | image_fixed, clinical=X)。
    """
    try:
        import shap  # type: ignore
    except ImportError:
        print('[SHAP] 未安装 shap 包，请先运行: pip install shap')
        return

    # 1. 构建 val/test DataLoader，batch_size=1，便于单病例处理
    val_loader, _ = build_dataloader(
        opt=opt,
        split='val',
        clinical_loader=None,
        use_weighted_sampler=False,
    )

    # 找到第一个 batch（一个病例）
    first_batch = None
    background_clinical: List[torch.Tensor] = []

    for i, batch in enumerate(val_loader):
        if first_batch is None:
            first_batch = batch
        # 收集背景临床特征（原始 clinical 向量）
        background_clinical.append(batch['clinical'][0].clone())
        if len(background_clinical) >= max_background:
            break

    if first_batch is None:
        print('[SHAP] 验证集为空，无法进行解释。')
        return

    # 2. 选定目标病例（使用第一个 batch 的第一个样本）
    target_batch_cpu = first_batch
    target_name = target_batch_cpu['name'][0] if isinstance(target_batch_cpu['name'], list) else str(
        target_batch_cpu['name']
    )
    target_clinical = target_batch_cpu['clinical'][0].clone()  # [D_clin]

    # 3. 构建 SHAP 模型包装：只改变 clinical，图像固定为该病例本身
    model.eval()

    # 先把完整 batch 搬到 device 上，用于探测临床特征维度（保持图像/临床 batch 维度一致）
    batch_dev_full = _move_batch_to_device(target_batch_cpu, device)

    # 通过 forward_with_features 看一下 clinical_feat 是否存在
    with torch.no_grad():
        tmp_out = model.forward_with_features(batch_dev_full)
    clinical_feat_dim = (
        0 if tmp_out['clinical_feat'] is None else tmp_out['clinical_feat'].shape[1]
    )
    print(f'[SHAP] 临床特征维度: {target_clinical.shape[0]} -> ClinicalMLP 输出维度: {clinical_feat_dim}')

    # 然后单独准备“固定图像”的 batch（仅单个样本），供 SHAP 过程中复用
    fixed_batch_dev = _move_batch_to_device(target_batch_cpu, device)
    fixed_images_full = fixed_batch_dev['images']  # dict 或 Tensor
    if isinstance(fixed_images_full, dict):
        fixed_images = {mod: img[0:1] for mod, img in fixed_images_full.items()}
    else:
        fixed_images = fixed_images_full[0:1]

    # 定义用于 SHAP 的预测函数
    task_key = (
        'lateral'
        if model.task_mode == 'single'
        else model.TASK_KEYS[task_idx]  # 'lnm' / 'central' / 'lateral'
    )

    def predict_proba_clinical(x_np: np.ndarray) -> np.ndarray:
        """
        x_np: [N, D_clin]，原始临床特征（与 DataLoader 输出一致）
        返回: [N]，指定任务概率
        """
        x = torch.from_numpy(x_np).float().to(device)

        # 构建 batch：复用固定的图像，多复制几份
        if isinstance(fixed_images, dict):
            images_rep = {
                mod: img.repeat(x.shape[0], 1, 1, 1)  # [1,C,H,W] -> [N,C,H,W]
                for mod, img in fixed_images.items()
            }
        else:
            images_rep = fixed_images.repeat(x.shape[0], 1, 1, 1)

        dummy_labels = torch.zeros(x.shape[0], len(opt.tasks), device=device)
        batch = {
            'images': images_rep,
            'clinical': x,
            'labels': dummy_labels,
        }

        with torch.no_grad():
            out = model.forward_with_features(batch)
        prob = torch.sigmoid(out['logits'][task_key]).cpu().numpy()
        return prob

    # 4. 准备 SHAP 背景和目标样本
    background_np = torch.stack(background_clinical, dim=0).numpy()  # [N_bg, D]
    target_np = target_clinical.unsqueeze(0).numpy()                # [1, D]

    explainer = shap.KernelExplainer(predict_proba_clinical, background_np)
    shap_values = explainer.shap_values(target_np, nsamples=min(256, 2 * background_np.shape[0]))

    # shap_values: [1, D]
    shap_vals_1d = np.array(shap_values)[0]
    base_value = explainer.expected_value

    feature_names = [f'f{i}' for i in range(target_np.shape[1])]

    # 5. 绘制瀑布图（全部临床特征），按贡献绝对值从大到小排序
    order = np.argsort(-np.abs(shap_vals_1d))
    shap_sorted = shap_vals_1d[order]
    feat_sorted = target_np[0][order]
    names_sorted = [feature_names[i] for i in order]

    # 只画前 K 个特征，避免太挤；其余合并为 "Others"
    K = min(20, len(names_sorted))
    shap_top = shap_sorted[:K]
    feat_top = feat_sorted[:K]
    names_top = names_sorted[:K]

    if len(names_sorted) > K:
        shap_rest = shap_sorted[K:]
        shap_top = np.concatenate([shap_top, [shap_rest.sum()]])
        feat_top = np.concatenate([feat_top, [0.0]])
        names_top = names_top + ['Others']

    # 瀑布图（手工实现一个简化版，避免对 shap 版本过于依赖）
    cum = base_value
    contrib_starts = [cum]
    contrib_ends = []
    for dv in shap_top:
        cum = cum + dv
        contrib_ends.append(cum)
        contrib_starts.append(cum)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(names_top))
    colors = ['red' if v > 0 else 'blue' for v in shap_top]
    ax.bar(x_pos, shap_top, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_top, rotation=45, ha='right')
    ax.set_ylabel('SHAP value (对概率的贡献)')
    ax.set_title(f'病例 {target_name} | 任务 {task_key} | 基线概率={base_value:.3f}')
    plt.tight_layout()

    save_dir = os.path.join(opt.exp_dir, 'shap')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'shap_clinical_case-{target_name}_task-{task_key}.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f'[SHAP] 已保存病例 {target_name} 的临床特征瀑布图至: {save_path}')


def visualize_cascade_flow_for_case(
    opt,
    model,
    device: torch.device,
):
    """
    对一个病例展示 multi_cascaded 模式下的概率流向：
    P(LNM), P(Central), P(Lateral)。

    为减小依赖，这里用简单的条形图和文字说明代替正式 Sankey 图，
    但结构清晰，便于医生理解。
    """
    # 选一个验证集中病例
    val_loader, _ = build_dataloader(
        opt=opt,
        split='val',
        clinical_loader=None,
        use_weighted_sampler=False,
    )
    try:
        batch_cpu = next(iter(val_loader))
    except StopIteration:
        print('[Cascade] 验证集为空，无法展示级联概率流向。')
        return

    name = batch_cpu['name'][0] if isinstance(batch_cpu['name'], list) else str(batch_cpu['name'])
    batch_dev = _move_batch_to_device(batch_cpu, device)

    model.eval()
    with torch.no_grad():
        out = model.forward_with_features(batch_dev)
    probs = out['probs']

    if model.task_mode == 'single':
        print('[Cascade] 当前为 single 模式，仅有 Lateral 任务，无级联结构。')
        return

    p_lnm = probs['lnm'][0].item()
    p_central = probs['central'][0].item()
    p_lateral = probs['lateral'][0].item()

    # 简单条形图展示三任务概率
    fig, ax = plt.subplots(figsize=(6, 4))
    tasks = ['LNM', 'CentralLNM', 'LateralLNM']
    vals = [p_lnm, p_central, p_lateral]
    ax.bar(tasks, vals, color=['C0', 'C1', 'C2'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title(f'病例 {name} 的级联概率输出')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    plt.tight_layout()

    save_dir = os.path.join(opt.exp_dir, 'cascade_flow')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cascade_probs_case-{name}.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(
        f'[Cascade] 病例 {name}: '
        f'P(LNM)={p_lnm:.3f}, P(Central)={p_central:.3f}, P(Lateral)={p_lateral:.3f} '
        f'已保存条形图至: {save_path}'
    )


def main():
    opt_manager = Options()
    opt = opt_manager.parse()

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f'[Explain] 使用设备: {device}')

    # 构建模型并加载 best checkpoint
    # 需要先构建一个 DataLoader 以写入 clinical_input_dim
    tmp_loader, clinical_loader = build_dataloader(
        opt=opt,
        split='train',
        clinical_loader=None,
        use_weighted_sampler=False,
    )
    clinical_input_dim = getattr(opt, 'clinical_input_dim', 0)

    model = build_model(opt, clinical_input_dim=clinical_input_dim)
    model = model.to(device)

    best_ckpt_path = os.path.join(opt.exp_dir, 'best_model.pth')
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f'找不到最佳模型权重: {best_ckpt_path}')

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[Explain] 已加载最佳模型，来自 Epoch {ckpt["epoch"]+1}')

    # 1) SHAP：临床特征贡献瀑布图
    explain_one_case_shap_clinical(
        opt=opt,
        model=model,
        device=device,
        max_background=64,
        task_idx=2,  # 重点解释 LateralLNM
    )

    # 2) 级联概率流向可视化
    visualize_cascade_flow_for_case(
        opt=opt,
        model=model,
        device=device,
    )


if __name__ == '__main__':
    main()

