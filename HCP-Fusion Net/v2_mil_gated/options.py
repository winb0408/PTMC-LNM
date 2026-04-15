# =============================================================================
# v2_mil_gated/options.py
# 全局实验参数配置（Attention-MIL + Gated Fusion 版本）
#
# --model_mode 控制 6 组消融实验：
#   clinical_only        → Baseline A（仅临床）
#   slice_mean_concat    → Baseline B（切片级均值池化+拼接，失败反例）
#   mean_mil_gated       → Ablation 1（Mean-MIL + 门控，验证 Attention 必要性）
#   attention_mil_concat → Ablation 2（Attention-MIL + 拼接，验证门控必要性）
#   attention_mil_gated  → Ablation 3（α=0）或 Ours（α>0）
#
# --penalty_alpha 控制先验约束损失权重（设 0 = 消融约束项）
# =============================================================================

import argparse
import json
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    p = argparse.ArgumentParser(
        description='多模态 Attention-MIL + Gated Fusion 实验框架',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 路径 ──
    g = p.add_argument_group('Paths')
    g.add_argument('--data_root', type=str,
                   default='/data2/wjb/code/PTMC_LNM/data/MultiModalDataset')
    g.add_argument('--groups_root', type=str, default='groups_cv5_central',
                   help='相对 data_root 的 5-fold 目录')
    g.add_argument('--save_dir', type=str, default='./checkpoints_v2',
                   help='检查点与结果输出目录')
    g.add_argument('--exp_name', type=str, default='mil_gated',
                   help='实验名称（用作子目录）')

    # ── 消融开关 ──
    g = p.add_argument_group('Ablation')
    g.add_argument('--model_mode', type=str, default='attention_mil_gated',
                   choices=[
                       'clinical_only',
                       'slice_mean_concat',
                       'mean_mil_gated',
                       'attention_mil_concat',
                       'attention_mil_gated',
                   ],
                   help='模型模式，决定使用哪种融合策略')
    g.add_argument('--penalty_alpha', type=float, default=1.0,
                   help='先验约束损失 max(0, P_central - P_lnm) 的权重 α')

    # ── 数据 ──
    g = p.add_argument_group('Data')
    g.add_argument('--modalities', type=str, nargs='+',
                   default=['B', 'VUSE', 'VCEUS'],
                   choices=['B', 'VUSE', 'VCEUS'])
    g.add_argument('--img_size', type=int, default=224)
    g.add_argument('--max_instances', type=int, default=12,
                   help='每位患者 bag 中最大切片数（超出则随机采样）')
    g.add_argument('--clinical_agg', type=str, default='mean+max',
                   choices=['mean', 'max', 'mean+max'],
                   help='临床特征患者级聚合方式')
    g.add_argument('--use_radiomics', type=str2bool, default=False,
                   help='是否启用患者级 ROI radiomics 分支（三源融合）')
    g.add_argument('--radiomics_root', type=str, default='',
                   help='ROI radiomics 输出根目录；内部应包含 fold_0/.../patient_features_selected.csv')
    g.add_argument('--radiomics_topk', type=int, default=0,
                   help='仅使用前 K 维 radiomics 特征；<=0 表示使用全部')

    # ── 模型 ──
    g = p.add_argument_group('Model')
    g.add_argument('--backbone', type=str, default='resnet18',
                   choices=[
                       'resnet18','resnet34','resnet50',
                       'efficientnet_b0','efficientnet_b2',
                       'convnext_tiny','vit_b_16','swin_t',
                       'mobilevit_s','mobilevit_xs',
                       'mamba_tiny'
                   ])
    g.add_argument('--pretrained', type=str2bool, default=True)
    g.add_argument('--freeze_backbone_epochs', type=int, default=3,
                   help='前 N 个 epoch 冻结 backbone 参数')
    g.add_argument('--hidden_dim', type=int, default=256,
                   help='Attention / Fusion / Head 统一隐藏维度 D')
    g.add_argument('--attn_dim', type=int, default=128,
                   help='Attention 网络内部维度')
    g.add_argument('--dropout', type=float, default=0.3)
    g.add_argument('--radiomics_fusion_mode', type=str, default='legacy',
                   choices=['legacy', 'align_proj', 'film'],
                   help='legacy=原拼接融合；align_proj=深度/组学对齐投影；film=组学调制深度特征')
    g.add_argument('--rad_proj_dim', type=int, default=128,
                   help='align_proj 共享隐空间维度')
    g.add_argument('--rad_align_fusion', type=str, default='add',
                   choices=['add', 'concat'],
                   help='align_proj 中 deep/rad 对齐后融合方式')
    g.add_argument('--film_hidden_dim', type=int, default=128,
                   help='FiLM 生成器隐藏层维度')
    g.add_argument('--film_gamma_scale', type=float, default=0.1,
                   help='FiLM 中 gamma 的残差缩放系数（gamma=1+s*tanh(.)）')

    # ── 损失 ──
    g = p.add_argument_group('Loss')
    g.add_argument('--lambda_lnm', type=float, default=1.0,
                   help='LNM 任务损失权重 λ₁')
    g.add_argument('--lambda_central', type=float, default=1.0,
                   help='CentralLNM 任务损失权重 λ₂')

    # ── 训练 ──
    g = p.add_argument_group('Training')
    g.add_argument('--folds', type=int, default=5)
    g.add_argument('--epochs', type=int, default=60)
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--lr_backbone', type=float, default=1e-5)
    g.add_argument('--weight_decay', type=float, default=1e-4)
    g.add_argument('--scheduler', type=str, default='cosine',
                   choices=['cosine', 'step', 'none'])
    g.add_argument('--warmup_epochs', type=int, default=3)
    g.add_argument('--grad_clip', type=float, default=1.0)
    g.add_argument('--early_stop_patience', type=int, default=15)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'])
    g.add_argument('--gpu_id', type=int, default=0,
                   help='CUDA 设备编号')
    g.add_argument('--num_workers', type=int, default=4)
    g.add_argument('--amp', type=str2bool, default=True,
                   help='混合精度训练')

    return p


def parse_args(args=None):
    """解析参数并执行后处理"""
    parser = get_parser()
    opt = parser.parse_args(args)

    # ── backbone 输出维度自动映射 ──
    _backbone_dim = {
        'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
        'efficientnet_b0': 1280, 'efficientnet_b2': 1408,
        'convnext_tiny': 768, 'vit_b_16': 768, 'swin_t': 768,
        'mobilevit_s': 640, 'mobilevit_xs': 512, 'mamba_tiny': 768
    }
    opt.backbone_out_dim = _backbone_dim.get(opt.backbone, 768)

    # ── 临床特征维度（45 维原始；mean+max 时翻倍）──
    opt.clinical_raw_dim = 45
    if opt.clinical_agg == 'mean+max':
        opt.clinical_input_dim = opt.clinical_raw_dim * 2
    else:
        opt.clinical_input_dim = opt.clinical_raw_dim
    opt.radiomics_input_dim = 0

    # ── 输出目录 ──
    opt.exp_dir = os.path.join(opt.save_dir, opt.exp_name)
    os.makedirs(opt.exp_dir, exist_ok=True)

    # ── 保存配置 ──
    cfg_path = os.path.join(opt.exp_dir, 'config.json')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(vars(opt), f, indent=2, ensure_ascii=False)

    return opt


if __name__ == '__main__':
    opt = parse_args()
    for k, v in sorted(vars(opt).items()):
        print(f'  {k:<30s}: {v}')
