# =============================================================================
# options.py
# 全局实验参数配置模块（基于 argparse）
# 支持多模态消融、任务模式切换、损失函数权重调整等所有实验维度
# =============================================================================

import argparse
import os
import json


def str2bool(v):
    """将字符串布尔参数转换为 Python bool，兼容多种输入格式"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    """
    构建并返回 ArgumentParser 对象。
    将 parser 的创建与解析解耦，方便在 Jupyter 或其他脚本中直接导入使用。
    """
    parser = argparse.ArgumentParser(
        description='多模态级联多任务医学影像分类实验框架',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =========================================================================
    # 一、路径与环境配置
    # =========================================================================
    path_group = parser.add_argument_group('Path & Environment')

    path_group.add_argument(
        '--data_root', type=str, default='/data2/wjb/code/PTMC_LNM/data/MultiModalDataset',
        help='数据根目录，下含 B/, VUSE/, VCEUS/ 图像子目录及 groups/ 划分文件'
    )
    path_group.add_argument(
        '--groups_dir', type=str, default='groups',
        help='存放 train.txt / test.txt 的子目录名（相对于 data_root）'
    )
    path_group.add_argument(
        '--clinical_csv', type=str, default='clinical_features.csv',
        help='临床特征 CSV 文件路径（绝对路径或相对于 data_root 的路径）'
    )
    path_group.add_argument(
        '--label_flag_file', type=str, default='lnm_clinical_flag.txt',
        help='图像名与临床特征行号对齐的映射文件'
    )
    path_group.add_argument(
        '--label_dirs', type=str, nargs='+',
        default=['LNM', 'CentralLNM', 'LateralLNM'],
        help='三个标签子目录名，与 --tasks 一一对应'
    )
    path_group.add_argument(
        '--save_dir', type=str, default='./checkpoints',
        help='模型检查点和日志保存目录'
    )
    path_group.add_argument(
        '--exp_name', type=str, default='exp_default',
        help='实验名称，用于创建子目录以隔离不同实验结果'
    )
    path_group.add_argument(
        '--resume', type=str, default='',
        help='从指定路径恢复训练（传入 checkpoint.pth 路径）；空字符串表示从头训练'
    )
    path_group.add_argument(
        '--seed', type=int, default=42,
        help='全局随机种子，保证实验可复现性'
    )
    path_group.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader 的工作进程数'
    )
    path_group.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='训练设备'
    )

    # =========================================================================
    # 二、数据与增强配置
    # =========================================================================
    data_group = parser.add_argument_group('Data & Augmentation')

    data_group.add_argument(
        '--modalities', type=str, nargs='+',
        default=['B', 'VUSE', 'VCEUS'],
        choices=['B', 'VUSE', 'VCEUS'],
        help=(
            '【消融实验核心参数】参与训练的图像模态列表。'
            '例如：--modalities B 表示单模态，--modalities B VUSE 表示双模态，'
            '--modalities B VUSE VCEUS 表示全三模态。'
            'networks.py 将根据此参数动态构建对应数量的 Backbone 并调整融合层维度。'
        )
    )
    data_group.add_argument(
        '--tasks', type=str, nargs='+',
        default=['LNM', 'CentralLNM', 'LateralLNM'],
        help='参与训练的任务名称列表，与 --label_dirs 一一对应'
    )
    data_group.add_argument(
        '--img_size', type=int, default=224,
        help='图像缩放尺寸（正方形）'
    )
    data_group.add_argument(
        '--use_clinical', type=str2bool, default=True,
        help=(
            '【消融实验参数】是否引入临床 CSV 特征。'
            'False 时 MLP 分支将被禁用，融合层维度相应缩减。'
        )
    )
    data_group.add_argument(
        '--clinical_feature_cols', type=str, nargs='*', default=None,
        help=(
            '指定参与模型的临床特征列名白名单（空格分隔）。'
            '不传则使用 CSV 中所有数值列（自动排除非数值列）。'
            '示例：--clinical_feature_cols age tumor_size gender_0 gender_1'
        )
    )
    data_group.add_argument(
        '--clinical_drop_cols', type=str, nargs='*', default=None,
        help=(
            '从 CSV 中排除的列名黑名单（空格分隔）。'
            '常用于剔除 ID 列、冗余列等。'
            '与 --clinical_feature_cols 互斥，优先级低于白名单。'
        )
    )
    data_group.add_argument(
        '--aug_hflip_prob', type=float, default=0.5,
        help='同步水平翻转概率（三模态共享同一随机状态）'
    )
    data_group.add_argument(
        '--aug_vflip_prob', type=float, default=0.3,
        help='同步垂直翻转概率'
    )
    data_group.add_argument(
        '--aug_rotate_deg', type=float, default=15.0,
        help='同步随机旋转最大角度（度）'
    )
    data_group.add_argument(
        '--aug_brightness', type=float, default=0.2,
        help='亮度抖动幅度（0 表示不启用）'
    )
    data_group.add_argument(
        '--aug_contrast', type=float, default=0.2,
        help='对比度抖动幅度（0 表示不启用）'
    )
    data_group.add_argument(
        '--normalize_mean', type=float, nargs=3,
        default=[0.485, 0.456, 0.406],
        help='图像归一化均值（RGB），使用 ImageNet 预训练权重的默认值'
    )
    data_group.add_argument(
        '--normalize_std', type=float, nargs=3,
        default=[0.229, 0.224, 0.225],
        help='图像归一化标准差（RGB）'
    )

    # =========================================================================
    # 三、模型架构配置
    # =========================================================================
    model_group = parser.add_argument_group('Model Architecture')

    model_group.add_argument(
        '--backbone', type=str, default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b2'],
        help='图像特征提取骨干网络类型'
    )
    model_group.add_argument(
        '--pretrained', type=str2bool, default=True,
        help='是否使用 ImageNet 预训练权重初始化骨干网络'
    )
    model_group.add_argument(
        '--freeze_backbone_epochs', type=int, default=0,
        help='前 N 个 epoch 冻结骨干网络参数，仅训练分类头（迁移学习热身策略）。0 表示不冻结。'
    )
    model_group.add_argument(
        '--backbone_out_dim', type=int, default=2048,
        help='骨干网络输出特征维度（ResNet50 默认 2048）'
    )
    model_group.add_argument(
        '--fusion_type', type=str, default='late',
        choices=['early', 'late'],
        help=(
            '【消融实验参数】图像模态融合方式。\n'
            'early: 将所有模态图像在通道维度拼接后送入单个 CNN（适合模态数固定场景）。\n'
            'late:  每个模态独立通过一个 Backbone 提取特征，之后在特征空间拼接融合（默认，推荐）。'
        )
    )
    model_group.add_argument(
        '--fusion_hidden_dim', type=int, default=512,
        help='融合层（Fusion MLP）的中间隐藏维度'
    )
    model_group.add_argument(
        '--fusion_dropout', type=float, default=0.3,
        help='融合层 Dropout 比率'
    )
    model_group.add_argument(
        '--clinical_mlp_dims', type=int, nargs='+',
        default=[128, 64],
        help='临床特征 MLP 各隐藏层维度列表，如 --clinical_mlp_dims 256 128 64'
    )
    model_group.add_argument(
        '--clinical_mlp_dropout', type=float, default=0.3,
        help='临床特征 MLP Dropout 比率'
    )
    model_group.add_argument(
        '--task_mode', type=str, default='multi_cascaded',
        choices=['single', 'multi_parallel', 'multi_cascaded'],
        help=(
            '【消融实验核心参数】预测任务模式。\n'
            'single:           仅预测 LateralLNM（末级任务）。\n'
            'multi_parallel:   LNM / CentralLNM / LateralLNM 三任务平行独立预测。\n'
            'multi_cascaded:   级联预测：先预测 LNM 与 CentralLNM，\n'
            '                  将其 logit 特征融合后再预测 LateralLNM（推荐，符合临床逻辑）。'
        )
    )
    model_group.add_argument(
        '--cascade_hidden_dim', type=int, default=64,
        help='级联预测头中间隐藏层维度'
    )

    # =========================================================================
    # 四、损失函数配置
    # =========================================================================
    loss_group = parser.add_argument_group('Loss Function')

    loss_group.add_argument(
        '--loss_type', type=str, default='focal',
        choices=['bce', 'focal'],
        help='分类损失类型：bce（标准二元交叉熵）或 focal（Focal Loss，适合类别不平衡）'
    )
    loss_group.add_argument(
        '--focal_alpha', type=float, default=0.25,
        help='Focal Loss 的 alpha 参数（正样本权重）'
    )
    loss_group.add_argument(
        '--focal_gamma', type=float, default=2.0,
        help='Focal Loss 的 gamma 参数（难例聚焦程度）'
    )
    loss_group.add_argument(
        '--task_loss_weights', type=float, nargs='+',
        default=[1.0, 1.0, 1.0],
        help=(
            '各任务损失的加权系数，与 --tasks 顺序对应。'
            '例如：--task_loss_weights 1.0 0.8 1.2 表示 LateralLNM 任务损失权重更大。'
        )
    )
    loss_group.add_argument(
        '--use_penalty_loss', type=str2bool, default=True,
        help=(
            '是否启用临床先验约束惩罚损失 Conditional Penalty Loss。'
            '该损失约束 P(LateralLNM) <= P(LNM)，符合临床逻辑：'
            '侧方转移不应高于整体转移概率。'
        )
    )
    loss_group.add_argument(
        '--penalty_weight', type=float, default=0.5,
        help=(
            '【消融实验参数】临床先验约束损失系数 α。'
            'Loss_total += α * max(0, P_lateral - P_lnm)。'
            '设为 0 等效于禁用约束损失。'
        )
    )
    loss_group.add_argument(
        '--pos_weight', type=float, nargs='+', default=None,
        help=(
            '各任务正样本权重（处理类别不平衡），与 --tasks 顺序对应。'
            '不指定则不加权。仅在 --loss_type bce 时生效。'
        )
    )

    # =========================================================================
    # 五、训练超参数配置
    # =========================================================================
    train_group = parser.add_argument_group('Training Hyperparameters')

    train_group.add_argument(
        '--epochs', type=int, default=100,
        help='总训练轮数'
    )
    train_group.add_argument(
        '--batch_size', type=int, default=16,
        help='训练批次大小'
    )
    train_group.add_argument(
        '--optimizer', type=str, default='adamw',
        choices=['adam', 'adamw', 'sgd'],
        help='优化器类型'
    )
    train_group.add_argument(
        '--lr', type=float, default=1e-4,
        help='初始学习率'
    )
    train_group.add_argument(
        '--lr_backbone', type=float, default=1e-5,
        help='骨干网络的独立学习率（通常小于分类头学习率，实现差分学习率）'
    )
    train_group.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='L2 权重衰减系数'
    )
    train_group.add_argument(
        '--momentum', type=float, default=0.9,
        help='SGD 动量（仅在 --optimizer sgd 时生效）'
    )
    train_group.add_argument(
        '--scheduler', type=str, default='cosine',
        choices=['cosine', 'step', 'plateau', 'none'],
        help='学习率调度策略'
    )
    train_group.add_argument(
        '--lr_step_size', type=int, default=30,
        help='StepLR 的步长（epoch 数），仅 --scheduler step 时生效'
    )
    train_group.add_argument(
        '--lr_gamma', type=float, default=0.1,
        help='StepLR 的衰减因子，仅 --scheduler step 时生效'
    )
    train_group.add_argument(
        '--warmup_epochs', type=int, default=5,
        help='线性 Warmup 的 epoch 数（0 表示不使用 Warmup）'
    )
    train_group.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='梯度裁剪阈值（0 表示不裁剪）'
    )
    train_group.add_argument(
        '--early_stop_patience', type=int, default=20,
        help='早停耐心值（验证集主指标连续 N 个 epoch 未提升则停止）。0 表示不早停。'
    )
    train_group.add_argument(
        '--monitor_metric', type=str, default='AUC_LateralLNM',
        help='用于模型选择和早停的监控指标名称（需与 metrics.py 返回字典的 key 一致）'
    )
    train_group.add_argument(
        '--val_split', type=float, default=0.2,
        help='从训练集中划分验证集的比例（0 表示不划分验证集，直接用测试集评估）'
    )
    train_group.add_argument(
        '--log_interval', type=int, default=10,
        help='每隔多少个 batch 打印一次训练日志'
    )
    train_group.add_argument(
        '--save_best_only', type=str2bool, default=True,
        help='是否只保存验证集最佳模型（False 则每 epoch 都保存）'
    )
    train_group.add_argument(
        '--amp', type=str2bool, default=True,
        help='是否启用自动混合精度（AMP）训练，可显著节省显存并加速训练'
    )

    return parser


class Options:
    """
    Options 封装类。
    负责解析命令行参数、进行参数合法性校验、
    创建输出目录并持久化实验配置。
    """

    def __init__(self):
        self.parser = get_parser()
        self.opt = None

    def parse(self, args=None):
        """
        解析参数并执行后处理逻辑。
        args=None 时从 sys.argv 读取（命令行模式）；
        传入字符串列表可在脚本中使用（如测试、Jupyter）。
        """
        self.opt = self.parser.parse_args(args)
        self._validate()
        self._post_process()
        return self.opt

    def _validate(self):
        """参数合法性校验，在训练开始前尽早暴露配置错误"""
        opt = self.opt

        # 检查任务数量与权重数量是否匹配
        n_tasks = len(opt.tasks)
        if len(opt.task_loss_weights) != n_tasks:
            raise ValueError(
                f'--task_loss_weights 数量 ({len(opt.task_loss_weights)}) '
                f'与 --tasks 数量 ({n_tasks}) 不一致。'
            )
        if opt.pos_weight and len(opt.pos_weight) != n_tasks:
            raise ValueError(
                f'--pos_weight 数量 ({len(opt.pos_weight)}) '
                f'与 --tasks 数量 ({n_tasks}) 不一致。'
            )

        # 级联模式至少需要 3 个任务
        if opt.task_mode == 'multi_cascaded' and n_tasks < 3:
            raise ValueError(
                'task_mode=multi_cascaded 要求至少 3 个任务 (LNM, CentralLNM, LateralLNM)。'
            )

        # early fusion 下不能使用多于 3 模态（通道过多会影响预训练权重）
        if opt.fusion_type == 'early' and len(opt.modalities) > 3:
            raise ValueError(
                'fusion_type=early 时建议模态数 <= 3，否则需要重新设计首层卷积。'
            )

        # single 任务模式下，任务列表应只含最终任务
        if opt.task_mode == 'single' and n_tasks != 1:
            print(
                f'[WARNING] task_mode=single 但 --tasks 包含 {n_tasks} 个任务，'
                f'将只预测 tasks 列表中的最后一个任务: {opt.tasks[-1]}'
            )

    def _post_process(self):
        """参数后处理：路径补全、目录创建、参数派生"""
        opt = self.opt

        # 构建实验输出目录
        opt.exp_dir = os.path.join(opt.save_dir, opt.exp_name)
        os.makedirs(opt.exp_dir, exist_ok=True)

        # 如果 clinical_csv 是相对路径，则拼接 data_root
        if not os.path.isabs(opt.clinical_csv):
            opt.clinical_csv = os.path.join(opt.data_root, opt.clinical_csv)

        # 构建 groups 目录完整路径
        opt.groups_path = os.path.join(opt.data_root, opt.groups_dir)

        # 根据 backbone 自动推断 backbone_out_dim（避免用户手动填写出错）
        backbone_dim_map = {
            'resnet18': 512, 'resnet34': 512,
            'resnet50': 2048, 'resnet101': 2048,
            'efficientnet_b0': 1280, 'efficientnet_b2': 1408,
        }
        if opt.backbone in backbone_dim_map:
            opt.backbone_out_dim = backbone_dim_map[opt.backbone]

        # 持久化本次实验配置，方便复现
        self._save_config()

    def _save_config(self):
        """将所有参数以 JSON 格式保存到实验目录"""
        config_path = os.path.join(self.opt.exp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(self.opt), f, indent=2, ensure_ascii=False)
        print(f'[Options] 实验配置已保存至: {config_path}')

    def print_options(self):
        """格式化打印所有参数，便于核查实验配置"""
        opt = self.opt
        print('\n' + '=' * 60)
        print(' 实验参数总览')
        print('=' * 60)
        for k, v in sorted(vars(opt).items()):
            print(f'  {k:<30s}: {v}')
        print('=' * 60 + '\n')


# =============================================================================
# 单独运行此文件时，打印所有默认参数（用于快速核查配置）
# =============================================================================
if __name__ == '__main__':
    opt_manager = Options()
    opt = opt_manager.parse()
    opt_manager.print_options()
