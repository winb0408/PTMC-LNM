# =============================================================================
# train.py
# 主训练脚本
#
# 使用示例（全三模态 + 级联多任务）：
#   python train.py \
#       --data_root /path/to/data \
#       --modalities B VUSE VCEUS \
#       --fusion_type late \
#       --task_mode multi_cascaded \
#       --use_clinical true \
#       --backbone resnet50 \
#       --epochs 100 \
#       --batch_size 16 \
#       --lr 1e-4 \
#       --penalty_weight 0.5 \
#       --exp_name exp_full_cascaded
#
# 消融实验示例（单模态，无级联）：
#   python train.py \
#       --modalities B \
#       --task_mode multi_parallel \
#       --use_clinical false \
#       --penalty_weight 0 \
#       --exp_name exp_ablation_single_modal
# =============================================================================

import os
import sys
import random
import time
import math
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# ── 项目内部模块（当前目录下的 Python 文件）
from options import Options
from dataset import build_dataloader, ClinicalCSVLoader
from networks import build_model
from loss import MultiTaskLoss
from metrics import MultiTaskMetrics, print_metrics, format_metrics


# =============================================================================
# 一、工具函数
# =============================================================================

def set_seed(seed: int):
    """全局随机种子固定，保证实验完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 以下两行会牺牲性能换取确定性，按需启用
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f'[Seed] 全局随机种子已设置为: {seed}')


def setup_logger(exp_dir: str, name: str = 'train') -> logging.Logger:
    """配置日志记录器（同时输出到控制台和文件）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 文件 Handler
    log_file = os.path.join(exp_dir, f'{name}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def build_optimizer(model: nn.Module, opt) -> torch.optim.Optimizer:
    """
    构建优化器（支持差分学习率）。
    骨干网络参数使用 lr_backbone，其余层参数使用 lr。
    """
    param_groups = model.get_parameter_groups(opt)

    if opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt.lr,  # 作为默认值，各 group 内会被覆盖
            weight_decay=opt.weight_decay,
        )
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=True,  # Nesterov 动量通常比标准 SGD 收敛更快
        )
    else:
        raise ValueError(f'未知的优化器: {opt.optimizer}')

    print(f'[Optimizer] 类型: {opt.optimizer} | LR: {opt.lr} | Backbone LR: {opt.lr_backbone}')
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, opt) -> Optional[object]:
    """构建学习率调度器"""
    if opt.scheduler == 'cosine':
        # 余弦退火：T_max 为总 epoch 数（扣除 warmup）
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=opt.epochs - opt.warmup_epochs,
            eta_min=opt.lr * 1e-3,  # 最低 LR = 初始 LR 的千分之一
        )
    elif opt.scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=opt.lr_step_size,
            gamma=opt.lr_gamma,
        )
    elif opt.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # 监控 AUC（越大越好）
            factor=0.5,
            patience=10,
            verbose=True,
        )
    else:
        scheduler = None

    print(f'[Scheduler] 策略: {opt.scheduler}')
    return scheduler


def warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int, base_lrs: list):
    """
    线性 Warmup 学习率调整。
    在前 warmup_epochs 个 epoch 内将 LR 从 0 线性增加到 base_lr。
    """
    alpha = (epoch + 1) / warmup_epochs
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = base_lrs[i] * alpha


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    opt,
):
    """保存模型检查点（包含完整训练状态，支持断点续训）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'opt': vars(opt),  # 保存完整实验配置
    }
    torch.save(checkpoint, save_path)
    print(f'[Checkpoint] 已保存至: {save_path}')


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    resume_path: str,
    device: torch.device,
) -> Tuple[int, Dict]:
    """加载检查点，返回 (起始 epoch, 最佳指标字典)"""
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f'找不到检查点文件: {resume_path}')

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_metrics = checkpoint.get('metrics', {})
    print(f'[Checkpoint] 从 epoch {checkpoint["epoch"]} 恢复训练')
    return start_epoch, best_metrics


# =============================================================================
# 二、单 Epoch 训练函数
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    metrics_tracker: MultiTaskMetrics,
    scaler: Optional[GradScaler],
    opt,
    epoch: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    执行单个 epoch 的训练。

    Args:
        model:            待训练模型
        loader:           训练 DataLoader
        optimizer:        优化器
        loss_fn:          多任务损失函数
        metrics_tracker:  指标追踪器（在 epoch 开始时已 reset）
        scaler:           AMP GradScaler（None 时不使用 AMP）
        opt:              实验配置
        epoch:            当前 epoch 编号（从 0 开始）
        logger:           日志记录器

    Returns:
        包含平均损失和各任务指标的字典
    """
    model.train()
    metrics_tracker.reset()

    # 损失累积（用于计算 epoch 平均损失）
    epoch_losses: Dict[str, float] = {}
    n_batches = len(loader)
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        # ── 将数据移至设备
        batch = _move_batch_to_device(batch, opt.device)

        # ── 前向传播（支持 AMP 混合精度）
        optimizer.zero_grad()

        if scaler is not None:
            # AMP 模式：在 autocast 上下文中执行前向传播
            with autocast():
                preds = model(batch)
                loss_dict = loss_fn(preds, batch['labels'])
                total_loss = loss_dict['loss_total']
        else:
            preds = model(batch)
            loss_dict = loss_fn(preds, batch['labels'])
            total_loss = loss_dict['loss_total']

        # ── 反向传播
        if scaler is not None:
            scaler.scale(total_loss).backward()
            # 梯度裁剪（在 unscale 之后才能正确裁剪）
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        # ── 累积损失
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        # ── 更新指标
        metrics_tracker.update(preds, batch['labels'])

        # ── 日志打印
        if (batch_idx + 1) % opt.log_interval == 0:
            elapsed = time.time() - t_start
            speed = (batch_idx + 1) / elapsed  # batches/sec
            loss_str = ' | '.join(
                f'{k}={v/(batch_idx+1):.4f}'
                for k, v in epoch_losses.items()
                if k != 'loss_total'
            )
            logger.info(
                f'Epoch [{epoch+1:03d}] '
                f'Batch [{batch_idx+1:04d}/{n_batches}] '
                f'({speed:.1f} batch/s) | '
                f'TotalLoss={epoch_losses.get("loss_total",0)/(batch_idx+1):.4f} | '
                f'{loss_str}'
            )

    # ── 计算 epoch 平均损失
    avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

    # ── 计算 epoch 指标
    epoch_metrics = metrics_tracker.compute()
    epoch_metrics.update(avg_losses)

    return epoch_metrics


# =============================================================================
# 三、验证函数
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: MultiTaskLoss,
    metrics_tracker: MultiTaskMetrics,
    opt,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    在验证集上评估模型性能。
    使用 @torch.no_grad() 禁用梯度计算以节省内存和加速。
    """
    model.eval()
    metrics_tracker.reset()

    epoch_losses: Dict[str, float] = {}
    n_batches = len(loader)

    for batch in loader:
        batch = _move_batch_to_device(batch, opt.device)

        # 验证阶段不需要 AMP（推理速度足够快且不需要节省显存）
        preds = model(batch)
        loss_dict = loss_fn(preds, batch['labels'])

        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        metrics_tracker.update(preds, batch['labels'])

    avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
    val_metrics = metrics_tracker.compute()
    val_metrics.update(avg_losses)

    return val_metrics


# =============================================================================
# 四、辅助工具函数
# =============================================================================

def _format_loss_items(metrics: Dict[str, float], prefix: str = '') -> str:
    """从 metrics 字典中提取并格式化 loss_* 项。"""
    loss_keys = [k for k in metrics.keys() if k.startswith('loss_')]
    if not loss_keys:
        return f'{prefix}(no loss items)'
    # 让 loss_total 排在最前面
    loss_keys = sorted(loss_keys, key=lambda k: (k != 'loss_total', k))
    return prefix + ' | '.join(f'{k}={metrics[k]:.4f}' for k in loss_keys)


def _move_batch_to_device(batch: Dict, device_str: str) -> Dict:
    """将 batch 中所有 Tensor 移至指定设备"""
    device = torch.device(device_str)
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            # Late Fusion 时 images 是字典
            moved[k] = {
                mod: t.to(device, non_blocking=True)
                for mod, t in v.items()
            }
        else:
            moved[k] = v  # 如 'name'（字符串列表）保持不变
    return moved


class EarlyStopper:
    """早停控制器，监控验证集主指标，连续 patience 个 epoch 未提升则停止"""

    def __init__(self, patience: int, mode: str = 'max'):
        """
        Args:
            patience: 最大容忍不提升的 epoch 数
            mode:     'max'（如 AUC，越大越好）或 'min'（如 Loss，越小越好）
        """
        self.patience = patience
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        更新最优值计数器。
        Returns: True 表示触发早停
        """
        if self.patience <= 0:
            return False

        improved = (
            (self.mode == 'max' and value > self.best_value)
            or (self.mode == 'min' and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# =============================================================================
# 五、主训练流程
# =============================================================================

def main():
    # ── Step 0: 解析参数
    opt_manager = Options()
    opt = opt_manager.parse()
    opt_manager.print_options()

    # ── Step 1: 初始化环境
    set_seed(opt.seed)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(opt.exp_dir)
    logger.info(f'使用设备: {device}')

    # ── Step 2: 构建数据流水线（train / val / test 三划分）
    logger.info('── 正在构建数据 DataLoader ──')

    # 训练集 DataLoader。
    # 若 use_clinical=True，build_dataloader 内部自动构建 ClinicalCSVLoader
    # 并将 feature_dim 写入 opt.clinical_input_dim，同时返回 loader 实例供复用。
    train_loader, clinical_loader = build_dataloader(
        opt=opt,
        split='train',
        clinical_loader=None,        # 首次调用，内部自动创建
        use_weighted_sampler=True,   # 处理类别不平衡
    )

    # 训练集“真实评估” DataLoader：不加权采样 + 关闭训练增强
    # 用于打印更可信的 Train 指标（避免 sampler/augmentation 造成 Train 虚高）
    train_eval_loader, _ = build_dataloader(
        opt=opt,
        split='train',
        clinical_loader=clinical_loader,
        use_weighted_sampler=False,
        eval_mode=True,
    )

    # 验证集 DataLoader（使用 groups/val.txt）
    val_loader, _ = build_dataloader(
        opt=opt,
        split='val',
        clinical_loader=clinical_loader,   # 复用已创建的 loader
        use_weighted_sampler=False,
    )

    # 测试集 DataLoader（使用 groups/test.txt，仅在训练结束后做最终评估）
    test_loader, _ = build_dataloader(
        opt=opt,
        split='test',
        clinical_loader=clinical_loader,
        use_weighted_sampler=False,
    )

    logger.info(
        f'训练集 batches: {len(train_loader)} | '
        f'训练集评估 batches: {len(train_eval_loader)} | '
        f'验证集 batches: {len(val_loader)} | '
        f'测试集 batches: {len(test_loader)}'
    )

    # ── Step 3: 构建模型
    # opt.clinical_input_dim 已由 build_dataloader 写入，直接读取
    logger.info('── 正在构建模型 ──')
    clinical_input_dim = getattr(opt, 'clinical_input_dim', 0)
    model = build_model(opt, clinical_input_dim=clinical_input_dim)
    model = model.to(device)

    # ── Step 4: 构建优化器、调度器、损失函数
    optimizer = build_optimizer(model, opt)
    scheduler = build_scheduler(optimizer, opt)
    loss_fn = MultiTaskLoss(opt).to(device)

    # AMP GradScaler（仅在 GPU + amp 模式下启用）
    scaler = GradScaler() if (opt.amp and device.type == 'cuda') else None
    if scaler:
        logger.info('[AMP] 自动混合精度已启用')

    # ── Step 5: 指标追踪器和早停控制器
    train_metrics_tracker = MultiTaskMetrics(opt)
    val_metrics_tracker = MultiTaskMetrics(opt)
    early_stopper = EarlyStopper(patience=opt.early_stop_patience, mode='max')

    # 记录 base LR（用于 Warmup 计算）
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    # ── Step 6: 断点恢复（可选）
    start_epoch = 0
    best_monitor_value = -float('inf')
    if opt.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, opt.resume, device)

    # ── Step 7: 主训练循环
    logger.info('═' * 60)
    logger.info(f' 开始训练 | 实验: {opt.exp_name} | 共 {opt.epochs} epoch')
    logger.info('═' * 60)

    for epoch in range(start_epoch, opt.epochs):
        epoch_start_time = time.time()

        # ── 7a. Warmup 学习率调整（前 N 个 epoch）
        if epoch < opt.warmup_epochs:
            warmup_lr(optimizer, epoch, opt.warmup_epochs, base_lrs)
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            logger.info(f'[Warmup] Epoch {epoch+1}/{opt.warmup_epochs} | LRs: {current_lrs}')

        # ── 7b. 骨干网络解冻控制（freeze_backbone_epochs）
        if epoch == 0 and opt.freeze_backbone_epochs > 0:
            model.freeze_backbones()
            logger.info(f'骨干网络已冻结，将在第 {opt.freeze_backbone_epochs} epoch 解冻')
        if epoch == opt.freeze_backbone_epochs and opt.freeze_backbone_epochs > 0:
            model.unfreeze_backbones()
            logger.info(f'第 {epoch+1} epoch: 骨干网络已解冻，开始端到端训练')

        # ── 7c. 训练一个 epoch（这里的 metrics 受 sampler/augmentation 影响，不用于最终展示）
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics_tracker=train_metrics_tracker,
            scaler=scaler,
            opt=opt,
            epoch=epoch,
            logger=logger,
        )

        # ── 7c2. 在“原始训练集（无 sampler、无增强）”上做一次评估，用于更可信的 Train 指标展示
        train_eval_metrics = validate(
            model=model,
            loader=train_eval_loader,
            loss_fn=loss_fn,
            metrics_tracker=train_metrics_tracker,  # 复用 tracker；validate 内部会 reset
            opt=opt,
            logger=logger,
        )

        # ── 7d. 验证
        val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            metrics_tracker=val_metrics_tracker,
            opt=opt,
            logger=logger,
        )

        epoch_time = time.time() - epoch_start_time

        # ── 7e. 日志记录（训练集和验证集指标）
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        logger.info(
            f'\nEpoch [{epoch+1:03d}/{opt.epochs}] '
            f'耗时: {epoch_time:.1f}s | '
            f'LR: {current_lrs}\n'
            f'── Train (training batches) ──\n{_format_loss_items(train_metrics, "  ")}\n'
            f'── Train (eval on original train set) ──\n{format_metrics(train_eval_metrics, "  ")}\n'
            f'── Val   ──\n{format_metrics(val_metrics, "  ")}'
        )

        # ── 7f. 学习率调度（Warmup 结束后）
        if epoch >= opt.warmup_epochs:
            if opt.scheduler == 'plateau':
                monitor_val = val_metrics.get(opt.monitor_metric, 0.0)
                scheduler.step(monitor_val)
            elif scheduler is not None:
                scheduler.step()

        # ── 7g. 模型保存
        monitor_value = val_metrics.get(opt.monitor_metric, 0.0)
        is_best = monitor_value > best_monitor_value

        if is_best:
            best_monitor_value = monitor_value
            best_ckpt_path = os.path.join(opt.exp_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_ckpt_path, opt)
            logger.info(
                f'★ 新最优模型 | {opt.monitor_metric}: {monitor_value:.4f} '
                f'(之前: {best_monitor_value:.4f})'
            )

        if not opt.save_best_only:
            # 每 epoch 都保存最新的检查点（支持完整断点续训）
            latest_ckpt_path = os.path.join(opt.exp_dir, 'latest_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, latest_ckpt_path, opt)

        # ── 7h. 早停判断
        if early_stopper.step(monitor_value):
            logger.info(
                f'[EarlyStop] 连续 {opt.early_stop_patience} 个 epoch 未提升，训练终止于 Epoch {epoch+1}。'
            )
            break

    # ── Step 8: 训练结束，加载最佳模型并做最终评估
    logger.info('\n' + '═' * 60)
    logger.info(' 训练完成！加载最佳模型进行最终评估')
    logger.info('═' * 60)

    best_ckpt_path = os.path.join(opt.exp_dir, 'best_model.pth')
    if os.path.exists(best_ckpt_path):
        # 重新加载最佳权重（不恢复优化器状态，仅用于推理）
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(
            f'最佳模型来自 Epoch {checkpoint["epoch"]+1} | '
            f'{opt.monitor_metric}: {best_monitor_value:.4f}'
        )

    # 最终在测试集上做完整评估
    final_metrics = validate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        metrics_tracker=val_metrics_tracker,
        opt=opt,
        logger=logger,
    )
    print_metrics(final_metrics, header='最终评估结果（测试集，最佳模型）')

    logger.info('训练流程完整结束。')


# =============================================================================
# 入口
# =============================================================================
if __name__ == '__main__':
    main()
