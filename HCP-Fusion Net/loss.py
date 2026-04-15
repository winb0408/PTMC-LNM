# =============================================================================
# models/loss.py
# 自定义损失函数模块
#
# 包含：
#   1. BinaryFocalLoss  —— 处理类别不平衡的 Focal Loss（二分类版本）
#   2. ConditionalPenaltyLoss —— 基于临床先验约束的惩罚损失
#                              max(0, P_lateral - P_lnm)
#   3. MultiTaskLoss    —— 统一管理多任务损失的组合损失
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# =============================================================================
# 一、带权重调整的 Binary Focal Loss
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """
    二分类 Focal Loss（Lin et al., 2017: "Focal Loss for Dense Object Detection"）

    动机：标准 BCE Loss 中，大量易分类的负样本会主导梯度，
    导致模型难以聚焦于少数难分类的正样本。
    Focal Loss 通过引入 (1 - p_t)^γ 调制因子，动态降低易分类样本的损失权重。

    公式：
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    其中：
        p_t = p      if y=1
              1-p    if y=0
        α_t = alpha  if y=1
              1-alpha if y=0

    Args:
        alpha: 正样本权重因子（0~1），通常 < 0.5 以压制负样本主导
        gamma: 难例聚焦程度，γ=0 退化为标准 BCE，γ=2 为论文推荐值
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B] 原始 logit（未经 Sigmoid）
            targets: [B] 二值标签 {0, 1}（float 类型）

        Returns:
            标量损失（reduction='mean' 时）
        """
        # 数值稳定：使用 log_sigmoid 计算 BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # [B]

        # 计算 p_t（预测正确类别的概率）
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal 调制因子
        focal_weight = (1 - p_t) ** self.gamma

        # alpha 加权（正负样本不对称权重）
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss  # [B]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# 二、临床先验约束惩罚损失（Conditional Penalty Loss）
# =============================================================================

class ConditionalPenaltyLoss(nn.Module):
    """
    基于临床先验的条件惩罚损失。

    临床先验约束：
        侧方淋巴结转移（LateralLNM）是整体淋巴结转移（LNM）的子集。
        因此，从概率角度，任何样本的侧方转移概率不应超过其整体转移概率：
            P(LateralLNM) ≤ P(LNM)

    若模型预测违反此约束，施加惩罚：
        L_penalty = mean( max(0, P_lateral - P_lnm) )
                  = mean( ReLU(P_lateral - P_lnm) )

    该损失：
    1. 仅对违反约束的样本（P_lateral > P_lnm）产生梯度
    2. 同时推动 P_lateral 下降和 P_lnm 上升，从两端修正约束违反
    3. 配合 --penalty_weight（α）控制约束强度

    注：此为软约束（soft constraint），不强制 100% 满足，
        允许模型在特殊样本上偶尔违反，但从统计层面引导正确行为。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        lateral_logit: torch.Tensor,
        lnm_logit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lateral_logit: [B] LateralLNM 预测 logit
            lnm_logit:     [B] LNM 预测 logit

        Returns:
            penalty_loss: 标量，表示约束违反的程度
        """
        p_lateral = torch.sigmoid(lateral_logit)  # [B]
        p_lnm = torch.sigmoid(lnm_logit)           # [B]

        # 惩罚项：仅对 P_lateral > P_lnm 的样本（违反临床先验）计算损失
        violation = F.relu(p_lateral - p_lnm)  # max(0, P_lateral - P_lnm)

        return violation.mean()  # 标量


# =============================================================================
# 三、统一多任务损失管理器
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    多任务损失统一管理模块。

    功能：
    1. 根据 task_mode 和 opt 动态决定参与计算的任务
    2. 对每个任务单独计算 BCE/Focal 损失，并按 task_loss_weights 加权求和
    3. 可选：叠加临床先验约束惩罚损失

    任务名与预测字典 key 的对应关系：
        'LNM'        → preds['lnm']     → labels[:, 0]
        'CentralLNM' → preds['central'] → labels[:, 1]
        'LateralLNM' → preds['lateral'] → labels[:, 2]
    """

    # 任务名 → 预测字典 key 的映射
    TASK_NAME_TO_KEY = {
        'LNM': 'lnm',
        'CentralLNM': 'central',
        'LateralLNM': 'lateral',
    }

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.task_mode = opt.task_mode

        # 构建各任务的基础分类损失函数
        if opt.loss_type == 'focal':
            self.criterion = BinaryFocalLoss(
                alpha=opt.focal_alpha,
                gamma=opt.focal_gamma,
                reduction='mean',
            )
        else:
            # 标准 BCE，支持正样本权重
            # pos_weight 在各任务之间可能不同，延迟到 forward 中处理
            self.criterion = None  # 将在 forward 中动态创建（以处理逐任务 pos_weight）

        # 临床先验约束损失
        self.penalty_loss_fn = ConditionalPenaltyLoss() if opt.use_penalty_loss else None

        # 任务损失权重
        self.task_weights = opt.task_loss_weights
        self.penalty_weight = opt.penalty_weight

        # pos_weight（仅 BCE 模式）
        self.pos_weights = opt.pos_weight  # List[float] or None

        # 记录任务名和对应列索引（用于从 labels 中提取正确的列）
        self.task_key_to_col = {}
        for col_idx, task_name in enumerate(opt.tasks):
            key = self.TASK_NAME_TO_KEY.get(task_name, task_name.lower())
            self.task_key_to_col[key] = col_idx

    def _get_task_criterion(self, task_idx: int, device: torch.device):
        """为指定任务构建/返回损失函数（处理逐任务 pos_weight）"""
        if self.opt.loss_type == 'focal':
            return self.criterion
        else:
            # BCE with optional per-task pos_weight
            if self.pos_weights is not None:
                pos_w = torch.tensor(
                    [self.pos_weights[task_idx]], dtype=torch.float32, device=device
                )
                return nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='mean')
            else:
                return nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失及各分项损失。

        Args:
            preds:  模型预测字典，如 {'lnm': [B], 'central': [B], 'lateral': [B]}
            labels: [B, N_tasks] 多任务标签矩阵，值为 0/1/-1（-1 表示该样本此任务无标签）

        Returns:
            loss_dict: 包含各分项损失和总损失的字典，例如：
                {
                    'loss_lnm': Tensor,
                    'loss_central': Tensor,
                    'loss_lateral': Tensor,
                    'loss_penalty': Tensor,
                    'loss_total': Tensor,  ← 用于反向传播
                }
        """
        device = labels.device
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # ── 各任务分类损失
        task_weight_sum = sum(self.task_weights)  # 用于归一化

        for task_key, pred_logit in preds.items():
            # 查找该任务在 labels 中的列索引
            col_idx = self.task_key_to_col.get(task_key)
            if col_idx is None:
                continue

            # 获取该任务的任务权重索引（与 opt.tasks 的顺序对应）
            task_names_keys = [
                self.TASK_NAME_TO_KEY.get(t, t.lower())
                for t in self.opt.tasks
            ]
            task_idx = task_names_keys.index(task_key) if task_key in task_names_keys else 0
            task_weight = self.task_weights[task_idx]

            # 获取标签，过滤掉无效标签（-1）
            task_labels = labels[:, col_idx]          # [B]
            valid_mask = (task_labels >= 0)            # 有效样本掩码
            if valid_mask.sum() == 0:
                continue

            valid_logits = pred_logit[valid_mask]      # [N_valid]
            valid_labels = task_labels[valid_mask]     # [N_valid]

            # 计算损失
            criterion = self._get_task_criterion(task_idx, device)
            loss_i = criterion(valid_logits, valid_labels)

            loss_dict[f'loss_{task_key}'] = loss_i

            # 加权累加
            total_loss = total_loss + task_weight * loss_i

        # ── 临床先验约束惩罚损失（仅在 lnm 和 lateral 都有预测时生效）
        if (
            self.penalty_loss_fn is not None
            and self.penalty_weight > 0
            and 'lateral' in preds
            and 'lnm' in preds
        ):
            penalty = self.penalty_loss_fn(preds['lateral'], preds['lnm'])
            loss_dict['loss_penalty'] = penalty
            total_loss = total_loss + self.penalty_weight * penalty
        else:
            loss_dict['loss_penalty'] = torch.tensor(0.0, device=device)

        loss_dict['loss_total'] = total_loss
        return loss_dict


# =============================================================================
# 模块自测
# =============================================================================
if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from options import Options

    opt = Options().parse([
        '--task_mode', 'multi_cascaded',
        '--loss_type', 'focal',
        '--use_penalty_loss', 'true',
        '--penalty_weight', '0.5',
        '--exp_name', 'loss_test',
    ])

    loss_fn = MultiTaskLoss(opt)

    B = 8
    preds = {
        'lnm':     torch.randn(B),
        'central': torch.randn(B),
        'lateral': torch.randn(B),
    }
    labels = torch.randint(0, 2, (B, 3)).float()

    loss_dict = loss_fn(preds, labels)
    print('损失分项:')
    for k, v in loss_dict.items():
        print(f'  {k}: {v.item():.4f}')

    # 验证惩罚损失约束方向正确：当 P_lateral >> P_lnm 时惩罚更大
    print('\n── 约束验证 ──')
    bad_preds = {'lnm': torch.full((4,), -3.0), 'lateral': torch.full((4,), 3.0)}  # 严重违反
    good_preds = {'lnm': torch.full((4,), 3.0), 'lateral': torch.full((4,), -3.0)}  # 满足约束
    penalty_fn = ConditionalPenaltyLoss()
    bad_penalty = penalty_fn(bad_preds['lateral'], bad_preds['lnm'])
    good_penalty = penalty_fn(good_preds['lateral'], good_preds['lnm'])
    print(f'  违反约束时惩罚: {bad_penalty.item():.4f}')
    print(f'  满足约束时惩罚: {good_penalty.item():.4f}')
    assert bad_penalty.item() > good_penalty.item(), '惩罚方向错误！'
    print('  ✓ 惩罚方向正确')
