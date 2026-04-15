# =============================================================================
# v2_mil_gated/loss.py
# 多任务损失 + 临床先验约束（CentralLNM ≤ LNM）
#
# L_total = λ₁ · BCE(P_lnm, Y_lnm)
#         + λ₂ · BCE(P_central, Y_central)
#         + α  · mean( max(0, P_central - P_lnm) )
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PriorConstraintLoss(nn.Module):
    """
    先验约束损失：P(CentralLNM) ≤ P(LNM)

    违反约束时施加惩罚：
        L_penalty = mean( ReLU(P_central - P_lnm) )

    临床含义：中央区转移是整体转移的子集，因此其预测概率不应超过整体转移概率。
    """

    def forward(
        self,
        logit_central: torch.Tensor,
        logit_lnm: torch.Tensor,
    ) -> torch.Tensor:
        p_central = torch.sigmoid(logit_central)
        p_lnm = torch.sigmoid(logit_lnm)
        violation = F.relu(p_central - p_lnm)
        return violation.mean()


class DualTaskLoss(nn.Module):
    """
    双任务 BCE + 先验约束的统一损失管理器。

    Args:
        lambda_lnm:     LNM 任务 BCE 权重 λ₁
        lambda_central: CentralLNM 任务 BCE 权重 λ₂
        penalty_alpha:  先验约束损失权重 α（0 = 消融约束项）
    """

    def __init__(
        self,
        lambda_lnm: float = 1.0,
        lambda_central: float = 1.0,
        penalty_alpha: float = 1.0,
        conditional_loss_alpha: float = 0.0,
        evidence_entropy_alpha: float = 0.0,
    ):
        super().__init__()
        self.lambda_lnm = lambda_lnm
        self.lambda_central = lambda_central
        self.penalty_alpha = penalty_alpha
        self.conditional_loss_alpha = conditional_loss_alpha
        self.evidence_entropy_alpha = evidence_entropy_alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.penalty_fn = PriorConstraintLoss()

    def forward(
        self,
        out_or_logit_lnm,
        logit_central_or_y_lnm=None,
        y_lnm: torch.Tensor = None,
        y_central: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'loss_lnm':     标量,
                'loss_central': 标量,
                'loss_penalty': 标量,
                'loss_total':   标量 (用于 backward),
            }
        """
        if isinstance(out_or_logit_lnm, dict):
            out = out_or_logit_lnm
            logit_lnm = out['logit_lnm']
            logit_central = out['logit_central']
            y_lnm_target = logit_central_or_y_lnm
            y_central_target = y_lnm
        else:
            out = {}
            logit_lnm = out_or_logit_lnm
            logit_central = logit_central_or_y_lnm
            y_lnm_target = y_lnm
            y_central_target = y_central

        loss_lnm = self.bce(logit_lnm, y_lnm_target)
        loss_central = self.bce(logit_central, y_central_target)

        total = self.lambda_lnm * loss_lnm + self.lambda_central * loss_central

        if self.conditional_loss_alpha > 0 and 'logit_central_cond' in out:
            pos_mask = (y_lnm_target > 0.5)
            if pos_mask.any():
                loss_cond = self.bce(out['logit_central_cond'][pos_mask], y_central_target[pos_mask])
                total = total + self.conditional_loss_alpha * loss_cond
            else:
                loss_cond = torch.tensor(0.0, device=logit_lnm.device)
        else:
            loss_cond = torch.tensor(0.0, device=logit_lnm.device)

        if self.penalty_alpha > 0:
            loss_pen = self.penalty_fn(logit_central, logit_lnm)
            total = total + self.penalty_alpha * loss_pen
        else:
            loss_pen = torch.tensor(0.0, device=logit_lnm.device)

        if self.evidence_entropy_alpha > 0 and out.get('evidence_weights', None) is not None:
            w = out['evidence_weights'].clamp(min=1e-8)
            entropy = -(w * torch.log(w)).sum(dim=1).mean()
            total = total - self.evidence_entropy_alpha * entropy
        else:
            entropy = torch.tensor(0.0, device=logit_lnm.device)

        return {
            'loss_lnm': loss_lnm,
            'loss_central': loss_central,
            'loss_conditional': loss_cond,
            'loss_penalty': loss_pen,
            'loss_evidence_entropy': entropy,
            'loss_total': total,
        }
