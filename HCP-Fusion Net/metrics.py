# =============================================================================
# utils/metrics.py
# 多任务评估指标计算模块
#
# 包含：
#   - 各任务的 AUC、Accuracy、Sensitivity、Specificity、F1
#   - 符合临床逻辑的约束满足率（Clinical Constraint Compliance Rate）
#   - 汇总函数和格式化打印
# =============================================================================

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve
)


# =============================================================================
# 一、单任务指标计算
# =============================================================================

def compute_task_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_name: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算单个任务的完整评估指标。

    Args:
        y_true:    真实标签数组，shape [N]，值为 {0, 1}
        y_score:   预测概率数组，shape [N]，值为 [0, 1]
        task_name: 任务名称（用于字典 key 命名）
        threshold: 二值化阈值（默认 0.5）

    Returns:
        包含以下 key 的字典（key 格式：{指标}_{任务名}）：
        AUC, Accuracy, Sensitivity(Recall), Specificity, F1, Threshold
    """
    prefix = task_name
    metrics = {}

    # ── 过滤无效标签
    valid_mask = (y_true >= 0)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0:
        return {f'{prefix}_{m}': float('nan') for m in ['AUC', 'Acc', 'Sens', 'Spec', 'F1']}

    # ── AUC（仅在两类都有样本时计算）
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float('nan')
    else:
        auc = float('nan')
    metrics[f'AUC_{prefix}'] = auc

    # ── 二值化预测
    y_pred = (y_score >= threshold).astype(int)

    # ── Accuracy
    metrics[f'Acc_{prefix}'] = accuracy_score(y_true, y_pred)

    # ── 混淆矩阵 → Sensitivity / Specificity
    if len(np.unique(y_true)) > 1:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-8)  # Recall / True Positive Rate
            specificity = tn / (tn + fp + 1e-8)  # True Negative Rate
        except Exception:
            sensitivity = float('nan')
            specificity = float('nan')
    else:
        # 单类情况
        sensitivity = float('nan')
        specificity = float('nan')
    metrics[f'Sens_{prefix}'] = sensitivity
    metrics[f'Spec_{prefix}'] = specificity

    # ── F1 Score
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        f1 = float('nan')
    metrics[f'F1_{prefix}'] = f1

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = 'youden',
) -> float:
    """
    在 ROC 曲线上找到最优分类阈值。

    Args:
        y_true:  真实标签
        y_score: 预测概率
        method:  'youden'（最大化 Sensitivity + Specificity - 1）
                 'f1'（最大化 F1）

    Returns:
        optimal_threshold: 最优阈值
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        return float(thresholds[optimal_idx])

    elif method == 'f1':
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.linspace(0.1, 0.9, 50):
            y_pred = (y_score >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        return best_thresh

    return 0.5


# =============================================================================
# 二、临床先验约束满足率
# =============================================================================

def compute_clinical_compliance(
    lateral_probs: np.ndarray,
    lnm_probs: np.ndarray,
) -> Dict[str, float]:
    """
    计算临床先验约束满足率。

    约束：P(LateralLNM) ≤ P(LNM) 对所有样本成立（临床逻辑：侧方转移 ⊆ 整体转移）

    Args:
        lateral_probs: [N] LateralLNM 预测概率
        lnm_probs:     [N] LNM 预测概率

    Returns:
        {
            'clinical_compliance_rate': 满足约束的样本比例（↑ 越好，临床可信度指标）
            'mean_violation':           违反约束时的平均超出量（↓ 越好）
            'max_violation':            最大约束违反量
        }
    """
    violations = np.maximum(0, lateral_probs - lnm_probs)  # max(0, P_lat - P_lnm)
    n_violated = (violations > 0).sum()
    n_total = len(violations)

    compliance_rate = 1.0 - (n_violated / n_total)
    mean_violation = violations[violations > 0].mean() if n_violated > 0 else 0.0
    max_violation = violations.max()

    return {
        'clinical_compliance_rate': float(compliance_rate),
        'mean_violation': float(mean_violation),
        'max_violation': float(max_violation),
    }


# =============================================================================
# 三、多任务汇总评估
# =============================================================================

class MultiTaskMetrics:
    """
    多任务评估指标累加器与计算器。

    使用方式（与 train.py 配合）：
        metrics = MultiTaskMetrics(opt)
        metrics.reset()
        for batch in loader:
            ...
            metrics.update(preds, labels)
        result = metrics.compute()
    """

    # 任务 key → 任务显示名的映射
    KEY_TO_DISPLAY = {
        'lnm': 'LNM',
        'central': 'CentralLNM',
        'lateral': 'LateralLNM',
    }

    def __init__(self, opt, threshold: float = 0.5):
        self.opt = opt
        self.threshold = threshold
        self.task_mode = opt.task_mode

        # 根据 task_mode 和 opt.tasks 确定需要评估的任务 key 列表
        if self.task_mode == 'single':
            # single 模式下，只评估 tasks 列表中的最后一个任务
            task_name_to_key = {'LNM': 'lnm', 'CentralLNM': 'central', 'LateralLNM': 'lateral'}
            if len(opt.tasks) >= 1:
                single_task_name = opt.tasks[-1]
                single_key = task_name_to_key.get(single_task_name, single_task_name.lower())
                self.active_keys = [single_key]
            else:
                self.active_keys = ['lateral']
        else:
            self.active_keys = ['lnm', 'central', 'lateral']

        self.reset()

    def reset(self):
        """清空累积的预测结果和标签，开始新的评估周期"""
        self._preds: Dict[str, List[float]] = {k: [] for k in self.active_keys}
        self._targets: Dict[str, List[float]] = {k: [] for k in self.active_keys}

        # 任务 key → labels 中的列索引
        task_name_to_key = {'LNM': 'lnm', 'CentralLNM': 'central', 'LateralLNM': 'lateral'}
        self._key_to_col = {}
        for col_i, task_name in enumerate(self.opt.tasks):
            key = task_name_to_key.get(task_name, task_name.lower())
            self._key_to_col[key] = col_i

    def update(
        self,
        preds: Dict[str, 'torch.Tensor'],
        labels: 'torch.Tensor',
    ):
        """
        将当前 batch 的预测结果和标签累积到内部缓冲区。

        Args:
            preds:  {'lnm': Tensor[B], 'central': Tensor[B], 'lateral': Tensor[B]}
                    （logit 或 prob 均可，内部统一转换为 prob）
            labels: Tensor[B, N_tasks]，值为 {0, 1, -1}（-1 表示无标签）
        """
        import torch

        for key in self.active_keys:
            if key not in preds:
                continue

            col_idx = self._key_to_col.get(key)
            if col_idx is None:
                continue

            # 将 logit 转换为概率
            logit = preds[key]
            prob = torch.sigmoid(logit).detach().cpu().numpy()

            # 获取对应列标签
            task_label = labels[:, col_idx].detach().cpu().numpy()

            # 逐样本添加（过滤无效标签）
            for p, t in zip(prob, task_label):
                if t >= 0:
                    self._preds[key].append(float(p))
                    self._targets[key].append(float(t))

    def compute(self) -> Dict[str, float]:
        """
        计算所有任务的完整评估指标。

        Returns:
            包含所有指标的字典，可直接用于 TensorBoard 记录或日志打印
        """
        all_metrics = {}

        # ── 各任务指标
        for key in self.active_keys:
            if not self._preds[key]:
                continue

            y_true = np.array(self._targets[key])
            y_score = np.array(self._preds[key])
            display_name = self.KEY_TO_DISPLAY.get(key, key)

            task_metrics = compute_task_metrics(y_true, y_score, display_name, self.threshold)
            all_metrics.update(task_metrics)

        # ── 临床约束满足率（需要同时有 lnm 和 lateral 的预测）
        if (
            'lateral' in self.active_keys and 'lnm' in self.active_keys
            and self._preds['lateral'] and self._preds['lnm']
        ):
            # 对齐两个任务的有效样本（取交集，以确保相同样本做对比）
            lat_probs = np.array(self._preds['lateral'])
            lnm_probs = np.array(self._preds['lnm'])

            # 只有当两者长度一致时才计算（通常如此，因为 multi_cascaded 同时预测所有任务）
            if len(lat_probs) == len(lnm_probs):
                compliance_metrics = compute_clinical_compliance(lat_probs, lnm_probs)
                all_metrics.update(compliance_metrics)

        return all_metrics

    def get_monitor_metric(self, metric_name: str) -> float:
        """获取指定监控指标的当前值（用于早停和模型选择）"""
        all_metrics = self.compute()
        return all_metrics.get(metric_name, 0.0)


# =============================================================================
# 四、格式化打印工具
# =============================================================================

def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """
    将指标字典格式化为可读的多行字符串。

    Args:
        metrics: 指标字典
        prefix:  每行前缀（如 '[Val]'）
    """
    lines = []
    # 按任务分组打印
    task_groups = {
        'LNM': ['AUC_LNM', 'Acc_LNM', 'Sens_LNM', 'Spec_LNM', 'F1_LNM'],
        'CentralLNM': ['AUC_CentralLNM', 'Acc_CentralLNM', 'Sens_CentralLNM', 'Spec_CentralLNM', 'F1_CentralLNM'],
        'LateralLNM': ['AUC_LateralLNM', 'Acc_LateralLNM', 'Sens_LateralLNM', 'Spec_LateralLNM', 'F1_LateralLNM'],
        'Clinical': ['clinical_compliance_rate', 'mean_violation', 'max_violation'],
    }

    for group_name, keys in task_groups.items():
        group_vals = [(k, metrics[k]) for k in keys if k in metrics]
        if not group_vals:
            continue
        vals_str = ' | '.join(
            f'{k.split("_")[0]}={v:.4f}' for k, v in group_vals
        )
        lines.append(f'{prefix}[{group_name}] {vals_str}')

    return '\n'.join(lines)


def print_metrics(metrics: Dict[str, float], header: str = 'Evaluation Results'):
    """打印评估结果"""
    print('\n' + '─' * 60)
    print(f' {header}')
    print('─' * 60)
    print(format_metrics(metrics, prefix='  '))

    # 单独打印临床约束指标
    compliance = metrics.get('clinical_compliance_rate')
    if compliance is not None:
        print(f'  [Clinical Constraint]')
        print(f'    Compliance Rate:  {compliance:.4f}  (↑ 临床先验满足率)')
        print(f'    Mean Violation:   {metrics.get("mean_violation", 0):.4f}  (↓ 违反幅度均值)')
        print(f'    Max Violation:    {metrics.get("max_violation", 0):.4f}  (↓ 最大违反量)')
    print('─' * 60 + '\n')


# =============================================================================
# 模块自测
# =============================================================================
if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from options import Options

    opt = Options().parse([
        '--task_mode', 'multi_cascaded',
        '--exp_name', 'metrics_test',
    ])

    # 模拟 100 个样本的预测
    B = 100
    np.random.seed(42)

    # 模拟：lnm >= lateral（临床约束基本满足）
    lnm_scores = np.random.beta(2, 2, B)
    lateral_scores = lnm_scores * np.random.uniform(0.5, 1.0, B)  # lateral ≤ lnm

    y_true_lnm = (lnm_scores > 0.5).astype(int)
    y_true_lat = (lateral_scores > 0.3).astype(int)

    metrics = MultiTaskMetrics(opt)
    metrics.reset()

    # 模拟 update
    for i in range(0, B, 10):
        batch_slice = slice(i, min(i+10, B))
        preds = {
            'lnm':     torch.from_numpy(np.log(lnm_scores[batch_slice] / (1 - lnm_scores[batch_slice] + 1e-6))),
            'central': torch.randn(10),
            'lateral': torch.from_numpy(np.log(lateral_scores[batch_slice] / (1 - lateral_scores[batch_slice] + 1e-6))),
        }
        labels_arr = np.stack([
            y_true_lnm[batch_slice],
            np.random.randint(0, 2, min(10, B-i)),
            y_true_lat[batch_slice],
        ], axis=1)
        labels = torch.from_numpy(labels_arr).float()
        metrics.update(preds, labels)

    result = metrics.compute()
    print_metrics(result, header='自测评估结果')
