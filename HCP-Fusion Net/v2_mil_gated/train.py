#!/usr/bin/env python3
# =============================================================================
# v2_mil_gated/train.py
# 主训练脚本：5-fold 交叉验证 + 完整评估 + 可视化
#
# 使用方式：
#   python train.py --model_mode attention_mil_gated --penalty_alpha 1.0
#   python train.py --model_mode clinical_only
#   python train.py --model_mode attention_mil_gated --penalty_alpha 0  # Ablation3
# =============================================================================

import os
import sys
import json
import time
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
)

# ── 内部模块 ──
from options import parse_args
from dataset import build_loader
from networks import build_model
from loss import DualTaskLoss


# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, p))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """计算完整评估指标"""
    y_pred = (y_prob >= threshold).astype(int)
    auc = safe_auc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUC': auc, 'ACC': acc, 'F1': f1,
        'Precision': prec, 'Recall': rec,
        'Sensitivity': sen, 'Specificity': spe,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


def compute_logical_compliance(
    p_central: np.ndarray,
    p_lnm: np.ndarray,
) -> float:
    """
    逻辑依从率：预测中满足 P_central ≤ P_lnm 的样本占比。
    """
    n = len(p_central)
    if n == 0:
        return 0.0
    compliant = np.sum(p_central <= p_lnm + 1e-6)
    return float(compliant / n)


# ═════════════════════════════════════════════════════════════════════════════
# 评估函数
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict:
    """
    在 loader 上评估模型，返回 CentralLNM 和 LNM 的指标及原始概率。
    """
    model.eval()
    all_y_lnm, all_p_lnm = [], []
    all_y_cen, all_p_cen = [], []
    all_pids = []

    for batch in loader:
        # 移至 GPU
        batch_gpu = {
            'bag': batch['bag'].to(device),
            'mask': batch['mask'].to(device),
            'clinical': batch['clinical'].to(device),
            'radiomics': batch['radiomics'].to(device),
        }

        with autocast(enabled=use_amp):
            out = model(batch_gpu)

        p_lnm = torch.sigmoid(out['logit_lnm']).cpu().numpy()
        p_cen = torch.sigmoid(out['logit_central']).cpu().numpy()
        y_lnm = batch['y_lnm'].numpy()
        y_cen = batch['y_central'].numpy()

        all_p_lnm.extend(p_lnm.tolist())
        all_p_cen.extend(p_cen.tolist())
        all_y_lnm.extend(y_lnm.tolist())
        all_y_cen.extend(y_cen.tolist())
        all_pids.extend(batch['pid'])

    y_lnm = np.array(all_y_lnm)
    p_lnm = np.array(all_p_lnm)
    y_cen = np.array(all_y_cen)
    p_cen = np.array(all_p_cen)

    m_central = compute_metrics(y_cen, p_cen)
    m_lnm = compute_metrics(y_lnm, p_lnm)
    compliance = compute_logical_compliance(p_cen, p_lnm)

    return {
        'central': m_central,
        'lnm': m_lnm,
        'compliance': compliance,
        'y_central': y_cen, 'p_central': p_cen,
        'y_lnm': y_lnm, 'p_lnm': p_lnm,
        'pids': all_pids,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 单折训练
# ═════════════════════════════════════════════════════════════════════════════

def train_one_fold(
    opt,
    fold: int,
    device: torch.device,
) -> Dict:
    """训练单折并返回该折的测试集评估结果"""
    print(f'\n{"="*70}')
    print(f'  Fold {fold}  |  mode={opt.model_mode}  α={opt.penalty_alpha}')
    print(f'{"="*70}')

    groups_dir = os.path.join(opt.groups_root, f'fold_{fold}')

    # ── DataLoader ──
    train_loader = build_loader(opt.data_root, groups_dir, 'train', opt, is_train=True)
    val_loader   = build_loader(opt.data_root, groups_dir, 'val',   opt, is_train=False)
    test_loader  = build_loader(opt.data_root, groups_dir, 'test',  opt, is_train=False)
    opt.radiomics_input_dim = getattr(train_loader.dataset, 'radiomics_dim', 0)

    # ── 模型 ──
    model = build_model(opt).to(device)

    # ── 冻结 backbone（前几个 epoch）──
    if opt.freeze_backbone_epochs > 0 and model.need_image:
        model.freeze_backbone()

    # ── 优化器 ──
    param_groups = model.get_parameter_groups(opt)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )

    # ── 学习率调度器 ──
    if opt.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs, eta_min=1e-7,
        )
    elif opt.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.1,
        )
    else:
        scheduler = None

    # ── 损失函数 ──
    criterion = DualTaskLoss(
        lambda_lnm=opt.lambda_lnm,
        lambda_central=opt.lambda_central,
        penalty_alpha=opt.penalty_alpha,
    )

    # ── AMP ──
    scaler = GradScaler(enabled=opt.amp)

    # ── 训练循环 ──
    best_val_auc = -1.0
    best_state = None
    patience_counter = 0
    history: List[Dict] = []

    for epoch in range(opt.epochs):
        # 解冻 backbone
        if epoch == opt.freeze_backbone_epochs and model.need_image:
            model.unfreeze_backbone()
            # 更新优化器参数组
            param_groups = model.get_parameter_groups(opt)
            optimizer = torch.optim.AdamW(
                param_groups, lr=opt.lr, weight_decay=opt.weight_decay,
            )
            if opt.scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=opt.epochs - epoch, eta_min=1e-7,
                )

        model.train()
        epoch_losses = defaultdict(list)

        for batch in train_loader:
            batch_gpu = {
                'bag': batch['bag'].to(device),
                'mask': batch['mask'].to(device),
                'clinical': batch['clinical'].to(device),
                'radiomics': batch['radiomics'].to(device),
            }
            y_lnm = batch['y_lnm'].to(device)
            y_cen = batch['y_central'].to(device)

            optimizer.zero_grad()

            with autocast(enabled=opt.amp):
                out = model(batch_gpu)
                loss_dict = criterion(
                    out['logit_lnm'], out['logit_central'],
                    y_lnm, y_cen,
                )

            scaler.scale(loss_dict['loss_total']).backward()

            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())

        if scheduler is not None:
            scheduler.step()

        # ── Validation ──
        val_result = evaluate(model, val_loader, device, opt.amp)
        val_auc_central = val_result['central']['AUC']

        # ── 日志 ──
        loss_str = ' '.join(
            f'{k}={np.mean(v):.4f}' for k, v in epoch_losses.items()
        )
        print(
            f'[F{fold}] Ep {epoch+1:03d}/{opt.epochs} '
            f'{loss_str} | '
            f'val_AUC_cen={val_auc_central:.4f} '
            f'val_AUC_lnm={val_result["lnm"]["AUC"]:.4f} '
            f'comply={val_result["compliance"]:.3f}'
        )

        history.append({
            'epoch': epoch + 1,
            'val_auc_central': val_auc_central,
            'val_auc_lnm': val_result['lnm']['AUC'],
            'val_compliance': val_result['compliance'],
            **{f'train_{k}': np.mean(v) for k, v in epoch_losses.items()},
        })

        # ── Early stopping / best model ──
        if val_auc_central == val_auc_central and val_auc_central > best_val_auc:
            best_val_auc = val_auc_central
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if opt.early_stop_patience > 0 and patience_counter >= opt.early_stop_patience:
                print(f'[F{fold}] Early stop at epoch {epoch+1}')
                break

    # ── 加载最优模型 & 测试 ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    test_result = evaluate(model, test_loader, device, opt.amp)
    val_final = evaluate(model, val_loader, device, opt.amp)

    # ── 保存折结果 ──
    fold_dir = os.path.join(opt.exp_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # 保存模型
    torch.save(best_state or model.state_dict(),
               os.path.join(fold_dir, 'best_model.pth'))

    # 保存预测概率（用于后续绘制 ROC）
    np.savez(
        os.path.join(fold_dir, 'predictions.npz'),
        y_central=test_result['y_central'],
        p_central=test_result['p_central'],
        y_lnm=test_result['y_lnm'],
        p_lnm=test_result['p_lnm'],
        pids=test_result['pids'],
    )

    # 保存指标
    fold_metrics = {
        'fold': fold,
        'best_val_auc_central': best_val_auc,
        'test': {
            'central': test_result['central'],
            'lnm': test_result['lnm'],
            'compliance': test_result['compliance'],
        },
        'val': {
            'central': val_final['central'],
            'lnm': val_final['lnm'],
            'compliance': val_final['compliance'],
        },
    }
    with open(os.path.join(fold_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_metrics, f, indent=2, ensure_ascii=False)

    # 保存训练历史
    with open(os.path.join(fold_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return fold_metrics


# ═════════════════════════════════════════════════════════════════════════════
# 可视化
# ═════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(opt, all_fold_data: List[Dict]):
    """为 CentralLNM 绘制 5-fold ROC 叠加图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARNING] matplotlib 不可用，跳过 ROC 绘图')
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fd in all_fold_data:
        fold = fd['fold']
        pred_path = os.path.join(opt.exp_dir, f'fold_{fold}', 'predictions.npz')
        data = np.load(pred_path, allow_pickle=True)
        y = data['y_central']
        p = data['p_central']
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = roc_auc_score(y, p)
        ax.plot(fpr, tpr, alpha=0.4, label=f'Fold {fold} (AUC={auc_val:.3f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean([safe_auc(
            np.load(os.path.join(opt.exp_dir, f'fold_{d["fold"]}', 'predictions.npz'))['y_central'],
            np.load(os.path.join(opt.exp_dir, f'fold_{d["fold"]}', 'predictions.npz'))['p_central'],
        ) for d in all_fold_data])
        ax.plot(mean_fpr, mean_tpr, 'b-', linewidth=2,
                label=f'Mean (AUC={mean_auc:.3f})')
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.15, color='blue')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'CentralLNM ROC ({opt.model_mode})')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(opt.exp_dir, 'roc_central.png'), dpi=200)
    plt.close(fig)
    print(f'[Plot] ROC saved: {opt.exp_dir}/roc_central.png')


def plot_auc_boxplot(opt, summary: Dict):
    """绘制 CentralLNM 5-fold AUC 箱线图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    aucs = summary.get('fold_auc_central', [])
    if not aucs:
        return

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(aucs, labels=[opt.model_mode])
    for i, v in enumerate(aucs):
        ax.plot(1, v, 'ro', alpha=0.5)
    ax.set_ylabel('AUC (CentralLNM)')
    ax.set_title(f'{opt.model_mode} 5-fold AUC')
    fig.tight_layout()
    fig.savefig(os.path.join(opt.exp_dir, 'auc_boxplot_central.png'), dpi=200)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

def main():
    opt = parse_args()
    set_seed(opt.seed)

    # 对 Transformer 系列在本环境下关闭 AMP，避免 dtype 不匹配错误
    if any(k in opt.backbone for k in ['vit', 'swin', 'mamba']):
        opt.amp = False

    # ── 设备 ──
    if opt.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu_id}')
        torch.cuda.set_device(device)
        print(f'[Device] {torch.cuda.get_device_name(device)}')
    else:
        device = torch.device('cpu')
        opt.amp = False
        print('[Device] CPU')

    # ── 5-fold 训练 ──
    all_fold_results: List[Dict] = []
    t0 = time.time()

    for fold in range(opt.folds):
        fold_result = train_one_fold(opt, fold, device)
        all_fold_results.append(fold_result)

    elapsed = time.time() - t0

    # ── 汇总统计 ──
    test_auc_central = [r['test']['central']['AUC'] for r in all_fold_results]
    test_auc_lnm = [r['test']['lnm']['AUC'] for r in all_fold_results]
    test_acc_central = [r['test']['central']['ACC'] for r in all_fold_results]
    test_f1_central = [r['test']['central']['F1'] for r in all_fold_results]
    test_sen_central = [r['test']['central']['Sensitivity'] for r in all_fold_results]
    test_spe_central = [r['test']['central']['Specificity'] for r in all_fold_results]
    test_compliance = [r['test']['compliance'] for r in all_fold_results]

    summary = {
        'model_mode': opt.model_mode,
        'penalty_alpha': opt.penalty_alpha,
        'backbone': opt.backbone,
        'folds': opt.folds,
        'epochs': opt.epochs,
        'elapsed_sec': elapsed,
        'fold_auc_central': test_auc_central,
        'fold_auc_lnm': test_auc_lnm,
        'CentralLNM': {
            'AUC_mean': float(np.nanmean(test_auc_central)),
            'AUC_std': float(np.nanstd(test_auc_central)),
            'ACC_mean': float(np.nanmean(test_acc_central)),
            'F1_mean': float(np.nanmean(test_f1_central)),
            'Sen_mean': float(np.nanmean(test_sen_central)),
            'Spe_mean': float(np.nanmean(test_spe_central)),
        },
        'LNM': {
            'AUC_mean': float(np.nanmean(test_auc_lnm)),
            'AUC_std': float(np.nanstd(test_auc_lnm)),
        },
        'Compliance_mean': float(np.nanmean(test_compliance)),
        'Compliance_std': float(np.nanstd(test_compliance)),
    }

    # 保存汇总
    summary_path = os.path.join(opt.exp_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存 CSV（便于后续汇总对比）
    csv_path = os.path.join(opt.exp_dir, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'AUC_central', 'AUC_lnm', 'ACC', 'F1',
                     'Sen', 'Spe', 'Compliance'])
        for i, r in enumerate(all_fold_results):
            w.writerow([
                i,
                f"{r['test']['central']['AUC']:.4f}",
                f"{r['test']['lnm']['AUC']:.4f}",
                f"{r['test']['central']['ACC']:.4f}",
                f"{r['test']['central']['F1']:.4f}",
                f"{r['test']['central']['Sensitivity']:.4f}",
                f"{r['test']['central']['Specificity']:.4f}",
                f"{r['test']['compliance']:.4f}",
            ])
        w.writerow([
            'mean±std',
            f"{np.nanmean(test_auc_central):.4f}±{np.nanstd(test_auc_central):.4f}",
            f"{np.nanmean(test_auc_lnm):.4f}±{np.nanstd(test_auc_lnm):.4f}",
            f"{np.nanmean(test_acc_central):.4f}±{np.nanstd(test_acc_central):.4f}",
            f"{np.nanmean(test_f1_central):.4f}±{np.nanstd(test_f1_central):.4f}",
            f"{np.nanmean(test_sen_central):.4f}±{np.nanstd(test_sen_central):.4f}",
            f"{np.nanmean(test_spe_central):.4f}±{np.nanstd(test_spe_central):.4f}",
            f"{np.nanmean(test_compliance):.4f}±{np.nanstd(test_compliance):.4f}",
        ])

    # ── 绘图 ──
    plot_roc_curves(opt, all_fold_results)
    plot_auc_boxplot(opt, summary)

    # ── 打印汇总 ──
    print(f'\n{"="*70}')
    print(f'  实验完成: {opt.model_mode}  (α={opt.penalty_alpha})')
    print(f'  耗时: {elapsed/60:.1f} min')
    print(f'  CentralLNM AUC: {summary["CentralLNM"]["AUC_mean"]:.4f} ± {summary["CentralLNM"]["AUC_std"]:.4f}')
    print(f'  LNM        AUC: {summary["LNM"]["AUC_mean"]:.4f} ± {summary["LNM"]["AUC_std"]:.4f}')
    print(f'  逻辑依从率:     {summary["Compliance_mean"]:.4f} ± {summary["Compliance_std"]:.4f}')
    print(f'  结果目录: {opt.exp_dir}')
    print(f'{"="*70}\n')

    return summary


if __name__ == '__main__':
    main()
