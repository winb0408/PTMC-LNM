# =============================================================================
# test.py
# 测试与推理脚本
#
# 使用示例：
#   python test.py \
#       --data_root /path/to/data \
#       --resume checkpoints/exp_full_cascaded/best_model.pth \
#       --exp_name exp_full_cascaded_test \
#       --modalities B VUSE VCEUS \
#       --task_mode multi_cascaded
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import torch

from options import Options
from data.dataset import build_dataloader, ClinicalCSVLoader
from models.networks import build_model
from models.loss import MultiTaskLoss
from utils.metrics import MultiTaskMetrics, print_metrics


@torch.no_grad()
def run_test(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f'[Test] 使用设备: {device}')

    # ── 构建测试 DataLoader
    # ClinicalCSVLoader 直接读取已预处理的全量 CSV，无需 fit，
    # 传 None 让 build_dataloader 内部自动创建即可。
    test_loader, _ = build_dataloader(
        opt=opt,
        split='test',
        clinical_loader=None,        # 内部自动从 opt.clinical_csv 创建
        use_weighted_sampler=False,
    )

    # ── 构建模型
    # opt.clinical_input_dim 已由 build_dataloader 写入
    clinical_input_dim = getattr(opt, 'clinical_input_dim', 0)
    model = build_model(opt, clinical_input_dim=clinical_input_dim)
    model = model.to(device)

    # ── 加载检查点
    if not opt.resume:
        raise ValueError('测试模式下必须通过 --resume 指定检查点路径！')
    checkpoint = torch.load(opt.resume, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'[Test] 已加载检查点: {opt.resume}')
    print(f'       来自 Epoch: {checkpoint.get("epoch", "unknown")}')

    model.eval()
    loss_fn = MultiTaskLoss(opt).to(device)
    metrics_tracker = MultiTaskMetrics(opt)
    metrics_tracker.reset()

    # ── 推理循环（收集所有预测结果）
    all_results = []

    for batch in test_loader:
        # 移至设备
        moved_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved_batch[k] = v.to(device)
            elif isinstance(v, dict):
                moved_batch[k] = {mod: t.to(device) for mod, t in v.items()}
            else:
                moved_batch[k] = v

        preds = model(moved_batch)
        metrics_tracker.update(preds, moved_batch['labels'])

        # 记录逐样本结果
        names = moved_batch['name']
        labels = moved_batch['labels'].cpu().numpy()  # [B, N_tasks]

        for i, name in enumerate(names):
            result = {'name': name}
            for key, logit in preds.items():
                prob = torch.sigmoid(logit[i]).item()
                result[f'prob_{key}'] = round(prob, 4)
                result[f'pred_{key}'] = int(prob >= 0.5)
            # 记录真实标签
            for col_i, task_name in enumerate(opt.tasks):
                result[f'label_{task_name}'] = int(labels[i, col_i])
            all_results.append(result)

    # ── 计算并打印指标
    final_metrics = metrics_tracker.compute()
    print_metrics(final_metrics, header='测试集评估结果')

    # ── 保存逐样本预测结果到 CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(opt.exp_dir, 'test_predictions.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'[Test] 逐样本预测结果已保存至: {csv_path}')

    # ── 保存汇总指标到 JSON
    metrics_path = os.path.join(opt.exp_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f'[Test] 汇总指标已保存至: {metrics_path}')

    return final_metrics


if __name__ == '__main__':
    opt_manager = Options()
    opt = opt_manager.parse()
    run_test(opt)
