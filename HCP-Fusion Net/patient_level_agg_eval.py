"""
patient_level_agg_eval.py

对现有实验（尤其是 5-fold CV）做“患者级聚合评估”：
- 先在 test 集按图像/切面推理得到每张图的概率 p
- 再按患者ID聚合（患者ID = 文件名第一个下划线前的数字串）
- 输出三种聚合策略的 patient-level AUC：
    1) max
    2) mean
    3) top-k mean

用法示例（评估 5 折 CV，每折一个 exp_dir）：

CUDA_VISIBLE_DEVICES=2 /data2/wjb/anaconda3/envs/pumc/bin/python patient_level_agg_eval.py \
  --save_dir ./checkpoints \
  --exp_prefix cvCentral_r50 \
  --folds 5 \
  --device cuda \
  --topk 3

说明：
- 该脚本会自动从每折实验目录的 config.json 读取 data_root / groups_dir / 模型配置。
- 默认只取 best_model.pth 进行评估。
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import build_dataloader
from networks import build_model


def patient_id_from_name(name: str) -> str:
    return str(name).split("_", 1)[0]


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
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


@dataclass
class PatientAggResult:
    pid: str
    y: int
    p_max: float
    p_mean: float
    p_topk_mean: float
    n_images: int


def aggregate_patient_probs(probs: List[float], topk: int) -> Tuple[float, float, float]:
    arr = np.array(probs, dtype=np.float32)
    p_max = float(arr.max())
    p_mean = float(arr.mean())
    k = int(topk)
    k = max(1, k)
    k = min(k, arr.shape[0])
    topk_vals = np.sort(arr)[-k:]
    p_topk_mean = float(topk_vals.mean())
    return p_max, p_mean, p_topk_mean


def load_opt_from_config(exp_dir: str) -> SimpleNamespace:
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到 config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # train.py 里需要 opt.exp_dir；这里也补上
    d["exp_dir"] = exp_dir
    return SimpleNamespace(**d)


def infer_sample_probs(opt, model, device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Returns:
      names: 样本名列表（与 DataLoader 同顺序）
      y_true: [N] 0/1
      y_prob: [N] 0~1
    """
    # 用 eval_mode=True：关闭训练增强，避免评估不稳定
    test_loader, _ = build_dataloader(
        opt=opt,
        split="test",
        clinical_loader=None,
        use_weighted_sampler=False,
        eval_mode=True,
    )

    model.eval()
    all_names: List[str] = []
    all_y: List[float] = []
    all_p: List[float] = []

    # single 模式下，模型输出 key 可能是 lnm/central/lateral
    if getattr(model, "task_mode", "") == "single":
        task_key = getattr(model, "single_task_key", None) or "lateral"
    else:
        # 多任务时，这里默认 central；你也可以扩展做三任务
        task_key = "central"

    with torch.no_grad():
        for batch in test_loader:
            batch_dev = _move_batch_to_device(batch, device)
            preds = model(batch_dev)  # Dict[str, Tensor[B]]
            if task_key not in preds:
                # fallback：取第一个 key
                task_key = list(preds.keys())[0]
            logit = preds[task_key]
            prob = torch.sigmoid(logit).detach().cpu().numpy()

            # labels: [B, 1]（Central 单任务）
            y = batch["labels"][:, 0].detach().cpu().numpy()

            names = batch["name"]
            if isinstance(names, list):
                all_names.extend(names)
            else:
                all_names.extend(list(names))
            all_y.extend([float(v) for v in y])
            all_p.extend([float(v) for v in prob])

    return all_names, np.array(all_y, dtype=np.float32), np.array(all_p, dtype=np.float32)


def patient_level_eval_for_exp(exp_dir: str, device_str: str, topk: int) -> Dict:
    opt = load_opt_from_config(exp_dir)

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    # 构建 DataLoader 以写入 clinical_input_dim（由 build_dataloader 内部写入 opt）
    _train_loader, clinical_loader = build_dataloader(
        opt=opt,
        split="train",
        clinical_loader=None,
        use_weighted_sampler=False,
        eval_mode=True,
    )
    clinical_input_dim = getattr(opt, "clinical_input_dim", 0)

    model = build_model(opt, clinical_input_dim=clinical_input_dim).to(device)
    best_path = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"找不到 best_model.pth: {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    names, y_true, y_prob = infer_sample_probs(opt, model, device)

    # 按患者聚合
    pid_to_probs: Dict[str, List[float]] = defaultdict(list)
    pid_to_y: Dict[str, List[int]] = defaultdict(list)

    for n, y, p in zip(names, y_true, y_prob):
        pid = patient_id_from_name(n)
        pid_to_probs[pid].append(float(p))
        pid_to_y[pid].append(int(y))

    patient_rows: List[PatientAggResult] = []
    for pid in sorted(pid_to_probs.keys()):
        ys = pid_to_y[pid]
        # 患者级 y：只要该患者任一切面为 1，则认为患者为 1（与你 CV 脚本一致）
        y_pid = 1 if max(ys) == 1 else 0
        p_max, p_mean, p_topk = aggregate_patient_probs(pid_to_probs[pid], topk=topk)
        patient_rows.append(
            PatientAggResult(
                pid=pid,
                y=y_pid,
                p_max=p_max,
                p_mean=p_mean,
                p_topk_mean=p_topk,
                n_images=len(pid_to_probs[pid]),
            )
        )

    y_pat = np.array([r.y for r in patient_rows], dtype=np.int32)
    p_max = np.array([r.p_max for r in patient_rows], dtype=np.float32)
    p_mean = np.array([r.p_mean for r in patient_rows], dtype=np.float32)
    p_topk = np.array([r.p_topk_mean for r in patient_rows], dtype=np.float32)

    def safe_auc(y, p) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))

    auc_max = safe_auc(y_pat, p_max)
    auc_mean = safe_auc(y_pat, p_mean)
    auc_topk = safe_auc(y_pat, p_topk)

    return {
        "exp_dir": exp_dir,
        "n_patients": int(len(patient_rows)),
        "n_images": int(len(names)),
        "topk": int(topk),
        "auc_patient_max": auc_max,
        "auc_patient_mean": auc_mean,
        "auc_patient_topk_mean": auc_topk,
    }


def main():
    ap = argparse.ArgumentParser("Patient-level aggregation evaluation (max/mean/topk-mean)")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--exp_prefix", type=str, required=True, help="例如 cvCentral_r50")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    results: List[Dict] = []
    for i in range(args.folds):
        exp_dir = os.path.join(args.save_dir, f"{args.exp_prefix}_fold{i}")
        print("\n" + "=" * 80)
        print(f"[PatientAgg] evaluating fold {i}: {exp_dir}")
        print("=" * 80)
        r = patient_level_eval_for_exp(exp_dir, device_str=args.device, topk=args.topk)
        results.append(r)
        print(
            f"[PatientAgg] fold{i} patients={r['n_patients']} images={r['n_images']} | "
            f"AUC(max)={r['auc_patient_max']:.4f} AUC(mean)={r['auc_patient_mean']:.4f} "
            f"AUC(top{args.topk})={r['auc_patient_topk_mean']:.4f}"
        )

    # 汇总输出
    out_json = os.path.join(args.save_dir, f"{args.exp_prefix}_patientAgg.json")
    out_csv = os.path.join(args.save_dir, f"{args.exp_prefix}_patientAgg.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "n_patients",
                "n_images",
                "topk",
                "auc_patient_max",
                "auc_patient_mean",
                "auc_patient_topk_mean",
            ],
        )
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow(
                {
                    "fold": i,
                    "n_patients": r["n_patients"],
                    "n_images": r["n_images"],
                    "topk": r["topk"],
                    "auc_patient_max": r["auc_patient_max"],
                    "auc_patient_mean": r["auc_patient_mean"],
                    "auc_patient_topk_mean": r["auc_patient_topk_mean"],
                }
            )

    # 打印均值/方差
    def _mean_std(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in results if r[key] == r[key]]  # 过滤 nan
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    m1, s1 = _mean_std("auc_patient_max")
    m2, s2 = _mean_std("auc_patient_mean")
    m3, s3 = _mean_std("auc_patient_topk_mean")

    print("\n" + "=" * 80)
    print("[PatientAgg] Summary (mean ± std across folds)")
    print(f"  max      : {m1:.4f} ± {s1:.4f}")
    print(f"  mean     : {m2:.4f} ± {s2:.4f}")
    print(f"  top{args.topk}-mean: {m3:.4f} ± {s3:.4f}")
    print(f"[PatientAgg] saved to:\n  {out_json}\n  {out_csv}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

