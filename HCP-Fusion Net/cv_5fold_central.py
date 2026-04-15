"""
cv_5fold_central.py

患者级 5-fold 交叉验证（仅 CentralLNM 单任务），用于更可靠地评估泛化能力。

核心规则：
- 患者ID = 文件名第一个下划线前的一串数字（如 "47495071_2_1_1" -> patient_id="47495071"）
- 同一患者的所有样本必须落在同一 fold（避免患者泄漏）
- fold 的分配按“患者级 CentralLNM 阳性率”做近似分层（stratified）

划分策略：
对每个 fold i：
  - test  = fold i
  - val   = fold (i+1)%5
  - train = 其余 3 个 fold

输出：
- 在 data_root 下生成 groups_cv5_central/fold_{k}/train.txt,val.txt,test.txt
- 逐折训练：checkpoints/<exp_prefix>_fold{k}/...
- 最终汇总：checkpoints/<exp_prefix>_cv_summary.json + .csv

使用示例（在 HCP-Fusion Net 目录下）：

CUDA_VISIBLE_DEVICES=2 /data2/wjb/anaconda3/envs/pumc/bin/python cv_5fold_central.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --base_groups_dir groups \
  --out_groups_dir groups_cv5_central \
  --exp_prefix cvCentral_r50 \
  --backbone resnet50 \
  --epochs 80 \
  --batch_size 16
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _read_names(groups_path: str, split: str) -> List[str]:
    p = os.path.join(groups_path, f"{split}.txt")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            stem, _ = os.path.splitext(s)
            out.append(stem)
    return out


def patient_id(name: str) -> str:
    return name.split("_", 1)[0]


def load_label(data_root: str, label_dir: str, name: str) -> Optional[int]:
    p = os.path.join(data_root, label_dir, f"{name}.txt")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            v = f.read().strip()
        y = int(v)
        if y not in (0, 1):
            return None
        return y
    except Exception:
        return None


@dataclass
class PatientRecord:
    pid: str
    samples: List[str]
    y: int  # patient-level label for CentralLNM


def build_patient_records(all_samples: List[str], data_root: str, central_label_dir: str) -> List[PatientRecord]:
    by_pid: Dict[str, List[str]] = defaultdict(list)
    for s in all_samples:
        by_pid[patient_id(s)].append(s)

    records: List[PatientRecord] = []
    missing_samples = 0
    for pid, samples in sorted(by_pid.items(), key=lambda x: x[0]):
        ys = []
        for name in samples:
            y = load_label(data_root, central_label_dir, name)
            if y is None:
                missing_samples += 1
                continue
            ys.append(y)
        # patient-level：只要该患者任一切面 Central=1，就认为患者 Central 阳性（临床更保守）
        y_pid = 1 if (len(ys) > 0 and max(ys) == 1) else 0
        records.append(PatientRecord(pid=pid, samples=samples, y=y_pid))

    if missing_samples > 0:
        print(f"[CV] 警告：发现 {missing_samples} 个样本缺少 Central 标签文件（将按患者聚合后可能被当作 0）。")

    return records


def stratified_group_kfold_patients(records: List[PatientRecord], k: int, seed: int) -> List[List[str]]:
    """
    近似分层的患者级 K 折划分：让每折的阳性患者数量尽量均衡。
    返回 folds，每个 fold 是患者ID列表。
    """
    rng = np.random.RandomState(seed)
    pos = [r.pid for r in records if r.y == 1]
    neg = [r.pid for r in records if r.y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    folds: List[List[str]] = [[] for _ in range(k)]
    # round-robin 分配，保证数量均衡
    for i, pid in enumerate(pos):
        folds[i % k].append(pid)
    for i, pid in enumerate(neg):
        folds[i % k].append(pid)

    # 再洗一次每折内部顺序
    for f in folds:
        rng.shuffle(f)
    return folds


def write_split_files(out_dir: str, train: List[str], val: List[str], test: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    for split, names in [("train", train), ("val", val), ("test", test)]:
        p = os.path.join(out_dir, f"{split}.txt")
        with open(p, "w", encoding="utf-8") as f:
            for n in names:
                f.write(f"{n}\n")


def run_one_fold(
    python_exe: str,
    fold_idx: int,
    data_root: str,
    groups_dir_rel: str,
    save_dir: str,
    exp_name: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_backbone: float,
    use_clinical: bool,
    device: str,
    amp: bool,
):
    cmd = [
        python_exe,
        "train.py",
        "--data_root",
        data_root,
        "--groups_dir",
        groups_dir_rel,
        "--modalities",
        "B",
        "VUSE",
        "VCEUS",
        "--fusion_type",
        "late",
        "--tasks",
        "CentralLNM",
        "--label_dirs",
        "CentralLNM",
        "--task_loss_weights",
        "1.0",
        "--task_mode",
        "single",
        "--use_clinical",
        "true" if use_clinical else "false",
        "--backbone",
        backbone,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--lr_backbone",
        str(lr_backbone),
        "--optimizer",
        "adamw",
        "--scheduler",
        "cosine",
        "--warmup_epochs",
        "5",
        "--loss_type",
        "bce",
        "--use_penalty_loss",
        "false",
        "--monitor_metric",
        "AUC_CentralLNM",
        "--save_dir",
        save_dir,
        "--exp_name",
        exp_name,
        "--device",
        device,
        "--amp",
        "true" if amp else "false",
    ]

    print("\n" + "=" * 80)
    print(f"[CV] Fold {fold_idx}: running train.py")
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True)


def evaluate_fold_test_auc(exp_dir: str) -> Optional[float]:
    """
    从 best_model.pth 中读取保存的 metrics（来自 val），再从 train.log 中解析最终 test AUC 不稳。
    为避免改动太大，这里先返回 best checkpoint 中记录的 AUC_CentralLNM（通常是 val AUC）。
    你更想要 test AUC 的话，我们可以下一步把 test 指标也落盘成 json。
    """
    best_path = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(best_path):
        return None
    import torch

    ckpt = torch.load(best_path, map_location="cpu")
    metrics = ckpt.get("metrics", {}) or {}
    auc = metrics.get("AUC_CentralLNM")
    return float(auc) if auc is not None else None


def main():
    ap = argparse.ArgumentParser("Patient-level 5-fold CV (CentralLNM)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--base_groups_dir", type=str, default="groups", help="原始 groups 目录（只用于取全量样本名）")
    ap.add_argument("--out_groups_dir", type=str, default="groups_cv5_central", help="输出 folds 的 groups 目录名（相对 data_root）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # 默认继承当前脚本解释器，避免误用系统 python 导致环境不一致
    ap.add_argument("--python", type=str, default=sys.executable, help="用于调用 train.py 的 python 可执行文件")

    ap.add_argument("--exp_prefix", type=str, default="cvCentral", help="每折实验名前缀")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")

    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--use_clinical", action="store_true", default=False, help="是否使用临床特征（默认 False，更稳）")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true", default=False)

    args = ap.parse_args()

    base_groups_path = os.path.join(args.data_root, args.base_groups_dir)
    all_samples = []
    for s in ["train", "val", "test"]:
        all_samples.extend(_read_names(base_groups_path, s))
    all_samples = sorted(list(set(all_samples)))

    if len(all_samples) == 0:
        raise RuntimeError(f"[CV] 在 {base_groups_path} 下未找到 train/val/test.txt，无法构建全量样本。")

    records = build_patient_records(all_samples, args.data_root, central_label_dir="CentralLNM")
    folds = stratified_group_kfold_patients(records, k=args.k, seed=args.seed)

    # 便于快速查询：pid -> samples
    pid_to_samples = {r.pid: r.samples for r in records}
    pid_to_y = {r.pid: r.y for r in records}

    out_groups_root = os.path.join(args.data_root, args.out_groups_dir)
    os.makedirs(out_groups_root, exist_ok=True)

    # 生成每折的 train/val/test.txt
    print("\n" + "=" * 80)
    print("[CV] 生成 folds 的 groups 文件")
    print("=" * 80)
    for i in range(args.k):
        test_pids = set(folds[i])
        val_pids = set(folds[(i + 1) % args.k])
        train_pids = set().union(*[set(folds[j]) for j in range(args.k) if j not in (i, (i + 1) % args.k)])

        def collect(pids: Set[str]) -> List[str]:
            names = []
            for pid in pids:
                names.extend(pid_to_samples.get(pid, []))
            return sorted(names)

        train_names = collect(train_pids)
        val_names = collect(val_pids)
        test_names = collect(test_pids)

        fold_dir = os.path.join(out_groups_root, f"fold_{i}")
        write_split_files(fold_dir, train=train_names, val=val_names, test=test_names)

        # 打印每折阳性率（患者级 + 样本级粗略）
        pos_train = sum(pid_to_y[pid] for pid in train_pids)
        pos_val = sum(pid_to_y[pid] for pid in val_pids)
        pos_test = sum(pid_to_y[pid] for pid in test_pids)
        print(
            f"fold_{i}: "
            f"train patients={len(train_pids)} pos={pos_train} | "
            f"val patients={len(val_pids)} pos={pos_val} | "
            f"test patients={len(test_pids)} pos={pos_test}"
        )

    # 逐折训练
    summary_rows = []
    for i in range(args.k):
        groups_dir_rel = os.path.join(args.out_groups_dir, f"fold_{i}")
        exp_name = f"{args.exp_prefix}_fold{i}"

        run_one_fold(
            python_exe=args.python,
            fold_idx=i,
            data_root=args.data_root,
            groups_dir_rel=groups_dir_rel,
            save_dir=args.save_dir,
            exp_name=exp_name,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_backbone=args.lr_backbone,
            use_clinical=args.use_clinical,
            device=args.device,
            amp=args.amp,
        )

        exp_dir = os.path.join(args.save_dir, exp_name)
        val_best_auc = evaluate_fold_test_auc(exp_dir)
        summary_rows.append({"fold": i, "exp_name": exp_name, "best_val_auc": val_best_auc})
        print(f"[CV] Fold {i} finished. best_val_auc={val_best_auc}")

    # 汇总输出
    out_json = os.path.join(args.save_dir, f"{args.exp_prefix}_cv_summary.json")
    out_csv = os.path.join(args.save_dir, f"{args.exp_prefix}_cv_summary.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "exp_name", "best_val_auc"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    aucs = [r["best_val_auc"] for r in summary_rows if r["best_val_auc"] is not None]
    if aucs:
        print("\n" + "=" * 80)
        print(f"[CV] Done. best_val_auc mean={float(np.mean(aucs)):.4f} std={float(np.std(aucs)):.4f}")
        print(f"[CV] Summary saved to:\n  {out_json}\n  {out_csv}")
        print("=" * 80 + "\n")
    else:
        print(f"[CV] Done. Summary saved to {out_json} / {out_csv} (no aucs found).")


if __name__ == "__main__":
    main()

