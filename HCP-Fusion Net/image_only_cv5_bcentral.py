"""
image_only_cv5_bcentral.py

患者级 5-fold 对照：Image-only（仅用 B 模态图像，不用临床特征），CentralLNM 单任务。

做两件事：
1) 逐折调用 train.py 训练（groups_dir 指向 groups_cv5_central/fold_k）
2) 训练完成后，对每折 best_model.pth 做患者级聚合评估（max/mean/topk-mean AUC）

输出：
- checkpoints/<exp_prefix>_fold{k}/...（train.py 输出）
- checkpoints/<exp_prefix>_patientAgg.csv/.json（聚合评估输出）

示例：
CUDA_VISIBLE_DEVICES=2 /data2/wjb/anaconda3/envs/pumc/bin/python image_only_cv5_bcentral.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --groups_root groups_cv5_central \
  --save_dir ./checkpoints \
  --exp_prefix cvCentral_Bonly_r50 \
  --python /data2/wjb/anaconda3/envs/pumc/bin/python \
  --epochs 80 \
  --batch_size 16 \
  --device cuda \
  --topk 3
"""

import argparse
import os
import subprocess
import sys


def run_fold(
    python_exe: str,
    data_root: str,
    groups_dir_rel: str,
    save_dir: str,
    exp_name: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_backbone: float,
    device: str,
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
        "false",
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
        "false",
    ]
    print("\n" + "=" * 80)
    print("[ImageOnlyCV] running:")
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser("Image-only (B) patient-level 5-fold CV: CentralLNM")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--groups_root", type=str, default="groups_cv5_central", help="相对 data_root 的 fold 根目录")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--exp_prefix", type=str, default="cvCentral_Bonly")
    # 默认继承当前脚本解释器，避免误用系统 python 导致环境不一致
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    # 1) train per fold
    for i in range(args.folds):
        groups_dir_rel = os.path.join(args.groups_root, f"fold_{i}")
        exp_name = f"{args.exp_prefix}_fold{i}"
        run_fold(
            python_exe=args.python,
            data_root=args.data_root,
            groups_dir_rel=groups_dir_rel,
            save_dir=args.save_dir,
            exp_name=exp_name,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_backbone=args.lr_backbone,
            device=args.device,
        )

    # 2) patient-level aggregation eval
    cmd_eval = [
        args.python,
        "patient_level_agg_eval.py",
        "--save_dir",
        args.save_dir,
        "--exp_prefix",
        args.exp_prefix,
        "--folds",
        str(args.folds),
        "--device",
        args.device,
        "--topk",
        str(args.topk),
    ]
    print("\n" + "=" * 80)
    print("[ImageOnlyCV] running patient-level aggregation eval:")
    print(" ".join(cmd_eval))
    print("=" * 80 + "\n")
    subprocess.run(cmd_eval, check=True)


if __name__ == "__main__":
    main()

