"""
clinical_only_cv5_central.py

临床特征（clinical_features.csv）only 的患者级 baseline：
- 仅预测 CentralLNM（患者级标签：该患者任一切面 Central=1 则为 1）
- 使用你已经生成的患者级 5-fold 划分文件：
    data_root/<groups_root>/fold_k/{train,val,test}.txt
- 从 lnm_clinical_flag.txt 建立 sample_name -> csv_row_idx 映射
- 从 clinical_features.csv 读取特征矩阵（默认无表头 header=None）

患者级特征聚合（可同时输出）：
1) mean（按患者所有样本的临床向量取均值）
2) max（按患者所有样本的临床向量取逐维最大值）
3) mean+max（拼接成 2D 维度，常用更强 baseline）

模型：
- StandardScaler + LogisticRegression（liblinear）
- 用 val 集选择 C（正则强度），再在 test 集汇报 AUC

输出：
- <save_dir>/<exp_prefix>_clinicalOnly_cv.csv / .json
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 split 文件: {path}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
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


def build_flag_mapping(flag_path: str) -> Dict[str, int]:
    """
    flag.txt: 每行一个图像文件名（可带扩展名），行号 = CSV 行号（0-based）
    返回 stem->row_idx
    """
    if not os.path.exists(flag_path):
        raise FileNotFoundError(f"找不到 flag 文件: {flag_path}")
    mapping: Dict[str, int] = {}
    with open(flag_path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            s = ln.strip()
            if not s:
                continue
            stem, _ = os.path.splitext(s)
            if stem not in mapping:
                mapping[stem] = i
    return mapping


def load_clinical_matrix(csv_path: str) -> np.ndarray:
    """
    读取 clinical_features.csv（默认无表头）。
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到临床 CSV: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    return df.values.astype(np.float32)


def patient_aggregate_features(
    names: List[str],
    data_root: str,
    label_dir: str,
    name_to_row: Dict[str, int],
    X: np.ndarray,
    agg: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    返回患者级 (X_pat, y_pat, pid_list)
    agg: 'mean' | 'max' | 'mean+max'
    """
    pid_to_rows: Dict[str, List[int]] = defaultdict(list)
    pid_to_labels: Dict[str, List[int]] = defaultdict(list)

    missing_row = 0
    missing_label = 0

    for n in names:
        pid = patient_id(n)
        row = name_to_row.get(n)
        if row is None or row < 0 or row >= X.shape[0]:
            missing_row += 1
            continue
        y = load_label(data_root, label_dir, n)
        if y is None:
            missing_label += 1
            continue
        pid_to_rows[pid].append(row)
        pid_to_labels[pid].append(y)

    if missing_row > 0:
        print(f"[ClinicalOnly] 警告：有 {missing_row} 个样本找不到合法的 CSV 行号映射（已跳过）")
    if missing_label > 0:
        print(f"[ClinicalOnly] 警告：有 {missing_label} 个样本缺少标签（已跳过）")

    pids = sorted(pid_to_rows.keys())
    feats = []
    ys = []
    for pid in pids:
        rows = pid_to_rows[pid]
        mat = X[rows]  # [n_i, D]
        f_mean = mat.mean(axis=0)
        f_max = mat.max(axis=0)
        if agg == "mean":
            f = f_mean
        elif agg == "max":
            f = f_max
        elif agg == "mean+max":
            f = np.concatenate([f_mean, f_max], axis=0)
        else:
            raise ValueError(f"未知 agg: {agg}")

        # 患者级标签：只要任一切面为 1，则患者=1
        y_pid = 1 if max(pid_to_labels[pid]) == 1 else 0
        feats.append(f)
        ys.append(y_pid)

    return np.stack(feats, axis=0), np.array(ys, dtype=np.int32), pids


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def train_lr_with_val_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[object, float, float, float]:
    """
    返回：best_model, best_C, val_auc, train_auc
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    Cs = [0.01, 0.1, 1.0, 3.0, 10.0]
    best = None
    best_auc = -1.0
    best_C = None
    best_train_auc = None

    for C in Cs:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=2000,
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)
        p_val = clf.predict_proba(X_val)[:, 1]
        p_tr = clf.predict_proba(X_train)[:, 1]
        val_auc = auc_safe(y_val, p_val)
        tr_auc = auc_safe(y_train, p_tr)
        if val_auc > best_auc:
            best_auc = val_auc
            best = clf
            best_C = C
            best_train_auc = tr_auc

    return best, float(best_C), float(best_auc), float(best_train_auc)


def main():
    ap = argparse.ArgumentParser("Clinical-only patient-level 5-fold baseline (CentralLNM)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--groups_root", type=str, default="groups_cv5_central", help="相对 data_root 的 folds 根目录")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--exp_prefix", type=str, default="cvCentral_clinicalOnly")
    ap.add_argument("--agg", type=str, default="all", choices=["all", "mean", "max", "mean+max"])
    args = ap.parse_args()

    data_root = args.data_root
    flag_path = os.path.join(data_root, "lnm_clinical_flag.txt")
    csv_path = os.path.join(data_root, "clinical_features.csv")
    name_to_row = build_flag_mapping(flag_path)
    X = load_clinical_matrix(csv_path)

    aggs = ["mean", "max", "mean+max"] if args.agg == "all" else [args.agg]
    rows_out: List[Dict] = []

    for fold in range(args.folds):
        fold_dir = os.path.join(data_root, args.groups_root, f"fold_{fold}")
        train_names = read_list(os.path.join(fold_dir, "train.txt"))
        val_names = read_list(os.path.join(fold_dir, "val.txt"))
        test_names = read_list(os.path.join(fold_dir, "test.txt"))

        for agg in aggs:
            X_tr, y_tr, pids_tr = patient_aggregate_features(
                train_names, data_root, "CentralLNM", name_to_row, X, agg=agg
            )
            X_val, y_val, pids_val = patient_aggregate_features(
                val_names, data_root, "CentralLNM", name_to_row, X, agg=agg
            )
            X_te, y_te, pids_te = patient_aggregate_features(
                test_names, data_root, "CentralLNM", name_to_row, X, agg=agg
            )

            model, best_C, val_auc, train_auc = train_lr_with_val_select(
                X_tr, y_tr, X_val, y_val, seed=args.seed
            )

            p_te = model.predict_proba(X_te)[:, 1]
            test_auc = auc_safe(y_te, p_te)

            rows_out.append(
                {
                    "fold": fold,
                    "agg": agg,
                    "n_patients_train": int(len(y_tr)),
                    "n_patients_val": int(len(y_val)),
                    "n_patients_test": int(len(y_te)),
                    "C": best_C,
                    "auc_train": train_auc,
                    "auc_val": val_auc,
                    "auc_test": float(test_auc),
                }
            )

            print(
                f"[ClinicalOnly] fold{fold} agg={agg:<8s} "
                f"AUC train={train_auc:.4f} val={val_auc:.4f} test={test_auc:.4f} (C={best_C})"
            )

    os.makedirs(args.save_dir, exist_ok=True)
    out_json = os.path.join(args.save_dir, f"{args.exp_prefix}_clinicalOnly_cv.json")
    out_csv = os.path.join(args.save_dir, f"{args.exp_prefix}_clinicalOnly_cv.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2, ensure_ascii=False)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold",
            "agg",
            "n_patients_train",
            "n_patients_val",
            "n_patients_test",
            "C",
            "auc_train",
            "auc_val",
            "auc_test",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[ClinicalOnly] saved to:\n  {out_json}\n  {out_csv}")


if __name__ == "__main__":
    main()

