"""
clinical_plus_deeprad_cv5_central.py

三路对照中的第三路：Clinical + Deep Rad-Score（患者级）baseline，CentralLNM 单任务。

思路：
- 先使用“image-only（B 模态）”训练好的每折 best_model.pth
- 对该折 train/val/test 的所有样本提取深度融合特征 fused（来自 model.forward_with_features 的 'fused'）
- 患者级聚合：对同一患者的 fused 向量取 mean（得到 patient-level embedding）
- 用 PCA 将 patient-level embedding 压缩为 1 维 DeepRadScore（只在 train 上 fit）
- 同时构建临床患者级特征（mean/max/mean+max，默认用 mean+max）
- 将 DeepRadScore 拼到临床特征后，训练 LogisticRegression（val 选 C），test 报告 AUC

依赖：
- 需要先跑完 image_only_cv5_bcentral.py，得到 checkpoints/<image_exp_prefix>_fold{k}/best_model.pth

示例：
CUDA_VISIBLE_DEVICES=2 /data2/wjb/anaconda3/envs/pumc/bin/python clinical_plus_deeprad_cv5_central.py \
  --save_dir ./checkpoints \
  --image_exp_prefix cvCentral_Bonly_r50 \
  --folds 5 \
  --device cuda
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

from dataset import build_dataloader
from networks import build_model


def patient_id(name: str) -> str:
    return str(name).split("_", 1)[0]


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            moved[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            moved[k] = v
    return moved


def load_opt(exp_dir: str) -> SimpleNamespace:
    cfg = os.path.join(exp_dir, "config.json")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"找不到 config.json: {cfg}")
    with open(cfg, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["exp_dir"] = exp_dir
    return SimpleNamespace(**d)


def build_flag_mapping(flag_path: str) -> Dict[str, int]:
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
    import pandas as pd

    df = pd.read_csv(csv_path, header=None)
    return df.values.astype(np.float32)


def patient_clinical_features(
    names: List[str],
    data_root: str,
    name_to_row: Dict[str, int],
    X: np.ndarray,
    agg: str = "mean+max",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns patient-level clinical feature + patient label (Central) + pid list.
    Label: patient positive if any slice Central=1.
    """
    def load_label(name: str) -> Optional[int]:
        p = os.path.join(data_root, "CentralLNM", f"{name}.txt")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            v = f.read().strip()
        try:
            y = int(v)
            return y if y in (0, 1) else None
        except Exception:
            return None

    pid_rows: Dict[str, List[int]] = defaultdict(list)
    pid_ys: Dict[str, List[int]] = defaultdict(list)
    for n in names:
        row = name_to_row.get(n)
        if row is None or row < 0 or row >= X.shape[0]:
            continue
        y = load_label(n)
        if y is None:
            continue
        pid = patient_id(n)
        pid_rows[pid].append(row)
        pid_ys[pid].append(y)

    pids = sorted(pid_rows.keys())
    feats = []
    ys = []
    for pid in pids:
        mat = X[pid_rows[pid]]
        f_mean = mat.mean(axis=0)
        f_max = mat.max(axis=0)
        if agg == "mean":
            f = f_mean
        elif agg == "max":
            f = f_max
        elif agg == "mean+max":
            f = np.concatenate([f_mean, f_max], axis=0)
        else:
            raise ValueError(agg)
        y_pid = 1 if max(pid_ys[pid]) == 1 else 0
        feats.append(f)
        ys.append(y_pid)

    return np.stack(feats, axis=0), np.array(ys, dtype=np.int32), pids


def extract_patient_fused_mean(
    opt,
    model,
    device: torch.device,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    提取 patient-level embedding（对同一患者的 fused 取 mean）。
    Returns: (E_pat [N_pat, D], y_pat [N_pat], pid_list)
    """
    loader, _ = build_dataloader(
        opt=opt,
        split=split,
        clinical_loader=None,
        use_weighted_sampler=False,
        eval_mode=True,
    )

    pid_to_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    pid_to_ys: Dict[str, List[int]] = defaultdict(list)

    model.eval()
    n_batches = len(loader)
    print(f"[Clin+DeepRad] extracting fused embeddings on split='{split}' ({n_batches} batches)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_dev = _move_batch_to_device(batch, device)
            out = model.forward_with_features(batch_dev)
            fused = out["fused"].detach().cpu().numpy()  # [B, D]
            y = batch["labels"][:, 0].detach().cpu().numpy().astype(int)
            names = batch["name"]
            for i in range(len(y)):
                pid = patient_id(names[i])
                pid_to_vecs[pid].append(fused[i])
                pid_to_ys[pid].append(int(y[i]))
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                print(f"[Clin+DeepRad]  split='{split}' progress: {batch_idx+1}/{n_batches} batches")

    pids = sorted(pid_to_vecs.keys())
    E = []
    Y = []
    for pid in pids:
        mat = np.stack(pid_to_vecs[pid], axis=0)
        E.append(mat.mean(axis=0))
        Y.append(1 if max(pid_to_ys[pid]) == 1 else 0)
    return np.stack(E, axis=0), np.array(Y, dtype=np.int32), pids


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def train_lr_val_select(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int,
) -> Tuple[object, float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    Cs = [0.01, 0.1, 1.0, 3.0, 10.0]
    best = None
    best_auc = -1.0
    best_C = None
    for C in Cs:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C, solver="liblinear", class_weight="balanced",
                                         random_state=seed, max_iter=2000)),
            ]
        )
        clf.fit(X_tr, y_tr)
        p_val = clf.predict_proba(X_val)[:, 1]
        auc = auc_safe(y_val, p_val)
        if auc > best_auc:
            best_auc = auc
            best = clf
            best_C = C
    return best, float(best_C), float(best_auc)


def main():
    ap = argparse.ArgumentParser("Clinical + DeepRadScore 5-fold baseline (Central)")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--image_exp_prefix", type=str, required=True, help="image-only 实验前缀，例如 cvCentral_Bonly_r50")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clinical_agg", type=str, default="mean+max", choices=["mean", "max", "mean+max"])
    ap.add_argument("--out_prefix", type=str, default=None, help="输出文件名前缀（默认= image_exp_prefix）")
    args = ap.parse_args()

    out_prefix = args.out_prefix or args.image_exp_prefix
    rows_out: List[Dict] = []

    for fold in range(args.folds):
        exp_dir = os.path.join(args.save_dir, f"{args.image_exp_prefix}_fold{fold}")
        print("\n" + "=" * 80)
        print(f"[Clin+DeepRad] fold {fold}: loading image-only model from {exp_dir}")
        print("=" * 80)
        opt = load_opt(exp_dir)
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

        # build to set clinical_input_dim (even if use_clinical=false, safe)
        _tmp, _ = build_dataloader(opt=opt, split="train", clinical_loader=None, use_weighted_sampler=False, eval_mode=True)
        clinical_input_dim = getattr(opt, "clinical_input_dim", 0)

        model = build_model(opt, clinical_input_dim=clinical_input_dim).to(device)
        best_path = os.path.join(exp_dir, "best_model.pth")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Deep embedding -> PCA1 -> score
        E_tr, y_tr, pids_tr = extract_patient_fused_mean(opt, model, device, split="train")
        E_val, y_val, pids_val = extract_patient_fused_mean(opt, model, device, split="val")
        E_te, y_te, pids_te = extract_patient_fused_mean(opt, model, device, split="test")

        from sklearn.decomposition import PCA

        print(f"[Clin+DeepRad] fold{fold}: fitting PCA(1) on train patient embeddings (n={E_tr.shape[0]}, d={E_tr.shape[1]})")
        pca = PCA(n_components=1, random_state=args.seed)
        s_tr = pca.fit_transform(E_tr).reshape(-1, 1)
        s_val = pca.transform(E_val).reshape(-1, 1)
        s_te = pca.transform(E_te).reshape(-1, 1)
        print(f"[Clin+DeepRad] fold{fold}: PCA done. ExplainedVarRatio={float(pca.explained_variance_ratio_[0]):.4f}")

        # Clinical patient features
        data_root = opt.data_root
        print(f"[Clin+DeepRad] fold{fold}: loading clinical CSV + flag mapping")
        name_to_row = build_flag_mapping(os.path.join(data_root, "lnm_clinical_flag.txt"))
        X_clin = load_clinical_matrix(os.path.join(data_root, "clinical_features.csv"))

        # 用 split 文件里的样本名生成患者临床特征（保证患者集合一致）
        # 直接从 groups_dir 的文件读
        groups_path = os.path.join(data_root, opt.groups_dir)
        def read_names(split: str) -> List[str]:
            with open(os.path.join(groups_path, f"{split}.txt"), "r", encoding="utf-8") as f:
                out=[]
                for ln in f:
                    s=ln.strip()
                    if not s: continue
                    out.append(os.path.splitext(s)[0])
                return out

        clin_tr, y_tr2, pids_tr2 = patient_clinical_features(read_names("train"), data_root, name_to_row, X_clin, agg=args.clinical_agg)
        clin_val, y_val2, pids_val2 = patient_clinical_features(read_names("val"), data_root, name_to_row, X_clin, agg=args.clinical_agg)
        clin_te, y_te2, pids_te2 = patient_clinical_features(read_names("test"), data_root, name_to_row, X_clin, agg=args.clinical_agg)

        # 对齐患者（取交集，避免不一致）：只对齐 score 和 clinical（同 split）
        def align_two(pids1, X1, y1, pids2, X2, y2):
            common = sorted(list(set(pids1) & set(pids2)))
            i1 = [pids1.index(p) for p in common]
            i2 = [pids2.index(p) for p in common]
            return X1[i1], y1[i1], X2[i2], y2[i2], common

        s_tr_a, y_tr_a, clin_tr_a, y_tr_b, common_tr = align_two(pids_tr, s_tr, y_tr, pids_tr2, clin_tr, y_tr2)
        s_val_a, y_val_a, clin_val_a, y_val_b, common_val = align_two(pids_val, s_val, y_val, pids_val2, clin_val, y_val2)
        s_te_a, y_te_a, clin_te_a, y_te_b, common_te = align_two(pids_te, s_te, y_te, pids_te2, clin_te, y_te2)

        # sanity: labels should match
        if not np.array_equal(y_tr_a, y_tr_b):
            print(f"[WARN] fold{fold} train patient labels mismatch after alignment.")
        if not np.array_equal(y_val_a, y_val_b):
            print(f"[WARN] fold{fold} val patient labels mismatch after alignment.")
        if not np.array_equal(y_te_a, y_te_b):
            print(f"[WARN] fold{fold} test patient labels mismatch after alignment.")

        Xtr = np.concatenate([clin_tr_a, s_tr_a], axis=1)
        Xval = np.concatenate([clin_val_a, s_val_a], axis=1)
        Xte = np.concatenate([clin_te_a, s_te_a], axis=1)

        print(f"[Clin+DeepRad] fold{fold}: training LogisticRegression on (n={Xtr.shape[0]}, d={Xtr.shape[1]}) and selecting C by val AUC")
        model_lr, best_C, val_auc = train_lr_val_select(Xtr, y_tr_a, Xval, y_val_a, seed=args.seed)
        p_te = model_lr.predict_proba(Xte)[:, 1]
        test_auc = auc_safe(y_te_a, p_te)

        rows_out.append(
            {
                "fold": fold,
                "clinical_agg": args.clinical_agg,
                "n_patients_train": int(Xtr.shape[0]),
                "n_patients_val": int(Xval.shape[0]),
                "n_patients_test": int(Xte.shape[0]),
                "C": best_C,
                "auc_val": float(val_auc),
                "auc_test": float(test_auc),
            }
        )
        print(f"[Clin+DeepRad] fold{fold} val_auc={val_auc:.4f} test_auc={test_auc:.4f} (C={best_C})")

    out_json = os.path.join(args.save_dir, f"{out_prefix}_clinPlusDeepRad_cv.json")
    out_csv = os.path.join(args.save_dir, f"{out_prefix}_clinPlusDeepRad_cv.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2, ensure_ascii=False)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "clinical_agg",
                "n_patients_train",
                "n_patients_val",
                "n_patients_test",
                "C",
                "auc_val",
                "auc_test",
            ],
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"[Clin+DeepRad] saved to:\n  {out_json}\n  {out_csv}")


if __name__ == "__main__":
    main()

