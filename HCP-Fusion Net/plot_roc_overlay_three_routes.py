import os
import json
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from dataset import build_dataloader
from networks import build_model


def patient_id(name: str) -> str:
    return str(name).split("_", 1)[0]


def load_opt(exp_dir: str) -> SimpleNamespace:
    with open(os.path.join(exp_dir, "config.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    d["exp_dir"] = exp_dir
    return SimpleNamespace(**d)


def _move_batch_to_device(batch, device):
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


def read_split_names(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            out.append(os.path.splitext(s)[0])
    return out


def load_label(data_root, label_dir, name):
    p = os.path.join(data_root, label_dir, f"{name}.txt")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            y = int(f.read().strip())
        return y if y in (0, 1) else None
    except Exception:
        return None


def build_flag_mapping(flag_path):
    mapping = {}
    with open(flag_path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            s = ln.strip()
            if not s:
                continue
            stem = os.path.splitext(s)[0]
            if stem not in mapping:
                mapping[stem] = i
    return mapping


def load_clinical_matrix(csv_path):
    import pandas as pd
    return pd.read_csv(csv_path, header=None).values.astype(np.float32)


def patient_clinical_features(names, data_root, name_to_row, X, agg="mean+max"):
    pid_rows = defaultdict(list)
    pid_ys = defaultdict(list)
    for n in names:
        row = name_to_row.get(n)
        if row is None or row < 0 or row >= X.shape[0]:
            continue
        y = load_label(data_root, "CentralLNM", n)
        if y is None:
            continue
        pid = patient_id(n)
        pid_rows[pid].append(row)
        pid_ys[pid].append(y)

    pids = sorted(pid_rows.keys())
    feats, ys = [], []
    for pid in pids:
        mat = X[pid_rows[pid]]
        f_mean = mat.mean(axis=0)
        f_max = mat.max(axis=0)
        if agg == "mean":
            f = f_mean
        elif agg == "max":
            f = f_max
        else:
            f = np.concatenate([f_mean, f_max], axis=0)
        y_pid = 1 if max(pid_ys[pid]) == 1 else 0
        feats.append(f)
        ys.append(y_pid)
    return np.stack(feats), np.array(ys, dtype=np.int32), pids


def train_lr_val_select(X_tr, y_tr, X_val, y_val, seed=42):
    Cs = [0.01, 0.1, 1.0, 3.0, 10.0]
    best, best_auc = None, -1.0
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
        fpr, tpr, _ = roc_curve(y_val, p_val)
        a = auc(fpr, tpr)
        if a > best_auc:
            best_auc = a
            best = clf
    return best


def infer_image_only_patient_probs(exp_dir, device):
    opt = load_opt(exp_dir)
    _tmp, _ = build_dataloader(opt=opt, split="train", clinical_loader=None, use_weighted_sampler=False, eval_mode=True)
    clinical_input_dim = getattr(opt, "clinical_input_dim", 0)
    model = build_model(opt, clinical_input_dim=clinical_input_dim).to(device)
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader, _ = build_dataloader(opt=opt, split="test", clinical_loader=None, use_weighted_sampler=False, eval_mode=True)
    pid_probs = defaultdict(list)
    pid_labels = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            bd = _move_batch_to_device(batch, device)
            preds = model(bd)
            key = getattr(model, "single_task_key", None) or list(preds.keys())[0]
            p = torch.sigmoid(preds[key]).detach().cpu().numpy()
            y = batch["labels"][:, 0].detach().cpu().numpy().astype(int)
            names = batch["name"]
            for i in range(len(y)):
                pid = patient_id(names[i])
                pid_probs[pid].append(float(p[i]))
                pid_labels[pid].append(int(y[i]))

    pids = sorted(pid_probs.keys())
    y_pat = np.array([1 if max(pid_labels[pid]) == 1 else 0 for pid in pids], dtype=np.int32)
    p_pat = np.array([float(np.mean(pid_probs[pid])) for pid in pids], dtype=np.float32)
    return y_pat, p_pat


def extract_patient_fused_mean(opt, model, device, split):
    loader, _ = build_dataloader(opt=opt, split=split, clinical_loader=None, use_weighted_sampler=False, eval_mode=True)
    pid_vecs = defaultdict(list)
    pid_ys = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            bd = _move_batch_to_device(batch, device)
            out = model.forward_with_features(bd)
            fused = out["fused"].detach().cpu().numpy()
            y = batch["labels"][:, 0].detach().cpu().numpy().astype(int)
            names = batch["name"]
            for i in range(len(y)):
                pid = patient_id(names[i])
                pid_vecs[pid].append(fused[i])
                pid_ys[pid].append(int(y[i]))
    pids = sorted(pid_vecs.keys())
    E = np.stack([np.mean(np.stack(pid_vecs[pid], axis=0), axis=0) for pid in pids], axis=0)
    Y = np.array([1 if max(pid_ys[pid]) == 1 else 0 for pid in pids], dtype=np.int32)
    return E, Y, pids


def align_two(pids1, X1, y1, pids2, X2, y2):
    common = sorted(list(set(pids1) & set(pids2)))
    i1 = [pids1.index(p) for p in common]
    i2 = [pids2.index(p) for p in common]
    return X1[i1], y1[i1], X2[i2], y2[i2], common


def infer_clinical_plus_deeprad_probs(exp_dir, device, clinical_agg="mean+max"):
    opt = load_opt(exp_dir)
    _tmp, _ = build_dataloader(opt=opt, split="train", clinical_loader=None, use_weighted_sampler=False, eval_mode=True)
    clinical_input_dim = getattr(opt, "clinical_input_dim", 0)
    model = build_model(opt, clinical_input_dim=clinical_input_dim).to(device)
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    E_tr, y_tr, pids_tr = extract_patient_fused_mean(opt, model, device, "train")
    E_val, y_val, pids_val = extract_patient_fused_mean(opt, model, device, "val")
    E_te, y_te, pids_te = extract_patient_fused_mean(opt, model, device, "test")

    pca = PCA(n_components=1, random_state=42)
    s_tr = pca.fit_transform(E_tr).reshape(-1, 1)
    s_val = pca.transform(E_val).reshape(-1, 1)
    s_te = pca.transform(E_te).reshape(-1, 1)

    data_root = opt.data_root
    groups_path = os.path.join(data_root, opt.groups_dir)
    tr_names = read_split_names(os.path.join(groups_path, "train.txt"))
    val_names = read_split_names(os.path.join(groups_path, "val.txt"))
    te_names = read_split_names(os.path.join(groups_path, "test.txt"))

    name_to_row = build_flag_mapping(os.path.join(data_root, "lnm_clinical_flag.txt"))
    Xclin = load_clinical_matrix(os.path.join(data_root, "clinical_features.csv"))
    c_tr, y_tr2, p_tr2 = patient_clinical_features(tr_names, data_root, name_to_row, Xclin, agg=clinical_agg)
    c_val, y_val2, p_val2 = patient_clinical_features(val_names, data_root, name_to_row, Xclin, agg=clinical_agg)
    c_te, y_te2, p_te2 = patient_clinical_features(te_names, data_root, name_to_row, Xclin, agg=clinical_agg)

    s_tr_a, y_tr_a, c_tr_a, y_tr_b, _ = align_two(pids_tr, s_tr, y_tr, p_tr2, c_tr, y_tr2)
    s_val_a, y_val_a, c_val_a, y_val_b, _ = align_two(pids_val, s_val, y_val, p_val2, c_val, y_val2)
    s_te_a, y_te_a, c_te_a, y_te_b, _ = align_two(pids_te, s_te, y_te, p_te2, c_te, y_te2)

    Xtr = np.concatenate([c_tr_a, s_tr_a], axis=1)
    Xval = np.concatenate([c_val_a, s_val_a], axis=1)
    Xte = np.concatenate([c_te_a, s_te_a], axis=1)

    clf = train_lr_val_select(Xtr, y_tr_a, Xval, y_val_a, seed=42)
    p_te = clf.predict_proba(Xte)[:, 1]
    return y_te_a, p_te


def infer_clinical_only_probs(data_root, groups_dir_rel, clinical_agg="mean+max"):
    groups_path = os.path.join(data_root, groups_dir_rel)
    tr_names = read_split_names(os.path.join(groups_path, "train.txt"))
    val_names = read_split_names(os.path.join(groups_path, "val.txt"))
    te_names = read_split_names(os.path.join(groups_path, "test.txt"))

    name_to_row = build_flag_mapping(os.path.join(data_root, "lnm_clinical_flag.txt"))
    Xclin = load_clinical_matrix(os.path.join(data_root, "clinical_features.csv"))
    Xtr, ytr, _ = patient_clinical_features(tr_names, data_root, name_to_row, Xclin, agg=clinical_agg)
    Xval, yval, _ = patient_clinical_features(val_names, data_root, name_to_row, Xclin, agg=clinical_agg)
    Xte, yte, _ = patient_clinical_features(te_names, data_root, name_to_row, Xclin, agg=clinical_agg)

    clf = train_lr_val_select(Xtr, ytr, Xval, yval, seed=42)
    pte = clf.predict_proba(Xte)[:, 1]
    return yte, pte


def main():
    save_dir = "./checkpoints"
    image_prefix = "cvCentral_Bonly_r50"
    folds = 5
    out_path = "../figures/roc_overlay_three_routes.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    routes = {
        "Clinical-only": [],
        "Image-only (B)": [],
        "Clinical + DeepRadScore": [],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(folds):
        exp_dir = os.path.join(save_dir, f"{image_prefix}_fold{fold}")
        opt = load_opt(exp_dir)
        data_root = opt.data_root
        groups_dir_rel = opt.groups_dir

        y_c, p_c = infer_clinical_only_probs(data_root, groups_dir_rel, clinical_agg="mean+max")
        y_i, p_i = infer_image_only_patient_probs(exp_dir, device)
        y_d, p_d = infer_clinical_plus_deeprad_probs(exp_dir, device, clinical_agg="mean+max")

        routes["Clinical-only"].append((y_c, p_c))
        routes["Image-only (B)"].append((y_i, p_i))
        routes["Clinical + DeepRadScore"].append((y_d, p_d))

    plt.figure(figsize=(7.6, 6.6))
    styles = {
        "Clinical-only": ("#1f77b4", "-"),
        "Image-only (B)": ("#d62728", "--"),
        "Clinical + DeepRadScore": ("#2ca02c", "-."),
    }

    for name, fold_data in routes.items():
        # macro-average by interpolation
        grid = np.linspace(0, 1, 200)
        tprs = []
        aucs = []
        for y, p in fold_data:
            fpr, tpr, _ = roc_curve(y, p)
            aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(grid, fpr, tpr))
        mean_tpr = np.mean(np.stack(tprs, axis=0), axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        color, ls = styles[name]
        plt.plot(grid, mean_tpr, ls=ls, color=color, lw=2.2, label=f"{name} (AUC={mean_auc:.3f}±{std_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Patient-level ROC Comparison (5-fold Mean Curves)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()

