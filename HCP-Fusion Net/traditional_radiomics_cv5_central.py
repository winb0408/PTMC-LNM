#!/usr/bin/env python3
"""
traditional_radiomics_cv5_central.py

CentralLNM 传统机器学习对照（患者级 5-fold）
统一比较三条 pipeline：
1) clinical_only         : 仅临床特征
2) radiomics_only        : 仅传统影像组学（由图像统计特征构成）
3) clinical_radiomics    : 临床 + 传统影像组学

影像组学特征（无分割mask版本，按切片RGB/灰度统计）:
- 每个样本提取 B 模态基础统计特征（强度直方图 + 一阶统计 + 颜色通道统计）
- 按患者聚合（mean / max / mean+max）

模型族：
- LogisticRegression
- RandomForest
- SVM(RBF)
- AdaBoost
- ExtraTrees
- XGBoost(若未安装则自动跳过)
- Stacking(先进集成: LR + RF + SVM -> LR)
- Voting(软投票: LR + RF + SVM)

输出：
- <save_dir>/<exp_prefix>_all_metrics.csv/.json
- <save_dir>/<exp_prefix>_summary_mean_std.csv
- <save_dir>/<exp_prefix>_roc_<pipeline>_<model>.png
- <save_dir>/<exp_prefix>_auc_boxplot_by_pipeline.png
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def read_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 split 文件: {path}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            out.append(os.path.splitext(s)[0])
    return out


def patient_id(name: str) -> str:
    return str(name).split("_", 1)[0]


def load_label(data_root: str, label_dir: str, name: str) -> Optional[int]:
    p = os.path.join(data_root, label_dir, f"{name}.txt")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            y = int(f.read().strip())
        return y if y in (0, 1) else None
    except Exception:
        return None


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

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到临床 CSV: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    return df.values.astype(np.float32)


def _find_image_path(modality_dir: str, name: str) -> Optional[str]:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    for ext in exts:
        p = os.path.join(modality_dir, f"{name}{ext}")
        if os.path.exists(p):
            return p
    return None


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def extract_basic_radiomics_features(img_path: str, bins: int = 16) -> np.ndarray:
    """
    简化传统影像组学：从单张B模态图像提取稳定统计特征。
    不依赖分割mask，便于与你当前数据直接对齐复现。
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    gray = np.mean(arr, axis=2)

    feats: List[float] = []

    # 灰度一阶统计
    g_flat = gray.reshape(-1)
    g_mean = float(np.mean(g_flat))
    g_std = float(np.std(g_flat))
    g_min = float(np.min(g_flat))
    g_max = float(np.max(g_flat))
    q25, q50, q75 = np.percentile(g_flat, [25, 50, 75]).tolist()
    g_energy = float(np.mean(g_flat * g_flat))
    g_abs_mean = float(np.mean(np.abs(g_flat - g_mean)))

    feats.extend([g_mean, g_std, g_min, g_max, q25, q50, q75, g_energy, g_abs_mean])

    # 直方图特征
    hist, _ = np.histogram(g_flat, bins=bins, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float32)
    hist = hist / (np.sum(hist) + 1e-12)
    hist_entropy = float(-np.sum(hist * np.log(hist + 1e-12)))
    feats.extend(hist.tolist())
    feats.append(hist_entropy)

    # RGB各通道统计
    for c in range(3):
        c_flat = arr[:, :, c].reshape(-1)
        c_mean = float(np.mean(c_flat))
        c_std = float(np.std(c_flat))
        c_p10, c_p90 = np.percentile(c_flat, [10, 90]).tolist()
        feats.extend([c_mean, c_std, c_p10, c_p90])

    # 简单纹理代理：梯度统计
    gy, gx = np.gradient(gray)
    gmag = np.sqrt(gx * gx + gy * gy).reshape(-1)
    feats.extend([
        float(np.mean(gmag)),
        float(np.std(gmag)),
        float(np.percentile(gmag, 90)),
    ])

    return np.asarray(feats, dtype=np.float32)


def build_patient_level_dataset(
    names: List[str],
    data_root: str,
    label_dir: str,
    name_to_row: Dict[str, int],
    X_clinical: np.ndarray,
    modality: str = "B",
    clinical_agg: str = "mean+max",
    rad_agg: str = "mean+max",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    返回：X_clin_pat, X_rad_pat, y_pat, pid_list
    """
    modality_dir = os.path.join(data_root, modality)

    pid_clin_rows: Dict[str, List[int]] = defaultdict(list)
    pid_rad_feats: Dict[str, List[np.ndarray]] = defaultdict(list)
    pid_labels: Dict[str, List[int]] = defaultdict(list)

    missing_row = 0
    missing_label = 0
    missing_img = 0

    for n in names:
        pid = patient_id(n)
        row = name_to_row.get(n)
        if row is None or row < 0 or row >= X_clinical.shape[0]:
            missing_row += 1
            continue
        y = load_label(data_root, label_dir, n)
        if y is None:
            missing_label += 1
            continue
        img_path = _find_image_path(modality_dir, n)
        if img_path is None:
            missing_img += 1
            continue

        rad = extract_basic_radiomics_features(img_path)

        pid_clin_rows[pid].append(row)
        pid_rad_feats[pid].append(rad)
        pid_labels[pid].append(y)

    if missing_row > 0:
        print(f"[WARN] 缺少临床映射样本: {missing_row}")
    if missing_label > 0:
        print(f"[WARN] 缺少标签样本: {missing_label}")
    if missing_img > 0:
        print(f"[WARN] 缺少图像样本: {missing_img}")

    pids = sorted(set(pid_clin_rows.keys()) & set(pid_rad_feats.keys()) & set(pid_labels.keys()))

    Xc_list: List[np.ndarray] = []
    Xr_list: List[np.ndarray] = []
    y_list: List[int] = []

    for pid in pids:
        c_mat = X_clinical[pid_clin_rows[pid]]
        r_mat = np.stack(pid_rad_feats[pid], axis=0)

        c_mean, c_max = c_mat.mean(axis=0), c_mat.max(axis=0)
        r_mean, r_max = r_mat.mean(axis=0), r_mat.max(axis=0)

        if clinical_agg == "mean":
            c_feat = c_mean
        elif clinical_agg == "max":
            c_feat = c_max
        else:
            c_feat = np.concatenate([c_mean, c_max], axis=0)

        if rad_agg == "mean":
            r_feat = r_mean
        elif rad_agg == "max":
            r_feat = r_max
        else:
            r_feat = np.concatenate([r_mean, r_max], axis=0)

        y_pid = 1 if max(pid_labels[pid]) == 1 else 0

        Xc_list.append(c_feat.astype(np.float32))
        Xr_list.append(r_feat.astype(np.float32))
        y_list.append(y_pid)

    return np.stack(Xc_list, axis=0), np.stack(Xr_list, axis=0), np.asarray(y_list, dtype=np.int32), pids


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def calc_metrics(y: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    pred = (prob >= thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "auc": auc_safe(y, prob),
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": _safe_div(tn, tn + fp),
        "sensitivity": _safe_div(tp, tp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_model_registry(seed: int) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models: Dict[str, object] = {}

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=seed, max_iter=4000)),
    ])
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
    ])
    adb = AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=seed)
    et = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)

    models["LogReg"] = lr
    models["RandomForest"] = rf
    models["SVM_RBF"] = svm
    models["AdaBoost"] = adb
    models["ExtraTrees"] = et

    # XGBoost 可选
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        models["XGBoost"] = xgb
    except Exception:
        print("[INFO] 未检测到 xgboost，已跳过 XGBoost 模型。")

    # 集成学习
    stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
        final_estimator=LogisticRegression(class_weight="balanced", solver="liblinear", random_state=seed, max_iter=3000),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    vote = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
        voting="soft",
        n_jobs=-1,
    )
    models["Stacking"] = stack
    models["SoftVoting"] = vote

    return models


def get_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        d = np.asarray(d, dtype=np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-12)
        return d
    raise RuntimeError("模型既无 predict_proba 也无 decision_function")


def plot_roc_multimodel(y_probs_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: str, title: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(7.0, 6.0))
    for model_name, (y, p) in y_probs_by_model.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={a:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_auc_boxplot(rows: List[Dict], out_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df[df["split"] == "test"]

    grouped = defaultdict(list)
    for _, r in df.iterrows():
        k = f"{r['pipeline']}\n{r['model']}"
        grouped[k].append(float(r["auc"]))

    keys = sorted(grouped.keys())
    vals = [grouped[k] for k in keys]

    plt.figure(figsize=(max(12, len(keys) * 0.5), 6.5))
    plt.boxplot(vals, labels=keys, showmeans=True)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("AUC (5-fold test)")
    plt.title("Traditional ML Comparison by Pipeline and Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Traditional radiomics-style ML baseline (CentralLNM, 5-fold)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--groups_root", type=str, default="groups_cv5_central")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--modality", type=str, default="B", choices=["B", "VUSE", "VCEUS"])
    ap.add_argument("--clinical_agg", type=str, default="mean+max", choices=["mean", "max", "mean+max"])
    ap.add_argument("--rad_agg", type=str, default="mean+max", choices=["mean", "max", "mean+max"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--exp_prefix", type=str, default="cvCentral_traditionalML")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    name_to_row = build_flag_mapping(os.path.join(args.data_root, "lnm_clinical_flag.txt"))
    X_clin = load_clinical_matrix(os.path.join(args.data_root, "clinical_features.csv"))

    model_registry = build_model_registry(seed=args.seed)

    all_rows: List[Dict] = []
    fold_prob_cache = defaultdict(dict)  # (pipeline -> model -> (y,p) for each fold append)

    for fold in range(args.folds):
        fold_dir = os.path.join(args.data_root, args.groups_root, f"fold_{fold}")
        tr_names = read_list(os.path.join(fold_dir, "train.txt"))
        va_names = read_list(os.path.join(fold_dir, "val.txt"))
        te_names = read_list(os.path.join(fold_dir, "test.txt"))

        Xc_tr, Xr_tr, y_tr, _ = build_patient_level_dataset(tr_names, args.data_root, "CentralLNM", name_to_row, X_clin, modality=args.modality, clinical_agg=args.clinical_agg, rad_agg=args.rad_agg)
        Xc_va, Xr_va, y_va, _ = build_patient_level_dataset(va_names, args.data_root, "CentralLNM", name_to_row, X_clin, modality=args.modality, clinical_agg=args.clinical_agg, rad_agg=args.rad_agg)
        Xc_te, Xr_te, y_te, _ = build_patient_level_dataset(te_names, args.data_root, "CentralLNM", name_to_row, X_clin, modality=args.modality, clinical_agg=args.clinical_agg, rad_agg=args.rad_agg)

        pipelines = {
            "clinical_only": (Xc_tr, Xc_va, Xc_te),
            "radiomics_only": (Xr_tr, Xr_va, Xr_te),
            "clinical_radiomics": (
                np.concatenate([Xc_tr, Xr_tr], axis=1),
                np.concatenate([Xc_va, Xr_va], axis=1),
                np.concatenate([Xc_te, Xr_te], axis=1),
            ),
        }

        for pipe_name, (Xtr, Xva, Xte) in pipelines.items():
            for model_name, model in model_registry.items():
                try:
                    model.fit(Xtr, y_tr)
                    p_tr = get_proba(model, Xtr)
                    p_va = get_proba(model, Xva)
                    p_te = get_proba(model, Xte)

                    m_tr = calc_metrics(y_tr, p_tr)
                    m_va = calc_metrics(y_va, p_va)
                    m_te = calc_metrics(y_te, p_te)

                    for split_name, m in [("train", m_tr), ("val", m_va), ("test", m_te)]:
                        row = {
                            "fold": fold,
                            "pipeline": pipe_name,
                            "model": model_name,
                            "split": split_name,
                            **m,
                        }
                        all_rows.append(row)

                    fold_prob_cache[(pipe_name, model_name)][fold] = (y_te.copy(), p_te.copy())

                    print(
                        f"[Fold{fold}] {pipe_name:18s} {model_name:12s} "
                        f"AUC train/val/test = {m_tr['auc']:.3f}/{m_va['auc']:.3f}/{m_te['auc']:.3f}"
                    )
                except Exception as e:
                    print(f"[WARN] fold{fold} {pipe_name}-{model_name} 训练失败: {e}")

    # 保存明细
    out_csv = os.path.join(args.save_dir, f"{args.exp_prefix}_all_metrics.csv")
    out_json = os.path.join(args.save_dir, f"{args.exp_prefix}_all_metrics.json")

    fieldnames = [
        "fold", "pipeline", "model", "split",
        "auc", "acc", "precision", "recall", "f1", "specificity", "sensitivity",
        "tn", "fp", "fn", "tp",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    # 汇总 mean±std（test）
    import pandas as pd

    df = pd.DataFrame(all_rows)
    dft = df[df["split"] == "test"].copy()
    sum_rows = []
    for (pipe, model), g in dft.groupby(["pipeline", "model"]):
        sum_rows.append({
            "pipeline": pipe,
            "model": model,
            "auc_mean": float(g["auc"].mean()),
            "auc_std": float(g["auc"].std(ddof=0)),
            "acc_mean": float(g["acc"].mean()),
            "f1_mean": float(g["f1"].mean()),
            "sensitivity_mean": float(g["sensitivity"].mean()),
            "specificity_mean": float(g["specificity"].mean()),
        })
    sum_df = pd.DataFrame(sum_rows).sort_values(["pipeline", "auc_mean"], ascending=[True, False])
    out_summary = os.path.join(args.save_dir, f"{args.exp_prefix}_summary_mean_std.csv")
    sum_df.to_csv(out_summary, index=False)

    # 每个pipeline画一张多模型ROC（聚合5折test）
    for pipe in sorted(dft["pipeline"].unique().tolist()):
        model_curves = {}
        for model in sorted(dft[dft["pipeline"] == pipe]["model"].unique().tolist()):
            ys, ps = [], []
            for fold in range(args.folds):
                key = (pipe, model)
                if fold not in fold_prob_cache[key]:
                    continue
                y, p = fold_prob_cache[key][fold]
                ys.append(y)
                ps.append(p)
            if not ys:
                continue
            model_curves[model] = (np.concatenate(ys), np.concatenate(ps))

        if model_curves:
            out_roc = os.path.join(args.save_dir, f"{args.exp_prefix}_roc_{pipe}.png")
            plot_roc_multimodel(model_curves, out_roc, f"ROC Comparison ({pipe})")

    # 箱线图
    out_box = os.path.join(args.save_dir, f"{args.exp_prefix}_auc_boxplot_by_pipeline.png")
    plot_auc_boxplot(all_rows, out_box)

    print("\n[Done] 输出文件:")
    print(f"  - {out_csv}")
    print(f"  - {out_json}")
    print(f"  - {out_summary}")
    print(f"  - {out_box}")
    print("  - ROC 图: <save_dir>/<exp_prefix>_roc_<pipeline>.png")


if __name__ == "__main__":
    main()
