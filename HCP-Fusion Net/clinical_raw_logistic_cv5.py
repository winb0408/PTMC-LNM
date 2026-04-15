#!/usr/bin/env python3
"""
clinical_raw_logistic_cv5.py

基于原始临床 Excel 的患者级 5-fold 临床回归基线：
1) 读取原始 Excel（Image / HISID / D~Z 特征）
2) 按 groups_cv5_central 的 train/val/test 划分对齐到同一患者级 split
3) 用 AC/AG/AE > 0 构造 LNM / CentralLNM / LateralLNM 标签
4) 训练集内完成缺失值处理、类别编码与单/多因素 Logistic 回归
5) 导出 OR / CI / p 值、逐折指标、ROC 图和汇总报表
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import warnings


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Raw clinical logistic baseline with patient-level 5-fold")
    p.add_argument("--excel_path", type=str, default="/data2/wjb/code/PTMC_LNM/data/(withLNM)data3_clinical_huizong.xlsx")
    p.add_argument("--data_root", type=str, default="/data2/wjb/code/PTMC_LNM/data/MultiModalDataset")
    p.add_argument("--groups_root", type=str, default="groups_cv5_central")
    p.add_argument("--save_dir", type=str, default="./checkpoints_raw_clinical")
    p.add_argument("--exp_name", type=str, default="raw_clinical_logistic_cv5")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--univariate_p", type=float, default=0.10)
    p.add_argument("--max_multivar_features", type=int, default=10)
    return p.parse_args()


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [os.path.splitext(ln.strip())[0] for ln in f if ln.strip()]


def normalize_image_stem(v) -> str:
    if pd.isna(v):
        return ""
    return os.path.splitext(str(v).strip())[0]


def normalize_hisid(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_categorical_value(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if not s:
        return np.nan
    s = s.replace("，", ",").replace("、", ",").replace(" ", "")
    if s in {"0.0", "1.0", "2.0", "3.0", "4.0"}:
        s = s[:-2]
    return s


def infer_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    categorical, numeric = [], []
    for col in feature_cols:
        s = df[col]
        if s.dtype == object:
            categorical.append(col)
            continue
        nunique = int(s.dropna().nunique())
        if pd.api.types.is_integer_dtype(s) and nunique <= 6:
            categorical.append(col)
        else:
            numeric.append(col)
    return categorical, numeric


def mode_or_nan(values: List):
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    return Counter(vals).most_common(1)[0][0]


def build_patient_feature_table(raw_df: pd.DataFrame, feature_cols: List[str], categorical_cols: List[str], numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    patient_rows = []
    grouped = raw_df.groupby("_patient_id", sort=True)
    for pid, g in grouped:
        row = {"patient_id": pid}
        for col in feature_cols:
            vals = g[col].tolist()
            non_na = [v for v in vals if not pd.isna(v)]
            inconsistent = len(pd.unique(non_na)) > 1 if non_na else False
            audit_rows.append({
                "patient_id": pid,
                "feature": col,
                "n_rows": int(len(vals)),
                "n_unique_non_na": int(len(pd.unique(non_na))) if non_na else 0,
                "inconsistent": int(inconsistent),
            })
            if col in categorical_cols:
                row[col] = mode_or_nan(vals)
            else:
                arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy()
                row[col] = float(np.median(arr)) if arr.size > 0 else np.nan

        # 以患者级任一图像 >0 构造患者标签
        row["LNM"] = int((pd.to_numeric(g["_label_lnm"], errors="coerce").fillna(0) > 0).any())
        row["CentralLNM"] = int((pd.to_numeric(g["_label_central"], errors="coerce").fillna(0) > 0).any())
        row["LateralLNM"] = int((pd.to_numeric(g["_label_lateral"], errors="coerce").fillna(0) > 0).any())
        row["n_images"] = int(len(g))
        patient_rows.append(row)

    return pd.DataFrame(patient_rows), pd.DataFrame(audit_rows)


def auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "AUC": auc_safe(y_true, y_prob),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(sen),
        "Specificity": float(spe),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def build_preprocessor(train_df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    pre.fit(train_df)

    num_features = list(numeric_cols)
    cat_features = []
    if categorical_cols:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_features = list(ohe.get_feature_names_out(categorical_cols))
    return pre, num_features + cat_features


def transform_df(preprocessor, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    X = preprocessor.transform(df)
    return pd.DataFrame(X, index=df.index, columns=feature_names)


def fit_logit(X: pd.DataFrame, y: pd.Series):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y.astype(float), Xc.astype(float))
    return model.fit(disp=0, maxiter=200)


def fit_logit_regularized(X: pd.DataFrame, y: pd.Series):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y.astype(float), Xc.astype(float))
    return model.fit_regularized(disp=0, maxiter=300, alpha=1e-4)


def run_univariate_screen(X_train: pd.DataFrame, y_train: pd.Series, p_threshold: float, max_keep: int, task: str, fold: int) -> Tuple[List[str], List[Dict]]:
    rows = []
    selected = []
    for col in X_train.columns:
        if np.nanstd(X_train[col].to_numpy()) < 1e-12:
            continue
        try:
            res = fit_logit(X_train[[col]], y_train)
            coef = float(res.params[col])
            ci = res.conf_int().loc[col].tolist()
            p = float(res.pvalues[col])
            rows.append({
                "task": task,
                "fold": fold,
                "feature": col,
                "coef": coef,
                "OR": float(np.exp(coef)),
                "CI_low": float(np.exp(ci[0])),
                "CI_high": float(np.exp(ci[1])),
                "p_value": p,
                "fit_mode": "standard",
            })
        except Exception:
            continue
    rows = sorted(rows, key=lambda r: (r["p_value"], -abs(r["coef"])))
    selected = [r["feature"] for r in rows if r["p_value"] < p_threshold]
    if not selected and rows:
        selected = [rows[0]["feature"]]
    if len(selected) > max_keep:
        selected = selected[:max_keep]
    return selected, rows


def run_multivariate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: Dict[str, pd.DataFrame],
    selected_features: List[str],
    task: str,
    fold: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict], str]:
    Xs = X_train[selected_features].copy()
    fit_mode = "standard"
    try:
        res = fit_logit(Xs, y_train)
    except Exception:
        fit_mode = "regularized"
        res = fit_logit_regularized(Xs, y_train)

    preds = {}
    for split, X_df in X_eval.items():
        Xc = sm.add_constant(X_df[selected_features], has_constant="add")
        preds[split] = np.asarray(res.predict(Xc), dtype=float)

    rows = []
    for feat in selected_features:
        coef = float(res.params.get(feat, np.nan))
        row = {
            "task": task,
            "fold": fold,
            "feature": feat,
            "coef": coef,
            "OR": float(np.exp(coef)) if np.isfinite(coef) else np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "p_value": np.nan,
            "fit_mode": fit_mode,
        }
        if fit_mode == "standard":
            ci = res.conf_int().loc[feat].tolist()
            row["CI_low"] = float(np.exp(ci[0]))
            row["CI_high"] = float(np.exp(ci[1]))
            row["p_value"] = float(res.pvalues[feat])
        rows.append(row)
    return preds, rows, fit_mode


def plot_task_roc(exp_dir: str, task: str, fold_test_preds: List[Dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for item in fold_test_preds:
        y = item["y"]
        p = item["prob"]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, alpha=0.35, label=f'Fold {item["fold"]} (AUC={auc:.3f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(float(auc))
    if tprs:
        mean_tpr = np.mean(np.vstack(tprs), axis=0)
        ax.plot(mean_fpr, mean_tpr, lw=2.2, label=f"Mean (AUC={np.mean(aucs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Raw Clinical ROC ({task})")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, f"roc_{task}.png"), dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    raw_df = pd.read_excel(args.excel_path, sheet_name=0)
    raw_df["_image_stem"] = raw_df["Image"].apply(normalize_image_stem)
    raw_df["_patient_id"] = raw_df["HISID"].apply(normalize_hisid)

    feature_cols = list(raw_df.columns[3:26])
    label_lnm_col = raw_df.columns[28]
    label_lateral_col = raw_df.columns[30]
    label_central_col = raw_df.columns[32]
    raw_df["_label_lnm"] = pd.to_numeric(raw_df[label_lnm_col], errors="coerce").fillna(0)
    raw_df["_label_lateral"] = pd.to_numeric(raw_df[label_lateral_col], errors="coerce").fillna(0)
    raw_df["_label_central"] = pd.to_numeric(raw_df[label_central_col], errors="coerce").fillna(0)

    # object 列统一做轻量规范化，避免 '1、2' / '0 ' 这类值扰动
    for col in feature_cols:
        if raw_df[col].dtype == object:
            raw_df[col] = raw_df[col].apply(normalize_categorical_value)

    categorical_cols, numeric_cols = infer_feature_types(raw_df, feature_cols)
    patient_df, audit_df = build_patient_feature_table(raw_df, feature_cols, categorical_cols, numeric_cols)
    audit_summary = (
        audit_df.groupby("feature")["inconsistent"]
        .sum()
        .reset_index()
        .rename(columns={"inconsistent": "n_inconsistent_patients"})
        .sort_values("n_inconsistent_patients", ascending=False)
    )
    audit_df.to_csv(os.path.join(exp_dir, "patient_feature_audit.csv"), index=False)
    audit_summary.to_csv(os.path.join(exp_dir, "patient_feature_audit_summary.csv"), index=False)

    image_to_patient = {
        normalize_image_stem(img): pid
        for img, pid in zip(raw_df["Image"], raw_df["_patient_id"])
        if normalize_image_stem(img) and pid
    }

    task_targets = {
        "LNM": "LNM",
        "CentralLNM": "CentralLNM",
        "LateralLNM": "LateralLNM",
    }

    all_metrics_rows = []
    summary_rows = []
    univariate_rows = []
    multivariate_rows = []
    fold_pred_cache = defaultdict(list)

    for fold in range(args.folds):
        fold_dir = os.path.join(args.data_root, args.groups_root, f"fold_{fold}")
        split_patients = {}
        for split in ["train", "val", "test"]:
            names = read_list(os.path.join(fold_dir, f"{split}.txt"))
            pids = sorted({image_to_patient[n] for n in names if n in image_to_patient})
            split_patients[split] = pids

        for task, target_col in task_targets.items():
            train_df = patient_df[patient_df["patient_id"].isin(split_patients["train"])].reset_index(drop=True)
            val_df = patient_df[patient_df["patient_id"].isin(split_patients["val"])].reset_index(drop=True)
            test_df = patient_df[patient_df["patient_id"].isin(split_patients["test"])].reset_index(drop=True)

            pre, transformed_feature_names = build_preprocessor(train_df[feature_cols], categorical_cols, numeric_cols)
            X_train = transform_df(pre, train_df[feature_cols], transformed_feature_names)
            X_val = transform_df(pre, val_df[feature_cols], transformed_feature_names)
            X_test = transform_df(pre, test_df[feature_cols], transformed_feature_names)

            y_train = train_df[target_col].astype(int)
            y_val = val_df[target_col].astype(int)
            y_test = test_df[target_col].astype(int)

            selected, uni_rows = run_univariate_screen(
                X_train, y_train, args.univariate_p, args.max_multivar_features, task, fold
            )
            univariate_rows.extend(uni_rows)

            preds, multi_rows, fit_mode = run_multivariate(
                X_train,
                y_train,
                {"train": X_train, "val": X_val, "test": X_test},
                selected,
                task,
                fold,
            )
            multivariate_rows.extend(multi_rows)

            for split, y_true, prob in [
                ("train", y_train.to_numpy(), preds["train"]),
                ("val", y_val.to_numpy(), preds["val"]),
                ("test", y_test.to_numpy(), preds["test"]),
            ]:
                met = compute_metrics(y_true, prob)
                all_metrics_rows.append({
                    "task": task,
                    "fold": fold,
                    "split": split,
                    "fit_mode": fit_mode,
                    "n_features_multivar": int(len(selected)),
                    **met,
                })
            fold_pred_cache[task].append({"fold": fold, "y": y_test.to_numpy(), "prob": preds["test"]})

            test_met = all_metrics_rows[-1]
            print(
                f"[RawClinical] fold={fold} task={task:<11s} "
                f"AUC_test={test_met['AUC']:.4f} ACC={test_met['ACC']:.4f} "
                f"F1={test_met['F1']:.4f} n_feat={len(selected)} mode={fit_mode}"
            )

    all_df = pd.DataFrame(all_metrics_rows)
    all_df.to_csv(os.path.join(exp_dir, "all_metrics.csv"), index=False)
    pd.DataFrame(univariate_rows).to_csv(os.path.join(exp_dir, "univariate_all_tasks.csv"), index=False)
    pd.DataFrame(multivariate_rows).to_csv(os.path.join(exp_dir, "multivariate_all_tasks.csv"), index=False)
    for task in task_targets:
        pd.DataFrame([r for r in univariate_rows if r["task"] == task]).to_csv(
            os.path.join(exp_dir, f"univariate_{task}.csv"), index=False
        )
        pd.DataFrame([r for r in multivariate_rows if r["task"] == task]).to_csv(
            os.path.join(exp_dir, f"multivariate_{task}.csv"), index=False
        )
        plot_task_roc(exp_dir, task, fold_pred_cache[task])

        test_df = all_df[(all_df["task"] == task) & (all_df["split"] == "test")]
        summary_rows.append({
            "task": task,
            "AUC_mean": float(test_df["AUC"].mean()),
            "AUC_std": float(test_df["AUC"].std(ddof=0)),
            "ACC_mean": float(test_df["ACC"].mean()),
            "F1_mean": float(test_df["F1"].mean()),
            "Sensitivity_mean": float(test_df["Sensitivity"].mean()),
            "Specificity_mean": float(test_df["Specificity"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(exp_dir, "summary_mean_std.csv"), index=False)

    report = {
        "exp_dir": exp_dir,
        "excel_path": args.excel_path,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "tasks": summary_rows,
    }
    with open(os.path.join(exp_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Done] Raw clinical baseline saved to: {exp_dir}")
    for row in summary_rows:
        print(
            f"  {row['task']}: "
            f"AUC={row['AUC_mean']:.4f}±{row['AUC_std']:.4f} "
            f"ACC={row['ACC_mean']:.4f} F1={row['F1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
