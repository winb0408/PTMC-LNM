#!/usr/bin/env python3
"""
roi_radiomics_cv5_central.py

标准患者级 ROI Radiomics 管线（CentralLNM）：
1) 基于 ROI mask 对切片提取 PyRadiomics 特征
2) 患者级 mean/max 聚合
3) 仅在训练集内执行 LASSO 选择
4) 导出 fold 级患者特征文件（供 Tri-source Deep 模型直接读取）
5) 对比 Baseline C: Radiomics-only / Baseline D: Clinical+Radiomics
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from sklearn.ensemble import StackingClassifier
    _HAS_STACKING = True
except Exception:
    _HAS_STACKING = False

SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def parse_args():
    p = argparse.ArgumentParser(description='ROI Radiomics CV5 for CentralLNM')
    p.add_argument('--data_root', type=str, default='/data2/wjb/code/PTMC_LNM/data/MultiModalDataset')
    p.add_argument('--groups_root', type=str, default='groups_cv5_central')
    p.add_argument('--mask_root', type=str, default='/data2/wjb/code/PTMC_LNM/data/masks')
    p.add_argument('--modalities', nargs='+', default=['B'], choices=['B', 'VUSE', 'VCEUS'])
    p.add_argument('--label_dir', type=str, default='CentralLNM')
    p.add_argument('--clinical_agg', type=str, default='mean+max', choices=['mean', 'max', 'mean+max'])
    p.add_argument('--rad_agg', type=str, default='mean+max', choices=['mean', 'max', 'mean+max'])
    p.add_argument('--save_dir', type=str, default='/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/checkpoints_roi_radiomics')
    p.add_argument('--exp_name', type=str, default='central_roi_radiomics_cv5')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lasso_min_keep', type=int, default=8)
    p.add_argument('--lasso_max_keep', type=int, default=30)
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)


def read_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [os.path.splitext(ln.strip())[0] for ln in f if ln.strip()]


def patient_id(name: str) -> str:
    return name.split('_', 1)[0]


def find_image(modality_dir: str, name: str) -> Optional[str]:
    for ext in SUPPORTED_EXTS:
        p = os.path.join(modality_dir, f'{name}{ext}')
        if os.path.exists(p):
            return p
    return None


def resolve_modality_dir(data_root: str, modality: str) -> str:
    """
    ROI radiomics 优先使用与原始 mask 对齐的原始图像目录。
    当前已确认：
      - B 原图位于 data/imgs，与 data/masks 尺寸一致
    其余模态若未来启用，再按需要补充原始对齐路径。
    """
    parent = os.path.dirname(data_root.rstrip('/'))
    if modality == 'B':
        return os.path.join(parent, 'imgs')
    return os.path.join(data_root, modality)


def load_label(data_root: str, label_dir: str, name: str) -> Optional[int]:
    p = os.path.join(data_root, label_dir, f'{name}.txt')
    if not os.path.exists(p):
        return None
    try:
        y = int(open(p, 'r', encoding='utf-8').read().strip())
        return y if y in (0, 1) else None
    except Exception:
        return None


def build_flag_mapping(flag_path: str) -> Dict[str, int]:
    mapping = {}
    with open(flag_path, 'r', encoding='utf-8') as f:
        for i, ln in enumerate(f):
            s = ln.strip()
            if not s:
                continue
            stem = os.path.splitext(s)[0]
            if stem not in mapping:
                mapping[stem] = i
    return mapping


def aggregate_matrix(mat: np.ndarray, mode: str) -> np.ndarray:
    mean_v = mat.mean(axis=0)
    max_v = mat.max(axis=0)
    if mode == 'mean':
        return mean_v
    if mode == 'max':
        return max_v
    return np.concatenate([mean_v, max_v], axis=0)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'AUC': auc,
        'ACC': acc,
        'F1': f1,
        'Precision': prec,
        'Recall': rec,
        'Sensitivity': sen,
        'Specificity': spe,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


def make_radiomics_extractor():
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except Exception as e:
        raise RuntimeError(
            '缺少 PyRadiomics 依赖，请先安装 radiomics 和 SimpleITK'
        ) from e

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        normalize=True,
        removeOutliers=3,
        force2D=True,
    )
    extractor.disableAllFeatures()
    for cls in ['firstorder', 'shape2D', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
        extractor.enableFeatureClassByName(cls)
    extractor.enableImageTypeByName('Original')
    return extractor, sitk


def extract_slice_radiomics(
    img_path: str,
    mask_path: str,
    extractor,
    sitk_mod,
) -> Optional[Dict[str, float]]:
    img = np.asarray(Image.open(img_path).convert('L'), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return None
    if img.shape != mask.shape:
        raise ValueError(f'图像与 mask 尺寸不一致: img={img.shape}, mask={mask.shape}, img_path={img_path}, mask_path={mask_path}')

    image_sitk = sitk_mod.GetImageFromArray(img)
    mask_sitk = sitk_mod.GetImageFromArray(mask)
    result = extractor.execute(image_sitk, mask_sitk)

    feats = {}
    for k, v in result.items():
        if not k.startswith('original_'):
            continue
        try:
            val = float(v)
        except Exception:
            continue
        if np.isfinite(val):
            feats[k] = val
    return feats if feats else None


def precompute_slice_features(args, all_names: List[str]) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], List[str]]:
    extractor, sitk_mod = make_radiomics_extractor()
    cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    feat_names = set()

    for idx, name in enumerate(sorted(set(all_names))):
        mask_path = find_image(args.mask_root, name)
        if mask_path is None:
            continue
        for mod in args.modalities:
            img_dir = resolve_modality_dir(args.data_root, mod)
            img_path = find_image(img_dir, name)
            if img_path is None:
                continue
            feats = extract_slice_radiomics(img_path, mask_path, extractor, sitk_mod)
            if feats is None:
                continue
            cache[(name, mod)] = feats
            feat_names.update(feats.keys())
        if (idx + 1) % 100 == 0:
            print(f'[Radiomics] 已提取 {idx + 1}/{len(all_names)} 张切片')

    feat_names = sorted(feat_names)
    print(f'[Radiomics] 有效切片特征数: {len(cache)} | 单切片特征维度: {len(feat_names)}')
    return cache, feat_names


def build_patient_dataset(
    names: List[str],
    args,
    slice_cache: Dict[Tuple[str, str], Dict[str, float]],
    feat_names: List[str],
    name_to_row: Dict[str, int],
    X_clinical: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    pid_to_rad = defaultdict(list)
    pid_to_rows = defaultdict(list)
    pid_to_y = defaultdict(list)

    for name in names:
        pid = patient_id(name)
        row = name_to_row.get(name)
        y = load_label(args.data_root, args.label_dir, name)
        if row is None or y is None:
            continue

        mod_vecs = []
        for mod in args.modalities:
            feats = slice_cache.get((name, mod))
            if feats is None:
                continue
            mod_vecs.append(np.asarray([feats.get(k, 0.0) for k in feat_names], dtype=np.float32))

        if not mod_vecs:
            continue

        rad_slice = np.concatenate(mod_vecs, axis=0)
        pid_to_rad[pid].append(rad_slice)
        pid_to_rows[pid].append(row)
        pid_to_y[pid].append(y)

    pids = sorted(set(pid_to_rad) & set(pid_to_rows) & set(pid_to_y))
    X_clin, X_rad, y_out = [], [], []
    for pid in pids:
        X_clin.append(aggregate_matrix(X_clinical[pid_to_rows[pid]], args.clinical_agg))
        X_rad.append(aggregate_matrix(np.stack(pid_to_rad[pid], axis=0), args.rad_agg))
        y_out.append(1 if any(v == 1 for v in pid_to_y[pid]) else 0)

    if not pids:
        return [], np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X_clin = np.nan_to_num(np.stack(X_clin).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_rad = np.nan_to_num(np.stack(X_rad).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_out = np.asarray(y_out, dtype=np.int64)
    return pids, X_clin, X_rad, y_out


def fit_lasso_projection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_keep: int,
    max_keep: int,
):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    cv = min(5, int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))
    cv = max(cv, 2)
    lasso = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12),
        cv=cv,
        penalty='l1',
        solver='liblinear',
        scoring='roc_auc',
        max_iter=4000,
        random_state=42,
    )
    lasso.fit(Xs, y_train)

    coef = lasso.coef_.reshape(-1)
    keep = np.flatnonzero(np.abs(coef) > 1e-8)
    if len(keep) < min_keep:
        rank = np.argsort(-np.abs(coef))
        keep = rank[:min(min_keep, X_train.shape[1])]
    if len(keep) > max_keep:
        rank = keep[np.argsort(-np.abs(coef[keep]))]
        keep = rank[:max_keep]

    return scaler, lasso, keep.astype(int)


def transform_selected(X: np.ndarray, scaler: StandardScaler, lasso: LogisticRegressionCV, keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xs = scaler.transform(X)
    X_keep = Xs[:, keep]
    coef = lasso.coef_.reshape(-1)[keep]
    rad_score = X_keep @ coef
    return X_keep.astype(np.float32), rad_score.astype(np.float32)


def build_models(seed: int) -> Dict[str, object]:
    models = {
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=seed),
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=seed)),
        ]),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=seed),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=400, random_state=seed),
    }
    base_estimators = [
        ('lr', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, random_state=seed))])),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=seed)),
        ('svm', Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=seed))])),
    ]
    models['SoftVoting'] = VotingClassifier(estimators=base_estimators, voting='soft')
    if _HAS_STACKING:
        models['Stacking'] = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
            stack_method='predict_proba',
        )
    return models


def fit_predict(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, 'decision_function'):
        s = model.decision_function(X_test)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    pred = model.predict(X_test)
    return pred.astype(np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    flag_path = os.path.join(args.data_root, 'lnm_clinical_flag.txt')
    clin_path = os.path.join(args.data_root, 'clinical_features.csv')
    name_to_row = build_flag_mapping(flag_path)
    X_clinical = pd.read_csv(clin_path, header=None).values.astype(np.float32)

    all_names = []
    fold_splits = {}
    for fold in range(args.folds):
        fold_dir = os.path.join(args.data_root, args.groups_root, f'fold_{fold}')
        split_map = {}
        for split in ['train', 'val', 'test']:
            names = read_list(os.path.join(fold_dir, f'{split}.txt'))
            split_map[split] = names
            all_names.extend(names)
        fold_splits[fold] = split_map

    slice_cache, feat_names = precompute_slice_features(args, all_names)
    pd.DataFrame({'feature_name': feat_names}).to_csv(os.path.join(exp_dir, 'slice_feature_names.csv'), index=False)

    all_rows = []
    summary_rows = []

    for fold in range(args.folds):
        print(f'\n{"=" * 70}\n[Fold {fold}] 构建患者级 radiomics\n{"=" * 70}')
        fold_out = os.path.join(exp_dir, f'fold_{fold}')
        os.makedirs(fold_out, exist_ok=True)

        split_data = {}
        for split in ['train', 'val', 'test']:
            pids, Xc, Xr, y = build_patient_dataset(
                fold_splits[fold][split], args, slice_cache, feat_names, name_to_row, X_clinical
            )
            split_data[split] = {'pid': pids, 'Xc': Xc, 'Xr': Xr, 'y': y}

        scaler, lasso, keep = fit_lasso_projection(
            split_data['train']['Xr'],
            split_data['train']['y'],
            args.lasso_min_keep,
            args.lasso_max_keep,
        )

        selected_cols = [f'rad_{i:03d}' for i in range(len(keep))]
        patient_rows = []
        fold_selected = {
            'n_raw_features': int(split_data['train']['Xr'].shape[1]),
            'n_selected_features': int(len(keep)),
            'selected_indices': keep.tolist(),
        }

        transformed = {}
        for split in ['train', 'val', 'test']:
            X_keep, rad_score = transform_selected(split_data[split]['Xr'], scaler, lasso, keep)
            transformed[split] = {'X_keep': X_keep, 'rad_score': rad_score}
            for pid, y, rs, vec in zip(split_data[split]['pid'], split_data[split]['y'], rad_score, X_keep):
                row = {'pid': pid, 'split': split, 'y': int(y), 'rad_score': float(rs)}
                for i, v in enumerate(vec):
                    row[selected_cols[i]] = float(v)
                patient_rows.append(row)

        pd.DataFrame(patient_rows).to_csv(os.path.join(fold_out, 'patient_features_selected.csv'), index=False)
        with open(os.path.join(fold_out, 'selected_features.json'), 'w', encoding='utf-8') as f:
            json.dump(fold_selected, f, indent=2, ensure_ascii=False)

        X_train_rad = transformed['train']['X_keep']
        X_test_rad = transformed['test']['X_keep']
        X_train_cd = np.concatenate([split_data['train']['Xc'], X_train_rad], axis=1)
        X_test_cd = np.concatenate([split_data['test']['Xc'], X_test_rad], axis=1)
        y_train = split_data['train']['y']
        y_test = split_data['test']['y']

        for pipeline_name, Xtr, Xte in [
            ('radiomics_only', X_train_rad, X_test_rad),
            ('clinical_radiomics', X_train_cd, X_test_cd),
        ]:
            models = build_models(args.seed + fold)
            for model_name, model in models.items():
                prob = fit_predict(model, Xtr, y_train, Xte)
                met = compute_metrics(y_test, prob)
                row = {
                    'fold': fold,
                    'pipeline': pipeline_name,
                    'model': model_name,
                    **met,
                }
                all_rows.append(row)

        # 记录该折用于深度三源的 radiomics 维度
        summary_rows.append({
            'fold': fold,
            'selected_dim': int(len(keep)),
            'train_patients': int(len(split_data['train']['pid'])),
            'test_patients': int(len(split_data['test']['pid'])),
        })

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(exp_dir, 'all_metrics.csv'), index=False)

    summary_df = all_df.groupby(['pipeline', 'model']).agg({
        'AUC': ['mean', 'std'],
        'ACC': 'mean',
        'F1': 'mean',
        'Sensitivity': 'mean',
        'Specificity': 'mean',
    }).reset_index()
    summary_df.columns = ['pipeline', 'model', 'AUC_mean', 'AUC_std', 'ACC_mean', 'F1_mean', 'Sensitivity_mean', 'Specificity_mean']
    summary_df = summary_df.sort_values(['pipeline', 'AUC_mean'], ascending=[True, False])
    summary_df.to_csv(os.path.join(exp_dir, 'summary_mean_std.csv'), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(exp_dir, 'fold_selected_dims.csv'), index=False)

    best_by_pipeline = summary_df.groupby('pipeline').head(1).to_dict(orient='records')
    report = {
        'exp_dir': exp_dir,
        'modalities': args.modalities,
        'best_by_pipeline': best_by_pipeline,
    }
    with open(os.path.join(exp_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'[Done] ROI radiomics 结果已保存: {exp_dir}')
    for item in best_by_pipeline:
        print(
            f'  {item["pipeline"]}: {item["model"]} | '
            f'AUC={item["AUC_mean"]:.4f}±{item["AUC_std"]:.4f}'
        )


if __name__ == '__main__':
    main()
