# =============================================================================
# v2_mil_gated/dataset.py
# 患者级多模态 Bag DataLoader
#
# 核心设计：
#   - 每次 __getitem__ 返回一位患者的完整 Bag：
#       bag_images: [N_slices, M*3, H, W]   (多模态通道拼接)
#       clinical:   [D_clin]                 (患者级聚合特征)
#       y_lnm:      scalar                   (患者级 LNM 标签)
#       y_central:  scalar                   (患者级 CentralLNM 标签)
#   - 由 collate_fn 处理变长 bag（padding + mask）
# =============================================================================

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


# ─────────────────── 辅助函数 ───────────────────

def _read_list(path: str) -> List[str]:
    """读取 split 文件，返回不含扩展名的样本名列表"""
    with open(path, 'r', encoding='utf-8') as f:
        return [os.path.splitext(ln.strip())[0] for ln in f if ln.strip()]


def _patient_id(name: str) -> str:
    """从样本名提取患者 ID（以首个 '_' 为分隔符）"""
    return name.split('_', 1)[0]


def _find_image(mod_dir: str, name: str) -> Optional[str]:
    """在模态目录下搜索图像文件"""
    for ext in SUPPORTED_EXTS:
        p = os.path.join(mod_dir, f'{name}{ext}')
        if os.path.exists(p):
            return p
    return None


def _load_label(data_root: str, label_dir: str, name: str) -> Optional[int]:
    """加载单切面标签（0/1），缺失返回 None"""
    p = os.path.join(data_root, label_dir, f'{name}.txt')
    if not os.path.exists(p):
        return None
    try:
        v = int(open(p, 'r', encoding='utf-8').read().strip())
        return v if v in (0, 1) else None
    except Exception:
        return None


def _build_flag_mapping(flag_path: str) -> Dict[str, int]:
    """解析 lnm_clinical_flag.txt → {stem: csv_row_idx}"""
    mapping: Dict[str, int] = {}
    with open(flag_path, 'r', encoding='utf-8') as f:
        for i, ln in enumerate(f):
            s = ln.strip()
            if not s:
                continue
            stem = os.path.splitext(s)[0]
            if stem not in mapping:
                mapping[stem] = i
    return mapping


def _load_patient_radiomics_map(
    radiomics_root: str,
    groups_dir: str,
    topk: int = 0,
) -> Dict[str, np.ndarray]:
    """
    从 ROI radiomics 结果目录读取患者级特征。
    约定文件位于:
      <radiomics_root>/fold_k/patient_features_selected.csv
    且包含列: pid, split, y, rad_score, ...
    """
    if not radiomics_root:
        return {}

    fold_name = os.path.basename(groups_dir.rstrip('/'))
    csv_path = os.path.join(radiomics_root, fold_name, 'patient_features_selected.csv')
    if not os.path.exists(csv_path):
        print(f'[Radiomics] 未找到特征文件: {csv_path}')
        return {}

    df = pd.read_csv(csv_path)
    if 'pid' not in df.columns:
        raise ValueError(f'Radiomics 文件缺少 pid 列: {csv_path}')

    drop_cols = {'pid', 'split', 'y'}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    if topk and topk > 0:
        feat_cols = feat_cols[:topk]
    if not feat_cols:
        raise ValueError(f'Radiomics 文件中没有可用特征列: {csv_path}')

    mapping: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        pid = str(row['pid'])
        mapping[pid] = row[feat_cols].to_numpy(dtype=np.float32)

    print(f'[Radiomics] 已加载 {len(mapping)} 位患者特征 | dim={len(feat_cols)} | topk={topk} | {csv_path}')
    return mapping


# ─────────────────── 核心 Dataset ───────────────────

class PatientBagDataset(Dataset):
    """
    患者级多模态 Bag 数据集。

    每位患者作为一个 Bag，其所有切面构成多个 Instance。
    支持多模态通道拼接（B + VUSE + VCEUS → [N, 9, H, W]）和临床特征聚合。
    """

    def __init__(
        self,
        data_root: str,
        groups_dir: str,       # 例如 'groups_cv5_central/fold_0'
        split: str,            # 'train' | 'val' | 'test'
        modalities: List[str],
        img_size: int = 224,
        max_instances: int = 12,
        clinical_agg: str = 'mean+max',
        radiomics_root: str = '',
        radiomics_topk: int = 0,
        is_train: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.radiomics_root = radiomics_root
        self.modalities = modalities
        self.img_size = img_size
        self.max_instances = max_instances
        self.clinical_agg = clinical_agg
        self.is_train = is_train
        self.radiomics_topk = radiomics_topk

        # ── 读取 split 样本名 ──
        split_path = os.path.join(data_root, groups_dir, f'{split}.txt')
        all_names = _read_list(split_path)

        # ── 按患者分组 ──
        pid_to_names: Dict[str, List[str]] = defaultdict(list)
        for n in all_names:
            pid_to_names[_patient_id(n)].append(n)
        self.pids = sorted(pid_to_names.keys())
        self.pid_to_names = pid_to_names

        # ── 患者级标签（任一切面为 1 → 患者为 1）──
        self.y_lnm: Dict[str, int] = {}
        self.y_central: Dict[str, int] = {}
        for pid, names in pid_to_names.items():
            lnm_vals = [_load_label(data_root, 'LNM', n) for n in names]
            cen_vals = [_load_label(data_root, 'CentralLNM', n) for n in names]
            self.y_lnm[pid] = 1 if any(v == 1 for v in lnm_vals if v is not None) else 0
            self.y_central[pid] = 1 if any(v == 1 for v in cen_vals if v is not None) else 0

        # ── 临床特征 ──
        flag_path = os.path.join(data_root, 'lnm_clinical_flag.txt')
        self.name_to_row = _build_flag_mapping(flag_path)
        csv_path = os.path.join(data_root, 'clinical_features.csv')
        self.X_clin = pd.read_csv(csv_path, header=None).values.astype(np.float32)

        # ── 预计算患者级临床特征 ──
        self.patient_clinical: Dict[str, np.ndarray] = {}
        for pid, names in pid_to_names.items():
            rows = [self.name_to_row[n] for n in names if n in self.name_to_row]
            if not rows:
                # 回退: 全零
                dim = self.X_clin.shape[1]
                self.patient_clinical[pid] = np.zeros(
                    dim * 2 if clinical_agg == 'mean+max' else dim,
                    dtype=np.float32,
                )
                continue
            mat = self.X_clin[rows]
            f_mean = mat.mean(axis=0)
            f_max = mat.max(axis=0)
            if clinical_agg == 'mean':
                self.patient_clinical[pid] = f_mean
            elif clinical_agg == 'max':
                self.patient_clinical[pid] = f_max
            else:  # mean+max
                self.patient_clinical[pid] = np.concatenate([f_mean, f_max])

        # ── 患者级 ROI radiomics ──
        self.patient_radiomics: Dict[str, np.ndarray] = {}
        self.radiomics_dim = 0
        if getattr(self, 'pids', None) and hasattr(self, 'pid_to_names'):
            rad_map = _load_patient_radiomics_map(
                getattr(self, 'radiomics_root', ''),
                groups_dir,
                getattr(self, 'radiomics_topk', 0),
            ) if hasattr(self, 'radiomics_root') else {}
            for pid in self.pids:
                if pid in rad_map:
                    self.patient_radiomics[pid] = rad_map[pid]
            if self.patient_radiomics:
                first_pid = next(iter(self.patient_radiomics))
                self.radiomics_dim = int(self.patient_radiomics[first_pid].shape[0])

        # ── 图像预处理 ──
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

        # ── 打印统计 ──
        n_pos_lnm = sum(self.y_lnm.values())
        n_pos_cen = sum(self.y_central.values())
        print(f'[PatientBag:{split}] {len(self.pids)} patients | '
              f'LNM+ {n_pos_lnm}/{len(self.pids)} | '
              f'Central+ {n_pos_cen}/{len(self.pids)}')

    def __len__(self) -> int:
        return len(self.pids)

    def _load_slice_tensor(self, name: str) -> Optional[torch.Tensor]:
        """
        加载一个切面的多模态图像，通道维度拼接 → [M*3, H, W]
        """
        tensors = []
        for mod in self.modalities:
            mod_dir = os.path.join(self.data_root, mod)
            img_path = _find_image(mod_dir, name)
            if img_path is None:
                return None
            img = Image.open(img_path).convert('RGB')
            tensors.append(self.transform(img))  # [3, H, W]
        return torch.cat(tensors, dim=0)  # [M*3, H, W]

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx]
        names = self.pid_to_names[pid]

        # ── 加载所有切面 ──
        bag_tensors: List[torch.Tensor] = []
        for n in names:
            t = self._load_slice_tensor(n)
            if t is not None:
                bag_tensors.append(t)

        # 兜底：空 bag 用全零占位
        n_channels = len(self.modalities) * 3
        if not bag_tensors:
            bag_tensors = [torch.zeros(n_channels, self.img_size, self.img_size)]

        # ── 截断/采样 ──
        if len(bag_tensors) > self.max_instances:
            if self.is_train:
                bag_tensors = random.sample(bag_tensors, self.max_instances)
            else:
                bag_tensors = bag_tensors[:self.max_instances]

        bag = torch.stack(bag_tensors, dim=0)  # [N, M*3, H, W]

        # ── 临床特征 ──
        clinical = torch.from_numpy(self.patient_clinical[pid].copy())
        if self.radiomics_dim > 0 and pid in self.patient_radiomics:
            radiomics = torch.from_numpy(self.patient_radiomics[pid].copy())
        else:
            radiomics = torch.zeros(self.radiomics_dim, dtype=torch.float32)

        # ── 标签 ──
        y_lnm = torch.tensor(self.y_lnm[pid], dtype=torch.float32)
        y_central = torch.tensor(self.y_central[pid], dtype=torch.float32)

        return {
            'pid': pid,
            'bag': bag,
            'clinical': clinical,
            'radiomics': radiomics,
            'y_lnm': y_lnm,
            'y_central': y_central,
            'n_instances': bag.size(0),
        }


# ─────────────────── Collate (变长 padding) ───────────────────

def collate_bags(batch: List[Dict]) -> Dict:
    """
    将变长 bag 通过 padding 对齐到同一 batch 中。
    返回额外的 mask 张量标记有效实例。
    """
    max_n = max(item['n_instances'] for item in batch)
    B = len(batch)
    C = batch[0]['bag'].size(1)
    H = batch[0]['bag'].size(2)
    W = batch[0]['bag'].size(3)

    bags = torch.zeros(B, max_n, C, H, W)
    mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item['n_instances']
        bags[i, :n] = item['bag']
        mask[i, :n] = True

    return {
        'pid': [item['pid'] for item in batch],
        'bag': bags,           # [B, max_N, C, H, W]
        'mask': mask,          # [B, max_N]  True=有效实例
        'clinical': torch.stack([item['clinical'] for item in batch]),
        'radiomics': torch.stack([item['radiomics'] for item in batch]),
        'y_lnm': torch.stack([item['y_lnm'] for item in batch]),
        'y_central': torch.stack([item['y_central'] for item in batch]),
    }


# ─────────────────── DataLoader 工厂 ───────────────────

def build_loader(
    data_root: str,
    groups_dir: str,
    split: str,
    opt,
    is_train: bool = False,
) -> DataLoader:
    ds = PatientBagDataset(
        data_root=data_root,
        groups_dir=groups_dir,
        split=split,
        modalities=opt.modalities,
        img_size=opt.img_size,
        max_instances=opt.max_instances,
        clinical_agg=opt.clinical_agg,
        radiomics_root=opt.radiomics_root if getattr(opt, 'use_radiomics', False) else '',
        radiomics_topk=getattr(opt, 'radiomics_topk', 0),
        is_train=is_train,
    )
    return DataLoader(
        ds,
        batch_size=4 if is_train else 8,
        shuffle=is_train,
        num_workers=opt.num_workers,
        pin_memory=(opt.device == 'cuda'),
        collate_fn=collate_bags,
        drop_last=(is_train and len(ds) > 4),
    )
