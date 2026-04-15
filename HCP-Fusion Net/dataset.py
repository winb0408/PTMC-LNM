# =============================================================================
# data/dataset.py
# 多模态医学影像数据集模块
#
# 核心设计要点：
#   1. 三模态图像同步数据增强（共享随机状态，保证空间一致性）
#   2. 多任务标签从多个子目录读取并按图像名对齐
#   3. 临床特征 CSV 已完成预处理（直接读取 → float32 矩阵），
#      通过 lnm_clinical_flag.txt 的行号与图像一一对应
#   4. 根据 opt.modalities 动态加载所需模态，无缝支持消融实验
# =============================================================================

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision import transforms


# =============================================================================
# 一、同步变换类（Synchronized Augmentation）
# =============================================================================

class SynchronizedTransform:
    """
    三模态图像同步数据增强类。

    医学影像中不同模态对应同一解剖位置，因此空间变换（翻转/旋转）
    必须在所有模态间完全同步，否则会破坏模态间的空间对应关系。
    本类通过 Python random 模块在变换前统一采样随机参数，
    确保所有模态接受完全相同的空间变换。

    颜色增强（亮度/对比度）对每个模态独立应用，
    因为不同超声模态的灰度响应特性各不相同。
    """

    def __init__(self, opt, is_train: bool = True):
        self.opt = opt
        self.is_train = is_train
        self.img_size = opt.img_size

        # ── 颜色增强（训练时每模态独立应用）
        self.color_jitter = transforms.ColorJitter(
            brightness=opt.aug_brightness,
            contrast=opt.aug_contrast,
        ) if (is_train and (opt.aug_brightness > 0 or opt.aug_contrast > 0)) else None

        # ── ToTensor + 归一化（所有模态共享相同参数）
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.normalize_mean, std=opt.normalize_std),
        ])

    def __call__(self, imgs: List[Image.Image]) -> List[torch.Tensor]:
        """
        对 PIL Image 列表进行同步增强后返回 Tensor 列表。

        Args:
            imgs: 各模态 PIL Image 列表，长度 = len(opt.modalities)

        Returns:
            各模态归一化 Tensor 列表，每个 shape = [3, H, W]
        """
        # ── Step 1: 所有模态统一 Resize（不依赖随机状态）
        imgs = [TF.resize(img, [self.img_size, self.img_size]) for img in imgs]

        if self.is_train:
            # ── Step 2: 同步随机水平翻转
            #    在所有模态变换前统一采样决策，保证结果一致
            if random.random() < self.opt.aug_hflip_prob:
                imgs = [TF.hflip(img) for img in imgs]

            # ── Step 3: 同步随机垂直翻转
            if random.random() < self.opt.aug_vflip_prob:
                imgs = [TF.vflip(img) for img in imgs]

            # ── Step 4: 同步随机旋转
            #    统一采样旋转角度后分别施加（结果完全一致）
            if self.opt.aug_rotate_deg > 0:
                angle = random.uniform(
                    -self.opt.aug_rotate_deg,
                    self.opt.aug_rotate_deg
                )
                imgs = [TF.rotate(img, angle) for img in imgs]

            # ── Step 5: 每模态独立颜色增强
            if self.color_jitter is not None:
                imgs = [self.color_jitter(img) for img in imgs]

        # ── Step 6: ToTensor + Normalize（统一后处理）
        return [self.to_tensor_norm(img) for img in imgs]


# =============================================================================
# 二、临床特征加载器（已预处理版本）
# =============================================================================

class ClinicalCSVLoader:
    """
    预处理后临床特征 CSV 加载器。

    假设：CSV 文件中的数值特征已完成所有预处理
    （连续变量 Z-score 标准化、离散变量 One-Hot 编码等）。
    本类仅负责：
        1. 读取 CSV → float32 NumPy 矩阵
        2. 支持按列名白名单/黑名单筛选特征列
        3. 对外提供 loader[row_idx] 的随机访问接口

    与图像的对齐完全由外部 flag 文件中的行号决定，
    本类不参与任何对齐逻辑（职责分离）。
    """

    def __init__(
        self,
        csv_path: str,
        feature_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            csv_path:     已预处理的 CSV 文件路径
            feature_cols: 列名白名单（list of str 或 None）。
                          不为 None 时只保留白名单内的列，适合做特征消融。
            drop_cols:    列名黑名单（list of str 或 None）。
                          仅在 feature_cols=None 时生效，用于排除 ID 等无关列。
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'找不到临床特征文件: {csv_path}')

        # 注意：本项目约定 clinical_features.csv **没有表头行**，
        # 所以需要显式 header=None，否则第一行数据会被当成列名丢掉。
        df = pd.read_csv(csv_path, header=None)
        print(f'[ClinicalCSV] 原始形状: {df.shape[0]} 行 × {df.shape[1]} 列')

        # ── 列筛选：白名单优先，黑名单次之，最后自动保留所有数值列
        if feature_cols is not None:
            # 白名单模式：严格按用户指定的列顺序保留
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f'--clinical_feature_cols 中以下列名在 CSV 中找不到: {missing}\n'
                    f'CSV 现有列名: {df.columns.tolist()}'
                )
            df = df[feature_cols]
            print(f'[ClinicalCSV] 白名单模式：保留 {len(feature_cols)} 列')
        else:
            # 黑名单模式
            if drop_cols:
                existing_drop = [c for c in drop_cols if c in df.columns]
                not_found = [c for c in drop_cols if c not in df.columns]
                if not_found:
                    print(f'[WARNING] --clinical_drop_cols 中找不到列: {not_found}，已跳过。')
                df = df.drop(columns=existing_drop)
                print(f'[ClinicalCSV] 黑名单模式：已删除 {existing_drop}')
            # 自动排除所有非数值列（如字符串 ID、备注等）
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                print(f'[ClinicalCSV] 自动排除非数值列: {non_numeric}')
                df = df.select_dtypes(include=[np.number])

        # ── 缺失值检查与填补
        #    理论上已预处理的 CSV 不含缺失值；若有，用均值填补并警告
        if df.isnull().any().any():
            n_missing = int(df.isnull().sum().sum())
            missing_cols = df.columns[df.isnull().any()].tolist()
            print(
                f'[WARNING] 临床特征矩阵存在 {n_missing} 个缺失值，'
                f'涉及列: {missing_cols}。将使用列均值填补，'
                f'建议在数据预处理阶段消除缺失值。'
            )
            df = df.fillna(df.mean())

        # ── 转换为 float32 矩阵并缓存到内存（全量，按行号索引）
        self.feature_matrix: np.ndarray = df.values.astype(np.float32)
        self.feature_dim: int = self.feature_matrix.shape[1]
        self.n_rows: int = self.feature_matrix.shape[0]
        self.col_names: List[str] = df.columns.tolist()

        print(
            f'[ClinicalCSV] 特征矩阵就绪: {self.n_rows} 行 × {self.feature_dim} 维\n'
            f'              列名: {self.col_names}'
        )

    def __getitem__(self, row_idx: int) -> np.ndarray:
        """
        按行号随机访问特征向量。

        Args:
            row_idx: CSV 行号（0-based），由 flag 文件解析结果提供

        Returns:
            shape [feature_dim] 的 float32 NumPy 数组
        """
        if row_idx < 0 or row_idx >= self.n_rows:
            raise IndexError(
                f'行号 {row_idx} 超出临床特征矩阵范围（共 {self.n_rows} 行）。'
                f'请检查 flag 文件中的行号是否与 CSV 的实际行数匹配。'
            )
        return self.feature_matrix[row_idx]  # 返回 ndarray 视图，零拷贝


# =============================================================================
# 三、Flag 文件解析器
# =============================================================================

class FlagFileParser:
    """
    lnm_clinical_flag.txt 解析器。

    文件格式（固定）：
    ┌────────────────────────────────────────────────────────────────┐
    │  每行一个图像文件名（含扩展名），行号即对应 CSV 的行索引（0-based）│
    │                                                                │
    │  第 0 行: 00044628_1_1_1.jpg  →  CSV 第 0 行（第 0 条临床记录）│
    │  第 1 行: 00044628_1_2_2.jpg  →  CSV 第 1 行                  │
    │  第 2 行: 00054321_2_1_1.jpg  →  CSV 第 2 行                  │
    │  ...                                                           │
    └────────────────────────────────────────────────────────────────┘

    解析结果：
        name_to_row: Dict[str, int]
            key   = 图像名的 stem（不含扩展名），与 train.txt/test.txt 的命名约定一致
            value = 该图像在 flag.txt 中的行号（即 CSV 的对应行索引）

    兼容性：
        - train.txt/test.txt 中的样本名可以带或不带扩展名，均会统一 strip 后匹配
        - flag.txt 中的图像名同样兼容有无扩展名两种写法
    """

    def __init__(self, flag_path: str, sample_names: List[str]):
        """
        Args:
            flag_path:    lnm_clinical_flag.txt 的完整路径
            sample_names: 来自 train.txt 或 test.txt 的样本名列表
                          （带或不带扩展名均可，内部统一 strip）
        """
        self.flag_path = flag_path
        # 统一将 sample_names 规范化为不含扩展名的 stem，保存原始名到 stem 的映射
        self.sample_names = sample_names
        self._raw_to_stem: Dict[str, str] = {
            name: os.path.splitext(name)[0] for name in sample_names
        }
        # stem → CSV 行号（主查询字典）
        self.name_to_row: Dict[str, int] = {}
        self._parse()

    @staticmethod
    def _to_stem(filename: str) -> str:
        """将文件名（有无扩展名均可）转换为不含扩展名的 stem"""
        return os.path.splitext(filename.strip())[0]

    def _parse(self):
        """
        解析 flag.txt：
            遍历每一行，以行号作为 CSV 行索引，以 stem 作为字典 key，
            构建 {stem: csv_row_idx} 映射。
        """
        if not os.path.exists(self.flag_path):
            raise FileNotFoundError(
                f'找不到 flag 文件: "{self.flag_path}"。\n'
                f'请通过 --label_flag_file 指定正确路径，'
                f'或确认文件已放置于 --data_root 目录下。'
            )

        with open(self.flag_path, 'r', encoding='utf-8') as f:
            raw_lines = [ln.strip() for ln in f if ln.strip()]

        if len(raw_lines) == 0:
            raise ValueError(f'flag 文件为空: "{self.flag_path}"')

        # 逐行建立 stem → 行号 的映射
        # 允许 flag.txt 中同一 stem 出现多次（重复行），以最先出现的行号为准
        for row_idx, line in enumerate(raw_lines):
            stem = self._to_stem(line)
            if stem not in self.name_to_row:
                self.name_to_row[stem] = row_idx
            # 重复出现时不覆盖，保留首次映射并给出提示
            elif self.name_to_row[stem] != row_idx:
                pass  # 静默忽略，使用首次出现的行号

        # ── 对齐验证：检查 split.txt 中的样本能否在 flag.txt 中找到映射
        all_stems = set(self._to_stem(n) for n in self.sample_names)
        matched_stems = all_stems & set(self.name_to_row.keys())
        unmatched_stems = all_stems - matched_stems

        print(
            f'[FlagParser] 已解析 flag.txt: 共 {len(raw_lines)} 条记录 '
            f'（{len(self.name_to_row)} 个唯一图像名）\n'
            f'             split 样本命中: {len(matched_stems)}/{len(all_stems)} 个'
        )

        if unmatched_stems:
            examples = sorted(unmatched_stems)[:5]
            raise RuntimeError(
                f'\n[FlagParser] 有 {len(unmatched_stems)} 个样本在 flag.txt 中找不到对应记录！\n'
                f'  未命中示例: {examples}\n'
                f'  请检查：\n'
                f'    1. train.txt/test.txt 中的图像名与 flag.txt 是否拼写一致\n'
                f'    2. flag.txt 是否涵盖了所有图像（train + test 的全集）\n'
                f'    3. 文件名大小写是否匹配（Linux 区分大小写）'
            )

    def get(self, name: str) -> int:
        """
        查询图像名对应的 CSV 行号。

        Args:
            name: 图像名（带或不带扩展名均可）

        Returns:
            CSV 行号（0-based），即 ClinicalCSVLoader 的索引参数

        Raises:
            KeyError: 若该图像名在 flag.txt 中不存在（正常情况下不应发生，
                      因为 __init__ 中已做全量验证）
        """
        stem = self._to_stem(name)
        if stem not in self.name_to_row:
            raise KeyError(
                f'图像 "{name}" (stem="{stem}") 在 flag.txt 中没有对应记录。'
            )
        return self.name_to_row[stem]



# =============================================================================
# 四、核心 Dataset 类
# =============================================================================

class MultiModalDataset(Dataset):
    """
    多模态医学影像数据集。

    数据目录约定：
    data_root/
    ├── B/                ← B-mode 图像目录
    ├── VUSE/             ← VUSE 图像目录
    ├── VCEUS/            ← VCEUS 图像目录
    ├── LNM/              ← LNM 标签目录（每样本一个同名 .txt，内容为 0 或 1）
    ├── CentralLNM/       ← 中央区转移标签目录
    ├── LateralLNM/       ← 侧方转移标签目录
    ├── groups/
    │   ├── train.txt     ← 训练集图像名列表（每行一个，不含扩展名）
    │   └── test.txt      ← 测试集图像名列表
    ├── lnm_clinical_flag.txt  ← 图像名 → CSV 行号映射
    └── clinical_features.csv  ← 已预处理完毕的临床特征（每行对应一个临床记录）
    """

    SUPPORTED_IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def __init__(
        self,
        opt,
        split: str = 'train',
        clinical_loader: Optional[ClinicalCSVLoader] = None,
        is_train_override: Optional[bool] = None,
    ):
        """
        Args:
            opt:             实验配置对象（来自 Options.parse()）
            split:           'train' 或 'test'
            clinical_loader: 已构建好的 ClinicalCSVLoader 实例。
                             训练集与测试集共享同一个 loader——
                             因为 CSV 是全量文件，行号由 flag 文件决定，
                             无需区分集合，也不存在数据泄漏问题。
        """
        super().__init__()
        self.opt = opt
        self.split = split
        self.is_train = (split == 'train') if (is_train_override is None) else bool(is_train_override)
        self.modalities = opt.modalities

        # ── 同步数据增强
        self.transform = SynchronizedTransform(opt, is_train=self.is_train)

        # ── 读取样本名列表
        list_path = os.path.join(opt.groups_path, f'{split}.txt')
        self.sample_names: List[str] = self._read_name_list(list_path)

        # ── 解析 flag 文件（图像名 → CSV 行号的映射）
        flag_path = os.path.join(opt.data_root, opt.label_flag_file)
        self.flag_parser = FlagFileParser(flag_path, self.sample_names)

        # ── 临床特征 loader
        self.use_clinical = opt.use_clinical and (clinical_loader is not None)
        self.clinical_loader = clinical_loader

        # ── 一次性将所有标签加载至内存（避免训练时高频 I/O）
        self.labels: Dict[str, List[Optional[int]]] = self._load_all_labels()

        # ── 过滤标签缺失的样本，构建有效索引列表
        self.valid_indices: List[int] = self._build_valid_indices()

        # ── 打印统计信息
        self._print_stats()

    # ─────────────────────────── 初始化辅助方法 ─────────────────────────────

    def _read_name_list(self, path: str) -> List[str]:
        """
        读取样本名列表文件。

        兼容两种写法：
        - 每行是不带扩展名的样本名，例如：00044628_1_1_1
        - 每行是带扩展名的文件名，例如：00044628_1_1_1.jpg

        最终统一转换为「不带扩展名」的样本名，方便与图像/标签文件对齐。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'找不到数据划分文件: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            names: List[str] = []
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                # 去掉可能存在的扩展名（如 .jpg/.png），仅保留 stem
                stem = os.path.splitext(s)[0]
                names.append(stem)
            return names

    def _load_single_label(self, label_dir: str, name: str) -> Optional[int]:
        """
        从标签子目录读取单个样本的二分类标签。
        约定：data_root/{label_dir}/{name}.txt，内容为单个整数 0 或 1。
        文件不存在或内容非法时返回 None。
        """
        label_path = os.path.join(self.opt.data_root, label_dir, f'{name}.txt')
        if not os.path.exists(label_path):
            return None
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            label = int(content)
            if label not in (0, 1):
                print(f'[WARNING] 标签值非法（应为 0/1）: {label_path} → "{content}"')
                return None
            return label
        except (ValueError, IOError) as e:
            print(f'[WARNING] 标签文件读取失败: {label_path}  ({e})')
            return None

    def _load_all_labels(self) -> Dict[str, List[Optional[int]]]:
        """
        一次性将所有样本、所有任务的标签加载进内存。
        返回：{sample_name: [label_task0, label_task1, label_task2]}
        缺失标签用 None 表示，__getitem__ 中将转换为 -1。
        """
        return {
            name: [
                self._load_single_label(label_dir, name)
                for label_dir in self.opt.label_dirs
            ]
            for name in self.sample_names
        }

    def _build_valid_indices(self) -> List[int]:
        """
        根据 task_mode 确定"必要任务"，过滤出所有必要标签均存在的样本索引。
          - single:        仅要求最后一个任务（LateralLNM）有标签
          - multi_*:       要求全部任务均有标签
        """
        n = len(self.opt.tasks)
        if self.opt.task_mode == 'single':
            required_cols = [n - 1]
        else:
            required_cols = list(range(n))

        valid = []
        for i, name in enumerate(self.sample_names):
            lbls = self.labels.get(name, [None] * n)
            if all(j < len(lbls) and lbls[j] is not None for j in required_cols):
                valid.append(i)
        return valid

    def _print_stats(self):
        """打印数据集统计信息（样本数、各任务标签分布）"""
        n_valid = len(self.valid_indices)
        n_total = len(self.sample_names)
        clin_str = f'(临床维度={self.clinical_loader.feature_dim})' if self.use_clinical else '(无临床特征)'
        print(
            f'[Dataset:{self.split}] 有效={n_valid}/{n_total} 样本 | '
            f'模态={self.modalities} {clin_str}'
        )
        for col_i, task_name in enumerate(self.opt.tasks):
            pos = sum(
                1 for i in self.valid_indices
                if self.labels[self.sample_names[i]][col_i] == 1
            )
            neg = sum(
                1 for i in self.valid_indices
                if self.labels[self.sample_names[i]][col_i] == 0
            )
            ratio = pos / (pos + neg) if (pos + neg) > 0 else 0
            print(f'  [{task_name:15s}] 正={pos:4d}  负={neg:4d}  阳性率={ratio:.3f}')

    # ─────────────────────────── 图像加载 ───────────────────────────────────

    def _find_image_path(self, modality: str, name: str) -> str:
        """在指定模态目录下枚举扩展名查找图像文件"""
        modality_dir = os.path.join(self.opt.data_root, modality)
        for ext in self.SUPPORTED_IMG_EXTS:
            candidate = os.path.join(modality_dir, f'{name}{ext}')
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f'在目录 "{modality_dir}" 中找不到样本 "{name}" 的图像。'
            f'已尝试扩展名: {self.SUPPORTED_IMG_EXTS}'
        )

    # ─────────────────────────── Dataset 接口 ───────────────────────────────

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回单个样本的数据字典：

        {
            'images':   Dict[str, Tensor[3,H,W]]  (late fusion 时)
                     或 Tensor[M*3, H, W]          (early fusion 时)
            'clinical': Tensor[D]                  (已预处理的临床特征向量)
                                                   (use_clinical=False 时为零标量占位)
            'labels':   Tensor[N_tasks]            (各任务标签；缺失处填 -1)
            'name':     str                        (样本名，推理/调试用)
        }
        """
        real_idx = self.valid_indices[idx]
        name = self.sample_names[real_idx]

        # ── 1. 加载所有模态图像（统一转为 RGB）
        pil_images = [
            Image.open(self._find_image_path(mod, name)).convert('RGB')
            for mod in self.modalities
        ]

        # ── 2. 同步数据增强
        tensor_list = self.transform(pil_images)  # List[Tensor[3,H,W]]

        # ── 3. 组织图像格式（根据 fusion_type）
        if self.opt.fusion_type == 'early':
            # 通道维度拼接：[M*3, H, W]
            images = torch.cat(tensor_list, dim=0)
        else:
            # 模态字典：{modality_name: Tensor[3,H,W]}
            images = {mod: tensor_list[i] for i, mod in enumerate(self.modalities)}

        # ── 4. 获取临床特征
        #      flag_parser.get(name) 通过 stem 匹配，返回该图像在 flag.txt 中的行号。
        #      行号即 CSV 行索引（0-based），与顺序完全对应。
        if self.use_clinical and self.clinical_loader is not None:
            row_idx = self.flag_parser.get(name)               # flag.txt 行号 = CSV 行索引
            clinical = torch.from_numpy(
                self.clinical_loader[row_idx].copy()            # .copy() 避免持有 ndarray 引用
            )
        else:
            clinical = torch.zeros(1, dtype=torch.float32)     # 占位，不参与计算

        # ── 5. 组装多任务标签
        labels = torch.tensor(
            [lbl if lbl is not None else -1 for lbl in self.labels[name]],
            dtype=torch.float32
        )

        return {
            'images':   images,
            'clinical': clinical,
            'labels':   labels,
            'name':     name,
        }

    # ─────────────────────────── 辅助工具 ───────────────────────────────────

    def get_label_distribution(self) -> Dict[str, Dict[int, int]]:
        """返回各任务正负样本计数，用于计算 pos_weight 和加权采样权重"""
        distribution: Dict[str, Dict[int, int]] = {}
        for col_i, task_name in enumerate(self.opt.tasks):
            counts: Dict[int, int] = {0: 0, 1: 0}
            for i in self.valid_indices:
                lbl = self.labels[self.sample_names[i]][col_i]
                if lbl is not None:
                    counts[lbl] = counts.get(lbl, 0) + 1
            distribution[task_name] = counts
        return distribution


# =============================================================================
# 五、Late Fusion 专用 collate_fn
# =============================================================================

def collate_fn_late_fusion(batch: List[Dict]) -> Dict:
    """
    Late Fusion 模式下的自定义 collate 函数。
    由于 'images' 字段是 dict 格式，PyTorch 默认 collate 无法正确处理，
    需手动对每个模态的 Tensor 在 batch 维度堆叠。
    """
    modalities = list(batch[0]['images'].keys())
    return {
        'images': {
            mod: torch.stack([item['images'][mod] for item in batch], dim=0)
            for mod in modalities
        },
        'clinical': torch.stack([item['clinical'] for item in batch], dim=0),
        'labels':   torch.stack([item['labels']   for item in batch], dim=0),
        'name':     [item['name'] for item in batch],
    }


# =============================================================================
# 六、DataLoader 工厂函数
# =============================================================================

def build_dataloader(
    opt,
    split: str,
    clinical_loader: Optional[ClinicalCSVLoader] = None,
    use_weighted_sampler: bool = False,
    eval_mode: bool = False,
) -> Tuple[DataLoader, Optional[ClinicalCSVLoader]]:
    """
    构建完整数据流水线的工厂函数。

    设计要点：
    - ClinicalCSVLoader 只需构建一次，训练集和测试集共享同一实例。
      原因：CSV 是全量文件，行号由 flag 文件决定，不存在训练/测试集的区分，
      也不需要 fit/transform（CSV 已预处理完毕）。
    - 第一次调用时（split='train'，clinical_loader=None）自动构建 loader，
      返回该 loader 以便后续传给测试集构建函数复用。

    Args:
        opt:                   实验配置对象
        split:                 'train' 或 'test'
        clinical_loader:       已构建的 ClinicalCSVLoader（首次调用传 None）
        use_weighted_sampler:  训练集是否使用加权随机采样（缓解类别不平衡）

    Returns:
        (loader, clinical_loader)
    """
    # ── Step 1: 若需要临床特征且尚未构建 loader，则在此创建
    if opt.use_clinical and clinical_loader is None:
        clinical_loader = ClinicalCSVLoader(
            csv_path=opt.clinical_csv,
            feature_cols=getattr(opt, 'clinical_feature_cols', None),
            drop_cols=getattr(opt, 'clinical_drop_cols', None),
        )
        # 将特征维度写入 opt，供 build_model() 直接读取，无需手动传参
        opt.clinical_input_dim = clinical_loader.feature_dim
    elif not opt.use_clinical:
        opt.clinical_input_dim = 0

    # ── Step 2: 构建 Dataset
    dataset = MultiModalDataset(
        opt=opt,
        split=split,
        clinical_loader=clinical_loader,
        is_train_override=(False if eval_mode else None),
    )

    # ── Step 3: 可选类别加权采样（仅训练集）
    sampler = None
    if use_weighted_sampler and (split == 'train') and (not eval_mode):
        dist = dataset.get_label_distribution()
        primary_task = opt.tasks[0]   # 用第一个任务（LNM）的分布作为采样依据
        n_pos = dist[primary_task].get(1, 1)
        n_neg = dist[primary_task].get(0, 1)
        n_total = n_pos + n_neg

        # 正/负样本权重：使期望的正负样本比例为 1:1
        sample_weights = [
            (n_total / (2.0 * n_pos))
            if dataset.labels[dataset.sample_names[i]][0] == 1
            else (n_total / (2.0 * n_neg))
            for i in dataset.valid_indices
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(f'[DataLoader:{split}] 加权采样 | 正={n_pos}  负={n_neg}')

    # ── Step 4: collate_fn（Late Fusion 需要特殊处理）
    collate = collate_fn_late_fusion if opt.fusion_type == 'late' else None

    loader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=(split == 'train' and sampler is None),
        sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=(opt.device == 'cuda'),
        drop_last=(split == 'train'),   # 丢弃末尾不完整 batch，避免 BN 单样本报错
        collate_fn=collate,
    )

    print(f'[DataLoader:{split}] {len(loader)} batches × batch_size={opt.batch_size}')
    return loader, clinical_loader


# =============================================================================
# 模块自测（需要实际数据目录）
# =============================================================================
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from options import Options

    opt = Options().parse([
        '--data_root',    './data_root',
        '--modalities',   'B', 'VUSE', 'VCEUS',
        '--fusion_type',  'late',
        '--task_mode',    'multi_cascaded',
        '--use_clinical', 'true',
        '--batch_size',   '4',
        '--exp_name',     'dataset_test',
    ])

    print('[Test] Dataset module imported successfully.')
    print(f'       Modalities:  {opt.modalities}')
    print(f'       Tasks:       {opt.tasks}')
    print('       （实际测试需将 --data_root 指向真实数据目录）')
