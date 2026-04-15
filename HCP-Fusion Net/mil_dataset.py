import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

SUPPORTED_IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


def _read_list(path: str) -> List[str]:
    out: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            stem, _ = os.path.splitext(s)
            out.append(stem)
    return out


def _find_image(mod_dir: str, name: str) -> Optional[str]:
    for ext in SUPPORTED_IMG_EXTS:
        p = os.path.join(mod_dir, f'{name}{ext}')
        if os.path.exists(p):
            return p
    return None


def _patient_id(name: str) -> str:
    return name.split('_', 1)[0]


class CentralMILBags(Dataset):
    """
    Build patient-level bags for CentralLNM:
      - Bag: all slices of the same patient_id from split file
      - Label: 1 if any slice has CentralLNM=1, else 0

    Args:
      data_root: dataset root
      groups_dir: e.g., 'groups_cv5_central/fold_k'
      split: 'train' | 'val' | 'test'
      modality: which image modality folder to use (e.g., 'B')
      img_size: resize to (img_size, img_size)
    """
    def __init__(self, data_root: str, groups_dir: str, split: str, modality: str = 'B', img_size: int = 224):
        super().__init__()
        self.data_root = data_root
        self.groups_dir = groups_dir
        self.split = split
        self.modality = modality
        self.img_size = img_size

        split_file = os.path.join(data_root, groups_dir, f'{split}.txt')
        names = _read_list(split_file)

        # group by patient
        pid_to_names: Dict[str, List[str]] = {}
        for n in names:
            pid = _patient_id(n)
            pid_to_names.setdefault(pid, []).append(n)

        # build labels at patient-level using CentralLNM directory
        lab_dir = os.path.join(data_root, 'CentralLNM')
        def load_lab(name: str) -> int:
            p = os.path.join(lab_dir, f'{name}.txt')
            if not os.path.exists(p):
                return 0
            try:
                v = int(open(p, 'r', encoding='utf-8').read().strip())
                return 1 if v == 1 else 0
            except Exception:
                return 0

        self.pids = sorted(pid_to_names.keys())
        self.pid_to_names = pid_to_names
        self.labels = []
        for pid in self.pids:
            y = 0
            for n in pid_to_names[pid]:
                y = max(y, load_lab(n))
            self.labels.append(y)

        self.mod_dir = os.path.join(data_root, modality)

    def __len__(self) -> int:
        return len(self.pids)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB').resize((self.img_size, self.img_size))
        import numpy as np
        arr = np.asarray(img, dtype='float32') / 255.0  # [H,W,3]
        t = torch.from_numpy(arr).permute(2,0,1)        # [3,H,W]
        # normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        t = (t - mean) / std
        return t

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx]
        names = self.pid_to_names[pid]
        imgs: List[torch.Tensor] = []
        for n in names:
            p = _find_image(self.mod_dir, n)
            if p is None:
                continue
            imgs.append(self._load_image_tensor(p))
        if len(imgs) == 0:
            # create a dummy instance to avoid crash; mark negative bag
            imgs = [torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)]
        bag = torch.stack(imgs, dim=0)  # [N, 3, H, W]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {'pid': pid, 'bag': bag, 'label': y}

