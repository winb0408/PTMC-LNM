"""
split_stats.py

统计 train/val/test 三划分的：
- 样本数（原始与有效样本数）
- 各任务阳性率（基于 label_dirs 中的标签文件）
- 同名样本是否跨 split 重复
- 疑似“患者泄漏”：按规则取患者ID = 文件名第一个下划线前的数字串，
  检查同一患者是否同时出现在不同 split 中。

用法（在 HCP-Fusion Net 目录下）：

python split_stats.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --groups_dir groups \
  --tasks LNM CentralLNM LateralLNM \
  --label_dirs LNM CentralLNM LateralLNM

说明：
- 会读取 data_root/groups/{train,val,test}.txt（每行样本名，可带 .jpg 后缀）
- 会读取 data_root/{label_dir}/{name}.txt（0/1）作为标签
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from options import Options


def read_split_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到划分文件: {path}")
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            stem, _ = os.path.splitext(s)
            names.append(stem)
    return names


def patient_id_from_name(name: str) -> str:
    """患者ID = 第一个下划线前的字符串（若无下划线则用整个 name）。"""
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


@dataclass
class TaskStats:
    pos: int = 0
    neg: int = 0
    missing: int = 0

    def total_labeled(self) -> int:
        return self.pos + self.neg

    def pos_rate(self) -> float:
        t = self.total_labeled()
        return (self.pos / t) if t > 0 else 0.0


def compute_split_stats(
    split_name: str,
    names: List[str],
    data_root: str,
    tasks: List[str],
    label_dirs: List[str],
    require_all_tasks: bool = True,
) -> Tuple[Dict[str, TaskStats], int, int]:
    """
    Returns:
      task_stats: per-task pos/neg/missing
      n_total: raw split size
      n_valid: number of samples with required labels (all tasks by default)
    """
    task_stats: Dict[str, TaskStats] = {t: TaskStats() for t in tasks}

    n_total = len(names)
    n_valid = 0

    for name in names:
        labels: List[Optional[int]] = []
        for task_name, label_dir in zip(tasks, label_dirs):
            y = load_label(data_root, label_dir, name)
            labels.append(y)
            if y is None:
                task_stats[task_name].missing += 1
            elif y == 1:
                task_stats[task_name].pos += 1
            else:
                task_stats[task_name].neg += 1

        # valid：要求所有任务都有标签（更严格，便于发现漏标）
        if require_all_tasks:
            if all(y is not None for y in labels):
                n_valid += 1
        else:
            # 至少一个任务有标签
            if any(y is not None for y in labels):
                n_valid += 1

    return task_stats, n_total, n_valid


def find_overlap(a: List[str], b: List[str]) -> Set[str]:
    return set(a) & set(b)


def main():
    opt = Options().parse()

    data_root = opt.data_root
    groups_path = os.path.join(data_root, opt.groups_dir)
    tasks: List[str] = list(opt.tasks)
    label_dirs: List[str] = list(opt.label_dirs)

    if len(tasks) != len(label_dirs):
        raise ValueError("--tasks 与 --label_dirs 数量必须一致")

    splits = {}
    for s in ["train", "val", "test"]:
        p = os.path.join(groups_path, f"{s}.txt")
        splits[s] = read_split_list(p)

    # 1) 同名样本跨 split 重复
    overlap_tv = find_overlap(splits["train"], splits["val"])
    overlap_tt = find_overlap(splits["train"], splits["test"])
    overlap_vt = find_overlap(splits["val"], splits["test"])

    # 2) 患者泄漏：同一 patient_id 出现在不同 split
    pid_sets = {s: set(patient_id_from_name(n) for n in names) for s, names in splits.items()}
    pid_tv = pid_sets["train"] & pid_sets["val"]
    pid_tt = pid_sets["train"] & pid_sets["test"]
    pid_vt = pid_sets["val"] & pid_sets["test"]

    print("\n" + "=" * 70)
    print("Split 基本信息")
    print("=" * 70)
    for s in ["train", "val", "test"]:
        print(f"- {s:<5s}: {len(splits[s])} 样本 | {len(set(splits[s]))} 唯一名称 | {len(pid_sets[s])} 患者")

    print("\n" + "=" * 70)
    print("同名样本跨 split 重复检查（应为 0）")
    print("=" * 70)
    print(f"train ∩ val : {len(overlap_tv)}")
    print(f"train ∩ test: {len(overlap_tt)}")
    print(f"val   ∩ test: {len(overlap_vt)}")
    if overlap_tv or overlap_tt or overlap_vt:
        ex = sorted(list(overlap_tv | overlap_tt | overlap_vt))[:10]
        print(f"示例（前 10 个）: {ex}")

    print("\n" + "=" * 70)
    print("疑似患者泄漏检查（按患者ID，理想应为 0）")
    print("=" * 70)
    print(f"patient(train) ∩ patient(val) : {len(pid_tv)}")
    print(f"patient(train) ∩ patient(test): {len(pid_tt)}")
    print(f"patient(val)   ∩ patient(test): {len(pid_vt)}")
    if pid_tv or pid_tt or pid_vt:
        ex = sorted(list(pid_tv | pid_tt | pid_vt))[:10]
        print(f"泄漏患者ID示例（前 10 个）: {ex}")

    # 3) 各 split 任务阳性率
    print("\n" + "=" * 70)
    print("各 split 的标签分布（按任务）")
    print("=" * 70)
    for s in ["train", "val", "test"]:
        task_stats, n_total, n_valid = compute_split_stats(
            split_name=s,
            names=splits[s],
            data_root=data_root,
            tasks=tasks,
            label_dirs=label_dirs,
            require_all_tasks=True,
        )
        print(f"\n[{s}] raw={n_total} | valid(all tasks labeled)={n_valid}")
        for t in tasks:
            st = task_stats[t]
            print(
                f"  - {t:<12s}: pos={st.pos:4d} neg={st.neg:4d} "
                f"pos_rate={st.pos_rate():.3f} missing={st.missing:4d}"
            )

    print("\n完成。")


if __name__ == "__main__":
    main()

