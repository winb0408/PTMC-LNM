"""
explain_module_contrib.py

目标：按“模块级别”展示单个病例在三个任务上的概率贡献拆解。

四个模块：
- B 超图像（B）
- VUSE 图像
- VCEUS 图像
- 临床特征（全部 CSV 汇总为一块）

方法：固定一个病例，分别计算：
- 全部信息：B + VUSE + VCEUS + Clinical → p_all
- 去掉某一块信息（置零或均值）：→ p_noX
模块贡献 = p_all - p_noX（正值表示该模块推高风险，负值表示降低风险）。
"""

import os
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

from options import Options
from dataset import build_dataloader
from networks import build_model


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
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


def compute_task_probs(model, batch_dev: Dict) -> Dict[str, float]:
    """返回单个样本在当前三任务上的预测概率（CPU float）。"""
    model.eval()
    with torch.no_grad():
        out = model.forward_with_features(batch_dev)
    probs = out["probs"]

    if model.task_mode == "single":
        return {"lateral": probs["lateral"][0].item()}

    return {
        "lnm": probs["lnm"][0].item(),
        "central": probs["central"][0].item(),
        "lateral": probs["lateral"][0].item(),
    }


def explain_case_module_contrib(opt, model, device: torch.device):
    """
    对一个验证集病例，计算四个模块对三个任务概率的贡献，并画出条形图。
    """
    # 选一个病例（val 集第一个样本）
    val_loader, _ = build_dataloader(
        opt=opt,
        split="val",
        clinical_loader=None,
        use_weighted_sampler=False,
    )
    try:
        batch_cpu = next(iter(val_loader))
    except StopIteration:
        print("[ModuleContrib] 验证集为空，无法进行解释。")
        return

    name = batch_cpu["name"][0] if isinstance(batch_cpu["name"], list) else str(batch_cpu["name"])

    # 只保留第一个样本
    batch_1 = {
        "images": {mod: img[0:1] for mod, img in batch_cpu["images"].items()},
        "clinical": batch_cpu["clinical"][0:1],
        "labels": batch_cpu["labels"][0:1],
        "name": [name],
    }
    batch_dev_full = _move_batch_to_device(batch_1, device)

    # baseline：四块信息全开
    probs_all = compute_task_probs(model, batch_dev_full)
    print(f"[ModuleContrib] 病例 {name} baseline 概率: {probs_all}")

    # 构造 4 种“去掉某一模块”的 batch
    # 1) 去掉 B：将 B 图像置零
    batch_noB = {
        "images": batch_dev_full["images"].copy(),
        "clinical": batch_dev_full["clinical"].clone(),
        "labels": batch_dev_full["labels"].clone(),
        "name": batch_1["name"],
    }
    if "B" in opt.modalities:
        b_img = batch_noB["images"]["B"]
        batch_noB["images"]["B"] = torch.zeros_like(b_img)

    # 2) 去掉 VUSE
    batch_noVUSE = {
        "images": batch_dev_full["images"].copy(),
        "clinical": batch_dev_full["clinical"].clone(),
        "labels": batch_dev_full["labels"].clone(),
        "name": batch_1["name"],
    }
    if "VUSE" in opt.modalities:
        vuse_img = batch_noVUSE["images"]["VUSE"]
        batch_noVUSE["images"]["VUSE"] = torch.zeros_like(vuse_img)

    # 3) 去掉 VCEUS
    batch_noVCEUS = {
        "images": batch_dev_full["images"].copy(),
        "clinical": batch_dev_full["clinical"].clone(),
        "labels": batch_dev_full["labels"].clone(),
        "name": batch_1["name"],
    }
    if "VCEUS" in opt.modalities:
        vceus_img = batch_noVCEUS["images"]["VCEUS"]
        batch_noVCEUS["images"]["VCEUS"] = torch.zeros_like(vceus_img)

    # 4) 去掉临床特征：clinical 置零
    batch_noClin = {
        "images": batch_dev_full["images"].copy(),
        "clinical": torch.zeros_like(batch_dev_full["clinical"]),
        "labels": batch_dev_full["labels"].clone(),
        "name": batch_1["name"],
    }

    # 计算四种 ablation 的概率
    probs_noB = compute_task_probs(model, batch_noB)
    probs_noVUSE = compute_task_probs(model, batch_noVUSE)
    probs_noVCEUS = compute_task_probs(model, batch_noVCEUS)
    probs_noClin = compute_task_probs(model, batch_noClin)

    print(f"[ModuleContrib] 病例 {name} 去掉 B 概率: {probs_noB}")
    print(f"[ModuleContrib] 病例 {name} 去掉 VUSE 概率: {probs_noVUSE}")
    print(f"[ModuleContrib] 病例 {name} 去掉 VCEUS 概率: {probs_noVCEUS}")
    print(f"[ModuleContrib] 病例 {name} 去掉 Clinical 概率: {probs_noClin}")

    # 对每个任务，计算 4 个模块的贡献值 p_all - p_no*
    task_keys = ["lnm", "central", "lateral"] if model.task_mode != "single" else ["lateral"]
    module_names = ["B", "VUSE", "VCEUS", "Clinical"]

    save_dir = os.path.join(opt.exp_dir, "module_contrib")
    os.makedirs(save_dir, exist_ok=True)

    for tk in task_keys:
        p_all = probs_all[tk]
        contribs = {
            "B": p_all - probs_noB[tk],
            "VUSE": p_all - probs_noVUSE[tk],
            "VCEUS": p_all - probs_noVCEUS[tk],
            "Clinical": p_all - probs_noClin[tk],
        }

        # 画条形图，按贡献绝对值排序
        mods_sorted = sorted(module_names, key=lambda m: -abs(contribs[m]))
        vals_sorted = [contribs[m] for m in mods_sorted]

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["red" if v > 0 else "blue" for v in vals_sorted]
        x = np.arange(len(mods_sorted))
        ax.bar(x, vals_sorted, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(mods_sorted)
        ax.set_ylabel("ΔProbability (p_all - p_no_module)")
        ax.set_title(f"Case {name} | Task {tk} | p_all={p_all:.2f}")
        for i, v in enumerate(vals_sorted):
            ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"case-{name}_task-{tk}_modules.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"[ModuleContrib] 已保存病例 {name} 任务 {tk} 的模块贡献图至: {save_path}")


def main():
    opt_manager = Options()
    opt = opt_manager.parse()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"[ModuleContrib] 使用设备: {device}")

    # 为了获得 clinical_input_dim，需要先构建一个 DataLoader
    train_loader, clinical_loader = build_dataloader(
        opt=opt,
        split="train",
        clinical_loader=None,
        use_weighted_sampler=False,
    )
    clinical_input_dim = getattr(opt, "clinical_input_dim", 0)

    model = build_model(opt, clinical_input_dim=clinical_input_dim)
    model = model.to(device)

    best_ckpt_path = os.path.join(opt.exp_dir, "best_model.pth")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"找不到最佳模型权重: {best_ckpt_path}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[ModuleContrib] 已加载最佳模型，来自 Epoch {ckpt['epoch']+1}")

    explain_case_module_contrib(opt, model, device)


if __name__ == "__main__":
    main()

