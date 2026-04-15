import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from types import SimpleNamespace

from dataset import build_dataloader
from networks import build_model


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


def patient_id(name: str) -> str:
    return str(name).split("_", 1)[0]


def infer_patient_probs(exp_dir: str, device_str: str = "cuda"):
    opt = load_opt(exp_dir)
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    # set clinical_input_dim if needed
    _tmp, _ = build_dataloader(opt=opt, split="train", clinical_loader=None, use_weighted_sampler=False, eval_mode=True)
    clinical_input_dim = getattr(opt, "clinical_input_dim", 0)

    model = build_model(opt, clinical_input_dim=clinical_input_dim).to(device)
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader, _ = build_dataloader(
        opt=opt,
        split="test",
        clinical_loader=None,
        use_weighted_sampler=False,
        eval_mode=True,
    )

    # use patient-mean probability
    pid_to_probs = {}
    pid_to_labels = {}

    with torch.no_grad():
        for batch in test_loader:
            batch_dev = _move_batch_to_device(batch, device)
            preds = model(batch_dev)
            # single task key
            task_key = getattr(model, "single_task_key", None) or list(preds.keys())[0]
            prob = torch.sigmoid(preds[task_key]).detach().cpu().numpy()
            y = batch["labels"][:, 0].detach().cpu().numpy().astype(int)
            names = batch["name"]
            for i in range(len(y)):
                pid = patient_id(names[i])
                pid_to_probs.setdefault(pid, []).append(float(prob[i]))
                pid_to_labels.setdefault(pid, []).append(int(y[i]))

    pids = sorted(pid_to_probs.keys())
    y_true = []
    y_score = []
    for pid in pids:
        y_true.append(1 if max(pid_to_labels[pid]) == 1 else 0)
        y_score.append(float(np.mean(pid_to_probs[pid])))
    return np.array(y_true), np.array(y_score)


def main():
    save_dir = "./checkpoints"
    exp_prefix = "cvCentral_Bonly_r50"
    folds = 5
    out_path = "../figures/roc_overlay_imageonly_b.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6.8, 6.2))
    aucs = []
    for fold in range(folds):
        exp_dir = os.path.join(save_dir, f"{exp_prefix}_fold{fold}")
        y, p = infer_patient_probs(exp_dir, device_str="cuda")
        fpr, tpr, _ = roc_curve(y, p)
        a = auc(fpr, tpr)
        aucs.append(a)
        plt.plot(fpr, tpr, lw=1.8, label=f"Fold {fold} (AUC={a:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Image-only(B) Patient-level ROC (mean AUC={np.mean(aucs):.3f})")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()

