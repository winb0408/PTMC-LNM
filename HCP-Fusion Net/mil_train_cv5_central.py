#!/usr/bin/env python3
"""
MIL 5-fold (patient-level) training for CentralLNM using AttentionMIL.
Uses groups_cv5_central/fold_k/{train,val,test}.txt
Saves per-fold metrics and patient-level predictions under --save_dir.
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from mil_dataset import CentralMILBags
from mil_models import AttentionMIL


def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def evaluate(model: AttentionMIL, loader, device: torch.device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            bag = batch['bag'].squeeze(0).to(device)  # [N,3,H,W]
            if bag.size(0) > loader.dataset.max_instances:
                bag = bag[:loader.dataset.max_instances]
            y = batch['label'].cpu().numpy().astype(float)
            logit, _, _ = model(bag)
            p = torch.sigmoid(logit).item()
            ys.append(float(y))
            ps.append(p)
    y = np.array(ys, dtype=float)
    p = np.array(ps, dtype=float)
    auc = float('nan') if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
    return auc, y, p


def train_one_fold(args, fold: int, device: torch.device):
    groups_dir = os.path.join(args.groups_root, f'fold_{fold}')
    ds_tr = CentralMILBags(args.data_root, groups_dir, 'train', modality=args.modality, img_size=args.img_size)
    ds_va = CentralMILBags(args.data_root, groups_dir, 'val', modality=args.modality, img_size=args.img_size)
    ds_te = CentralMILBags(args.data_root, groups_dir, 'test', modality=args.modality, img_size=args.img_size)
    ds_tr.max_instances = args.max_instances
    ds_va.max_instances = args.max_instances
    ds_te.max_instances = args.max_instances

    # data loader with batch_size=1 (one bag per batch)
    def make_loader(ds, shuffle: bool):
        return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=2, pin_memory=(device.type=='cuda'))

    tr_loader = make_loader(ds_tr, True)
    va_loader = make_loader(ds_va, False)
    te_loader = make_loader(ds_te, False)

    model = AttentionMIL(backbone=args.backbone, pretrained=True, attn_dim=128, bag_hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tr_loader:
            bag = batch['bag'].squeeze(0).to(device)  # [N,3,H,W]
            if bag.size(0) > args.max_instances:
                idx = torch.randperm(bag.size(0), device=bag.device)[:args.max_instances]
                bag = bag[idx]
            y = batch['label'].to(device)
            logit, _, _ = model(bag)
            loss = bce(logit.view(1), y.view(1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        # val
        val_auc, _, _ = evaluate(model, va_loader, device)
        if val_auc == val_auc and val_auc > best_val:  # not nan and better
            best_val = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        # simple console line (caller captures to logs)
        print(f"[Fold{fold}] Epoch {epoch+1}/{args.epochs} train_loss={np.mean(losses):.4f} val_auc={val_auc:.4f}")

    # load best and test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_auc, y_te, p_te = evaluate(model, te_loader, device)

    out_dir = os.path.join(args.save_dir, f"{args.exp_prefix}_fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "fold": fold,
            "val_auc_best": float(best_val),
            "test_auc": float(test_auc),
            "elapsed_sec": float(time.time()-start),
            "n_train_patients": len(ds_tr),
            "n_val_patients": len(ds_va),
            "n_test_patients": len(ds_te),
        }, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(out_dir, "test_y.npy"), y_te)
    np.save(os.path.join(out_dir, "test_p.npy"), p_te)
    return best_val, test_auc


def main():
    ap = argparse.ArgumentParser("MIL CentralLNM 5-fold")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--groups_root", type=str, default="groups_cv5_central")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--modality", type=str, default="B")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--max_instances", type=int, default=8, help="limit instances per bag to avoid GPU OOM")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--exp_prefix", type=str, default="cvCentral_MIL")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    vals, tests = [], []
    for k in range(args.folds):
        bv, bt = train_one_fold(args, k, device)
        vals.append(bv); tests.append(bt)
    summary = {
        "val_auc_mean": float(np.nanmean(vals)),
        "val_auc_std":  float(np.nanstd(vals)),
        "test_auc_mean": float(np.nanmean(tests)),
        "test_auc_std":  float(np.nanstd(tests)),
        "folds": int(args.folds),
    }
    with open(os.path.join(args.save_dir, f"{args.exp_prefix}_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[MIL] Summary:", summary)

if __name__ == "__main__":
    main()

