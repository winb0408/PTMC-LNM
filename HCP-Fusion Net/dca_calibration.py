#!/usr/bin/env python3
"""
Compute Decision Curve Analysis (DCA) and Calibration (Brier/ECE, reliability plot)
from patient-level predictions CSV/NPY pairs and write figures to --out_dir.
This script expects y.npy (0/1) and p.npy (prob) arrays or a CSV with columns: y,p.
"""
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = np.mean((y_prob[mask] >= 0.5) == y_true[mask])
        conf = np.mean(y_prob[mask])
        w = np.mean(mask)
        ece += w * abs(acc - conf)
    return float(ece)


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Net benefit curve (simple DCA): NB = TP/N - FP/N * (pt/(1-pt)), where pt is threshold.
    """
    N = len(y_true)
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    out = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        NB = (TP / N) - (FP / N) * (pt / (1-pt + 1e-12))
        out.append(NB)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser("DCA & Calibration")
    ap.add_argument("--y", type=str, required=True, help="path to y.npy or csv with y column")
    ap.add_argument("--p", type=str, default="", help="path to p.npy or csv with p column (if separate)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--label", type=str, default="model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load arrays
    if args.y.endswith(".npy") and args.p.endswith(".npy"):
        y = np.load(args.y)
        p = np.load(args.p)
    else:
        import pandas as pd
        df = pd.read_csv(args.y)
        if 'p' in df.columns and 'y' in df.columns:
            y = df['y'].values
            p = df['p'].values
        else:
            raise RuntimeError("CSV must contain columns y,p or provide npy files for both.")

    # calibration
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y, p)
    ece = ece_score(y, p, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'k--',label='Perfectly calibrated')
    plt.plot(mean_pred, frac_pos, 'o-', label=args.label)
    plt.xlabel('Mean predicted value'); plt.ylabel('Fraction of positives')
    plt.title('Reliability Diagram')
    plt.legend(); plt.tight_layout()
    rel_path = os.path.join(args.out_dir, f"calibration_{args.label}.png")
    plt.savefig(rel_path, dpi=200); plt.close()

    # DCA
    thresholds = np.linspace(0.01, 0.99, 50)
    nb_model = decision_curve(y, p, thresholds)
    nb_all = (np.mean(y) - (1-np.mean(y)) * (thresholds/(1-thresholds+1e-12)))  # treat-all
    nb_none = np.zeros_like(thresholds)  # treat-none
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, nb_model, label=args.label)
    plt.plot(thresholds, nb_all, '--', label='All')
    plt.plot(thresholds, nb_none, '--', label='None')
    plt.xlabel('Threshold probability'); plt.ylabel('Net benefit')
    plt.title('Decision Curve Analysis'); plt.legend(); plt.tight_layout()
    dca_path = os.path.join(args.out_dir, f"dca_{args.label}.png")
    plt.savefig(dca_path, dpi=200); plt.close()

    with open(os.path.join(args.out_dir, f"calibration_dca_{args.label}.json"), 'w', encoding='utf-8') as f:
        json.dump({"brier": float(brier), "ece": float(ece),
                   "calibration_png": rel_path, "dca_png": dca_path}, f, indent=2, ensure_ascii=False)
    print(f"[DCA/CAL] saved to: {rel_path}, {dca_path}")

if __name__ == "__main__":
    main()

