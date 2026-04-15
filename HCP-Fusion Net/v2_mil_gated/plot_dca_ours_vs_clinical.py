#!/usr/bin/env python3
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_pred(base):
    ys=[]; ps=[]
    for k in range(5):
        p=os.path.join(base, f'fold_{k}', 'predictions.npz')
        if not os.path.exists(p): continue
        d=np.load(p, allow_pickle=True)
        ys.append(d['y_central']); ps.append(d['p_central'])
    y=np.concatenate(ys); p=np.concatenate(ps)
    return y.astype(int), p.astype(float)

# simple DCA
def net_benefit(y, p, thresh):
    n=len(y)
    pred=(p>=thresh).astype(int)
    tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum()
    return tp/n - fp/n * (thresh/(1-thresh))

if __name__=='__main__':
    ours='/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2/ours_full'
    clin='/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2/baseline_a_clinical_only'
    y_o, p_o = load_pred(ours)
    y_c, p_c = load_pred(clin)
    assert (y_o==y_c).all()
    ts=np.linspace(0.05,0.95,19)
    nb_o=[net_benefit(y_o,p_o,t) for t in ts]
    nb_c=[net_benefit(y_c,p_c,t) for t in ts]
    prev = y_o.mean()
    nb_all=[prev - (1-prev)*(t/(1-t)) for t in ts]  # treat-all
    nb_none=[0.0 for _ in ts]  # treat-none
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(ts, nb_o, label='Ours')
    ax.plot(ts, nb_c, label='Clinical-only')
    ax.plot(ts, nb_all, 'k--', alpha=0.7, label='Treat-all (prevalence baseline)')
    ax.plot(ts, nb_none, color='gray', linestyle=':', alpha=0.9, label='Treat-none')
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
    ax.set_title('Decision Curve Analysis (CentralLNM)')
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.legend(); fig.tight_layout()
    out='/data2/wjb/code/PTMC_LNM/figures/dca_ours_vs_clinical.png'
    fig.savefig(out,dpi=220); print(out)
