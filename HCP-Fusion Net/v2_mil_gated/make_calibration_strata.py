#!/usr/bin/env python3
import os, numpy as np, pandas as pd
from sklearn.calibration import calibration_curve

base='/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2/ours_full'
ys=[]; ps=[]
for k in range(5):
    p=os.path.join(base, f'fold_{k}', 'predictions.npz')
    if not os.path.exists(p): continue
    d=np.load(p, allow_pickle=True)
    ys.append(d['y_central']); ps.append(d['p_central'])

y=np.concatenate(ys); p=np.concatenate(ps)
# 分箱：固定阈值或分位数，这里用固定阈值0.1间隔
bins=np.arange(0.0, 1.01, 0.1)
inds=np.digitize(p, bins, right=True)
rows=[]
for b in range(1, len(bins)):
    mask=(inds==b); n=mask.sum()
    if n==0:
        rows.append({'bin':f'[{bins[b-1]:.1f},{bins[b]:.1f}]','n':0,'pos':0,'rate':0.0})
        continue
    yb=y[mask]
    pos=int((yb==1).sum())
    rate=pos/n
    rows.append({'bin':f'[{bins[b-1]:.1f},{bins[b]:.1f}]','n':int(n),'pos':pos,'rate':rate})

df=pd.DataFrame(rows)
# 导出CSV与LaTeX表
out_csv='/data2/wjb/code/PTMC_LNM/overleaf_package_mil/figures/calibration_strata_ours.csv'
df.to_csv(out_csv, index=False)
out_tex='/data2/wjb/code/PTMC_LNM/overleaf_package_mil/figures/calibration_strata_ours.tex'
df2=df.copy(); df2['rate']=df2['rate'].map(lambda x: f"{x:.3f}")
df2.to_latex(out_tex, index=False, column_format='lrrr', caption='Calibration strata of Ours', label='tab:calib_strata')
print(out_csv) ; print(out_tex)
