#!/usr/bin/env python3
import os, json, pandas as pd
root='/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2'
exps=[
'baseline_a_clinical_only','baseline_b_slice_mean_concat','ablation1_mean_mil_gated',
'ablation2_attention_mil_concat','ablation3_no_penalty','ours_full']

rows=[]
for e in exps:
    for k in range(5):
        p=os.path.join(root,e,f'fold_{k}','metrics.json')
        if not os.path.exists(p): continue
        d=json.load(open(p,'r',encoding='utf-8'))
        test=d['test']['central']
        rows.append({'exp':e,'fold':k,'AUC':test['AUC'],'ACC':test['ACC'],'F1':test['F1'],'Sen':test['Sensitivity'],'Spe':test['Specificity']})

df=pd.DataFrame(rows)
# 输出CSV与按exp分表LaTeX
out_csv=os.path.join(root,'ablation_per_fold_metrics.csv')
df.to_csv(out_csv,index=False)

out_dir='/data2/wjb/code/PTMC_LNM/overleaf_package_mil/figures'
os.makedirs(out_dir, exist_ok=True)
for e, sub in df.groupby('exp'):
    t=sub.sort_values('fold')
    tex=os.path.join(out_dir, f'{e}_per_fold.tex')
    t2=t.copy();
    for col in ['AUC','ACC','F1','Sen','Spe']:
        t2[col]=t2[col].map(lambda x: f"{x:.3f}")
    t2.to_latex(tex, index=False, column_format='lrrrrr', caption=f'Per-fold metrics: {e}', label=f'tab:{e}_per_fold')
    print(tex)
print(out_csv)
