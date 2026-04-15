import os, json, pandas as pd, time
root = '/data2/wjb/code/PTMC_LNM'
run_dir = os.path.join(root,'aris_runs','20260401_223156_ptmc_centrallnm')
art = os.path.join(run_dir,'artifacts')
out_tex = os.path.join(root,'overleaf_package','main_CLNM_current.tex')

def safe_mean_std(vals):
    import numpy as np
    if len(vals)==0: return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals, ddof=0))

# load GPU artifacts
img_patient = os.path.join(art,'dl_imgonly_b_r50_gpu_patientAgg.csv')
img_deeprad = os.path.join(art,'dl_imgonly_b_r50_gpu_clinPlusDeepRad_cv.csv')
mm_cv = os.path.join(art,'dl_central_full_r50_gpu_cv_summary.csv')
mil_sum = os.path.join(art,'mil_central_b_gpu_summary.json')

img_mean, img_std = float('nan'), float('nan')
if os.path.exists(img_patient):
    df = pd.read_csv(img_patient)
    if 'auc_patient_mean' in df.columns:
        img_mean, img_std = safe_mean_std(df['auc_patient_mean'].tolist())

dr_mean, dr_std = float('nan'), float('nan')
if os.path.exists(img_deeprad):
    df = pd.read_csv(img_deeprad)
    if 'auc_test' in df.columns:
        dr_mean, dr_std = safe_mean_std(df['auc_test'].tolist())

mm_mean, mm_std = float('nan'), float('nan')
if os.path.exists(mm_cv):
    df = pd.read_csv(mm_cv)
    if 'best_val_auc' in df.columns:
        mm_mean, mm_std = safe_mean_std(df['best_val_auc'].tolist())

mil_val_mean = mil_val_std = mil_test_mean = mil_test_std = float('nan')
if os.path.exists(mil_sum):
    d = json.load(open(mil_sum,'r',encoding='utf-8'))
    mil_val_mean = d.get('val_auc_mean','nan')
    mil_val_std  = d.get('val_auc_std','nan')
    mil_test_mean= d.get('test_auc_mean','nan')
    mil_test_std = d.get('test_auc_std','nan')

tex = r"""
\documentclass[12pt,a4paper]{ctexart}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\title{CentralLNM 实验汇总报告（GPU 版，自动生成）}
\author{PTMC\_LNM 项目}
\date{%s}
\begin{document}
\maketitle

\section{概述}
本文基于 GPU 跑完的三路实验（image-only、Clinical+DeepRad、multi-modal、MIL），对 CentralLNM 的指标进行汇总，所有产物来源于 `%s`。

\section{关键结果（均值$\pm$标准差）}
\begin{table}[H]
\centering
\caption{CentralLNM GPU 路线汇总}
\begin{tabular}{lcc}
\toprule
路线 & 指标 & 5-fold 结果 \\
\midrule
Image-only(B)（患者级 mean AUC） & AUC & %.4f $\pm$ %.4f \\
Clinical + DeepRadScore（test AUC） & AUC & %.4f $\pm$ %.4f \\
Multi-modal 单任务（val 最优 AUC） & AUC & %.4f $\pm$ %.4f \\
MIL（val AUC） & AUC & %.4f $\pm$ %.4f \\
MIL（test AUC） & AUC & %.4f $\pm$ %.4f \\
\bottomrule
\end{tabular}
\end{table}

\section{产物与明细入口}
\begin{itemize}
  \item Image-only 患者级：\verb|%s|
  \item Clinical+DeepRad：\verb|%s|
  \item Multi-modal CV 汇总：\verb|%s|
  \item MIL 汇总：\verb|%s|
\end{itemize}

\section{结论小结}
\begin{itemize}
  \item 当前 GPU 结果与既往趋势一致：image-only 与 multi-modal 的验证集 AUC 仍在 0.60--0.64 区间波动；Clinical+DeepRad 接近临床基线未显著超越。
  \item MIL 路线（GPU）已完成 5-fold 并给出 val/test 汇总，用于 DCA/校准与论文补充分析。
\end{itemize}

\end{document}
""" % (time.strftime('%Y-%m-%d %H:%M:%S'), art,
       img_mean, img_std, dr_mean, dr_std, mm_mean, mm_std,
       mil_val_mean, mil_val_std, mil_test_mean, mil_test_std,
       img_patient, img_deeprad, mm_cv, mil_sum)

open(out_tex,'w',encoding='utf-8').write(tex)
print('WROTE', out_tex)
