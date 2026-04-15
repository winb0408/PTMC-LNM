#!/usr/bin/env python3
"""
统一串联三源实验：
1) 先生成 fold 级 ROI radiomics 特征
2) Baseline C / D 从 radiomics 报表读取
3) Ours V1: Deep MIL + Clinical Gated（EffNet-B0）
4) Ours V2: Tri-source Gated Fusion（EffNet-B0 + ROI radiomics）

说明：
- 该脚本主要负责组织命令与目录，不强制并行。
- 运行前请先确保已安装 PyRadiomics / SimpleITK。
"""

import argparse
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    p = argparse.ArgumentParser(description='Run tri-source experiments')
    p.add_argument('--python', type=str, default=sys.executable)
    p.add_argument('--data_root', type=str, default='/data2/wjb/code/PTMC_LNM/data/MultiModalDataset')
    p.add_argument('--mask_root', type=str, default='/data2/wjb/code/PTMC_LNM/data/masks')
    p.add_argument('--radiomics_save_dir', type=str, default=os.path.join(ROOT, 'checkpoints_roi_radiomics'))
    p.add_argument('--radiomics_exp_name', type=str, default='central_roi_radiomics_cv5')
    p.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'checkpoints_v2_tri_source'))
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--pretrained', type=str, default='True')
    p.add_argument('--run_radiomics', action='store_true')
    p.add_argument('--run_v1', action='store_true')
    p.add_argument('--run_v2', action='store_true')
    return p.parse_args()


def run_cmd(cmd):
    print('\n[CMD]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def run_cmd_with_pretrain_fallback(cmd):
    """
    先按用户期望尝试 pretrained=True。
    如果当前离线环境导致权重下载失败，则自动回退到 pretrained=False 继续跑，
    避免整条流水线中断。
    """
    print('\n[CMD]', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if '--pretrained' in cmd:
            idx = cmd.index('--pretrained') + 1
            if idx < len(cmd) and str(cmd[idx]).lower() == 'true':
                retry_cmd = cmd.copy()
                retry_cmd[idx] = 'False'
                print('[WARN] 预训练权重下载失败，自动回退为 pretrained=False 继续实验')
                print('[RETRY]', ' '.join(retry_cmd))
                subprocess.run(retry_cmd, check=True)
                return
        raise


def main():
    args = parse_args()
    rad_root = os.path.join(args.radiomics_save_dir, args.radiomics_exp_name)

    if args.run_radiomics:
        cmd = [
            args.python,
            os.path.join(ROOT, 'roi_radiomics_cv5_central.py'),
            '--data_root', args.data_root,
            '--mask_root', args.mask_root,
            '--save_dir', args.radiomics_save_dir,
            '--exp_name', args.radiomics_exp_name,
            '--folds', str(args.folds),
            '--modalities', 'B',
        ]
        run_cmd(cmd)

    if args.run_v1:
        cmd = [
            args.python,
            os.path.join(os.path.dirname(__file__), 'train.py'),
            '--data_root', args.data_root,
            '--groups_root', 'groups_cv5_central',
            '--save_dir', args.save_dir,
            '--exp_name', 'ours_v1_effb0',
            '--model_mode', 'attention_mil_gated',
            '--backbone', 'efficientnet_b0',
            '--pretrained', args.pretrained,
            '--epochs', str(args.epochs),
            '--folds', str(args.folds),
            '--gpu_id', str(args.gpu_id),
            '--modalities', 'B', 'VUSE', 'VCEUS',
        ]
        run_cmd_with_pretrain_fallback(cmd)

    if args.run_v2:
        cmd = [
            args.python,
            os.path.join(os.path.dirname(__file__), 'train.py'),
            '--data_root', args.data_root,
            '--groups_root', 'groups_cv5_central',
            '--save_dir', args.save_dir,
            '--exp_name', 'ours_v2_tri_source_effb0',
            '--model_mode', 'attention_mil_gated',
            '--backbone', 'efficientnet_b0',
            '--pretrained', args.pretrained,
            '--epochs', str(args.epochs),
            '--folds', str(args.folds),
            '--gpu_id', str(args.gpu_id),
            '--modalities', 'B', 'VUSE', 'VCEUS',
            '--use_radiomics', 'True',
            '--radiomics_root', rad_root,
        ]
        run_cmd_with_pretrain_fallback(cmd)


if __name__ == '__main__':
    main()
