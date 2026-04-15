#!/usr/bin/env python3
import os, json, subprocess, sys

PY='/data2/wjb/anaconda3/envs/pumc/bin/python'
ROOT=os.path.dirname(__file__)
SAVE=os.path.join(ROOT,'checkpoints_v2_backbones')
os.makedirs(SAVE, exist_ok=True)

# Candidate backbones (will be tried; unavailable ones are skipped inside train)
BACKBONES=[
  'resnet18', 'resnet50',
  'efficientnet_b0', 'efficientnet_b2',
  'convnext_tiny', 'vit_b_16', 'swin_t'
  # MobileViT/Mamba require extra deps; leave for future extension
]

def run_one(bb: str):
    exp=f'backbone_{bb}'
    cmd=[PY, 'train.py',
         '--model_mode','attention_mil_gated','--penalty_alpha','1.0',
         '--backbone', bb,
         '--save_dir', SAVE,
         '--exp_name', exp,
         '--epochs','60','--seed','42','--gpu_id','0']
    print('RUN', ' '.join(cmd))
    try:
        subprocess.check_call(cmd, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        print('SKIP/FAIL', bb, 'rc=', e.returncode)

if __name__=='__main__':
    for bb in BACKBONES:
        run_one(bb)
