#!/usr/bin/env python3
import os, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from dataset import PatientBagDataset
from networks import build_model
from options import parse_args

def save_patient_panel(opt, model, data_root, groups_dir, split, out_dir, n_cases=6, modality='B'):
    os.makedirs(out_dir, exist_ok=True)
    ds = PatientBagDataset(data_root, groups_dir, split, modalities=opt.modalities, img_size=224, max_instances=12, clinical_agg=opt.clinical_agg, is_train=False)
    # 为了得到每位患者的切面文件名，重建pid->names映射
    split_file = os.path.join(data_root, groups_dir, f'{split}.txt')
    names = []
    with open(split_file,'r',encoding='utf-8') as f:
        for ln in f:
            s=ln.strip();
            if s: names.append(os.path.splitext(s)[0])
    pid_to_names={}
    for n in names:
        pid=n.split('_',1)[0]
        pid_to_names.setdefault(pid,[]).append(n)

    # 采样病例
    pids = ds.pids
    random.seed(42)
    sample_pids = random.sample(pids, min(n_cases, len(pids)))

    device = torch.device('cuda:0' if (opt.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    model.eval(); model.to(device)

    for pid in sample_pids:
        # 组装bag
        bag_tensors=[]; thumbs=[]
        for n in pid_to_names.get(pid, []):
            # 仅用指定模态B做缩略图
            img_path=None
            for ext in ['.jpg','.png','.jpeg','.bmp','.tif','.tiff']:
                p=os.path.join(data_root, modality, f'{n}{ext}')
                if os.path.exists(p): img_path=p; break
            if img_path is None: continue
            im=Image.open(img_path).convert('RGB').resize((224,224))
            thumbs.append(np.asarray(im))
        # 用dataset生成的bag（含多模态通道），以获得与模型一致的预处理
        idx = ds.pids.index(pid)
        sample = ds[idx]
        bag = sample['bag'].unsqueeze(0).to(device)  # [1,N,C,H,W]
        mask = torch.ones(bag.size(1), dtype=torch.bool, device=device).unsqueeze(0)
        clin = sample['clinical'].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model({'bag':bag,'mask':mask,'clinical':clin})
            attn = out['attn_weights'].squeeze(0).detach().cpu().numpy() if out['attn_weights'] is not None else None
        # 画板：上排图像缩略图，下排对应注意力条形图
        ninst=len(thumbs)
        if ninst==0 or attn is None: continue
        fig=plt.figure(figsize=(min(12, 2*ninst), 4))
        # 显示缩略图
        for i,arr in enumerate(thumbs):
            ax=fig.add_subplot(2,ninst,i+1)
            ax.imshow(arr); ax.axis('off'); ax.set_title(f'i={i}')
        # 注意力条
        ax2=fig.add_subplot(2,1,2)
        k=min(ninst, len(attn))
        ax2.bar(range(k), attn[:k]); ax2.set_ylabel('Attention'); ax2.set_xlabel('Instance index')
        fig.suptitle(f'PID={pid} (Attention weights)')
        fig.tight_layout(rect=[0,0,1,0.93])
        out_path=os.path.join(out_dir, f'attn_panel_pid_{pid}.png')
        fig.savefig(out_path, dpi=180); plt.close(fig)
        print(out_path)

if __name__=='__main__':
    # 使用 ours_full/fold_*/best_model.pth
    base_ckpt='/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2/ours_full'
    data_root='/data2/wjb/code/PTMC_LNM/data/MultiModalDataset'
    # 用fold_0路径构造 groups_dir，并采样test集
    fold=0
    groups_dir='groups_cv5_central/fold_0'
    # 构造与训练一致的opt
    opt = parse_args(['--model_mode','attention_mil_gated','--penalty_alpha','1.0','--gpu_id','0'])
    model = build_model(opt)
    ckpt=os.path.join(base_ckpt, f'fold_{fold}', 'best_model.pth')
    state=torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state)
    out_dir='/data2/wjb/code/PTMC_LNM/figures/attn_panels'
    save_patient_panel(opt, model, data_root, groups_dir, 'test', out_dir, n_cases=6, modality='B')
