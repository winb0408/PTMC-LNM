#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from options import parse_args
from dataset import PatientBagDataset
from networks import build_model


def denorm(img_t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = img_t.cpu() * std + mean
    x = torch.clamp(x, 0, 1)
    return x.permute(1, 2, 0).numpy()


def overlay(img, cam, alpha=0.42):
    cmap = plt.get_cmap('jet')
    hm = cmap(np.clip(cam, 0, 1))[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * hm, 0, 1)


class HookCAM:
    def __init__(self, model):
        self.fmap = None
        self.grad = None
        layer = model.instance_encoder.backbone.layer4
        self.h1 = layer.register_forward_hook(self._fh)
        self.h2 = layer.register_full_backward_hook(self._bh)

    def _fh(self, m, i, o):
        self.fmap = o

    def _bh(self, m, gi, go):
        self.grad = go[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()


def cam_from_fg(fmap, grad):
    w = grad.view(grad.size(0), -1).mean(dim=1)
    cam = (w[:, None, None] * fmap).sum(dim=0)
    cam = torch.relu(cam)
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-6)
    return cam.detach().cpu().numpy()


def infer_prob(model, bag, mask, clin):
    with torch.no_grad():
        out = model({'bag': bag, 'mask': mask, 'clinical': clin})
        p = torch.sigmoid(out['logit_central'])[0].item()
    return p


def make_case_figures(model, hook, bag, mask, clin, pid, y_true, p_central, rank_tag, out_dir):
    # forward for attention
    with torch.no_grad():
        out = model({'bag': bag, 'mask': mask, 'clinical': clin})
        attn = out['attn_weights'][0].detach().cpu().numpy()

    # backward for grad
    model.zero_grad()
    out = model({'bag': bag, 'mask': mask, 'clinical': clin})
    out['logit_central'][0].backward(retain_graph=True)
    fmap = hook.fmap.detach()  # [N,C,h,w]
    grad = hook.grad.detach()  # [N,C,h,w]

    # ---- 多切面图（Top4）----
    N = bag.size(1)
    k = min(4, N)
    sel = np.argsort(-attn)[:k]
    fig, axes = plt.subplots(2, k, figsize=(3.1 * k, 5.8))
    if k == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, i in enumerate(sel):
        img_b = denorm(bag[0, i, 0:3])
        cam = cam_from_fg(fmap[i], grad[i])
        cam_up = F.interpolate(torch.tensor(cam)[None, None], size=img_b.shape[:2], mode='bilinear', align_corners=False)[0, 0].numpy()
        ov = overlay(img_b, cam_up)
        axes[0, j].imshow(img_b)
        axes[0, j].axis('off')
        axes[0, j].set_title(f'Slice {i}')
        axes[1, j].imshow(ov)
        axes[1, j].axis('off')
        axes[1, j].set_title(f'Grad-CAM (w={attn[i]:.2f})')
    fig.suptitle(f'PID={pid} | y={int(y_true)} | p_central={p_central:.3f} | group={rank_tag}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out1 = os.path.join(out_dir, f'gradcam_slices_case_{rank_tag}.png')
    fig.savefig(out1, dpi=220)
    plt.close(fig)

    # ---- 多模态图（Top1切面）----
    top_i = int(np.argmax(attn))
    mods = [('B', 0, 3), ('VUSE', 3, 6), ('VCEUS', 6, 9)]
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.8))
    for j, (name, l, r) in enumerate(mods):
        bag_m = bag.clone()
        keep = torch.zeros_like(bag_m)
        keep[:, :, l:r, :, :] = bag[:, :, l:r, :, :]
        bag_m = keep

        model.zero_grad()
        out_m = model({'bag': bag_m, 'mask': mask, 'clinical': clin})
        out_m['logit_central'][0].backward(retain_graph=True)
        fmap_m = hook.fmap.detach()
        grad_m = hook.grad.detach()

        img = denorm(bag[0, top_i, l:r])
        cam = cam_from_fg(fmap_m[top_i], grad_m[top_i])
        cam_up = F.interpolate(torch.tensor(cam)[None, None], size=img.shape[:2], mode='bilinear', align_corners=False)[0, 0].numpy()
        ov = overlay(img, cam_up)

        axes[0, j].imshow(img)
        axes[0, j].axis('off')
        axes[0, j].set_title(f'{name} input')
        axes[1, j].imshow(ov)
        axes[1, j].axis('off')
        axes[1, j].set_title(f'{name} Grad-CAM')

    fig.suptitle(f'PID={pid} | y={int(y_true)} | p_central={p_central:.3f} | top-slice={top_i} | group={rank_tag}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = os.path.join(out_dir, f'gradcam_modalities_case_{rank_tag}.png')
    fig.savefig(out2, dpi=220)
    plt.close(fig)

    print(out1)
    print(out2)


def main():
    data_root = '/data2/wjb/code/PTMC_LNM/data/MultiModalDataset'
    ckpt = '/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2/ours_full/fold_0/best_model.pth'
    out_dir = '/data2/wjb/code/PTMC_LNM/figures'
    os.makedirs(out_dir, exist_ok=True)

    opt = parse_args(['--model_mode', 'attention_mil_gated', '--penalty_alpha', '1.0', '--gpu_id', '0'])
    device = torch.device('cuda:0' if (opt.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    ds = PatientBagDataset(data_root, 'groups_cv5_central/fold_0', 'test', opt.modalities, opt.img_size, opt.max_instances, opt.clinical_agg, False)

    model = build_model(opt).to(device)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # 得到每个患者概率，选高/中/低
    probs = []
    ys = []
    for i in range(len(ds.pids)):
        s = ds[i]
        bag = s['bag'].unsqueeze(0).to(device)
        mask = torch.ones((1, bag.size(1)), dtype=torch.bool, device=device)
        clin = s['clinical'].unsqueeze(0).to(device)
        p = infer_prob(model, bag, mask, clin)
        probs.append(p)
        ys.append(float(s['y_central'].item()))

    order = np.argsort(probs)
    idx_high = int(order[-1])
    idx_mid = int(order[len(order)//2])
    idx_low = int(order[0])

    hook = HookCAM(model)
    for tag, idx in [('high', idx_high), ('mid', idx_mid), ('low', idx_low)]:
        s = ds[idx]
        pid = ds.pids[idx]
        bag = s['bag'].unsqueeze(0).to(device)
        mask = torch.ones((1, bag.size(1)), dtype=torch.bool, device=device)
        clin = s['clinical'].unsqueeze(0).to(device)
        p = probs[idx]
        y = ys[idx]
        make_case_figures(model, hook, bag, mask, clin, pid, y, p, tag, out_dir)

    hook.close()


if __name__ == '__main__':
    main()
