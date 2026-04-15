#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import time

PY = '/data2/wjb/anaconda3/envs/pumc/bin/python'
ROOT = os.path.abspath(os.path.dirname(__file__) or '.')
SAVE = os.path.join(ROOT, 'checkpoints_v2_backbones_pretrained')
os.makedirs(SAVE, exist_ok=True)

# 统一协议：pretrained=True；ResNet18 直接复用既有消融主线结果（同协议）
JOBS = [
    {'bb': 'resnet50', 'gpu': 1},
    {'bb': 'efficientnet_b0', 'gpu': 2},
    {'bb': 'efficientnet_b2', 'gpu': 3},
    {'bb': 'convnext_tiny', 'gpu': 0},
    {'bb': 'vit_b_16', 'gpu': 1},
    {'bb': 'swin_t', 'gpu': 2},
    {'bb': 'mobilevit_s', 'gpu': 3},
]

STATUS = os.path.join(SAVE, 'sweep_status.json')
LOG = os.path.join(SAVE, 'sweep_launcher.log')
MAX_PAR = 4


def cmd(bb, gpu):
    return [
        PY, 'train.py',
        '--model_mode', 'attention_mil_gated',
        '--penalty_alpha', '1.0',
        '--backbone', bb,
        '--save_dir', SAVE,
        '--exp_name', f'backbone_{bb}',
        '--epochs', '60',
        '--seed', '42',
        '--gpu_id', str(gpu),
        '--pretrained', 'True',
    ]


def reuse_resnet18():
    src = os.path.join(ROOT, 'checkpoints_v2', 'ours_full')
    dst = os.path.join(SAVE, 'backbone_resnet18')
    if os.path.exists(dst):
        return
    if not os.path.exists(src):
        raise FileNotFoundError(f'无法复用 ResNet18：源目录不存在 {src}')
    shutil.copytree(src, dst)
    # 追加一个复用标记文件，便于后续写报告说明
    marker = os.path.join(dst, 'REUSED_FROM.txt')
    with open(marker, 'w', encoding='utf-8') as f:
        f.write('Reused from checkpoints_v2/ours_full (same training protocol: pretrained=True, freeze=3, 5-fold).\n')


if __name__ == '__main__':
    reuse_resnet18()

    queue = JOBS.copy()
    running = []
    done = [{'bb': 'resnet18', 'gpu': -1, 'rc': 0, 'note': 'reused'}]

    with open(LOG, 'w', encoding='utf-8') as lf:
        lf.write('=== backbone sweep (pretrained=True) start ===\n')
        lf.write('[REUSE] resnet18 from checkpoints_v2/ours_full\n')

    with open(STATUS, 'w', encoding='utf-8') as sf:
        json.dump({'running': [], 'queued': [j['bb'] for j in queue], 'done': done}, sf, ensure_ascii=False)

    while queue or running:
        while queue and len(running) < MAX_PAR:
            j = queue.pop(0)
            f = open(os.path.join(SAVE, f'log_{j["bb"]}.txt'), 'w', encoding='utf-8')
            p = subprocess.Popen(cmd(j['bb'], j['gpu']), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            running.append({'p': p, 'j': j, 'f': f})
            with open(LOG, 'a', encoding='utf-8') as lf:
                lf.write(f'[START] {j["bb"]} gpu={j["gpu"]} pid={p.pid}\n')

        time.sleep(6)
        still = []
        for r in running:
            rc = r['p'].poll()
            if rc is None:
                still.append(r)
            else:
                r['f'].close()
                done.append({'bb': r['j']['bb'], 'gpu': r['j']['gpu'], 'rc': rc})
                with open(LOG, 'a', encoding='utf-8') as lf:
                    lf.write(f'[DONE] {r["j"]["bb"]} rc={rc}\n')
        running = still

        with open(STATUS, 'w', encoding='utf-8') as sf:
            json.dump(
                {
                    'running': [{'bb': r['j']['bb'], 'pid': r['p'].pid, 'gpu': r['j']['gpu']} for r in running],
                    'queued': [j['bb'] for j in queue],
                    'done': done
                },
                sf,
                ensure_ascii=False
            )

    with open(LOG, 'a', encoding='utf-8') as lf:
        lf.write('=== all done ===\n')
