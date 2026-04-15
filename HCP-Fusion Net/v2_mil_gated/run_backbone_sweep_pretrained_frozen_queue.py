#!/usr/bin/env python3
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

PY = '/data2/wjb/anaconda3/envs/pumc/bin/python'
ROOT = os.path.abspath(os.path.dirname(__file__) or '.')
SAVE = os.path.join(ROOT, 'checkpoints_v2_backbones_pretrained_frozen')
os.makedirs(SAVE, exist_ok=True)

BACKBONES = [
    'resnet18',
    'resnet50',
    'efficientnet_b0',
    'efficientnet_b2',
    'convnext_tiny',
    'vit_b_16',
    'swin_t',
    'mobilevit_s',
]

STATUS = os.path.join(SAVE, 'sweep_status.json')
LOG = os.path.join(SAVE, 'sweep_launcher.log')
MAX_PAR = 4
CHECK_SEC = 8
GPU_IDS = [0, 1, 2, 3]


def cmd(bb: str, gpu: int) -> List[str]:
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
        '--freeze_backbone_epochs', '999',  # 全程冻结骨干，不在训练中解冻
    ]


def query_gpu_state() -> Dict[int, Dict[str, int]]:
    """
    返回每张GPU的内存占用与利用率：
      {gpu_id: {'mem_used': MiB, 'util': %}}
    """
    out = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    )
    states: Dict[int, Dict[str, int]] = {}
    for ln in out.strip().splitlines():
        idx_s, mem_s, util_s = [x.strip() for x in ln.split(',')]
        idx = int(idx_s)
        states[idx] = {'mem_used': int(mem_s), 'util': int(util_s)}
    return states


def pick_idle_gpu(occupied: set) -> Optional[int]:
    """
    空闲判定（保守）：
      memory.used < 1500 MiB 且 util < 10%
    """
    states = query_gpu_state()
    for g in GPU_IDS:
        if g in occupied:
            continue
        st = states.get(g)
        if st is None:
            continue
        if st['mem_used'] < 1500 and st['util'] < 10:
            return g
    return None


def write_status(running, queue, done):
    with open(STATUS, 'w', encoding='utf-8') as sf:
        json.dump(
            {
                'running': [
                    {'bb': r['bb'], 'pid': r['p'].pid, 'gpu': r['gpu']}
                    for r in running
                ],
                'queued': [j['bb'] for j in queue],
                'done': done,
            },
            sf,
            ensure_ascii=False,
        )


if __name__ == '__main__':
    queue = [{'bb': bb} for bb in BACKBONES]
    running = []
    done = []

    with open(LOG, 'w', encoding='utf-8') as lf:
        lf.write('=== backbone sweep (pretrained=True, frozen backbone) start ===\n')

    write_status(running, queue, done)

    while queue or running:
        # 回收已结束任务
        still = []
        for r in running:
            rc = r['p'].poll()
            if rc is None:
                still.append(r)
            else:
                r['f'].close()
                done.append({'bb': r['bb'], 'gpu': r['gpu'], 'rc': rc})
                with open(LOG, 'a', encoding='utf-8') as lf:
                    lf.write(f'[DONE] {r["bb"]} gpu={r["gpu"]} rc={rc}\n')
        running = still

        # 尝试调度新任务（有空闲GPU就发车，不够就排队）
        while queue and len(running) < MAX_PAR:
            occupied = {r['gpu'] for r in running}
            g = pick_idle_gpu(occupied)
            if g is None:
                break
            job = queue.pop(0)
            bb = job['bb']
            f = open(os.path.join(SAVE, f'log_{bb}.txt'), 'w', encoding='utf-8')
            p = subprocess.Popen(cmd(bb, g), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            running.append({'bb': bb, 'gpu': g, 'p': p, 'f': f})
            with open(LOG, 'a', encoding='utf-8') as lf:
                lf.write(f'[START] {bb} gpu={g} pid={p.pid}\n')

        write_status(running, queue, done)
        time.sleep(CHECK_SEC)

    with open(LOG, 'a', encoding='utf-8') as lf:
        lf.write('=== all done ===\n')
