#!/usr/bin/env python3
import os, time, subprocess, json
PY='/data2/wjb/anaconda3/envs/pumc/bin/python'
ROOT=os.path.abspath(os.path.dirname(__file__) or '.')
SAVE=os.path.join(ROOT,'checkpoints_v2_backbones')
os.makedirs(SAVE, exist_ok=True)

# 扩展多骨干（均使用相同预训练策略，公平对比）
JOBS=[
  {'bb':'resnet18',        'gpu':0},
  {'bb':'resnet50',        'gpu':1},
  {'bb':'efficientnet_b0', 'gpu':2},
  {'bb':'efficientnet_b2', 'gpu':3},
  {'bb':'convnext_tiny',   'gpu':0},
  {'bb':'vit_b_16',        'gpu':1},
  {'bb':'swin_t',          'gpu':2},
  {'bb':'mobilevit_s',     'gpu':3},
]

status=os.path.join(SAVE,'sweep_status.json')
log=os.path.join(SAVE,'sweep_launcher.log')

MAX_PAR=4

def cmd(bb,gpu):
    return [PY,'train.py','--model_mode','attention_mil_gated','--penalty_alpha','1.0',
            '--backbone',bb,'--save_dir',SAVE,'--exp_name',f'backbone_{bb}',
            '--epochs','60','--seed','42','--gpu_id',str(gpu),
            '--pretrained','False']

if __name__=='__main__':
    queue=JOBS.copy(); running=[]; done=[]
    with open(log,'w') as lf:
        lf.write('=== backbone sweep start ===\n')
    with open(status,'w') as sf:
        json.dump({'running':[], 'queued':[j['bb'] for j in queue], 'done':[]}, sf)
    while queue or running:
        while queue and len(running)<MAX_PAR:
            j=queue.pop(0)
            f=open(os.path.join(SAVE,f'log_{j["bb"]}.txt'),'w')
            p=subprocess.Popen(cmd(j['bb'],j['gpu']),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            running.append({'p':p,'j':j,'f':f})
            with open(log,'a') as lf: lf.write(f"[START] {j['bb']} gpu={j['gpu']} pid={p.pid}\n")
        time.sleep(6)
        still=[]
        for r in running:
            rc=r['p'].poll()
            if rc is None:
                still.append(r)
            else:
                r['f'].close(); done.append({'bb':r['j']['bb'],'gpu':r['j']['gpu'],'rc':rc})
                with open(log,'a') as lf: lf.write(f"[DONE] {r['j']['bb']} rc={rc}\n")
        running=still
        with open(status,'w') as sf:
            json.dump({'running':[{'bb':r['j']['bb'],'pid':r['p'].pid,'gpu':r['j']['gpu']} for r in running],
                       'queued':[j['bb'] for j in queue], 'done':done}, sf)
    with open(log,'a') as lf: lf.write('=== all done ===\n')
