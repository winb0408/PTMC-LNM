#!/usr/bin/env python3
import os, time, subprocess, json

ROOT = "/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated"
PYTHON = "/data2/wjb/anaconda3/envs/pumc/bin/python"
SAVE_ROOT = os.path.join(ROOT, "checkpoints_v2")
os.makedirs(SAVE_ROOT, exist_ok=True)

JOBS = [
    {"name":"baseline_a_clinical_only","mode":"clinical_only","alpha":"0"},
    {"name":"baseline_b_slice_mean_concat","mode":"slice_mean_concat","alpha":"0"},
    {"name":"ablation1_mean_mil_gated","mode":"mean_mil_gated","alpha":"1.0"},
    {"name":"ablation2_attention_mil_concat","mode":"attention_mil_concat","alpha":"1.0"},
    {"name":"ablation3_no_penalty","mode":"attention_mil_gated","alpha":"0"},
    {"name":"ours_full","mode":"attention_mil_gated","alpha":"1.0"},
]

GPUS = [0,1,2,3]
EPOCHS = 60
SEED = 42

status_path = os.path.join(SAVE_ROOT, "parallel_launcher_status.json")
log_path = os.path.join(SAVE_ROOT, "run_all_parallel.log")


def make_cmd(job, gpu):
    return [
        PYTHON, "train.py",
        "--model_mode", job["mode"],
        "--penalty_alpha", job["alpha"],
        "--exp_name", job["name"],
        "--save_dir", SAVE_ROOT,
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--gpu_id", str(gpu),
    ]


def dump_status(running, queued, done):
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "running": running,
        "queued": queued,
        "done": done,
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    os.chdir(ROOT)
    queue = JOBS.copy()
    running = {}  # gpu -> {proc, job, logf}
    done = []

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("=== 4-GPU parallel launcher start ===\n")

    while queue or running:
        # 分配空闲GPU
        for gpu in GPUS:
            if gpu in running:
                continue
            if not queue:
                continue
            job = queue.pop(0)
            job_log = os.path.join(SAVE_ROOT, f"log_{job['name']}.txt")
            f = open(job_log, "w", encoding="utf-8")
            proc = subprocess.Popen(make_cmd(job, gpu), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            running[gpu] = {"proc": proc, "job": job, "logf": f}
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[START] {job['name']} gpu={gpu} pid={proc.pid}\n")

        dump_status(
            running=[{"gpu":g, "name":v['job']['name'], "pid":v['proc'].pid} for g,v in running.items()],
            queued=[q['name'] for q in queue],
            done=done,
        )

        time.sleep(8)

        finished_gpus = []
        for gpu, rec in running.items():
            rc = rec["proc"].poll()
            if rc is not None:
                rec["logf"].close()
                done.append({"name": rec['job']['name'], "gpu": gpu, "rc": rc})
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"[DONE] {rec['job']['name']} gpu={gpu} rc={rc}\n")
                finished_gpus.append(gpu)
        for g in finished_gpus:
            running.pop(g, None)

    dump_status(running=[], queued=[], done=done)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("=== all jobs done ===\n")


if __name__ == "__main__":
    main()
