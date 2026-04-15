#!/usr/bin/env bash
set -euo pipefail
PY="/data2/wjb/anaconda3/envs/pumc/bin/python"
ROOT="/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated"
SAVE="$ROOT/checkpoints_v2_backbones"
QUEUE_LOG="$SAVE/rerun_failed_queue.log"
STATUS_JSON="$SAVE/rerun_failed_status.json"

# 不改参数，仅重跑失败模型
MODELS=("convnext_tiny" "vit_b_16" "mobilevit_s")
GPUS=(1 2)
MAX_PAR=2

echo "=== rerun failed backbones start $(date '+%F %T') ===" > "$QUEUE_LOG"
printf '{"running":[],"queued":["convnext_tiny","vit_b_16","mobilevit_s"],"done":[]}' > "$STATUS_JSON"

running_pids=()
running_models=()
running_gpus=()
done_items=()
queue=("${MODELS[@]}")

spawn_job() {
  local model="$1"
  local gpu="$2"
  local lf="$SAVE/log_${model}.txt"
  : > "$lf"
  "$PY" "$ROOT/train.py" \
    --model_mode attention_mil_gated --penalty_alpha 1.0 \
    --backbone "$model" --save_dir "$SAVE" --exp_name "backbone_${model}" \
    --epochs 60 --seed 42 --gpu_id "$gpu" --pretrained False \
    > "$lf" 2>&1 &
  local pid=$!
  running_pids+=("$pid")
  running_models+=("$model")
  running_gpus+=("$gpu")
  echo "[START] $model gpu=$gpu pid=$pid" >> "$QUEUE_LOG"
}

write_status() {
  local running_json=""
  local queued_json=""
  local done_json=""

  for i in "${!running_pids[@]}"; do
    running_json+="{\"bb\":\"${running_models[$i]}\",\"pid\":${running_pids[$i]},\"gpu\":${running_gpus[$i]}},"
  done
  running_json="${running_json%,}"

  for q in "${queue[@]:-}"; do
    queued_json+="\"$q\"," 
  done
  queued_json="${queued_json%,}"

  for d in "${done_items[@]:-}"; do
    done_json+="$d,"
  done
  done_json="${done_json%,}"

  printf '{"running":[%s],"queued":[%s],"done":[%s]}' "$running_json" "$queued_json" "$done_json" > "$STATUS_JSON"
}

while ((${#queue[@]} > 0)) || ((${#running_pids[@]} > 0)); do
  # 尽量填满并发槽位
  while ((${#queue[@]} > 0)) && ((${#running_pids[@]} < MAX_PAR)); do
    model="${queue[0]}"
    queue=("${queue[@]:1}")

    # 选择一个当前未被本脚本占用的GPU
    gpu=""
    for g in "${GPUS[@]}"; do
      used=0
      for rg in "${running_gpus[@]:-}"; do
        if [[ "$rg" == "$g" ]]; then used=1; break; fi
      done
      if [[ $used -eq 0 ]]; then gpu="$g"; break; fi
    done
    [[ -z "$gpu" ]] && break

    spawn_job "$model" "$gpu"
    write_status
  done

  sleep 8

  # 回收完成任务
  new_pids=()
  new_models=()
  new_gpus=()
  for i in "${!running_pids[@]}"; do
    pid="${running_pids[$i]}"
    model="${running_models[$i]}"
    gpu="${running_gpus[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_models+=("$model")
      new_gpus+=("$gpu")
    else
      wait "$pid" || rc=$?
      rc=${rc:-0}
      echo "[DONE] $model gpu=$gpu rc=$rc" >> "$QUEUE_LOG"
      done_items+=("{\"bb\":\"$model\",\"gpu\":$gpu,\"rc\":$rc}")
      rc=0
    fi
  done
  running_pids=("${new_pids[@]:-}")
  running_models=("${new_models[@]:-}")
  running_gpus=("${new_gpus[@]:-}")
  write_status

done

echo "=== rerun failed backbones done $(date '+%F %T') ===" >> "$QUEUE_LOG"
