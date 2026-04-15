#!/usr/bin/env bash
set -euo pipefail
PY=/data2/wjb/anaconda3/envs/pumc/bin/python
SAVE="/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/v2_mil_gated/checkpoints_v2_backbones_pretrained"
LOG="$SAVE/rerun_failed_pretrained_queue.log"
mkdir -p "$SAVE"

run_one () {
  local bb="$1"
  local gpu="$2"
  echo "[START] $(date '+%F %T') $bb gpu=$gpu" | tee -a "$LOG"
  "$PY" train.py \
    --model_mode attention_mil_gated \
    --penalty_alpha 1.0 \
    --backbone "$bb" \
    --save_dir "$SAVE" \
    --exp_name "backbone_${bb}" \
    --epochs 60 \
    --seed 42 \
    --gpu_id "$gpu" \
    --pretrained True \
    > "$SAVE/log_${bb}_rerun.txt" 2>&1
  local rc=$?
  echo "[DONE]  $(date '+%F %T') $bb rc=$rc" | tee -a "$LOG"
  return $rc
}

echo "=== rerun failed pretrained backbones queue start ===" | tee "$LOG"
# 串行排队，避免OOM；不改模型参数
run_one efficientnet_b2 0
run_one vit_b_16 0
run_one mobilevit_s 0

echo "=== all reruns done ===" | tee -a "$LOG"
