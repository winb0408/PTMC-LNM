#!/bin/bash
# =============================================================================
# run_all_experiments.sh
# 批量运行 6 组消融实验（Baseline A/B + Ablation 1/2/3 + Ours）
#
# 使用方式：
#   bash run_all_experiments.sh [GPU_ID]
#   默认 GPU_ID=0
# =============================================================================

set -e

PYTHON="/data2/wjb/anaconda3/envs/pumc/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_ID="${1:-0}"
SAVE_ROOT="${SCRIPT_DIR}/checkpoints_v2"
EPOCHS=60
SEED=42

echo "=============================================="
echo "  Attention-MIL + Gated Fusion 消融实验"
echo "  GPU: ${GPU_ID}  Epochs: ${EPOCHS}  Seed: ${SEED}"
echo "  Save: ${SAVE_ROOT}"
echo "=============================================="

cd "${SCRIPT_DIR}"

# ── Baseline A: Clinical-only ──
echo -e "\n\n>>> [1/6] Baseline A: clinical_only"
${PYTHON} train.py \
    --model_mode clinical_only \
    --penalty_alpha 0 \
    --exp_name baseline_a_clinical_only \
    --save_dir "${SAVE_ROOT}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu_id ${GPU_ID} \
    2>&1 | tee "${SAVE_ROOT}/log_baseline_a.txt"

# ── Baseline B: Slice-level Mean Concat ──
echo -e "\n\n>>> [2/6] Baseline B: slice_mean_concat"
${PYTHON} train.py \
    --model_mode slice_mean_concat \
    --penalty_alpha 0 \
    --exp_name baseline_b_slice_mean_concat \
    --save_dir "${SAVE_ROOT}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu_id ${GPU_ID} \
    2>&1 | tee "${SAVE_ROOT}/log_baseline_b.txt"

# ── Ablation 1: Mean-MIL + Gated ──
echo -e "\n\n>>> [3/6] Ablation 1: mean_mil_gated"
${PYTHON} train.py \
    --model_mode mean_mil_gated \
    --penalty_alpha 1.0 \
    --exp_name ablation1_mean_mil_gated \
    --save_dir "${SAVE_ROOT}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu_id ${GPU_ID} \
    2>&1 | tee "${SAVE_ROOT}/log_ablation1.txt"

# ── Ablation 2: Attention-MIL + Concat ──
echo -e "\n\n>>> [4/6] Ablation 2: attention_mil_concat"
${PYTHON} train.py \
    --model_mode attention_mil_concat \
    --penalty_alpha 1.0 \
    --exp_name ablation2_attention_mil_concat \
    --save_dir "${SAVE_ROOT}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu_id ${GPU_ID} \
    2>&1 | tee "${SAVE_ROOT}/log_ablation2.txt"

# ── Ablation 3: Full Architecture, α=0 ──
echo -e "\n\n>>> [5/6] Ablation 3: attention_mil_gated (α=0)"
${PYTHON} train.py \
    --model_mode attention_mil_gated \
    --penalty_alpha 0 \
    --exp_name ablation3_no_penalty \
    --save_dir "${SAVE_ROOT}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu_id ${GPU_ID} \
    2>&1 | tee "${SAVE_ROOT}/log_ablation3.txt"

# ── Ours: Full Architecture, α=1.0 ──
echo -e "\n\n>>> [6/6] Ours: attention_mil_gated (α=1.0)"
${PYTHON} train.py \
    --model_mode attention_mil_gated \
    --penalty_alpha 1.0 \
    --exp_name ours_full \
    --save_dir "${SAVE_ROOT}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu_id ${GPU_ID} \
    2>&1 | tee "${SAVE_ROOT}/log_ours.txt"


echo -e "\n\n=============================================="
echo "  全部 6 组实验完成！"
echo "  结果目录: ${SAVE_ROOT}"
echo "=============================================="
