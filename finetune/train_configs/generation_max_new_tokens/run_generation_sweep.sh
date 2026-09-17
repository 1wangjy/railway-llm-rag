#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data16T/wjy/Learn_llm/铁路大模型/constrast_infer
EXP=/data16T/wjy/Learn_llm/铁路大模型/new_7b/train_configs/generation_max_new_tokens
MODEL=/data16T/wjy/models/Qwen2.5-7B-Instruct/qwen25_7b_data7236_lr3e-4_r64alpha64_warmup50_seed42_from_base_best
PYTHON=/data16T/wjy/anaconda3/envs/unsloth/bin/python

cd "$PROJECT"
mkdir -p "$EXP/logs" "$EXP/outputs"

for TOKENS in 256 512 1024; do
  LOG="$EXP/logs/qwen25_7b_seed42_max${TOKENS}.log"
  OUTPUT="$EXP/outputs/qwen25_7b_seed42_max${TOKENS}_predictions.json"
  echo "[$(date '+%F %T')] start max_new_tokens=${TOKENS}" | tee -a "$LOG"
  env -u LD_LIBRARY_PATH \
    INFER_CUDA_VISIBLE_DEVICES=4 \
    PYTHONUNBUFFERED=1 \
    "$PYTHON" -u run_infer.py \
    --model qwen25_7b \
    --model_path "$MODEL" \
    --max_samples 799 \
    --max_new_tokens "$TOKENS" \
    --max_seq_length 2048 \
    --output_path "$OUTPUT" \
    >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] finished max_new_tokens=${TOKENS}" | tee -a "$LOG"
done
