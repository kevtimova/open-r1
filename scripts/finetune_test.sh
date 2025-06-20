#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export RUN_ID=$(date +%Y%m%d_%H%M%S)

ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero3_test.yaml \
src/open_r1/grpo.py \
--config recipes/Qwen2.5-1.5B/grpo/config_mot_test.yaml \
--output_dir "data/Qwen2.5-1.5B-MoT-GRPO-$RUN_ID" \
--vllm_mode colocate
