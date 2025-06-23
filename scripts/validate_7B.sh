#!/bin/bash

ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
src/open_r1/grpo.py \
--config recipes/Qwen2.5-7B/config_mot_eval.yaml \
--output_dir "/mnt/ai_researcher_volume_nyc2_1750677290711/Qwen2.5-7B-MoT-GRPO-20250623_003621" \
--vllm_mode colocate
