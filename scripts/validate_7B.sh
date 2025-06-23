#!/bin/bash

ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
src/open_r1/grpo.py \
--config recipes/Qwen2.5-7B/config_mot_eval.yaml \
--vllm_mode colocate
