#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# === Configuration ===
NUM_GPUS=1
MODEL_NAME=Qwen2.5-1.5B-MoT-GRPO-2025-06-14-22-39/checkpoint-2200
MODEL=data/$MODEL_NAME
TASK=livecodebenchv5
OUTPUT_DIR=data/evals/$MODEL_NAME/$TASK

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# === Run Evaluation ===
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR