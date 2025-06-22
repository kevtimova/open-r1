#!/bin/bash

NUM_GPUS=8

export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_NAME=Qwen2.5-1.5B-MoT-GRPO-20250622_153449/checkpoint-20
MODEL=data/$MODEL_NAME
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL_NAME

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
--use-chat-template \
--output-dir $OUTPUT_DIR