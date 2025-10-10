#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
rollout=8


top_p_perception_tokens=0.4
advantage_scaling_min=0.9
entropy_penalty_coef=0.06


EXP_NAME="perc${top_p_perception_tokens}_advsc${advantage_scaling_min}_pen${entropy_penalty_coef}_ep${TOTAL_EPOCHES}_rollout${rollout}"

CONGI_FILE="examples/configs/config.yaml"
TRAIN_FILE="chamber111/VPPO_ViRL39K_train"
VAL_FILE="chamber111/VPPO_MMK12_validation"

FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    trainer.project_name="7b_vppo" \
    trainer.logger=['console','swanlab'] \
    algorithm.use_vppo_on_entropy=False \
    algorithm.use_vppo_on_perception=True \
    algorithm.use_advantage_shaping=True \
    algorithm.use_entropy_penalty=True \
    algorithm.top_p_perception_tokens=${top_p_perception_tokens} \
    algorithm.entropy_penalty_coef=${entropy_penalty_coef} \
    algorithm.advantage_scaling_min=${advantage_scaling_min} \
    worker.rollout.n=${rollout} \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.micro_batch_size_per_device_for_update=16