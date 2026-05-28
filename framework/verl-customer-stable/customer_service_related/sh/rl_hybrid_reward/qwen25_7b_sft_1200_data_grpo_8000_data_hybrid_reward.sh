#!/bin/bash
# Customer service dialogue PPO training script
# Supports both Friday API and vLLM reward model backends

# source /opt/rh/devtoolset-8/enable

# Cannot find libcudnn_adv.so.9
# export LD_LIBRARY_PATH=<python_env>/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# Prioritize loading PyTorch core libraries (includes libcusparse.so.12, etc.)
# export LD_LIBRARY_PATH=<python_env>/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
# Add NVIDIA JIT link library path (provides __nvJitLinkComplete_12_4 symbols, etc.)
# export LD_LIBRARY_PATH=<python_env>/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# Add CUDA runtime library path (provides GPU runtime support)
# export LD_LIBRARY_PATH=<python_env>/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PYTHONPATH=framework/verl-customer-stable:$PYTHONPATH
export RAY_DEBUG_POST_MORTEM=0
set -x

# ============================================================================
# Configuration Parameters
# ============================================================================

# Data paths
DATA_DIR=${DATA_DIR:-"data/rl_training_data_paper/8000_sample_ordered_no_transfer"}
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"
# /rl_training_data_paper
LOG_DIR="framework/verl-customer-stable/customer_service_related/log_dir"
OUTPUT_DIR="framework/verl-customer-stable/customer_service_related/output"
TRAINING_NAME="qwen25_7b_diverse_sft_fix_format_diversejson_general_prompt_data_grpo_8000_data_hybrid_reward"
VALID_DATA_DIR="$LOG_DIR/$TRAINING_NAME/valid_data"
ROLLOUT_DATA_DIR="$LOG_DIR/$TRAINING_NAME/rollout_data"
TENSORBOARD_DIR="$LOG_DIR/$TRAINING_NAME/tensorboard"

MODEL_SAVE_DIR="model/hybrid_rl/$TRAINING_NAME/"

# Model paths
# ACTOR_MODEL="model/customer_service_rl/qwen25-7b-sft-v2_data-930_genrm-friday-dpsk-3.2_rule-condition_algo-gspo-lr-warmup/epoch2" 
ACTOR_MODEL="model/qwen25-7b-diverse-sft-fix-format_diversejson" 
# ACTOR_MODEL="model/qwen25-7b-diverse-sft-fix-format"
# model/qwen25-7b-sft-1200-data
# model/qwen25-14b-diverse-sft-fix-format

# Training parameters
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=128
LEARNING_RATE=1e-6
TOTAL_EPOCHS=2
SAVE_FREQ=40
TEST_FREQ=5

# GPU configuration
CUDA_DEVICES="0,1,2,3,4,5,6,7"
N_GPUS=8
NNODES=1


# ============================================================================
# Set Reward function environment variables
REWARD_BACKEND=${REWARD_BACKEND:-"friday"}  # friday or vllm
REWARD_MODEL=${REWARD_MODEL:-"deepseek-chat"}     # Friday API model name or vLLM model name
REWARD_API_KEY=${REWARD_API_KEY:-"<your_api_key>"}  # Friday API key
REWARD_API_URL=${REWARD_API_URL:-"<your_api_url>"}  # API URL
REWARD_MAX_WORKERS=${REWARD_MAX_WORKERS:-512}  # Concurrency


echo "============================================================================"
echo "Training Configuration"
echo "============================================================================"
echo "Reward Backend: ${REWARD_BACKEND}"
echo "Reward Model: ${REWARD_MODEL}"
echo "Reward API URL: ${REWARD_API_URL}"
echo "Reward Max Workers: ${REWARD_MAX_WORKERS}"
echo ""
echo "Data Directory: ${DATA_DIR}"
echo "Train File: ${TRAIN_FILE}"
echo "Val File: ${VAL_FILE}"
echo ""
echo "Actor Model: ${ACTOR_MODEL}"
echo "Train Batch Size: ${TRAIN_BATCH_SIZE}"
echo "PPO Mini Batch Size: ${PPO_MINI_BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Total Epochs: ${TOTAL_EPOCHS}"
echo ""
echo "CUDA Devices: ${CUDA_DEVICES}"
echo "N GPUs: ${N_GPUS}"
echo "N Nodes: ${NNODES}"
echo ""
echo "Project Name: ${PROJECT_NAME}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "============================================================================"

# ============================================================================
# Check data files
# ============================================================================

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Error: Training file does not exist: ${TRAIN_FILE}"
    exit 1
fi

if [ ! -f "${VAL_FILE}" ]; then
    echo "Error: Validation file does not exist: ${VAL_FILE}"
    exit 1
fi

echo "Data file check passed"

# ============================================================================
# Start training
# ============================================================================
export REWARD_BACKEND
export REWARD_MODEL
export REWARD_API_KEY
export REWARD_API_URL
export REWARD_MAX_WORKERS

export RAY_DEDUP_LOGS=1 
export HYDRA_FULL_ERROR=1
export LOGLEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export TENSORBOARD_DIR=${TENSORBOARD_DIR}
    
    # nohup \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=4000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.model.path=${ACTOR_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=framework/verl-customer-stable/customer_service_related/config/interaction_config_vllm_no_transfer.yaml \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=customer_batch \
    custom_reward_function.path=framework/verl-customer-stable/customer_service_related/reward_function/hybrid_reward_batch.py \
    custom_reward_function.name=compute_hybrid_score_batch \
    trainer.logger='["console","tensorboard"]' \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=True \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode='resume_path' \
    trainer.resume_from_path="model/hybrid_rl/qwen25_7b_diverse_sft_fix_format_diversejson_general_prompt_data_grpo_8000_data_hybrid_reward/global_step_80" \
    trainer.validation_data_dir=${VALID_DATA_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.default_local_dir=${MODEL_SAVE_DIR} \
    trainer.log_val_generations=1
    # > "$OUTPUT_DIR/$TRAINING_NAME.log" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo $TRAIN_PID > "$OUTPUT_DIR/training.pid"
echo "To stop training: kill $TRAIN_PID"

    # actor_rollout_ref.model.enable_gradient_checkpointing=True \
    # actor_rollout_ref.model.enable_activation_offload=True \
    # actor_rollout_ref.actor.fsdp_config.param_offload=True \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \

    # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    # actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \