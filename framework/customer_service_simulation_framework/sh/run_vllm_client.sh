# source /opt/rh/devtoolset-8/enable
export CUDA_VISIBLE_DEVICES=6,7
NODES_NUM=2
# MODEL_NAME=model/hybrid_rl/qwen25_7b_diverse_sft_fix_format_diversejson_general_prompt_data_grpo_8000_data_hybrid_reward/after_merge/global_step_120
MODEL_NAME=model/hybrid_rl/ab_budget_15percentage_qwen25_7b_general_prompt_data_grpo_8000_data_hybrid_reward/after_merge/global_step_120
set -x
# set -x
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8003 \
    --tensor-parallel-size $NODES_NUM \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --trust-remote-code