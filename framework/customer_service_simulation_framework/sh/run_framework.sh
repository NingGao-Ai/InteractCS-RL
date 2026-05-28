#!/bin/bash

# Set working directory
WORKSPACE_DIR="<project_root>"
FRAMEWORK_DIR="${WORKSPACE_DIR}/framework/customer_service_simulation_framework"
CONFIG_FILE="${FRAMEWORK_DIR}/config/vllm_customer.yaml"


# Switch to framework directory
cd "${FRAMEWORK_DIR}" || exit 1

# Set PYTHONPATH
export PYTHONPATH="${FRAMEWORK_DIR}:${PYTHONPATH}"

# MODEL_NAME=model/hybrid_rl/qwen25_7b_diverse_sft_fix_format_diversejson_general_prompt_data_grpo_8000_data_hybrid_reward/after_merge/global_step_120
# MODEL_NAME=model/hybrid_rl/qwen25_7b_diverse_sft_fix_format_diversejson_general_prompt_data_grpo_8000_data_hybrid_reward/after_merge/global_step_120
MODEL_NAME=model/hybrid_rl/ab_budget_15percentage_qwen25_7b_general_prompt_data_grpo_8000_data_hybrid_reward/after_merge/global_step_120
OUTPUT_DIR=framework/customer_service_simulation_framework/outputs/ab_budget_15percentage_qwen25_7b_general_prompt_data_grpo_8000_data_hybrid_reward

# Run main program
python main.py --config "${CONFIG_FILE}" \
    --agent.customer.llm.model=$MODEL_NAME \
    --session.RLSimulation.output_dir=$OUTPUT_DIR \
    --agent.RLSimulation.max_workers=20

