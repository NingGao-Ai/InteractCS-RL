#!/bin/bash

# Customer Service Evaluation Framework - Main Script
# Use --key=value format to specify parameters directly

set -x

# Project root directory
PROJECT_ROOT="<project_root>"
cd "$PROJECT_ROOT/code/customer_service_evaluation"

# Use --key=value format to specify parameters directly
echo "Starting customer service dialogue simulation"
echo "Executing command: python ./main.py "
echo ""

python ./main.py \
    --user_llm.client_type="openai" \
    --user_llm.model="gpt-4.1" \
    --user_llm.api_url="<your_api_url>" \
    --user_llm.api_key="<your_api_key>" \
    --user_llm.temperature=1.2 \
    --user_llm.max_tokens=8000 \
    --user_llm.timeout=180 \
    --user_llm.max_retries=3 \
    --customer_llm.client_type="vllm" \
    --customer_llm.model="<project_root>/verl/huggingface.co/Qwen/Qwen2.5-7B-Instruct" \
    --customer_llm.api_url="http://localhost:8000/v1" \
    --customer_llm.temperature=0.7 \
    --customer_llm.max_tokens=8000 \
    --customer_llm.timeout=180 \
    --customer_llm.max_retries=3 \
    --user_simulator.prompt_file="code/prompt/user-simulator-prompt.txt" \
    --user_simulator.user_profiles_file="data/user_roleplay_data/user_roleplay_data_stage_6_user_profiles.json" \
    --user_simulator.system_signals_file="data/user_roleplay_data/user_roleplay_data_stage_7_system_signals_pool.jsonl" \
    --user_simulator.core_need_file="data/user_roleplay_data/user_roleplay_data_stage_8_core_need.json" \
    --customer_service.prompt_file="code/prompt/customer-service-prompt-v2.txt" \
    --parallel.max_workers=10 \
    --parallel.batch_size=10 \
    --parallel.max_turns_per_conversation=20 \
    --output.output_dir="data/customer_service_evaluation/qwen25-7b" \
    --output.save_format="json" \
    --simulation.num_conversations=80 \
    --simulation.user_category_distribution='{"1": 5, "2": 25, "3": 25, "4": 20, "5": 5}' \

