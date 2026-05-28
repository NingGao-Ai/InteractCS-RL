#!/bin/bash

# Customer Service Dialogue Evaluation Script
# Use --key=value format to specify parameters directly

set -x

INPUT_FILE="data/customer_service_evaluation/qwen25_32b/customer_service_conversations_20251104_095854.jsonl"
OUTPUT_DIR="data/customer_service_evaluation/qwen25_32b"


# Project root directory
PROJECT_ROOT="<project_root>"
cd "$PROJECT_ROOT/code/customer_service_evaluation"

# Activate conda environment
# Use --key=value format to specify parameters directly
echo "Starting customer service dialogue evaluation"
echo "Executing command: python ./main.py $@"
echo ""

python ./main.py \
    --input.input_file=$INPUT_FILE \
    --input.max_conversations=100 \
    --llm.client_type="openai" \
    --llm.model="gpt-4.1" \
    --llm.api_url="<your_api_url>" \
    --llm.api_key="<your_api_key>" \
    --llm.temperature=0 \
    --llm.max_tokens=6000 \
    --llm.timeout=180 \
    --llm.max_retries=3 \
    --prompts.speech_evaluation_prompt_file="code/prompt/evaluation-speech.txt" \
    --prompts.logic_evaluation_prompt_file="code/prompt/evaluation-logic.txt" \
    --prompts.compensation_evaluation_prompt_file="code/prompt/evaluation-compensation.txt" \
    --parallel.max_workers=20 \
    --parallel.batch_size=20 \
    --output.evaluation_dir="" \
    --output.save_format="jsonl" \
    "$@"
