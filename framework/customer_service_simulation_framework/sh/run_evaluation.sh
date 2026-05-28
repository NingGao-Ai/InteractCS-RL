#!/bin/bash

# Customer service dialogue evaluation script

# Set working directory
WORKSPACE_DIR="<project_root>"
FRAMEWORK_DIR="${WORKSPACE_DIR}/framework/customer_service_simulation_framework"
CONFIG_FILE="${FRAMEWORK_DIR}/config/evaluation.yaml"

INPUT_FILE=framework/customer_service_simulation_framework/outputs/<your_experiment>/conversation.jsonl
OUTPUT_DIR=framework/customer_service_simulation_framework/outputs/<your_experiment>/evaluation

# Switch to framework directory
cd "${FRAMEWORK_DIR}" || exit 1

# Set PYTHONPATH
export PYTHONPATH="${FRAMEWORK_DIR}:${PYTHONPATH}"

# Run evaluation
echo "Starting evaluation..."
python main.py --config "${CONFIG_FILE}" \
    --session.Evaluation.input_file=$INPUT_FILE \
    --session.Evaluation.output_dir=$OUTPUT_DIR \
    --session.Evaluation.max_workers=80
