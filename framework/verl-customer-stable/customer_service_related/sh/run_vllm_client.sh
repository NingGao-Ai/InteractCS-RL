# source /opt/rh/devtoolset-8/enable
export CUDA_VISIBLE_DEVICES=4,5
NODES_NUM=2
MODEL_NAME=<cluster_path>/banma_aigc/model/huggingface.co/Qwen/Qwen2.5-32B-Instruct

LOG_FILE="framework/verl-customer-stable/customer_service_related/output/vllm_output/vllm_server_2.log"
PID_FILE="framework/verl-customer-stable/customer_service_related/output/vllm_output/vllm_server_2.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat $PID_FILE)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "vLLM Server is already running with PID: $OLD_PID"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f $PID_FILE
    fi
fi

set -x

# Start service
nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8089 \
    --tensor-parallel-size $NODES_NUM \
    --gpu-memory-utilization 0.8 \
    --max-num-batched-tokens 65536 \
    --max-model-len 32768 \
    --max-num-seqs 512 \
    --disable-log-requests \
    --trust-remote-code \
    > $LOG_FILE 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > $PID_FILE

echo "=========================================="
echo "vLLM Server started successfully!"
echo "PID: $VLLM_PID"
echo "Log: $LOG_FILE"
echo "=========================================="
echo ""
echo "To view logs: tail -f $LOG_FILE"
echo "To stop server: kill $VLLM_PID"
echo "Or run: kill \$(cat $PID_FILE)"