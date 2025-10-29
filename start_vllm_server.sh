#!/bin/bash

# vLLM Server Startup Script for GPT-OSS 20B
# Runs on GPU 0, 1 with tensor parallelism

set -e

echo "==================================================="
echo "Starting vLLM Server for GPT-OSS 20B"
echo "==================================================="

# Configuration
MODEL_NAME="openai/gpt-oss-20b"
GPUS="0,1"
TENSOR_PARALLEL=2
PORT=8000
LOG_DIR="/data/arclang/logs"
LOG_FILE="$LOG_DIR/vllm_server_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPUs: $GPUS"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "  Port: $PORT"
echo "  Log: $LOG_FILE"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "vLLM not found. Installing..."
    pip install vllm
else
    echo "vLLM is already installed."
fi

echo ""
echo "Starting vLLM server..."
echo "View logs: tail -f $LOG_FILE"
echo ""

# Start vLLM server with optimized memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 128 \
    --disable-custom-all-reduce \
    2>&1 | tee "$LOG_FILE"
