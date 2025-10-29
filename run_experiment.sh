#!/bin/bash

# ARC Experiment Runner Script
set -e

cd /home/ubuntu/arc-lang-public

export PYTHONPATH=/home/ubuntu/arc-lang-public
export MAX_CONCURRENCY=20
export LOCAL_LOGS_ONLY=1
export LOCAL_VLLM_URL=http://localhost:8000/v1

# Load .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Starting ARC experiment..."
echo "PYTHONPATH: $PYTHONPATH"
echo "MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "LOCAL_VLLM_URL: $LOCAL_VLLM_URL"
echo ""

python -u src/run.py
