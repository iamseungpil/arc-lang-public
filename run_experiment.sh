#!/bin/bash

# ARC Experiment Runner Script
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"
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
