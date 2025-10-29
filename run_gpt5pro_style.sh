#!/bin/bash
set -e

cd /home/ubuntu/arc-lang-public

export PYTHONPATH=/home/ubuntu/arc-lang-public
export MAX_CONCURRENCY=4
export LOCAL_LOGS_ONLY=1
export LOCAL_VLLM_URL=http://localhost:8000/v1

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Starting ARC experiment with GPT-5 Pro style config..."
echo "PYTHONPATH: $PYTHONPATH"
echo "MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "LOCAL_VLLM_URL: $LOCAL_VLLM_URL"
echo "Config: times=(2,2,3,2), max_tokens up to 20k"
echo ""

python -u src/run.py
