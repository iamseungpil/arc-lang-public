#!/bin/bash

# Continuous progress monitor for ARC validation experiment

LOG_FILE="/data/arclang/logs/arc_validation_400_final.log"

while true; do
    clear
    echo "=== ARC Validation 400 - Live Progress Monitor ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check if process is running
    if ps aux | grep -q "[p]ython -u src/run.py"; then
        echo "✓ Experiment process is RUNNING"
    else
        echo "✗ Experiment process is NOT RUNNING"
    fi

    # Check vLLM server
    if ps aux | grep -q "[v]llm.entrypoints.openai.api_server"; then
        echo "✓ vLLM server is RUNNING"
    else
        echo "✗ vLLM server is NOT RUNNING"
    fi

    echo ""
    echo "=== Challenge Progress ==="

    # Count challenges started
    STARTED=$(grep -c "Starting to solve challenge:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "Challenges started: $STARTED"

    # Count challenges completed
    COMPLETED=$(grep -c "completed!" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "Challenges completed: $COMPLETED / 400"

    # Calculate percentage
    if [ "$COMPLETED" -gt 0 ]; then
        PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/400)*100}")
        echo "Progress: $PERCENT%"
    fi

    # Show latest completed
    LATEST_COMPLETED=$(grep "completed!" "$LOG_FILE" 2>/dev/null | tail -1 || echo "None yet")
    echo "Latest completed: $LATEST_COMPLETED"

    # Count correct
    CORRECT_LINE=$(grep "correct count" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
    if [ -n "$CORRECT_LINE" ]; then
        echo "Latest result: $CORRECT_LINE"
    fi

    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -2

    echo ""
    echo "=== Recent Log (last 5 lines) ==="
    tail -5 "$LOG_FILE" 2>/dev/null || echo "No log yet"

    echo ""
    echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
