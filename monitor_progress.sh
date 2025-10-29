#!/bin/bash

# Monitor ARC validation experiment progress

LOG_FILE="/data/arclang/logs/arc_validation_400_unbuffered.log"
ATTEMPTS_FILE="/data/arclang/attempts/arc-prize-2024/arc-agi_evaluation_attempts.json"

echo "=== ARC Validation 400 Progress Monitor ==="
echo "Log file: $LOG_FILE"
echo "Attempts file: $ATTEMPTS_FILE"
echo ""

# Check if process is running
if ps aux | grep -q "[p]ython -u src/run.py"; then
    echo "✓ Experiment process is running"
else
    echo "✗ Experiment process is NOT running"
fi

# Check vLLM server
if ps aux | grep -q "[v]llm.entrypoints.openai.api_server"; then
    echo "✓ vLLM server is running"
else
    echo "✗ vLLM server is NOT running"
fi

echo ""
echo "=== Challenge Progress ==="

# Count challenges started
STARTED=$(grep -c "Starting to solve challenge:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Challenges started: $STARTED"

# Count challenges completed
COMPLETED=$(grep -c "completed!" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Challenges completed: $COMPLETED"

# Count correct solutions
CORRECT=$(grep "correct count" "$LOG_FILE" 2>/dev/null | tail -1 || echo "No results yet")
echo "Latest result: $CORRECT"

# Show last few log lines
echo ""
echo "=== Recent Log (last 10 lines) ==="
tail -10 "$LOG_FILE"
