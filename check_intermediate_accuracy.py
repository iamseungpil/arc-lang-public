#!/usr/bin/env python3
"""Check accuracy of completed challenges so far"""
import json
from pathlib import Path
from pydantic import TypeAdapter

# Paths
attempts_path = Path("/data/arclang/attempts/arc-prize-2024/arc-agi_evaluation_attempts.json")
solutions_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_solutions.json")

# Load data
if not attempts_path.exists():
    print("No attempts file found yet")
    exit(0)

attempts_data = attempts_path.read_text()
if not attempts_data or attempts_data == "{}":
    print("Attempts file is empty")
    exit(0)

attempts = json.loads(attempts_data)
solutions = json.loads(solutions_path.read_text())

# Calculate accuracy
total_count = 0
correct_count = 0
completed_challenges = 0

for challenge_id, attempt_list in attempts.items():
    if challenge_id not in solutions:
        continue
    
    completed_challenges += 1
    truth_grids = solutions[challenge_id]
    
    for i, truth_grid in enumerate(truth_grids):
        if i >= len(attempt_list):
            continue
            
        total_count += 1
        attempt = attempt_list[i]
        
        # Check both attempts
        if attempt.get("attempt_1") == truth_grid:
            correct_count += 1
        elif attempt.get("attempt_2") == truth_grid:
            correct_count += 1

print(f"\n=== Intermediate Accuracy ===")
print(f"Completed challenges: {completed_challenges} / 400")
print(f"Total test cases: {total_count}")
print(f"Correct: {correct_count}")
if total_count > 0:
    accuracy = correct_count / total_count * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No test cases completed yet")
