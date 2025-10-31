#!/usr/bin/env python3
"""
Extract LLM's natural language explanations for how it solved (or failed) selected ARC tasks.
This generates detailed explanations from the model about its reasoning process.
"""

import json
import asyncio
from pathlib import Path
import sys
sys.path.append('/home/ubuntu/arc-lang-public')

from openai import AsyncOpenAI

# Initialize local vLLM client
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

async def get_explanation_for_task(task_id: str, task_data: dict, solution: list, attempts: list, is_solved: bool) -> dict:
    """Get LLM's explanation for how it approached this task."""

    # Build prompt asking for explanation
    status = "correctly solved" if is_solved else "attempted but failed to solve"

    # Format training examples
    train_examples = []
    for i, example in enumerate(task_data['train'], 1):
        input_str = format_grid(example['input'])
        output_str = format_grid(example['output'])
        train_examples.append(f"Example {i}:\nInput:\n{input_str}\nOutput:\n{output_str}")

    train_text = "\n\n".join(train_examples)

    # Format test case
    test_input = format_grid(task_data['test'][0]['input'])
    ground_truth = format_grid(solution)

    # Format attempts
    attempt_text = ""
    for i, attempt in enumerate(attempts, 1):
        if f'attempt_{i}' in attempt and attempt[f'attempt_{i}']:
            attempt_grid = format_grid(attempt[f'attempt_{i}'])
            attempt_text += f"\n\nAttempt {i}:\n{attempt_grid}"

    prompt = f"""You are analyzing an ARC (Abstract Reasoning Corpus) task that you {status}.

Task ID: {task_id}

Training Examples:
{train_text}

Test Input:
{test_input}

Ground Truth Output:
{ground_truth}

Your Predictions:{attempt_text}

Please provide a detailed explanation addressing:
1. **Pattern Recognition**: What is the transformation rule? Describe the pattern you identified from the training examples.
2. **Your Approach**: How did you attempt to apply this rule to the test input?
3. **{'Success Factors' if is_solved else 'Failure Analysis'}**: {'What made this task solvable?' if is_solved else 'Why did this approach fail? What was missing from your understanding?'}
4. **Key Insights**: What are the critical elements of this transformation (colors, positions, shapes, symmetry, etc.)?

Write a clear, technical explanation as if you were explaining your reasoning to another AI researcher."""

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7,
        )

        explanation = response.choices[0].message.content
        return {
            "task_id": task_id,
            "status": "solved" if is_solved else "unsolved",
            "explanation": explanation,
            "ground_truth": solution,
            "attempts": attempts
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "solved" if is_solved else "unsolved",
            "explanation": f"Error generating explanation: {str(e)}",
            "error": str(e)
        }

def format_grid(grid: list) -> str:
    """Format a grid for display."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

async def main():
    print("Loading data files...")

    # Load files
    challenges_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_solutions.json")
    attempts_path = Path("/data/arclang/attempts/arc-prize-2024/arc-agi_evaluation_attempts.json")
    selected_tasks_path = Path("/home/ubuntu/arc-lang-public/analysis/selected_tasks.json")

    with open(challenges_path) as f:
        challenges = json.load(f)

    with open(solutions_path) as f:
        solutions = json.load(f)

    with open(attempts_path) as f:
        attempts = json.load(f)

    with open(selected_tasks_path) as f:
        selected_tasks = json.load(f)

    explanations = {
        "solved": [],
        "unsolved": []
    }

    # Process solved tasks
    print("\n=== Generating explanations for SOLVED tasks ===")
    for task_info in selected_tasks['solved']:
        task_id = task_info['id']
        print(f"Processing solved task: {task_id}")

        explanation = await get_explanation_for_task(
            task_id=task_id,
            task_data=challenges[task_id],
            solution=solutions[task_id][0] if task_id in solutions else [],
            attempts=attempts[task_id] if task_id in attempts else [],
            is_solved=True
        )
        explanations["solved"].append(explanation)
        print(f"  ✓ Generated explanation for {task_id}")

    # Process unsolved tasks
    print("\n=== Generating explanations for UNSOLVED tasks ===")
    for task_info in selected_tasks['unsolved']:
        task_id = task_info['id']
        print(f"Processing unsolved task: {task_id}")

        explanation = await get_explanation_for_task(
            task_id=task_id,
            task_data=challenges[task_id],
            solution=solutions[task_id][0] if task_id in solutions else [],
            attempts=attempts[task_id] if task_id in attempts else [],
            is_solved=False
        )
        explanations["unsolved"].append(explanation)
        print(f"  ✓ Generated explanation for {task_id}")

    # Save explanations
    output_path = Path("/home/ubuntu/arc-lang-public/analysis/llm_explanations.json")
    with open(output_path, 'w') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    print(f"\n✓ All explanations saved to: {output_path}")

    # Also create a readable markdown version
    md_path = Path("/home/ubuntu/arc-lang-public/analysis/llm_explanations.md")
    with open(md_path, 'w') as f:
        f.write("# GPT-OSS 20B의 ARC 문제 풀이 설명\n\n")
        f.write("이 문서는 GPT-OSS 20B 모델이 선택된 ARC 문제들을 어떻게 이해하고 풀었는지(또는 실패했는지)에 대한 자세한 설명을 담고 있습니다.\n\n")
        f.write("---\n\n")

        f.write("## 풀린 문제 (Solved Tasks)\n\n")
        for exp in explanations["solved"]:
            f.write(f"### Task {exp['task_id']} ✅\n\n")
            f.write(f"{exp['explanation']}\n\n")
            f.write("---\n\n")

        f.write("## 못 푼 문제 (Unsolved Tasks)\n\n")
        for exp in explanations["unsolved"]:
            f.write(f"### Task {exp['task_id']} ❌\n\n")
            f.write(f"{exp['explanation']}\n\n")
            f.write("---\n\n")

    print(f"✓ Readable markdown saved to: {md_path}")

if __name__ == "__main__":
    asyncio.run(main())
