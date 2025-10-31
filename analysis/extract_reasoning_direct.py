#!/usr/bin/env python3
"""
Direct vLLM API call to extract reasoning for specific ARC tasks.
Uses the same vLLM server (GPU 0,1) - will queue behind current experiment.
"""

import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

# Initialize vLLM client (same server as main experiment)
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

def format_grid(grid):
    """Format grid for display."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

async def get_reasoning_for_task(task_id: str, task_data: dict, solution: list, is_solved: bool):
    """Get GPT-OSS reasoning for a specific task."""

    # Format training examples
    train_examples = []
    for i, example in enumerate(task_data['train'], 1):
        input_str = format_grid(example['input'])
        output_str = format_grid(example['output'])
        train_examples.append(f"Example {i}:\nInput:\n{input_str}\nOutput:\n{output_str}")

    train_text = "\n\n".join(train_examples)
    test_input = format_grid(task_data['test'][0]['input'])
    ground_truth = format_grid(solution) if solution else "N/A"

    # Create prompt asking for reasoning
    prompt = f"""You are solving an ARC (Abstract Reasoning Corpus) task.

Task ID: {task_id}

Training Examples:
{train_text}

Test Input:
{test_input}

Ground Truth Output:
{ground_truth}

Please provide a detailed explanation of:
1. **Pattern Recognition**: What is the transformation rule from input to output?
2. **Analysis**: Describe the pattern step-by-step
3. **Application**: How would you apply this rule to the test input?
4. **Reasoning**: Explain your thought process in detail

Provide your reasoning and then the final grid output in JSON format."""

    try:
        print(f"\n{'='*60}")
        print(f"Requesting reasoning for {task_id} ({'SOLVED' if is_solved else 'UNSOLVED'})")
        print(f"{'='*60}")

        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16000,  # Higher limit for reasoning
            temperature=0.3,
            # NO JSON mode - let model generate natural text with reasoning
        )

        # Extract reasoning if available
        reasoning_text = getattr(response.choices[0].message, 'reasoning', None)
        content_text = response.choices[0].message.content

        print(f"\n✓ Response received for {task_id}")
        print(f"  - Finish reason: {response.choices[0].finish_reason}")
        print(f"  - Content length: {len(content_text) if content_text else 0} chars")
        print(f"  - Reasoning available: {reasoning_text is not None}")
        if reasoning_text:
            print(f"  - Reasoning length: {len(reasoning_text)} chars")

        return {
            "task_id": task_id,
            "status": "solved" if is_solved else "unsolved",
            "reasoning": reasoning_text,
            "response_content": content_text,
            "finish_reason": response.choices[0].finish_reason,
            "ground_truth": solution
        }

    except Exception as e:
        print(f"\n✗ Error for {task_id}: {str(e)}")
        return {
            "task_id": task_id,
            "status": "solved" if is_solved else "unsolved",
            "error": str(e)
        }

async def main():
    print("="*60)
    print("EXTRACTING REASONING FROM VLLM (GPU 0,1)")
    print("Main experiment (PID 45157) continues - using vLLM queue")
    print("="*60)

    # Load data
    challenges_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_solutions.json")

    with open(challenges_path) as f:
        challenges = json.load(f)

    with open(solutions_path) as f:
        solutions = json.load(f)

    # Select 2 tasks: 1 solved, 1 unsolved
    tasks_to_extract = [
        {"id": "833dafe3", "is_solved": True},   # Solved (from visualization)
        {"id": "ad7e01d0", "is_solved": False},  # Unsolved (from visualization)
    ]

    results = []

    # Process tasks in parallel (vLLM will queue them)
    tasks = []
    for task_info in tasks_to_extract:
        task_id = task_info["id"]
        is_solved = task_info["is_solved"]

        task = get_reasoning_for_task(
            task_id=task_id,
            task_data=challenges[task_id],
            solution=solutions.get(task_id, [None])[0],
            is_solved=is_solved
        )
        tasks.append(task)

    # Execute in parallel
    results = await asyncio.gather(*tasks)

    # Save results
    output_path = Path("/home/ubuntu/arc-lang-public/analysis/llm_reasoning_extracted.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    for result in results:
        task_id = result['task_id']
        status = result['status']
        has_reasoning = 'reasoning' in result and result['reasoning'] is not None

        print(f"\n{task_id} ({status.upper()}):")
        print(f"  - Reasoning extracted: {'YES' if has_reasoning else 'NO'}")
        if has_reasoning:
            print(f"  - Reasoning length: {len(result['reasoning'])} chars")
            print(f"  - Preview: {result['reasoning'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
