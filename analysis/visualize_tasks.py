#!/usr/bin/env python3
"""
Visualize selected ARC tasks with train examples, test input, ground truth, and model predictions.
Reference format: /home/ubuntu/TinyRecursiveModels/_archive_visualizations/trm_arc_visualizations/
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# ARC color palette (colors 0-9)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Light Blue
    '#870C25',  # 9: Dark Red
]

def plot_grid(ax, grid, title=''):
    """Plot a single grid with ARC colors."""
    grid = np.array(grid)
    height, width = grid.shape

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw cells
    for i in range(height):
        for j in range(width):
            color_idx = int(grid[i, j])
            color = ARC_COLORS[color_idx]
            rect = patches.Rectangle((j, height - i - 1), 1, 1,
                                     facecolor=color,
                                     edgecolor='white',
                                     linewidth=0.5)
            ax.add_patch(rect)

    # Add title
    if title:
        ax.text(width / 2, height + 0.5, title,
               ha='center', va='bottom', fontsize=10, fontweight='bold')

def visualize_task(task_id, challenges, solutions, attempts, output_path, is_solved):
    """Create visualization for a single task."""

    if task_id not in challenges:
        print(f"Task {task_id} not found in challenges")
        return

    task = challenges[task_id]
    train_examples = task['train']
    test_examples = task['test']

    # Get ground truth and predictions
    ground_truth = solutions.get(task_id, [])
    predictions = attempts.get(task_id, [])

    # Calculate layout
    n_train = len(train_examples)
    n_test = len(test_examples)

    # Create figure with subplots
    # Layout: Train examples (input/output pairs) in top rows, test examples in bottom rows
    fig = plt.figure(figsize=(20, 4 * (n_train + n_test)))

    status = "SOLVED" if is_solved else "UNSOLVED"
    fig.suptitle(f'Task {task_id} - {status}', fontsize=16, fontweight='bold', y=0.98)

    current_row = 0

    # Plot training examples
    for idx, example in enumerate(train_examples):
        input_grid = example['input']
        output_grid = example['output']

        # Input
        ax_in = plt.subplot(n_train + n_test, 4, current_row * 4 + 1)
        plot_grid(ax_in, input_grid, f'Train {idx+1} Input')

        # Output
        ax_out = plt.subplot(n_train + n_test, 4, current_row * 4 + 2)
        plot_grid(ax_out, output_grid, f'Train {idx+1} Output')

        current_row += 1

    # Plot test examples
    for idx, example in enumerate(test_examples):
        test_input = example['input']

        # Test Input
        ax_test = plt.subplot(n_train + n_test, 4, current_row * 4 + 1)
        plot_grid(ax_test, test_input, f'Test {idx+1} Input')

        # Ground Truth
        if idx < len(ground_truth):
            ax_truth = plt.subplot(n_train + n_test, 4, current_row * 4 + 2)
            plot_grid(ax_truth, ground_truth[idx], 'Ground Truth')

        # Model Prediction 1
        if idx < len(predictions) and predictions[idx].get('attempt_1'):
            ax_pred1 = plt.subplot(n_train + n_test, 4, current_row * 4 + 3)
            pred1 = predictions[idx]['attempt_1']
            # Check if correct
            is_correct_1 = (idx < len(ground_truth) and pred1 == ground_truth[idx])
            title = 'Prediction 1 ✓' if is_correct_1 else 'Prediction 1 ✗'
            plot_grid(ax_pred1, pred1, title)

        # Model Prediction 2
        if idx < len(predictions) and predictions[idx].get('attempt_2'):
            ax_pred2 = plt.subplot(n_train + n_test, 4, current_row * 4 + 4)
            pred2 = predictions[idx]['attempt_2']
            # Check if correct
            is_correct_2 = (idx < len(ground_truth) and pred2 == ground_truth[idx])
            title = 'Prediction 2 ✓' if is_correct_2 else 'Prediction 2 ✗'
            plot_grid(ax_pred2, pred2, title)

        current_row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {output_path}")

def main():
    # Load data
    challenges_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("/home/ubuntu/arc-lang-public/data/arc-prize-2024/arc-agi_evaluation_solutions.json")
    attempts_path = Path("/data/arclang/attempts/arc-prize-2024/arc-agi_evaluation_attempts.json")
    selected_tasks_path = Path("/home/ubuntu/arc-lang-public/analysis/selected_tasks.json")

    print("Loading data files...")
    with open(challenges_path) as f:
        challenges = json.load(f)

    with open(solutions_path) as f:
        solutions = json.load(f)

    with open(attempts_path) as f:
        attempts = json.load(f)

    with open(selected_tasks_path) as f:
        selected_tasks = json.load(f)

    # Create visualizations for solved tasks
    print("\n=== Creating visualizations for SOLVED tasks ===")
    for task_info in selected_tasks['solved']:
        task_id = task_info['id']
        output_path = f"/home/ubuntu/arc-lang-public/analysis/solved_{task_id}.png"
        visualize_task(task_id, challenges, solutions, attempts, output_path, is_solved=True)

    # Create visualizations for unsolved tasks
    print("\n=== Creating visualizations for UNSOLVED tasks ===")
    for task_info in selected_tasks['unsolved']:
        task_id = task_info['id']
        output_path = f"/home/ubuntu/arc-lang-public/analysis/unsolved_{task_id}.png"
        visualize_task(task_id, challenges, solutions, attempts, output_path, is_solved=False)

    print("\n✓ All visualizations completed!")
    print(f"Output directory: /home/ubuntu/arc-lang-public/analysis/")

if __name__ == "__main__":
    main()
