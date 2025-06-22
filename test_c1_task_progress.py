#!/usr/bin/env python3
"""
Test script for C-1 task progress measurement.

This script demonstrates how C-1 contribution rewards measure the degree to which
each action reduces the distance to solving the overall task through state-value differences.
"""

import os
import sys
sys.path.append('.')

from verl.utils.contribution import contrib_value_delta, contrib_value_delta_advanced


def test_basic_task_progress():
    """Test basic task progress measurement."""
    print("=== Basic Task Progress Measurement ===")
    
    # Simulate a multi-step task where the agent makes progress
    steps = [
        (0.0, 0.2),   # Initial progress
        (0.2, 0.4),   # More progress
        (0.4, 0.6),   # Continued progress
        (0.6, 0.8),   # Near completion
        (0.8, 1.0),   # Task completed
    ]
    
    for i, (prev_v, cur_v) in enumerate(steps):
        contribution = contrib_value_delta(prev_v, cur_v, step=i)
        progress = cur_v - prev_v
        print(f"Step {i}: {prev_v:.1f} → {cur_v:.1f} (progress: {progress:.1f}, contribution: {contribution:.3f})")


def test_task_complexity_scaling():
    """Test how task complexity affects contribution rewards."""
    print("\n=== Task Complexity Scaling ===")
    
    # Same value improvement, different task complexities
    prev_v, cur_v = 0.3, 0.5
    value_improvement = cur_v - prev_v
    
    complexities = [0.5, 1.0, 2.0, 5.0]
    
    for complexity in complexities:
        contribution = contrib_value_delta(prev_v, cur_v, step=0, task_complexity=complexity)
        print(f"Complexity {complexity}: improvement {value_improvement:.1f} → contribution {contribution:.3f}")


def test_step_based_scaling():
    """Test how step number affects contribution rewards."""
    print("\n=== Step-Based Scaling ===")
    
    # Same value improvement at different steps
    prev_v, cur_v = 0.4, 0.6
    value_improvement = cur_v - prev_v
    
    steps = [0, 10, 25, 50, 75, 100]
    
    for step in steps:
        contribution = contrib_value_delta(prev_v, cur_v, step=step)
        print(f"Step {step}: improvement {value_improvement:.1f} → contribution {contribution:.3f}")


def test_advanced_task_progress():
    """Test advanced task progress with task-specific information."""
    print("\n=== Advanced Task Progress ===")
    
    # Multi-step task
    multi_step_info = {
        'task_type': 'multi_step',
        'expected_steps': 50,
        'current_progress': 0.3
    }
    
    # Exploration task
    exploration_info = {
        'task_type': 'exploration',
        'expected_steps': 20,
        'current_progress': 0.1
    }
    
    # General task
    general_info = {
        'task_type': 'general',
        'expected_steps': 30,
        'current_progress': 0.5
    }
    
    prev_v, cur_v = 0.3, 0.6
    value_improvement = cur_v - prev_v
    
    for task_type, task_info in [('multi_step', multi_step_info), 
                                ('exploration', exploration_info), 
                                ('general', general_info)]:
        contribution = contrib_value_delta_advanced(prev_v, cur_v, step=10, task_info=task_info)
        print(f"{task_type}: improvement {value_improvement:.1f} → contribution {contribution:.3f}")


def test_no_progress_scenarios():
    """Test scenarios where no progress is made."""
    print("\n=== No Progress Scenarios ===")
    
    scenarios = [
        (0.5, 0.3, "Value decreased"),
        (0.5, 0.5, "No change"),
        (0.8, 0.7, "Slight regression"),
    ]
    
    for prev_v, cur_v, description in scenarios:
        contribution = contrib_value_delta(prev_v, cur_v, step=0)
        print(f"{description}: {prev_v:.1f} → {cur_v:.1f} → contribution {contribution:.3f}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Edge Cases ===")
    
    edge_cases = [
        (None, 0.5, "None prev_v"),
        (0.5, None, "None cur_v"),
        ("0.3", "0.6", "String values"),
        (0.0, 0.0, "Zero values"),
        (-0.5, 0.5, "Negative to positive"),
    ]
    
    for prev_v, cur_v, description in edge_cases:
        try:
            contribution = contrib_value_delta(prev_v, cur_v, step=0)
            print(f"{description}: {prev_v} → {cur_v} → contribution {contribution:.3f}")
        except Exception as e:
            print(f"{description}: {prev_v} → {cur_v} → ERROR: {e}")


def simulate_task_episode():
    """Simulate a complete task episode with C-1 rewards."""
    print("\n=== Complete Task Episode Simulation ===")
    
    # Simulate a complex task with varying progress rates
    episode_data = [
        (0.0, 0.1, 0),    # Initial exploration
        (0.1, 0.15, 1),   # Small progress
        (0.15, 0.15, 2),  # No progress (exploration)
        (0.15, 0.25, 3),  # Breakthrough
        (0.25, 0.35, 4),  # Continued progress
        (0.35, 0.35, 5),  # Plateau
        (0.35, 0.45, 6),  # Another breakthrough
        (0.45, 0.6, 7),   # Major progress
        (0.6, 0.7, 8),    # Steady progress
        (0.7, 0.85, 9),   # Near completion
        (0.85, 1.0, 10),  # Task completed
    ]
    
    total_contribution = 0.0
    total_progress = 0.0
    
    print("Step | Prev_V | Cur_V | Progress | Contribution | Cumulative")
    print("-" * 65)
    
    for prev_v, cur_v, step in episode_data:
        progress = cur_v - prev_v
        contribution = contrib_value_delta(prev_v, cur_v, step=step, task_complexity=2.0)
        total_progress += progress
        total_contribution += contribution
        
        print(f"{step:4d} | {prev_v:6.2f} | {cur_v:5.2f} | {progress:8.2f} | {contribution:11.3f} | {total_contribution:10.3f}")
    
    print("-" * 65)
    print(f"Total progress: {total_progress:.2f}")
    print(f"Total contribution reward: {total_contribution:.3f}")
    print(f"Average contribution per step: {total_contribution/len(episode_data):.3f}")


def main():
    """Run all C-1 task progress tests."""
    print("C-1 Task Progress Measurement Tests")
    print("=" * 60)
    print("This demonstrates how C-1 measures task progress through")
    print("state-value differences and reduces distance to task completion.")
    print("=" * 60)
    
    test_basic_task_progress()
    test_task_complexity_scaling()
    test_step_based_scaling()
    test_advanced_task_progress()
    test_no_progress_scenarios()
    test_edge_cases()
    simulate_task_episode()
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("- C-1 rewards positive progress toward task completion")
    print("- Early progress is more valuable (step-based scaling)")
    print("- Task complexity affects reward normalization")
    print("- No reward for regression or stagnation")
    print("- Advanced version supports task-specific progress modeling")


if __name__ == "__main__":
    main() 