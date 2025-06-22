#!/usr/bin/env python3
"""
Example script demonstrating the contribution reward system.

This script shows how the contribution rewards work with sample data
and different contribution types.
"""

import os
import sys
sys.path.append('.')

from verl.utils.reward_score.rlla import compute_score


def example_c0_contribution():
    """Example of C-0 (binary) contribution based on blackboard changes."""
    print("=== C-0 Contribution Example ===")
    
    # Set environment for C-0
    os.environ["CONTRIBUTION"] = "1"
    os.environ["CONTRIB_TYPE"] = "C0"
    os.environ["BETA"] = "0.05"
    os.environ["EXPERIMENT_NAME"] = "qwen_test"
    
    # Sample solution and ground truth
    solution_str = "<|im_start|>assistant<think>I need to find information about LAX airport.</think><tool_call>\n{\"name\": \"search\", \"parameters\": {\"query\": \"LAX airport\"}}\n</tool_call><|im_end|>"
    ground_truth = "<think>I need to find information about LAX airport.</think><tool_call>\n{\"name\": \"search\", \"parameters\": {\"query\": \"LAX airport\"}}\n</tool_call>"
    
    # Test with no blackboard change
    kwargs_no_change = {
        "prev_step_dict": {"bb_hash": "{}"},
        "cur_step_dict": {"bb_hash": "{}"}
    }
    
    score1, format1, correct1, length1 = compute_score(
        solution_str, ground_truth, step=0, **kwargs_no_change
    )
    print(f"No blackboard change: score = {score1:.3f}")
    
    # Test with blackboard change
    kwargs_change = {
        "prev_step_dict": {"bb_hash": "{}"},
        "cur_step_dict": {"bb_hash": "{\"LAX\": \"Los Angeles International Airport\"}"}
    }
    
    score2, format2, correct2, length2 = compute_score(
        solution_str, ground_truth, step=0, **kwargs_change
    )
    print(f"With blackboard change: score = {score2:.3f}")
    print(f"Contribution reward: {score2 - score1:.3f}")


def example_c1_contribution():
    """Example of C-1 (value-delta) contribution based on critic value changes."""
    print("\n=== C-1 Contribution Example ===")
    
    # Set environment for C-1
    os.environ["CONTRIBUTION"] = "1"
    os.environ["CONTRIB_TYPE"] = "C1"
    os.environ["BETA"] = "0.05"
    os.environ["EXPERIMENT_NAME"] = "qwen_test"
    
    # Sample solution and ground truth
    solution_str = "<|im_start|>assistant<think>I need to calculate the distance between LAX and JFK airports.</think><tool_call>\n{\"name\": \"calculator\", \"parameters\": {\"expression\": \"distance(LAX, JFK)\"}}\n</tool_call><|im_end|>"
    ground_truth = "<think>I need to calculate the distance between LAX and JFK airports.</think><tool_call>\n{\"name\": \"calculator\", \"parameters\": {\"expression\": \"distance(LAX, JFK)\"}}\n</tool_call>"
    
    # Test with no value improvement
    kwargs_no_improvement = {
        "prev_step_dict": {"value_est": 0.8},
        "cur_step_dict": {"value_est": 0.6}
    }
    
    score1, format1, correct1, length1 = compute_score(
        solution_str, ground_truth, step=0, **kwargs_no_improvement
    )
    print(f"No value improvement: score = {score1:.3f}")
    
    # Test with value improvement
    kwargs_improvement = {
        "prev_step_dict": {"value_est": 0.6},
        "cur_step_dict": {"value_est": 0.8}
    }
    
    score2, format2, correct2, length2 = compute_score(
        solution_str, ground_truth, step=0, **kwargs_improvement
    )
    print(f"With value improvement: score = {score2:.3f}")
    print(f"Contribution reward: {score2 - score1:.3f}")


def example_no_contribution():
    """Example without contribution rewards (baseline)."""
    print("\n=== Baseline (No Contribution) Example ===")
    
    # Disable contribution
    os.environ["CONTRIBUTION"] = "0"
    os.environ["EXPERIMENT_NAME"] = "qwen_test"
    
    # Sample solution and ground truth
    solution_str = "<|im_start|>assistant<think>Let me solve this step by step.</think><tool_call>\n{\"name\": \"search\", \"parameters\": {\"query\": \"weather\"}}\n</tool_call><|im_end|>"
    ground_truth = "<think>Let me solve this step by step.</think><tool_call>\n{\"name\": \"search\", \"parameters\": {\"query\": \"weather\"}}\n</tool_call>"
    
    score, format_score, correct_score, length_score = compute_score(
        solution_str, ground_truth, step=0
    )
    print(f"Baseline score: {score:.3f}")
    print(f"  - Format: {format_score:.3f}")
    print(f"  - Correctness: {correct_score:.3f}")
    print(f"  - Length: {length_score:.3f}")


def main():
    """Run all examples."""
    print("Contribution Reward System Examples")
    print("=" * 50)
    
    example_no_contribution()
    example_c0_contribution()
    example_c1_contribution()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- C-0 rewards any change to the blackboard state")
    print("- C-1 rewards improvements in critic value estimates")
    print("- Both are weighted by the BETA parameter")
    print("- The system is backward compatible when CONTRIBUTION=0")


if __name__ == "__main__":
    main() 