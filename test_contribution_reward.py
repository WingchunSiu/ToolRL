#!/usr/bin/env python3
"""
Test script to validate the contribution-enhanced reward function.
"""

import os
import sys

# Set environment variables for testing
os.environ["ENABLE_CONTRIBUTION"] = "1"
os.environ["CONTRIBUTION_BETA"] = "0.5"
os.environ["EXPERIMENT_NAME"] = "grpo-test-qwen2.5-1.5b"

# Add project path
sys.path.append('/u/siu1/ToolRL')

from verl.utils.reward_score.rlla import compute_score

def test_contribution_reward():
    print("Testing Contribution-Enhanced Reward Function")
    print("=" * 60)
    
    # Test Case 1: Complete correct response
    print("\n1. COMPLETE CORRECT RESPONSE:")
    solution_1 = """<|im_start|>assistant
<think>
The user wants information about refugees in Germany. I should use the getRefugeeInfo tool with country parameter set to Germany.
</think>
<tool_call>
{"name": "getRefugeeInfo", "parameters": {"country": "Germany"}}
</tool_call>
<response>I'll help you get the latest statistics on refugees in Germany for your report.</response>
<|im_end|>"""
    
    ground_truth_1 = """<think>
The user wants information about refugees in Germany. I should use the getRefugeeInfo tool with country parameter set to Germany.
</think>
<tool_call>
{"name": "getRefugeeInfo", "parameters": {"country": "Germany"}}
</tool_call>
<response>I'll help you get the latest statistics on refugees in Germany for your report.</response>"""
    
    score_1, format_1, correct_1, length_1 = compute_score(solution_1, ground_truth_1, step=50)
    
    # Test Case 2: Partial response (thinking only)
    print("\n" + "="*60)
    print("2. PARTIAL RESPONSE (THINKING ONLY):")
    solution_2 = """<|im_start|>assistant
<think>
The user wants information about refugees in Germany. I need to analyze what tool to use. The getRefugeeInfo tool seems appropriate for this request.
</think>
<|im_end|>"""
    
    score_2, format_2, correct_2, length_2 = compute_score(solution_2, ground_truth_1, step=50)
    
    # Test Case 3: Wrong tool but correct structure
    print("\n" + "="*60)
    print("3. WRONG TOOL BUT CORRECT STRUCTURE:")
    solution_3 = """<|im_start|>assistant
<think>
The user wants refugee information. Let me use a tool.
</think>
<tool_call>
{"name": "getWeatherInfo", "parameters": {"country": "Germany"}}
</tool_call>
<response>Here's the information you requested.</response>
<|im_end|>"""
    
    score_3, format_3, correct_3, length_3 = compute_score(solution_3, ground_truth_1, step=50)
    
    # Test Case 4: No structure (baseline)
    print("\n" + "="*60)
    print("4. NO STRUCTURE (BASELINE):")
    solution_4 = """<|im_start|>assistant
I can help you get information about refugees in Germany. Let me check the latest statistics for your report.
<|im_end|>"""
    
    score_4, format_4, correct_4, length_4 = compute_score(solution_4, ground_truth_1, step=50)
    
    # Test Case 5: Early training step (structure focus)
    print("\n" + "="*60)
    print("5. EARLY TRAINING STEP (STRUCTURE FOCUS):")
    solution_5 = """<|im_start|>assistant
<think>
Basic thinking about the task.
</think>
<tool_call>
{"name": "wrongTool", "parameters": {}}
</tool_call>
<|im_end|>"""
    
    score_5, format_5, correct_5, length_5 = compute_score(solution_5, ground_truth_1, step=5)  # Early step
    
    # Test Case 6: Late training step (correctness focus)
    print("\n" + "="*60)
    print("6. LATE TRAINING STEP (CORRECTNESS FOCUS):")
    
    score_6, format_6, correct_6, length_6 = compute_score(solution_5, ground_truth_1, step=150)  # Late step
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF CONTRIBUTION REWARD RESULTS")
    print("="*60)
    
    test_cases = [
        ("Complete Correct", score_1, format_1, correct_1),
        ("Partial (Think Only)", score_2, format_2, correct_2),
        ("Wrong Tool", score_3, format_3, correct_3),
        ("No Structure", score_4, format_4, correct_4),
        ("Early Step", score_5, format_5, correct_5),
        ("Late Step", score_6, format_6, correct_6)
    ]
    
    print(f"{'Test Case':<20} {'Total':<8} {'Format':<8} {'Correct':<8}")
    print("-" * 50)
    for name, total, fmt, corr in test_cases:
        print(f"{name:<20} {total:<8.3f} {fmt:<8.3f} {corr:<8.3f}")
    
    # Verify contribution reward is working
    contribution_enabled = score_1 > (format_1 + correct_1 + length_1)
    print(f"\nContribution reward enabled: {contribution_enabled}")
    
    if contribution_enabled:
        print("✅ Contribution reward is working!")
        print("Expected behavior:")
        print("- Complete responses should get highest scores")
        print("- Partial structure should get some reward")
        print("- Early steps should focus more on structure")
        print("- Late steps should focus more on correctness")
    else:
        print("❌ Contribution reward not working properly")
    
    return test_cases

if __name__ == "__main__":
    test_contribution_reward() 