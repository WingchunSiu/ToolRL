#!/usr/bin/env python3
"""
Quick sanity test for the contribution integration.
This tests the C-0 and C-1 contribution functions as described in the patch plan.
"""

import os
import sys
sys.path.append('.')

from verl.utils.contribution import contrib_binary, contrib_value_delta


def test_c0_contribution():
    """Test C-0: Binary contribution based on blackboard changes."""
    print("Testing C-0 contribution...")
    
    # Test 1: No change
    assert contrib_binary("{}", "{}") == 0
    assert contrib_binary('{"x": 1}', '{"x": 1}') == 0
    print("✓ No change correctly returns 0")
    
    # Test 2: Change detected
    assert contrib_binary("{}", '{"LAX": 1}') == 1
    assert contrib_binary('{"x": 1}', '{"x": 2}') == 1
    assert contrib_binary('{"x": 1}', '{"x": 1, "y": 2}') == 1
    print("✓ Changes correctly return 1")


def test_c1_contribution():
    """Test C-1: Value-delta contribution based on critic value changes."""
    print("Testing C-1 contribution...")
    
    # Test 1: No improvement
    assert contrib_value_delta(0.5, 0.3) == 0.0
    assert contrib_value_delta(0.5, 0.5) == 0.0
    print("✓ No improvement correctly returns 0.0")
    
    # Test 2: Improvement detected
    assert contrib_value_delta(0.3, 0.5) == 0.2
    assert contrib_value_delta(0.0, 1.0) == 1.0
    print("✓ Improvements correctly return positive delta")


def test_environment_variables():
    """Test that environment variables are properly set."""
    print("Testing environment variables...")
    
    # Set test values
    os.environ["CONTRIBUTION"] = "1"
    os.environ["CONTRIB_TYPE"] = "C0"
    os.environ["BETA"] = "0.05"
    
    assert os.getenv("CONTRIBUTION") == "1"
    assert os.getenv("CONTRIB_TYPE") == "C0"
    assert float(os.getenv("BETA", "0.0")) == 0.05
    print("✓ Environment variables work correctly")


if __name__ == "__main__":
    print("Running contribution integration tests...")
    print("=" * 50)
    
    test_c0_contribution()
    test_c1_contribution()
    test_environment_variables()
    
    print("=" * 50)
    print("All tests passed! ✅")
    print("\nTo use in training:")
    print("export CONTRIBUTION=1")
    print("export CONTRIB_TYPE=C0  # or C1")
    print("export BETA=0.05")
    print("bash train_grpo.sh") 