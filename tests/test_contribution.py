# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.utils.contribution import contrib_binary, contrib_value_delta, contrib_info_gain
import numpy as np


def test_binary():
    """Test binary contribution reward function."""
    # Test with meaningful content
    response1 = "<think>Let me solve this step by step. First, I need to understand the problem.</think>"
    assert contrib_binary(response=response1, step=0) == 1
    
    response2 = "<think>Short</think>"
    assert contrib_binary(response=response2, step=0) == 0  # Too short
    
    response3 = "<tool_call>\n{\"name\": \"calculator\", \"parameters\": {\"expression\": \"2+2\"}}\n</tool_call>"
    assert contrib_binary(response=response3, step=0) == 1  # Has tool calls
    
    # Test with no meaningful content
    response4 = "<think></think>"
    assert contrib_binary(response=response4, step=0) == 0
    
    response5 = "Just some text without structure"
    assert contrib_binary(response=response5, step=0) == 0


def test_value_delta():
    """Test value-delta contribution reward function."""
    # Test with good content
    response1 = "<think>This is a detailed analysis with many words to test the length scoring mechanism.</think>"
    result1 = contrib_value_delta(response=response1, step=0, current_score=1.0)
    assert result1 > 0.0
    assert result1 <= 1.0
    
    # Test with tool calls
    response2 = "<think>Let me calculate this.</think><tool_call>\n{\"name\": \"calculator\"}\n</tool_call>"
    result2 = contrib_value_delta(response=response2, step=0, current_score=1.0)
    assert result2 > 0.0
    
    # Test step decay
    result3 = contrib_value_delta(response=response1, step=0, current_score=1.0)
    result4 = contrib_value_delta(response=response1, step=25, current_score=1.0)
    assert result3 > result4  # Earlier steps should have higher scores


def test_info_gain():
    """Test information gain contribution reward function."""
    # Test with diverse vocabulary
    response1 = "<think>This response contains many different unique words to test information density calculation.</think>"
    result1 = contrib_info_gain(response=response1, step=0)
    assert isinstance(result1, float)
    assert result1 >= 0.0
    assert result1 <= 1.0
    
    # Test with tool complexity
    response2 = "<think>Let me use tools.</think><tool_call>\n{\"name\": \"tool1\"}\n{\"name\": \"tool2\"}\n{\"name\": \"tool3\"}\n</tool_call>"
    result2 = contrib_info_gain(response=response2, step=0)
    assert result2 > 0.0
    
    # Test with repetitive content
    response3 = "<think>word word word word word word word word word word</think>"
    result3 = contrib_info_gain(response=response3, step=0)
    assert result3 < 0.5  # Low information density due to repetition


if __name__ == "__main__":
    print("Running contribution reward tests...")
    test_binary()
    test_value_delta()
    test_info_gain()
    print("All tests passed!") 