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

import hashlib
import json
import numpy as np
import re


def _h(bb_str):
    """Deterministic hash of blackboard string."""
    return hashlib.md5(bb_str.encode()).hexdigest()


def _extract_think_content(response):
    """Extract the think content from a response."""
    if "<think>" in response and "</think>" in response:
        think_start = response.find("<think>") + len("<think>")
        think_end = response.find("</think>")
        return response[think_start:think_end].strip()
    return ""


def _extract_tool_calls(response):
    """Extract tool calls from a response."""
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_start = response.find("<tool_call>") + len("<tool_call>")
        tool_call_end = response.find("</tool_call>")
        return response[tool_call_start:tool_call_end].strip()
    return ""


# -------- C-0  binary progress ----------
def contrib_binary(prev_bb, cur_bb):
    """C-0: Binary contribution reward based on blackboard changes.
    
    Args:
        prev_bb: Previous blackboard state (JSON string)
        cur_bb: Current blackboard state (JSON string)
    
    Returns:
        1 if blackboard changed, 0 otherwise
    """
    return int(_h(prev_bb) != _h(cur_bb))


# -------- C-1  value-delta --------------
def contrib_value_delta(prev_v, cur_v, step=0, task_complexity=1.0, **kwargs):
    """C-1: Value-delta contribution reward measuring task progress.
    
    This function measures the degree to which each action reduces the distance 
    to solving the overall task through state-value differences.
    
    Args:
        prev_v: Previous critic value estimate (distance to task completion)
        cur_v: Current critic value estimate (distance to task completion)
        step: Current step in the episode
        task_complexity: Complexity factor for the task (higher = more complex)
        **kwargs: Additional context (e.g., task_type, progress_indicators)
    
    Returns:
        Positive reward proportional to progress toward task completion
    """
    # Ensure values are numeric
    try:
        prev_v = float(prev_v) if prev_v is not None else 0.0
        cur_v = float(cur_v) if cur_v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0
    
    # Calculate the improvement in value (reduction in distance to completion)
    value_improvement = cur_v - prev_v
    
    # Only reward positive progress (reduction in distance to task completion)
    if value_improvement <= 0:
        return 0.0
    
    # Normalize by task complexity to account for different task difficulties
    normalized_improvement = value_improvement / max(task_complexity, 0.1)
    
    # Apply step-based scaling: early progress is more valuable
    # This encourages exploration and early task progress
    step_factor = max(0.1, 1.0 - step / 100.0)  # Decay over 100 steps
    
    # Calculate final contribution reward
    contribution = normalized_improvement * step_factor
    
    # Clip to reasonable bounds
    contribution = np.clip(contribution, 0.0, 1.0)
    
    return float(contribution)


def contrib_value_delta_advanced(prev_v, cur_v, step=0, task_info=None, **kwargs):
    """Advanced C-1: Enhanced value-delta with task-specific progress modeling.
    
    This version includes more sophisticated logic for measuring task progress
    based on the specific characteristics of the task being solved.
    
    Args:
        prev_v: Previous critic value estimate
        cur_v: Current critic value estimate  
        step: Current step in the episode
        task_info: Dictionary containing task-specific information
        **kwargs: Additional context
    
    Returns:
        Enhanced contribution reward based on task progress
    """
    task_info = task_info or {}
    
    # Basic value improvement
    value_improvement = cur_v - prev_v
    
    if value_improvement <= 0:
        return 0.0
    
    # Task-specific progress factors
    task_type = task_info.get('task_type', 'general')
    expected_steps = task_info.get('expected_steps', 50)
    current_progress = task_info.get('current_progress', 0.0)
    
    # Adjust reward based on task type
    if task_type == 'multi_step':
        # Multi-step tasks: reward consistent progress
        progress_factor = min(1.0, step / expected_steps)
        contribution = value_improvement * (1.0 + progress_factor)
    elif task_type == 'exploration':
        # Exploration tasks: reward early discoveries
        early_bonus = max(0.0, 1.0 - step / 20.0)  # Bonus for early steps
        contribution = value_improvement * (1.0 + early_bonus)
    else:
        # General tasks: standard progress reward
        contribution = value_improvement
    
    # Apply diminishing returns for very large improvements
    contribution = contribution / (1.0 + abs(contribution))  # Simple sigmoid-like function
    
    return float(contribution)


# -------- C-2  info-gain ----------------
def contrib_info_gain(response, step, **kwargs):
    """Information gain contribution reward: based on response complexity and novelty."""
    # This is kept for backward compatibility but not used in the new implementation
    return 0.0 