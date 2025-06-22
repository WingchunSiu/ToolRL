# Contribution Reward System Implementation

This document describes the implementation of real C-0 and C-1 contribution signals in the ToolRL framework, as outlined in the step-by-step patch plan.

## Overview

The contribution reward system adds an additional reward component to the existing reward computation:

```
R_total = R_format + R_correctness + R_length + β · R_contrib
```

where `R_contrib` is either C-0 (binary blackboard changes) or C-1 (value-delta from critic).

## Implementation Details

### 1. Contribution Functions (`verl/utils/contribution.py`)

#### C-0: Binary Contribution
```python
def contrib_binary(prev_bb, cur_bb):
    """C-0: Binary contribution reward based on blackboard changes."""
    return int(_h(prev_bb) != _h(cur_bb))
```

- **Input**: Previous and current blackboard states (JSON strings)
- **Output**: 1 if blackboard changed, 0 otherwise
- **Use case**: Rewards any change to the shared blackboard state

#### C-1: Value-Delta Contribution
```python
def contrib_value_delta(prev_v, cur_v):
    """C-1: Value-delta contribution reward based on critic value changes."""
    return max(cur_v - prev_v, 0.0)
```

- **Input**: Previous and current critic value estimates
- **Output**: Positive change in value estimate, 0 if no improvement
- **Use case**: Rewards improvements in the critic's value estimate

### 2. Reward Computation Integration (`verl/utils/reward_score/rlla.py`)

The `compute_score()` function has been modified to accept step dictionaries:

```python
def compute_score(solution_str, ground_truth, step=0, **kwargs):
    # ... existing reward computation ...
    
    # Contribution reward
    if os.getenv("CONTRIBUTION", "0") == "1":
        contrib_type = os.getenv("CONTRIB_TYPE", "C0").upper()
        beta = float(os.getenv("BETA", "0.05"))
        
        prev_step_dict = kwargs.get("prev_step_dict", {})
        cur_step_dict = kwargs.get("cur_step_dict", {})
        
        if contrib_type == "C0":
            prev_bb = prev_step_dict.get("bb_hash", "{}")
            cur_bb = cur_step_dict.get("bb_hash", "{}")
            R_contrib = contrib_binary(prev_bb, cur_bb)
        elif contrib_type == "C1":
            prev_v = prev_step_dict.get("value_est", 0.0)
            cur_v = cur_step_dict.get("value_est", 0.0)
            R_contrib = contrib_value_delta(prev_v, cur_v)
        
        score += beta * R_contrib
```

### 3. Reward Manager Integration (`verl/trainer/main_ppo.py`)

The `RewardManager.__call__()` method now passes step dictionaries to the reward computation:

```python
# Prepare kwargs for contribution calculation
kwargs = {}
if os.getenv("CONTRIBUTION", "0") == "1":
    prev_step_dict = data_item.non_tensor_batch.get('prev_step_dict', {})
    cur_step_dict = data_item.non_tensor_batch.get('cur_step_dict', {})
    kwargs['prev_step_dict'] = prev_step_dict
    kwargs['cur_step_dict'] = cur_step_dict

score, format_score, correctness_score, length_score = compute_score_fn(
    solution_str=sequences_str, 
    ground_truth=ground_truth, 
    step=step,
    **kwargs
)
```

## Usage

### Environment Variables

The contribution system is controlled by environment variables:

```bash
# Enable contribution rewards
export CONTRIBUTION=1

# Choose contribution type
export CONTRIB_TYPE=C0    # Binary blackboard changes
# OR
export CONTRIB_TYPE=C1    # Value-delta from critic

# Set contribution weight
export BETA=0.05         # Weight for contribution reward
```

### Training Scripts

The contribution system is already integrated into the GRPO training script:

```bash
# examples/grpo_trainer/run_grpo.sh
export CONTRIBUTION=1      # 0 to disable
export CONTRIB_TYPE=C0     # C0 / C1 / C2
export BETA=0.05

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # ... other parameters ...
```

### Testing

Run the integration test to verify the system works:

```bash
python test_contrib_integration.py
```

## Required Data Flow

For the contribution system to work, the following data must be available in the step dictionaries:

### For C-0 (Binary Contribution)
- `prev_step_dict["bb_hash"]`: Hash of previous blackboard state
- `cur_step_dict["bb_hash"]`: Hash of current blackboard state

### For C-1 (Value-Delta Contribution)
- `prev_step_dict["value_est"]`: Previous critic value estimate
- `cur_step_dict["value_est"]`: Current critic value estimate

## Implementation Notes

1. **Backward Compatibility**: The system is designed to be backward compatible. If `CONTRIBUTION=0` or step dictionaries are not provided, the system falls back to the original reward computation.

2. **Default Values**: If step dictionaries are missing required fields, sensible defaults are used:
   - `bb_hash` defaults to `"{}"` (empty blackboard)
   - `value_est` defaults to `0.0`

3. **Error Handling**: The system gracefully handles missing data and continues training even if contribution data is unavailable.

4. **Logging**: When contribution is enabled, the system logs:
   - Contribution type being used
   - Contribution reward value
   - Beta weight
   - Total contribution to final reward

## Future Extensions

The system is designed to be easily extensible:

1. **Additional Contribution Types**: New contribution functions can be added to `verl/utils/contribution.py`
2. **Dynamic Beta**: The beta weight could be made step-dependent or adaptive
3. **Multiple Contributions**: The system could support combining multiple contribution types
4. **Custom Metrics**: Additional metrics could be added to the step dictionaries for more sophisticated contribution calculations

## Troubleshooting

### Common Issues

1. **No contribution reward**: Check that `CONTRIBUTION=1` and step dictionaries contain required data
2. **Zero contribution**: Verify that the blackboard is actually changing (C-0) or critic values are improving (C-1)
3. **Import errors**: Ensure all required modules are available in the Python path

### Debugging

Enable verbose logging by setting environment variables:

```bash
export CONTRIBUTION=1
export CONTRIB_TYPE=C0
export BETA=0.05
# Add debug prints in the reward computation
```

The system will print contribution reward details during training when enabled. 