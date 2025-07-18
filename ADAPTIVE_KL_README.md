# Adaptive KL Divergence for GRPO Training

This repository has been extended with adaptive KL divergence functionality to address pathological length exploitation in GRPO (Generalized Reward Policy Optimization) training.

## Problem Statement

In GRPO training, the loss function typically combines reward and KL divergence terms:

```
L = -E[r(x)] + β * KL(π_θ || π_ref)
```

When the reward term saturates (approaches maximum reward), models often exploit the KL term by generating longer completions to reduce KL divergence. This happens because longer sequences spread probability mass more diffusely, leading to lower per-token surprise and reduced KL divergence.

## Solution: Adaptive KL Methods

This implementation provides several methods to prevent pathological KL exploitation:

### 1. **KL Clipping** (`kl_adaptation_method="clip"`)
Hard clips KL divergence at a specified threshold:
```python
kl_term = min(KL(p || p_ref), kl_clip_threshold)
```

**Parameters:**
- `kl_clip_threshold`: Maximum allowed KL value (default: 1.0)

**Use case:** When you want to prevent extreme KL values while maintaining gradient flow up to the threshold.

### 2. **Sigmoid KL Clipping** (`kl_adaptation_method="sigmoid"`)
Applies soft clipping using a sigmoid function:
```python
kl_term = sigmoid(KL(p || p_ref) - kl_target) * KL(p || p_ref)
```

**Parameters:**
- `kl_target`: Target KL divergence value (default: 0.1)

**Use case:** When you want smooth, differentiable clipping that gradually reduces KL contribution.

### 3. **Length-Normalized KL** (`kl_adaptation_method="length_normalized"`)
Normalizes KL divergence by sequence length:
```python
kl_per_token = kl_divergence / completion_length
```

**Use case:** When you want to prevent longer sequences from artificially reducing KL divergence.

### 4. **Dynamic Beta Scheduling** (`kl_adaptation_method="dynamic"`)
Dynamically adjusts the β coefficient based on reward convergence:
- Reduces β when average reward exceeds threshold
- Sets β to minimum after specified training steps

**Parameters:**
- `dynamic_beta_decay`: Decay factor for β (default: 0.9)
- `reward_threshold`: Reward level triggering β reduction (default: 0.8)
- `beta_min`: Minimum β value (default: 0.0)
- `beta_schedule_steps`: Steps after which β → beta_min (default: 1000)

**Use case:** When you want to automatically reduce KL pressure as training progresses.

## Usage

### Basic Usage

```python
from trl import GRPOConfig, GRPOTrainer

# Example: KL Clipping
training_args = GRPOConfig(
    # ... other arguments ...
    kl_adaptation_method="clip",
    kl_clip_threshold=0.5,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_function],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### All Available Parameters

```python
training_args = GRPOConfig(
    # Standard GRPO parameters
    temperature=1.0,
    learning_rate=5e-6,
    beta=0.001,  # Base KL coefficient
    
    # Adaptive KL parameters
    kl_adaptation_method="dynamic",      # Method: none, clip, sigmoid, dynamic, length_normalized
    kl_clip_threshold=1.0,               # For clipping methods
    kl_target=0.1,                       # For sigmoid clipping
    dynamic_beta_decay=0.9,              # For dynamic scheduling
    reward_threshold=0.8,                # For dynamic scheduling
    beta_min=0.0,                        # For dynamic scheduling
    beta_schedule_steps=1000,            # For dynamic scheduling
)
```

## Metrics and Monitoring

When using adaptive KL methods, additional metrics are logged:

- `adaptive_kl/raw_kl`: Original KL divergence value
- `adaptive_kl/adapted_kl`: KL divergence after adaptation
- `adaptive_kl/current_beta`: Current β coefficient value

These metrics help you monitor the effectiveness of adaptive KL methods during training.

## Examples

### Example 1: Preventing Length Exploitation

```python
# Use length-normalized KL to prevent length exploitation
training_args = GRPOConfig(
    kl_adaptation_method="length_normalized",
    # ... other args
)
```

### Example 2: Progressive KL Reduction

```python
# Dynamically reduce KL pressure as training progresses
training_args = GRPOConfig(
    kl_adaptation_method="dynamic",
    reward_threshold=0.7,     # Start reducing β when reward > 0.7
    dynamic_beta_decay=0.9,   # Reduce β by 10% each time
    beta_min=0.001,          # Don't go below 0.001
    beta_schedule_steps=500,  # After 500 steps, set β = beta_min
)
```

### Example 3: Soft KL Limiting

```python
# Use sigmoid to smoothly limit KL contribution
training_args = GRPOConfig(
    kl_adaptation_method="sigmoid",
    kl_target=0.2,  # Target KL around 0.2
)
```

## Complete Demo Script

See `blackwell/test_adaptive_kl_grpo.py` for a complete demonstration of all adaptive KL methods. The script shows:

1. Baseline training (no adaptation)
2. KL clipping demonstration
3. Length normalization example
4. Dynamic beta scheduling
5. Sigmoid clipping example

## Implementation Details

### Architecture

The adaptive KL functionality is implemented through:

1. **Configuration Parameters**: Added to `GRPOConfig` class
2. **Utility Functions**: Core adaptation logic in `rl_replacements.py`
3. **Loss Integration**: Modified `compute_loss` function applies adaptations
4. **Metrics Logging**: Tracks adaptation effectiveness

### Key Files Modified

- `unsloth/models/rl.py`: Added configuration parameters to GRPOConfig
- `unsloth/models/rl_replacements.py`: Core adaptive KL implementation
- `blackwell/test_adaptive_kl_grpo.py`: Complete demonstration script

### Backward Compatibility

All new parameters have sensible defaults that maintain original behavior when not specified. Existing code will continue to work without modification.

## Best Practices

1. **Start with Length Normalization**: Often the most effective single method
2. **Monitor Metrics**: Use the logged metrics to understand adaptation behavior
3. **Tune Gradually**: Start with conservative parameters and adjust based on results
4. **Combine Methods**: Some methods can be combined (e.g., dynamic + length normalization)

## Troubleshooting

### Common Issues

1. **KL becomes too small**: Increase `kl_clip_threshold` or `kl_target`
2. **No adaptation effect**: Check that `kl_adaptation_method` is not "none"
3. **Training instability**: Reduce adaptation strength or use softer methods

### Debug Tips

- Monitor `adaptive_kl/raw_kl` vs `adaptive_kl/adapted_kl` to see adaptation effect
- Check `adaptive_kl/current_beta` for dynamic scheduling behavior
- Compare completion lengths between methods to verify length exploitation prevention

## Performance Considerations

The adaptive KL methods add minimal computational overhead:
- **Clipping**: Simple tensor operations
- **Sigmoid**: One additional sigmoid computation
- **Length Normalization**: Division by sequence length
- **Dynamic**: Lightweight reward tracking

## Future Enhancements

Potential future improvements:
- Additional adaptation methods (e.g., exponential decay)
- Automatic hyperparameter tuning
- Integration with other RL training methods
- Advanced reward-based adaptation strategies

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [PPO KL Regularization](https://arxiv.org/abs/1707.06347)
- [Length Bias in RL](https://arxiv.org/abs/2209.00588)

---

This implementation provides a robust foundation for preventing pathological length exploitation in GRPO training while maintaining the flexibility to adapt to different use cases and requirements.
