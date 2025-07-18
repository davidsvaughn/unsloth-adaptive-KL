# KL Clipping for GRPO Training

This repository implements simple KL clipping functionality to prevent pathological length exploitation in GRPO (Generalized Reward Policy Optimization) training.

## Problem Statement

In GRPO training, the loss function combines reward and KL divergence terms:

```
L = -E[r(x)] + β * KL(π_θ || π_ref)
```

When the reward term saturates, models often exploit the KL term by generating longer completions to reduce KL divergence. This happens because longer sequences spread probability mass more diffusely, leading to lower per-token surprise.

## Solution: Simple KL Clipping

This implementation provides two simple methods to prevent pathological KL exploitation:

### 1. **Hard Clipping** (`kl_clip_method="hard"`)
Uses a ReLU-like function to clip KL divergence:
```python
kl_term = max(0, KL(p || p_ref) - kl_clip_threshold)
```

**Use case:** When you want to completely eliminate KL values below the threshold while preserving gradient flow above it.

### 2. **Soft Clipping** (`kl_clip_method="soft"`)
Uses a softplus function for smooth clipping:
```python
import torch.nn.functional as F
kl_term = F.softplus(KL(p || p_ref) - kl_clip_threshold)
```

**Use case:** When you want smooth, differentiable clipping that gradually reduces KL contribution below the threshold.

## Usage

### Basic Usage

```python
from trl import GRPOConfig, GRPOTrainer

# Hard clipping example
training_args = GRPOConfig(
    # ... other arguments ...
    kl_clip_method="hard",
    kl_clip_threshold=0.5,
)

# Soft clipping example  
training_args = GRPOConfig(
    # ... other arguments ...
    kl_clip_method="soft",
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

### Available Parameters

```python
training_args = GRPOConfig(
    # Standard GRPO parameters
    temperature=1.0,
    learning_rate=5e-6,
    beta=0.001,  # Base KL coefficient
    
    # KL clipping parameters
    kl_clip_method="hard",           # Options: "none", "hard", "soft"
    kl_clip_threshold=0.5,           # Threshold for clipping
    use_per_token_kl_threshold=True, # If True, threshold is per-token (default: True)
)
```

### Per-Token vs Total KL Thresholds

**Per-Token Threshold (Recommended)**: When `use_per_token_kl_threshold=True` (default), the `kl_clip_threshold` is interpreted as a per-token KL divergence threshold. This is scaled by the reference sequence length to compute the actual total KL threshold used for clipping.

**Total KL Threshold**: When `use_per_token_kl_threshold=False`, the `kl_clip_threshold` is used directly as the total KL threshold without any scaling.

#### Benefits of Per-Token Thresholds

1. **Intuitive threshold setting**: "0.1 nats per token" is much more interpretable than total KL values
2. **Length-agnostic**: Same threshold works whether generating 10 tokens or 100 tokens
3. **Transferable**: Thresholds learned on one task/length distribution transfer to others
4. **Prevents gaming**: Model can't manipulate threshold by generating longer sequences
5. **Stable**: Threshold is based on reference sequence length, not generated length

#### How It Works

```python
# For per-token thresholds
total_kl_threshold = per_token_kl_threshold * reference_sequence_length

# Example: 0.1 nats per token × 50 tokens = 5.0 total nats threshold
```

**Example:**
```python
# Per-token threshold (recommended)
training_args = GRPOConfig(
    kl_clip_method="soft",
    kl_clip_threshold=0.1,           # 0.1 nats per token
    use_per_token_kl_threshold=True, # Scale by reference length
)

# Total KL threshold (legacy behavior)
training_args = GRPOConfig(
    kl_clip_method="soft",
    kl_clip_threshold=5.0,            # 5.0 total nats
    use_per_token_kl_threshold=False, # Use threshold directly
)
```

## Methods Comparison

| Method | Function | Gradient Behavior | Use Case |
|--------|----------|-------------------|----------|
| **None** | `KL(p \|\| p_ref)` | Standard | Baseline GRPO |
| **Hard** | `max(0, KL - threshold)` | Zero below threshold | Sharp cutoff |
| **Soft** | `F.softplus(KL - threshold)` | Smooth everywhere | Gradual reduction |

## Complete Example

See `blackwell/test_simple_kl_grpo.py` for a complete demonstration showing:

1. Baseline training (no clipping)
2. Hard clipping demonstration
3. Soft clipping demonstration

The script compares all three methods on the same mathematical reasoning task.

## Implementation Details

The KL clipping functionality is implemented in:
- `unsloth/models/rl_replacements.py`: Core clipping functions
- Configuration validation ensures only valid methods are used
- Metrics logging tracks the effect of clipping during training

## Best Practices

1. **Start with `kl_clip_threshold=0.5`**: A reasonable default for most tasks
2. **Monitor KL values**: Watch training logs to see if clipping is effective
3. **Choose method based on needs**:
   - Use **hard** clipping for sharp cutoffs
   - Use **soft** clipping for smooth transitions
4. **Compare with baseline**: Always test against standard GRPO to measure improvement

## Mathematical Details

### Hard Clipping
- **Function**: `max(0, KL - threshold)`
- **Gradient**: 0 when `KL < threshold`, 1 when `KL > threshold`
- **Effect**: Completely removes KL penalty below threshold

### Soft Clipping  
- **Function**: `F.softplus(KL - threshold) = log(1 + exp(KL - threshold))`
- **Gradient**: `sigmoid(KL - threshold)`
- **Effect**: Smoothly reduces KL penalty below threshold

## Performance Considerations

Both clipping methods add minimal computational overhead:
- **Hard clipping**: Simple max operation
- **Soft clipping**: One softplus computation per training step

## Troubleshooting

### Common Issues

1. **KL values don't change**: Check that `kl_clip_threshold` is appropriate for your KL scale
2. **Training becomes unstable**: Try a higher threshold or switch to soft clipping
3. **No improvement over baseline**: Monitor if your KL values actually exceed the threshold

### Debug Tips

- Check training logs for KL values to set appropriate thresholds
- Compare completion lengths between methods
- Monitor reward progression to ensure clipping helps

---

This simplified implementation provides effective KL clipping to prevent pathological length exploitation while maintaining code simplicity and ease of use.
