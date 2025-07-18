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

### 1. **Hard Clipping** (`kl_adaptation_method="hard"`)
Uses a ReLU-like function to clip KL divergence:
```python
kl_term = max(0, KL(p || p_ref) - kl_clip_threshold)
```

**Use case:** When you want to completely eliminate KL values below the threshold while preserving gradient flow above it.

### 2. **Soft Clipping** (`kl_adaptation_method="soft"`)
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
    kl_adaptation_method="hard",
    kl_clip_threshold=0.5,
)

# Soft clipping example  
training_args = GRPOConfig(
    # ... other arguments ...
    kl_adaptation_method="soft",
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
    kl_adaptation_method="hard",  # Options: "none", "hard", "soft"
    kl_clip_threshold=0.5,       # Threshold for clipping
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
