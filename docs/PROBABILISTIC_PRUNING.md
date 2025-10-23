# Probabilistic Pruning in OctreeGS

This document describes the probabilistic pruning feature implemented in OctreeGS, which uses learnable binary masks to dynamically prune Gaussians during training.

## Overview

Probabilistic pruning introduces a differentiable masking mechanism that learns which Gaussians should be kept or dropped during training. This approach:

1. **Adds learnable mask parameters** (`mask_logit_keep` and `mask_logit_drop`) for each Gaussian
2. **Uses Gumbel-Softmax sampling** to generate binary masks with temperature annealing
3. **Applies mask regularization** to encourage sparsity while maintaining quality
4. **Performs statistical pruning** based on multiple mask samples over training iterations

## Key Components

### 1. Masking Utilities (`utils/masking.py`)

#### `gumbel_softmax_binary(logits_keep, logits_drop, tau, hard=True)`
Performs binary Gumbel-Softmax sampling to generate masks:
- **Input**: Keep/drop logits for each Gaussian, temperature τ
- **Output**: Soft probabilities and hard binary masks (0 or 1)
- **Straight-through estimator**: Uses hard masks in forward pass, soft probabilities for gradients

#### `TempScheduler(t_start=1.0, t_end=0.4, total_steps=30000)`
Linear temperature annealing for Gumbel-Softmax:
- **Start**: High temperature (1.0) → more exploration
- **End**: Low temperature (0.4) → sharper decisions
- **Purpose**: Gradually transition from soft to hard masking

### 2. Pruning Utilities (`utils/pruning.py`)

#### `ProbabilisticPruner(num_gaussians, k_trials=10)`
Accumulates mask samples and prunes Gaussians with zero hits:
- **Accumulation**: Tracks how many times each Gaussian is selected across k trials
- **Pruning decision**: Removes Gaussians with 0 hits out of k samples
- **Protection mask**: Can protect newly created Gaussians from premature pruning

### 3. Model Modifications (`scene/gaussian_model.py`)

**New parameters** for each Gaussian:
```python
self._mask_logit_keep  # [N, 1] learnable logit for keeping
self._mask_logit_drop  # [N, 1] learnable logit for dropping
self._protect_mask     # [N] protection status (0 or 1)
```

**New methods**:
- `newborn_protect_mask()`: Returns protection mask for recently created Gaussians
- `remove_gaussians(idx)`: Removes specified Gaussians from all parameters and optimizer state

**Initialization**:
- New Gaussians start with `keep=0.8, drop=0.0` (bias towards keeping)
- Protection mask can be set to prevent pruning for N local cycles

### 4. Training Integration (`train.py`)

**Mask sampling** (each iteration):
```python
tau = temp_sched.value(iteration)
p_keep, M = gumbel_softmax_binary(
    gaussians.get_mask_logit_keep,
    gaussians.get_mask_logit_drop,
    tau=tau, hard=True
)
pruner.accumulate(M)
```

**Mask regularization loss**:
```python
lambda_m = get_lambda_m(iteration, total_steps)
L_mask = (M.mean()) ** 2
loss = L_render + lambda_m * L_mask
```

Lambda schedule (encourages increasing sparsity):
- [0%, 30%]: λ = 5×10⁻⁴ (warmup)
- [30%, 70%]: λ = 8×10⁻⁴ (mid-training)
- [70%, 100%]: λ = 1×10⁻³ (late training)

**Pruning triggers**:
1. **After densification**: Every 10 local cycles (k=10 samples)
2. **Periodic check**: Every 1000 iterations

## Configuration

See `configs/mask_pruning.yaml` for all hyperparameters.

### Default Settings
```yaml
tau_start: 1.0          # Starting temperature
tau_end: 0.4            # Ending temperature
k_trials: 10            # Samples before pruning decision
protect_cycles: 2       # Protection duration for new Gaussians
mask_grad_clip: 1.0     # Gradient clipping for mask logits
```

### Presets

**Quality-first** (more conservative pruning):
- Higher final temperature (0.6)
- Lower late-stage λ (5×10⁻⁴)
- More samples (k=12)
- Longer protection (3 cycles)

**Compress-first** (aggressive pruning):
- Lower final temperature (0.4)
- Higher late-stage λ (1.2×10⁻³)
- Fewer samples (k=8)
- Standard protection (2 cycles)

**Balanced** (default):
- Moderate settings between the two extremes

## Usage

The feature is automatically enabled when running training. No special flags are needed.

### Monitoring

TensorBoard logs (updated every 100 iterations):
- `mask/p_keep_mean`: Average keep probability
- `mask/retained_ratio`: Fraction of Gaussians retained
- `mask/lambda_m`: Current regularization coefficient
- `mask/num_retained`: Number of retained Gaussians
- `mask/num_total`: Total number of Gaussians

### Testing

Run the test suite to verify the implementation:
```bash
python test_masking.py
```

Tests cover:
- Gumbel-Softmax binary sampling
- Temperature scheduling
- Probabilistic pruning with/without protection masks
- Integration of all components

## Implementation Notes

### Design Decisions

1. **Python-level masking**: Currently implemented at the Python level rather than in CUDA kernels. This provides flexibility and ease of implementation. Future optimization could move masking into the rasterization kernel if needed.

2. **Straight-through estimator**: Essential for training binary masks with gradient-based optimization. Allows gradients to flow through discrete decisions.

3. **Statistical pruning**: Using k=10 samples provides robust pruning decisions while limiting overhead. Each sample contributes information about Gaussian importance.

4. **Protection mechanism**: Prevents newly created Gaussians from being immediately pruned before they can be optimized.

5. **Staged regularization**: Increasing λ over training encourages progressive sparsification as the model converges.

### Gradient Flow

The mask parameters receive gradients from:
1. **Reconstruction loss** (indirect): Through straight-through estimator
2. **Mask regularization** (direct): L_mask = (mean(M))²

Gradient clipping (max_norm=1.0) prevents large updates and stabilizes training.

### Memory and Compute

**Additional memory**:
- 2 × N parameters for mask logits (typically small compared to other parameters)
- k × N integers for hit counting (temporary, cleared after pruning)

**Additional compute**:
- Gumbel sampling: ~O(N) per iteration
- Hit accumulation: ~O(N) per iteration
- Pruning: ~O(N) when triggered (infrequent)

Overall overhead is minimal compared to rendering and backward pass.

## Future Enhancements

Possible improvements:
1. **CUDA kernel integration**: Pass masks to rasterizer for true contribution gating
2. **Adaptive k_trials**: Adjust sampling frequency based on training stage
3. **Hierarchical masking**: Apply masks at different LOD levels
4. **Mask visualization**: Render images showing masked/unmasked Gaussians
5. **Fine-grained protection**: Track Gaussian "age" for more precise protection

## References

- Gumbel-Softmax: "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2017)
- Straight-through estimators: "Estimating or Propagating Gradients Through Stochastic Neurons" (Bengio et al., 2013)
