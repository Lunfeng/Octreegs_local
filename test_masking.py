"""
Simple test script to verify masking and pruning utilities.
This should be run in an environment with torch installed.

Usage:
    python test_masking.py
"""

import torch
from utils.masking import gumbel_softmax_binary, TempScheduler
from utils.pruning import ProbabilisticPruner


def test_gumbel_softmax_binary():
    """Test binary Gumbel-Softmax sampling."""
    print("Testing gumbel_softmax_binary...")
    
    # Create test logits
    N = 100
    logits_keep = torch.randn(N, 1, device="cuda") * 0.5 + 0.8
    logits_drop = torch.randn(N, 1, device="cuda") * 0.5
    
    # Test with different temperatures
    for tau in [1.0, 0.5, 0.1]:
        p_keep, M = gumbel_softmax_binary(logits_keep, logits_drop, tau=tau, hard=True)
        
        assert p_keep.shape == (N, 1), f"p_keep shape mismatch: {p_keep.shape}"
        assert M.shape == (N, 1), f"M shape mismatch: {M.shape}"
        assert torch.all((M == 0) | (M == 1)), "M should be binary"
        assert 0 <= p_keep.min() <= 1 and 0 <= p_keep.max() <= 1, "p_keep should be in [0,1]"
        
        print(f"  tau={tau}: mean_p_keep={p_keep.mean().item():.3f}, retained={M.sum().item()}/{N}")
    
    print("✓ gumbel_softmax_binary tests passed")


def test_temp_scheduler():
    """Test temperature scheduler."""
    print("\nTesting TempScheduler...")
    
    scheduler = TempScheduler(t_start=1.0, t_end=0.4, total_steps=1000)
    
    # Test at different steps
    assert scheduler.value(0) == 1.0, "Initial temperature should be 1.0"
    assert scheduler.value(1000) == 0.4, "Final temperature should be 0.4"
    
    mid_temp = scheduler.value(500)
    assert 0.4 < mid_temp < 1.0, f"Mid temperature should be between 0.4 and 1.0, got {mid_temp}"
    
    # Test monotonicity
    for i in range(0, 1000, 100):
        t1 = scheduler.value(i)
        t2 = scheduler.value(i + 100)
        assert t1 >= t2, f"Temperature should decrease monotonically: {t1} >= {t2}"
    
    print(f"  Initial: {scheduler.value(0):.3f}")
    print(f"  Mid: {scheduler.value(500):.3f}")
    print(f"  Final: {scheduler.value(1000):.3f}")
    print("✓ TempScheduler tests passed")


def test_probabilistic_pruner():
    """Test probabilistic pruner."""
    print("\nTesting ProbabilisticPruner...")
    
    N = 100
    k_trials = 10
    pruner = ProbabilisticPruner(num_gaussians=N, k_trials=k_trials)
    
    # Simulate k_trials mask samples where some gaussians are never selected
    for trial in range(k_trials):
        # Create a mask where first 20 gaussians are never selected (always 0)
        M = torch.ones(N, 1, device="cuda")
        M[:20] = 0
        pruner.accumulate(M)
    
    # Mark and reset should identify the first 20 gaussians for deletion
    del_idx = pruner.mark_and_reset()
    
    assert del_idx.shape[0] == 20, f"Should identify 20 gaussians for deletion, got {del_idx.shape[0]}"
    assert torch.all(del_idx < 20), "Deleted indices should be in [0, 20)"
    
    # After reset, counters should be zero
    assert pruner.trial_count == 0, "Trial count should be reset"
    
    print(f"  Correctly identified {del_idx.shape[0]} gaussians for deletion")
    print("✓ ProbabilisticPruner tests passed")
    
    # Test with protection mask
    print("\nTesting with protection mask...")
    pruner = ProbabilisticPruner(num_gaussians=N, k_trials=k_trials)
    
    for trial in range(k_trials):
        M = torch.ones(N, 1, device="cuda")
        M[:30] = 0  # First 30 never selected
        pruner.accumulate(M)
    
    # Protect first 10 gaussians
    protect_mask = torch.zeros(N, dtype=torch.int32, device="cuda")
    protect_mask[:10] = 1
    
    del_idx = pruner.mark_and_reset(protect_mask=protect_mask)
    
    # Should only delete gaussians 10-29 (30 total - 10 protected = 20)
    assert del_idx.shape[0] == 20, f"Should delete 20 gaussians (30-10 protected), got {del_idx.shape[0]}"
    assert torch.all(del_idx >= 10), "Protected gaussians should not be deleted"
    assert torch.all(del_idx < 30), "Deleted indices should be in [10, 30)"
    
    print(f"  Correctly protected first 10 gaussians")
    print(f"  Deleted {del_idx.shape[0]} unprotected gaussians")
    print("✓ Protection mask tests passed")


def test_integration():
    """Test integration of masking and pruning."""
    print("\nTesting integration...")
    
    N = 50
    k_trials = 5
    
    # Initialize
    logits_keep = torch.randn(N, 1, device="cuda") * 0.5 + 0.8
    logits_drop = torch.randn(N, 1, device="cuda") * 0.5
    
    pruner = ProbabilisticPruner(num_gaussians=N, k_trials=k_trials)
    scheduler = TempScheduler(t_start=1.0, t_end=0.4, total_steps=100)
    
    # Simulate training loop
    for step in range(k_trials):
        tau = scheduler.value(step * 20)
        p_keep, M = gumbel_softmax_binary(logits_keep, logits_drop, tau=tau, hard=True)
        pruner.accumulate(M)
    
    del_idx = pruner.mark_and_reset()
    
    print(f"  Simulated {k_trials} iterations")
    print(f"  Identified {del_idx.shape[0]} gaussians for pruning")
    print(f"  Retention rate: {(N - del_idx.shape[0])/N*100:.1f}%")
    print("✓ Integration tests passed")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)
    
    print("Running masking and pruning tests...\n")
    test_gumbel_softmax_binary()
    test_temp_scheduler()
    test_probabilistic_pruner()
    test_integration()
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
