"""
Probabilistic pruning utilities for OctreeGS.
Implements statistical pruning based on multiple mask samples.
"""

import torch


class ProbabilisticPruner:
    """
    Probabilistic pruner that accumulates mask samples and removes
    gaussians that are never selected across k trials.
    """
    
    def __init__(self, num_gaussians: int, k_trials: int = 10):
        """
        Initialize probabilistic pruner.
        
        Args:
            num_gaussians: number of gaussians to track
            k_trials: number of sampling trials before pruning decision
        """
        self.k = k_trials
        self.hits = torch.zeros(num_gaussians, dtype=torch.int32, device="cuda")
        self.trial_count = 0
    
    @torch.no_grad()
    def accumulate(self, M: torch.Tensor):
        """
        Accumulate mask sample.
        
        Args:
            M: [N, 1] binary mask where 1 means the gaussian was selected
        """
        # Resize hits buffer if needed (for newly added gaussians)
        current_size = M.shape[0]
        if current_size > self.hits.shape[0]:
            # Extend hits buffer with zeros for new gaussians
            new_hits = torch.zeros(current_size - self.hits.shape[0], 
                                   dtype=torch.int32, device="cuda")
            self.hits = torch.cat([self.hits, new_hits], dim=0)
        elif current_size < self.hits.shape[0]:
            # This shouldn't happen in normal operation
            # but handle it by truncating
            self.hits = self.hits[:current_size]
        
        self.hits += M.view(-1).to(torch.int32)
        self.trial_count += 1
    
    @torch.no_grad()
    def mark_and_reset(self, protect_mask: torch.Tensor = None):
        """
        Identify gaussians to delete and reset counters.
        
        Args:
            protect_mask: [N] binary tensor where 1 means protect from pruning
        
        Returns:
            idx: indices of gaussians to delete
        """
        # Only perform pruning if we've completed enough trials
        if self.trial_count < self.k:
            return torch.tensor([], dtype=torch.long, device="cuda")
        
        # Mark gaussians that were never hit
        to_delete = (self.hits == 0)
        
        # Apply protection mask if provided
        if protect_mask is not None:
            protect_mask = protect_mask.view(-1)
            # Resize protection mask if needed
            if protect_mask.shape[0] < to_delete.shape[0]:
                pad = torch.zeros(to_delete.shape[0] - protect_mask.shape[0], 
                                 dtype=protect_mask.dtype, device="cuda")
                protect_mask = torch.cat([protect_mask, pad], dim=0)
            elif protect_mask.shape[0] > to_delete.shape[0]:
                protect_mask = protect_mask[:to_delete.shape[0]]
            
            # Don't delete protected gaussians
            to_delete = to_delete & (protect_mask == 0)
        
        idx = torch.nonzero(to_delete, as_tuple=False).view(-1)
        
        # Reset counters
        self.hits.zero_()
        self.trial_count = 0
        
        return idx
    
    def resize(self, new_size: int):
        """Resize the hits buffer (called when gaussians are added/removed)."""
        if new_size > self.hits.shape[0]:
            # Add new entries
            new_hits = torch.zeros(new_size - self.hits.shape[0], 
                                   dtype=torch.int32, device="cuda")
            self.hits = torch.cat([self.hits, new_hits], dim=0)
        elif new_size < self.hits.shape[0]:
            # Truncate
            self.hits = self.hits[:new_size]
