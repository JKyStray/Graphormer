import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class RelativePositionBias(nn.Module):
    """
    Implements relative position bias for Transformer attention mechanisms.
    This adds position-aware bias terms to attention scores based on the relative positions
    between tokens, allowing the model to leverage sequence order information.
    
    The implementation uses a bucket-based approach to handle arbitrary sequence lengths
    while keeping the number of parameters manageable.
    """
    def __init__(self, num_buckets=64, max_distance=256, out_dim=2):
        """
        Initialize the relative position bias module.
        
        Args:
            num_buckets (int): Number of buckets for relative positions. Positions are hashed into these buckets.
                            Default: 64
            max_distance (int): Maximum distance to consider for exact position differences.
                            Positions beyond this will be hashed. Default: 256
            out_dim (int): Output dimension of the position bias. Typically matches the attention head dimension.
                         Default: 2
        """
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # Embedding layer that maps bucket indices to bias values
        self.relative_attention_bias = nn.Embedding(self.num_buckets, out_dim)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        """
        Convert relative positions to bucket indices using a mixed exact/logarithmic bucketing scheme.
        
        Args:
            relative_position: Tensor of relative positions [seq_len, seq_len]
            num_buckets: Total number of buckets to use (will be split between positive/negative)
            max_distance: Maximum distance for exact position differences
            
        Returns:
            Tensor of bucket indices with same shape as relative_position
            
        The bucketing strategy:
        1. First half of buckets are for exact positions (small distances)
        2. Second half use log-space bucketing for larger distances
        3. Negative positions are mapped to separate buckets
        """
        # Split buckets between positive and negative positions
        num_buckets //= 2
        # Handle negative positions by offsetting their bucket indices
        ret = (relative_position < 0).to(relative_position) * num_buckets
        # Work with absolute positions for bucketing
        relative_position = torch.abs(relative_position)
        
        # For small distances, use exact position values
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).long()
        )
        
        # Clamp values to valid bucket range
        val_if_large = torch.min(
            val_if_large, 
            torch.full_like(val_if_large, num_buckets - 1)
        )

        # Combine small and large distance bucketing
        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)
        return rp_bias
