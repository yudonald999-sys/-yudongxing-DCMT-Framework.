"""
Hierarchical Representation Module

Implements multi-level transformer encoders with bidirectional connections
as described in Equation (2):
h^l = TransformerBlock(h^{l-1} + TopDown(h^{l+1}))

Author: Dongxing Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class TopDownProjection(nn.Module):
    """
    Top-down projection for feedback connections from higher levels.
    
    Projects higher-level representations to lower-level dimensions
    and applies learned gating for information flow control.
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        lower: torch.Tensor,
        higher: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply top-down projection with gating.
        
        Args:
            lower: Lower-level representation [B, L, D]
            higher: Higher-level representation [B, L', D]
            
        Returns:
            Gated top-down signal [B, L, D]
        """
        # Project higher-level representation
        if higher.shape[1] != lower.shape[1]:
            # Interpolate if sequence lengths differ
            higher = F.interpolate(
                higher.transpose(1, 2),
                size=lower.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        projected = self.projection(higher)
        
        # Compute gate
        combined = torch.cat([lower, projected], dim=-1)
        gate = self.gate(combined)
        
        return gate * projected


class HierarchicalLevel(nn.Module):
    """Single level in the hierarchical representation."""
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.top_down = TopDownProjection(d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        top_down_signal: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input with optional top-down modulation.
        
        Implements: h^l = TransformerBlock(h^{l-1} + TopDown(h^{l+1}))
        
        Args:
            x: Input representation h^{l-1} [B, L, D]
            top_down_signal: Higher-level representation h^{l+1} [B, L', D]
            mask: Attention mask
            
        Returns:
            Output representation h^l [B, L, D]
        """
        # Add top-down signal if available
        if top_down_signal is not None:
            td = self.top_down(x, top_down_signal)
            x = x + td
        
        # Apply transformer block
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class HierarchicalRepresentation(nn.Module):
    """
    Hierarchical Representation Network with Bidirectional Connections
    
    Implements multi-level processing with both bottom-up (feedforward)
    and top-down (feedback) information flow, as shown in Figure 2b.
    
    The representation at level l is computed as:
        h^l = TransformerBlock(h^{l-1} + TopDown(h^{l+1}))
    
    Example for processing "cat on mat":
        - h^0 = word embeddings [cat, on, mat]
        - h^1 = TransformerBlock(h^0 + TopDown(h^2))
        - h^2 = TransformerBlock(h^1)  # highest level, no top-down
    
    Args:
        d_model: Model dimensionality (default: 768)
        n_levels: Number of hierarchical levels (default: 3)
        n_heads: Number of attention heads per level (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_levels: int = 3,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_levels = n_levels
        self.d_model = d_model
        
        # Create hierarchical levels
        self.levels = nn.ModuleList([
            HierarchicalLevel(d_model, n_heads, d_ff, dropout)
            for _ in range(n_levels)
        ])
        
        # Chunk-aware pooling for each level
        self.chunk_pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh()
            )
            for _ in range(n_levels)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model * n_levels, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_all_levels: bool = False
    ) -> torch.Tensor:
        """
        Apply hierarchical processing with bidirectional connections.
        
        Args:
            x: Input features [batch_size, seq_length, d_model]
            boundaries: Chunk boundary indicators [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length]
            return_all_levels: Whether to return representations from all levels
            
        Returns:
            Hierarchical representation [batch_size, seq_length, d_model]
            or list of representations if return_all_levels=True
        """
        B, L, D = x.shape
        
        # First pass: bottom-up processing
        level_outputs = []
        current = x
        
        for level in self.levels:
            current = level(current, top_down_signal=None, mask=mask)
            level_outputs.append(current)
        
        # Second pass: top-down refinement
        # Process from highest to lowest level
        refined_outputs = [level_outputs[-1]]  # Highest level unchanged
        
        for i in range(self.n_levels - 2, -1, -1):
            # Get top-down signal from the level above
            top_down_signal = refined_outputs[-1]
            
            # Refine current level with top-down information
            refined = self.levels[i](
                level_outputs[i],
                top_down_signal=top_down_signal,
                mask=mask
            )
            refined_outputs.append(refined)
        
        # Reverse to get low-to-high order
        refined_outputs = refined_outputs[::-1]
        
        if return_all_levels:
            return refined_outputs
        
        # Apply chunk-aware pooling and combine levels
        pooled_outputs = []
        for i, (output, pool) in enumerate(zip(refined_outputs, self.chunk_pool)):
            if boundaries is not None:
                # Weight by boundary proximity (emphasize chunk boundaries)
                weights = 1.0 + 0.5 * boundaries.unsqueeze(-1)
                weighted = output * weights
                pooled = pool(weighted)
            else:
                pooled = pool(output)
            pooled_outputs.append(pooled)
        
        # Concatenate and project
        combined = torch.cat(pooled_outputs, dim=-1)
        output = self.output_proj(combined)
        
        return output
    
    def get_level_representations(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get representations from all hierarchical levels.
        
        Useful for analysis and visualization of level-specific features.
        
        Args:
            x: Input features [B, L, D]
            mask: Attention mask [B, L]
            
        Returns:
            List of representations, one per level
        """
        return self.forward(x, boundaries=None, mask=mask, return_all_levels=True)
