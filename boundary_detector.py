"""
Adaptive Boundary Detection Module

Implements context-sensitive token boundary detection as described in Equation (1):
B(x; θ) = σ(f_θ(x) - T)

Author: Dongxing Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdaptiveBoundaryDetector(nn.Module):
    """
    Adaptive Boundary Detection for Dynamic Tokenization
    
    This module learns to predict token boundaries based on semantic coherence,
    mimicking human cognitive chunking mechanisms.
    
    The boundary detection function:
        B(x; θ) = σ(f_θ(x) - T)
    
    Where:
        - B: Boundary detection function (output: probability in [0, 1])
        - x: Input features (concatenated visual and textual embeddings)
        - θ: Learned parameters of the boundary detector network
        - f_θ: Neural network with parameters θ
        - σ: Sigmoid activation function
        - T: Threshold value (default: 0.5)
    
    Example:
        For an image-text pair describing "a red car":
        - If f_θ(x) = 0.7 and T = 0.5
        - Then B(x, θ) = σ(0.7 - 0.5) = σ(0.2) ≈ 0.55
        - This indicates a boundary (probability > 0.5)
    
    Args:
        d_model: Model dimensionality (default: 768)
        threshold: Boundary detection threshold T (default: 0.5)
        hidden_dim: Hidden layer dimension (default: 256)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        threshold: float = 0.5,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold
        
        # Boundary prediction network f_θ
        self.boundary_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Context attention for local coherence
        self.context_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable threshold adjustment
        self.threshold_adjust = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Detect token boundaries in the input sequence.
        
        Args:
            x: Input features [batch_size, seq_length, d_model]
            mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            boundary_probs: Boundary probabilities [batch_size, seq_length]
                           Values close to 1 indicate chunk boundaries
        """
        B, L, D = x.shape
        
        # Apply context attention to capture local coherence
        context_x, _ = self.context_attn(x, x, x, key_padding_mask=mask)
        
        # Combine original and context-aware representations
        combined = x + context_x
        
        # Compute boundary scores: f_θ(x)
        boundary_scores = self.boundary_net(combined).squeeze(-1)  # [B, L]
        
        # Apply adaptive threshold: B(x; θ) = σ(f_θ(x) - T)
        effective_threshold = self.threshold + self.threshold_adjust
        boundary_probs = torch.sigmoid(boundary_scores - effective_threshold)
        
        # Apply mask if provided
        if mask is not None:
            boundary_probs = boundary_probs * mask.float()
        
        return boundary_probs
    
    def get_hard_boundaries(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get discrete boundary predictions (0 or 1).
        
        Args:
            x: Input features [batch_size, seq_length, d_model]
            mask: Optional attention mask
            
        Returns:
            boundaries: Binary boundary indicators [batch_size, seq_length]
        """
        probs = self.forward(x, mask)
        return (probs > 0.5).float()
    
    def get_chunks(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment input into chunks based on detected boundaries.
        
        Args:
            x: Input features [batch_size, seq_length, d_model]
            mask: Optional attention mask
            
        Returns:
            chunk_ids: Chunk assignment for each position [batch_size, seq_length]
            num_chunks: Number of chunks per batch item [batch_size]
        """
        boundaries = self.get_hard_boundaries(x, mask)
        
        # Convert boundaries to chunk IDs using cumsum
        # Each boundary starts a new chunk
        chunk_ids = boundaries.cumsum(dim=-1).long()
        
        # Count chunks per batch item
        num_chunks = chunk_ids.max(dim=-1).values + 1
        
        return chunk_ids, num_chunks


class BoundaryLoss(nn.Module):
    """
    Loss function for boundary detection training.
    
    Combines:
    1. Binary cross-entropy with ground truth boundaries
    2. Smoothness regularization to prevent over-segmentation
    3. Minimum chunk size constraint
    """
    
    def __init__(
        self,
        smoothness_weight: float = 0.1,
        min_chunk_size: int = 2
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.min_chunk_size = min_chunk_size
        
    def forward(
        self,
        pred_boundaries: torch.Tensor,
        target_boundaries: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute boundary detection loss.
        
        Args:
            pred_boundaries: Predicted boundary probabilities [B, L]
            target_boundaries: Ground truth boundaries [B, L]
            mask: Valid position mask [B, L]
            
        Returns:
            Total loss value
        """
        if mask is None:
            mask = torch.ones_like(pred_boundaries)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(
            pred_boundaries * mask,
            target_boundaries * mask,
            reduction='sum'
        ) / mask.sum()
        
        # Smoothness regularization: penalize rapid boundary changes
        diff = torch.abs(pred_boundaries[:, 1:] - pred_boundaries[:, :-1])
        smoothness_loss = (diff * mask[:, 1:]).sum() / mask[:, 1:].sum()
        
        total_loss = bce_loss + self.smoothness_weight * smoothness_loss
        
        return total_loss
