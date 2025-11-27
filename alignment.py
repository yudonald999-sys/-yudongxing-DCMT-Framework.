"""
Cross-Modal Alignment Module

Implements contrastive learning for visual-linguistic correspondence
as described in Equation (3):
L_align = -log(exp(S(V_i, t_i) / τ) / Σ_j exp(S(V_i, t_j) / τ))

Author: Dongxing Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CrossModalAlignment(nn.Module):
    """
    Cross-Modal Alignment Module
    
    Learns to align visual and textual representations using contrastive
    learning, promoting correspondence between matched pairs while
    separating unmatched pairs.
    
    The alignment loss (Equation 3):
        L_align = -log(exp(S(V_i, t_i) / τ) / Σ_j exp(S(V_i, t_j) / τ))
    
    This formula is the contrastive alignment loss (similar to CLIP) which
    trains paired embeddings to be more similar than mismatched pairs.
    
    Where:
        - S(V_i, t_i): Similarity between visual embedding V_i and text embedding t_i
        - τ (tau): Temperature parameter controlling distribution sharpness
        - The numerator represents the matched pair
        - The denominator sums over all pairs in the batch
    
    Example:
        For a matched visual-text pair with:
        - S(V_i, t_i) = 0.9 (high similarity for matched pair)
        - Average S(V_i, t_j) = 0.2 for unmatched pairs
        - τ = 0.07
        - L_align ≈ 0.15 (low loss indicates good alignment)
    
    Args:
        d_model: Model dimensionality (default: 768)
        temperature: Temperature parameter τ (default: 0.07)
        projection_dim: Dimension of projection space (default: 256)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        temperature: float = 0.07,
        projection_dim: int = 256
    ):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Projection heads for visual and textual features
        self.visual_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
        # Cross-attention for fine-grained alignment
        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def compute_similarity(
        self,
        visual: torch.Tensor,
        textual: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix S(V_i, t_j) for all pairs.
        
        Args:
            visual: Visual embeddings [batch_size, projection_dim]
            textual: Text embeddings [batch_size, projection_dim]
            
        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        # L2 normalize
        visual = F.normalize(visual, p=2, dim=-1)
        textual = F.normalize(textual, p=2, dim=-1)
        
        # Cosine similarity
        similarity = torch.matmul(visual, textual.T)
        
        return similarity
    
    def contrastive_loss(
        self,
        similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss (Equation 3).
        
        L_align = -log(exp(S(V_i, t_i) / τ) / Σ_j exp(S(V_i, t_j) / τ))
        
        Args:
            similarity: Similarity matrix [batch_size, batch_size]
            
        Returns:
            Contrastive loss value
        """
        B = similarity.shape[0]
        temperature = self.log_temperature.exp()
        
        # Scale by temperature
        logits = similarity / temperature
        
        # Labels: diagonal entries are positive pairs
        labels = torch.arange(B, device=similarity.device)
        
        # Symmetric loss (V->T and T->V)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        
        loss = (loss_v2t + loss_t2v) / 2
        
        return loss
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute cross-modal alignment.
        
        Args:
            visual_features: Visual representations [B, N_v, D]
            text_features: Text representations [B, N_t, D]
            visual_mask: Visual attention mask [B, N_v]
            text_mask: Text attention mask [B, N_t]
            
        Returns:
            Dictionary containing:
                - 'loss': Contrastive alignment loss
                - 'similarity': Similarity matrix [B, B]
                - 'v2t_attn': Visual-to-text attention weights
                - 't2v_attn': Text-to-visual attention weights
                - 'aligned_visual': Text-aligned visual features
                - 'aligned_text': Visual-aligned text features
        """
        B = visual_features.shape[0]
        
        # Cross-attention for fine-grained alignment
        aligned_visual, v2t_attn = self.cross_attn_v2t(
            visual_features, text_features, text_features,
            key_padding_mask=text_mask
        )
        aligned_text, t2v_attn = self.cross_attn_t2v(
            text_features, visual_features, visual_features,
            key_padding_mask=visual_mask
        )
        
        # Pool to get sequence-level representations
        if visual_mask is not None:
            visual_mask_expanded = visual_mask.unsqueeze(-1).float()
            visual_pooled = (aligned_visual * visual_mask_expanded).sum(1) / visual_mask_expanded.sum(1)
        else:
            visual_pooled = aligned_visual.mean(dim=1)
            
        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            text_pooled = (aligned_text * text_mask_expanded).sum(1) / text_mask_expanded.sum(1)
        else:
            text_pooled = aligned_text.mean(dim=1)
        
        # Project to alignment space
        visual_proj = self.visual_proj(visual_pooled)  # [B, projection_dim]
        text_proj = self.text_proj(text_pooled)  # [B, projection_dim]
        
        # Compute similarity matrix
        similarity = self.compute_similarity(visual_proj, text_proj)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(similarity)
        
        return {
            'loss': loss,
            'similarity': similarity,
            'v2t_attn': v2t_attn,
            't2v_attn': t2v_attn,
            'aligned_visual': aligned_visual,
            'aligned_text': aligned_text,
            'visual_proj': visual_proj,
            'text_proj': text_proj
        }
    
    def get_alignment_scores(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get token-level alignment scores between visual and text tokens.
        
        Args:
            visual_features: [B, N_v, D]
            text_features: [B, N_t, D]
            
        Returns:
            Alignment scores [B, N_v, N_t]
        """
        # Normalize features
        visual_norm = F.normalize(visual_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)
        
        # Compute pairwise similarities
        alignment = torch.bmm(visual_norm, text_norm.transpose(1, 2))
        
        return alignment


class MutualInformationEstimator(nn.Module):
    """
    Mutual Information Neural Estimation (MINE) for measuring
    cross-modal information sharing.
    
    Used to quantify MI reduction between human neural data and model
    representations (as mentioned in the paper: "MI reduction: 68%").
    """
    
    def __init__(self, d_model: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        visual: torch.Tensor,
        textual: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate mutual information between visual and textual features.
        
        Uses the MINE lower bound:
        I(V; T) >= E[T(v, t)] - log(E[exp(T(v, t'))])
        
        Args:
            visual: Visual features [B, D]
            textual: Text features [B, D]
            
        Returns:
            MI estimate (in bits)
        """
        B = visual.shape[0]
        
        # Joint samples (matched pairs)
        joint = torch.cat([visual, textual], dim=-1)
        joint_scores = self.network(joint)
        
        # Marginal samples (shuffled/unmatched pairs)
        perm = torch.randperm(B, device=visual.device)
        textual_shuffled = textual[perm]
        marginal = torch.cat([visual, textual_shuffled], dim=-1)
        marginal_scores = self.network(marginal)
        
        # MINE lower bound
        mi_estimate = joint_scores.mean() - torch.logsumexp(marginal_scores, dim=0) + torch.log(torch.tensor(B, dtype=torch.float))
        
        # Convert nats to bits
        mi_bits = mi_estimate / torch.log(torch.tensor(2.0))
        
        return mi_bits
