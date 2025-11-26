"""Fusion Models for Multimodal Protein Function Prediction.

Implements multiple fusion techniques for fair comparison:
- Concatenation (baseline)
- Average pooling (baseline)
- Gated fusion (learned weighted combination)
- Transformer attention fusion (cross-attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModalityEncoder(nn.Module):
    """Projects modality embedding to common hidden space."""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# Fusion Modules
# ============================================================================

class ConcatFusion(nn.Module):
    """Concatenation fusion - simple baseline."""
    
    def __init__(self, hidden_dim, n_modalities=4, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, H, M):
        """
        Args:
            H: [B, 4, hidden_dim] - encoded modalities
            M: [B, 4] - binary masks (1=present, 0=missing)
        Returns:
            z: [B, hidden_dim] - fused representation
            weights: [B, 4] - uniform weights for compatibility
        """
        B = H.size(0)
        # Flatten and project
        H_flat = H.view(B, -1)  # [B, 4*hidden_dim]
        z = self.projection(H_flat)
        
        # Return uniform weights for logging compatibility
        weights = M / M.sum(dim=1, keepdim=True).clamp(min=1)
        return z, weights


class AverageFusion(nn.Module):
    """Average pooling fusion - simple baseline."""
    
    def __init__(self, hidden_dim, n_modalities=4):
        super().__init__()
        # No learnable parameters
    
    def forward(self, H, M):
        """
        Args:
            H: [B, 4, hidden_dim] - encoded modalities
            M: [B, 4] - binary masks
        Returns:
            z: [B, hidden_dim] - fused representation
            weights: [B, 4] - uniform weights
        """
        M_expanded = M.unsqueeze(-1)  # [B, 4, 1]
        H_masked = H * M_expanded
        
        # Average over available modalities
        z = H_masked.sum(dim=1) / M.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Return uniform weights
        weights = M / M.sum(dim=1, keepdim=True).clamp(min=1)
        return z, weights


class GatedFusion(nn.Module):
    """Gated fusion with learned attention weights."""
    
    def __init__(self, hidden_dim, n_modalities=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, H, M):
        """
        Args:
            H: [B, 4, hidden_dim] - encoded modalities
            M: [B, 4] - binary masks
        Returns:
            z: [B, hidden_dim] - fused representation
            weights: [B, 4] - learned attention weights
        """
        # Compute gate scores
        raw = self.gate(H).squeeze(-1)  # [B, 4]
        
        # Mask out missing modalities
        raw = raw + (M - 1) * 1e9
        
        # Softmax over available modalities
        weights = torch.softmax(raw, dim=1)  # [B, 4]
        
        # Weighted combination
        weights_expanded = weights.unsqueeze(-1)  # [B, 4, 1]
        z = (weights_expanded * H).sum(dim=1)  # [B, hidden_dim]
        
        return z, weights


class TransformerFusion(nn.Module):
    """Transformer attention fusion with cross-attention between modalities."""
    
    def __init__(self, hidden_dim, n_modalities=4, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        
        # Learnable modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(n_modalities, hidden_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Cross-attention for final aggregation
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, H, M):
        """
        Args:
            H: [B, 4, hidden_dim] - encoded modalities
            M: [B, 4] - binary masks
        Returns:
            z: [B, hidden_dim] - fused representation
            weights: [B, 4] - attention weights
        """
        B = H.size(0)
        
        # Add modality embeddings
        H = H + self.modality_embeddings.unsqueeze(0)
        
        # Create attention mask (True = masked out)
        attn_mask = (M == 0)  # [B, 4]
        
        # Self-attention among modalities
        H_transformed = self.transformer(
            H, 
            src_key_padding_mask=attn_mask
        )
        
        # Cross-attention: query attends to modalities
        query = self.query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        
        z, attn_weights = self.cross_attn(
            query, H_transformed, H_transformed,
            key_padding_mask=attn_mask
        )
        
        z = self.output_norm(z.squeeze(1))  # [B, hidden_dim]
        weights = attn_weights.squeeze(1)  # [B, 4]
        
        return z, weights


class SeqAnchoredFusion(nn.Module):
    """Sequence-anchored cross-attention fusion.
    
    Uses sequence as the primary representation (query).
    Other modalities provide supplementary context (keys/values).
    Preserves sequence information via residual connection.
    """
    
    def __init__(self, hidden_dim, n_modalities=4, n_heads=4, n_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Modality embeddings for context modalities (text, struct, ppi)
        self.context_embeddings = nn.Parameter(
            torch.randn(n_modalities - 1, hidden_dim) * 0.02
        )
        
        # Cross-attention: sequence queries other modalities
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Layer norms and FFN for each layer
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(n_layers)
        ])
        
        # Learned residual weight (how much to trust other modalities)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, H, M):
        """
        Args:
            H: [B, 4, hidden_dim] - [seq, text, struct, ppi]
            M: [B, 4] - binary masks
        Returns:
            z: [B, hidden_dim] - fused representation
            weights: [B, 4] - attention weights (seq weight is residual contribution)
        """
        B = H.size(0)
        
        # Split sequence (query) from context (key/value)
        h_seq = H[:, 0:1, :]  # [B, 1, hidden_dim]
        h_context = H[:, 1:, :]  # [B, 3, hidden_dim] - text, struct, ppi
        
        # Add modality embeddings to context
        h_context = h_context + self.context_embeddings.unsqueeze(0)
        
        # Mask for context modalities (text, struct, ppi)
        context_mask = (M[:, 1:] == 0)  # [B, 3], True = masked
        
        # Store attention weights from last layer
        attn_weights = None
        
        # Cross-attention layers
        q = h_seq
        for i, (cross_attn, ln1, ln2, ffn) in enumerate(zip(
            self.cross_attn_layers, self.layer_norms1, self.layer_norms2, self.ffns
        )):
            # Cross-attention: sequence attends to context
            attn_out, attn_weights = cross_attn(
                q, h_context, h_context,
                key_padding_mask=context_mask
            )
            
            # Residual + LayerNorm
            q = ln1(q + attn_out)
            
            # FFN + Residual + LayerNorm
            q = ln2(q + ffn(q))
        
        # q is now sequence enriched by context: [B, 1, hidden_dim]
        h_enriched = q.squeeze(1)  # [B, hidden_dim]
        h_seq_orig = H[:, 0, :]  # [B, hidden_dim]
        
        # Gated residual: preserve original sequence + enriched
        gate = torch.sigmoid(self.residual_gate)
        z = (1 - gate) * h_seq_orig + gate * h_enriched
        
        # Construct full attention weights [B, 4]
        # First position is "how much we kept original sequence"
        seq_weight = (1 - gate).expand(B, 1)
        context_weights = gate * attn_weights.squeeze(1)  # [B, 3]
        full_weights = torch.cat([seq_weight, context_weights], dim=1)  # [B, 4]
        
        return z, full_weights


# ============================================================================
# Main Model
# ============================================================================

class MultiModalFusionModel(nn.Module):
    """Multimodal protein function prediction with configurable fusion.
    
    Args:
        fusion_type: One of 'concat', 'average', 'gated', 'transformer', 'seq_anchored'
        modality_dropout: Dropout rate for struct/ppi modalities during training
    """
    
    FUSION_TYPES = ['concat', 'average', 'gated', 'transformer', 'seq_anchored']
    
    def __init__(
        self,
        seq_dim,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=512,
        num_go_terms=100,
        dropout=0.3,
        fusion_type='gated',
        modality_dropout=0.0,
        # Transformer-specific
        n_heads=4,
        n_layers=2,
    ):
        super().__init__()
        
        assert fusion_type in self.FUSION_TYPES, f"fusion_type must be one of {self.FUSION_TYPES}"
        
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        
        # Per-modality encoders
        self.enc_seq = ModalityEncoder(seq_dim, hidden_dim, dropout)
        self.enc_text = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.enc_struct = ModalityEncoder(struct_dim, hidden_dim, dropout)
        self.enc_ppi = ModalityEncoder(ppi_dim, hidden_dim, dropout)
        
        # Fusion module
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(hidden_dim, n_modalities=4, dropout=dropout)
        elif fusion_type == 'average':
            self.fusion = AverageFusion(hidden_dim, n_modalities=4)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(hidden_dim, n_modalities=4)
        elif fusion_type == 'transformer':
            self.fusion = TransformerFusion(
                hidden_dim, n_modalities=4, 
                n_heads=n_heads, n_layers=n_layers, dropout=dropout
            )
        elif fusion_type == 'seq_anchored':
            self.fusion = SeqAnchoredFusion(
                hidden_dim, n_modalities=4,
                n_heads=n_heads, n_layers=1, dropout=dropout
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def _apply_modality_dropout(self, masks, training):
        """Randomly drop struct and ppi modalities during training."""
        if not training or self.modality_dropout == 0:
            return masks
        
        # Only drop struct and ppi (indices 2, 3)
        dropout_mask = torch.rand_like(masks[:, 2:]) > self.modality_dropout
        
        new_masks = masks.clone()
        new_masks[:, 2:] = new_masks[:, 2:] * dropout_mask.float()
        
        return new_masks
    
    def forward(self, seq, seq_mask, text, text_mask, struct, struct_mask, ppi, ppi_mask):
        """
        Forward pass.
        
        Returns:
            logits: [B, num_go_terms]
            fusion_weights: [B, 4] - weights for each modality
        """
        # Encode modalities
        h_seq = self.enc_seq(seq)
        h_text = self.enc_text(text)
        h_struct = self.enc_struct(struct)
        h_ppi = self.enc_ppi(ppi)
        
        # Stack: [B, 4, hidden_dim]
        H = torch.stack([h_seq, h_text, h_struct, h_ppi], dim=1)
        
        # Masks: [B, 4]
        M = torch.cat([seq_mask, text_mask, struct_mask, ppi_mask], dim=1)
        
        # Apply modality dropout during training
        M = self._apply_modality_dropout(M, self.training)
        
        # Zero out hidden states for missing modalities
        M_expanded = M.unsqueeze(-1)
        H = H * M_expanded
        
        # Fuse modalities
        z, fusion_weights = self.fusion(H, M)
        
        # Classify
        logits = self.classifier(z)
        
        return logits, fusion_weights