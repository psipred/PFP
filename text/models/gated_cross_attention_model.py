"""Gated cross-attention fusion model."""

import torch
import torch.nn as nn


class GatedCrossAttentionFusion(nn.Module):
    """
    Combines cross-attention with gating.
    First does bidirectional cross-attention, then uses gate to balance.
    """
    
    def __init__(self, num_go_terms, text_dim=768, esm_dim=1280, hidden_dim=512):
        super().__init__()
        
        # Text backbone
        self.field_projections = nn.ModuleList([
            nn.Linear(text_dim, hidden_dim) for _ in range(17)
        ])
        self.text_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        # ESM projection
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        
        # Cross-attention layers
        self.text_to_esm = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.esm_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Gate for cross-attended features
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def forward(self, text_fields, esm_emb):
        """
        Args:
            text_fields: List of 17 tensors [batch, hidden_dim]
            esm_emb: Tensor [batch, esm_dim]
        """
        # Process text fields
        text_feats = torch.stack([
            proj(field) for proj, field in zip(self.field_projections, text_fields)
        ], dim=1)  # [batch, 17, hidden_dim]
        text_feats = self.text_attention(text_feats)
        
        # Process ESM
        esm_feat = self.esm_proj(esm_emb).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention in both directions
        text_attended, _ = self.text_to_esm(
            query=text_feats,
            key=esm_feat,
            value=esm_feat
        )
        text_pooled = text_attended.mean(dim=1)  # [batch, hidden_dim]
        
        esm_attended, _ = self.esm_to_text(
            query=esm_feat,
            key=text_feats,
            value=text_feats
        )
        esm_pooled = esm_attended.squeeze(1)  # [batch, hidden_dim]
        
        # Gate the two directions
        gate_input = torch.cat([text_pooled, esm_pooled], dim=-1)
        alpha = self.gate(gate_input)  # [batch, 1]
        
        # Weighted fusion
        fused = alpha * text_pooled + (1 - alpha) * esm_pooled
        
        # Classify
        logits = self.classifier(fused)
        
        return logits