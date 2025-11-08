"""Improved pairwise modality adaptive fusion model with PPI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ImprovedPairwiseModalityFusion(nn.Module):
    def __init__(self, modality_pair, dim_config, 
                 hidden_dim=512, num_go_terms=677, dropout=0.3,
                 use_ppi=False, ppi_dim=512,
                 modality_dropout_rate=0.1, ppi_dropout_rate=0.4):
        super().__init__()
        
        self.modality_pair = modality_pair
        self.use_ppi = use_ppi
        self.modality_dropout_rate = modality_dropout_rate
        self.ppi_dropout_rate = ppi_dropout_rate
        
        # Parse modalities
        mod1, mod2 = modality_pair.split('_')
        self.mod1_name = mod1
        self.mod2_name = mod2
        dim1 = dim_config[mod1]
        dim2 = dim_config[mod2]
        
        # Transform to common dimension with residual connections
        self.mod1_transform = nn.Sequential(
            nn.Linear(dim1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mod2_transform = nn.Sequential(
            nn.Linear(dim2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # PPI handling with learnable missing embedding
        if use_ppi:
            self.ppi_transform = nn.Sequential(
                nn.Linear(ppi_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            # Learnable missing PPI embedding
            self.ppi_missing_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        # Presence-aware gate network (CONDITIONS ON FLAGS)
        gate_input_dim = hidden_dim * 2 + (1 if use_ppi else 0)
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Temperature with better initialization
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Cross-attention for richer fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # FiLM modulation from gates
        self.film_generator = nn.Linear(2, hidden_dim * 2)
        
        # Enhanced fusion with proper input dimension
        fusion_dim = hidden_dim * 3  # h1, h2, cross_attended
        if use_ppi:
            fusion_dim += hidden_dim  # Add PPI
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
        
        # Per-aspect prediction heads (optional - can be added later)
        self.use_aspect_heads = False
        
    def compute_decorrelation_loss(self, h1, h2):
        """Barlow Twins-style decorrelation (better than diversity loss)."""
        batch_size = h1.size(0)
        
        # Normalize
        h1 = (h1 - h1.mean(dim=0, keepdim=True)) / (h1.std(dim=0, keepdim=True) + 1e-6)
        h2 = (h2 - h2.mean(dim=0, keepdim=True)) / (h2.std(dim=0, keepdim=True) + 1e-6)
        
        # Cross-correlation matrix
        c = torch.mm(h1.T, h2) / batch_size
        
        # Decorrelation loss (minimize off-diagonal)
        loss = (c ** 2).mean()
        return loss
    
    def compute_gate_entropy(self, gates):
        """Entropy regularization to prevent collapse."""
        eps = 1e-8
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1).mean()
        return entropy  # Return positive entropy, not negative
    
    def forward(self, mod1_features, mod2_features, ppi_features=None, ppi_flag=None):
        batch_size = mod1_features.size(0)
        device = mod1_features.device
        
        # MODALITY DROPOUT during training
        if self.training:
            # Randomly drop modalities
            drop_mod1 = torch.rand(1).item() < self.modality_dropout_rate
            drop_mod2 = torch.rand(1).item() < self.modality_dropout_rate
            
            # Never drop both primary modalities
            if drop_mod1 and drop_mod2:
                drop_mod2 = False
                
            if drop_mod1:
                mod1_features = torch.zeros_like(mod1_features)
            if drop_mod2:
                mod2_features = torch.zeros_like(mod2_features)
                
            # Drop PPI more aggressively
            if self.use_ppi and ppi_flag is not None:
                ppi_dropout_mask = torch.rand(batch_size, 1, device=device) < self.ppi_dropout_rate
                ppi_flag = ppi_flag * (~ppi_dropout_mask).float()
        
        # Transform modalities
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        # Handle PPI with learnable missing embedding
        if self.use_ppi:
            if ppi_features is not None and ppi_flag is not None:
                ppi_h = self.ppi_transform(ppi_features)
                # Blend real PPI with learned missing embedding
                ppi_flag = ppi_flag.view(batch_size, 1)
                ppi_h = ppi_flag * ppi_h + (1 - ppi_flag) * self.ppi_missing_embed.unsqueeze(0)
            else:
                ppi_h = self.ppi_missing_embed.unsqueeze(0).expand(batch_size, -1)
                ppi_flag = torch.zeros(batch_size, 1, device=device)
        
        # PRESENCE-AWARE GATES
        if self.use_ppi:
            gate_input = torch.cat([h1, h2, ppi_flag], dim=-1)
        else:
            gate_input = torch.cat([h1, h2], dim=-1)
            
        gate_logits = self.gate_network(gate_input)
        
        # Temperature-controlled softmax with clamping
        temperature = F.softplus(self.temperature) + 0.5  # Ensure positive and bounded
        gates = F.softmax(gate_logits / temperature, dim=-1)
        gate1, gate2 = gates[:, 0:1], gates[:, 1:2]
        
        # FiLM modulation
        film_params = self.film_generator(gates)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Apply FiLM before gating
        h1_modulated = h1 * (1 + gamma) + beta
        h2_modulated = h2 * (1 + gamma) + beta
        
        # Apply gates
        gated_h1 = h1_modulated * gate1
        gated_h2 = h2_modulated * gate2
        
        # CROSS-ATTENTION for richer interaction
        h1_unsqueezed = gated_h1.unsqueeze(1)  # [B, 1, D]
        h2_unsqueezed = gated_h2.unsqueeze(1)  # [B, 1, D]
        
        # h1 attends to h2
        h1_attended, _ = self.cross_attention(
            h1_unsqueezed, h2_unsqueezed, h2_unsqueezed
        )
        h1_attended = h1_attended.squeeze(1)
        
        # Combine all features
        if self.use_ppi:
            combined = torch.cat([gated_h1, gated_h2, h1_attended, ppi_h], dim=-1)
        else:
            combined = torch.cat([gated_h1, gated_h2, h1_attended], dim=-1)
        
        # Final prediction
        output = self.fusion(combined)
        
        # Compute regularization losses
        decorr_loss = self.compute_decorrelation_loss(h1, h2) if self.training else None
        entropy_loss = self.compute_gate_entropy(gates) if self.training else None
        
        return output, decorr_loss, entropy_loss


class ClassBalancedBCELoss(nn.Module):
    """Class-balanced BCE loss for extreme imbalance."""
    
    def __init__(self, pos_counts, neg_counts, beta=0.99):
        super().__init__()
        self.beta = beta
        
        # Handle zero counts
        pos_counts = np.maximum(pos_counts, 1)
        neg_counts = np.maximum(neg_counts, 1)
        
        # Effective number of samples
        effective_pos = (1 - beta ** pos_counts) / (1 - beta + 1e-8)
        effective_neg = (1 - beta ** neg_counts) / (1 - beta + 1e-8)
        
        # Class weights
        pos_weight = (1.0 / effective_pos) / (1.0 / effective_pos + 1.0 / effective_neg)
        neg_weight = (1.0 / effective_neg) / (1.0 / effective_pos + 1.0 / effective_neg)
        
        # Convert to tensors and register as buffers (will auto-move to device)
        self.register_buffer('pos_weight', torch.FloatTensor(pos_weight))
        self.register_buffer('neg_weight', torch.FloatTensor(neg_weight))
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Weighted BCE
        pos_loss = -targets * torch.log(probs + 1e-8) * self.pos_weight.unsqueeze(0)
        neg_loss = -(1 - targets) * torch.log(1 - probs + 1e-8) * self.neg_weight.unsqueeze(0)
        
        return (pos_loss + neg_loss).mean()