"""Simplified and fixed pairwise modality fusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimplifiedPairwiseFusion(nn.Module):
    def __init__(self, modality_pair, dim_config, 
                 hidden_dim=512, num_go_terms=677, dropout=0.3,
                 use_ppi=False, ppi_dim=512):
        super().__init__()
        
        self.modality_pair = modality_pair
        self.use_ppi = use_ppi
        
        # Parse modalities
        mod1, mod2 = modality_pair.split('_')
        self.mod1_name = mod1
        self.mod2_name = mod2
        dim1 = dim_config[mod1]
        dim2 = dim_config[mod2]
        
        # Simple transform to common dimension
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
            # Learnable missing PPI embedding (KEY IMPROVEMENT)
            self.ppi_missing_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        # FIXED: Presence-aware gate with better initialization
        gate_input_dim = hidden_dim * 2
        if use_ppi:
            gate_input_dim += 1  # Add PPI presence flag
            
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Initialize last layer to zero for balanced gates
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)
        
        # FIXED: Higher initial temperature to prevent collapse
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
        
        # Simplified fusion (removed cross-attention for stability)
        fusion_dim = hidden_dim * 2  # Just gated features
        if use_ppi:
            fusion_dim += hidden_dim  # Add PPI
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def compute_gates(self, h1, h2, ppi_flag=None):
        """Compute adaptive gates with presence awareness."""
        if self.use_ppi and ppi_flag is not None:
            gate_input = torch.cat([h1, h2, ppi_flag], dim=-1)
        else:
            gate_input = torch.cat([h1, h2], dim=-1)
        
        gate_logits = self.gate_network(gate_input)
        
        # FIXED: Proper temperature scaling with minimum value
        temperature = torch.exp(self.log_temperature) + 0.5  # Min temp = 0.5
        gates = F.softmax(gate_logits / temperature, dim=-1)
        
        return gates[:, 0:1], gates[:, 1:2]
    
    def forward(self, mod1_features, mod2_features, ppi_features=None, ppi_flag=None):
        batch_size = mod1_features.size(0)
        device = mod1_features.device
        
        # Transform modalities
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        # Handle PPI with learnable missing embedding (KEY IMPROVEMENT)
        if self.use_ppi:
            if ppi_features is not None and ppi_flag is not None:
                ppi_h = self.ppi_transform(ppi_features)
                ppi_flag = ppi_flag.view(batch_size, 1).float()
                # Blend real PPI with learned missing embedding
                ppi_h = ppi_flag * ppi_h + (1 - ppi_flag) * self.ppi_missing_embed.unsqueeze(0)
            else:
                ppi_h = self.ppi_missing_embed.unsqueeze(0).expand(batch_size, -1)
                ppi_flag = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
        
        # Compute presence-aware gates
        gate1, gate2 = self.compute_gates(h1, h2, ppi_flag if self.use_ppi else None)
        
        # Apply gates with residual connection to prevent complete suppression
        alpha = 0.1  # Residual weight
        gated_h1 = gate1 * h1 + alpha * h1
        gated_h2 = gate2 * h2 + alpha * h2
        
        # Simple fusion
        if self.use_ppi:
            combined = torch.cat([gated_h1, gated_h2, ppi_h], dim=-1)
        else:
            combined = torch.cat([gated_h1, gated_h2], dim=-1)
        
        output = self.fusion(combined)
        
        # Return gates for monitoring
        return output, gate1.mean(), gate2.mean()


def train_epoch_fixed(model, loader, optimizer, criterion, scheduler, device, entropy_weight=0.02):
    """Training with stronger entropy regularization."""
    model.train()
    total_loss = 0
    gate_stats = []
    
    for batch in loader:
        mod1 = batch['mod1'].to(device)
        mod2 = batch['mod2'].to(device)
        ppi = batch['ppi'].to(device) if 'ppi' in batch else None
        ppi_flag = batch['ppi_flag'].to(device) if 'ppi_flag' in batch else None
        labels = batch['labels'].to(device)
        
        logits, gate1_mean, gate2_mean = model(mod1, mod2, ppi, ppi_flag)
        main_loss = criterion(logits, labels)
        
        # STRONG entropy regularization to prevent collapse
        gates = torch.stack([gate1_mean, gate2_mean])
        entropy = -(gates * torch.log(gates + 1e-8) + (1-gates) * torch.log(1-gates + 1e-8)).mean()
        
        loss = main_loss - entropy_weight * entropy  # Maximize entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        gate_stats.append([gate1_mean.item(), gate2_mean.item()])
    
    avg_gates = torch.tensor(gate_stats).mean(0)
    return total_loss / len(loader), avg_gates[0].item(), avg_gates[1].item()


class FocalBCELoss(nn.Module):
    """Focal loss for extreme imbalance - simpler than class-balanced."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()