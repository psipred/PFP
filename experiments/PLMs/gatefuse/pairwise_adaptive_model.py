"""Pairwise modality adaptive fusion model with optional PPI complement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseModalityAdaptiveFusion(nn.Module):
    def __init__(self, modality_pair, dim_config, 
                 hidden_dim=512, num_go_terms=677, dropout=0.3,
                 use_ppi=False, ppi_dim=512,
                 use_diversity_loss=False, diversity_weight=0.01,
                 use_gate_entropy=True, gate_entropy_weight=0.001,
                 temperature=1.5, learnable_temperature=True):
        super().__init__()
        
        self.modality_pair = modality_pair
        self.use_diversity_loss = use_diversity_loss
        self.diversity_weight = diversity_weight
        self.use_gate_entropy = use_gate_entropy
        self.gate_entropy_weight = gate_entropy_weight
        self.use_ppi = use_ppi
        
        # Get dimensions from config
        mod1, mod2 = modality_pair.split('_')
        self.mod1_name = mod1
        self.mod2_name = mod2
        dim1 = dim_config[mod1]
        dim2 = dim_config[mod2]
        
        # Transform to common dimension
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
        
        # PPI transform (if used)
        if use_ppi:
            self.ppi_transform = nn.Sequential(
                nn.Linear(ppi_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Gate adjuster
        self.gate_adjuster = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )
        
        # Interaction layer
        self.interaction = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Fusion network - adjust input size based on PPI usage
        if use_ppi:
            fusion_input_dim = hidden_dim * 4 + 1  # gated_h1 + gated_h2 + interact + ppi_h + ppi_flag
        else:
            fusion_input_dim = hidden_dim * 3
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_go_terms)
        )
    
    def compute_diversity_loss(self, h1, h2):
        """Encourage diversity between representations."""
        sim = F.cosine_similarity(h1, h2, dim=-1).mean()
        return sim
    
    def compute_adaptive_gates(self, h1, h2):
        """Compute gates from normalized features."""
        concat_features = torch.cat([h1, h2], dim=-1)
        
        gate_logits = self.gate_network(concat_features)
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        
        return gates[:, 0:1], gates[:, 1:2]
    
    def compute_gate_entropy(self, gates):
        """Encourage balanced gate usage."""
        gates = torch.cat(gates, dim=-1)
        entropy = -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean()
        return -entropy
    
    def forward(self, mod1_features, mod2_features, ppi_features=None, ppi_flag=None):
        # Transform to common space
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        # Compute gates
        gate1, gate2 = self.compute_adaptive_gates(h1, h2)
        
        # Apply gates
        gated_h1 = h1 * gate1
        gated_h2 = h2 * gate2
        
        # Interaction
        interact = F.relu(self.interaction(
            torch.cat([gated_h1, gated_h2], dim=-1)
        ))
        
        # Add PPI if available
        if self.use_ppi and ppi_features is not None:
            ppi_h = self.ppi_transform(ppi_features)
            ppi_h = ppi_h * ppi_flag  # Mask out proteins without PPI
            combined = torch.cat([gated_h1, gated_h2, interact, ppi_h, ppi_flag], dim=-1)
        else:
            combined = torch.cat([gated_h1, gated_h2, interact], dim=-1)
        
        # Fusion
        output = self.fusion(combined)
        
        # Regularization losses
        diversity_loss = None
        gate_entropy_loss = None
        
        if self.training:
            if self.use_diversity_loss:
                diversity_loss = self.compute_diversity_loss(h1, h2)
            if self.use_gate_entropy:
                gate_entropy_loss = self.compute_gate_entropy([gate1, gate2])
        
        return output, diversity_loss, gate_entropy_loss