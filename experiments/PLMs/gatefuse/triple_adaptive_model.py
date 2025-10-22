"""Triple modality adaptive fusion model for PLM experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleModalityAdaptiveFusion(nn.Module):
    def __init__(self, text_dim=768, prott5_dim=1024, esm_dim=1280, 
                 hidden_dim=384, num_go_terms=677, dropout=0.3,  # Reduced size
                 use_diversity_loss=False, diversity_weight=0.01,
                 gate_entropy_weight=0.001):
        super().__init__()
        


        self.use_diversity_loss = use_diversity_loss
        # Transform to common dimension
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(prott5_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.esm_transform = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gates computed from NORMALIZED features (not raw)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Takes transformed features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        # Simpler gate adjuster
        self.gate_adjuster = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        # Simplified interactions
        self.text_esm_interaction = nn.Linear(hidden_dim * 2, hidden_dim)
        self.prott5_esm_interaction = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Smaller fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_go_terms)
        )
        
        self.use_diversity_loss = use_diversity_loss
        self.diversity_weight = diversity_weight
        self.gate_entropy_weight = gate_entropy_weight




        
    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        """Encourage diversity between representations."""
        sim_tp = F.cosine_similarity(text_h, prott5_h, dim=-1).mean()
        sim_te = F.cosine_similarity(text_h, esm_h, dim=-1).mean()
        sim_pe = F.cosine_similarity(prott5_h, esm_h, dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0


        
          
    def compute_adaptive_gates(self, text_h, prott5_h, esm_h):
        """Compute gates from NORMALIZED features."""
        concat_features = torch.cat([text_h, prott5_h, esm_h], dim=-1)
        
        gate_logits = self.gate_network(concat_features)
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
    
    def compute_gate_entropy(self, gates):
        """Encourage balanced gate usage (prevent collapse)."""
        # gates: [batch, 3]
        gates = torch.cat([gates[0], gates[1], gates[2]], dim=-1)
        entropy = -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean()
        return -entropy  # Negative because we want to maximize entropy
        
    def forward(self, text_features, prott5_features, esm_features):
        # Transform to common space (normalized)
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)
        
        # Compute gates from NORMALIZED features
        text_gate, prott5_gate, esm_gate = self.compute_adaptive_gates(
            text_h, prott5_h, esm_h
        )
        
        # Apply gates
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate
        
        # Simplified interactions
        text_esm_interact = F.relu(self.text_esm_interaction(
            torch.cat([gated_text, gated_esm], dim=-1)
        ))
        prott5_esm_interact = F.relu(self.prott5_esm_interaction(
            torch.cat([gated_prott5, gated_esm], dim=-1)
        ))
        
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_esm_interact, prott5_esm_interact
        ], dim=-1)
        
        output = self.fusion(combined)
        
        # Regularization losses
        diversity_loss = None
        gate_entropy_loss = None
        
        if self.training:
            if self.use_diversity_loss:
                diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
            
            gate_entropy_loss = self.compute_gate_entropy(
                [text_gate, prott5_gate, esm_gate]
            )
        
        return output, diversity_loss, gate_entropy_loss