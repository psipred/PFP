"""Triple modality adaptive fusion model for PLM experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleModalityAdaptiveFusion(nn.Module):
    """Triple modality fusion combining text, ProtT5, and ESM embeddings."""
    
    def __init__(self, text_dim=768, prott5_dim=1024, esm_dim=1280, 
                 hidden_dim=512, num_go_terms=677, dropout=0.2):
        super().__init__()
        
        self.text_dim = text_dim
        self.prott5_dim = prott5_dim
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        
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
        
        # Adaptive gates with temperature
        self.gate_network = nn.Sequential(
            nn.Linear(text_dim + prott5_dim + esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 modalities
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        # Context-aware adjustment
        self.gate_adjuster = nn.Sequential(
            nn.Linear(text_dim + prott5_dim + esm_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        # Pairwise interactions
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final fusion (3 modalities + 2 interactions)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
        
        # Diversity regularization
        self.diversity_weight = nn.Parameter(torch.tensor(0.01))
        
    def compute_adaptive_gates(self, text_features, prott5_features, esm_features):
        """Compute temperature-scaled gates for three modalities."""
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        
        # Base gate values
        gate_logits = self.gate_network(concat_features)
        
        # Context-aware adjustments
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        
        # Temperature-controlled softmax
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
    
    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        """Encourage diversity between representations."""
        sim_tp = F.cosine_similarity(text_h, prott5_h, dim=-1).mean()
        sim_te = F.cosine_similarity(text_h, esm_h, dim=-1).mean()
        sim_pe = F.cosine_similarity(prott5_h, esm_h, dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0
        
    def forward(self, text_features, prott5_features, esm_features):
        """Forward pass through the fusion network."""
        # Transform features
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)
        
        # Compute adaptive gates
        text_gate, prott5_gate, esm_gate = self.compute_adaptive_gates(
            text_features, prott5_features, esm_features
        )
        
        # Apply gates
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate
        
        # Compute pairwise interactions
        text_esm_interact = self.text_esm_interaction(
            torch.cat([gated_text, gated_esm], dim=-1)
        )
        prott5_esm_interact = self.prott5_esm_interaction(
            torch.cat([gated_prott5, gated_esm], dim=-1)
        )
        
        # Combine all representations
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_esm_interact, prott5_esm_interact
        ], dim=-1)
        
        # Final prediction
        output = self.fusion(combined)
        
        # Add diversity regularization during training
        diversity_loss = None
        if self.training:
            diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)
        
        return output, diversity_loss