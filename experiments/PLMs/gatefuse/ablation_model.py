"""Ablation study model for pairwise fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseAblationModel(nn.Module):
    """Pairwise fusion with ablation options."""
    
    def __init__(self, modality_pair, dim_config, 
                 hidden_dim=384, num_go_terms=677, dropout=0.3,
                 # Ablation flags
                 use_adaptive_gates=True,
                 use_gate_adjuster=True,
                 use_interaction=True,
                 use_diversity_loss=False,
                 use_gate_entropy=True,
                 learnable_temperature=True,
                 temperature=1.5,
                 fixed_gate_weights=None,  # [weight_mod1, weight_mod2] if not None
                 diversity_weight=0.01,
                 gate_entropy_weight=0.001):
        super().__init__()
        
        self.modality_pair = modality_pair
        self.use_adaptive_gates = use_adaptive_gates
        self.use_gate_adjuster = use_gate_adjuster
        self.use_interaction = use_interaction
        self.use_diversity_loss = use_diversity_loss
        self.use_gate_entropy = use_gate_entropy
        self.diversity_weight = diversity_weight
        self.gate_entropy_weight = gate_entropy_weight
        
        # Get dimensions
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
        
        # Gate mechanism (only if adaptive)
        if use_adaptive_gates:
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
            
            # Gate adjuster (optional)
            if use_gate_adjuster:
                self.gate_adjuster = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 2),
                    nn.Tanh()
                )
        else:
            # Fixed gates
            if fixed_gate_weights is None:
                fixed_gate_weights = [0.5, 0.5]
            self.register_buffer('fixed_gate1', torch.tensor([fixed_gate_weights[0]]))
            self.register_buffer('fixed_gate2', torch.tensor([fixed_gate_weights[1]]))
        
        # Interaction layer (optional)
        if use_interaction:
            self.interaction = nn.Linear(hidden_dim * 2, hidden_dim)
            fusion_input_dim = hidden_dim * 3
        else:
            fusion_input_dim = hidden_dim * 2
        
        # Fusion network
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
        """Compute adaptive gates."""
        concat_features = torch.cat([h1, h2], dim=-1)
        
        gate_logits = self.gate_network(concat_features)
        
        if self.use_gate_adjuster:
            adjustments = self.gate_adjuster(concat_features)
            gate_logits = gate_logits + adjustments * 0.5
        
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        
        return gates[:, 0:1], gates[:, 1:2]
    
    def compute_gate_entropy(self, gates):
        """Encourage balanced gate usage."""
        gates = torch.cat(gates, dim=-1)
        entropy = -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean()
        return -entropy
    
    def forward(self, mod1_features, mod2_features):
        # Transform to common space
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        # Compute gates
        if self.use_adaptive_gates:
            gate1, gate2 = self.compute_adaptive_gates(h1, h2)
        else:
            # Fixed gates
            batch_size = h1.size(0)
            gate1 = self.fixed_gate1.expand(batch_size, 1)
            gate2 = self.fixed_gate2.expand(batch_size, 1)
        
        # Apply gates
        gated_h1 = h1 * gate1
        gated_h2 = h2 * gate2
        
        # Interaction (optional)
        if self.use_interaction:
            interact = F.relu(self.interaction(
                torch.cat([gated_h1, gated_h2], dim=-1)
            ))
            combined = torch.cat([gated_h1, gated_h2, interact], dim=-1)
        else:
            combined = torch.cat([gated_h1, gated_h2], dim=-1)
        
        # Fusion
        output = self.fusion(combined)
        
        # Regularization losses
        diversity_loss = None
        gate_entropy_loss = None
        
        if self.training:
            if self.use_diversity_loss:
                diversity_loss = self.compute_diversity_loss(h1, h2)
            if self.use_gate_entropy and self.use_adaptive_gates:
                gate_entropy_loss = self.compute_gate_entropy([gate1, gate2])
        
        return output, diversity_loss, gate_entropy_loss