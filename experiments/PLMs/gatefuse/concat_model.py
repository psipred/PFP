"""Simple concatenation baseline model."""

import torch
import torch.nn as nn


class ConcatenationModel(nn.Module):
    """Simple baseline that concatenates all modality embeddings."""
    
    def __init__(self, text_dim=768, prott5_dim=1024, esm_dim=1280,
                 hidden_dim=512, num_go_terms=677, dropout=0.3):
        super().__init__()
        
        # Total input dimension after concatenation
        total_dim = text_dim + prott5_dim + esm_dim  # 768 + 1024 + 1280 = 3072
        
        # Simple MLP classifier
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def forward(self, text_features, prott5_features, esm_features):
        """
        Args:
            text_features: (batch, 768)
            prott5_features: (batch, 1024)
            esm_features: (batch, 1280)
        
        Returns:
            logits: (batch, num_go_terms)
        """
        # Simple concatenation
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        
        # Pass through classifier
        logits = self.classifier(concat_features)
        
        return logits