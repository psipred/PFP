"""Simple function-only text model for protein function prediction."""

import torch
import torch.nn as nn


class SimpleFunctionModel(nn.Module):
    """
    Simple model using only Function field text embeddings.
    Architecture mirrors ESMClassifier for fair comparison.
    """
    
    def __init__(self, num_go_terms, hidden_dim=768, dropout=0.3):
        super().__init__()
        
        # Simple MLP classifier (same as ESM model)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_go_terms)
        )
    
    def forward(self, function_embedding):
        """
        Args:
            function_embedding: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        
        Returns:
            logits: [batch_size, num_go_terms]
        """
        # Mean pool if sequence dimension exists
        if function_embedding.dim() == 3:
            pooled = function_embedding.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            pooled = function_embedding
        
        logits = self.classifier(pooled)
        return logits