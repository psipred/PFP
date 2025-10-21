"""Simple MLP classifier for PLM embeddings."""

import torch.nn as nn


class PLMClassifier(nn.Module):
    """
    3-layer MLP classifier for PLM embeddings.
    Same architecture as the text experiments' ESM baseline.
    """
    
    def __init__(self, num_go_terms, embedding_dim=1280, dropout=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # # Layer 2
            # nn.Linear(512, 512),
            # nn.LayerNorm(512),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(512, num_go_terms)
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, embedding_dim)
        
        Returns:
            logits: (batch, num_go_terms)
        """
        return self.classifier(embeddings)