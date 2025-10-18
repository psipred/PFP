"""Concatenation baseline model - simple concatenation of all field embeddings."""

import torch
import torch.nn as nn


class ConcatModel(nn.Module):
    """
    Simple concatenation baseline that concatenates all field embeddings
    and applies a classifier on top.
    """
    
    def __init__(self, num_go_terms, hidden_dim=768, dropout=0.1):
        super().__init__()
        
        self.num_fields = 17  # Number of text fields
        
        # Classifier on concatenated embeddings
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_fields, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def forward(self, field_hidden_states):
        """
        Args:
            field_hidden_states: List of tensors
                - Most fields: [batch_size, 1, hidden_dim] (CLS token only)
                - Function field (idx=3): [batch_size, seq_len, hidden_dim] (full sequence)
        
        Returns:
            logits: [batch_size, num_go_terms]
        """
        cls_tokens = []
        
        for idx, h in enumerate(field_hidden_states):
            if h.dim() == 3:
                if h.size(1) == 1:
                    # CLS token only: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
                    cls_tokens.append(h.squeeze(1))
                else:
                    # Full sequence (Function field): mean pool over sequence
                    # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
                    cls_tokens.append(h.mean(dim=1))
            elif h.dim() == 2:
                # Already [batch_size, hidden_dim]
                cls_tokens.append(h)
            else:
                raise ValueError(f"Unexpected tensor shape at field {idx}: {h.shape}")
        
        # Concatenate all field representations
        concat_features = torch.cat(cls_tokens, dim=1)  # [batch_size, hidden_dim * num_fields]
        
        # Classify
        logits = self.classifier(concat_features)
        
        return logits