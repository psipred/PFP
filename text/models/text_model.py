"""Text-based GO prediction model."""

import torch
import torch.nn as nn


class TextFusionModel(nn.Module):
    """
    Multi-field text fusion model for GO prediction.
    
    Architecture:
    1. Per-field transformer encoders (17 fields)
    2. Cross-attention between non-function fields and function field
    3. Classification head
    """
    
    def __init__(self, num_go_terms: int, hidden_dim: int = 768):
        super().__init__()
        
        self.num_fields = 17
        self.hidden_dim = hidden_dim
        
        # Per-field encoders
        self.field_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4, 
                dropout=0.1, 
                batch_first=True
            ) 
            for _ in range(self.num_fields)
        ])
        
        # Encoder for non-function fields
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dropout=0.1, 
            batch_first=True
        )
        self.suffix_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Cross-attention layers
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, 
                num_heads=4, 
                dropout=0.1, 
                batch_first=True
            ) 
            for _ in range(4)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(4)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_go_terms)
        )
    
    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: List of 17 field embeddings
                - Fields 0-2, 4-16: (batch, 1, hidden_dim) [CLS only]
                - Field 3: (batch, seq_len, hidden_dim) [Full function text]
        
        Returns:
            logits: (batch, num_go_terms)
        """
        # Encode each field
        field_outputs = [
            self.field_encoders[i](hidden_states_list[i]) 
            for i in range(self.num_fields)
        ]
        
        # Extract CLS tokens from non-function fields
        cls_tokens = [
            field_outputs[i][:, 0, :].unsqueeze(1) 
            for i in range(len(field_outputs)) if i != 3
        ]
        cls_sequence = torch.cat(cls_tokens, dim=1)  # (batch, 16, hidden_dim)
        
        # Encode CLS sequence
        cls_encoded = self.suffix_encoder(cls_sequence)
        
        # Get function field encoding
        function_field = field_outputs[3]  # (batch, seq_len, hidden_dim)
        
        # Cross-attention: CLS sequence attends to function field
        x = cls_encoded
        for i in range(4):
            residual = x
            x, _ = self.cross_attentions[i](x, function_field, function_field)
            x = self.layer_norms[i](x + residual)
        
        # Pool and classify
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits