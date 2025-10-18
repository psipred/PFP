"""ESM model with sequence branch architecture - simplified version."""
import torch
import torch.nn as nn
from transformers import EsmModel

class ESMSequenceBranch(nn.Module):
    """
    Simplified ESM sequence branch model.
    Single transformer layer on top of frozen ESM embeddings.
    """
    
    def __init__(self, num_go_terms, esm_model_name="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        
        # Load ESM model
        self.esm = EsmModel.from_pretrained(esm_model_name)
        
        # Freeze ESM parameters
        for param in self.esm.parameters():
            param.requires_grad = False
        
        # Get embedding dimension from ESM config
        embedding_dim = self.esm.config.hidden_size  # 1280
        
        # Single transformer layer for refinement
        self.refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dropout=0.3,
                batch_first=True
            ),
            num_layers=1
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_go_terms)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_go_terms]
        """
        # Get ESM embeddings (frozen)
        with torch.no_grad():
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            seq_outputs = outputs.last_hidden_state  # [batch, seq_len, 1280]
        
        # Apply refinement transformer
        refined = self.refiner(seq_outputs)  # [batch, seq_len, 1280]
        
        # Mean pooling (excluding padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(refined.size()).float()
        sum_embeddings = torch.sum(refined * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        pooled = sum_embeddings / sum_mask.clamp(min=1e-9)  # [batch, 1280]
        
        # Predict
        logits = self.classifier(pooled)  # [batch, num_go_terms]
        
        return logits