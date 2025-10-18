"""Cross-modal attention fusion model."""

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between text and ESM.
    Uses TextFusionModel's architecture for text encoding.
    
    Architecture:
    1. Text encoding (same as TextFusionModel):
       - Per-field transformer encoders (17 fields)
       - Cross-attention between non-function fields and function field
    2. ESM projection
    3. Bidirectional cross-attention:
       - Text queries ESM
       - ESM queries text
    4. Fusion and classification
    """
    
    def __init__(self, num_go_terms: int, esm_dim: int = 1280, hidden_dim: int = 768):
        super().__init__()
        
        self.num_fields = 17
        self.hidden_dim = hidden_dim
        
        # ============ TEXT ENCODING (from TextFusionModel) ============
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
        
        # Cross-attention layers (text internal)
        self.text_cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, 
                num_heads=4, 
                dropout=0.1, 
                batch_first=True
            ) 
            for _ in range(4)
        ])
        self.text_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(4)
        ])
        
        # ============ ESM PROCESSING ============
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        
        # ============ CROSS-MODAL ATTENTION ============
        # Text queries ESM
        self.text_to_esm = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # ESM queries text
        self.esm_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # ============ FUSION ============
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ============ CLASSIFICATION ============
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_go_terms)
        )
    
    def encode_text(self, hidden_states_list):
        """
        Encode text using TextFusionModel architecture.
        
        Args:
            hidden_states_list: List of 17 field embeddings
                - Fields 0-2, 4-16: (batch, 1, hidden_dim) [CLS only]
                - Field 3: (batch, seq_len, hidden_dim) [Full function text]
        
        Returns:
            text_representation: (batch, 16, hidden_dim) - encoded text features
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
            x, _ = self.text_cross_attentions[i](x, function_field, function_field)
            x = self.text_layer_norms[i](x + residual)
        
        return x  # (batch, 16, hidden_dim)
    
    def forward(self, hidden_states_list, esm_emb):
        """
        Args:
            hidden_states_list: List of 17 field embeddings (from text)
            esm_emb: Tensor [batch, esm_dim] (from ESM)
        
        Returns:
            logits: (batch, num_go_terms)
        """
        # ============ ENCODE TEXT ============
        text_features = self.encode_text(hidden_states_list)  # (batch, 16, hidden_dim)
        
        # ============ PROCESS ESM ============
        esm_feat = self.esm_proj(esm_emb).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # ============ BIDIRECTIONAL CROSS-ATTENTION ============
        # Text queries ESM: "What sequence features support these annotations?"
        text_attended, _ = self.text_to_esm(
            query=text_features,
            key=esm_feat,
            value=esm_feat
        )  # (batch, 16, hidden_dim)
        text_pooled = text_attended.mean(dim=1)  # (batch, hidden_dim)
        
        # ESM queries text: "Which annotations explain this sequence?"
        esm_attended, _ = self.esm_to_text(
            query=esm_feat,
            key=text_features,
            value=text_features
        )  # (batch, 1, hidden_dim)
        esm_pooled = esm_attended.squeeze(1)  # (batch, hidden_dim)
        
        # ============ FUSE BOTH DIRECTIONS ============
        fused = torch.cat([text_pooled, esm_pooled], dim=-1)  # (batch, hidden_dim*2)
        fused = self.fusion(fused)  # (batch, hidden_dim)
        
        # ============ CLASSIFY ============
        logits = self.classifier(fused)  # (batch, num_go_terms)
        
        return logits