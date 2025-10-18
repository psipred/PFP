"""Final Gated Fusion Model with Scale-Balanced Gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionModel(nn.Module):
    """
    Gated fusion between text and ESM with scale-balanced gate input.
    
    Key improvements:
    1. Layer normalization to balance feature scales
    2. L2 normalization specifically for gate input (not for fusion)
    3. Temperature-scaled sigmoid for gate flexibility
    4. Comprehensive diagnostics
    
    Architecture:
    1. Text encoding (from TextFusionModel):
       - Per-field transformer encoders (17 fields)
       - Cross-attention between non-function fields and function field
    2. ESM projection
    3. Scale-balanced gating mechanism
    4. Classification
    """
    
    def __init__(self, num_go_terms: int, esm_dim: int = 1280, hidden_dim: int = 768):
        super().__init__()
        
        self.num_fields = 17
        self.hidden_dim = hidden_dim
        
        # Layer norms for feature balancing
        self.txt_ln = nn.LayerNorm(hidden_dim)
        self.esm_ln = nn.LayerNorm(hidden_dim)
        
        # ============ TEXT ENCODING ============
        self.field_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4, 
                dropout=0.1, 
                batch_first=True
            ) 
            for _ in range(self.num_fields)
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dropout=0.1, 
            batch_first=True
        )
        self.suffix_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
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
        
        # ============ SCALE-BALANCED GATING ============
        # Larger capacity gate with layer normalization
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        
        # Initialize gate to output ~0 (sigmoid(0) = 0.5 for balanced start)
        nn.init.xavier_uniform_(self.gate[-1].weight, gain=0.01)
        nn.init.zeros_(self.gate[-1].bias)
        
        # Learnable temperature for gate
        self.gate_temperature = nn.Parameter(torch.ones(1))
        
        # ============ CLASSIFICATION ============
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_go_terms)
        )
        
        # Diagnostics tracking
        self.register_buffer('step_count', torch.tensor(0))
        self.print_every = 100  # Print diagnostics every N steps
    
    def encode_text(self, hidden_states_list):
        """
        Encode text using TextFusionModel architecture.
        
        Args:
            hidden_states_list: List of 17 field embeddings
                - Fields 0-2, 4-16: (batch, 1, hidden_dim) [CLS only]
                - Field 3: (batch, seq_len, hidden_dim) [Full function text]
        
        Returns:
            text_representation: (batch, hidden_dim) - pooled text features
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
        
        # Pool to single vector
        pooled = x.mean(dim=1)  # (batch, hidden_dim)
        
        return pooled
    
    def forward(self, hidden_states_list, esm_emb):
        """
        Forward pass with scale-balanced gate.
        
        Args:
            hidden_states_list: List of 17 field embeddings (from text)
            esm_emb: Tensor [batch, esm_dim] (from ESM)
        
        Returns:
            logits: (batch, num_go_terms)
        """
        batch_size = esm_emb.shape[0]
        
        # ============ ENCODE TEXT & ESM ============
        text_feat = self.encode_text(hidden_states_list)  # (batch, hidden_dim)
        esm_feat = self.esm_proj(esm_emb)  # (batch, hidden_dim)
        
        # ============ FEATURE STATISTICS (raw magnitudes) ============
        text_feat_norm = text_feat.norm(dim=-1).mean().item()
        esm_feat_norm = esm_feat.norm(dim=-1).mean().item()
        
        # ============ NORMALIZE FEATURES ============
        # Apply layer normalization to balance magnitudes
        text_feat_ln = self.txt_ln(text_feat)
        esm_feat_ln = self.esm_ln(esm_feat)
        
        # Statistics after LayerNorm
        text_ln_norm = text_feat_ln.norm(dim=-1).mean().item()
        esm_ln_norm = esm_feat_ln.norm(dim=-1).mean().item()
        
        # ============ SCALE-BALANCED GATE INPUT ============
        # L2 normalize ONLY for gate input to ensure equal influence
        # This prevents the 10x magnitude difference from biasing the gate
        text_gate_input = F.normalize(text_feat_ln, p=2, dim=-1)
        esm_gate_input = F.normalize(esm_feat_ln, p=2, dim=-1)
        gate_input = torch.cat([text_gate_input, esm_gate_input], dim=-1)
        
        # Gate logits before sigmoid
        gate_logits = self.gate(gate_input)
        
        # Temperature-scaled sigmoid
        alpha = torch.sigmoid(gate_logits / self.gate_temperature)
        
        # ============ GATE STATISTICS ============
        alpha_mean = alpha.mean().item()
        alpha_std = alpha.std().item()
        alpha_min = alpha.min().item()
        alpha_max = alpha.max().item()
        temp = self.gate_temperature.item()
        
        # ============ WEIGHTED FUSION ============
        # CRITICAL: Use original features for fusion, not L2-normalized ones
        # This preserves the actual signal strength of each modality
        fused = alpha * text_feat + (1 - alpha) * esm_feat
        
        # ============ CLASSIFY ============
        logits = self.classifier(fused)
        
        # # ============ DIAGNOSTIC PRINTING ============
        # self.step_count += 1
        
        # if self.training and self.step_count % self.print_every == 0:
        #     print(f"\n{'='*70}")
        #     print(f"GATE DIAGNOSTICS (Step {self.step_count.item()})")
        #     print(f"{'='*70}")
        #     print(f"Feature Magnitudes:")
        #     print(f"  Raw features:")
        #     print(f"    Text L2 norm: {text_feat_norm:.4f}")
        #     print(f"    ESM L2 norm:  {esm_feat_norm:.4f}")
        #     print(f"    Ratio (Text/ESM): {text_feat_norm/esm_feat_norm:.4f}x")
        #     print(f"  After LayerNorm:")
        #     print(f"    Text L2 norm: {text_ln_norm:.4f}")
        #     print(f"    ESM L2 norm:  {esm_ln_norm:.4f}")
        #     print(f"    Ratio: {text_ln_norm/esm_ln_norm:.4f}x")
        #     print(f"\nGate Statistics:")
        #     print(f"  Alpha (text weight):")
        #     print(f"    Mean:  {alpha_mean:.4f}")
        #     print(f"    Std:   {alpha_std:.4f}")
        #     print(f"    Min:   {alpha_min:.4f}")
        #     print(f"    Max:   {alpha_max:.4f}")
        #     print(f"  Temperature: {temp:.4f}")
        #     print(f"  Gate logit mean: {gate_logits.mean().item():.4f}")
        #     print(f"\nInterpretation:")
        #     if alpha_mean > 0.7:
        #         print(f"  → HEAVILY favoring TEXT ({alpha_mean:.1%} weight)")
        #     elif alpha_mean < 0.3:
        #         print(f"  → HEAVILY favoring ESM ({(1-alpha_mean):.1%} weight)")
        #     else:
        #         print(f"  → BALANCED fusion (text={alpha_mean:.1%}, esm={1-alpha_mean:.1%})")
            
        #     if alpha_std < 0.05:
        #         print(f"  ⚠️  Low variance: gate is nearly constant!")
        #     else:
        #         print(f"  ✓ Good variance: gate is adaptive")
        #     print(f"{'='*70}\n")
        
        # # Print summary during evaluation
        # elif not self.training:
        #     print(f"[Val/Test] Alpha mean: {alpha_mean:.4f} ± {alpha_std:.4f} "
        #           f"(range: [{alpha_min:.4f}, {alpha_max:.4f}])")
        
        return logits