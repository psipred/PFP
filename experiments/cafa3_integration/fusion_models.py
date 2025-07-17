import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math

# Add this to fusion_models.py

class ConcatFusion(nn.Module):
    """Simple concatenation fusion for any two modalities."""
    
    def __init__(self, dim1, dim2, hidden_dim=512, output_dim=677):
        super().__init__()
        
        # Project concatenated features
        self.fusion = nn.Sequential(
            nn.Linear(dim1 + dim2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, feat1, feat2):
        # Simple concatenation
        fused = torch.cat([feat1, feat2], dim=-1)
        output = self.fusion(fused)
        
        # Return tuple for consistency with other fusion models
        return output, {}

class ModalitySpecificExpert(nn.Module):
    """Expert network specialized for a specific modality."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        
    def forward(self, x):
        return self.network(x)


class AdaptiveMoEFusion(nn.Module):
    """
    Mixture of Experts fusion that learns to combine ESM and text features
    adaptively based on the input characteristics.
    """
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, 
                 output_dim=677, num_experts_per_modality=3):
        super().__init__()
        
        # Modality-specific experts
        self.esm_experts = nn.ModuleList([
            ModalitySpecificExpert(esm_dim, hidden_dim, hidden_dim)
            for _ in range(num_experts_per_modality)
        ])
        
        self.text_experts = nn.ModuleList([
            ModalitySpecificExpert(text_dim, hidden_dim, hidden_dim)
            for _ in range(num_experts_per_modality)
        ])
        
        # Gating networks - these decide which experts to use
        self.esm_gate = nn.Sequential(
            nn.Linear(esm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts_per_modality)
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts_per_modality)
        )
        
        # Cross-modal attention for feature refinement
        self.cross_attention = CrossModalAttention(hidden_dim)
        
        # Final fusion and prediction
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable temperature for gating
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, esm_features, text_features):
        # Compute expert weights with sparse gating
        esm_weights = F.softmax(self.esm_gate(esm_features) / self.temperature, dim=-1)
        text_weights = F.softmax(self.text_gate(text_features) / self.temperature, dim=-1)
        
        # Apply top-k sparsity (use top 2 experts)
        k = 2
        esm_topk = torch.topk(esm_weights, k, dim=-1)
        text_topk = torch.topk(text_weights, k, dim=-1)
        
        # Compute expert outputs
        esm_output = torch.zeros(esm_features.size(0), self.esm_experts[0].network[-1].out_features).to(esm_features.device)
        for i, expert in enumerate(self.esm_experts):
            mask = (esm_topk.indices == i).any(dim=-1).float().unsqueeze(-1)
            weight = esm_weights[:, i].unsqueeze(-1)
            esm_output += mask * weight * expert(esm_features)
            
        text_output = torch.zeros(text_features.size(0), self.text_experts[0].network[-1].out_features).to(text_features.device)
        for i, expert in enumerate(self.text_experts):
            mask = (text_topk.indices == i).any(dim=-1).float().unsqueeze(-1)
            weight = text_weights[:, i].unsqueeze(-1)
            text_output += mask * weight * expert(text_features)
        
        # Apply cross-modal attention
        esm_refined, text_refined = self.cross_attention(esm_output, text_output)
        
        # Final fusion
        fused = torch.cat([esm_refined, text_refined], dim=-1)
        output = self.fusion(fused)
        
        return output, {
            'esm_weights': esm_weights,
            'text_weights': text_weights,
            'esm_expert_outputs': esm_output,
            'text_expert_outputs': text_output
        }


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for feature refinement."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-head cross attention
        self.esm_to_text = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.text_to_esm = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, esm_features, text_features):
        # Add batch dimension if needed for attention
        esm_features = esm_features.unsqueeze(1)  # [B, 1, D]
        text_features = text_features.unsqueeze(1)  # [B, 1, D]
        
        # Cross attention: ESM attends to text
        esm_refined, _ = self.text_to_esm(esm_features, text_features, text_features)
        esm_refined = self.ln1(esm_refined + esm_features)
        
        # Cross attention: Text attends to ESM
        text_refined, _ = self.esm_to_text(text_features, esm_features, esm_features)
        text_refined = self.ln2(text_refined + text_features)
        
        return esm_refined.squeeze(1), text_refined.squeeze(1)





class GatedMultimodalFusion(nn.Module):
    """
    Gated fusion that learns modality-specific gates to control information flow.
    This prevents one modality from dominating and preserves complementary information.
    """
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, output_dim=677):
        super().__init__()
        
        # Transform each modality to same dimension
        self.esm_transform = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Gating mechanism - learns when to use which modality
        self.esm_gate = nn.Sequential(
            nn.Linear(esm_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(esm_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Modality-specific processing after gating
        self.esm_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, esm_features, text_features):
        # Concatenate raw features for gate computation
        concat_features = torch.cat([esm_features, text_features], dim=-1)
        
        # Transform features
        esm_hidden = self.esm_transform(esm_features)
        text_hidden = self.text_transform(text_features)
        
        # Compute gates
        esm_gate = self.esm_gate(concat_features)
        text_gate = self.text_gate(concat_features)
        
        # Apply gates
        gated_esm = esm_hidden * esm_gate
        gated_text = text_hidden * text_gate
        
        # Process gated features
        processed_esm = self.esm_processor(gated_esm)
        processed_text = self.text_processor(gated_text)
        
        # Add residual connections
        final_esm = processed_esm + self.residual_weight * esm_hidden
        final_text = processed_text + self.residual_weight * text_hidden
        
        # Concatenate and predict
        fused = torch.cat([final_esm, final_text], dim=-1)
        output = self.fusion(fused)
        
        return output, {
            'esm_gate': esm_gate.mean(dim=-1),
            'text_gate': text_gate.mean(dim=-1),
            'gate_correlation': torch.corrcoef(torch.stack([esm_gate.mean(dim=-1), text_gate.mean(dim=-1)]))[0, 1]
        }


class ImprovedGatedFusion(nn.Module):
    """
    Enhanced version with attention-based gating and feature calibration.
    Based on "Learning Deep Multimodal Feature Representation with Asymmetric Multi-layer Fusion" (MM 2020)
    """
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, output_dim=677):
        super().__init__()
        
        # Feature calibration modules
        self.esm_calibration = FeatureCalibration(esm_dim, hidden_dim)
        self.text_calibration = FeatureCalibration(text_dim, hidden_dim)
        
        # Asymmetric fusion paths
        self.esm_to_text_fusion = AsymmetricFusion(hidden_dim, hidden_dim)
        self.text_to_esm_fusion = AsymmetricFusion(hidden_dim, hidden_dim)
        
        # Self-attention for each modality
        self.esm_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Dynamic weight learning
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, esm_features, text_features):
        # Calibrate features
        esm_calibrated = self.esm_calibration(esm_features)
        text_calibrated = self.text_calibration(text_features)
        
        # Self-attention (add sequence dimension)
        esm_calibrated_seq = esm_calibrated.unsqueeze(1)
        text_calibrated_seq = text_calibrated.unsqueeze(1)
        
        esm_attended, _ = self.esm_self_attn(esm_calibrated_seq, esm_calibrated_seq, esm_calibrated_seq)
        text_attended, _ = self.text_self_attn(text_calibrated_seq, text_calibrated_seq, text_calibrated_seq)
        
        esm_attended = esm_attended.squeeze(1)
        text_attended = text_attended.squeeze(1)
        
        # Asymmetric fusion
        esm_enhanced = self.esm_to_text_fusion(esm_attended, text_attended)
        text_enhanced = self.text_to_esm_fusion(text_attended, esm_attended)
        
        # Dynamic weighting
        concat_enhanced = torch.cat([esm_enhanced, text_enhanced], dim=-1)
        weights = self.weight_predictor(concat_enhanced)
        
        # Weighted combination
        fused = weights[:, 0:1] * esm_enhanced + weights[:, 1:2] * text_enhanced
        
        # Final prediction
        output = self.output_projection(fused)
        
        return output, {
            'esm_weight': weights[:, 0],
            'text_weight': weights[:, 1],
            'calibrated_esm': esm_calibrated,
            'calibrated_text': text_calibrated
        }


class FeatureCalibration(nn.Module):
    """Calibrate features to reduce modality gap."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.scale = nn.Parameter(torch.ones(output_dim))
        self.shift = nn.Parameter(torch.zeros(output_dim))
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        x = x * self.scale + self.shift
        return x


class AsymmetricFusion(nn.Module):
    """Asymmetric fusion module that enhances one modality with another."""
    
    def __init__(self, main_dim, auxiliary_dim):
        super().__init__()
        self.main_transform = nn.Linear(main_dim, main_dim)
        self.aux_transform = nn.Linear(auxiliary_dim, main_dim)
        self.gate = nn.Sequential(
            nn.Linear(main_dim + auxiliary_dim, main_dim),
            nn.ReLU(),
            nn.Linear(main_dim, main_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(main_dim)
        
    def forward(self, main_features, aux_features):
        main_proj = self.main_transform(main_features)
        aux_proj = self.aux_transform(aux_features)
        
        gate_input = torch.cat([main_features, aux_features], dim=-1)
        gate = self.gate(gate_input)
        
        enhanced = main_proj + gate * aux_proj
        return self.norm(enhanced)


class MultimodalTransformerFusion(nn.Module):
    """
    Transformer-based fusion that treats each modality as a token sequence
    and learns cross-modal interactions through self-attention.
    Based on "Perceiver: General Perception with Iterative Attention" (ICML 2021)
    and "FLAVA: A Foundational Language And Vision Alignment Model" (CVPR 2022)
    """
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, 
                 output_dim=677, num_layers=4, num_heads=8):
        super().__init__()
        
        # Project modalities to common dimension
        self.esm_projection = nn.Linear(esm_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Modality embeddings (like positional embeddings but for modality type)
        self.modality_embeddings = nn.Embedding(2, hidden_dim)  # 0: ESM, 1: Text
        
        # Learnable query tokens for different GO aspects
        self.num_query_tokens = 8
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads for different granularities
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_query_tokens, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Auxiliary modality-specific heads for regularization
        self.esm_aux_head = nn.Linear(hidden_dim, output_dim)
        self.text_aux_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, esm_features, text_features):
        batch_size = esm_features.size(0)
        
        # Project features
        esm_proj = self.esm_projection(esm_features)  # [B, hidden_dim]
        text_proj = self.text_projection(text_features)  # [B, hidden_dim]
        
        # Add modality embeddings
        esm_proj = esm_proj + self.modality_embeddings(torch.zeros(batch_size, dtype=torch.long, device=esm_features.device))
        text_proj = text_proj + self.modality_embeddings(torch.ones(batch_size, dtype=torch.long, device=text_features.device))
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Create token sequence: [query_tokens, esm_token, text_token]
        esm_proj = esm_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        text_proj = text_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        token_sequence = torch.cat([query_tokens, esm_proj, text_proj], dim=1)
        
        # Apply transformer
        encoded = self.transformer(token_sequence)
        
        # Extract query token outputs
        query_outputs = encoded[:, :self.num_query_tokens, :]  # [B, num_queries, hidden_dim]
        query_outputs_flat = query_outputs.reshape(batch_size, -1)  # [B, num_queries * hidden_dim]
        
        # Main prediction
        output = self.output_head(query_outputs_flat)
        
        # Auxiliary predictions for regularization
        esm_token_output = encoded[:, self.num_query_tokens, :]
        text_token_output = encoded[:, self.num_query_tokens + 1, :]
        
        esm_aux = self.esm_aux_head(esm_token_output)
        text_aux = self.text_aux_head(text_token_output)
        
        return output, {
            'esm_auxiliary': esm_aux,
            'text_auxiliary': text_aux,
            'query_outputs': query_outputs,
            'attention_weights': None  # Can be extracted if needed
        }


class CrossModalTransformer(nn.Module):
    """
    Efficient cross-modal transformer that processes modalities separately
    then fuses through cross-attention layers.
    Based on "ALIGN: Scaling Up Visual and Vision-Language Representation Learning" (ICML 2021)
    """
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, 
                 output_dim=677, num_self_layers=2, num_cross_layers=2):
        super().__init__()
        
        # Modality-specific encoders
        self.esm_encoder = ModalityEncoder(esm_dim, hidden_dim, num_self_layers)
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, num_self_layers)
        
        # Cross-modal fusion layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim) for _ in range(num_cross_layers)
        ])
        
        # Pooling strategy
        self.pool = AttentivePooling(hidden_dim)
        
        # Final classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 = esm + text + fused
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, esm_features, text_features):
        # Encode each modality
        esm_encoded = self.esm_encoder(esm_features)
        text_encoded = self.text_encoder(text_features)
        
        # Cross-modal fusion
        esm_fused, text_fused = esm_encoded, text_encoded
        for cross_layer in self.cross_layers:
            esm_fused, text_fused = cross_layer(esm_fused, text_fused)
        
        # Pool features
        esm_pooled = self.pool(esm_fused)
        text_pooled = self.pool(text_fused)
        fused_pooled = self.pool(torch.cat([esm_fused, text_fused], dim=1))
        
        # Concatenate all representations
        combined = torch.cat([esm_pooled, text_pooled, fused_pooled], dim=-1)
        
        # Final prediction
        output = self.classifier(combined)
        
        return output, {
            'esm_encoded': esm_encoded,
            'text_encoded': text_encoded,
            'fused_representation': fused_pooled
        }


class ModalityEncoder(nn.Module):
    """Self-attention encoder for a single modality."""
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # Project and add sequence dimension
        x = self.projection(x)
        x = self.norm(x)
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Self-attention layers
        for layer in self.layers:
            x = layer(x)
            
        return x


class CrossAttentionLayer(nn.Module):
    """Bidirectional cross-attention between modalities."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.esm_to_text = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.text_to_esm = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, esm_features, text_features):
        # ESM attends to text
        esm_attended, _ = self.esm_to_text(esm_features, text_features, text_features)
        esm_features = self.norm1(esm_features + esm_attended)
        
        # Text attends to ESM
        text_attended, _ = self.text_to_esm(text_features, esm_features, esm_features)
        text_features = self.norm2(text_features + text_attended)
        
        # FFN for both
        esm_features = esm_features + self.ffn(esm_features)
        text_features = text_features + self.ffn(text_features)
        
        return esm_features, text_features


class AttentivePooling(nn.Module):
    """Learned pooling using attention mechanism."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: [B, seq_len, hidden_dim]
        weights = self.attention(x)  # [B, seq_len, 1]
        pooled = (x * weights).sum(dim=1)  # [B, hidden_dim]
        return pooled


import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveMultimodalFusion(nn.Module):
    """
    Contrastive learning approach that aligns ESM and text representations
    before fusion, ensuring they encode complementary information.
    Based on "Contrastive Language-Image Pre-training" (CLIP) adapted for proteins.
    """
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, 
                 output_dim=677, temperature=0.07):
        super().__init__()
        
        # Modality-specific encoders with projection heads
        self.esm_encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        
        # Projection to shared space for contrastive learning
        self.esm_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Multimodal fusion after alignment
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # GO prediction heads with hierarchical structure
        self.go_predictor = HierarchicalGOPredictor(hidden_dim, output_dim)
        
        # Auxiliary task: predict functional similarity
        self.similarity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, esm_features, text_features, labels=None):
        batch_size = esm_features.size(0)
        
        # Encode modalities
        esm_encoded = self.esm_encoder(esm_features)
        text_encoded = self.text_encoder(text_features)
        
        # Project to shared space
        esm_proj = self.esm_proj(esm_encoded)
        text_proj = self.text_proj(text_encoded)
        
        # Normalize for contrastive loss
        esm_proj_norm = F.normalize(esm_proj, p=2, dim=-1)
        text_proj_norm = F.normalize(text_proj, p=2, dim=-1)
        
        # Compute similarity matrix for contrastive loss
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * esm_proj_norm @ text_proj_norm.T
        
        # Attention-based fusion
        # Stack encoded features for attention
        esm_encoded_seq = esm_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
        text_encoded_seq = text_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Self-attention over both modalities
        combined_seq = torch.cat([esm_encoded_seq, text_encoded_seq], dim=1)  # [B, 2, hidden_dim]
        fused, attention_weights = self.fusion_attention(combined_seq, combined_seq, combined_seq)
        
        # Aggregate fused representation
        fused_representation = fused.mean(dim=1)  # [B, hidden_dim]
        
        # GO prediction
        go_predictions = self.go_predictor(fused_representation)
        
        # Compute contrastive loss if training
        contrastive_loss = None
        if labels is not None:
            # Create positive pairs based on shared GO terms
            label_similarity = compute_label_similarity(labels)
            contrastive_loss = self.compute_contrastive_loss(similarity, label_similarity)
        
        # Auxiliary prediction: functional similarity
        concat_features = torch.cat([esm_encoded, text_encoded], dim=-1)
        similarity_pred = self.similarity_predictor(concat_features)
        
        return go_predictions, {
            'contrastive_loss': contrastive_loss,
            'similarity_matrix': similarity,
            'attention_weights': attention_weights,
            'similarity_prediction': similarity_pred,
            'esm_projection': esm_proj_norm,
            'text_projection': text_proj_norm
        }
    
    def compute_contrastive_loss(self, similarity_matrix, label_similarity):
        """
        Compute contrastive loss that encourages proteins with similar functions
        to have aligned ESM and text representations.
        """
        # Convert label similarity to target distribution
        target = F.softmax(label_similarity / 0.1, dim=-1)
        
        # Compute cross-entropy loss both ways
        loss_esm_to_text = -torch.sum(target * F.log_softmax(similarity_matrix, dim=-1), dim=-1).mean()
        loss_text_to_esm = -torch.sum(target.T * F.log_softmax(similarity_matrix.T, dim=-1), dim=-1).mean()
        
        return (loss_esm_to_text + loss_text_to_esm) / 2


class HierarchicalGOPredictor(nn.Module):
    """
    GO prediction head that respects the hierarchical structure of GO terms.
    """
    
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        
        # Multiple prediction heads for different GO depths
        self.depth_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Aggregation layer
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final prediction
        self.final_predictor = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Get predictions at different granularities
        depth_outputs = []
        for predictor in self.depth_predictors:
            depth_outputs.append(predictor(x))
        
        # Aggregate
        aggregated = torch.cat(depth_outputs, dim=-1)
        aggregated = self.aggregator(aggregated)
        
        # Final prediction
        output = self.final_predictor(aggregated)
        
        return output


def compute_label_similarity(labels):
    """
    Compute pairwise similarity between proteins based on GO annotations.
    """
    # Normalize labels
    labels_norm = F.normalize(labels.float(), p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = labels_norm @ labels_norm.T
    
    return similarity


class MultiModalContrastiveLoss(nn.Module):
    """
    Combined loss for GO prediction with contrastive alignment.
    """
    
    def __init__(self, alpha=0.5, beta=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta   # Weight for auxiliary tasks
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets, auxiliary_outputs):
        # Main GO prediction loss
        go_loss = self.bce_loss(predictions, targets)
        
        # Contrastive loss (if available)
        contrastive_loss = auxiliary_outputs.get('contrastive_loss', 0)
        
        # Auxiliary similarity prediction loss (if available)
        similarity_loss = 0
        if 'similarity_prediction' in auxiliary_outputs and 'true_similarity' in auxiliary_outputs:
            similarity_loss = F.mse_loss(
                auxiliary_outputs['similarity_prediction'],
                auxiliary_outputs['true_similarity']
            )
        
        # Combined loss
        total_loss = go_loss + self.alpha * contrastive_loss + self.beta * similarity_loss
        
        return total_loss, {
            'go_loss': go_loss.item(),
            'contrastive_loss': contrastive_loss.item() if torch.is_tensor(contrastive_loss) else contrastive_loss,
            'similarity_loss': similarity_loss.item() if torch.is_tensor(similarity_loss) else similarity_loss,
            'total_loss': total_loss.item()
        }



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class EnhancedGatedFusion(nn.Module):
    """Fixed Gated Fusion with better initialization and no pre-sigmoid normalization."""
    
    def __init__(self, esm_dim=1280, text_dim=768, hidden_dim=512, output_dim=677,
                 freeze_gates=False, gate_temperature=1.0, init_gate_bias=-2.0):
        super().__init__()
        
        # Modality transformations
        self.esm_transform = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Gate temperature
        self.gate_temperature = gate_temperature  # Fixed value, not learnable initially
        
        # Gating networks - IMPROVED
        self.esm_gate = nn.Sequential(
            nn.Linear(esm_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # No activation here
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(esm_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # No activation here
        )
        
        # Initialize gates with different biases to break symmetry
        with torch.no_grad():
            # ESM gate slightly positive bias (favor ESM initially)
            self.esm_gate[-1].bias.data.fill_(0.5)
            # Text gate slightly negative bias (less text initially)
            self.text_gate[-1].bias.data.fill_(-0.5)
        
        # Auxiliary heads
        self.esm_aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.text_aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Main fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # No residual initially to encourage gate learning
        self.use_residual = False
        self.freeze_gates = freeze_gates
        
    def forward(self, esm_features, text_features):
        batch_size = esm_features.size(0)
        device = esm_features.device
        
        # Transform features
        esm_hidden = self.esm_transform(esm_features)
        text_hidden = self.text_transform(text_features)
        
        # Compute gates
        concat_features = torch.cat([esm_features, text_features], dim=-1)
        
        if self.freeze_gates:
            esm_gate = torch.full_like(esm_hidden, 0.5)
            text_gate = torch.full_like(text_hidden, 0.5)
        else:
            # Raw gate logits
            esm_gate_logits = self.esm_gate(concat_features)
            text_gate_logits = self.text_gate(concat_features)
            
            # Apply temperature and sigmoid (no layer norm!)
            esm_gate = torch.sigmoid(esm_gate_logits / self.gate_temperature)
            text_gate = torch.sigmoid(text_gate_logits / self.gate_temperature)
        
        # Apply gates
        gated_esm = esm_hidden * esm_gate
        gated_text = text_hidden * text_gate
        
        # Add small residual if enabled
        if self.use_residual:
            gated_esm = gated_esm + 0.1 * esm_hidden
            gated_text = gated_text + 0.1 * text_hidden
        
        # Auxiliary predictions
        esm_aux = self.esm_aux_head(esm_hidden)
        text_aux = self.text_aux_head(text_hidden)
        
        # Main fusion
        fused = torch.cat([gated_esm, gated_text], dim=-1)
        main_output = self.fusion(fused)
        
        # Detailed gate statistics
        gate_stats = {
            'esm_gate_mean': esm_gate.mean(dim=-1),
            'text_gate_mean': text_gate.mean(dim=-1),
            'esm_gate_std': esm_gate.std(dim=-1),
            'text_gate_std': text_gate.std(dim=-1),
            'gate_diff': (esm_gate - text_gate).mean(dim=-1),
            'esm_aux': esm_aux,
            'text_aux': text_aux,
            'esm_gate_raw': esm_gate,  # Full gate values for debugging
            'text_gate_raw': text_gate,
            'temperature': self.gate_temperature
        }
        
        return main_output, gate_stats
class GateDiagnostics:
    """Diagnostics for gate behavior analysis."""
    
    def __init__(self):
        self.gate_history = defaultdict(list)
        self.protein_gate_patterns = defaultdict(dict)
        self.go_term_preferences = defaultdict(lambda: defaultdict(list))
        
    def update(self, names, gate_stats, labels, go_terms):
        """Update diagnostics with batch data."""
        esm_gates = gate_stats['esm_gate_mean'].detach().cpu().numpy()
        text_gates = gate_stats['text_gate_mean'].detach().cpu().numpy()
        
        for i, name in enumerate(names):
            # Store gate values for each protein
            self.protein_gate_patterns[name] = {
                'esm_gate': float(esm_gates[i]),
                'text_gate': float(text_gates[i]),
                'preference': 'esm' if esm_gates[i] > text_gates[i] else 'text',
                'difference': float(esm_gates[i] - text_gates[i])
            }
            
            # Track GO term preferences
            active_terms = torch.where(labels[i] > 0)[0].cpu().numpy()


            for term_idx in active_terms:
                go_term = go_terms[term_idx]
                self.go_term_preferences[go_term]['esm_gates'].append(float(esm_gates[i]))
                self.go_term_preferences[go_term]['text_gates'].append(float(text_gates[i]))
        
        # Global statistics
        self.gate_history['esm_mean'].append(float(esm_gates.mean()))
        self.gate_history['text_mean'].append(float(text_gates.mean()))
        self.gate_history['esm_std'].append(float(esm_gates.std()))
        self.gate_history['text_std'].append(float(text_gates.std()))
    
    def analyze_patterns(self):
        """Analyze gate patterns."""
        patterns = {
            'global_stats': {
                'esm_mean': np.mean(self.gate_history['esm_mean']),
                'text_mean': np.mean(self.gate_history['text_mean']),
                'esm_dominance': sum(1 for p in self.protein_gate_patterns.values() 
                                    if p['preference'] == 'esm') / len(self.protein_gate_patterns)
            },
            'go_term_preferences': {},
            'extreme_cases': {
                'strong_esm': [],
                'strong_text': [],
                'balanced': []
            }
        }
        
        # Analyze GO term preferences
        for go_term, gates in self.go_term_preferences.items():
            if len(gates['esm_gates']) > 5:  # Need enough samples
                esm_mean = np.mean(gates['esm_gates'])
                text_mean = np.mean(gates['text_gates'])
                patterns['go_term_preferences'][go_term] = {
                    'esm_preference': float(esm_mean),
                    'text_preference': float(text_mean),
                    'modality': 'esm' if esm_mean > text_mean else 'text'
                }
        
        # Find extreme cases
        for name, gates in self.protein_gate_patterns.items():
            diff = abs(gates['difference'])
            if diff > 0.3:
                if gates['preference'] == 'esm':
                    patterns['extreme_cases']['strong_esm'].append(name)
                else:
                    patterns['extreme_cases']['strong_text'].append(name)
            elif diff < 0.1:
                patterns['extreme_cases']['balanced'].append(name)
        
        return patterns


def train_epoch_with_diagnostics(model, loader, optimizer, criterion, device, cfg, epoch, diagnostics, go_terms):
    """Enhanced training with diagnostics."""
    model.train()
    total_loss = 0
    aux_weight = cfg['model'].get('aux_loss_weight', 0.3)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        names, features_batch, labels = batch
        labels = labels.to(device)
        
        # Forward pass

        feat1, feat2 = cfg['dataset']['features']
        predictions, gate_stats = model(
            features_batch[feat1].to(device), 
            features_batch[feat2].to(device)
        )
        
        # Main loss
        main_loss = criterion(predictions, labels)
        
        # Auxiliary losses  
        esm_aux_loss = criterion(gate_stats['esm_aux'], labels)
        text_aux_loss = criterion(gate_stats['text_aux'], labels)
        aux_loss = aux_weight * (esm_aux_loss + text_aux_loss) / 2
        
        # Total loss
        
        loss = main_loss + aux_loss
        
        # Gradient penalty for balanced gates (optional)
        if cfg['model'].get('gate_balance_penalty', 0) > 0:
            gate_diff = gate_stats['gate_diff'].abs().mean()
            loss += cfg['model']['gate_balance_penalty'] * gate_diff
        
        # Update diagnostics
        diagnostics.update(names, gate_stats, labels, go_terms)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient monitoring
        if epoch % 10 == 0 and cfg.get('monitor_gradients', False):
            monitor_gradients(model)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim'].get('gradient_clip', 1.0))
        optimizer.step()
        
        total_loss += loss.item()
        # Update progress bar every 10 batches
        if (batch_idx + 1) % 100 == 0:
            print( main_loss, aux_loss, esm_aux_loss, text_aux_loss, aux_weight, loss)
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)


def monitor_gradients(model):
    """Monitor gradient flow through the model."""
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item()
            }
    
    # Check for dead gradients
    esm_grad = np.mean([v['norm'] for k, v in grad_stats.items() if 'esm' in k])
    text_grad = np.mean([v['norm'] for k, v in grad_stats.items() if 'text' in k])
    
    if esm_grad < 1e-7:
        print("WARNING: Dead gradients in ESM branch!")
    if text_grad < 1e-7:
        print("WARNING: Dead gradients in text branch!")


def staged_training(model, train_loader, valid_loader, cfg, device, diagnostics, go_terms):
    """Staged training approach."""
    criterion = nn.BCEWithLogitsLoss()
    
    # Stage 1: Train modality encoders only
    print("Stage 1: Training modality encoders...")
    encoder_params = []
    for name, param in model.named_parameters():
        if 'transform' in name or 'aux_head' in name:
            param.requires_grad = True
            encoder_params.append(param)
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(encoder_params, lr=cfg['optim']['lr'])
    for epoch in range(cfg['staged'].get('encoder_epochs', 5)):
        train_epoch_with_diagnostics(model, train_loader, optimizer, criterion, device, cfg, epoch, diagnostics, go_terms)
    
    # Stage 2: Train fusion only
    print("Stage 2: Training fusion mechanism...")
    fusion_params = []
    for name, param in model.named_parameters():
        if 'gate' in name or 'fusion' in name:
            param.requires_grad = True
            fusion_params.append(param)
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(fusion_params, lr=cfg['optim']['lr'] * 0.5)
    for epoch in range(cfg['staged'].get('fusion_epochs', 5)):
        train_epoch_with_diagnostics(model, train_loader, optimizer, criterion, device, cfg, epoch, diagnostics, go_terms)
    
    # Stage 3: Fine-tune everything
    print("Stage 3: Fine-tuning all parameters...")
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'] * 0.1)
    return optimizer