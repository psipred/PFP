"""Fusion Models for Multimodal Protein Function Prediction.

This module implements various fusion techniques for combining protein modalities:
- Concatenation (baseline)
- Bilinear gated fusion (pairwise interactions)
- Aux-only uniform fusion (used for aux-head-only ablations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class ModalityEncoder(nn.Module):
    """Projects modality embedding to common hidden space.

    Architecture: Linear -> LayerNorm -> ReLU -> Dropout

    Args:
        input_dim: Input embedding dimension
        hidden_dim: Output hidden dimension
        dropout: Dropout rate
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseFusion(nn.Module):
    """Base class for all fusion modules.

    Defines the common interface and utility methods for fusion operations.
    """

    def __init__(self, hidden_dim: int, n_modalities: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.dropout_rate = dropout

    def forward(self, H: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: [B, n_modalities, hidden_dim] - encoded modalities
            M: [B, n_modalities] - binary masks (1=present, 0=missing)
        Returns:
            z: [B, hidden_dim] - fused representation
            weights: [B, n_modalities] - fusion weights
        """
        raise NotImplementedError

    def _mask_scores(self, scores: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Apply mask to scores before softmax."""
        neg_inf = torch.finfo(scores.dtype).min
        return scores.masked_fill(M == 0, neg_inf)

    def _safe_softmax(self, scores: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Softmax with NaN handling for all-masked edge cases."""
        weights = torch.softmax(scores, dim=dim)
        return torch.nan_to_num(weights, nan=0.0)

    def _weighted_sum(self, H: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of modalities."""
        return (weights.unsqueeze(-1) * H).sum(dim=1)


class ConcatFusion(BaseFusion):
    """Concatenation fusion - simple baseline.

    Concatenates all modality embeddings and projects back to hidden_dim.
    """

    def __init__(self, hidden_dim: int, n_modalities: int = 4, dropout: float = 0.3):
        super().__init__(hidden_dim, n_modalities, dropout)
        self.projection = nn.Linear(hidden_dim * n_modalities, hidden_dim)

    def forward(self, H: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = H.size(0)
        H_flat = H.view(B, -1)
        z = self.projection(H_flat)

        # Uniform weights over available modalities
        weights = M / M.sum(dim=1, keepdim=True).clamp(min=1)
        return z, weights


class BilinearGatedFusion(BaseFusion):
    """Bilinear gated fusion with pairwise modality interactions.

    Key insight: Some modality pairs may be synergistic (seq + struct)
    while others may be redundant (text + text-derived features).

    Combines unary (per-modality) and pairwise (between-modality) scores
    to learn adaptive weighting based on modality relationships.
    """

    def __init__(self, hidden_dim: int, n_modalities: int = 4, dropout: float = 0.3):
        super().__init__(hidden_dim, n_modalities, dropout)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Unary scores: per-modality importance
        self.unary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Learned pairwise interaction matrix
        self.pairwise = nn.Parameter(torch.zeros(n_modalities, n_modalities))
        nn.init.xavier_uniform_(self.pairwise)

        # Projection for similarity computation
        self.proj = nn.Linear(hidden_dim, hidden_dim // 4)
        self.dropout = nn.Dropout(dropout)

        # Learnable balance between unary and pairwise
        self.pairwise_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, H: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H_norm = self.layer_norm(H)

        # Unary scores
        unary_scores = self.unary(H_norm).squeeze(-1)  # [B, n_modalities]

        # Pairwise scores via projected dot product
        H_proj = self.dropout(self.proj(H_norm))  # [B, n_modalities, hidden_dim // 4]
        similarity = torch.bmm(H_proj, H_proj.transpose(1, 2))  # [B, n, n]

        # Mask pairwise interactions
        mask_2d = M.unsqueeze(1) * M.unsqueeze(2)
        similarity = similarity * mask_2d

        # Weighted pairwise contribution
        pairwise_contrib = (similarity * self.pairwise.unsqueeze(0)).sum(dim=2)

        # Combine unary and pairwise
        raw = unary_scores + self.pairwise_weight * pairwise_contrib
        raw = self._mask_scores(raw, M)
        weights = self._safe_softmax(raw)
        z = self._weighted_sum(H, weights)

        return z, weights


class AuxOnlyUniformFusion(BaseFusion):
    """Uniform average over available modalities.

    This zero-parameter fusion is used for the aux-head-only ablation where the
    final prediction is produced by averaging per-modality auxiliary logits over
    available modalities.
    """

    def __init__(self, hidden_dim: int, n_modalities: int = 4, dropout: float = 0.3):
        super().__init__(hidden_dim, n_modalities, dropout)

    def forward(self, H: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = M / M.sum(dim=1, keepdim=True).clamp(min=1)
        z = self._weighted_sum(H, weights)
        return z, weights


# Fusion Registry
FUSION_REGISTRY: Dict[str, type] = {
    'concat': ConcatFusion,
    'gated_bilinear': BilinearGatedFusion,
    'aux_only': AuxOnlyUniformFusion,
}


class ResidualAuxHead(nn.Module):
    """Auxiliary head with residual connections for per-modality predictions.

    Architecture: 2 residual blocks followed by classifier.
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)
        x = x + h

        # Block 2
        h = self.fc2(x)
        h = self.ln2(h)
        h = self.act(h)
        h = self.dropout(h)
        x = x + h

        return self.classifier(x)


class MultiModalFusionModel(nn.Module):
    """Multimodal protein function prediction with configurable fusion.

    Modality order: [seq, text, struct, ppi]

    Args:
        seq_dim: Dimension of sequence embeddings (ESM: 1280, ProtT5: 1024)
        text_dim: Dimension of text embeddings (PubMedBERT: 768)
        struct_dim: Dimension of structure embeddings (ESM-IF: 512)
        ppi_dim: Dimension of PPI embeddings (512)
        hidden_dim: Common hidden dimension for fusion
        num_go_terms: Number of GO terms to predict
        dropout: Dropout rate
        fusion_type: One of 'concat', 'gated_bilinear', 'aux_only'
        modality_dropout: Dropout rate for non-sequence modalities during training
        protect_seq: If True, never drop sequence modality
        use_late_fusion: If True, enable auxiliary heads and hybrid gated fusion
        late_output_mode: 'hybrid' or 'aux_only' when late fusion is enabled
    """

    MODALITY_NAMES = ['seq', 'text', 'struct', 'ppi']

    def __init__(
        self,
        seq_dim: int,
        text_dim: int = 768,
        struct_dim: int = 512,
        ppi_dim: int = 512,
        hidden_dim: int = 512,
        num_go_terms: int = 100,
        dropout: float = 0.3,
        fusion_type: str = 'gated_bilinear',
        modality_dropout: float = 0.0,
        protect_seq: bool = True,
        use_late_fusion: bool = False,
        late_output_mode: str = 'hybrid',
        **kwargs
    ):
        super().__init__()

        if fusion_type not in FUSION_REGISTRY:
            raise ValueError(
                f"fusion_type must be one of {list(FUSION_REGISTRY.keys())}, got '{fusion_type}'"
            )

        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.protect_seq = protect_seq
        self.n_modalities = len(self.MODALITY_NAMES)
        self.use_late_fusion = use_late_fusion
        self.late_output_mode = late_output_mode
        self.num_go_terms = num_go_terms

        if late_output_mode not in {'hybrid', 'aux_only'}:
            raise ValueError(
                f"late_output_mode must be one of ['hybrid', 'aux_only'], got '{late_output_mode}'"
            )
        if late_output_mode == 'aux_only' and not use_late_fusion:
            raise ValueError("late_output_mode='aux_only' requires use_late_fusion=True")

        # Per-modality encoders
        self.encoders = nn.ModuleDict({
            'seq': ModalityEncoder(seq_dim, hidden_dim, dropout),
            'text': ModalityEncoder(text_dim, hidden_dim, dropout),
            'struct': ModalityEncoder(struct_dim, hidden_dim, dropout),
            'ppi': ModalityEncoder(ppi_dim, hidden_dim, dropout),
        })

        # Fusion module
        fusion_cls = FUSION_REGISTRY[fusion_type]
        self.fusion = fusion_cls(hidden_dim, n_modalities=self.n_modalities, dropout=dropout)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )

        # Late fusion components
        if use_late_fusion:
            self.aux_heads = nn.ModuleDict({
                name: ResidualAuxHead(hidden_dim, num_go_terms, dropout)
                for name in self.MODALITY_NAMES
            })
            self.late_fusion_beta = nn.Parameter(torch.tensor(0.0))

    def _apply_modality_dropout(self, M: torch.Tensor) -> torch.Tensor:
        """Randomly drop non-sequence modalities during training."""
        if not self.training or self.modality_dropout == 0:
            return M

        dropout_mask = torch.rand_like(M) > self.modality_dropout

        if self.protect_seq:
            dropout_mask[:, 0] = True

        return M * dropout_mask.float()

    def forward(
        self,
        seq: torch.Tensor, seq_mask: torch.Tensor,
        text: torch.Tensor, text_mask: torch.Tensor,
        struct: torch.Tensor, struct_mask: torch.Tensor,
        ppi: torch.Tensor, ppi_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            seq: [B, seq_dim] - sequence embeddings
            seq_mask: [B, 1] - sequence availability mask
            text: [B, text_dim] - text embeddings
            text_mask: [B, 1] - text availability mask
            struct: [B, struct_dim] - structure embeddings
            struct_mask: [B, 1] - structure availability mask
            ppi: [B, ppi_dim] - PPI embeddings
            ppi_mask: [B, 1] - PPI availability mask

        Returns:
            logits: [B, num_go_terms] - GO term predictions
            fusion_weights: [B, 4] - weights for [seq, text, struct, ppi]
            aux_outputs: Dict with auxiliary info (if use_late_fusion), else None
        """
        # Encode modalities
        h_seq = self.encoders['seq'](seq)
        h_text = self.encoders['text'](text)
        h_struct = self.encoders['struct'](struct)
        h_ppi = self.encoders['ppi'](ppi)

        # Stack: [B, 4, hidden_dim]
        H = torch.stack([h_seq, h_text, h_struct, h_ppi], dim=1)

        # Masks: [B, 4]
        M = torch.cat([seq_mask, text_mask, struct_mask, ppi_mask], dim=1)
        M = M.to(dtype=H.dtype, device=H.device)

        # Apply modality dropout during training
        M = self._apply_modality_dropout(M)

        # Zero out missing modalities
        H = H * M.unsqueeze(-1)

        # Fuse
        z_early, fusion_weights = self.fusion(H, M)

        # Classifier
        logits_early = self.classifier(z_early)

        if not self.use_late_fusion:
            return logits_early, fusion_weights, None

        # Late Fusion
        h_list = [h_seq, h_text, h_struct, h_ppi]
        aux_logits = {}
        aux_logits_stacked = []

        for i, name in enumerate(self.MODALITY_NAMES):
            h_masked = h_list[i] * M[:, i:i+1]
            aux_logits[name] = self.aux_heads[name](h_masked)
            aux_logits_stacked.append(aux_logits[name])

        aux_logits_tensor = torch.stack(aux_logits_stacked, dim=1)
        weights_expanded = fusion_weights.unsqueeze(-1)
        logits_late = (weights_expanded * aux_logits_tensor).sum(dim=1)

        if self.late_output_mode == 'aux_only':
            logits = logits_late
            aux_outputs = {
                'aux_logits': aux_logits,
                'logits_early': logits_early,
                'logits_late': logits_late,
                'late_output_mode': 'aux_only',
            }
            return logits, fusion_weights, aux_outputs

        # Hybrid residual integration
        lam = torch.sigmoid(self.late_fusion_beta)
        logits = (1 - lam) * logits_early + lam * logits_late

        aux_outputs = {
            'aux_logits': aux_logits,
            'logits_early': logits_early,
            'logits_late': logits_late,
            'lambda': lam,
            'late_output_mode': 'hybrid',
        }

        return logits, fusion_weights, aux_outputs

    @property
    def modality_names(self) -> List[str]:
        """Return modality names in order."""
        return self.MODALITY_NAMES.copy()

    @classmethod
    def available_fusion_types(cls) -> List[str]:
        """Return available fusion types."""
        return list(FUSION_REGISTRY.keys())


def create_model(fusion_type: str, **kwargs) -> nn.Module:
    """Factory function to create fusion model.

    Args:
        fusion_type: Type of fusion ('concat', 'gated_bilinear', 'aux_only')
        **kwargs: Model arguments (seq_dim, hidden_dim, num_go_terms, etc.)

    Returns:
        MultiModalFusionModel instance
    """
    return MultiModalFusionModel(fusion_type=fusion_type, **kwargs)


def count_fusion_parameters(fusion_type: str, hidden_dim: int = 512,
                           n_modalities: int = 4) -> int:
    """Count parameters in a fusion module."""
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    fusion_cls = FUSION_REGISTRY[fusion_type]
    fusion = fusion_cls(hidden_dim, n_modalities=n_modalities, dropout=0.3)
    return sum(p.numel() for p in fusion.parameters())
