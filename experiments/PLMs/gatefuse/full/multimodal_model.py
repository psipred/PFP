"""4-Modality Fusion Model with Adaptive Gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Encoder that projects modality to common hidden space."""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiModalGate(nn.Module):
    """Gating network for 4 modalities with mask support."""
    
    def __init__(self, hidden_dim, n_modalities=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, H, M):
        """
        Args:
            H: [B, 4, hidden_dim] - encoded modalities
            M: [B, 4] - binary masks (1=present, 0=missing)
        
        Returns:
            w: [B, 4] - softmax weights
        """
        raw = self.fc(H)  # [B, 4, 1]
        raw = raw.squeeze(-1)  # [B, 4]
        
        # Set logits of missing modalities to very negative
        raw = raw + (M - 1) * 1e9  # if M=0 → raw-1e9
        
        w = torch.softmax(raw, dim=1)  # [B, 4]
        return w


class FourModalityFusion(nn.Module):
    """4-modality fusion: seq + text + struct + ppi."""
    
    def __init__(
        self,
        seq_dim,  # 1024 for ProtT5, 1280 for ESM
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=512,
        num_go_terms=100,
        dropout=0.3,
        use_aux_heads=True,
        modality_dropout=0.0  # Probability to randomly drop modalities during training
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.text_dim = text_dim
        self.struct_dim = struct_dim
        self.ppi_dim = ppi_dim
        self.hidden_dim = hidden_dim
        self.use_aux_heads = use_aux_heads
        self.modality_dropout = modality_dropout
        
        # Per-modality encoders
        self.enc_seq = ModalityEncoder(seq_dim, hidden_dim, dropout)
        self.enc_text = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.enc_struct = ModalityEncoder(struct_dim, hidden_dim, dropout)
        self.enc_ppi = ModalityEncoder(ppi_dim, hidden_dim, dropout)
        
        # Gating network
        self.gate = MultiModalGate(hidden_dim, n_modalities=4)
        
        # Main classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
        
        # Auxiliary heads (optional)
        if use_aux_heads:
            self.aux_head_seq = nn.Linear(hidden_dim, num_go_terms)
            self.aux_head_text = nn.Linear(hidden_dim, num_go_terms)
            self.aux_head_struct = nn.Linear(hidden_dim, num_go_terms)
            self.aux_head_ppi = nn.Linear(hidden_dim, num_go_terms)
    
    def _apply_modality_dropout(self, masks, training):
        """Randomly drop available modalities during training."""
        if not training or self.modality_dropout == 0:
            return masks
        
        # Don't drop seq and text (base modalities)
        # Only drop struct and ppi
        dropout_mask = torch.rand_like(masks[:, 2:]) > self.modality_dropout
        
        new_masks = masks.clone()
        new_masks[:, 2:] = new_masks[:, 2:] * dropout_mask.float()
        
        return new_masks
    
    def forward(self, seq, seq_mask, text, text_mask, struct, struct_mask, ppi, ppi_mask):
        """
        Args:
            seq: [B, seq_dim]
            seq_mask: [B, 1] (1=present, 0=missing)
            ... same for text, struct, ppi
        
        Returns:
            logits: [B, num_go_terms]
            aux_logits: dict with auxiliary logits (if use_aux_heads)
            gate_weights: [B, 4] attention weights
        """
        batch_size = seq.size(0)
        
        # Encode all modalities
        h_seq = self.enc_seq(seq)
        h_text = self.enc_text(text)
        h_struct = self.enc_struct(struct)
        h_ppi = self.enc_ppi(ppi)
        
        # Stack: [B, 4, hidden_dim]
        H = torch.stack([h_seq, h_text, h_struct, h_ppi], dim=1)
        
        # Masks: [B, 4]
        M = torch.cat([seq_mask, text_mask, struct_mask, ppi_mask], dim=1)
        
        # Apply modality dropout during training
        M = self._apply_modality_dropout(M, self.training)
        
        # Zero out hidden states for missing modalities
        M_expanded = M.unsqueeze(-1)  # [B, 4, 1]
        H = H * M_expanded
        
        # Gating
        gate_weights = self.gate(H, M)  # [B, 4]
        
        # Fused representation
        gate_expanded = gate_weights.unsqueeze(-1)  # [B, 4, 1]
        z = (gate_expanded * H).sum(dim=1)  # [B, hidden_dim]
        
        # Main classifier
        logits = self.classifier(z)
        
        # Auxiliary heads
        aux_logits = {}
        if self.use_aux_heads:
            aux_logits['seq'] = self.aux_head_seq(h_seq)
            aux_logits['text'] = self.aux_head_text(h_text)
            aux_logits['struct'] = self.aux_head_struct(h_struct)
            aux_logits['ppi'] = self.aux_head_ppi(h_ppi)
        
        return logits, aux_logits, gate_weights


class FocalBCELoss(nn.Module):
    """Focal loss for multi-label classification."""
    
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()