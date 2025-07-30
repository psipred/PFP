#!/usr/bin/env python3
"""
Refined Ablation Study Report Pipeline
Focuses on performance metrics and overfitting analysis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import statements remain the same
sys.path.append('/SAN/bioinf/PFP/PFP')

from experiments.cafa3_integration.train_cafa3 import (
    CAFA3Dataset, collate_batch, validate, calculate_fmax, 
    train_epoch, generate_predictions
)
from Network.model_utils import EarlyStop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AblationModel(nn.Module):
    """Base class for ablation models with interpretability hooks."""
    
    def __init__(self, text_dim=768, prott5_dim=1024, esm_dim=1280, hidden_dim=512, output_dim=677):
        super().__init__()
        self.text_dim = text_dim
        self.prott5_dim = prott5_dim
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.interpretability_data = {}
        
    def get_interpretability_data(self):
        """Return collected interpretability data."""
        return self.interpretability_data


class SingleModalityBaseline(AblationModel):
    """Baseline: Single modality model."""
    
    def __init__(self, modality='text', **kwargs):
        super().__init__(**kwargs)
        self.modality = modality
        if modality == 'text':
            input_dim = self.text_dim
        elif modality == 'prott5':
            input_dim = self.prott5_dim
        elif modality == 'esm':
            input_dim = self.esm_dim
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, text_features=None, prott5_features=None, esm_features=None):
        if self.modality == 'text':
            features = text_features
        elif self.modality == 'prott5':
            features = prott5_features
        elif self.modality == 'esm':
            features = esm_features
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
        if features is None:
            raise ValueError(f"Expected {self.modality} features but got None")
        output = self.network(features)
        
        # Store which modality was used
        self.interpretability_data = {
            'modality_used': self.modality,
            'feature_magnitude': features.norm(dim=-1).mean().item()
        }
        
        return output, self.interpretability_data


class SimpleConcatenation(AblationModel):
    """Level 1: Simple concatenation without any transformation."""
    
    def __init__(self, modalities=['text', 'prott5'], **kwargs):
        super().__init__(**kwargs)
        self.modalities = modalities
        
        # Calculate input dimension based on modalities
        input_dim = 0
        if 'text' in modalities:
            input_dim += self.text_dim
        if 'prott5' in modalities:
            input_dim += self.prott5_dim
        if 'esm' in modalities:
            input_dim += self.esm_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Direct concatenation
        features_list = []
        if text_features is not None:
            features_list.append(text_features)
        if prott5_features is not None:
            features_list.append(prott5_features)
        if esm_features is not None:
            features_list.append(esm_features)
        fused = torch.cat(features_list, dim=-1)
        output = self.fusion(fused)
        
        # Analyze relative magnitudes
        interp = {}
        if text_features is not None:
            interp['text_magnitude'] = text_features.norm(dim=-1).mean().item()
        if prott5_features is not None:
            interp['prott5_magnitude'] = prott5_features.norm(dim=-1).mean().item()
        if text_features is not None and prott5_features is not None:
            interp['magnitude_ratio'] = (text_features.norm(dim=-1) / (prott5_features.norm(dim=-1) + 1e-8)).mean().item()
        self.interpretability_data = interp
        
        return output, self.interpretability_data


class TransformedConcatenation(AblationModel):
    """Level 2: Concatenation with feature transformation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Transform to common dimension
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Transform features
        transformed = []
        if text_features is not None:
            text_hidden = self.text_transform(text_features)
            transformed.append(text_hidden)
        else:
            text_hidden = None
        if prott5_features is not None:
            prott5_hidden = self.prott5_transform(prott5_features)
            transformed.append(prott5_hidden)
        else:
            prott5_hidden = None
        if esm_features is not None:
            esm_hidden = self.esm_transform(esm_features)
            transformed.append(esm_hidden)
        else:
            esm_hidden = None
        # Concatenate and fuse
        fused = torch.cat(transformed, dim=-1)
        output = self.fusion(fused)
        
        # Analyze transformed representations
        interp = {}
        if text_hidden is not None:
            interp['text_hidden_magnitude'] = text_hidden.norm(dim=-1).mean().item()
        if prott5_hidden is not None:
            interp['prott5_hidden_magnitude'] = prott5_hidden.norm(dim=-1).mean().item()
        if text_hidden is not None and prott5_hidden is not None:
            interp['cosine_similarity'] = F.cosine_similarity(text_hidden, prott5_hidden, dim=-1).mean().item()
        self.interpretability_data = interp
        
        return output, self.interpretability_data


class SimpleGatedFusion(AblationModel):
    """Level 3: Add basic gating mechanism."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Transform features
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Simple gates (no cross-modal input)
        self.text_gate = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.prott5_gate = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.esm_gate = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Transform features
        transformed = []
        gates = []
        interp = {}
        if text_features is not None:
            text_hidden = self.text_transform(text_features)
            text_gate = self.text_gate(text_features)
            gated_text = text_hidden * text_gate
            transformed.append(gated_text)
            gates.append(('text_gate', text_gate))
        if prott5_features is not None:
            prott5_hidden = self.prott5_transform(prott5_features)
            prott5_gate = self.prott5_gate(prott5_features)
            gated_prott5 = prott5_hidden * prott5_gate
            transformed.append(gated_prott5)
            gates.append(('prott5_gate', prott5_gate))
        if esm_features is not None:
            esm_hidden = self.esm_transform(esm_features)
            esm_gate = self.esm_gate(esm_features)
            gated_esm = esm_hidden * esm_gate
            transformed.append(gated_esm)
            gates.append(('esm_gate', esm_gate))
        # Fuse
        fused = torch.cat(transformed, dim=-1)
        output = self.fusion(fused)
        # Store gate values for analysis
        for name, gate in gates:
            interp[f'{name}_mean'] = gate.mean().item()
            interp[f'{name}_std'] = gate.std().item()
        if 'text_gate' in dict(gates) and 'prott5_gate' in dict(gates):
            interp['gate_difference'] = (dict(gates)['text_gate'].mean(dim=-1) - dict(gates)['prott5_gate'].mean(dim=-1)).mean().item()
        self.interpretability_data = interp
        return output, self.interpretability_data


class CrossModalGatedFusion(AblationModel):
    """Level 4: Gating with cross-modal information."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Transform features
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross-modal gates
        self.text_gate = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.prott5_gate = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # For interpretability
        self.register_buffer('gate_history', torch.zeros(1000, 2))
        self.history_idx = 0
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Concatenate for gate computation
        concat_features = torch.cat([text_features, prott5_features], dim=-1)
        
        # Transform features
        text_hidden = self.text_transform(text_features)
        prott5_hidden = self.prott5_transform(prott5_features)
        
        # Compute cross-modal gates
        text_gate = self.text_gate(concat_features)
        prott5_gate = self.prott5_gate(concat_features)
        
        # Apply gates
        gated_text = text_hidden * text_gate
        gated_prott5 = prott5_hidden * prott5_gate
        
        # Fuse
        fused = torch.cat([gated_text, gated_prott5], dim=-1)
        output = self.fusion(fused)
        
        # Store gate values
        if self.training and self.history_idx < 1000:
            self.gate_history[self.history_idx, 0] = text_gate.mean().item()
            self.gate_history[self.history_idx, 1] = prott5_gate.mean().item()
            self.history_idx += 1
        
        self.interpretability_data = {
            'text_gate_mean': text_gate.mean().item(),
            'prott5_gate_mean': prott5_gate.mean().item(),
            'gate_correlation': torch.corrcoef(torch.stack([
                text_gate.mean(dim=-1), prott5_gate.mean(dim=-1)
            ]))[0, 1].item() if text_gate.shape[0] > 1 else 0.0,
            'dominant_modality': 'text' if text_gate.mean() > prott5_gate.mean() else 'prott5'
        }
        
        return output, self.interpretability_data


class FullGatedFusion(AblationModel):
    """Level 5: Full gated fusion with all components."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Transform features
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross-modal gates
        self.text_gate = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.prott5_gate = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Modality-specific processors
        self.text_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # For detailed analysis
        self.sample_counter = 0
        self.gate_statistics = defaultdict(list)
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Concatenate for gate computation
        concat_features = torch.cat([text_features, prott5_features], dim=-1)
        
        # Transform features
        text_hidden = self.text_transform(text_features)
        prott5_hidden = self.prott5_transform(prott5_features)
        
        # Compute gates
        text_gate = self.text_gate(concat_features)
        prott5_gate = self.prott5_gate(concat_features)
        
        # Apply gates
        gated_text = text_hidden * text_gate
        gated_prott5 = prott5_hidden * prott5_gate
        
        # Process gated features
        processed_text = self.text_processor(gated_text)
        processed_prott5 = self.prott5_processor(gated_prott5)
        
        # Add residual connections
        final_text = processed_text + self.residual_weight * text_hidden
        final_prott5 = processed_prott5 + self.residual_weight * prott5_hidden
        
        # Concatenate and predict
        fused = torch.cat([final_text, final_prott5], dim=-1)
        output = self.fusion(fused)
        
        # Comprehensive interpretability data
        self.interpretability_data = {
            'text_gate_mean': text_gate.mean().item(),
            'prott5_gate_mean': prott5_gate.mean().item(),
            'residual_weight': self.residual_weight.item(),
            'text_contribution': (text_gate * text_hidden).norm(dim=-1).mean().item(),
            'prott5_contribution': (prott5_gate * prott5_hidden).norm(dim=-1).mean().item(),
            'gate_sparsity': {
                'text': (text_gate < 0.1).float().mean().item(),
                'prott5': (prott5_gate < 0.1).float().mean().item()
            }
        }
        
        # Store per-sample statistics
        if self.training:
            self.sample_counter += text_features.shape[0]
            self.gate_statistics['text_gates'].extend(text_gate.mean(dim=-1).detach().cpu().tolist())
            self.gate_statistics['prott5_gates'].extend(prott5_gate.mean(dim=-1).detach().cpu().tolist())
        
        return output, self.interpretability_data


class AdaptiveGatedFusion(AblationModel):
    """Level 6: Adaptive gating with temperature control and better regularization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Transform features
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Adaptive gates with temperature
        self.text_gate = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.prott5_gate = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Learnable temperature parameters for adaptive gating
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
        # Context-aware gate adjustment
        self.gate_adjuster = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2),  # Adjustment factors for each gate
            nn.Tanh()  # Output in [-1, 1] for adjustment
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def compute_adaptive_gates(self, text_features, prott5_features):
        """Compute gates with temperature scaling and adaptive adjustment."""
        # Base gate values
        text_gate_logit = self.text_gate(text_features).squeeze(-1)
        prott5_gate_logit = self.prott5_gate(prott5_features).squeeze(-1)
        
        # Get context-aware adjustments
        concat_features = torch.cat([text_features, prott5_features], dim=-1)
        adjustments = self.gate_adjuster(concat_features)
        
        # Apply adjustments to logits
        text_gate_logit = text_gate_logit + adjustments[:, 0] * 0.5
        prott5_gate_logit = prott5_gate_logit + adjustments[:, 1] * 0.5
        
        # Stack for softmax (ensures gates sum to 1)
        gate_logits = torch.stack([text_gate_logit, prott5_gate_logit], dim=-1)
        
        # Apply temperature-controlled softmax
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        
        return gates[:, 0].unsqueeze(-1), gates[:, 1].unsqueeze(-1)
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Transform features
        text_hidden = self.text_transform(text_features)
        prott5_hidden = self.prott5_transform(prott5_features)
        
        # Compute adaptive gates
        text_gate, prott5_gate = self.compute_adaptive_gates(text_features, prott5_features)
        
        # Apply gates
        gated_text = text_hidden * text_gate
        gated_prott5 = prott5_hidden * prott5_gate
        
        # Fuse
        fused = torch.cat([gated_text, gated_prott5], dim=-1)
        output = self.fusion(fused)
        
        # Interpretability data
        self.interpretability_data = {
            'text_gate_mean': text_gate.mean().item(),
            'prott5_gate_mean': prott5_gate.mean().item(),
            'temperature': self.temperature.item(),
            'gate_sum': (text_gate + prott5_gate).mean().item(),  # Should be ~1.0
            'gate_entropy': -(text_gate * torch.log(text_gate + 1e-8) + 
                             prott5_gate * torch.log(prott5_gate + 1e-8)).mean().item(),
            'gate_variance': {
                'text': text_gate.var().item(),
                'prott5': prott5_gate.var().item()
            }
        }
        
        return output, self.interpretability_data


# Add this to ablation_study.py in the __init__ method after defining ablation_models:
# '6_adaptive_gated': AdaptiveGatedFusion()

class AttentionFusion(AblationModel):
    """Level 7: Multi-head attention fusion between text and prott5."""
    
    def __init__(self, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        
        # Transform modalities to same dimension
        self.text_transform = nn.Linear(self.text_dim, self.hidden_dim)
        self.prott5_transform = nn.Linear(self.prott5_dim, self.hidden_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )
        
        # Modality-specific processors with residual
        self.text_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Adaptive weighting based on attention scores
        self.weight_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Multi-scale fusion layers
        self.scale_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim // (2**i))
            for i in range(3)  # 512, 256, 128
        ])
        
        # Final fusion with all scales
        total_dim = self.hidden_dim + self.hidden_dim // 2 + self.hidden_dim // 4
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Transform both modalities
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        
        # Stack for attention (batch, 2, hidden_dim)
        modalities = torch.stack([text_h, prott5_h], dim=1)
        
        # Cross-attention between modalities
        attended, attention_weights = self.cross_attention(
            modalities, modalities, modalities,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Process each attended modality
        text_processed = self.text_processor(attended[:, 0])
        prott5_processed = self.prott5_processor(attended[:, 1])
        
        # Add residual connections
        text_processed = text_processed + text_h * 0.1
        prott5_processed = prott5_processed + prott5_h * 0.1
        
        # Adaptive weighting
        concat_processed = torch.cat([text_processed, prott5_processed], dim=-1)
        weights = self.weight_predictor(concat_processed)
        
        # Weighted combination
        weighted = text_processed * weights[:, 0:1] + prott5_processed * weights[:, 1:2]
        
        # Multi-scale representations
        scales = []
        for scale_layer in self.scale_layers:
            scales.append(scale_layer(weighted))
        
        # Concatenate all scales
        multi_scale = torch.cat(scales, dim=-1)
        
        # Final prediction
        output = self.final_fusion(multi_scale)
        
        # Interpretability
        self.interpretability_data = {
            'text_weight': weights[:, 0].mean().item(),
            'prott5_weight': weights[:, 1].mean().item(),
            'attention_text_to_prott5': attention_weights[:, 0, 1].mean().item(),
            'attention_prott5_to_text': attention_weights[:, 1, 0].mean().item(),
            'weight_variance': weights.var(dim=0).mean().item()
        }
        
        return output, self.interpretability_data


class HierarchicalFusion(AblationModel):
    """Level 8: Hierarchical fusion inspired by GO structure with regularization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # First level: modality-specific encoding
        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_encoder = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Second level: interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Hierarchical gates for different GO levels
        self.hierarchical_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 2),
                nn.Softmax(dim=-1)
            ) for _ in range(3)  # 3 levels of hierarchy
        ])
        
        # Third level: global fusion with skip connections
        self.global_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # 2 modalities + interaction
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Aspect-aware projection
        self.aspect_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Regularization: feature diversity loss weight
        self.diversity_weight = 0.01
        
    def compute_diversity_loss(self, text_enc, prott5_enc):
        """Encourage diversity between feature representations."""
        sim = F.cosine_similarity(text_enc, prott5_enc, dim=-1)
        return sim.mean()
    
    def forward(self, text_features, prott5_features, esm_features=None):
        # Level 1: Encode modalities
        text_enc = self.text_encoder(text_features)
        prott5_enc = self.prott5_encoder(prott5_features)
        
        # Level 2: Interaction
        interaction_input = torch.cat([text_enc, prott5_enc], dim=-1)
        interaction = self.interaction_layer(interaction_input)
        
        # Hierarchical gating at different levels
        hierarchical_outputs = []
        for i, gate_module in enumerate(self.hierarchical_gates):
            gates = gate_module(interaction_input)
            level_output = gates[:, 0:1] * text_enc + gates[:, 1:2] * prott5_enc
            hierarchical_outputs.append(level_output)
        
        # Average hierarchical outputs
        hierarchical_fusion = torch.stack(hierarchical_outputs).mean(dim=0)
        
        # Level 3: Global fusion with skip connections
        global_input = torch.cat([text_enc, prott5_enc, interaction], dim=-1)
        global_features = self.global_fusion(global_input)
        
        # Combine with hierarchical fusion
        final_features = global_features + 0.1 * hierarchical_fusion
        output = self.aspect_projection(final_features)
        
        # Compute diversity loss for regularization
        diversity_loss = self.compute_diversity_loss(text_enc, prott5_enc)
        
        # Interpretability
        gate_stats = []
        for i, gates in enumerate(hierarchical_outputs):
            gate_module = self.hierarchical_gates[i]
            g = gate_module(interaction_input)
            gate_stats.append({
                'text_gate': g[:, 0].mean().item(),
                'prott5_gate': g[:, 1].mean().item()
            })
        
        self.interpretability_data = {
            'level_0_text_gate': gate_stats[0]['text_gate'],
            'level_1_text_gate': gate_stats[1]['text_gate'],
            'level_2_text_gate': gate_stats[2]['text_gate'],
            'diversity_loss': diversity_loss.item(),
            'feature_magnitudes': {
                'text': text_enc.norm(dim=-1).mean().item(),
                'prott5': prott5_enc.norm(dim=-1).mean().item(),
                'interaction': interaction.norm(dim=-1).mean().item()
            }
        }
        
        # Add diversity loss to regularize (if training)
        if self.training:
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)
        
        return output, self.interpretability_data


class HierarchicalFusion(AblationModel):
    """Level 8: Hierarchical fusion inspired by GO structure with regularization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # First level: modality-specific encoding
        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_encoder = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Second level: interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Hierarchical gates for different GO levels
        self.hierarchical_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 2),
                nn.Softmax(dim=-1)
            ) for _ in range(3)  # 3 levels of hierarchy
        ])
        
        # Third level: global fusion with skip connections
        self.global_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # 2 modalities + interaction
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Aspect-aware projection
        self.aspect_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Regularization: feature diversity loss weight
        self.diversity_weight = 0.01
        
    def compute_diversity_loss(self, text_enc, prott5_enc):
        """Encourage diversity between feature representations."""
        sim = F.cosine_similarity(text_enc, prott5_enc, dim=-1)
        return sim.mean()
    
    def forward(self, text_features, prott5_features, esm_features=None):
        # Level 1: Encode modalities
        text_enc = self.text_encoder(text_features)
        prott5_enc = self.prott5_encoder(prott5_features)
        
        # Level 2: Interaction
        interaction_input = torch.cat([text_enc, prott5_enc], dim=-1)
        interaction = self.interaction_layer(interaction_input)
        
        # Hierarchical gating at different levels
        hierarchical_outputs = []
        for i, gate_module in enumerate(self.hierarchical_gates):
            gates = gate_module(interaction_input)
            level_output = gates[:, 0:1] * text_enc + gates[:, 1:2] * prott5_enc
            hierarchical_outputs.append(level_output)
        
        # Average hierarchical outputs
        hierarchical_fusion = torch.stack(hierarchical_outputs).mean(dim=0)
        
        # Level 3: Global fusion with skip connections
        global_input = torch.cat([text_enc, prott5_enc, interaction], dim=-1)
        global_features = self.global_fusion(global_input)
        
        # Combine with hierarchical fusion
        final_features = global_features + 0.1 * hierarchical_fusion
        output = self.aspect_projection(final_features)
        
        # Compute diversity loss for regularization
        diversity_loss = self.compute_diversity_loss(text_enc, prott5_enc)
        
        # Interpretability
        gate_stats = []
        for i, gates in enumerate(hierarchical_outputs):
            gate_module = self.hierarchical_gates[i]
            g = gate_module(interaction_input)
            gate_stats.append({
                'text_gate': g[:, 0].mean().item(),
                'prott5_gate': g[:, 1].mean().item()
            })
        
        self.interpretability_data = {
            'level_0_text_gate': gate_stats[0]['text_gate'],
            'level_1_text_gate': gate_stats[1]['text_gate'],
            'level_2_text_gate': gate_stats[2]['text_gate'],
            'diversity_loss': diversity_loss.item(),
            'feature_magnitudes': {
                'text': text_enc.norm(dim=-1).mean().item(),
                'prott5': prott5_enc.norm(dim=-1).mean().item(),
                'interaction': interaction.norm(dim=-1).mean().item()
            }
        }
        
        # Add diversity loss to regularize (if training)
        if self.training:
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)
        
        return output, self.interpretability_data

class EnsembleFusion(AblationModel):
    """Level 9: Ensemble approach combining best elements from all models."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Component 1: Simple transformation (from model 2 - best performer)
        self.simple_path = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Component 2: Gated path with improvements
        self.gated_text = nn.Linear(self.text_dim, self.hidden_dim)
        self.gated_prott5 = nn.Linear(self.prott5_dim, self.hidden_dim)
        
        # Adaptive gates using both modalities
        gate_input_dim = self.text_dim + self.prott5_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Component 3: Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Path weighting with temperature
        self.path_temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Final projection with regularization
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, text_features, prott5_features, esm_features=None):
        # Path 1: Simple concatenation and transformation
        concat_features = torch.cat([text_features, prott5_features], dim=-1)
        simple_output = self.simple_path(concat_features)
        
        # Path 2: Gated fusion
        text_h = self.gated_text(text_features)
        prott5_h = self.gated_prott5(prott5_features)
        
        # Compute gates using both modalities
        gates = self.gate_network(concat_features)
        
        # Apply gates
        gated_output = gates[:, 0:1] * text_h + gates[:, 1:2] * prott5_h
        
        # Path 3: Direct weighted average (baseline)
        avg_output = (text_h + prott5_h) / 2
        
        # Combine paths using attention
        combined = torch.stack([simple_output, gated_output], dim=1)
        attention_input = torch.cat([simple_output, gated_output], dim=-1)
        attention_weights = self.attention(attention_input)
        
        # Temperature-scaled combination
        path_weights = F.softmax(attention_weights / self.path_temperature, dim=-1)
        
        # Weighted combination with residual from average
        ensemble_output = (
            path_weights[:, 0:1] * simple_output +
            path_weights[:, 1:2] * gated_output +
            0.1 * avg_output  # Small residual from average
        )
        
        # Final prediction
        output = self.final_projection(ensemble_output)
        
        # Interpretability
        self.interpretability_data = {
            'text_gate': gates[:, 0].mean().item(),
            'prott5_gate': gates[:, 1].mean().item(),
            'simple_path_weight': path_weights[:, 0].mean().item(),
            'gated_path_weight': path_weights[:, 1].mean().item(),
            'path_temperature': self.path_temperature.item(),
            'gate_entropy': -(gates * torch.log(gates + 1e-8)).sum(dim=-1).mean().item()
        }
        
        return output, self.interpretability_data


class TripleModalityAdaptiveFusion(AblationModel):
    """Triple modality fusion combining best elements from top performers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_esm = True
        
        # Simple transformations (from model 2's success)
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Adaptive triple gates with temperature (from model 6)
        self.gate_network = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 3)  # 3 modalities
        )
        
        # Learnable temperature with better initialization
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        # Context-aware adjustment (from model 6)
        self.gate_adjuster = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        # Pairwise interactions (new component)
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final fusion with skip connections
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),  # 3 modalities + 2 interactions
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Regularization weight for diversity
        self.diversity_weight = nn.Parameter(torch.tensor(0.01))
        
    def compute_adaptive_gates(self, text_features, prott5_features, esm_features):
        """Compute temperature-scaled gates for three modalities."""
        # Concatenate all features
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        
        # Base gate values
        gate_logits = self.gate_network(concat_features)
        
        # Context-aware adjustments
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        
        # Temperature-controlled softmax
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
    
    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        """Encourage diversity between representations."""
        sim_tp = F.cosine_similarity(text_h, prott5_h, dim=-1).mean()
        sim_te = F.cosine_similarity(text_h, esm_h, dim=-1).mean()
        sim_pe = F.cosine_similarity(prott5_h, esm_h, dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0
        
    def forward(self, text_features, prott5_features, esm_features):
        if esm_features is None:
            raise ValueError("This model requires ESM features")
            
        # Transform features
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)
        
        # Compute adaptive gates
        text_gate, prott5_gate, esm_gate = self.compute_adaptive_gates(
            text_features, prott5_features, esm_features
        )
        
        # Apply gates
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate
        
        # Compute pairwise interactions (important for capturing relationships)
        text_esm_interact = self.text_esm_interaction(
            torch.cat([gated_text, gated_esm], dim=-1)
        )
        prott5_esm_interact = self.prott5_esm_interaction(
            torch.cat([gated_prott5, gated_esm], dim=-1)
        )
        
        # Combine all representations
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_esm_interact, prott5_esm_interact
        ], dim=-1)
        
        # Final prediction
        output = self.fusion(combined)
        
        # Add diversity regularization during training
        if self.training:
            diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)
        
        # Interpretability data
        # Stack gates into shape (batch, 3) for analysis
        gates_tensor = torch.cat([text_gate, prott5_gate, esm_gate], dim=1)  # (B, 3)
        batch_mean_gates = gates_tensor.mean(dim=0)  # (3,)
        dominant_idx = int(batch_mean_gates.argmax().item())
        dominant_name = ['text', 'prott5', 'esm'][dominant_idx]

        gate_entropy = -(gates_tensor * torch.log(gates_tensor + 1e-8)).sum(dim=1).mean().item()

        self.interpretability_data = {
            'text_gate': text_gate.mean().item(),
            'prott5_gate': prott5_gate.mean().item(),
            'esm_gate': esm_gate.mean().item(),
            'temperature': self.temperature.item(),
            'gate_entropy': gate_entropy,   
            'dominant_modality': dominant_name,
            'diversity_loss': diversity_loss.item() if self.training else 0.0
        }
        
        return output, self.interpretability_data


# --- Model 11: Enhanced triple modality fusion with vector gates and complete interactions ---
class EnhancedTripleModalityFusion(AblationModel):
    """Model 11: Enhanced triple modality fusion with vector gates and complete interactions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_esm = True

        # Modality transformations (same as model 10)
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Vector gates (per-dimension) instead of scalar
        gate_in = self.text_dim + self.prott5_dim + self.esm_dim
        self.text_gate_network = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.prott5_gate_network = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.esm_gate_network = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Temperature parameter for vector gates
        self.temperature = nn.Parameter(torch.tensor(1.5))

        # Context-aware gate adjustments (now vector-based)
        self.gate_adjuster = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 3),
            nn.Tanh()
        )

        # Complete pairwise interactions (including text-prott5)
        self.text_prott5_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Triple interaction module
        self.triple_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Final fusion with all components
        # 3 gated modalities + 3 pairwise interactions + 1 triple interaction = 7 * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 7, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        # Learnable diversity regularization weight
        self.diversity_weight = nn.Parameter(torch.tensor(0.01))

    def compute_vector_gates(self, text_features, prott5_features, esm_features):
        """Compute per-dimension gates using temperature-scaled softmax."""
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)

        # Base vector gate logits (B, H)
        text_gate_logits = self.text_gate_network(concat_features)
        prott5_gate_logits = self.prott5_gate_network(concat_features)
        esm_gate_logits = self.esm_gate_network(concat_features)

        # Vector adjustments
        adjustments = self.gate_adjuster(concat_features)  # (B, 3H)
        adj_text = adjustments[:, :self.hidden_dim]
        adj_prott5 = adjustments[:, self.hidden_dim:2*self.hidden_dim]
        adj_esm = adjustments[:, 2*self.hidden_dim:]

        text_gate_logits = text_gate_logits + 0.5 * adj_text
        prott5_gate_logits = prott5_gate_logits + 0.5 * adj_prott5
        esm_gate_logits = esm_gate_logits + 0.5 * adj_esm

        # Stack and apply temperature softmax across modality axis for each hidden dim
        gate_logits = torch.stack([text_gate_logits, prott5_gate_logits, esm_gate_logits], dim=1)  # (B, 3, H)
        gates = F.softmax(gate_logits / self.temperature, dim=1)  # (B, 3, H)
        return gates[:, 0, :], gates[:, 1, :], gates[:, 2, :]

    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        """Encourage diversity between representations (for reporting; not added to loss here)."""
        text_norm = F.normalize(text_h, p=2, dim=-1)
        prott5_norm = F.normalize(prott5_h, p=2, dim=-1)
        esm_norm = F.normalize(esm_h, p=2, dim=-1)
        sim_tp = (text_norm * prott5_norm).sum(dim=-1).mean()
        sim_te = (text_norm * esm_norm).sum(dim=-1).mean()
        sim_pe = (prott5_norm * esm_norm).sum(dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0

    def forward(self, text_features, prott5_features, esm_features):
        if esm_features is None:
            raise ValueError("This model requires ESM features")

        # Transform
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)

        # Vector gates
        text_gate, prott5_gate, esm_gate = self.compute_vector_gates(
            text_features, prott5_features, esm_features
        )  # each (B, H)

        # Apply gates (broadcast across hidden dim)
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate

        # All pairwise interactions
        text_prott5_interact = self.text_prott5_interaction(torch.cat([gated_text, gated_prott5], dim=-1))
        text_esm_interact = self.text_esm_interaction(torch.cat([gated_text, gated_esm], dim=-1))
        prott5_esm_interact = self.prott5_esm_interaction(torch.cat([gated_prott5, gated_esm], dim=-1))

        # Triple interaction
        triple_interact = self.triple_interaction(torch.cat([gated_text, gated_prott5, gated_esm], dim=-1))

        # Final fusion
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_prott5_interact, text_esm_interact, prott5_esm_interact,
            triple_interact
        ], dim=-1)
        output = self.fusion(combined)

        # Diversity (reported only; training pipeline remains unchanged)
        diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
        # Apply diversity regularization inside the model (internal style)
        if self.training:
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)

        # Interpretability
        gates_tensor = torch.stack([text_gate, prott5_gate, esm_gate], dim=1)  # (B, 3, H)
        gate_entropy_perdim = -(gates_tensor * torch.log(gates_tensor + 1e-8)).sum(dim=1).mean().item()
        dominant_idx = int(torch.stack([
            text_gate.mean(), prott5_gate.mean(), esm_gate.mean()
        ]).argmax().item())
        dominant_name = ['text', 'prott5', 'esm'][dominant_idx]

        self.interpretability_data = {
            'text_gate_mean': text_gate.mean().item(),
            'prott5_gate_mean': prott5_gate.mean().item(),
            'esm_gate_mean': esm_gate.mean().item(),
            'text_gate_std': text_gate.std().item(),
            'prott5_gate_std': prott5_gate.std().item(),
            'esm_gate_std': esm_gate.std().item(),
            'temperature': self.temperature.item(),
            'gate_entropy_perdim': gate_entropy_perdim,
            'diversity_loss': diversity_loss.item(),
            'dominant_modality': dominant_name
        }

        # Keep pipeline signature: return (output, interpretability_data)
        return output, self.interpretability_data


# --- Model 12: EnhancedTripleModalityFusionLoss (returns diversity loss to add in trainer) ---
class EnhancedTripleModalityFusionLoss(AblationModel):
    """Model 12: Same as model 11 but returns diversity loss to add in the trainer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_esm = True

        # Modality transformations
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        gate_in = self.text_dim + self.prott5_dim + self.esm_dim
        self.text_gate_network = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.prott5_gate_network = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.esm_gate_network = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(1.5))
        self.gate_adjuster = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 3),
            nn.Tanh()
        )

        self.text_prott5_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.triple_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 7, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def compute_vector_gates(self, text_features, prott5_features, esm_features):
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        text_gate_logits = self.text_gate_network(concat_features)
        prott5_gate_logits = self.prott5_gate_network(concat_features)
        esm_gate_logits = self.esm_gate_network(concat_features)
        adjustments = self.gate_adjuster(concat_features)
        adj_text = adjustments[:, :self.hidden_dim]
        adj_prott5 = adjustments[:, self.hidden_dim:2*self.hidden_dim]
        adj_esm = adjustments[:, 2*self.hidden_dim:]
        text_gate_logits = text_gate_logits + 0.5 * adj_text
        prott5_gate_logits = prott5_gate_logits + 0.5 * adj_prott5
        esm_gate_logits = esm_gate_logits + 0.5 * adj_esm
        gate_logits = torch.stack([text_gate_logits, prott5_gate_logits, esm_gate_logits], dim=1)
        gates = F.softmax(gate_logits / self.temperature, dim=1)
        return gates[:, 0, :], gates[:, 1, :], gates[:, 2, :]

    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        text_norm = F.normalize(text_h, p=2, dim=-1)
        prott5_norm = F.normalize(prott5_h, p=2, dim=-1)
        esm_norm = F.normalize(esm_h, p=2, dim=-1)
        sim_tp = (text_norm * prott5_norm).sum(dim=-1).mean()
        sim_te = (text_norm * esm_norm).sum(dim=-1).mean()
        sim_pe = (prott5_norm * esm_norm).sum(dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0

    def forward(self, text_features, prott5_features, esm_features):
        if esm_features is None:
            raise ValueError("This model requires ESM features")

        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)

        text_gate, prott5_gate, esm_gate = self.compute_vector_gates(
            text_features, prott5_features, esm_features
        )

        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate

        text_prott5_interact = self.text_prott5_interaction(torch.cat([gated_text, gated_prott5], dim=-1))
        text_esm_interact = self.text_esm_interaction(torch.cat([gated_text, gated_esm], dim=-1))
        prott5_esm_interact = self.prott5_esm_interaction(torch.cat([gated_prott5, gated_esm], dim=-1))
        triple_interact = self.triple_interaction(torch.cat([gated_text, gated_prott5, gated_esm], dim=-1))

        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_prott5_interact, text_esm_interact, prott5_esm_interact,
            triple_interact
        ], dim=-1)
        output = self.fusion(combined)

        diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)

        gates_tensor = torch.stack([text_gate, prott5_gate, esm_gate], dim=1)
        gate_entropy_perdim = -(gates_tensor * torch.log(gates_tensor + 1e-8)).sum(dim=1).mean().item()
        dominant_idx = int(torch.stack([
            text_gate.mean(), prott5_gate.mean(), esm_gate.mean()
        ]).argmax().item())
        dominant_name = ['text', 'prott5', 'esm'][dominant_idx]

        self.interpretability_data = {
            'text_gate_mean': text_gate.mean().item(),
            'prott5_gate_mean': prott5_gate.mean().item(),
            'esm_gate_mean': esm_gate.mean().item(),
            'text_gate_std': text_gate.std().item(),
            'prott5_gate_std': prott5_gate.std().item(),
            'esm_gate_std': esm_gate.std().item(),
            'temperature': self.temperature.item(),
            'gate_entropy_perdim': gate_entropy_perdim,
            'diversity_loss': diversity_loss.item(),
            'dominant_modality': dominant_name
        }

        # Return diversity loss for trainer to add into criterion
        return output, self.interpretability_data, diversity_loss


# --- Model 11A: Add text-prott5 interaction to Model 10 ---
class Model11A_TextProtT5Interaction(AblationModel):
    """Model 11A: Add text-prott5 interaction to Model 10."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_esm = True
        
        # Same as Model 10
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Adaptive gates (same as Model 10)
        self.gate_network = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 3)
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        self.gate_adjuster = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        # ADD: text-prott5 interaction (previously missing in Model 10)
        self.text_prott5_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Existing interactions from Model 10
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final fusion updated for 3 interactions instead of 2
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2),  # 3 modalities + 3 interactions
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        self.diversity_weight = nn.Parameter(torch.tensor(0.01))
        
    def compute_adaptive_gates(self, text_features, prott5_features, esm_features):
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        gate_logits = self.gate_network(concat_features)
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
    
    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        sim_tp = F.cosine_similarity(text_h, prott5_h, dim=-1).mean()
        sim_te = F.cosine_similarity(text_h, esm_h, dim=-1).mean()
        sim_pe = F.cosine_similarity(prott5_h, esm_h, dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0
        
    def forward(self, text_features, prott5_features, esm_features):
        if esm_features is None:
            raise ValueError("This model requires ESM features")
            
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)
        
        text_gate, prott5_gate, esm_gate = self.compute_adaptive_gates(
            text_features, prott5_features, esm_features
        )
        
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate
        
        # All pairwise interactions (including new text-prott5)
        text_prott5_interact = self.text_prott5_interaction(
            torch.cat([gated_text, gated_prott5], dim=-1)
        )
        text_esm_interact = self.text_esm_interaction(
            torch.cat([gated_text, gated_esm], dim=-1)
        )
        prott5_esm_interact = self.prott5_esm_interaction(
            torch.cat([gated_prott5, gated_esm], dim=-1)
        )
        
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_prott5_interact, text_esm_interact, prott5_esm_interact
        ], dim=-1)
        
        output = self.fusion(combined)
        
        if self.training:
            diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)
        
        gates_tensor = torch.cat([text_gate, prott5_gate, esm_gate], dim=1)
        batch_mean_gates = gates_tensor.mean(dim=0)
        dominant_idx = int(batch_mean_gates.argmax().item())
        dominant_name = ['text', 'prott5', 'esm'][dominant_idx]
        gate_entropy = -(gates_tensor * torch.log(gates_tensor + 1e-8)).sum(dim=1).mean().item()

        self.interpretability_data = {
            'text_gate': text_gate.mean().item(),
            'prott5_gate': prott5_gate.mean().item(),
            'esm_gate': esm_gate.mean().item(),
            'temperature': self.temperature.item(),
            'gate_entropy': gate_entropy,
            'dominant_modality': dominant_name,
            'diversity_loss': diversity_loss.item() if self.training else 0.0
        }
        
        return output, self.interpretability_data


# --- Model 11B: Add dynamic diversity weight scheduling to Model 10 ---
class Model11B_DynamicDiversity(AblationModel):
    """Model 11B: Add dynamic diversity weight scheduling to Model 10."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_esm = True
        
        # Same as Model 10
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 3)
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        self.gate_adjuster = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # ADD: Dynamic diversity weight based on gate entropy
        self.base_diversity_weight = nn.Parameter(torch.tensor(0.01))
        self.diversity_scaler = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def compute_adaptive_gates(self, text_features, prott5_features, esm_features):
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        gate_logits = self.gate_network(concat_features)
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
    
    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        sim_tp = F.cosine_similarity(text_h, prott5_h, dim=-1).mean()
        sim_te = F.cosine_similarity(text_h, esm_h, dim=-1).mean()
        sim_pe = F.cosine_similarity(prott5_h, esm_h, dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0
        
    def forward(self, text_features, prott5_features, esm_features):
        if esm_features is None:
            raise ValueError("This model requires ESM features")
            
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)
        
        text_gate, prott5_gate, esm_gate = self.compute_adaptive_gates(
            text_features, prott5_features, esm_features
        )
        
        # Compute gate entropy for dynamic diversity weight
        gates_tensor = torch.cat([text_gate, prott5_gate, esm_gate], dim=1)
        gate_entropy = -(gates_tensor * torch.log(gates_tensor + 1e-8)).sum(dim=1, keepdim=True)
        
        # Dynamic diversity weight: higher when gates are more uniform (high entropy)
        diversity_scale = self.diversity_scaler(gate_entropy.mean().unsqueeze(0))
        dynamic_diversity_weight = self.base_diversity_weight * (1 + 2 * diversity_scale)
        
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate
        
        text_esm_interact = self.text_esm_interaction(
            torch.cat([gated_text, gated_esm], dim=-1)
        )
        prott5_esm_interact = self.prott5_esm_interaction(
            torch.cat([gated_prott5, gated_esm], dim=-1)
        )
        
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_esm_interact, prott5_esm_interact
        ], dim=-1)
        
        output = self.fusion(combined)
        
        if self.training:
            diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
            output = output - dynamic_diversity_weight * diversity_loss.unsqueeze(-1)
        
        batch_mean_gates = gates_tensor.mean(dim=0)
        dominant_idx = int(batch_mean_gates.argmax().item())
        dominant_name = ['text', 'prott5', 'esm'][dominant_idx]

        self.interpretability_data = {
            'text_gate': text_gate.mean().item(),
            'prott5_gate': prott5_gate.mean().item(),
            'esm_gate': esm_gate.mean().item(),
            'temperature': self.temperature.item(),
            'gate_entropy': gate_entropy.mean().item(),
            'dominant_modality': dominant_name,
            'diversity_weight': dynamic_diversity_weight.item() if self.training else self.base_diversity_weight.item(),
            'diversity_loss': diversity_loss.item() if self.training else 0.0
        }
        
        return output, self.interpretability_data


# --- Model 11C: Add mixture of experts for aspect-specific learning to Model 10 ---
class Model11C_MixtureOfExperts(AblationModel):
    """Model 11C: Add mixture of experts for aspect-specific learning to Model 10."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_esm = True
        
        # Same as Model 10
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 3)
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        self.gate_adjuster = nn.Sequential(
            nn.Linear(self.text_dim + self.prott5_dim + self.esm_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.text_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.prott5_esm_interaction = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ADD: Mixture of 3 experts (one per GO aspect)
        self.num_experts = 3
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(self.num_experts)
        ])
        
        # Expert selection gating
        self.expert_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Final projection (shared)
        self.final_projection = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.diversity_weight = nn.Parameter(torch.tensor(0.01))
        
    def compute_adaptive_gates(self, text_features, prott5_features, esm_features):
        concat_features = torch.cat([text_features, prott5_features, esm_features], dim=-1)
        gate_logits = self.gate_network(concat_features)
        adjustments = self.gate_adjuster(concat_features)
        gate_logits = gate_logits + adjustments * 0.5
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
    
    def compute_diversity_loss(self, text_h, prott5_h, esm_h):
        sim_tp = F.cosine_similarity(text_h, prott5_h, dim=-1).mean()
        sim_te = F.cosine_similarity(text_h, esm_h, dim=-1).mean()
        sim_pe = F.cosine_similarity(prott5_h, esm_h, dim=-1).mean()
        return (sim_tp + sim_te + sim_pe) / 3.0
        
    def forward(self, text_features, prott5_features, esm_features):
        if esm_features is None:
            raise ValueError("This model requires ESM features")
            
        text_h = self.text_transform(text_features)
        prott5_h = self.prott5_transform(prott5_features)
        esm_h = self.esm_transform(esm_features)
        
        text_gate, prott5_gate, esm_gate = self.compute_adaptive_gates(
            text_features, prott5_features, esm_features
        )
        
        gated_text = text_h * text_gate
        gated_prott5 = prott5_h * prott5_gate
        gated_esm = esm_h * esm_gate
        
        text_esm_interact = self.text_esm_interaction(
            torch.cat([gated_text, gated_esm], dim=-1)
        )
        prott5_esm_interact = self.prott5_esm_interaction(
            torch.cat([gated_prott5, gated_esm], dim=-1)
        )
        
        combined = torch.cat([
            gated_text, gated_prott5, gated_esm,
            text_esm_interact, prott5_esm_interact
        ], dim=-1)
        
        # Mixture of experts
        expert_weights = self.expert_gate(combined)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(combined))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, hidden_dim)
        
        # Weighted combination of experts
        mixed_output = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        output = self.final_projection(mixed_output)
        
        if self.training:
            diversity_loss = self.compute_diversity_loss(text_h, prott5_h, esm_h)
            output = output - self.diversity_weight * diversity_loss.unsqueeze(-1)
        
        gates_tensor = torch.cat([text_gate, prott5_gate, esm_gate], dim=1)
        batch_mean_gates = gates_tensor.mean(dim=0)
        dominant_idx = int(batch_mean_gates.argmax().item())
        dominant_name = ['text', 'prott5', 'esm'][dominant_idx]
        gate_entropy = -(gates_tensor * torch.log(gates_tensor + 1e-8)).sum(dim=1).mean().item()

        self.interpretability_data = {
            'text_gate': text_gate.mean().item(),
            'prott5_gate': prott5_gate.mean().item(),
            'esm_gate': esm_gate.mean().item(),
            'temperature': self.temperature.item(),
            'gate_entropy': gate_entropy,
            'dominant_modality': dominant_name,
            'expert_weights': expert_weights.mean(dim=0).tolist(),
            'diversity_loss': diversity_loss.item() if self.training else 0.0
        }
        
        return output, self.interpretability_data



class AblationStudy:
    """Refined ablation study coordinator with focus on performance metrics."""
    
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aspect_output_dims = {
            'BPO': 3992,
            'CCO': 551,
            'MFO': 677
        }
        self.results = {}
        self.performance_metrics = defaultdict(lambda: defaultdict(dict))
        
    def prepare_datasets(self, aspect: str):
        """Prepare datasets for the given aspect."""
        data_dir = Path("/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/data")
        
        embeddings_dir = {
            'text': '/SAN/bioinf/PFP/embeddings/cafa3/text',
            'prott5': '/SAN/bioinf/PFP/embeddings/cafa3/prott5',
            'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm'
        }
        
        train_dataset = CAFA3Dataset(
            names_file=str(data_dir / f"{aspect}_train_names.npy"),
            labels_file=str(data_dir / f"{aspect}_train_labels.npz"),
            features=['text', 'prott5', 'esm'],
            embeddings_dir=embeddings_dir
        )
        
        valid_dataset = CAFA3Dataset(
            names_file=str(data_dir / f"{aspect}_valid_names.npy"),
            labels_file=str(data_dir / f"{aspect}_valid_labels.npz"),
            features=['text', 'prott5', 'esm'],
            embeddings_dir=embeddings_dir
        )
        
        return train_dataset, valid_dataset
    
    def train_model(self, model, model_name: str, aspect: str,
                   train_dataset, valid_dataset, epochs: int = 30):
        """Train a single ablation model with performance tracking."""
        logger.info(f"\nTraining {model_name} on {aspect}")
        
        use_esm = getattr(model, 'requires_esm', False) or model_name.endswith('_esm')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=32,
            shuffle=True,
            collate_fn=collate_batch
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_batch
        )
        
        # Setup training
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        early_stop = EarlyStop(patience=5, min_epochs=10)
        diversity_weight = 0.01
        
        # Tracking metrics
        best_fmax = 0.0
        best_epoch = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'fmax': [],
            'overfitting_gap': []
        }
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_losses = []
            
            for batch in train_loader:
                names, features, labels = batch
                labels = labels.to(self.device)
    
                try:
                    text_input = features.get('text', None)
                    prott5_input = features.get('prott5', None)
                    esm_input = features.get('esm', None) if use_esm else None

                    out = model(
                        text_input.to(self.device) if text_input is not None else None,
                        prott5_input.to(self.device) if prott5_input is not None else None,
                        esm_input.to(self.device) if esm_input is not None else None
                    )
                    
                    if isinstance(out, tuple) and len(out) == 3:
                        predictions, interp_data, diversity_loss = out
                    else:
                        predictions, interp_data = out
                        diversity_loss = None
                        
                except ValueError as e:
                    logger.warning(f"Skipping batch: {e}")
                    continue

                main_loss = criterion(predictions, labels)
                loss = main_loss + (diversity_weight * diversity_loss if diversity_loss is not None else 0.0)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in valid_loader:
                    names, features, labels = batch
                    labels = labels.to(self.device)

                    try:
                        text_input = features.get('text', None)
                        prott5_input = features.get('prott5', None)
                        esm_input = features.get('esm', None) if use_esm else None

                        out = model(
                            text_input.to(self.device) if text_input is not None else None,
                            prott5_input.to(self.device) if prott5_input is not None else None,
                            esm_input.to(self.device) if esm_input is not None else None
                        )
                        
                        predictions = out[0] if isinstance(out, tuple) else out
                        
                    except ValueError as e:
                        logger.warning(f"Skipping validation batch: {e}")
                        continue

                    loss = criterion(predictions, labels)
                    val_losses.append(loss.item())

                    all_preds.append(torch.sigmoid(predictions).cpu())
                    all_labels.append(labels.cpu())
            
            # Calculate metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            if all_preds:
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                fmax = calculate_fmax(all_preds, all_labels)
            else:
                fmax = 0.0
            
            overfitting_gap = avg_val_loss - avg_train_loss
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['fmax'].append(fmax)
            history['overfitting_gap'].append(overfitting_gap)
            
            # Log progress
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, F-max: {fmax:.4f}, "
                       f"Overfit Gap: {overfitting_gap:.4f}")
            
            # Early stopping
            early_stop(-fmax, fmax, model)
            
            if fmax > best_fmax:
                best_fmax = fmax
                best_epoch = epoch
                torch.save(model.state_dict(), 
                          self.output_dir / f"{model_name}_{aspect}_best.pt")
            
            if early_stop.stop():
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Calculate final metrics
        final_metrics = {
            'best_fmax': best_fmax,
            'best_epoch': best_epoch,
            'final_epoch': epoch,
            'avg_overfitting_gap': np.mean(history['overfitting_gap'][-5:]),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'convergence_epoch': self._find_convergence_epoch(history['val_loss']),
            'history': history
        }
        
        return final_metrics
    
    def _find_convergence_epoch(self, val_losses, threshold=0.001, window=5):
        """Find epoch where model converged (val loss stabilized)."""
        if len(val_losses) < window:
            return len(val_losses)
            
        for i in range(window, len(val_losses)):
            recent_losses = val_losses[i-window:i]
            if np.std(recent_losses) < threshold:
                return i
        return len(val_losses)
    
    def run_complete_study(self):
        """Run the complete ablation study."""
        logger.info("Starting refined ablation study")
        
        # Import model classes (assuming they're defined in the same file or imported)
        from ablation_study import (
            SingleModalityBaseline, SimpleConcatenation, TransformedConcatenation,
            SimpleGatedFusion, CrossModalGatedFusion, FullGatedFusion,
            AdaptiveGatedFusion, AttentionFusion, HierarchicalFusion,
            EnsembleFusion, TripleModalityAdaptiveFusion, 
            EnhancedTripleModalityFusion, EnhancedTripleModalityFusionLoss
        )
        
        aspects = ['MFO', 'BPO', 'CCO']
        
        for aspect in aspects:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {aspect}")
            logger.info(f"{'='*50}")
            
            output_dim = self.aspect_output_dims[aspect]
            
            # Define models to test
            ablation_models = {
                '0_baseline_text': SingleModalityBaseline(modality='text', output_dim=output_dim),
                '0_baseline_prott5': SingleModalityBaseline(modality='prott5', output_dim=output_dim),
                '0_baseline_esm': SingleModalityBaseline(modality='esm', output_dim=output_dim),
                '1_simple_concat': SimpleConcatenation(output_dim=output_dim),
                '2_transformed_concat': TransformedConcatenation(output_dim=output_dim),
                '3_simple_gated': SimpleGatedFusion(output_dim=output_dim),
                '4_crossmodal_gated': CrossModalGatedFusion(output_dim=output_dim),
                '5_full_gated': FullGatedFusion(output_dim=output_dim),
                '6_adaptive_gated': AdaptiveGatedFusion(output_dim=output_dim),
                '7_attention_fusion': AttentionFusion(output_dim=output_dim),
                '8_hierarchical_fusion': HierarchicalFusion(output_dim=output_dim),
                '9_ensemble_fusion': EnsembleFusion(output_dim=output_dim),
                '10_triple_adaptive': TripleModalityAdaptiveFusion(output_dim=output_dim),
                '11_enhanced_triple': EnhancedTripleModalityFusion(output_dim=output_dim),
                'Model11A_TextProtT5Interaction': Model11A_TextProtT5Interaction(output_dim=output_dim),
                'Model11B_DynamicDiversity': Model11B_DynamicDiversity(output_dim=output_dim),
                'Model11C_MixtureOfExperts': Model11C_MixtureOfExperts(output_dim=output_dim),
                '12_enhanced_triple_loss': EnhancedTripleModalityFusionLoss(output_dim=output_dim),
            }
            
            # Prepare datasets
            train_dataset, valid_dataset = self.prepare_datasets(aspect)
            
            # Train all models
            for model_name, model in ablation_models.items():
                metrics = self.train_model(model, model_name, aspect,
                                         train_dataset, valid_dataset)
                self.performance_metrics[aspect][model_name] = metrics
                
                # Save individual results
                result_file = self.output_dir / f"{model_name}_{aspect}_metrics.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        'model': model_name,
                        'aspect': aspect,
                        'best_fmax': metrics['best_fmax'],
                        'best_epoch': metrics['best_epoch'],
                        'final_epoch': metrics['final_epoch'],
                        'avg_overfitting_gap': metrics['avg_overfitting_gap'],
                        'convergence_epoch': metrics['convergence_epoch']
                    }, f, indent=2)
            
            # Generate aspect report
            self.generate_aspect_report(aspect)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Create visualizations
        self.create_performance_visualizations()
    
    def generate_aspect_report(self, aspect: str):
        """Generate performance report for a single aspect."""
        report_file = self.output_dir / f"{aspect}_performance_report.md"
        metrics = self.performance_metrics[aspect]
        
        with open(report_file, 'w') as f:
            f.write(f"# Performance Report: {aspect}\n\n")
            
            # Performance comparison table
            f.write("## Model Performance Comparison\n\n")
            f.write("| Model | F-max | Best Epoch | Convergence | Overfit Gap | Status |\n")
            f.write("|-------|-------|------------|-------------|-------------|--------|\n")
            
            # Find best baseline for comparison
            baseline_scores = {
                name: metrics[name]['best_fmax'] 
                for name in metrics.keys() 
                if '0_baseline' in name
            }
            
            if baseline_scores:
                best_baseline_name = max(baseline_scores, key=baseline_scores.get)
                best_baseline_score = baseline_scores[best_baseline_name]
            else:
                best_baseline_score = 0.0
                logger.warning(f"No baseline found for {aspect}")
            
            # Sort models by performance
            sorted_models = sorted(metrics.items(), 
                                 key=lambda x: x[1]['best_fmax'], 
                                 reverse=True)
            
            for model_name, model_metrics in sorted_models:
                fmax = model_metrics['best_fmax']
                best_epoch = model_metrics['best_epoch']
                convergence = model_metrics['convergence_epoch']
                overfit_gap = model_metrics['avg_overfitting_gap']
                
                # Determine status
                if overfit_gap > 0.5:
                    status = "⚠️ High Overfit"
                elif overfit_gap > 0.2:
                    status = "⚡ Moderate Overfit"
                else:
                    status = "✅ Good"
                
                # Calculate improvement
                if '0_baseline' not in model_name and best_baseline_score > 0:
                    improvement = ((fmax - best_baseline_score) / best_baseline_score) * 100
                    fmax_str = f"{fmax:.4f} (+{improvement:.1f}%)"
                else:
                    fmax_str = f"{fmax:.4f}"
                
                f.write(f"| {model_name} | {fmax_str} | {best_epoch} | "
                       f"{convergence} | {overfit_gap:.3f} | {status} |\n")
            
            # Training dynamics analysis
            f.write("\n## Training Dynamics\n\n")
            
            # Models with concerning overfitting
            overfit_models = [
                (name, metrics[name]['avg_overfitting_gap'])
                for name in metrics.keys()
                if metrics[name]['avg_overfitting_gap'] > 0.2
            ]
            
            if overfit_models:
                f.write("### ⚠️ Models with High Overfitting\n\n")
                for model_name, gap in sorted(overfit_models, key=lambda x: x[1], reverse=True):
                    f.write(f"- **{model_name}**: Gap = {gap:.3f}\n")
            
            # Best performing fusion models
            f.write("\n### 🏆 Top Fusion Models\n\n")
            fusion_models = {
                name: metrics[name]['best_fmax']
                for name in metrics.keys()
                if '0_baseline' not in name
            }
            
            top_3 = sorted(fusion_models.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (name, score) in enumerate(top_3, 1):
                improvement = ((score - best_baseline_score) / best_baseline_score) * 100 if best_baseline_score > 0 else 0
                f.write(f"{i}. **{name}**: F-max = {score:.4f} (+{improvement:.1f}% vs best baseline)\n")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report across all aspects."""
        report_file = self.output_dir / "comprehensive_performance_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Ablation Study Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # Overall best models
            f.write("### Best Models by Aspect\n\n")
            f.write("| Aspect | Best Model | F-max | Improvement | Overfit Status |\n")
            f.write("|--------|------------|-------|-------------|----------------|\n")
            
            for aspect in ['BPO', 'CCO', 'MFO']:
                if aspect not in self.performance_metrics:
                    continue
                    
                metrics = self.performance_metrics[aspect]
                
                # Find best model
                best_model = max(metrics.items(), key=lambda x: x[1]['best_fmax'])
                model_name, model_metrics = best_model
                
                # Find best baseline
                baseline_scores = {
                    name: metrics[name]['best_fmax']
                    for name in metrics.keys()
                    if '0_baseline' in name
                }
                
                if baseline_scores:
                    best_baseline = max(baseline_scores.values())
                    improvement = ((model_metrics['best_fmax'] - best_baseline) / best_baseline) * 100
                else:
                    improvement = 0.0
                
                # Overfit status
                overfit_gap = model_metrics['avg_overfitting_gap']
                if overfit_gap > 0.5:
                    status = "High"
                elif overfit_gap > 0.2:
                    status = "Moderate"
                else:
                    status = "Low"
                
                f.write(f"| {aspect} | {model_name} | {model_metrics['best_fmax']:.4f} | "
                       f"+{improvement:.1f}% | {status} |\n")
            
            # Component analysis
            f.write("\n### Component Impact Analysis\n\n")
            self._analyze_component_impact(f)
            
            # Overfitting analysis
            f.write("\n### Overfitting Analysis\n\n")
            self._analyze_overfitting_patterns(f)
            
            # Model recommendations
            f.write("\n### Recommendations\n\n")
            self._generate_recommendations(f)
    
    def _analyze_component_impact(self, f):
        """Analyze impact of each architectural component."""
        components = [
            ('1_simple_concat', 'Concatenation'),
            ('2_transformed_concat', '+ Transformation'),
            ('3_simple_gated', '+ Simple Gating'),
            ('4_crossmodal_gated', '+ Cross-modal Gating'),
            ('5_full_gated', '+ Full Architecture')
        ]
        
        f.write("| Component | Avg Improvement | Consistency |\n")
        f.write("|-----------|-----------------|-------------|\n")
        
        for model_id, component_name in components:
            improvements = []
            
            for aspect in ['BPO', 'CCO', 'MFO']:
                if aspect not in self.performance_metrics:
                    continue
                    
                metrics = self.performance_metrics[aspect]
                if model_id not in metrics:
                    continue
                
                # Compare to best baseline
                baseline_scores = [
                    metrics[name]['best_fmax']
                    for name in metrics.keys()
                    if '0_baseline' in name
                ]
                
                if baseline_scores:
                    baseline = max(baseline_scores)
                    improvement = ((metrics[model_id]['best_fmax'] - baseline) / baseline) * 100
                    improvements.append(improvement)
            
            if improvements:
                avg_improvement = np.mean(improvements)
                consistency = np.std(improvements)
                
                f.write(f"| {component_name} | +{avg_improvement:.1f}% | "
                       f"{'High' if consistency < 5 else 'Low'} |\n")
    
    def _analyze_overfitting_patterns(self, f):
        """Analyze overfitting patterns across models."""
        overfit_data = []
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            if aspect not in self.performance_metrics:
                continue
                
            for model_name, metrics in self.performance_metrics[aspect].items():
                overfit_data.append({
                    'model': model_name,
                    'aspect': aspect,
                    'gap': metrics['avg_overfitting_gap'],
                    'convergence': metrics['convergence_epoch']
                })
        
        # Models with highest overfitting
        worst_overfit = sorted(overfit_data, key=lambda x: x['gap'], reverse=True)[:5]
        
        f.write("**Models Most Prone to Overfitting:**\n\n")
        for item in worst_overfit:
            f.write(f"- {item['model']} ({item['aspect']}): Gap = {item['gap']:.3f}\n")
        
        # Models with best generalization
        best_general = sorted(overfit_data, key=lambda x: x['gap'])[:5]
        
        f.write("\n**Models with Best Generalization:**\n\n")
        for item in best_general:
            f.write(f"- {item['model']} ({item['aspect']}): Gap = {item['gap']:.3f}\n")
    
    def _generate_recommendations(self, f):
        """Generate recommendations based on analysis."""
        # Find overall best models
        best_models = {}
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            if aspect not in self.performance_metrics:
                continue
                
            metrics = self.performance_metrics[aspect]
            
            # Filter models with acceptable overfitting
            good_models = {
                name: m for name, m in metrics.items()
                if m['avg_overfitting_gap'] < 0.2
            }
            
            if good_models:
                best = max(good_models.items(), key=lambda x: x[1]['best_fmax'])
                best_models[aspect] = best[0]
        
        f.write("**Recommended Models by Task:**\n\n")
        for aspect, model in best_models.items():
            f.write(f"- **{aspect}**: {model} (balanced performance and generalization)\n")
        
        f.write("\n**General Recommendations:**\n\n")
        f.write("1. **For production**: Use models with overfitting gap < 0.2\n")
        f.write("2. **For research**: Enhanced triple modality models show promise\n")
        f.write("3. **For efficiency**: Adaptive gated models balance performance and complexity\n")
    
    def create_performance_visualizations(self):
        """Create performance visualization plots."""
        # 1. Performance comparison across aspects
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, aspect in enumerate(['BPO', 'CCO', 'MFO']):
            if aspect not in self.performance_metrics:
                continue
                
            ax = axes[idx]
            metrics = self.performance_metrics[aspect]
            
            # Extract data
            models = list(metrics.keys())
            scores = [metrics[m]['best_fmax'] for m in models]
            overfit_gaps = [metrics[m]['avg_overfitting_gap'] for m in models]
            
            # Create scatter plot
            scatter = ax.scatter(scores, overfit_gaps, s=100, alpha=0.6)
            
            # Add labels for interesting points
            for i, model in enumerate(models):
                if '0_baseline' in model or i % 2 == 0:  # Label some points
                    ax.annotate(model.split('_')[1], (scores[i], overfit_gaps[i]), 
                              fontsize=8, alpha=0.7)
            
            ax.set_xlabel('F-max Score')
            ax.set_ylabel('Overfitting Gap')
            ax.set_title(f'{aspect}: Performance vs Overfitting')
            ax.grid(True, alpha=0.2)
            
            # Add regions
            ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='High overfit threshold')
            ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Good generalization')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_vs_overfitting.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Learning curves for top models
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for aspect in ['BPO', 'CCO', 'MFO']:
            if aspect not in self.performance_metrics:
                continue
                
            # Get top 2 models
            metrics = self.performance_metrics[aspect]
            top_models = sorted(metrics.items(), 
                              key=lambda x: x[1]['best_fmax'], 
                              reverse=True)[:2]
            
            for model_name, model_metrics in top_models:
                if plot_idx >= 6:
                    break
                    
                ax = axes[plot_idx]
                history = model_metrics['history']
                
                epochs = range(1, len(history['train_loss']) + 1)
                ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
                ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{model_name} ({aspect})')
                ax.legend()
                ax.grid(True, alpha=0.2)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refined Ablation Study")
    parser.add_argument('--config', type=str, 
                       default='/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/configs/ablation_config.yaml',
                       help='Configuration file')
    parser.add_argument('--output-dir', type=str,
                       default='/SAN/bioinf/PFP/PFP/experiments/gated_fusion_ablation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create base config if needed
    if not Path(args.config).exists():
        base_config = {
            'experiment_name': 'gated_fusion_ablation',
            'dataset': {
                'features': ['text', 'prott5', 'esm'],
                'batch_size': 32
            },
            'model': {
                'hidden_dim': 512
            },
            'optim': {
                'lr': 1e-4,
                'weight_decay': 0.01,
                'epochs': 30,
                'patience': 5,
                'min_epochs': 10
            }
        }
        
        Path(args.config).parent.mkdir(parents=True, exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(base_config, f)
    
    # Run study
    study = AblationStudy(args.config, args.output_dir)
    study.run_complete_study()


if __name__ == "__main__":
    main()