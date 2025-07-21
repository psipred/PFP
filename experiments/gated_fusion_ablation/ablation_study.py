#!/usr/bin/env python3
"""
Bottom-up Ablation Study for Gated Fusion Model
Analyzes component contributions and modality importance
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
            nn.Dropout(0.1)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
            nn.Dropout(0.1)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
            nn.Dropout(0.1)
        )
        
        self.prott5_transform = nn.Sequential(
            nn.Linear(self.prott5_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.esm_transform = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
            nn.Dropout(0.1)
        )
        
        self.prott5_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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


class AblationStudy:
    """Main ablation study coordinator."""
    
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Correct output dimension per GO aspect
        self.aspect_output_dims = {
            'BPO': 3992,
            'CCO': 551,
            'MFO': 677
        }
        # Define ablation models
        self.ablation_models = {
            '0_baseline_text': SingleModalityBaseline(modality='text'),
            '0_baseline_prott5': SingleModalityBaseline(modality='prott5'),
            '0_baseline_esm': SingleModalityBaseline(modality='esm'),
            '1_simple_concat': SimpleConcatenation(),
            '2_transformed_concat': TransformedConcatenation(),
            '3_simple_gated': SimpleGatedFusion(),
            '4_crossmodal_gated': CrossModalGatedFusion(),
            '5_full_gated': FullGatedFusion()
        }
        
        self.results = {}
        self.interpretability_results = defaultdict(list)
        
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
    
    def train_model(self, model: AblationModel, model_name: str, aspect: str,
                   train_dataset, valid_dataset, epochs: int = 100):
        """Train a single ablation model."""
        logger.info(f"\nTraining {model_name} on {aspect}")
        # Only the dedicated ESM baseline should consume ESM embeddings
        use_esm = model_name.endswith('_esm')
        
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
        
        best_fmax = 0.0
        history = []
        interpretability_samples = []
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                names, features, labels = batch
                labels = labels.to(self.device)

                # Forward pass with graceful handling of missing modality features
                try:
                    # Prepare modality‑specific inputs
                    text_input = features.get('text', None)
                    prott5_input = features.get('prott5', None)
                    esm_input = features.get('esm', None) if use_esm else None

                    predictions, interp_data = model(
                        text_input.to(self.device) if text_input is not None else None,
                        prott5_input.to(self.device) if prott5_input is not None else None,
                        esm_input.to(self.device) if esm_input is not None else None
                    )
                except ValueError as e:
                    logger.warning(f"Skipping batch in training due to missing modality features: {e}")
                    continue

                loss = criterion(predictions, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Collect interpretability data
                if len(interpretability_samples) < 100:
                    interpretability_samples.append(interp_data)
            
            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            val_interp_data = []
            
            with torch.no_grad():
                for batch in valid_loader:
                    names, features, labels = batch
                    labels = labels.to(self.device)

                    try:
                        text_input = features.get('text', None)
                        prott5_input = features.get('prott5', None)
                        esm_input = features.get('esm', None) if use_esm else None

                        predictions, interp_data = model(
                            text_input.to(self.device) if text_input is not None else None,
                            prott5_input.to(self.device) if prott5_input is not None else None,
                            esm_input.to(self.device) if esm_input is not None else None
                        )
                    except ValueError as e:
                        logger.warning(f"Skipping batch in validation due to missing modality features: {e}")
                        continue

                    loss = criterion(predictions, labels)
                    val_loss += loss.item()

                    all_preds.append(torch.sigmoid(predictions).cpu())
                    all_labels.append(labels.cpu())
                    val_interp_data.append(interp_data)
            
            # Calculate metrics
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            fmax = calculate_fmax(all_preds, all_labels)
            
            # Log progress
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(valid_loader)
            
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, F-max: {fmax:.4f}")
            
            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'fmax': fmax
            })
            
            # Early stopping
            early_stop(-fmax, fmax, model)
            
            if fmax > best_fmax:
                best_fmax = fmax
                torch.save(model.state_dict(), 
                          self.output_dir / f"{model_name}_{aspect}_best.pt")
            
            if early_stop.stop():
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Analyze interpretability data
        interp_analysis = self.analyze_interpretability(
            interpretability_samples + val_interp_data,
            model_name, aspect
        )
        
        return {
            'best_fmax': best_fmax,
            'final_epoch': epoch,
            'history': history,
            'interpretability': interp_analysis
        }
    
    def analyze_interpretability(self, interp_data_list: List[Dict], 
                                model_name: str, aspect: str) -> Dict:
        """Analyze interpretability data collected during training."""
        if not interp_data_list:
            return {}
        
        analysis = defaultdict(list)
        
        # Aggregate data
        for data in interp_data_list:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    analysis[key].append(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        analysis[f"{key}_{subkey}"].append(subvalue)
        
        # Compute statistics
        stats = {}
        for key, values in analysis.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Save detailed analysis
        analysis_file = self.output_dir / f"{model_name}_{aspect}_interpretability.json"
        with open(analysis_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def run_complete_study(self):
        """Run the complete ablation study."""
        logger.info("Starting complete ablation study")
        
        aspects = ['BPO', 'CCO', 'MFO']
        
        for aspect in aspects:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {aspect}")
            logger.info(f"{'='*50}")
            
            output_dim = self.aspect_output_dims.get(aspect, 677)

            ablation_models = {
                '0_baseline_text':   SingleModalityBaseline(modality='text',   output_dim=output_dim),
                '0_baseline_prott5': SingleModalityBaseline(modality='prott5', output_dim=output_dim),
                '0_baseline_esm':    SingleModalityBaseline(modality='esm',    output_dim=output_dim),
                '1_simple_concat':   SimpleConcatenation(output_dim=output_dim),
                '2_transformed_concat': TransformedConcatenation(output_dim=output_dim),
                '3_simple_gated':    SimpleGatedFusion(output_dim=output_dim),
                '4_crossmodal_gated': CrossModalGatedFusion(output_dim=output_dim),
                '5_full_gated':      FullGatedFusion(output_dim=output_dim)
            }
            # Prepare datasets once per aspect to reuse the in‑memory cache
            train_dataset, valid_dataset = self.prepare_datasets(aspect)
            aspect_results = {}
            
            for model_name, model in ablation_models.items():
                results = self.train_model(model, model_name, aspect,
                                           train_dataset, valid_dataset)
                aspect_results[model_name] = results
                
                # Save individual results
                result_file = self.output_dir / f"{model_name}_{aspect}_results.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        'model': model_name,
                        'aspect': aspect,
                        'best_fmax': results['best_fmax'],
                        'final_epoch': results['final_epoch'],
                        'interpretability_summary': results['interpretability']
                    }, f, indent=2)
            
            self.results[aspect] = aspect_results
            
            # Generate aspect report
            self.generate_aspect_report(aspect, aspect_results)
            self.ablation_models = ablation_models  # keep reference for later analysis
        
        # Generate final comprehensive report
        self.generate_comprehensive_report()
    
    def generate_aspect_report(self, aspect: str, results: Dict):
        """Generate report for a single aspect."""
        report_file = self.output_dir / f"{aspect}_ablation_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Ablation Study Report for {aspect}\n\n")
            
            # Performance comparison table
            f.write("## Performance Comparison\n\n")
            f.write("| Model | Best F-max | Improvement | Key Insights |\n")
            f.write("|-------|------------|-------------|-------------|\n")
            
            # Find a baseline to use as reference
            baseline_fmax = None
            baseline_name = None
            for model_name in ['0_baseline_text', '0_baseline_esm', '0_baseline_prott5']:
                if model_name in results:
                    baseline_fmax = results[model_name]['best_fmax']
                    baseline_name = model_name
                    break
            
            if baseline_fmax is None:
                logger.warning("No baseline model found")
                return
            
            for model_name in sorted(results.keys()):
                model_results = results[model_name]
                fmax = model_results['best_fmax']
                improvement = ((fmax - baseline_fmax) / baseline_fmax) * 100
                
                # Extract key insights
                insights = self.extract_insights(model_name, model_results['interpretability'])
                
                f.write(f"| {model_name} | {fmax:.4f} | {improvement:+.2f}% | {insights} |\n")

            # Interpretability analysis
            f.write("\n## Interpretability Analysis\n\n")
                
            for model_name in ['4_crossmodal_gated', '5_full_gated']:
                if model_name in results:
                    interp = results[model_name]['interpretability']
                    f.write(f"\n### {model_name}\n\n")
                    
                    if 'text_gate_mean' in interp:
                        f.write(f"- Average text gate: {interp['text_gate_mean']['mean']:.3f} "
                               f"(±{interp['text_gate_mean']['std']:.3f})\n")
                    if 'prott5_gate_mean' in interp:
                        f.write(f"- Average ProtT5 gate: {interp['prott5_gate_mean']['mean']:.3f} "
                               f"(±{interp['prott5_gate_mean']['std']:.3f})\n")
                    if 'dominant_modality' in interp:
                        # Count dominant modality
                        text_dominant = sum(1 for d in results[model_name]['interpretability'].get('dominant_modality', []) 
                                          if d == 'text')
                        total = len(results[model_name]['interpretability'].get('dominant_modality', []))
                        if total > 0:
                            f.write(f"- Text dominant in {text_dominant/total*100:.1f}% of samples\n")
    
    def extract_insights(self, model_name: str, interp_data: Dict) -> str:
        """Extract key insights from interpretability data."""
        insights = []
        
        if '0_baseline' in model_name:
            if 'text' in model_name:
                modality = 'Text'
            elif 'prott5' in model_name:
                modality = 'ProtT5'
            elif 'esm' in model_name:
                modality = 'ESM'
            else:
                modality = 'Unknown'
            insights.append(f"{modality} only baseline")
            
        elif '1_simple_concat' in model_name:
            if 'magnitude_ratio' in interp_data:
                ratio = interp_data['magnitude_ratio']['mean']
                insights.append(f"Text/ProtT5 magnitude ratio: {ratio:.2f}")
                
        elif '2_transformed' in model_name:
            if 'cosine_similarity' in interp_data:
                sim = interp_data['cosine_similarity']['mean']
                insights.append(f"Feature similarity: {sim:.3f}")
                
        elif 'gated' in model_name:
            if 'text_gate_mean' in interp_data and 'prott5_gate_mean' in interp_data:
                text_gate = interp_data['text_gate_mean']['mean']
                prott5_gate = interp_data['prott5_gate_mean']['mean']
                
                if text_gate > prott5_gate * 1.2:
                    insights.append("Text-dominant gating")
                elif prott5_gate > text_gate * 1.2:
                    insights.append("ProtT5-dominant gating")
                else:
                    insights.append("Balanced gating")
                    
            if 'gate_sparsity_text' in interp_data:
                sparsity = interp_data['gate_sparsity_text']['mean']
                if sparsity > 0.3:
                    insights.append(f"High text gate sparsity ({sparsity:.2f})")
        
        return "; ".join(insights) if insights else "N/A"
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive report across all aspects."""
        report_file = self.output_dir / "comprehensive_ablation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Gated Fusion Ablation Study\n\n")
            f.write("## Executive Summary\n\n")
            
            # Overall performance gains
            f.write("### Performance Gains by Component\n\n")
            
            components = {
                '1_simple_concat': 'Simple Concatenation',
                '2_transformed_concat': 'Feature Transformation',
                '3_simple_gated': 'Basic Gating',
                '4_crossmodal_gated': 'Cross-Modal Gating',
                '5_full_gated': 'Full Model (with residuals & processors)'
            }
            
            f.write("| Component | BPO Gain | CCO Gain | MFO Gain | Average |\n")
            f.write("|-----------|----------|----------|----------|----------|\n")
            
            for model_id, model_name in components.items():
                gains = []
                for aspect in ['BPO', 'CCO', 'MFO']:
                    if aspect in self.results and model_id in self.results[aspect]:
                        baseline = self.results[aspect]['0_baseline_text']['best_fmax']
                        current = self.results[aspect][model_id]['best_fmax']
                        gain = ((current - baseline) / baseline) * 100
                        gains.append(gain)
                    else:
                        gains.append(0)
                
                avg_gain = np.mean(gains)
                f.write(f"| {model_name} | {gains[0]:+.1f}% | {gains[1]:+.1f}% | "
                       f"{gains[2]:+.1f}% | {avg_gain:+.1f}% |\n")
            
            # Modality importance analysis
            f.write("\n### Modality Importance by GO Aspect\n\n")
            self.analyze_modality_importance(f)
            
            # Key findings
            f.write("\n### Key Findings\n\n")
            self.summarize_key_findings(f)
            
            # Visualizations
            self.create_visualizations()
            f.write("\n### Visualizations\n\n")
            f.write("See generated plots in the output directory:\n")
            f.write("- `ablation_performance.png`: Performance progression\n")
            f.write("- `gate_distributions.png`: Gate value distributions\n")
            f.write("- `modality_importance.png`: Modality importance by aspect\n")
    
    def analyze_modality_importance(self, f):
        """Analyze which modality is more important for each aspect."""
        importance_data = []
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            if aspect not in self.results:
                continue
                
            # Compare single modality baselines
            text_fmax = self.results[aspect].get('0_baseline_text', {}).get('best_fmax', 0)
            prott5_fmax = self.results[aspect].get('0_baseline_prott5', {}).get('best_fmax', 0)
            esm_fmax = self.results[aspect].get('0_baseline_esm', {}).get('best_fmax', 0)
            
            # Analyze gate values from full model
            if '5_full_gated' in self.results[aspect]:
                interp = self.results[aspect]['5_full_gated']['interpretability']
                text_gate = interp.get('text_gate_mean', {}).get('mean', 0.5)
                prott5_gate = interp.get('prott5_gate_mean', {}).get('mean', 0.5)
                
                importance_data.append({
                    'Aspect': aspect,
                    'Text Baseline': text_fmax,
                    'ProtT5 Baseline': prott5_fmax,
                    'ESM Baseline': esm_fmax,
                    'Text Gate': text_gate,
                    'ProtT5 Gate': prott5_gate,
                    'Dominant': 'Text' if text_gate > prott5_gate else 'ProtT5'
                })
        
        # Write table
        f.write("| Aspect | Text F-max | ProtT5 F-max | ESM F-max | Avg Text Gate | Avg ProtT5 Gate | Dominant |\n")
        f.write("|--------|------------|--------------|-----------|---------------|-----------------|----------|\n")
        
        for data in importance_data:
            f.write(f"| {data['Aspect']} | {data['Text Baseline']:.4f} | "
                   f"{data['ProtT5 Baseline']:.4f} | {data['ESM Baseline']:.4f} | "
                   f"{data['Text Gate']:.3f} | {data['ProtT5 Gate']:.3f} | {data['Dominant']} |\n")
    
    def summarize_key_findings(self, f):
        """Summarize key findings from the ablation study."""
        findings = []
        
        # 1. Which components provide the most gain?
        avg_gains = defaultdict(list)
        for aspect in self.results:
            # Find a baseline
            baseline = None
            for baseline_name in ['0_baseline_text', '0_baseline_esm', '0_baseline_prott5']:
                if baseline_name in self.results[aspect]:
                    baseline = self.results[aspect][baseline_name]['best_fmax']
                    break
            
            if baseline is None:
                continue
                
            for model_name, results in self.results[aspect].items():
                if '0_baseline' not in model_name:
                    gain = ((results['best_fmax'] - baseline) / baseline) * 100
                    avg_gains[model_name].append(gain)
        
        best_component = max(avg_gains.items(), key=lambda x: np.mean(x[1]))
        findings.append(f"1. **Largest performance gain**: {best_component[0]} "
                       f"(avg {np.mean(best_component[1]):.1f}% improvement)")
        
        # 2. Is cross-modal gating beneficial?
        if '3_simple_gated' in avg_gains and '4_crossmodal_gated' in avg_gains:
            simple_avg = np.mean(avg_gains['3_simple_gated'])
            cross_avg = np.mean(avg_gains['4_crossmodal_gated'])
            benefit = cross_avg - simple_avg
            findings.append(f"2. **Cross-modal gating benefit**: {benefit:.1f}% additional gain")
        
        # 3. Modality preferences by aspect
        modality_prefs = []
        for aspect in ['BPO', 'CCO', 'MFO']:
            if aspect in self.results and '5_full_gated' in self.results[aspect]:
                interp = self.results[aspect]['5_full_gated']['interpretability']
                if 'text_gate_mean' in interp and 'prott5_gate_mean' in interp:
                    text_gate = interp['text_gate_mean']['mean']
                    prott5_gate = interp['prott5_gate_mean']['mean']
                    if text_gate > prott5_gate * 1.2:
                        modality_prefs.append(f"{aspect}: Text-dominant")
                    elif prott5_gate > text_gate * 1.2:
                        modality_prefs.append(f"{aspect}: ProtT5-dominant")
                    else:
                        modality_prefs.append(f"{aspect}: Balanced")
        
        if modality_prefs:
            findings.append(f"3. **Modality preferences**: {', '.join(modality_prefs)}")
        
        # Write findings
        for finding in findings:
            f.write(f"\n{finding}\n")
    
    def create_visualizations(self):
        """Create visualization plots for the ablation study."""
        # 1. Performance progression plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, aspect in enumerate(['BPO', 'CCO', 'MFO']):
            if aspect not in self.results:
                continue
                
            ax = axes[idx]
            
            # Extract data
            models = []
            performances = []
            
            for model_name in sorted(self.results[aspect].keys()):
                models.append(model_name.split('_', 1)[1])  # Remove number prefix
                performances.append(self.results[aspect][model_name]['best_fmax'])
            
            # Plot
            ax.plot(range(len(models)), performances, 'o-', linewidth=2, markersize=8)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('F-max')
            ax.set_title(f'{aspect} Performance Progression')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Gate distributions (if available)
        if any('5_full_gated' in self.results.get(aspect, {}) for aspect in ['BPO', 'CCO', 'MFO']):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, aspect in enumerate(['BPO', 'CCO', 'MFO']):
                if aspect in self.results and '5_full_gated' in self.results[aspect]:
                    ax = axes[idx]
                    
                    model = self.ablation_models['5_full_gated']
                    if hasattr(model, 'gate_statistics') and model.gate_statistics:
                        text_gates = model.gate_statistics.get('text_gates', [])
                        prott5_gates = model.gate_statistics.get('prott5_gates', [])
                        
                        if text_gates and prott5_gates:
                            ax.hist(text_gates, bins=30, alpha=0.5, label='Text', density=True)
                            ax.hist(prott5_gates, bins=30, alpha=0.5, label='ProtT5', density=True)
                            ax.set_xlabel('Gate Value')
                            ax.set_ylabel('Density')
                            ax.set_title(f'{aspect} Gate Distributions')
                            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'gate_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gated Fusion Ablation Study")
    parser.add_argument('--config', type=str, 
                       default='/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/configs/ablation_config.yaml',
                       help='Configuration file')
    parser.add_argument('--output-dir', type=str,
                       default='/SAN/bioinf/PFP/PFP/experiments/gated_fusion_ablation',
                       help='Output directory for results')
    parser.add_argument('--aspects', nargs='+', default=['BPO', 'CCO', 'MFO'],
                       help='GO aspects to evaluate')
    
    args = parser.parse_args()
    
    # Create base config if it doesn't exist
    if not Path(args.config).exists():
        base_config = {
            'experiment_name': 'gated_fusion_ablation',
            'dataset': {
                'features': ['text', 'prott5'],
                'batch_size': 32
            },
            'model': {
                'hidden_dim': 512,
                'output_dim': 677  # Will be updated per aspect
            },
            'optim': {
                'lr': 1e-4,
                'weight_decay': 0.01,
                'epochs': 30,
                'patience': 5,
                'min_epochs': 10,
                'gradient_clip': 1.0
            },
            'log': {
                'out_dir': args.output_dir
            }
        }
        
        Path(args.config).parent.mkdir(parents=True, exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(base_config, f)
    
    # Run ablation study
    study = AblationStudy(args.config, args.output_dir)
    study.run_complete_study()


if __name__ == "__main__":
    main()